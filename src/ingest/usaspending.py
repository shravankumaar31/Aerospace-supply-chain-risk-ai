"""
USASpending.gov contract award ingestion for aerospace NAICS codes.

Pulls contract awards from the USASpending Awards Search API (no key required),
filters to aerospace manufacturing NAICS codes, and saves raw + clean outputs.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

# Aerospace manufacturing NAICS codes (FAR Part 19 / SIC crosswalk)
AEROSPACE_NAICS = [
    "336411",  # Aircraft Manufacturing
    "336412",  # Aircraft Engine & Engine Parts Manufacturing
    "336413",  # Other Aircraft Parts & Equipment Manufacturing
    "336414",  # Guided Missile & Space Vehicle Manufacturing
    "336415",  # Guided Missile & Space Vehicle Propulsion Unit Manufacturing
    "336419",  # Other Guided Missile & Space Vehicle Parts Manufacturing
]

# Number of full fiscal years to look back from the current award_fiscal_year
FISCAL_YEARS = [2022, 2023, 2024]

# USASpending hard-caps results at 100 per page
PAGE_SIZE = 100

RAW_OUT = Path("data/raw/usaspending_raw.json")
CLEAN_OUT = Path("data/raw/usaspending_clean.csv")


# ---------------------------------------------------------------------------
# API request helpers
# ---------------------------------------------------------------------------

def build_payload(naics_codes: list[str], fiscal_years: list[int], page: int) -> dict:
    """
    Build the POST body for the spending_by_award endpoint.

    Key parameters:
      filters.award_type_codes  – "A","B","C","D" = definitive contracts
      filters.naics_codes       – list of 6-digit NAICS strings
      filters.time_period       – fiscal-year date ranges
      fields                    – Title-Case column names the API recognises
      sort / order              – descending by award amount for deterministic pages
      page / limit              – pagination controls

    Note: USASpending field names in the `fields` array are Title Case, not
    snake_case.  Passing snake_case names returns null values for every record.
    """
    # Build one date range per fiscal year.  US fiscal year runs Oct 1 – Sep 30.
    time_periods = [
        {
            "start_date": f"{fy - 1}-10-01",   # FY starts Oct 1 of prior calendar year
            "end_date":   f"{fy}-09-30",        # FY ends Sep 30
        }
        for fy in fiscal_years
    ]

    return {
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],   # definitive contract types only
            "naics_codes": naics_codes,
            "time_period": time_periods,
        },
        "fields": [
            "Recipient Name",                    # company / organisation that received the award
            "Award Amount",                      # total obligated dollar value
            "Place of Performance State Code",   # 2-letter state abbreviation
            "NAICS Code",                        # 6-digit NAICS on the award (may differ from filter)
            "Start Date",                        # period of performance start
        ],
        "sort": "Award Amount",
        "order": "desc",
        "page": page,
        "limit": PAGE_SIZE,
    }


def fetch_page(payload: dict) -> dict:
    """POST to the API and return the parsed JSON response."""
    resp = requests.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Pagination loop
# ---------------------------------------------------------------------------

def pull_all_awards() -> list[dict]:
    """
    Iterate through every page of results and return a flat list of award records.

    USASpending returns a `page_metadata.hasNext` boolean to signal more pages.
    """
    all_records: list[dict] = []
    page = 1

    print(f"Fetching aerospace contract awards (NAICS: {', '.join(AEROSPACE_NAICS)}) "
          f"for FY {FISCAL_YEARS[0]}–{FISCAL_YEARS[-1]} …")

    while True:
        payload = build_payload(AEROSPACE_NAICS, FISCAL_YEARS, page)
        data = fetch_page(payload)

        results = data.get("results", [])
        all_records.extend(results)

        has_next = data.get("page_metadata", {}).get("hasNext", False)
        print(f"  Page {page:>3}: {len(results):>4} records  |  cumulative: {len(all_records):>6}", end="\r")

        if not has_next:
            break
        page += 1

    print()  # newline after the carriage-return progress line
    return all_records


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_raw(records: list[dict]) -> None:
    """Write the full list of raw API records to JSON for auditability."""
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_OUT, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Raw JSON saved → {RAW_OUT}  ({RAW_OUT.stat().st_size / 1_048_576:.1f} MB)")


def build_clean_df(records: list[dict]) -> pd.DataFrame:
    """
    Normalise raw records into a tidy DataFrame with standardised column names.

    The API returns Title-Case column names ("Recipient Name", "Award Amount",
    etc.) which we rename to snake_case for downstream use.
    """
    df = pd.DataFrame(records)

    # Rename Title-Case API names → snake_case output names
    df = df.rename(columns={
        "Recipient Name":                  "recipient_name",
        "Award Amount":                    "award_amount",
        "Place of Performance State Code": "state",
        "NAICS Code":                      "naics_code",
        "Start Date":                      "start_date",
    })

    # Keep only the five target columns; drop anything extra the API may return
    target_cols = ["recipient_name", "award_amount", "state", "naics_code", "start_date"]
    df = df[[c for c in target_cols if c in df.columns]]

    # Coerce award_amount to numeric (API may return strings or None)
    df["award_amount"] = pd.to_numeric(df["award_amount"], errors="coerce")

    return df


def save_clean(df: pd.DataFrame) -> None:
    """Write the clean DataFrame to CSV."""
    CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_OUT, index=False)
    print(f"Clean CSV saved → {CLEAN_OUT}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the pulled dataset."""
    total_records = len(df)
    total_value_b = df["award_amount"].sum() / 1e9

    top5 = (
        df.groupby("recipient_name", dropna=False)["award_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    print("\n" + "=" * 60)
    print("  USASpending Aerospace Contract Pull — Summary")
    print("=" * 60)
    print(f"  Total records pulled : {total_records:>10,}")
    print(f"  Total award value    : ${total_value_b:>10.2f}B")
    print("\n  Top 5 recipients by award value:")
    for rank, (name, amt) in enumerate(top5.items(), start=1):
        print(f"    {rank}. {str(name):<45} ${amt / 1e9:.2f}B")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate fetch → save raw → clean → save clean → summarise."""
    try:
        records = pull_all_awards()
    except requests.HTTPError as exc:
        print(f"API request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("No records returned — check NAICS codes and fiscal year filters.")
        sys.exit(1)

    save_raw(records)

    df = build_clean_df(records)
    save_clean(df)
    print_summary(df)


if __name__ == "__main__":
    main()
