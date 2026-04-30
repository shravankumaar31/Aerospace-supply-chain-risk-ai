"""
Census Bureau International Trade (exports) ingestion for aerospace HS codes.

Pulls annual export values by destination country from the Census Bureau
International Trade API (no API key required) for HS chapters covering
aircraft and spacecraft (8801–8805), years 2022–2024.

Outputs:
    data/raw/census_trade_raw.json   — raw API response list-of-lists
    data/raw/census_trade_clean.csv  — tidy DataFrame with typed columns
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.census.gov/data/timeseries/intltrade/exports/hs"

# HS commodity codes for aircraft and spacecraft (4-digit chapter level)
AEROSPACE_HS_CODES = ["8801", "8802", "8803", "8804", "8805"]

# Query December of each year: ALL_VAL_YR is year-to-date, so December = full-year total
YEAR_MONTHS = ["2022-12", "2023-12", "2024-12"]

# Fields to request — note Census uses E_COMMODITY (not HS_COMMODITY)
GET_FIELDS = "CTY_CODE,CTY_NAME,ALL_VAL_YR,E_COMMODITY,E_COMMODITY_LDESC"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_trade_data(hs_code: str, year_month: str) -> list[list] | None:
    """
    Fetch export records for one HS code and one year-month (e.g. '2022-12').

    ALL_VAL_YR is a year-to-date cumulative field; querying December gives the
    full-year annual total. Returns raw list-of-lists (first row = headers),
    or None if the API returns 204 No Content (code has no data for that period).
    """
    params = {
        "get": GET_FIELDS,
        "E_COMMODITY": hs_code,
        "time": year_month,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    if resp.status_code == 204:
        # 204 = no data for this HS code / period combination (e.g. retired codes)
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_all(hs_codes: list[str], year_months: list[str]) -> list[list]:
    """
    Fetch and concatenate records for all HS codes × year-months, keeping one header.
    """
    all_rows: list[list] = []
    header: list[str] | None = None

    for code in hs_codes:
        for ym in year_months:
            print(f"  Fetching HS {code}  {ym} …")
            data = fetch_trade_data(code, ym)

            if data is None:
                print(f"    (no data — skipping)")
                continue

            if header is None:
                # First response — keep the header row
                header = data[0]
                all_rows.extend(data)
            else:
                # Subsequent responses — skip the repeated header row
                all_rows.extend(data[1:])

    return all_rows


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean(raw: list[list]) -> pd.DataFrame:
    """
    Convert raw list-of-lists into a tidy DataFrame.

    Column mapping:
        E_COMMODITY         → hs_code
        E_COMMODITY_LDESC   → hs_description
        CTY_CODE            → country_code
        CTY_NAME            → country_name
        ALL_VAL_YR          → export_value  (USD, cast to Int64 — nullable int)
        time                → year          (int, extracted from YYYY-12 string)
    """
    headers = raw[0]
    rows = raw[1:]

    # Census echoes the filter field (E_COMMODITY) as a second column; deduplicate
    # by position so pandas doesn't raise on duplicate column names.
    seen: dict[str, int] = {}
    deduped_headers = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            deduped_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            deduped_headers.append(h)

    df = pd.DataFrame(rows, columns=deduped_headers)

    # Drop aggregate / grouping rows; keep only individual country codes.
    # Aggregate patterns observed in Census data:
    #   "-"     → overall world total
    #   "XXXX"  → continental regions (e.g. 5XXX = Asia, 4XXX = Europe)
    #   "00xx"  → economic groupings (OECD=0022, NATO=0023, APEC=0026, etc.)
    df = df[
        (df["CTY_CODE"] != "-") &
        (~df["CTY_CODE"].str.contains("X", na=False)) &
        (~df["CTY_CODE"].str.startswith("00"))
    ].copy()

    # Rename to snake_case output columns
    df = df.rename(columns={
        "E_COMMODITY": "hs_code",
        "E_COMMODITY_LDESC": "hs_description",
        "CTY_CODE": "country_code",
        "CTY_NAME": "country_name",
        "ALL_VAL_YR": "export_value",
        "time": "year",
    })

    # Cast types — export_value may be null for suppressed cells
    df["export_value"] = pd.to_numeric(df["export_value"], errors="coerce").astype("Int64")
    # time column is "YYYY-12"; extract just the year integer
    df["year"] = df["year"].str[:4].astype(int)

    return df[["hs_code", "hs_description", "country_code", "country_name", "export_value", "year"]]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print top-10 export destinations and total export value."""
    total = df["export_value"].sum()
    print(f"\nTotal aerospace export value (2022–2024): ${total:,.0f}")

    top10 = (
        df.groupby(["country_code", "country_name"])["export_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top10["export_value_bn"] = (top10["export_value"] / 1e9).round(2)

    print("\nTop 10 export destinations by total value (all HS codes, all years):")
    print(f"  {'Rank':<5} {'Country':<35} {'Total USD (B)':>14}")
    print("  " + "-" * 56)
    for rank, row in top10.iterrows():
        print(f"  {rank + 1:<5} {row['country_name']:<35} {row['export_value_bn']:>13.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch, save, and summarize Census aerospace trade data."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching Census Bureau aerospace export data …")
    raw = fetch_all(AEROSPACE_HS_CODES, YEAR_MONTHS)

    # Save raw JSON
    raw_path = RAW_DIR / "census_trade_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Raw data saved → {raw_path}  ({len(raw) - 1} data rows)")

    # Clean and save CSV
    df = clean(raw)
    csv_path = RAW_DIR / "census_trade_clean.csv"
    df.to_csv(csv_path, index=False)
    print(f"Clean data saved → {csv_path}  ({len(df)} rows)")

    print_summary(df)


if __name__ == "__main__":
    main()
