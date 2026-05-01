"""
BLS Public Data API v1 ingestion for aerospace manufacturing employment.

Pulls monthly employment (CES series) for NAICS 3364 sub-industries covering
aircraft manufacturing, aircraft engine & parts, and other aircraft parts,
for years 2022–2024.  No API key required (v1 endpoint).

Series pulled:
    CEU3136411101 — employment, aircraft manufacturing
    CEU3136411201 — employment, aircraft engine & engine parts
    CEU3136411301 — employment, other aircraft parts & auxiliary equipment

Outputs:
    data/raw/bls_employment_raw.json   — raw JSON response from BLS
    data/raw/bls_employment_clean.csv  — tidy DataFrame with typed columns
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"

# CES (Current Employment Statistics) series for NAICS 3364 sub-industries.
# Series ID format: CEU + 31 (durable goods supersector) + 6-digit NAICS + 01 (all employees).
SERIES_IDS = [
    "CEU3133641101",  # NAICS 336411 — aircraft manufacturing
    "CEU3133641201",  # NAICS 336412 — aircraft engine & engine parts
    "CEU3133641301",  # NAICS 336413 — other aircraft parts & auxiliary equipment
]

# Human-readable labels keyed by series ID
SERIES_LABELS = {
    "CEU3133641101": "Aircraft Manufacturing",
    "CEU3133641201": "Aircraft Engine & Engine Parts",
    "CEU3133641301": "Other Aircraft Parts & Auxiliary Equipment",
}

START_YEAR = "2022"
END_YEAR = "2024"

# Resolve paths relative to this file so the script runs from any cwd
ROOT = Path(__file__).resolve().parents[2]
RAW_JSON = ROOT / "data" / "raw" / "bls_employment_raw.json"
CLEAN_CSV = ROOT / "data" / "raw" / "bls_employment_clean.csv"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_bls_data() -> dict:
    """POST to BLS v1 API and return the raw JSON response as a dict."""
    payload = {
        "seriesid": SERIES_IDS,
        "startyear": START_YEAR,
        "endyear": END_YEAR,
    }
    response = requests.post(BLS_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_bls_response(raw: dict) -> pd.DataFrame:
    """
    Flatten the BLS response structure into a tidy DataFrame.

    BLS v1 wraps each series under raw["Results"]["series"], where each
    element has a "seriesID" key and a "data" list of monthly observations.
    """
    status = raw.get("status", "UNKNOWN")
    if status != "REQUEST_SUCCEEDED":
        # Surface any BLS-level error messages before aborting
        messages = raw.get("message", [])
        raise RuntimeError(f"BLS API returned status '{status}': {messages}")

    rows = []
    for series in raw["Results"]["series"]:
        sid = series["seriesID"]
        label = SERIES_LABELS.get(sid, sid)  # fall back to raw ID if unmapped
        for obs in series["data"]:
            rows.append(
                {
                    "series_id": sid,
                    "year": int(obs["year"]),
                    # period is "M01"–"M12"; strip leading zero for readability
                    "period": obs["period"],
                    "period_name": obs["periodName"],
                    # value is a string in BLS response; cast to float (thousands)
                    "value": float(obs["value"]),
                    "series_label": label,
                }
            )

    df = pd.DataFrame(rows)
    # Sort chronologically within each series
    df = df.sort_values(["series_id", "year", "period"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print average monthly employment and peak month per series."""
    print("\n=== BLS Aerospace Employment Summary ===")
    print(f"  Years: {START_YEAR}–{END_YEAR}  |  Rows: {len(df)}\n")

    for sid, group in df.groupby("series_id"):
        label = SERIES_LABELS.get(sid, sid)
        avg = group["value"].mean()
        peak_row = group.loc[group["value"].idxmax()]
        print(f"  {label}")
        print(f"    Avg monthly employment : {avg:,.1f} thousand")
        print(
            f"    Peak month             : {peak_row['period_name']} {peak_row['year']}"
            f"  ({peak_row['value']:,.1f} thousand)"
        )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch, save, and summarise BLS aerospace employment data."""
    print(f"Fetching BLS employment data for series: {SERIES_IDS}")

    raw = fetch_bls_data()

    # Persist raw response exactly as returned by the API
    RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSON.write_text(json.dumps(raw, indent=2))
    print(f"Raw JSON saved → {RAW_JSON}")

    df = parse_bls_response(raw)

    df.to_csv(CLEAN_CSV, index=False)
    print(f"Clean CSV saved → {CLEAN_CSV}  ({len(df)} rows)")

    print_summary(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
