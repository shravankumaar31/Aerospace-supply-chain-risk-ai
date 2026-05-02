"""
unify.py — Day 5 of 21: Aerospace Supply Chain Risk AI
Merges three cleaned data sources into a unified supplier_segments table and writes
it to both SQLite (data/processed/supply_chain.db) and CSV.
"""

import sqlite3
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROCESSED_DIR / "supply_chain.db"
CSV_PATH = PROCESSED_DIR / "supplier_segments.csv"

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

NAICS_LABELS: dict[str, str] = {
    "336411": "Aircraft Manufacturing",
    "336412": "Aircraft Engine & Engine Parts Manufacturing",
    "336413": "Other Aircraft Parts & Equipment Manufacturing",
    "336414": "Guided Missile & Space Vehicle Manufacturing",
    "336415": "Guided Missile & Space Vehicle Propulsion Unit Manufacturing",
    "336419": "Other Guided Missile & Space Vehicle Parts Manufacturing",
}

# HS product code → NAICS code (aerospace crosswalk)
HS_TO_NAICS: dict[str, str] = {
    "8801": "336411",  # balloons, gliders, non-powered aircraft → aircraft mfg
    "8802": "336411",  # powered aircraft (planes, helicopters) → aircraft mfg
    "8804": "336413",  # parachutes, rotochutes → other aircraft parts
    "8805": "336414",  # aircraft launchers, aircraft carriers → guided missile/space
}

# BLS CES series ID → NAICS code
SERIES_TO_NAICS: dict[str, str] = {
    "CEU3133641101": "336411",
    "CEU3133641201": "336412",
    "CEU3133641301": "336413",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATE_ABBREVS: set[str] = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "GU", "VI",
}


def normalize_state(value: str | None) -> str | None:
    """Return a 2-letter state abbreviation or None if the value is invalid."""
    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    return cleaned if cleaned in _STATE_ABBREVS else None


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------


def load_usaspending() -> pd.DataFrame:
    """
    Aggregate USASpending contract awards by (naics_code, state).

    NOTE: The USASpending API returns null for NAICS Code at the individual
    award level (a known upstream limitation; see Day 2 notes). All rows
    therefore have an empty naics_code. The aggregation is preserved as-is so
    that contract totals remain available at the state level; naics_code will
    be NaN for these rows in the unified table.
    """
    df = pd.read_csv(RAW_DIR / "usaspending_clean.csv", dtype=str)

    # Normalize state to 2-letter abbreviation
    df["state"] = df["state"].apply(normalize_state)

    # Treat empty naics_code as NaN so groupby keeps it as a proper null
    df["naics_code"] = df["naics_code"].replace("", pd.NA)

    df["award_amount"] = pd.to_numeric(df["award_amount"], errors="coerce")

    aggregated = (
        df.groupby(["naics_code", "state"], dropna=False)
        .agg(
            total_contract_value=("award_amount", "sum"),
            recipient_count=("award_amount", "count"),
        )
        .reset_index()
    )

    return aggregated


def load_census_trade() -> pd.DataFrame:
    """
    Aggregate Census Bureau export values by NAICS code via the HS crosswalk.

    Sums export_value across all destination countries for each mapped NAICS code.
    HS codes without a crosswalk entry are dropped.
    """
    df = pd.read_csv(RAW_DIR / "census_trade_clean.csv", dtype=str)
    df["export_value"] = pd.to_numeric(df["export_value"], errors="coerce")

    # Map HS code → NAICS code; rows without a mapping are excluded
    df["naics_code"] = df["hs_code"].map(HS_TO_NAICS)
    df = df.dropna(subset=["naics_code"])

    aggregated = (
        df.groupby("naics_code")["export_value"]
        .sum()
        .reset_index()
        .rename(columns={"export_value": "export_value"})
    )

    return aggregated


def load_bls_employment() -> pd.DataFrame:
    """
    Derive employment counts by NAICS code from BLS CES series.

    Values are monthly employment in thousands; we average across all available
    months to get a representative annual mean, then convert to whole workers.
    Series IDs without a NAICS crosswalk entry are dropped.
    """
    df = pd.read_csv(RAW_DIR / "bls_employment_clean.csv", dtype=str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["naics_code"] = df["series_id"].map(SERIES_TO_NAICS)
    df = df.dropna(subset=["naics_code"])

    aggregated = (
        df.groupby("naics_code")["value"]
        .mean()
        # BLS reports employment in thousands of workers; convert to whole workers
        .mul(1000)
        .round(0)
        .astype(int)
        .reset_index()
        .rename(columns={"value": "employment_count"})
    )

    return aggregated


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------


def build_unified_table(
    usaspending: pd.DataFrame,
    census: pd.DataFrame,
    bls: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the three source dataframes into a single supplier_segments table.

    The merge strategy:
    - Outer-join usaspending (naics_code × state) with census (naics_code only)
      and BLS (naics_code only) on naics_code so that every combination of
      NAICS code and state present in any source appears in the output.
    - Rows from usaspending where naics_code is NaN preserve state-level
      contract totals even when NAICS is unavailable upstream.
    - naics_label is derived from NAICS_LABELS; rows with unrecognised or null
      naics_code receive a null label.
    """
    # Start from usaspending as the spine (preserves state dimension)
    merged = usaspending.copy()

    # Left-join census export_value on naics_code
    merged = merged.merge(census, on="naics_code", how="outer")

    # Left-join BLS employment_count on naics_code
    merged = merged.merge(bls, on="naics_code", how="outer")

    # Attach human-readable NAICS label
    merged["naics_label"] = merged["naics_code"].map(NAICS_LABELS)

    # Enforce canonical column order
    final_cols = [
        "naics_code",
        "naics_label",
        "state",
        "total_contract_value",
        "recipient_count",
        "export_value",
        "employment_count",
    ]
    return merged[final_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------


def write_sqlite(df: pd.DataFrame) -> None:
    """Write the unified dataframe to the supplier_segments table in SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("supplier_segments", conn, if_exists="replace", index=False)


def print_summary(df: pd.DataFrame) -> None:
    """Print schema and first 5 rows of the unified table."""
    print("\n=== supplier_segments schema ===")
    print(df.dtypes.to_string())

    print("\n=== First 5 rows ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.head(5).to_string(index=False))

    print(f"\nTotal rows: {len(df)}")
    print(f"SQLite:     {DB_PATH}")
    print(f"CSV:        {CSV_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Load, transform, merge, and persist the unified supplier_segments table."""
    print("Loading sources...")
    usaspending = load_usaspending()
    census = load_census_trade()
    bls = load_bls_employment()

    print(
        f"  USASpending rows (by naics×state): {len(usaspending)}"
        f"  (note: naics_code is null for all rows — upstream API limitation)"
    )
    print(f"  Census export rows (by naics):     {len(census)}")
    print(f"  BLS employment rows (by naics):    {len(bls)}")

    print("\nBuilding unified table...")
    unified = build_unified_table(usaspending, census, bls)

    print("Writing outputs...")
    write_sqlite(unified)
    unified.to_csv(CSV_PATH, index=False)

    print_summary(unified)


if __name__ == "__main__":
    main()
