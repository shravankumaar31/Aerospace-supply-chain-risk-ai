"""
unify.py — Day 5 of 21: Aerospace Supply Chain Risk AI
Merges three cleaned data sources into a unified supplier_segments table and writes
it to both SQLite (data/processed/supply_chain.db) and CSV.

Day 7 quality fixes:
  1. Drop orphan rows with both null naics_code and null state (the 280-row
     null-state aggregate from USASpending that collapsed into a single blank row).
  2. Pad any of the 50 US states absent from supplier_segments with zero-value rows
     so the choropleth on Day 16 has complete geographic coverage.
  3. Pre-populate hhi_score=None / concentration_risk_label="N/A" on NAICS-keyed
     rows so downstream risk scorers never crash on unexpected nulls.
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

# All 50 US states required for choropleth coverage (Fix 2)
US_50_STATES: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

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

    Day 7 quality fixes applied here:
    - Fix 1: orphan rows (null naics_code AND null state) are dropped.
    - Fix 3: hhi_score and concentration_risk_label are initialised so
      downstream scorers never see unexpected nulls; NAICS-keyed rows use
      "N/A" for the label because HHI is a state-level metric only.
    """
    # Start from usaspending as the spine (preserves state dimension)
    merged = usaspending.copy()

    # Outer-join census export_value on naics_code
    merged = merged.merge(census, on="naics_code", how="outer")

    # Outer-join BLS employment_count on naics_code
    merged = merged.merge(bls, on="naics_code", how="outer")

    # Attach human-readable NAICS label
    merged["naics_label"] = merged["naics_code"].map(NAICS_LABELS)

    # Fix 1: the USASpending aggregation produces one row with naics_code=null
    # AND state=null (the 280 awards with no state recorded). Drop it — it adds
    # no geographic or sector information.
    merged = merged.dropna(subset=["naics_code", "state"], how="all")

    # Fix 3: initialise risk columns; NAICS-keyed rows can never have an HHI
    # score (HHI is computed from state-level award distributions, not NAICS).
    naics_mask = merged["naics_code"].notna()
    merged["hhi_score"] = None
    merged["concentration_risk_label"] = None
    merged.loc[naics_mask, "concentration_risk_label"] = "N/A"

    # Fix 1: label each row by its key dimension
    merged["row_type"] = None
    merged.loc[naics_mask, "row_type"] = "NAICS"
    merged.loc[merged["naics_code"].isna() & merged["state"].notna(), "row_type"] = "STATE"

    final_cols = [
        "naics_code",
        "naics_label",
        "state",
        "row_type",
        "total_contract_value",
        "recipient_count",
        "export_value",
        "employment_count",
        "hhi_score",
        "concentration_risk_label",
    ]
    return merged[final_cols].reset_index(drop=True)


def _pad_missing_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix 2: insert zero-value state rows for each of the 50 US states that
    does not already appear in the table so the choropleth on Day 16 has
    complete geographic coverage.

    Zero rows use total_contract_value=0 and recipient_count=0; all other
    numeric columns are left null so they don't distort aggregations.
    """
    state_keyed = df[df["naics_code"].isna() & df["state"].notna()]
    present = set(state_keyed["state"].dropna())
    missing = [s for s in US_50_STATES if s not in present]

    if not missing:
        return df

    zero_rows = pd.DataFrame({
        "naics_code": None,
        "naics_label": None,
        "state": missing,
        "row_type": "STATE",
        "total_contract_value": 0.0,
        "recipient_count": 0,
        "export_value": None,
        "employment_count": None,
        "hhi_score": None,
        "concentration_risk_label": None,
    })

    return pd.concat([df, zero_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

FINAL_COL_ORDER: list[str] = [
    "row_type", "naics_code", "naics_label", "state",
    "total_contract_value", "recipient_count", "export_value",
    "employment_count", "hhi_score", "concentration_risk_label",
    "geo_risk_score", "workforce_risk_score",
]

_FLOAT_COLS: list[str] = [
    "total_contract_value", "export_value", "hhi_score",
    "geo_risk_score", "workforce_risk_score",
]


def export_final_csv() -> pd.DataFrame:
    """
    Read the fully-scored supplier_segments table from SQLite, reorder columns
    to the canonical 12-column schema, round floats to 2 dp, replace NaN with
    empty string, and write the definitive CSV.

    This is the only function that should produce supplier_segments.csv after
    risk scripts have run. The risk scripts (concentration, geographic,
    workforce) must not write the CSV themselves.
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM supplier_segments", conn)

    for col in FINAL_COL_ORDER:
        if col not in df.columns:
            df[col] = None

    df = df[FINAL_COL_ORDER]

    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    df = df.where(df.notna(), other="")
    df.to_csv(CSV_PATH, index=False)
    print(f"  Saved {CSV_PATH}  ({len(df)} rows, {len(df.columns)} columns)")
    return df


def _clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 3: replace every NaN/NA value with None so CSV cells are empty string."""
    return df.where(df.notna(), other=None)


def write_sqlite(df: pd.DataFrame) -> None:
    """Write the unified dataframe to the supplier_segments table in SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("supplier_segments", conn, if_exists="replace", index=False)


def print_summary(df: pd.DataFrame) -> None:
    """Print schema, row counts, and coverage checks for the unified table."""
    print("\n=== supplier_segments schema ===")
    print(df.dtypes.to_string())

    print("\n=== First 10 rows ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(df.head(10).to_string(index=False))

    state_mask = df["naics_code"].isna() & df["state"].notna()
    naics_mask = df["naics_code"].notna()
    orphan_mask = df["naics_code"].isna() & df["state"].isna()
    missing50 = sorted(set(US_50_STATES) - set(df[state_mask]["state"].dropna()))

    print(f"\nTotal rows:           {len(df)}")
    print(f"  NAICS-keyed rows:   {naics_mask.sum()}")
    print(f"  State-keyed rows:   {state_mask.sum()}")
    print(f"  Orphan rows:        {orphan_mask.sum()}  ← should be 0")
    print(f"  Missing from 50:    {missing50}  ← should be []")
    print(f"SQLite:               {DB_PATH}")
    print(f"CSV:                  {CSV_PATH}")


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

    print("Padding missing state rows...")
    unified = _pad_missing_states(unified)

    print("Writing outputs...")
    unified = _clean_nulls(unified)
    write_sqlite(unified)
    unified.to_csv(CSV_PATH, index=False)

    print_summary(unified)


if __name__ == "__main__":
    main()
