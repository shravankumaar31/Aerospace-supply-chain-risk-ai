"""
geographic.py — Day 7/21: Aerospace Supply Chain Risk AI

Computes geographic concentration risk for aerospace export HS codes and
rolls the scores up to NAICS level. Writes geo_risk_score back to the
supplier_segments table in SQLite and CSV.

Geographic risk formula per HS code:
    top_country_share  = (largest single-country export / total exports) × 100   [%]
    country_count      = number of unique destination countries

    geo_risk_score = (top_country_share * 0.7) + ((1 / country_count) * 30 * 100)
    → capped at 100

Interpretation:
    - Term 1 captures export dependence on a single buyer country.
    - Term 2 captures destination breadth: fewer countries → higher risk.

NAICS aggregation: average geo_risk_score across all HS codes mapped to the
same NAICS code (some NAICS receive multiple HS codes, e.g. 336411 gets both
8801 and 8802).

HS → NAICS crosswalk:
    8801 → 336411  (balloons, gliders, non-powered aircraft → Aircraft Manufacturing)
    8802 → 336411  (powered aircraft                        → Aircraft Manufacturing)
    8804 → 336413  (parachutes, rotochutes                  → Other Aircraft Parts)
    8805 → 336414  (aircraft launchers / carriers           → Guided Missile & Space)

Usage:
    python src/risk/geographic.py
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
DB_PATH = PROCESSED_DIR / "supply_chain.db"
CSV_PATH = PROCESSED_DIR / "supplier_segments.csv"

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# HS product code → NAICS code crosswalk
HS_TO_NAICS: dict[int, str] = {
    8801: "336411",
    8802: "336411",
    8804: "336413",
    8805: "336414",
}

NAICS_LABELS: dict[str, str] = {
    "336411": "Aircraft Manufacturing",
    "336412": "Aircraft Engine & Engine Parts Manufacturing",
    "336413": "Other Aircraft Parts & Equipment Manufacturing",
    "336414": "Guided Missile & Space Vehicle Manufacturing",
    "336415": "Guided Missile & Space Vehicle Propulsion Unit Manufacturing",
    "336419": "Other Guided Missile & Space Vehicle Parts Manufacturing",
}


# ---------------------------------------------------------------------------
# HS-level risk computation
# ---------------------------------------------------------------------------


def compute_hs_geo_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute geographic concentration risk metrics for each HS code.

    Args:
        df: Census trade data with columns [hs_code, country_name, export_value].

    Returns:
        DataFrame with one row per HS code and columns:
        hs_code, top_country, top_country_share (%), country_count, geo_risk_score.
    """
    records = []
    for hs_code, group in df.groupby("hs_code"):
        # Aggregate export value per destination country to handle any duplicate rows
        country_totals = group.groupby("country_name")["export_value"].sum()
        total = country_totals.sum()

        if total == 0:
            continue  # skip HS codes with no recorded exports

        top_country = country_totals.idxmax()
        top_country_share_pct = (country_totals.max() / total) * 100  # as percent
        country_count = len(country_totals)

        # Geo risk: 70% weight on single-buyer dependency, 30% on destination breadth
        raw_score = (top_country_share_pct * 0.7) + ((1 / country_count) * 30 * 100)
        geo_risk_score = min(raw_score, 100.0)

        records.append({
            "hs_code": hs_code,
            "top_country": top_country,
            "top_country_share": round(top_country_share_pct, 2),
            "country_count": country_count,
            "geo_risk_score": round(geo_risk_score, 2),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# NAICS-level aggregation
# ---------------------------------------------------------------------------


def aggregate_to_naics(hs_risk: pd.DataFrame) -> pd.DataFrame:
    """
    Average geo_risk_score across HS codes that share a NAICS code.

    For NAICS codes covered by multiple HS codes (e.g. 336411 has 8801 and
    8802), the top_country / top_country_share / country_count columns are
    taken from whichever HS code has the highest top_country_share so the
    printed summary highlights the worst-case exposure.

    Args:
        hs_risk: Output of compute_hs_geo_risk().

    Returns:
        DataFrame with columns: naics_code, naics_label, top_country,
        top_country_share, country_count, geo_risk_score.
    """
    hs_risk = hs_risk.copy()
    hs_risk["naics_code"] = hs_risk["hs_code"].map(HS_TO_NAICS)
    hs_risk = hs_risk.dropna(subset=["naics_code"])

    # Average score across HS codes per NAICS
    avg_score = (
        hs_risk.groupby("naics_code")["geo_risk_score"]
        .mean()
        .reset_index()
    )

    # Representative top-country info: pick HS code with highest share per NAICS
    worst_idx = hs_risk.groupby("naics_code")["top_country_share"].idxmax()
    worst = hs_risk.loc[worst_idx, ["naics_code", "top_country", "top_country_share", "country_count"]]

    naics_risk = avg_score.merge(worst, on="naics_code")
    naics_risk["naics_label"] = naics_risk["naics_code"].map(NAICS_LABELS)
    naics_risk["geo_risk_score"] = naics_risk["geo_risk_score"].round(2)

    return naics_risk


# ---------------------------------------------------------------------------
# Database update
# ---------------------------------------------------------------------------


def update_db(naics_risk: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """
    Add geo_risk_score to NAICS-keyed rows in supplier_segments via
    ALTER TABLE + UPDATE — never replaces the whole table, so other risk
    columns added by peer scripts are preserved.

    State-keyed rows receive NULL (geo risk is a NAICS-level metric).
    """
    score_map = naics_risk.set_index("naics_code")["geo_risk_score"]

    with sqlite3.connect(str(db_path)) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(supplier_segments)")}
        if "geo_risk_score" not in existing:
            conn.execute("ALTER TABLE supplier_segments ADD COLUMN geo_risk_score REAL")

        conn.execute("UPDATE supplier_segments SET geo_risk_score = NULL")
        conn.executemany(
            "UPDATE supplier_segments SET geo_risk_score = ? "
            "WHERE CAST(naics_code AS INTEGER) = CAST(? AS INTEGER) AND state IS NULL",
            [(score, naics) for naics, score in score_map.items()],
        )

        return pd.read_sql("SELECT * FROM supplier_segments", conn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Compute geographic risk scores and merge them into supplier_segments."""
    print("=== Geographic Risk — Day 7/21 ===\n")

    # Load Census Bureau export data
    df = pd.read_csv(RAW_DIR / "census_trade_clean.csv")
    df["export_value"] = pd.to_numeric(df["export_value"], errors="coerce").fillna(0)
    print(f"Loaded {len(df)} trade rows across {df['hs_code'].nunique()} HS codes\n")

    # Compute risk at HS-code level
    print("Computing HS-level geographic risk ...")
    hs_risk = compute_hs_geo_risk(df)
    print(f"  {len(hs_risk)} HS codes scored\n")

    # Roll up to NAICS level
    print("Aggregating to NAICS level ...")
    naics_risk = aggregate_to_naics(hs_risk)
    print(f"  {len(naics_risk)} NAICS codes scored\n")

    # Persist to SQLite
    print(f"Updating {DB_PATH} ...")
    updated = update_db(naics_risk, DB_PATH)
    geo_count = updated["geo_risk_score"].notna().sum()
    print(f"  Table: {len(updated)} rows, {geo_count} with geo_risk_score\n")

    # Print results table
    header = f"{'NAICS':<8}  {'Label':<48}  {'Top Country':<25}  {'Share%':>7}  {'Countries':>9}  {'GeoRisk':>7}"
    print("Results by NAICS:\n")
    print(header)
    print("-" * len(header))
    for _, row in naics_risk.sort_values("geo_risk_score", ascending=False).iterrows():
        print(
            f"{row['naics_code']:<8}  {row['naics_label']:<48}  "
            f"{row['top_country']:<25}  {row['top_country_share']:>7.1f}  "
            f"{row['country_count']:>9}  {row['geo_risk_score']:>7.1f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
