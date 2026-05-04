"""
workforce.py — Compute workforce stress risk scores for aerospace NAICS segments.

Scores are written to the NAICS-keyed rows of the supplier_segments table in
data/processed/supply_chain.db (rows where state IS NULL).

Scoring uses absolute thresholds so each series reflects its true risk level
rather than being stretched to fill 0-100 relative to the peer set.

Component scores (each 0-100):
    peak_score      — based on % drop from peak employment
                        > 10% → 70-100 (high stress)
                        5-10% → 30-70  (moderate)
                        < 5%  → 0-30   (low)
    volatility_score — based on std-dev of MoM changes (thousands of workers)
                        > 5k  → 70-100 (high)
                        2-5k  → 30-70  (moderate)
                        < 2k  → 0-30   (low)
    trend_penalty   — 20 if avg MoM trend is negative, else 0

Final formula:
    workforce_risk_score = (peak_score * 0.5) + (volatility_score * 0.3)
                         + (trend_penalty * 0.2)
    Capped at [0, 100].

BLS series covered:
    CEU3133641101 → 336411  Aircraft Manufacturing
    CEU3133641201 → 336412  Aircraft Engine & Engine Parts Manufacturing
    CEU3133641301 → 336413  Other Aircraft Parts & Equipment Manufacturing
    (no series)   → 336414  Guided Missile & Space Vehicle Manufacturing
"""

import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
BLS_CSV = ROOT / "data" / "raw" / "bls_employment_clean.csv"
DB_PATH = ROOT / "data" / "processed" / "supply_chain.db"
OUT_CSV = ROOT / "data" / "processed" / "supplier_segments.csv"

# BLS series → NAICS code mapping
SERIES_TO_NAICS = {
    "CEU3133641101": "336411",
    "CEU3133641201": "336412",
    "CEU3133641301": "336413",
}


def load_bls(path: Path) -> pd.DataFrame:
    """Load and sort BLS employment data by series and chronological order."""
    df = pd.read_csv(path)
    # Convert period M01-M12 to an integer month for sorting
    df["month"] = df["period"].str.lstrip("M").astype(int)
    df = df.sort_values(["series_id", "year", "month"]).reset_index(drop=True)
    return df


def _peak_score(pct: float) -> float:
    """Map peak-to-current drop % to a 0-100 component score using absolute thresholds."""
    if pct > 10:
        # Linear interpolation within 70-100 for drops above 10 %
        # Clamp at 100 for drops ≥ 40 % (extreme scenario ceiling)
        return min(70 + (pct - 10) / 30 * 30, 100)
    elif pct >= 5:
        # 5-10 % → 30-70
        return 30 + (pct - 5) / 5 * 40
    else:
        # 0-5 % → 0-30
        return pct / 5 * 30


def _volatility_score(std_dev: float) -> float:
    """Map MoM std-dev (thousands of workers) to a 0-100 component score."""
    if std_dev > 5:
        return min(70 + (std_dev - 5) / 5 * 30, 100)
    elif std_dev >= 2:
        # 2-5k → 30-70
        return 30 + (std_dev - 2) / 3 * 40
    else:
        # 0-2k → 0-30
        return std_dev / 2 * 30


def compute_workforce_metrics(series_df: pd.DataFrame) -> dict:
    """
    Compute workforce stress metrics and component scores for a single BLS series.

    Returns a dict with keys:
        series_label, latest_employment, peak_employment,
        peak_to_current_ratio, volatility,
        peak_score, volatility_score, trend_penalty,
        workforce_risk_score
    """
    values = series_df["value"].astype(float).values
    label = series_df["series_label"].iloc[0]

    # Month-over-month changes (thousands of employees)
    mom_changes = np.diff(values)

    avg_mom = float(np.mean(mom_changes))
    volatility = float(np.std(mom_changes))

    latest = float(values[-1])
    peak = float(values.max())

    peak_to_current = (peak - latest) / peak * 100 if peak > 0 else 0.0

    # Component scores using absolute thresholds (independent of peer set)
    p_score = _peak_score(peak_to_current)
    v_score = _volatility_score(volatility)
    # Fixed 20-point penalty when the average monthly trend is declining
    t_penalty = 20.0 if avg_mom < 0 else 0.0

    final = max(0.0, min(100.0, p_score * 0.5 + v_score * 0.3 + t_penalty * 0.2))

    return {
        "series_label": label,
        "latest_employment": latest,
        "peak_employment": peak,
        "peak_to_current_ratio": peak_to_current,
        "volatility": volatility,
        "peak_score": round(p_score, 2),
        "volatility_score": round(v_score, 2),
        "trend_penalty": t_penalty,
        "workforce_risk_score": round(final, 2),
    }


def update_database(metrics_by_naics: dict) -> pd.DataFrame:
    """Add/update workforce_risk_score column in supplier_segments and persist."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM supplier_segments", conn)

    # Add column if it doesn't exist yet
    if "workforce_risk_score" not in df.columns:
        df["workforce_risk_score"] = None

    # Write scores only to NAICS-keyed rows (state IS NULL)
    for naics, m in metrics_by_naics.items():
        mask = df["naics_code"] == naics
        df.loc[mask, "workforce_risk_score"] = m["workforce_risk_score"]
    # 336414 has no BLS series → score stays None (already None)

    # Persist to SQLite
    df.to_sql("supplier_segments", conn, if_exists="replace", index=False)
    conn.close()

    # Persist to CSV
    df.to_csv(OUT_CSV, index=False)

    return df


def print_results(metrics_by_naics: dict) -> None:
    """Pretty-print per-series workforce risk results including component scores."""
    col_w = 112
    print("\n" + "=" * col_w)
    print("Workforce Risk Scores (absolute thresholds)")
    print("=" * col_w)
    header = (
        f"{'NAICS':<8} {'Label':<46} {'Latest':>9} {'Peak':>9} "
        f"{'Pk→Cur%':>8} {'Volatility':>11} "
        f"{'PeakScr':>8} {'VolScr':>7} {'TrendPen':>9} {'FinalScore':>11}"
    )
    print(header)
    print("-" * col_w)

    for naics, m in sorted(metrics_by_naics.items()):
        print(
            f"{naics:<8} {m['series_label']:<46} "
            f"{m['latest_employment']:>9,.1f} "
            f"{m['peak_employment']:>9,.1f} "
            f"{m['peak_to_current_ratio']:>7.2f}% "
            f"{m['volatility']:>11.2f} "
            f"{m['peak_score']:>8.2f} "
            f"{m['volatility_score']:>7.2f} "
            f"{m['trend_penalty']:>9.1f} "
            f"{m['workforce_risk_score']:>11.2f}"
        )

    print("-" * col_w)
    print("NAICS 336414 — no BLS series, workforce_risk_score = None")
    print("=" * col_w + "\n")


def main():
    """Entry point: compute workforce risk scores and persist results."""
    bls = load_bls(BLS_CSV)

    # Compute per-series metrics
    metrics_by_naics = {}
    for series_id, naics in SERIES_TO_NAICS.items():
        subset = bls[bls["series_id"] == series_id]
        if subset.empty:
            raise ValueError(f"No BLS rows found for series {series_id}")
        metrics_by_naics[naics] = compute_workforce_metrics(subset)

    # Write to DB and CSV
    update_database(metrics_by_naics)

    # Report
    print_results(metrics_by_naics)
    print(f"Updated: {DB_PATH}")
    print(f"Updated: {OUT_CSV}")


if __name__ == "__main__":
    main()
