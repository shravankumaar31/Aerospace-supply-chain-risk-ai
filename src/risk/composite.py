"""
composite.py — Day 9/21: Aerospace Supply Chain Risk AI

Computes a composite risk score and risk tier for every row in supplier_segments,
combining the dimension-specific scores produced by Days 6–8:

  STATE rows  → normalize hhi_score (TEXT, 0–10000) to 0–100 scale
  NAICS rows  → weighted average of geo_risk_score and workforce_risk_score
                (0.50 / 0.50 when both present; single score used when only one)

Risk tiers:
  >= 70  → High
  >= 40  → Medium
  >=  1  → Low
  == 0   → Minimal
  None   → Unknown
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.db"

# Make src/ importable so we can call export_final_csv from unify.py
sys.path.insert(0, str(ROOT / "src" / "transform"))
from unify import export_final_csv  # noqa: E402


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def classify_tier(score: float | None) -> str:
    """Map a composite risk score to a risk tier label."""
    if score is None:
        return "Unknown"
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    if score >= 1:
        return "Low"
    return "Minimal"


# ---------------------------------------------------------------------------
# Score computation helpers
# ---------------------------------------------------------------------------

def _composite_for_naics(geo: float | None, workforce: float | None) -> float | None:
    """
    Weighted average of available NAICS-level scores.
    Falls back to whichever single score is present; returns None only when
    both are absent.
    """
    if geo is not None and workforce is not None:
        return geo * 0.50 + workforce * 0.50
    if geo is not None:
        return geo
    if workforce is not None:
        return workforce
    return None


def _add_columns_if_missing(conn: sqlite3.Connection) -> None:
    """ALTER TABLE to add composite columns; no-op when they already exist."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(supplier_segments)")}
    if "composite_risk_score" not in existing:
        conn.execute("ALTER TABLE supplier_segments ADD COLUMN composite_risk_score REAL")
    if "risk_tier" not in existing:
        conn.execute("ALTER TABLE supplier_segments ADD COLUMN risk_tier TEXT")


# ---------------------------------------------------------------------------
# Main scoring logic
# ---------------------------------------------------------------------------

def score_state_rows(conn: sqlite3.Connection) -> None:
    """
    For STATE rows: hhi_score is stored as TEXT (inherited from pandas write).
    Cast to REAL, then normalize from [0, 10000] to [0, 100].
    Zero-value states (e.g. ND, SD, WY) naturally evaluate to 0.0.
    """
    conn.execute("""
        UPDATE supplier_segments
        SET composite_risk_score = (CAST(hhi_score AS REAL) / 10000.0) * 100.0
        WHERE row_type = 'STATE'
    """)


def score_naics_rows(conn: sqlite3.Connection) -> None:
    """
    For NAICS rows: pull existing scores into Python, apply weighted logic,
    write back individually. Pure SQL CASE can't cleanly express the
    'use single score when only one is present' rule without repeating
    column references, so Python is clearer here.
    """
    rows = conn.execute(
        "SELECT naics_code, geo_risk_score, workforce_risk_score "
        "FROM supplier_segments WHERE row_type = 'NAICS'"
    ).fetchall()

    updates = []
    for naics_code, geo, workforce in rows:
        score = _composite_for_naics(geo, workforce)
        updates.append((score, naics_code))

    conn.executemany(
        "UPDATE supplier_segments "
        "SET composite_risk_score = ? "
        "WHERE row_type = 'NAICS' AND naics_code = ?",
        updates,
    )


def apply_risk_tiers(conn: sqlite3.Connection) -> None:
    """
    Classify every row into a risk tier based on composite_risk_score.
    Done in a single SQL CASE statement after scores are finalised.
    """
    conn.execute("""
        UPDATE supplier_segments
        SET risk_tier = CASE
            WHEN composite_risk_score IS NULL THEN 'Unknown'
            WHEN composite_risk_score >= 70    THEN 'High'
            WHEN composite_risk_score >= 40    THEN 'Medium'
            WHEN composite_risk_score >= 1     THEN 'Low'
            ELSE                                    'Minimal'
        END
    """)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_state_results(conn: sqlite3.Connection) -> None:
    """Print the top 10 states by composite risk score."""
    rows = conn.execute("""
        SELECT state, hhi_score, composite_risk_score, risk_tier
        FROM supplier_segments
        WHERE row_type = 'STATE'
        ORDER BY CAST(composite_risk_score AS REAL) DESC
        LIMIT 10
    """).fetchall()

    print("\nTop 10 STATE rows by composite risk score:")
    print(f"{'State':<8} {'HHI Score':>12}  {'Composite':>10}  Tier")
    print("-" * 48)
    for state, hhi, composite, tier in rows:
        hhi_val = float(hhi) if hhi not in (None, "") else 0.0
        comp_val = composite if composite is not None else 0.0
        print(f"{state:<8} {hhi_val:>12.1f}  {comp_val:>10.2f}  {tier}")


def print_naics_results(conn: sqlite3.Connection) -> None:
    """Print all 4 NAICS rows with their composite scores and tiers."""
    rows = conn.execute("""
        SELECT naics_code, naics_label, geo_risk_score, workforce_risk_score,
               composite_risk_score, risk_tier
        FROM supplier_segments
        WHERE row_type = 'NAICS'
        ORDER BY naics_code
    """).fetchall()

    print("\nAll NAICS rows — composite scores and tiers:")
    print(f"{'NAICS':<8} {'Label':<52} {'Geo':>6}  {'WF':>6}  {'Composite':>10}  Tier")
    print("-" * 96)
    for code, label, geo, wf, composite, tier in rows:
        geo_s = f"{geo:.2f}" if geo is not None else "  N/A"
        wf_s = f"{wf:.2f}" if wf is not None else "  N/A"
        comp_s = f"{composite:.2f}" if composite is not None else "  N/A"
        print(f"{code:<8} {label:<52} {geo_s:>6}  {wf_s:>6}  {comp_s:>10}  {tier}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compute composite risk scores, update SQLite, export CSV."""
    print("=== Composite Risk Scoring — Day 9/21 ===\n")

    with sqlite3.connect(DB_PATH) as conn:
        print(f"Connected to {DB_PATH}")

        _add_columns_if_missing(conn)
        print("  Added composite_risk_score and risk_tier columns (if missing)")

        print("  Scoring STATE rows (HHI normalization)...")
        score_state_rows(conn)

        print("  Scoring NAICS rows (weighted geo + workforce)...")
        score_naics_rows(conn)

        print("  Applying risk tier classification...")
        apply_risk_tiers(conn)

        conn.commit()

        print_state_results(conn)
        print_naics_results(conn)

    print("\nExporting supplier_segments.csv with all 14 columns...")
    export_final_csv()
    print("\nDone.")


if __name__ == "__main__":
    main()
