"""
test_risk.py — Day 10/21

Pytest suite validating the risk-scoring pipeline:
  - HHI math and normalization (Day 6)
  - Geographic and workforce score bounds (Days 7–8)
  - Composite formula and tier classification (Day 9)
  - End-to-end structure of supplier_segments
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.db"

sys.path.insert(0, str(ROOT / "src"))

from risk.composite import _composite_for_naics, classify_tier, score_state_rows
from risk.concentration import compute_hhi


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_df() -> pd.DataFrame:
    """Snapshot of supplier_segments as it exists on disk."""
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}"
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM supplier_segments", conn)


# ---------------------------------------------------------------------------
# 1. HHI calculation
# ---------------------------------------------------------------------------

def test_hhi_calculation():
    """Verify HHI formula on textbook splits."""
    monopoly = pd.DataFrame([
        {"state": "AA", "recipient_name": "OnlyCo", "award_amount": 1_000_000},
    ])
    duopoly = pd.DataFrame([
        {"state": "BB", "recipient_name": "Co1", "award_amount": 500_000},
        {"state": "BB", "recipient_name": "Co2", "award_amount": 500_000},
    ])
    quartet = pd.DataFrame([
        {"state": "CC", "recipient_name": f"Co{i}", "award_amount": 250_000}
        for i in range(4)
    ])

    hhi_monopoly = compute_hhi(monopoly).set_index("state")["hhi_score"]
    hhi_duopoly = compute_hhi(duopoly).set_index("state")["hhi_score"]
    hhi_quartet = compute_hhi(quartet).set_index("state")["hhi_score"]

    assert hhi_monopoly["AA"] == pytest.approx(10_000.0)
    assert hhi_duopoly["BB"] == pytest.approx(5_000.0)
    assert hhi_quartet["CC"] == pytest.approx(2_500.0)


# ---------------------------------------------------------------------------
# 2. HHI normalization to 0–100 composite scale
# ---------------------------------------------------------------------------

def test_hhi_normalization():
    """STATE rows: composite = (hhi / 10000) * 100, exercised end-to-end."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE supplier_segments (
            state TEXT,
            row_type TEXT,
            hhi_score TEXT,
            composite_risk_score REAL
        )
    """)
    conn.executemany(
        "INSERT INTO supplier_segments (state, row_type, hhi_score) VALUES (?, 'STATE', ?)",
        [("HI", "10000"), ("MD", "5000"), ("ZR", "0")],
    )
    score_state_rows(conn)

    rows = dict(conn.execute(
        "SELECT state, composite_risk_score FROM supplier_segments"
    ).fetchall())
    conn.close()

    assert rows["HI"] == pytest.approx(100.0)
    assert rows["MD"] == pytest.approx(50.0)
    assert rows["ZR"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Risk tier thresholds
# ---------------------------------------------------------------------------

def test_risk_tier_thresholds():
    assert classify_tier(75) == "High"
    assert classify_tier(55) == "Medium"
    assert classify_tier(25) == "Low"
    assert classify_tier(0) == "Minimal"


# ---------------------------------------------------------------------------
# 4. NAICS composite formula
# ---------------------------------------------------------------------------

def test_composite_formula_naics():
    assert _composite_for_naics(60.42, 32.44) == pytest.approx(46.43)
    assert _composite_for_naics(60.42, None) == pytest.approx(60.42)
    assert _composite_for_naics(None, 32.44) == pytest.approx(32.44)
    assert _composite_for_naics(None, None) is None


# ---------------------------------------------------------------------------
# 5. geo_risk_score bounds (live data)
# ---------------------------------------------------------------------------

def test_geo_risk_bounds(db_df):
    geo = db_df["geo_risk_score"].dropna()
    assert not geo.empty, "Expected at least one geo_risk_score in the table"
    assert (geo >= 0).all()
    assert (geo <= 100).all()


# ---------------------------------------------------------------------------
# 6. workforce_risk_score bounds (live data)
# ---------------------------------------------------------------------------

def test_workforce_risk_bounds(db_df):
    wf = db_df["workforce_risk_score"].dropna()
    assert not wf.empty, "Expected at least one workforce_risk_score in the table"
    assert (wf >= 0).all()
    assert (wf <= 100).all()


# ---------------------------------------------------------------------------
# 7. composite_risk_score bounds + completeness (live data)
# ---------------------------------------------------------------------------

def test_composite_score_bounds(db_df):
    comp = db_df["composite_risk_score"]
    assert comp.isna().sum() == 0, "composite_risk_score must have no NULLs"
    assert (comp >= 0).all()
    assert (comp <= 100).all()


# ---------------------------------------------------------------------------
# 8. supplier_segments structural invariants
# ---------------------------------------------------------------------------

def test_supplier_segments_structure(db_df):
    expected_columns = {
        "naics_code", "naics_label", "state", "row_type",
        "total_contract_value", "recipient_count", "export_value",
        "employment_count", "hhi_score", "concentration_risk_label",
        "geo_risk_score", "workforce_risk_score",
        "composite_risk_score", "risk_tier",
    }

    assert len(db_df) == 56
    assert (db_df["row_type"] == "NAICS").sum() == 4
    assert (db_df["row_type"] == "STATE").sum() == 52
    assert set(db_df.columns) == expected_columns
    assert len(expected_columns) == 14

    both_populated = db_df["naics_code"].notna() & db_df["state"].notna()
    assert not both_populated.any(), "No row may carry both naics_code and state"

    zero_states = db_df[db_df["state"].isin(["ND", "SD", "WY"])]
    assert len(zero_states) == 3
    assert (zero_states["composite_risk_score"] == 0.0).all()
