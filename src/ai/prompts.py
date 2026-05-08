"""
prompts.py — Day 12/21: Aerospace Supply Chain Risk AI

Builds the system + user prompts that drive the Claude-generated risk briefs.
The user-facing function, build_user_prompt(), takes one row from
supplier_segments (as a dict) and returns a structured prompt string ready
for the Anthropic SDK.

This module does NOT call the API — it only constructs prompts. Day 13 wires
these into the Claude client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
SEGMENTS_CSV = ROOT / "data" / "processed" / "supplier_segments.csv"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior aerospace procurement analyst with 15+ years of
experience advising primes and Tier-1 suppliers on supply chain risk. You write
concise, decision-grade risk briefs for procurement directors and program managers.

Your task: produce a one-page risk brief for the supplier segment described in the
user message. Ground every claim in the numeric data provided — do not invent
figures, suppliers, or events. When a metric is missing, say so explicitly rather
than guessing.

The brief MUST follow this exact structure and section order:

1. Executive Summary
   - Exactly 2 sentences. State the overall risk posture and the single most
     important driver.

2. Risk Drivers
   - Exactly 3 bullet points.
   - Each bullet must cite a specific number from the data (e.g. HHI, composite
     score, employment count, contract value, recipient count).
   - Note on metrics: the "export concentration risk score" reflects the
     percentage of exports going to a single destination country — it is a
     trade-dependency measure, NOT a measure of physical plant location or
     regional clustering. Interpret and reference it accordingly.

3. Market Context
   - One short paragraph (3-5 sentences) framing the segment within the broader
     U.S. aerospace industrial base. Reference relevant aerospace dynamics
     (consolidation, single-source exposure, regional clustering, workforce
     pipelines) where the data supports it.

4. Recommended Actions
   - Exactly 3 bullet points.
   - Each action must be concrete, owner-implied, and tied to a driver above.

Tone: analytical, neutral, executive-ready. No marketing language, no hedging
filler ("it is worth noting that..."), no emoji. Keep the full brief under
350 words.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_num(value: Any, *, decimals: int = 2) -> str:
    """Format a numeric value with thousands separators; return 'N/A' if missing."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    """Format an integer-like value with thousands separators; 'N/A' if missing."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and pd.isna(value):
        return "N/A"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_money(value: Any) -> str:
    """Format a USD value as `$X,XXX,XXX`; 'N/A' if missing."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and pd.isna(value):
        return "N/A"
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_text(value: Any) -> str:
    """Return string form of value, or 'N/A' for missing values."""
    if value is None:
        return "N/A"
    if isinstance(value, float) and pd.isna(value):
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(segment_data: dict) -> str:
    """
    Build a structured user prompt for a single supplier-segment row.

    Parameters
    ----------
    segment_data : dict
        One row from supplier_segments (STATE or NAICS), keyed by column name.

    Returns
    -------
    str
        A clean, multi-line prompt string ready to send to the Claude API.
    """
    row_type = _fmt_text(segment_data.get("row_type")).upper()
    composite = _fmt_num(segment_data.get("composite_risk_score"))
    tier = _fmt_text(segment_data.get("risk_tier"))

    if row_type == "STATE":
        identifier = _fmt_text(segment_data.get("state"))
        header = f"Supplier Segment: U.S. State — {identifier}"
        metrics = [
            f"- Row type: STATE",
            f"- State: {identifier}",
            f"- Total contract value: {_fmt_money(segment_data.get('total_contract_value'))}",
            f"- Recipient count: {_fmt_int(segment_data.get('recipient_count'))}",
            f"- HHI score (0-10000): {_fmt_num(segment_data.get('hhi_score'))}",
            f"- Concentration risk label: {_fmt_text(segment_data.get('concentration_risk_label'))}",
            f"- Composite risk score (0-100): {composite}",
            f"- Risk tier: {tier}",
        ]
    elif row_type == "NAICS":
        code = _fmt_text(segment_data.get("naics_code"))
        label = _fmt_text(segment_data.get("naics_label"))
        header = f"Supplier Segment: NAICS {code} — {label}"
        metrics = [
            f"- Row type: NAICS",
            f"- NAICS code: {code}",
            f"- NAICS label: {label}",
            f"- Export value: {_fmt_money(segment_data.get('export_value'))}",
            f"- Employment count: {_fmt_int(segment_data.get('employment_count'))}",
            f"- Export concentration risk score (0-100) — measures dependency on single export destination country: {_fmt_num(segment_data.get('geo_risk_score'))}",
            f"- Workforce risk score (0-100): {_fmt_num(segment_data.get('workforce_risk_score'))}",
            f"- Composite risk score (0-100): {composite}",
            f"- Risk tier: {tier}",
        ]
    else:
        header = f"Supplier Segment: Unknown row_type ({row_type})"
        metrics = [f"- {key}: {_fmt_text(value)}" for key, value in segment_data.items()]

    body = "\n".join(metrics)
    return (
        f"{header}\n\n"
        f"Risk metrics for this segment:\n{body}\n\n"
        "Using only the metrics above, write the one-page risk brief in the exact "
        "four-section format defined in your instructions."
    )


# ---------------------------------------------------------------------------
# Standalone preview
# ---------------------------------------------------------------------------

def _pick_top_state(df: pd.DataFrame) -> dict:
    """Return the highest-risk STATE row (composite_risk_score == 100)."""
    states = df[df["row_type"] == "STATE"].copy()
    states = states.sort_values("composite_risk_score", ascending=False)
    return states.iloc[0].to_dict()


def _pick_aircraft_naics(df: pd.DataFrame) -> dict:
    """Return the NAICS 336411 (Aircraft Manufacturing) row."""
    naics = df[df["row_type"] == "NAICS"].copy()
    naics["naics_code"] = naics["naics_code"].astype("Int64").astype(str)
    match = naics[naics["naics_code"] == "336411"]
    return match.iloc[0].to_dict()


def main() -> None:
    """Load the segments CSV, build prompts for two demo rows, print them."""
    print("=== Prompt Preview — Day 12/21 ===\n")
    print(f"Loading {SEGMENTS_CSV}")
    df = pd.read_csv(SEGMENTS_CSV)

    state_row = _pick_top_state(df)
    naics_row = _pick_aircraft_naics(df)

    print("\n" + "=" * 72)
    print("SYSTEM PROMPT")
    print("=" * 72)
    print(SYSTEM_PROMPT)

    print("\n" + "=" * 72)
    print(f"USER PROMPT — STATE ({state_row.get('state')})")
    print("=" * 72)
    print(build_user_prompt(state_row))

    print("\n" + "=" * 72)
    print(f"USER PROMPT — NAICS {naics_row.get('naics_code')}")
    print("=" * 72)
    print(build_user_prompt(naics_row))


if __name__ == "__main__":
    main()
