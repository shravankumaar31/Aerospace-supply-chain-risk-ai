"""
brief_generator.py — Day 13/21: Aerospace Supply Chain Risk AI

Wires the Day-12 prompts into the Anthropic Claude API to generate
decision-grade risk briefs for individual supplier segments.

Public API:
    generate_brief(segment_data) -> str
    save_brief(segment_data, brief_text) -> Path
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.ai.prompts import SYSTEM_PROMPT, build_user_prompt

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
SEGMENTS_CSV = ROOT / "data" / "processed" / "supplier_segments.csv"
BRIEFS_DIR = ROOT / "outputs" / "briefs"

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1000

_TOKEN_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def _client() -> Anthropic:
    """Build an Anthropic client from the ANTHROPIC_API_KEY env var."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env before running."
        )
    return Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Brief generation
# ---------------------------------------------------------------------------

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(Exception),
)
def generate_brief(segment_data: dict) -> str:
    """
    Generate a one-page risk brief for a single supplier-segment row.

    Calls the Anthropic Messages API with the Day-12 system prompt and a
    structured user prompt built from `segment_data`. Retries up to 3 times
    with exponential backoff on any exception.

    Parameters
    ----------
    segment_data : dict
        One row from supplier_segments (STATE or NAICS), keyed by column name.

    Returns
    -------
    str
        The brief text returned by the model.
    """
    client = _client()
    user_prompt = build_user_prompt(segment_data)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    _TOKEN_USAGE["input_tokens"] += input_tokens
    _TOKEN_USAGE["output_tokens"] += output_tokens
    _TOKEN_USAGE["calls"] += 1

    logger.info(
        "Anthropic call OK — input_tokens=%d, output_tokens=%d",
        input_tokens,
        output_tokens,
    )

    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _identifier_for(segment_data: dict) -> str:
    """Return the file-safe identifier for a segment row."""
    row_type = str(segment_data.get("row_type", "")).upper()
    if row_type == "STATE":
        state = segment_data.get("state")
        if state is None or (isinstance(state, float) and pd.isna(state)):
            raise ValueError("STATE row missing 'state' value")
        return str(state).strip().upper()
    if row_type == "NAICS":
        code = segment_data.get("naics_code")
        if code is None or (isinstance(code, float) and pd.isna(code)):
            raise ValueError("NAICS row missing 'naics_code' value")
        return str(int(float(code)))
    raise ValueError(f"Unknown row_type: {row_type!r}")


def save_brief(segment_data: dict, brief_text: str) -> Path:
    """
    Save a generated brief to outputs/briefs/{identifier}.txt.

    The identifier is the state code for STATE rows or the NAICS code for
    NAICS rows. The output directory is created if it does not exist.

    Parameters
    ----------
    segment_data : dict
        The segment row used to generate the brief.
    brief_text : str
        The brief text returned by `generate_brief`.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    identifier = _identifier_for(segment_data)
    out_path = BRIEFS_DIR / f"{identifier}.txt"
    out_path.write_text(brief_text, encoding="utf-8")
    logger.info("Saved brief to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def _pick_wi_state(df: pd.DataFrame) -> dict[str, Any]:
    """Return the WI STATE row (composite risk score == 100)."""
    match = df[(df["row_type"] == "STATE") & (df["state"] == "WI")]
    if match.empty:
        raise RuntimeError("No STATE row found for WI in supplier_segments.csv")
    return match.iloc[0].to_dict()


def _pick_aircraft_naics(df: pd.DataFrame) -> dict[str, Any]:
    """Return the NAICS 336411 (Aircraft Manufacturing) row."""
    naics = df[df["row_type"] == "NAICS"].copy()
    naics["naics_code"] = naics["naics_code"].astype("Int64").astype(str)
    match = naics[naics["naics_code"] == "336411"]
    if match.empty:
        raise RuntimeError("No NAICS 336411 row found in supplier_segments.csv")
    return match.iloc[0].to_dict()


def main() -> None:
    """Generate and save briefs for the WI STATE row and NAICS 336411."""
    print("=== Brief Generator — Day 13/21 ===\n")
    print(f"Loading {SEGMENTS_CSV}")
    df = pd.read_csv(SEGMENTS_CSV)

    rows: list[tuple[str, dict[str, Any]]] = [
        ("STATE — WI", _pick_wi_state(df)),
        ("NAICS 336411 — Aircraft Manufacturing", _pick_aircraft_naics(df)),
    ]

    for label, row in rows:
        print("\n" + "=" * 72)
        print(f"GENERATING BRIEF: {label}")
        print("=" * 72)
        brief = generate_brief(row)
        path = save_brief(row, brief)
        print(brief)
        print(f"\n[saved to {path}]")

    print("\n" + "=" * 72)
    print("TOKEN USAGE SUMMARY")
    print("=" * 72)
    print(f"Calls:         {_TOKEN_USAGE['calls']}")
    print(f"Input tokens:  {_TOKEN_USAGE['input_tokens']:,}")
    print(f"Output tokens: {_TOKEN_USAGE['output_tokens']:,}")
    print(
        f"Total tokens:  "
        f"{_TOKEN_USAGE['input_tokens'] + _TOKEN_USAGE['output_tokens']:,}"
    )


if __name__ == "__main__":
    main()
