"""
brief_generator.py — Day 14/21: Aerospace Supply Chain Risk AI

Wires the Day-12 prompts into the Anthropic Claude API to generate
decision-grade risk briefs for individual supplier segments, and
batches generation across the highest-risk states plus all NAICS rows.

Public API:
    generate_brief(segment_data) -> str
    save_brief(segment_data, brief_text) -> Path
    batch_generate_briefs(n: int = 10) -> list[Path]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
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
from tqdm import tqdm

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
MANIFEST_PATH = BRIEFS_DIR / "manifest.json"

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
# Batch generation
# ---------------------------------------------------------------------------

def _select_batch_segments(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Select the rows to include in a batch run.

    Returns the top `n` STATE rows by composite_risk_score (descending) plus
    every NAICS row, in that order. Ties are broken by the original CSV order.
    """
    states = (
        df[df["row_type"] == "STATE"]
        .sort_values("composite_risk_score", ascending=False, kind="stable")
        .head(n)
    )
    naics = df[df["row_type"] == "NAICS"]
    return pd.concat([states, naics], ignore_index=True)


def _segment_label(segment_data: dict) -> str:
    """Return a human-readable label for log/progress output."""
    row_type = str(segment_data.get("row_type", "")).upper()
    if row_type == "STATE":
        return f"STATE {segment_data.get('state')}"
    if row_type == "NAICS":
        code = segment_data.get("naics_code")
        try:
            code_str = str(int(float(code)))
        except (TypeError, ValueError):
            code_str = str(code)
        return f"NAICS {code_str}"
    return row_type or "UNKNOWN"


def _manifest_record(
    segment_data: dict,
    brief_path: Path,
    tokens_used: int,
    generated_at: str,
) -> dict[str, Any]:
    """Build a single manifest record for a generated/cached brief."""
    row_type = str(segment_data.get("row_type", "")).upper()
    state_val = segment_data.get("state")
    naics_val = segment_data.get("naics_code")

    state_out: str | None = None
    if state_val is not None and not (isinstance(state_val, float) and pd.isna(state_val)):
        state_out = str(state_val)

    naics_out: str | None = None
    if naics_val is not None and not (isinstance(naics_val, float) and pd.isna(naics_val)):
        try:
            naics_out = str(int(float(naics_val)))
        except (TypeError, ValueError):
            naics_out = str(naics_val)

    score = segment_data.get("composite_risk_score")
    score_out = None if (score is None or (isinstance(score, float) and pd.isna(score))) else float(score)

    return {
        "row_type": row_type,
        "state": state_out,
        "naics_code": naics_out,
        "composite_risk_score": score_out,
        "risk_tier": segment_data.get("risk_tier"),
        "brief_file_path": str(brief_path.relative_to(ROOT)),
        "tokens_used": tokens_used,
        "generated_at": generated_at,
    }


def batch_generate_briefs(n: int = 10) -> list[Path]:
    """
    Generate risk briefs for the top-N highest-risk STATE rows plus all NAICS rows.

    Loads ``data/processed/supplier_segments.csv``, selects the top ``n`` STATE
    rows by ``composite_risk_score`` plus every NAICS row, and ensures each has
    a saved brief in ``outputs/briefs/``. Segments whose brief file already
    exists are loaded from disk instead of re-calling the Anthropic API. After
    all briefs are produced, a ``manifest.json`` summarising the run is written
    next to the briefs.

    Parameters
    ----------
    n : int, default 10
        Number of top-risk STATE rows to include. All NAICS rows are included
        regardless of score.

    Returns
    -------
    list[Path]
        Absolute paths to every brief covered by this run, in the order they
        were processed.
    """
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", SEGMENTS_CSV)
    df = pd.read_csv(SEGMENTS_CSV)

    batch = _select_batch_segments(df, n)
    logger.info(
        "Batch selected: %d STATE rows + %d NAICS rows = %d total",
        (batch["row_type"] == "STATE").sum(),
        (batch["row_type"] == "NAICS").sum(),
        len(batch),
    )

    paths: list[Path] = []
    manifest: list[dict[str, Any]] = []

    for _, row in tqdm(
        batch.iterrows(),
        total=len(batch),
        desc="Generating briefs",
        unit="brief",
    ):
        segment = row.to_dict()
        identifier = _identifier_for(segment)
        out_path = BRIEFS_DIR / f"{identifier}.txt"
        label = _segment_label(segment)

        if out_path.exists():
            tqdm.write(f"[skip] {label}: brief already exists at {out_path.name}")
            generated_at = datetime.fromtimestamp(
                out_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            tokens_used = 0
        else:
            tqdm.write(f"[gen ] {label}: calling Anthropic API")
            before = _TOKEN_USAGE["input_tokens"] + _TOKEN_USAGE["output_tokens"]
            brief_text = generate_brief(segment)
            save_brief(segment, brief_text)
            after = _TOKEN_USAGE["input_tokens"] + _TOKEN_USAGE["output_tokens"]
            tokens_used = after - before
            generated_at = datetime.now(timezone.utc).isoformat()

        paths.append(out_path)
        manifest.append(
            _manifest_record(segment, out_path, tokens_used, generated_at)
        )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote manifest to %s", MANIFEST_PATH)

    return paths


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _print_manifest_summary() -> None:
    """Pretty-print the saved manifest as a compact table."""
    if not MANIFEST_PATH.exists():
        print("No manifest.json found.")
        return

    records = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    print("\n" + "=" * 88)
    print(f"MANIFEST SUMMARY — {len(records)} briefs ({MANIFEST_PATH})")
    print("=" * 88)
    header = f"{'#':>2}  {'TYPE':<5}  {'KEY':<10}  {'SCORE':>6}  {'TIER':<8}  {'TOKENS':>6}  PATH"
    print(header)
    print("-" * len(header))
    for i, rec in enumerate(records, start=1):
        key = rec.get("state") if rec["row_type"] == "STATE" else rec.get("naics_code")
        score = rec.get("composite_risk_score")
        score_str = f"{score:6.2f}" if isinstance(score, (int, float)) else "  n/a "
        print(
            f"{i:>2}  {rec['row_type']:<5}  {str(key or ''):<10}  {score_str}  "
            f"{str(rec.get('risk_tier') or ''):<8}  {rec['tokens_used']:>6}  "
            f"{rec['brief_file_path']}"
        )


def main() -> None:
    """Generate the Day-14 batch of briefs and print a manifest summary."""
    print("=== Brief Generator — Day 14/21 ===\n")
    paths = batch_generate_briefs(n=10)

    print("\n" + "=" * 72)
    print("TOKEN USAGE SUMMARY (this run)")
    print("=" * 72)
    print(f"Calls:         {_TOKEN_USAGE['calls']}")
    print(f"Input tokens:  {_TOKEN_USAGE['input_tokens']:,}")
    print(f"Output tokens: {_TOKEN_USAGE['output_tokens']:,}")
    print(
        f"Total tokens:  "
        f"{_TOKEN_USAGE['input_tokens'] + _TOKEN_USAGE['output_tokens']:,}"
    )
    print(f"\nBriefs covered: {len(paths)}")

    _print_manifest_summary()


if __name__ == "__main__":
    main()
