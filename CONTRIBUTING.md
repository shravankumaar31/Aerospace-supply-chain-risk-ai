# Contributing

Thanks for your interest in the Aerospace Supply Chain Risk AI project. This
guide covers everything you need to extend the risk model, run the suite, and
keep style consistent with the rest of the codebase.

---

## Repository Layout

```
src/
  ingest/       — pulls raw data from USASpending, Census, BLS
  transform/    — unifies raw inputs into supplier_segments (SQLite + CSV)
  risk/         — scoring modules (one per risk dimension) + composite
  ai/           — Claude prompt builders, brief generator, PDF export
  app/          — Streamlit dashboard
tests/          — pytest suite
data/
  raw/          — raw API outputs (gitignored)
  processed/    — supply_chain.db, supplier_segments.csv
outputs/        — charts, generated briefs, PDFs
```

The pipeline is deliberately linear: ingest → unify → score → composite → CSV.
Each risk dimension lives in its own module under `src/risk/` and writes a
single column back into `supplier_segments`.

---

## Adding a New Risk Metric Module

Every risk dimension follows the same contract: read inputs, compute a 0–100
score, write the column back to `supplier_segments` via `ALTER TABLE` + `UPDATE`
(never replace the table), and have the composite step pick it up.

### 1. Create the module file

Add a new file under `src/risk/`, named after the dimension:

```
src/risk/<dimension>.py
```

For example, `src/risk/cyber.py` for a cyber-exposure score.

### 2. Implement the scoring contract

Use the existing modules (`concentration.py`, `geographic.py`,
`workforce.py`) as the reference shape. At minimum you need:

```python
"""<dimension>.py — short description of the risk dimension.

Scores are written to <STATE|NAICS> rows of supplier_segments
in data/processed/supply_chain.db.
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "supply_chain.db"


def compute_<dimension>_risk(source_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 0–100 score for each scored entity.

    Args:
        source_df: Cleaned source data with the columns needed for scoring.

    Returns:
        DataFrame with one row per entity and a ``<dimension>_risk_score``
        column on the 0–100 scale.
    """
    ...


def update_db(scores: pd.DataFrame) -> pd.DataFrame:
    """Add the new score column to supplier_segments without dropping others.

    Args:
        scores: Output of ``compute_<dimension>_risk``.

    Returns:
        Snapshot of the updated supplier_segments table.
    """
    with sqlite3.connect(DB_PATH) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(supplier_segments)")}
        if "<dimension>_risk_score" not in existing:
            conn.execute("ALTER TABLE supplier_segments ADD COLUMN <dimension>_risk_score REAL")
        conn.execute("UPDATE supplier_segments SET <dimension>_risk_score = NULL")
        conn.executemany(
            "UPDATE supplier_segments SET <dimension>_risk_score = ? "
            "WHERE <... key match ...>",
            [...],
        )
        return pd.read_sql("SELECT * FROM supplier_segments", conn)


def main() -> None:
    """Entry point: compute scores and persist them."""
    ...


if __name__ == "__main__":
    main()
```

Key rules:

- Use **absolute thresholds**, not percentile ranking. Risk should reflect
  real-world severity, not the peer distribution.
- Always cap scores to `[0, 100]`.
- Update `supplier_segments` via `ALTER TABLE` + `UPDATE` — never
  `to_sql(..., if_exists="replace")`. Other modules' columns must survive.
- Use `Path(__file__).resolve().parents[2]` to anchor paths relative to the
  repo root so the script runs from any cwd.

### 3. Wire the module into the pipeline runner

In `src/transform/run_pipeline.py`, add your script to the `STEPS` list in the
order it should execute (typically after the source-aligned scorers and before
`composite.py`):

```python
STEPS = [
    [sys.executable, str(ROOT / "src" / "transform" / "unify.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "concentration.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "geographic.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "workforce.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "<dimension>.py")],  # new
]
```

### 4. Fold the new score into the composite

If the new dimension should influence `composite_risk_score`, edit
`src/risk/composite.py`:

- Extend `_composite_for_naics` (or add a `_composite_for_state` analogue) to
  include the new column.
- Update the weighted-average formula so weights sum to 1.0.
- If the new score is state-keyed, also update `score_state_rows`.

### 5. Update the CSV schema

In `src/transform/unify.py`, append the new column to `FINAL_COL_ORDER` so
`export_final_csv` writes it out. Add the column name to `_FLOAT_COLS` if it's
numeric — that triggers 2-dp rounding on export.

### 6. Add tests

In `tests/test_risk.py`, add at minimum:

- A unit test for the scoring function on a small fixture (verify formula
  edge cases).
- A bounds test that asserts the live column is within `[0, 100]` with no
  unexpected nulls.

See `test_geo_risk_bounds` and `test_workforce_risk_bounds` for the pattern.

### 7. Update the dashboard (optional)

If the new metric should appear in `src/app/app.py`:

- Add it to `DISPLAY_COLUMNS` and `COLUMN_RENAME`.
- Render it in `render_detail_panel` alongside the existing metrics.

### 8. Update the prompt (optional)

If the new metric should influence AI-generated briefs, add a line to
`build_user_prompt` in `src/ai/prompts.py` so Claude sees the new field.

---

## Running the Test Suite

```bash
# Run everything, verbose
pytest tests/ -v

# Run a single test
pytest tests/test_risk.py::test_hhi_calculation -v

# Stop on the first failure
pytest tests/ -x
```

Some tests (`test_geo_risk_bounds`, `test_workforce_risk_bounds`,
`test_composite_score_bounds`, `test_supplier_segments_structure`) read the
live `data/processed/supply_chain.db`. You must run the pipeline at least once
before these tests will pass — see below.

---

## Running the Full Pipeline Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Open .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 3. Pull raw data (only needed when refreshing inputs)
python src/ingest/usaspending.py
python src/ingest/census_trade.py
python src/ingest/bls_employment.py

# 4. Run the unify → score → composite pipeline end-to-end
python src/transform/run_pipeline.py

# 5. Verify it landed cleanly
pytest tests/ -v

# 6. (Optional) Launch the dashboard
streamlit run src/app/app.py
```

Outputs land in:

- `data/processed/supply_chain.db` — full SQLite store
- `data/processed/supplier_segments.csv` — canonical 14-column CSV
- `outputs/briefs/` — AI-generated briefs (when run)

---

## Code Style Guidelines

### Docstrings — Google style

Every public function (and any non-trivial private helper) takes a Google-style
docstring. The first line is a one-sentence summary; longer functions add an
`Args:` / `Returns:` / `Raises:` block.

```python
def compute_hhi(awards: pd.DataFrame) -> pd.DataFrame:
    """Compute Herfindahl-Hirschman Index per state from recipient award totals.

    Args:
        awards: Award-level rows with ``state``, ``recipient_name``,
            and ``award_amount`` columns.

    Returns:
        DataFrame with one row per state and columns ``state``, ``hhi_score``,
        ``concentration_risk_label``.
    """
```

For one-line entry points (`main`, `_classify`, etc.), a single sentence is
enough — don't pad with empty `Args:` blocks.

### Type Hints — required on every signature

All function parameters and return types must be annotated. Use `from __future__
import annotations` at the top of modules that need PEP 604 union syntax
(`int | None`) on older Python versions.

```python
def normalize_state(value: str | None) -> str | None: ...
def fetch_page(payload: dict) -> dict: ...
def main() -> None: ...
```

For pandas types, annotate as `pd.DataFrame` or `pd.Series`. For Plotly figures,
use `plotly.graph_objects.Figure`. For "anything goes" arguments, use
`typing.Any` rather than leaving the parameter bare.

### Naming and Module Conventions

- Modules: `lower_snake_case.py`
- Functions: `lower_snake_case`, private helpers prefixed with `_`
- Constants: `UPPER_SNAKE_CASE`
- Anchor file paths with `Path(__file__).resolve().parents[N]`, not relative
  strings — scripts must run from any cwd.

### Imports

Order: standard library → third-party → local, separated by blank lines. Inside
each group, sort alphabetically. Avoid `from x import *`.

### Database Writes

Always use `ALTER TABLE` + `UPDATE` to add columns to `supplier_segments`.
Never use `to_sql(..., if_exists="replace")` from a risk module — it will
wipe peer modules' columns.

### Logging vs Printing

Pipeline scripts use `print(...)` for human-readable progress output (the
runner captures stdout for the GitHub Actions log). Library code that may run
inside other processes (e.g. `src/ai/brief_generator.py`) uses the standard
`logging` module.

### Tests

- Unit tests live in `tests/test_<area>.py`.
- Use `pytest.approx` for floating-point comparisons.
- Mark integration tests that depend on the live DB clearly — they should
  read, never write.

---

## Pull Requests

Before opening a PR:

1. Pipeline runs clean: `python src/transform/run_pipeline.py`
2. Tests pass: `pytest tests/ -v`
3. New functions have Google docstrings and type hints.
4. README or CONTRIBUTING is updated if you changed user-facing behavior.

Keep PRs focused — one risk dimension, one bug fix, or one feature per PR.
