"""
run_pipeline.py — End-to-end aerospace supply chain risk pipeline.

Execution order:
    1. unify.py          — build base supplier_segments table in SQLite
    2. concentration.py  — ALTER + UPDATE hhi_score, concentration_risk_label
    3. geographic.py     — ALTER + UPDATE geo_risk_score
    4. workforce.py      — ALTER + UPDATE workforce_risk_score
    5. composite.py      — ALTER + UPDATE composite_risk_score, risk_tier
    6. export_final_csv  — write canonical 14-column CSV from unify.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STEPS = [
    [sys.executable, str(ROOT / "src" / "transform" / "unify.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "concentration.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "geographic.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "workforce.py")],
    [sys.executable, str(ROOT / "src" / "risk" / "composite.py")],
]

# unify.py is in the same package directory; importable directly
from unify import export_final_csv  # noqa: E402


def main() -> None:
    """Run every pipeline step in order, then export the canonical CSV."""
    for cmd in STEPS:
        label = Path(cmd[1]).name
        print(f"\n{'=' * 60}\nRunning: {label}\n{'=' * 60}")
        subprocess.run(cmd, cwd=ROOT, check=True)

    print(f"\n{'=' * 60}\nExporting final CSV\n{'=' * 60}")
    df = export_final_csv()

    print("\nFirst 6 rows:")
    print(df.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
