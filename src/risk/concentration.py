"""
concentration.py — Day 6/21

Computes Herfindahl-Hirschman Index (HHI) concentration risk for each US state
based on recipient-level aerospace contract award data. Writes hhi_score and
concentration_risk_label back to the supplier_segments table in SQLite and CSV.

HHI formula:  sum((recipient_award_value / state_total) ** 2) * 10000
Classification:
    HHI > 2500  → High
    1500–2500   → Moderate
    < 1500      → Low
"""

import sqlite3
import pandas as pd


DB_PATH = "data/processed/supply_chain.db"
CSV_INPUT = "data/raw/usaspending_clean.csv"
CSV_OUTPUT = "data/processed/supplier_segments.csv"

HHI_HIGH = 2500
HHI_MODERATE = 1500


def load_awards(path: str) -> pd.DataFrame:
    """Load recipient-level award data and drop rows without a state."""
    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=["state"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with null state from {path}")
    return df


def compute_hhi(awards: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HHI per state from recipient-level award amounts.

    Returns a DataFrame with columns: state, hhi_score, concentration_risk_label
    """
    # Sum award_amount per (state, recipient) — a single recipient can have
    # multiple rows; aggregate first so each entity counts once per state.
    recipient_totals = (
        awards.groupby(["state", "recipient_name"], as_index=False)["award_amount"]
        .sum()
        .rename(columns={"award_amount": "recipient_total"})
    )

    state_totals = (
        recipient_totals.groupby("state")["recipient_total"]
        .sum()
        .rename("state_total")
        .reset_index()
    )

    # Merge state totals back to compute each recipient's share
    merged = recipient_totals.merge(state_totals, on="state")
    merged["share"] = merged["recipient_total"] / merged["state_total"]

    # HHI = sum of squared shares * 10000
    hhi = (
        merged.groupby("state")["share"]
        .apply(lambda s: (s**2).sum() * 10_000)
        .rename("hhi_score")
        .reset_index()
    )

    hhi["concentration_risk_label"] = hhi["hhi_score"].apply(_classify)
    return hhi


def _classify(score: float) -> str:
    """Map a raw HHI score to a risk label."""
    if score > HHI_HIGH:
        return "High"
    if score > HHI_MODERATE:
        return "Moderate"
    return "Low"


def update_db(hhi: pd.DataFrame, db_path: str) -> pd.DataFrame:
    """
    Merge hhi_score and concentration_risk_label onto state-keyed rows in
    supplier_segments, then write the full updated table back to SQLite.

    - State-keyed rows  → get computed HHI score and risk label.
    - NAICS-keyed rows  → hhi_score=None, concentration_risk_label="N/A"
      (HHI is a state-level metric; setting "N/A" explicitly prevents
      downstream scorers from crashing on null label values).

    Returns the full updated DataFrame.
    """
    with sqlite3.connect(db_path) as conn:
        segments = pd.read_sql("SELECT * FROM supplier_segments", conn)

        state_mask = segments["naics_code"].isna() & segments["state"].notna()

        # State rows: drop stale risk columns then left-join fresh HHI scores
        state_rows = (
            segments[state_mask]
            .drop(columns=["hhi_score", "concentration_risk_label"], errors="ignore")
            .copy()
            .merge(
                hhi[["state", "hhi_score", "concentration_risk_label"]],
                on="state",
                how="left",
            )
        )

        # NAICS-keyed rows: HHI is inapplicable — fix labels explicitly
        naics_rows = segments[~state_mask].copy()
        naics_rows["hhi_score"] = None
        naics_rows["concentration_risk_label"] = "N/A"

        # Reconstruct in canonical column order (base cols + risk cols appended)
        base_cols = [
            c for c in segments.columns
            if c not in ("hhi_score", "concentration_risk_label")
        ]
        final_cols = base_cols + ["hhi_score", "concentration_risk_label"]
        updated = pd.concat(
            [naics_rows[final_cols], state_rows[final_cols]],
            ignore_index=True,
        )

        updated.to_sql("supplier_segments", conn, if_exists="replace", index=False)

    return updated


def update_csv(df: pd.DataFrame, path: str) -> None:
    """Write the full updated DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"  Saved {path}  ({len(df)} rows)")


def print_top10(hhi: pd.DataFrame) -> None:
    """Print the 10 states with the highest HHI concentration scores."""
    top = hhi.nlargest(10, "hhi_score")[["state", "hhi_score", "concentration_risk_label"]]
    print("\nTop 10 states by HHI concentration risk:")
    print(f"{'State':<8} {'HHI Score':>12}  Label")
    print("-" * 35)
    for _, row in top.iterrows():
        print(f"{row['state']:<8} {row['hhi_score']:>12.1f}  {row['concentration_risk_label']}")


def main() -> None:
    print("=== Concentration Risk — Day 6/21 ===\n")

    print(f"Loading awards from {CSV_INPUT} ...")
    awards = load_awards(CSV_INPUT)
    print(f"  {len(awards)} award rows, {awards['state'].nunique()} states\n")

    print("Computing HHI per state ...")
    hhi = compute_hhi(awards)
    label_counts = hhi["concentration_risk_label"].value_counts().to_dict()
    print(f"  {len(hhi)} states scored — {label_counts}\n")

    print(f"Updating {DB_PATH} ...")
    updated = update_db(hhi, DB_PATH)
    print(f"  Table has {len(updated)} rows ({updated['hhi_score'].notna().sum()} with HHI scores)\n")

    print(f"Updating {CSV_OUTPUT} ...")
    update_csv(updated, CSV_OUTPUT)

    print_top10(hhi)
    print("\nDone.")


if __name__ == "__main__":
    main()
