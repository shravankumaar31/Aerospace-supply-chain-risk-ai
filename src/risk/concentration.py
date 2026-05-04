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
    Add hhi_score and concentration_risk_label to supplier_segments via
    ALTER TABLE + UPDATE — never replaces the whole table, so other risk
    columns added by peer scripts are preserved.
    """
    with sqlite3.connect(db_path) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(supplier_segments)")}
        if "hhi_score" not in existing:
            conn.execute("ALTER TABLE supplier_segments ADD COLUMN hhi_score REAL")
        if "concentration_risk_label" not in existing:
            conn.execute("ALTER TABLE supplier_segments ADD COLUMN concentration_risk_label TEXT")

        # NAICS rows: HHI is not applicable at the product-category level
        conn.execute(
            "UPDATE supplier_segments "
            "SET hhi_score = NULL, concentration_risk_label = 'N/A' "
            "WHERE naics_code IS NOT NULL"
        )
        # State rows default to 0 / Low (handles states with no award records)
        conn.execute(
            "UPDATE supplier_segments "
            "SET hhi_score = 0.0, concentration_risk_label = 'Low' "
            "WHERE naics_code IS NULL AND state IS NOT NULL"
        )
        # Overwrite with actual computed HHI values
        conn.executemany(
            "UPDATE supplier_segments "
            "SET hhi_score = ?, concentration_risk_label = ? "
            "WHERE naics_code IS NULL AND state = ?",
            [
                (row["hhi_score"], row["concentration_risk_label"], row["state"])
                for _, row in hhi.iterrows()
            ],
        )

        return pd.read_sql("SELECT * FROM supplier_segments", conn)


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

    print_top10(hhi)

    print("\n=== First 10 rows of updated supplier_segments ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(updated.head(10).to_string(index=False))
    print(f"\nTotal rows confirmed: {len(updated)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
