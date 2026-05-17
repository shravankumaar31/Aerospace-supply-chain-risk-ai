"""Streamlit dashboard for the aerospace supply chain risk intelligence project.

Renders a choropleth map, filterable segment table, detail panel with metrics
and AI-generated risk briefs, and sector overview cards. Reads the unified
supplier_segments.csv produced by the transform pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.brief_generator import generate_brief, save_brief  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "supplier_segments.csv"
BRIEFS_DIR = PROJECT_ROOT / "outputs" / "briefs"

RISK_TIER_COLORS = {
    "High": "#d9534f",
    "Medium": "#f0ad4e",
    "Low": "#5cb85c",
    "Minimal": "#9e9e9e",
}

DISPLAY_COLUMNS = [
    "row_type",
    "state",
    "naics_label",
    "composite_risk_score",
    "risk_tier",
    "hhi_score",
    "geo_risk_score",
    "workforce_risk_score",
]

COLUMN_RENAME = {
    "row_type": "Type",
    "state": "State",
    "naics_label": "Sector",
    "composite_risk_score": "Risk Score",
    "risk_tier": "Risk Tier",
    "hhi_score": "HHI",
    "geo_risk_score": "Geo Risk",
    "workforce_risk_score": "Workforce Risk",
}

SECTOR_SHORT_NAMES = {
    "336411": "Aircraft Mfg",
    "336412": "Engines & Parts",
    "336413": "Other Aircraft Parts",
    "336414": "Missiles & Space",
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Load the supplier_segments CSV from disk, with Streamlit caching.

    Args:
        path: Path to ``supplier_segments.csv``.

    Returns:
        The full supplier-segments DataFrame.
    """
    return pd.read_csv(path)


def brief_identifier(row: pd.Series) -> Optional[str]:
    """Return the file-stem identifier for a segment row's brief.

    Args:
        row: One row from supplier_segments.

    Returns:
        Two-letter state code for STATE rows, NAICS code string for NAICS
        rows, or ``None`` when neither is available.
    """
    if row.get("row_type") == "STATE" and pd.notna(row.get("state")):
        return str(row["state"])
    if row.get("row_type") == "NAICS" and pd.notna(row.get("naics_code")):
        return str(int(row["naics_code"]))
    return None


def load_brief(identifier: str) -> Optional[str]:
    """Load a saved brief from ``outputs/briefs/``.

    Args:
        identifier: State code or NAICS code stem.

    Returns:
        Brief text, or ``None`` if no file exists for the identifier.
    """
    path = BRIEFS_DIR / f"{identifier}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def risk_badge(tier: str) -> str:
    """Render a colored HTML pill for a risk tier.

    Args:
        tier: Risk tier label (``High``/``Medium``/``Low``/``Minimal``).

    Returns:
        Inline HTML span styled with the tier color.
    """
    color = RISK_TIER_COLORS.get(tier, "#6c757d")
    return (
        f"<span style='background-color:{color};color:white;"
        f"padding:4px 12px;border-radius:12px;font-weight:600;"
        f"font-size:0.85rem;'>{tier}</span>"
    )


def fmt_number(val: Any, decimals: int = 2) -> str:
    """Format a numeric value with thousands separators.

    Args:
        val: Numeric value (may be ``None`` or NaN).
        decimals: Decimal places to render.

    Returns:
        Formatted number, or an em-dash for missing values.
    """
    if val is None or pd.isna(val):
        return "—"
    return f"{val:,.{decimals}f}"


def fmt_metric(val: Any, na_label: str = "N/A", decimals: int = 2) -> str:
    """Format a metric value for ``st.metric`` display.

    Args:
        val: Numeric value (may be ``None`` or NaN).
        na_label: Label to display when the value is missing.
        decimals: Decimal places to render.

    Returns:
        Formatted number, or ``na_label`` when missing.
    """
    if val is None or pd.isna(val):
        return na_label
    return f"{val:,.{decimals}f}"


def fmt_int(val: Any) -> str:
    """Format an integer count with thousands separators, or ``N/A``.

    Args:
        val: Integer-like value (may be ``None`` or NaN).

    Returns:
        Formatted integer string.
    """
    if val is None or pd.isna(val):
        return "N/A"
    return f"{int(val):,}"


def fmt_money(val: Any) -> str:
    """Format a USD amount with B/M shorthand for large values.

    Args:
        val: Dollar amount (may be ``None`` or NaN).

    Returns:
        Compact dollar string (``$1.23B``/``$45.6M``) or ``N/A``.
    """
    if val is None or pd.isna(val):
        return "N/A"
    if val >= 1e9:
        return f"${val / 1e9:,.2f}B"
    if val >= 1e6:
        return f"${val / 1e6:,.2f}M"
    return f"${val:,.0f}"


def segment_label(row: pd.Series) -> str:
    """Build a human-readable label for a supplier-segment row.

    Args:
        row: One row from supplier_segments.

    Returns:
        ``State: XX`` for STATE rows or ``NAICS NNNNNN — Label`` for NAICS rows.
    """
    if row.get("row_type") == "STATE":
        return f"State: {row.get('state', '')}"
    return f"NAICS {row.get('naics_code', '')} — {row.get('naics_label', '')}"


def render_detail_panel(row: pd.Series) -> None:
    """Render the selected segment's metrics and risk-brief panel.

    Args:
        row: One row from supplier_segments to display.
    """
    st.markdown("### Selected Segment")

    name_col, tier_col = st.columns([3, 1])
    with name_col:
        st.markdown(f"#### {segment_label(row)}")
    with tier_col:
        tier = row.get("risk_tier") if pd.notna(row.get("risk_tier")) else "N/A"
        st.markdown(risk_badge(tier), unsafe_allow_html=True)

    row_type = row.get("row_type")
    is_state = row_type == "STATE"
    is_naics = row_type == "NAICS"

    m1, m2, m3 = st.columns(3)
    m1.metric("Composite Risk", fmt_metric(row.get("composite_risk_score")))
    if is_state:
        m2.metric("Geo Risk", "N/A", delta="State-level metric", delta_color="off")
        m3.metric("Workforce Risk", "N/A", delta="State-level metric", delta_color="off")
    else:
        m2.metric("Geo Risk", fmt_metric(row.get("geo_risk_score")))
        m3.metric("Workforce Risk", fmt_metric(row.get("workforce_risk_score")))

    m4, m5, m6 = st.columns(3)
    if is_naics:
        m4.metric("HHI", "N/A", delta="Sector-level metric", delta_color="off")
    else:
        m4.metric("HHI", fmt_metric(row.get("hhi_score")))
    m5.metric("Contract Value", fmt_money(row.get("total_contract_value")))
    m6.metric("Recipients", fmt_int(row.get("recipient_count")))

    identifier = brief_identifier(row)
    st.markdown("#### Risk Brief")

    if identifier is None:
        st.info("No brief identifier available for this segment.")
        return

    existing_brief = load_brief(identifier)
    brief_slot = st.empty()

    if existing_brief:
        with brief_slot.container(border=True):
            st.markdown(existing_brief)
        button_label = "Regenerate Brief"
    else:
        button_label = "Generate AI Brief"

    if st.button(button_label, key=f"gen_brief_{identifier}"):
        with st.spinner("Generating brief with Claude AI..."):
            try:
                segment_data = row.to_dict()
                new_brief = generate_brief(segment_data)
                save_brief(segment_data, new_brief)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to generate brief: {exc}")
            else:
                with brief_slot.container(border=True):
                    st.markdown(new_brief)
                st.success("Brief generated successfully")
    elif not existing_brief:
        brief_slot.info("No brief available for this segment. Click 'Generate AI Brief' to create one.")


def build_choropleth(state_df: pd.DataFrame) -> Figure:
    """Build the USA-state choropleth of composite risk scores.

    Args:
        state_df: DataFrame restricted to STATE rows with non-null
            ``state`` and ``composite_risk_score``.

    Returns:
        A Plotly Figure ready to pass to ``st.plotly_chart``.
    """
    fig = px.choropleth(
        state_df,
        locations="state",
        locationmode="USA-states",
        color="composite_risk_score",
        scope="usa",
        color_continuous_scale="RdYlGn_r",
        range_color=(0, 100),
        hover_data={
            "state": True,
            "composite_risk_score": ":.2f",
            "risk_tier": True,
            "total_contract_value": ":,.0f",
            "recipient_count": ":,.0f",
            "hhi_score": ":.2f",
        },
        labels={
            "composite_risk_score": "Risk Score",
            "risk_tier": "Risk Tier",
            "total_contract_value": "Contract Value",
            "recipient_count": "Recipients",
            "hhi_score": "HHI",
        },
        title="Aerospace Supply Chain Concentration Risk by State",
        height=550,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_colorbar=dict(title="Risk Score"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        geo=dict(bgcolor="white", lakecolor="white"),
    )
    return fig


def render_sector_cards(df: pd.DataFrame) -> None:
    """Render one card per NAICS sector with composite score and sub-metrics.

    Args:
        df: Full supplier_segments DataFrame; NAICS rows are filtered internally.
    """
    st.markdown("### Sector Risk Overview")
    naics_rows = df[df["row_type"] == "NAICS"].copy()
    if naics_rows.empty:
        st.info("No NAICS sector data available.")
        return

    naics_rows = naics_rows.sort_values("naics_code")
    cols = st.columns(len(naics_rows))
    for col, (_, row) in zip(cols, naics_rows.iterrows()):
        code = str(int(row["naics_code"]))
        short = SECTOR_SHORT_NAMES.get(code, row.get("naics_label", code))
        tier = row.get("risk_tier") if pd.notna(row.get("risk_tier")) else "N/A"
        with col:
            with st.container(border=True):
                st.markdown(f"**{short}**")
                st.caption(f"NAICS {code}")
                st.metric("Composite", fmt_number(row.get("composite_risk_score")))
                st.markdown(risk_badge(tier), unsafe_allow_html=True)
                st.markdown(
                    f"<div style='margin-top:8px;font-size:0.85rem;'>"
                    f"Geo: <b>{fmt_number(row.get('geo_risk_score'))}</b> · "
                    f"Workforce: <b>{fmt_number(row.get('workforce_risk_score'))}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def find_default_selection(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the highest-risk STATE row, used as the default panel selection.

    Args:
        df: Full supplier_segments DataFrame.

    Returns:
        The top STATE row by ``composite_risk_score``, or ``None`` if no
        STATE rows exist.
    """
    states = df[df["row_type"] == "STATE"].copy()
    if states.empty:
        return None
    top = states.sort_values("composite_risk_score", ascending=False, na_position="last")
    return top.iloc[0]


def main() -> None:
    """Compose and render the Streamlit dashboard page."""
    st.set_page_config(
        page_title="Aerospace Supply Chain Risk Intelligence",
        layout="wide",
    )

    df = load_data(DATA_PATH)

    st.sidebar.title("Aerospace Supply Chain Risk Intelligence")

    tier_options = ["High", "Medium", "Low", "Minimal"]
    selected_tiers = st.sidebar.multiselect(
        "Risk Tier",
        options=tier_options,
        default=tier_options,
    )

    row_type_choice = st.sidebar.radio(
        "Row Type",
        options=["All", "STATE only", "NAICS only"],
        index=0,
    )

    min_score = st.sidebar.slider(
        "Min composite risk score",
        min_value=0,
        max_value=100,
        value=0,
    )

    filtered = df.copy()
    if selected_tiers:
        filtered = filtered[filtered["risk_tier"].isin(selected_tiers)]
    else:
        filtered = filtered.iloc[0:0]

    if row_type_choice == "STATE only":
        filtered = filtered[filtered["row_type"] == "STATE"]
    elif row_type_choice == "NAICS only":
        filtered = filtered[filtered["row_type"] == "NAICS"]

    filtered = filtered[filtered["composite_risk_score"].fillna(0) >= min_score]
    st.sidebar.markdown(f"**Records after filtering:** {len(filtered)}")

    st.title("Aerospace Supply Chain Risk Intelligence")
    st.caption("Powered by USASpending.gov · Census Bureau · BLS · Claude AI")

    total_segments = len(df)
    high_risk_count = int((df["risk_tier"] == "High").sum())
    highest_score = df["composite_risk_score"].max()
    state_rows_all = df[df["row_type"] == "STATE"]
    total_contract_value_b = state_rows_all["total_contract_value"].sum() / 1e9

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total segments analyzed", f"{total_segments:,}")
    c2.metric("High risk segments", f"{high_risk_count:,}")
    c3.metric(
        "Highest risk score",
        f"{highest_score:.1f}" if pd.notna(highest_score) else "N/A",
    )
    c4.metric("Total contract value", f"${total_contract_value_b:,.2f}B")

    # ---------- SECTION 1: Choropleth Map ----------
    state_df = df[df["row_type"] == "STATE"].copy()
    state_df = state_df.dropna(subset=["state", "composite_risk_score"])

    fig = build_choropleth(state_df)
    map_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="risk_map",
    )

    st.markdown(
        "<div style='text-align:center;font-size:0.9rem;margin-top:-10px;'>"
        "<span style='color:#5cb85c;font-size:1.1rem;'>&#9679;</span> Low risk (0–40)"
        "&nbsp;&nbsp;&nbsp;"
        "<span style='color:#f0ad4e;font-size:1.1rem;'>&#9679;</span> Medium risk (40–70)"
        "&nbsp;&nbsp;&nbsp;"
        "<span style='color:#d9534f;font-size:1.1rem;'>&#9679;</span> High risk (70–100)"
        "</div>",
        unsafe_allow_html=True,
    )

    clicked_state: Optional[str] = None
    if map_event and getattr(map_event, "selection", None):
        points = map_event.selection.get("points", []) if isinstance(map_event.selection, dict) else []
        if points:
            loc = points[0].get("location")
            if loc:
                clicked_state = loc

    # ---------- SECTION 3: Table (built early so selection can drive panel) ----------
    table_source = (
        filtered[DISPLAY_COLUMNS + ["naics_code", "total_contract_value", "recipient_count"]]
        .sort_values("composite_risk_score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    display_df = table_source[DISPLAY_COLUMNS].rename(columns=COLUMN_RENAME)
    display_df = display_df.where(display_df.notna(), "")

    st.subheader("Supplier segments")
    st.caption("Click any row to load its risk brief in the panel below.")

    table_event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="segments_table",
        column_config={
            "Risk Score": st.column_config.NumberColumn(format="%.2f"),
            "HHI": st.column_config.NumberColumn(format="%.2f"),
            "Geo Risk": st.column_config.NumberColumn(format="%.2f"),
            "Workforce Risk": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    selected_table_row: Optional[pd.Series] = None
    if table_event and getattr(table_event, "selection", None):
        rows = table_event.selection.get("rows", []) if isinstance(table_event.selection, dict) else []
        if rows:
            selected_table_row = table_source.iloc[rows[0]]

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name="supplier_segments_filtered.csv",
        mime="text/csv",
    )

    # ---------- SECTION 2: Detail Panel ----------
    st.divider()

    panel_row: Optional[pd.Series] = None
    if selected_table_row is not None:
        match = df[
            (df["row_type"] == selected_table_row["row_type"])
            & (df["state"].fillna("") == (selected_table_row["state"] if pd.notna(selected_table_row["state"]) else ""))
            & (df["naics_code"].fillna(-1) == (selected_table_row["naics_code"] if pd.notna(selected_table_row["naics_code"]) else -1))
        ]
        if not match.empty:
            panel_row = match.iloc[0]
    elif clicked_state:
        match = df[(df["row_type"] == "STATE") & (df["state"] == clicked_state)]
        if not match.empty:
            panel_row = match.iloc[0]

    if panel_row is None:
        panel_row = find_default_selection(df)

    if panel_row is not None:
        render_detail_panel(panel_row)
    else:
        st.info("No segment available to display.")

    # ---------- SECTION 4: Sector Cards ----------
    st.divider()
    render_sector_cards(df)


if __name__ == "__main__":
    main()
