"""
pdf_export.py — Day 19/21: Aerospace Supply Chain Risk AI

Render generated risk briefs as polished one-page PDFs with branded header,
metric panel, parsed body, and footer.

Public API:
    export_brief_pdf(segment_data, brief_text) -> Path
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[2]
SEGMENTS_CSV = ROOT / "data" / "processed" / "supplier_segments.csv"
BRIEFS_DIR = ROOT / "outputs" / "briefs"

TIER_COLORS: dict[str, tuple[int, int, int]] = {
    "High": (192, 57, 43),
    "Medium": (211, 126, 27),
    "Low": (39, 119, 78),
    "Minimal": (90, 110, 130),
}

NAVY = (20, 38, 66)
SLATE = (90, 100, 115)
RULE = (180, 188, 200)


def _is_missing(value: Any) -> bool:
    """Return True for None / NaN / empty string values."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _identifier_for(segment_data: dict) -> str:
    """Return the file-safe identifier for a STATE or NAICS row."""
    row_type = str(segment_data.get("row_type", "")).upper()
    if row_type == "STATE":
        state = segment_data.get("state")
        if _is_missing(state):
            raise ValueError("STATE row missing 'state' value")
        return str(state).strip().upper()
    if row_type == "NAICS":
        code = segment_data.get("naics_code")
        if _is_missing(code):
            raise ValueError("NAICS row missing 'naics_code' value")
        return str(int(float(code)))
    raise ValueError(f"Unknown row_type: {row_type!r}")


def _segment_title(segment_data: dict) -> str:
    """Return a human-readable segment title for the subheader."""
    row_type = str(segment_data.get("row_type", "")).upper()
    if row_type == "STATE":
        return f"State of {segment_data.get('state')}"
    if row_type == "NAICS":
        code = _identifier_for(segment_data)
        label = segment_data.get("naics_label") or ""
        return f"NAICS {code} — {label}".rstrip(" —")
    return row_type or "UNKNOWN"


def _fmt_score(value: Any) -> str:
    """Format a 0-100 score with two decimals, or 'n/a' if missing."""
    if _is_missing(value):
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_int(value: Any) -> str:
    """Format an integer count, or 'n/a' if missing."""
    if _is_missing(value):
        return "n/a"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_money(value: Any) -> str:
    """Format a dollar amount in B/M/K shorthand, or 'n/a' if missing."""
    if _is_missing(value):
        return "n/a"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if amount >= 1e9:
        return f"${amount / 1e9:.2f}B"
    if amount >= 1e6:
        return f"${amount / 1e6:.2f}M"
    if amount >= 1e3:
        return f"${amount / 1e3:.2f}K"
    return f"${amount:,.0f}"


def _metrics_for(segment_data: dict) -> list[tuple[str, str]]:
    """Return the (label, value) pairs to display in the metric panel."""
    row_type = str(segment_data.get("row_type", "")).upper()
    if row_type == "STATE":
        return [
            ("Composite Risk", _fmt_score(segment_data.get("composite_risk_score"))),
            ("HHI Score", _fmt_score(segment_data.get("hhi_score"))),
            ("Contract Value", _fmt_money(segment_data.get("total_contract_value"))),
            ("Recipients", _fmt_int(segment_data.get("recipient_count"))),
        ]
    return [
        ("Composite Risk", _fmt_score(segment_data.get("composite_risk_score"))),
        ("Geo Risk", _fmt_score(segment_data.get("geo_risk_score"))),
        ("Workforce Risk", _fmt_score(segment_data.get("workforce_risk_score"))),
    ]


_UNICODE_REPLACEMENTS: dict[str, str] = {
    "•": "-",   # bullet •
    "–": "-",   # en-dash –
    "—": "-",   # em-dash —
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "…": "...",  # ellipsis
    " ": " ",   # non-breaking space
}


def _to_latin1(text: str) -> str:
    """Replace common Unicode punctuation so it renders in the Latin-1 core font."""
    for src, dst in _UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text


class BriefPDF(FPDF):
    """A4 PDF with a fixed footer for the supply-chain brief layout."""

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(left=15, top=15, right=15)

    def footer(self) -> None:
        """Render the footer on every page."""
        self.set_y(-15)
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*SLATE)
        self.cell(
            0,
            4,
            "Generated by Claude AI | Data: USASpending.gov, Census Bureau, BLS",
            align="L",
        )
        self.cell(
            0,
            4,
            f"Generated {datetime.now().strftime('%Y-%m-%d')}",
            align="R",
        )


def _draw_header(pdf: BriefPDF, segment_data: dict) -> None:
    """Render the title, subtitle, and tier badge."""
    pdf.set_text_color(*NAVY)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 9, "AEROSPACE SUPPLY CHAIN RISK INTELLIGENCE", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(50, 60, 75)
    title = _to_latin1(_segment_title(segment_data))
    tier = str(segment_data.get("risk_tier") or "Unknown")

    badge_w = 26.0
    badge_h = 7.0
    available = pdf.w - pdf.l_margin - pdf.r_margin - badge_w - 2
    pdf.cell(available, badge_h, title, align="L")

    badge_x = pdf.get_x()
    badge_y = pdf.get_y()
    color = TIER_COLORS.get(tier, SLATE)
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.rect(badge_x, badge_y, badge_w, badge_h, style="F")
    pdf.set_xy(badge_x, badge_y)
    pdf.cell(badge_w, badge_h, tier.upper(), align="C")
    pdf.ln(badge_h + 2)

    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)


def _draw_metric_panel(pdf: BriefPDF, metrics: list[tuple[str, str]]) -> None:
    """Render the metric panel as evenly spaced columns."""
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = usable / len(metrics)
    start_y = pdf.get_y()
    label_h = 5.0
    value_h = 8.0

    for i, (label, value) in enumerate(metrics):
        x = pdf.l_margin + i * col_w
        pdf.set_xy(x, start_y)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*SLATE)
        pdf.cell(col_w, label_h, label.upper(), align="L")

        pdf.set_xy(x, start_y + label_h)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(*NAVY)
        pdf.cell(col_w, value_h, value, align="L")

    pdf.set_y(start_y + label_h + value_h + 2)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)


def _draw_body(pdf: BriefPDF, brief_text: str) -> None:
    """Parse the markdown-style brief and render it with styled sections."""
    pdf.set_text_color(35, 40, 50)
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    lines = brief_text.splitlines()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            pdf.ln(2)
            continue

        if line.startswith("# "):
            continue

        if line.startswith("## "):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*NAVY)
            pdf.multi_cell(usable_w, 6, line[3:].strip(), align="L")
            pdf.ln(1)
            continue

        if line.lstrip().startswith(("- ", "• ", "* ")):
            stripped = line.lstrip()
            bullet_body = stripped[2:].strip()
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(35, 40, 50)
            bullet_x = pdf.l_margin
            pdf.set_xy(bullet_x, pdf.get_y())
            pdf.cell(4, 5, "-")
            pdf.set_xy(bullet_x + 4, pdf.get_y())
            pdf.multi_cell(usable_w - 4, 5, bullet_body, align="L", markdown=True)
            pdf.ln(0.5)
            continue

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(35, 40, 50)
        pdf.multi_cell(usable_w, 5, line, align="L", markdown=True)
        pdf.ln(0.5)


def export_brief_pdf(segment_data: dict, brief_text: str) -> Path:
    """
    Render a brief as a styled PDF and save it under outputs/briefs/.

    The PDF includes a branded header, segment subheader with risk-tier badge,
    a metric panel (state or NAICS variants), the parsed brief body with bold
    section headers, and a footer with attribution and generation date.

    Parameters
    ----------
    segment_data : dict
        One row from supplier_segments.csv (STATE or NAICS), keyed by column.
    brief_text : str
        Markdown-flavored brief text (typically produced by the brief generator).

    Returns
    -------
    Path
        Absolute path to the saved PDF.
    """
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    identifier = _identifier_for(segment_data)
    out_path = BRIEFS_DIR / f"{identifier}.pdf"

    pdf = BriefPDF()
    pdf.add_page()

    safe_text = _escape_markdown_preserving_bold(brief_text)

    _draw_header(pdf, segment_data)
    _draw_metric_panel(pdf, _metrics_for(segment_data))
    _draw_body(pdf, safe_text)

    pdf.output(str(out_path))
    return out_path


def _escape_markdown_preserving_bold(text: str) -> str:
    """
    Sanitize the brief text for fpdf2's markdown renderer and Latin-1 font.

    Replaces Unicode punctuation with ASCII equivalents (the core Helvetica
    font is Latin-1 only) and drops markdown sequences that fpdf2 would
    otherwise interpret as strikethrough (``--``) or italic (``__``), while
    keeping ``**bold**`` runs intact.
    """
    sanitized = _to_latin1(text)
    sanitized = re.sub(r"(?<!-)--(?!-)", "-", sanitized)
    sanitized = sanitized.replace("__", "")
    return sanitized


def main() -> None:
    """Render PDFs for the highest-risk STATE (WI) and NAICS 336411."""
    df = pd.read_csv(SEGMENTS_CSV)

    states = df[df["row_type"] == "STATE"].copy()
    top_state = states.sort_values(
        "composite_risk_score", ascending=False, kind="stable"
    ).iloc[0]
    if str(top_state["state"]).upper() != "WI":
        wi_row = states[states["state"].str.upper() == "WI"]
        if not wi_row.empty:
            top_state = wi_row.iloc[0]

    naics_row = df[
        (df["row_type"] == "NAICS") & (df["naics_code"].astype(float) == 336411)
    ].iloc[0]

    saved: list[Path] = []
    for segment in (top_state.to_dict(), naics_row.to_dict()):
        identifier = _identifier_for(segment)
        txt_path = BRIEFS_DIR / f"{identifier}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Expected brief at {txt_path}")
        brief_text = txt_path.read_text(encoding="utf-8")
        pdf_path = export_brief_pdf(segment, brief_text)
        saved.append(pdf_path)

    print("Saved PDFs:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
