# Aerospace Supply Chain Risk AI

[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aerospace-supply-chain-risk-ai-lp5mayxt7onstfjvpjyhf8.streamlit.app/)

An end-to-end data pipeline and AI system that scores US aerospace supply chain risk across 50 states and 4 NAICS sectors, then generates analyst-style procurement risk briefs using the Claude API.

---

## Why This Exists

The US aerospace supply chain is dangerously concentrated, and that concentration is invisible until someone quantifies it. **47 out of 52 states score High concentration risk** when measured by the Herfindahl-Hirschman Index against five years of federal contract data. Five states are effectively single-vendor monopolies. This project turns three public government datasets into a single risk score per state and sector, and uses Claude to write the one-page procurement brief an analyst would otherwise spend a day producing.

---

## Live Demo

**App:** https://aerospace-supply-chain-risk-ai-lp5mayxt7onstfjvpjyhf8.streamlit.app/

- Filter the choropleth map by NAICS sector and risk tier to see concentration hotspots
- Click any state to see HHI, export dependency, workforce volatility, and composite score
- Generate an AI procurement brief for any high-risk segment and export it as PDF

---

## Architecture

**Ingest → Unify → Score → Composite → Brief → Dashboard**

1. **Ingest** — `src/ingest/usaspending.py`, `src/ingest/census_trade.py`, `src/ingest/bls_employment.py`
2. **Unify** — `src/transform/unify.py` merges all sources into `data/processed/supply_chain.db`
3. **Score** — `src/risk/concentration.py`, `src/risk/geographic.py`, `src/risk/workforce.py`
4. **Composite** — `src/risk/composite.py` combines scores into `composite_risk_score` + `risk_tier`
5. **Brief** — `src/ai/brief_generator.py` sends segment data to Claude API and returns an analyst-style brief
6. **Dashboard** — `src/app/app.py` Streamlit app with Plotly choropleth and AI brief panel

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas | Data transformation |
| SQLite | Unified data store |
| requests | API ingestion |
| Anthropic Claude API | AI brief generation |
| Streamlit | Interactive dashboard |
| Plotly | Choropleth map and charts |
| ReportLab | PDF export |
| pytest | Risk scoring validation |
| GitHub Actions | Weekly automated pipeline |

---

## Data Sources

| Source | What it provides | Records |
|---|---|---|
| USASpending.gov | Aerospace contract awards FY2022-2024 | 10,000 awards |
| Census Bureau International Trade | Aerospace export values by country | 718 country rows |
| BLS Public Data API | Aerospace employment by sector | 108 monthly records |

---

## Key Findings

- **47 of 52 states score High concentration risk** — the aerospace supplier base is dangerously concentrated almost everywhere in the US
- **5 states are single-vendor monopolies** (HHI = 10,000): Idaho, Maine, Puerto Rico, Rhode Island, Wisconsin
- **Aircraft Manufacturing has the highest sector risk** at composite score 46.43, driven by 14% export dependency on Saudi Arabia and high monthly employment volatility
- **California and New Mexico are the only competitive markets** with Moderate HHI scores, due to deep contractor ecosystems
- **Missouri is the highest-risk large-spend state** — $97B in contracts but composite risk score of 100 (Boeing dominance)

---

## Charts

### Top 20 Highest Risk States
![Top 20 States](outputs/chart1_top20_states.png)

### Risk Score Distribution
![Risk Distribution](outputs/chart2_risk_distribution.png)

### NAICS Sector Risk Heatmap
![NAICS Heatmap](outputs/chart3_naics_heatmap.png)

### Contract Value vs Risk Score
![Contract vs Risk](outputs/chart4_contract_vs_risk.png)

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/shravankumaar31/Aerospace-supply-chain-risk-ai.git
cd Aerospace-supply-chain-risk-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the full pipeline
python src/transform/run_pipeline.py

# 5. Run the test suite
pytest tests/

# 6. Launch the dashboard
streamlit run src/app/app.py
```

---

## Project Structure

- `src/ingest/` — Data ingestion scripts (USASpending, Census, BLS)
- `src/transform/` — Data unification and pipeline runner
- `src/risk/` — Risk scoring modules (concentration, geographic, workforce, composite)
- `src/ai/` — Claude API brief generator
- `src/app/` — Streamlit dashboard
- `data/raw/` — Raw API responses (gitignored)
- `data/processed/` — SQLite database and final CSV
- `tests/` — pytest suite (8 tests, all passing)
- `notebooks/` — EDA notebook
- `outputs/` — Charts and AI-generated briefs
- `.github/workflows/` — GitHub Actions weekly pipeline

---

## Test Suite

```
tests/test_risk.py::test_hhi_calculation              PASSED
tests/test_risk.py::test_hhi_normalization            PASSED
tests/test_risk.py::test_risk_tier_thresholds         PASSED
tests/test_risk.py::test_composite_formula_naics      PASSED
tests/test_risk.py::test_geo_risk_bounds              PASSED
tests/test_risk.py::test_workforce_risk_bounds        PASSED
tests/test_risk.py::test_composite_score_bounds       PASSED
tests/test_risk.py::test_supplier_segments_structure  PASSED
8 passed in 2.04s
```

---

## Progress

| Phase | Days | Status |
|---|---|---|
| Phase 1 — Data Ingestion | Days 1-5 | Complete |
| Phase 2 — Risk Scoring Engine | Days 6-11 | Complete |
| Phase 3 — AI Brief Generator | Days 12-17 | Complete |
| Phase 4 — Polish and Release | Days 18-21 | In progress |

---

## Future Work

- Add CMMC compliance risk layer using public DFARS data
- Integrate real-time news sentiment per supplier using Claude API
- Expand to Tier 2 and Tier 3 supplier mapping using SAM.gov data

---

## About

Built by Shravan Kumaar — Data Analytics professional with an MS in Business Analytics from Cal State East Bay and a background in AI/ML annotation pipelines at Amazon.
