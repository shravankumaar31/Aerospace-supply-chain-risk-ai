# Aerospace Supply Chain Risk AI

An end-to-end pipeline that ingests public aerospace supply chain data (BLS, FAA, FRED), scores supplier and commodity risk using a composite model, and generates AI-written procurement briefs via the Claude API. Results are surfaced through an interactive Streamlit dashboard with Plotly visualizations.

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Python, `requests`, BLS Public Data API |
| Data processing | `pandas` |
| Risk scoring | Custom composite model (`src/risk/`) |
| AI brief generation | Claude API (`Sonnet 4.6`) |
| Dashboard | Streamlit + Plotly |
| PDF export | fpdf2 |
| Testing | pytest |
| CI | GitHub Actions |

## Project Structure

```
src/ingest/      # API clients and raw data fetchers
src/transform/   # Cleaning, normalisation, feature engineering
src/risk/        # Risk scoring models and aggregation
src/ai/          # ClaudeAI prompt templates and brief generation
src/app/         # Streamlit dashboard
tests/           # pytest unit and integration tests
outputs/briefs/  # Generated PDF procurement briefs
notebooks/       # Exploratory analysis
```

## Setup

```bash
cp .env.example .env   # add your API keys
pip install -r requirements.txt
streamlit run src/app/dashboard.py
```

## Results

> _Placeholder — to be populated after pipeline run._
