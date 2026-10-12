# CycleTwin AI — Phases 4 to 8

## Phase 4 — Explainability
- SHAPExplainabilityAgent
- Patient-specific SHAP contributions
- Dashboard chart explaining model output

## Phase 5 — Database
- SQLite database at `data/cycletwin.db`
- Stores users
- Stores each assessment
- Stores probability, prediction, risk band, input JSON, SHAP JSON and recommendations JSON
- Trend/history page reads from the database

## Phase 6 — Dashboard
- Streamlit + Plotly
- Custom CSS
- Purple/white product-style UI
- Sidebar navigation
- KPI cards
- SHAP charts
- Recommendations cards
- Trend chart
- Agent status cards
- Assessment, analysis, trends, recommendations and reports pages

## Phase 7 — What-if simulator
- Allows changing selected binary inputs
- Re-runs the trained model
- Shows current probability vs simulated probability
- Explicitly labels this as simulation, not causal prediction

## Phase 8 — PDF Report
- Report Agent
- Includes assessment, model probability, all input values, top SHAP drivers and recommendations
- Downloadable from Streamlit

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The current model remains the Phase-2 Random Forest built from the uploaded PCOS dataset. This is an educational prototype and not a medical diagnosis system.
