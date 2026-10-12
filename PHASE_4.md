# Phase 4 — Explainability + Dashboard

This phase uses the user's Phase-2 workflow/model:
- same cleaned PCOS dataset
- same target `PCOS (Y/N)`
- same 80/20 stratified split
- same Random Forest: `n_estimators=300, random_state=42`

## Added

1. SHAP Explainability Agent
2. Patient-specific feature contribution chart
3. Risk-awareness summary
4. Recommendation section
5. Agent pipeline visualization
6. Streamlit dashboard

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

The dashboard is a research/college prototype. SHAP explains the model's behavior; it does not establish that a feature causes PCOS.
