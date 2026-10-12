# CycleTwin AI — Master Build (Phases 3–8 merged)

This is the two uploaded projects (**Phase 3** and **Phases 4–8**) combined into a
single, working codebase, with three bugs fixed that were present in the
Phase 4–8 code (see "Fixes applied" below).

Phase 4–8's agents were already an exact superset of Phase 3's, so this build
uses Phase 4–8 as the base and adds back Phase 3's CLI demo script for quick
testing without launching the dashboard.

## How to run it

Open this folder in VS Code, open a terminal (`` Ctrl+` ``), then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will start a local server and print a URL — open it in your browser:

```
Local URL: http://localhost:8501
```

That's the main app: a purple/white dashboard with pages for Assessment,
Analysis (SHAP), Trends, What-if Simulator, Recommendations, and Reports.

### Optional: quick CLI check (no browser)

```bash
python run_demo.py
```

This runs one sample patient through the full pipeline and prints the
prediction, SHAP drivers, recommendations, and generates a PDF report at
`reports/cycletwin_report.pdf` — useful to sanity-check the pipeline is wired
correctly before touching the UI.

## File flow (pipeline)

```
User input (Streamlit form or run_demo.py)
        |
        v
CycleTwinOrchestrator (orchestrator.py)
        |
        v
  1. Lifestyle Agent        agents/lifestyle_agent.py
        |
        v
  2. Symptom Agent          agents/symptom_agent.py
        |
        v
  3. Prediction Agent       agents/prediction_agent.py
        |   loads models/rf_pcos_model.pkl + models/feature_schema.pkl
        v
  4. SHAP Explainability    agents/shap_agent.py
        |   per-patient feature contributions
        v
  5. Recommendation Agent   agents/recommendation_agent.py
        |
        v
  6. Report Agent (on demand)  agents/report_agent.py
        |   builds reports/cycletwin_report.pdf
        v
     Result dict --> displayed in app.py / saved via database.py
```

Supporting modules:
- `database.py` — SQLite storage (`data/cycletwin.db`) of users + past
  assessments, powering the Trends page.
- `what_if.py` — re-runs the model with edited inputs for the What-if
  Simulator page.
- `app.py` — the Streamlit dashboard; the single entry point for normal use.
- `run_demo.py` — CLI entry point for quick pipeline testing.

## Reference docs (kept from both source projects)

- `PHASE_3.md` — original Phase 3 agent-pipeline design notes
- `PHASE_4.md` — Phase 4 explainability/dashboard notes
- `PHASES_4_TO_8.md` — Phase 5–8 notes (database, dashboard, what-if, PDF report)

## Fixes applied while merging

1. **`database.py`** was missing `import json` and `import pandas as pd`,
   which it uses internally — would crash on the first save/history read.
2. **`agents/prediction_agent.py`** expected the model file to be a
   `{"model": ..., "features": ...}` bundle, but `models/rf_pcos_model.pkl`
   is actually the raw `RandomForestClassifier`. Fixed to load the model
   directly and pull the feature list from `models/feature_schema.pkl`
   (matching how `shap_agent.py` already loads it).
3. **`agents/report_agent.py`**: the orchestrator's `generate_report()`
   called a `.generate()` method that didn't exist, and the report body
   referenced a `state["explanation"]["patient_signals"]` key that the
   Phase 4–8 pipeline no longer produces (it was replaced by SHAP output).
   Added a `generate()` method and rewrote the report body to use the
   current `symptoms` and `shap_top_positive` / `shap_top_negative` state.
4. **`app.py` line 247**: a single-quoted string contained an unescaped
   apostrophe (`women's`), which closed the string early and produced a
   genuine `SyntaxError` — this would have stopped the whole app from
   starting. Fixed by switching that string to double quotes.

## Verification performed

- Every `.py` file compiled cleanly (`py_compile`) — no syntax errors.
- `python run_demo.py` runs the full pipeline and produces a PDF report.
- All 7 dashboard pages were executed programmatically via Streamlit's
  `AppTest` framework with no exceptions.
- The three interactive buttons (Run AI Assessment, Simulate change,
  Generate PDF Report) were each clicked and verified to produce real
  output: a saved row in `data/cycletwin.db` and a valid 1-page PDF at
  `reports/cycletwin_report.pdf`.

## Note on scope

This is a college/educational prototype (per the original project's own
docs) — model output is a "risk-awareness" signal, not a medical diagnosis.
