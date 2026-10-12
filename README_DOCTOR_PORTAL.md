# CycleTwin AI — Final Doctor Portal

This folder is the uploaded **CycleTwin_AI_Master** project with a new
doctor-facing portal layered on top of the existing Phase 3–8 pipeline.

## Existing AI pipeline retained

The portal continues to use:

1. Lifestyle Agent
2. Symptom Agent
3. Prediction Agent
4. SHAP Explainability Agent
5. Recommendation Agent
6. Report/PDF functionality

The existing Random Forest model and feature schema are used unchanged.

## New doctor portal

- Attractive doctor login screen
- Demo doctor authentication
- Doctor dashboard
- Patient registry
- New patient creation
- Kaggle sample-patient loading
- Baseline vs follow-up assessment
- Persistent SQLite doctor/patient/assessment records
- Patient-specific SHAP explanation
- Agent trace/status
- Longitudinal probability trend
- Interval-wise follow-up comparison
- "What Changed?" feature comparison
- What-if model simulator
- Downloadable PDF report

## Run

From this folder:

```bash
pip install -r requirements_doctor_portal.txt
streamlit run app_doctor.py
```

Open the Streamlit URL, normally:

```text
http://localhost:8501
```

### Demo login

```text
Email:    doctor@cycletwin.demo
Password: 1234
```

## Recommended college demo flow

```text
Doctor Login
    ↓
Dashboard
    ↓
Kaggle Demo
    ↓
Load Random Kaggle Patient
    ↓
Health Assessment
    ↓
Run AI Assessment
    ↓
AI Analysis / SHAP
    ↓
Recommendations
    ↓
Create follow-up assessment
    ↓
Trends & History
    ↓
Interval-wise Trend
    ↓
What Changed?
    ↓
What-if Simulator
    ↓
Reports
    ↓
Generate PDF
```

## Database

The doctor portal creates:

```text
data/cycletwin_doctor.db
```

It contains:

- doctors
- patients
- assessments

The original application's database remains separate.

## Important scope

This is a college/educational research prototype. SHAP explains model
behavior; it does not establish causality. Model probability is a
risk-awareness signal and is not a medical diagnosis or treatment
recommendation.
