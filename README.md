# CycleTwin AI — PCOS Risk-Awareness Doctor Portal

CycleTwin AI is a research prototype that estimates a PCOS risk-awareness
probability from clinical, symptom and lifestyle inputs, explains each
prediction with SHAP, and presents results in a Streamlit doctor portal.

> **Scope:** educational/research prototype. The output is a risk-awareness
> signal, **not a medical diagnosis** and not a treatment recommendation.
> SHAP describes model behaviour; it does not establish causality.

## Quick start

```bash
pip install -r requirements_doctor_portal.txt
python -m streamlit run app_doctor.py
```

Open http://localhost:8501.

Demo login (demo only): `doctor@cycletwin.demo` / `1234`

Command-line smoke test (no browser):

```bash
python run_demo.py
```

This runs one sample patient through all agents and writes
`reports/cycletwin_report.pdf`.

## Portal features

- Doctor login, dashboard and patient registry
- New patient entry and Kaggle sample-patient loading
- Baseline and follow-up assessments with trend and "What changed?" views
- Patient-specific SHAP explanation
- What-if simulator (re-scores the model with edited inputs)
- Downloadable PDF report
- Persistent SQLite storage (`data/cycletwin_doctor.db`, created automatically,
  git-ignored)

## Pipeline

```
Input (portal form or run_demo.py)
  -> orchestrator.py
     1. Lifestyle Agent        agents/lifestyle_agent.py
     2. Symptom Agent          agents/symptom_agent.py
     3. Prediction Agent       agents/prediction_agent.py   (models/*.pkl)
     4. SHAP Agent             agents/shap_agent.py
     5. Recommendation Agent   agents/recommendation_agent.py
     6. Report Agent           agents/report_agent.py       (on demand)
```

The agents are deterministic, rule-based modules coordinated by one
orchestrator. They are not autonomous LLM agents. Only the Prediction and SHAP
agents use the trained model.

Risk bands in `prediction_agent.py` (below 0.40, 0.40–0.70, 0.70 and above)
are heuristic and have not been calibrated.

## Project layout

```
app_doctor.py            Streamlit doctor portal
doctor_database.py       SQLite storage (doctors, patients, assessments)
orchestrator.py          runs the agent pipeline
what_if.py               what-if simulation
run_demo.py              CLI smoke test
agents/                  the six agents
models/                  rf_pcos_model.pkl, feature_schema.pkl, feature_medians.json
data/pcos_clean.csv      raw Kaggle data (cycle coded 2/4/5)
data/run_final/          one reproducible data run (see below)
data/PCOS_TVAE_Synthetic_Data_Generator.ipynb   split + synthetic data
data/PCOS_Analysis.ipynb                         model training and evaluation
```

## Data and model pipeline (reproducible)

Run in this order. All seeds are 42.

1. `data/PCOS_TVAE_Synthetic_Data_Generator.ipynb`
   - Reads raw `pcos_clean.csv` (541 records, 177 positive = 32.7%).
   - Recodes `Cycle(R/I)`: 2 -> 0 (Regular), 4 -> 1 (Irregular), and the single
     record coded 5 -> 1 (Irregular).
   - Stratified split **before** any modelling: 432 train / 109 test.
   - Fits TVAE on the **train split only** and samples 1,500 synthetic records
     with the same positive rate as train. Checks that no synthetic record
     copies a train or test record.
   - Writes everything to `data/run_final/` with `run_manifest.json`
     (seed, library versions, SHA-256 hashes, quality scores).
2. `data/PCOS_Analysis.ipynb`
   - Trains Logistic Regression, Random Forest, XGBoost, SVM and Gradient
     Boosting on (a) real train only and (b) real train + synthetic.
   - Every model is scored on the same 109 real test records. The test set is
     never used by TVAE or by training.
   - Model choice uses 5-fold CV on real train data only.
   - Saves `models/rf_pcos_model.pkl`, `feature_schema.pkl` and
     `feature_medians.json` (medians from real train only).

### Main results (real test set, n = 109, 36 positive)

| Setting | RF AUC | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Real train only (deployed) | 0.932 | 0.936 | 0.833 | 0.968 | 0.896 |
| Real train + 1,500 synthetic | 0.943 | 0.881 | 0.833 | 0.811 | 0.822 |

- RF 5-fold CV AUC on real train: 0.955.
- Paired bootstrap of the AUC gain from synthetic data: 95% CI -0.018 to 0.045.
  The interval includes 0, so synthetic augmentation shows **no proven benefit**.
  The deployed model is therefore trained on real data only.
- The test set is small, so differences between models are within noise.

### Synthetic data quality (TVAE, train vs synthetic)

- Column shape score 0.84; pair-trend score 0.96.
- Weakest columns: Blood Group, II beta-HCG, Pulse rate, Hb, AMH.
- Distance to closest real record: minimum 2.29, no exact copies.
- Distance checks are not a formal privacy guarantee.

## Encoding note

`Cycle(R/I)` is 0 = Regular, 1 = Irregular everywhere (training data, saved
medians, agents, portal). Older material that uses 2/4 is obsolete.

## Known limitations

- One public Kaggle dataset (Kerala, India); no external validation.
- AMH, follicle counts and ultrasound features are close to diagnostic criteria,
  so performance is optimistic for screening without a scan.
- Small test set (109 records).
- Demo authentication stores the doctor password in plain text. Do not use real
  patient data with this portal.

## Reproducibility

Versions used (see `data/run_final/run_manifest.json`): Python 3.13.5,
numpy 2.1.3, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.15.3, torch 2.11.0,
ctgan 0.12.1. The pickled model needs the pinned scikit-learn version.