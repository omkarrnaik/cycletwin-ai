# Phase 3 — Agent Pipeline

## Goal

Wrap the Phase-2 Random Forest model in a modular agent pipeline.

## Pipeline

User Input
→ Lifestyle Agent
→ Symptom Agent
→ Prediction Agent
→ Explainability Agent
→ Recommendation Agent
→ Report Agent

## Why each agent exists

### Lifestyle Agent
Extracts lifestyle-related signals that exist in the current PCOS dataset:
- Fast food
- Regular exercise

### Symptom Agent
Summarizes symptom/cycle inputs:
- Irregular cycle
- Weight gain
- Hair growth
- Skin darkening
- Hair loss
- Pimples/acne

### Prediction Agent
Loads the exact Phase-2 Random Forest model and produces a probability and risk-awareness band.

### Explainability Agent
Uses the Random Forest's feature importance plus patient-specific flags. In Phase 4, this can be upgraded to SHAP for per-patient explanations.

### Recommendation Agent
Converts structured findings into general wellness suggestions.

### Report Agent
Creates a PDF summary.

## Important

The agents are intentionally modular. They are not six independent LLMs. Each has a narrow responsibility and passes structured state to the next agent. This is appropriate for a college project because the agentic behavior is explainable and testable.

## Run

```bash
pip install -r requirements.txt
python run_demo.py
```
