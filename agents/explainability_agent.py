from .base import Agent
import pickle
import pandas as pd
import numpy as np

class ExplainabilityAgent(Agent):
    name = "Explainability Agent"

    def __init__(self, model_path="models/rf_pcos_model.pkl"):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.features = bundle["features"]

    def run(self, state):
        x = state["input"]
        row = pd.DataFrame([{f: x[f] for f in self.features}], columns=self.features)

        # Phase 3 uses global RF feature importance plus patient-specific
        # directional flags. This is transparent and works without requiring
        # a separate SHAP dependency. SHAP can be added in Phase 4.
        importances = pd.Series(
            self.model.feature_importances_,
            index=self.features
        ).sort_values(ascending=False)

        patient_flags = []
        checks = [
            ("Cycle(R/I)", 1, "Irregular cycle"),
            ("Weight gain(Y/N)", 1, "Weight gain"),
            ("hair growth(Y/N)", 1, "Excess hair growth"),
            ("Skin darkening (Y/N)", 1, "Skin darkening"),
            ("Hair loss(Y/N)", 1, "Hair loss"),
            ("Pimples(Y/N)", 1, "Pimples/acne"),
            ("Fast food (Y/N)", 1, "Fast-food intake"),
            ("Reg.Exercise(Y/N)", 0, "No regular exercise")
        ]
        for col, expected, label in checks:
            if x.get(col) == expected:
                patient_flags.append(label)

        state["explanation"] = {
            "global_top_features": [
                {"feature": k, "importance": float(v)}
                for k, v in importances.head(8).items()
            ],
            "patient_signals": patient_flags,
            "note": "Feature importance describes model behavior; it does not establish causation."
        }
        state["trace"].append(self.name)
        return state
