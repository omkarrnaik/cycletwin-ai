from .base import Agent
import pickle
import pandas as pd

class PredictionAgent(Agent):
    name = "Prediction Agent"

    def __init__(self, model_path="models/rf_pcos_model.pkl", features_path="models/feature_schema.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(features_path, "rb") as f:
            self.features = pickle.load(f)

    def run(self, state):
        row = {feature: state["input"][feature] for feature in self.features}
        X = pd.DataFrame([row], columns=self.features)

        probability = float(self.model.predict_proba(X)[0, 1])
        label = int(probability >= 0.50)

        state["prediction"] = {
            "label": label,
            "probability": probability,
            "risk_band": (
                "Higher risk-awareness pattern" if probability >= 0.70
                else "Moderate risk-awareness pattern" if probability >= 0.40
                else "Lower risk-awareness pattern"
            )
        }
        state["trace"].append(self.name)
        return state
