import pickle
import pandas as pd
import shap

class SHAPExplainabilityAgent:
    """
    Produces a patient-specific explanation for the Phase-2 Random Forest.
    """

    name = "SHAP Explainability Agent"

    def __init__(self, model_path="models/rf_pcos_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open("models/feature_schema.pkl", "rb") as f:
            self.features = pickle.load(f)
        self.explainer = shap.TreeExplainer(self.model)

    def run(self, user_input):
        row = pd.DataFrame([{f: user_input[f] for f in self.features}], columns=self.features)
        shap_values = self.explainer.shap_values(row)

        # SHAP versions differ: binary RF may return a list or a 3-D array.
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            arr = shap_values
            if arr.ndim == 3:
                values = arr[0, :, 1]
            else:
                values = arr[0]

        result = pd.DataFrame({
            "feature": self.features,
            "value": [row.iloc[0][f] for f in self.features],
            "shap_value": values
        })
        result["abs_shap"] = result["shap_value"].abs()
        result = result.sort_values("abs_shap", ascending=False)

        return {
            "all_features": result,
            "top_positive": result[result.shap_value > 0].head(6),
            "top_negative": result[result.shap_value < 0].head(6)
        }
