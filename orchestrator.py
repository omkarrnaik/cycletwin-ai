import pickle
import pandas as pd
from agents.lifestyle_agent import LifestyleAgent
from agents.symptom_agent import SymptomAgent
from agents.prediction_agent import PredictionAgent
from agents.shap_agent import SHAPExplainabilityAgent
from agents.recommendation_agent import RecommendationAgent
from agents.report_agent import ReportAgent

class CycleTwinOrchestrator:
    def __init__(self):
        self.lifestyle = LifestyleAgent()
        self.symptom = SymptomAgent()
        self.prediction = PredictionAgent()
        self.shap = SHAPExplainabilityAgent()
        self.recommendation = RecommendationAgent()
        self.report = ReportAgent()

    def run(self, user_input):
        state = {"input": user_input, "trace": []}

        state = self.lifestyle.run(state)
        state = self.symptom.run(state)

        # Prediction Agent
        state = self.prediction.run(state)

        # SHAP Agent
        shap_result = self.shap.run(user_input)
        state["shap_rows"] = shap_result["all_features"].to_dict("records")
        state["shap_top_positive"] = shap_result["top_positive"].to_dict("records")
        state["shap_top_negative"] = shap_result["top_negative"].to_dict("records")
        state["trace"].append(self.shap.name)

        state = self.recommendation.run(state)
        return state

    def generate_report(self, state, path):
        state["report_path"] = self.report.generate(state, path)
        state["trace"].append(self.report.name)
        return state["report_path"]
