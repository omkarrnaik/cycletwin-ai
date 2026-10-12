from .base import Agent

class SymptomAgent(Agent):
    name = "Symptom Agent"

    def run(self, state):
        x = state["input"]
        mapping = {
            "Weight gain(Y/N)": "weight gain",
            "hair growth(Y/N)": "excess hair growth",
            "Skin darkening (Y/N)": "skin darkening",
            "Hair loss(Y/N)": "hair loss",
            "Pimples(Y/N)": "pimples/acne",
            "Cycle(R/I)": "irregular cycle"
        }

        findings = []
        for col, label in mapping.items():
            value = x.get(col)
            if col == "Cycle(R/I)":
                if value == 1:
                    findings.append(label)
            elif value == 1:
                findings.append(label)

        state["symptoms"] = {
            "findings": findings,
            "count": len(findings)
        }
        state["trace"].append(self.name)
        return state
