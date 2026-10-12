from .base import Agent

class LifestyleAgent(Agent):
    name = "Lifestyle Agent"

    def run(self, state):
        x = state["input"]
        findings = []

        if x.get("Fast food (Y/N)") == 1:
            findings.append("Fast-food intake is recorded.")
        if x.get("Reg.Exercise(Y/N)") == 0:
            findings.append("Regular exercise is not recorded.")

        state["lifestyle"] = {
            "findings": findings,
            "status": "needs_attention" if findings else "no_flag"
        }
        state["trace"].append(self.name)
        return state