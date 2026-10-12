from .base import Agent

class RecommendationAgent(Agent):
    name = "Recommendation Agent"

    def run(self, state):
        x = state["input"]
        recs = []

        if x.get("Reg.Exercise(Y/N)") == 0:
            recs.append("Consider adding sustainable regular physical activity, if appropriate for you.")
        if x.get("Fast food (Y/N)") == 1:
            recs.append("Consider reducing highly processed or fast-food meals.")
        if x.get("Cycle(R/I)") == 1:
            recs.append("Track cycle timing consistently and discuss persistent irregularity with a qualified clinician.")
        if state["symptoms"]["count"] >= 3:
            recs.append("Consider discussing persistent or multiple symptoms with a qualified clinician.")

        state["recommendations"] = recs or [
            "Continue tracking your cycle and maintaining consistent healthy routines."
        ]
        state["trace"].append(self.name)
        return state
