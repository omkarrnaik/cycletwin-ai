class Agent:
    """Small interface used by all CycleTwin agents."""

    name = "BaseAgent"

    def run(self, state):
        raise NotImplementedError
