import pandas as pd

def simulate(model, base_input, changes):
    current = dict(base_input)
    current.update(changes)

    X = pd.DataFrame([current], columns=list(base_input.keys()))
    probability = float(model.predict_proba(X)[0, 1])
    return probability
