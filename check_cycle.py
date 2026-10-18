import json, pickle
from orchestrator import CycleTwinOrchestrator

med = json.load(open("models/feature_medians.json"))
features = pickle.load(open("models/feature_schema.pkl", "rb"))
orch = CycleTwinOrchestrator()

for cyc in (0, 1):
    x = {f: med[f] for f in features}
    x["Cycle(R/I)"] = cyc
    r = orch.run(x)
    row = next(d for d in r["shap_rows"] if d["feature"] == "Cycle(R/I)")
    print(f"Cycle={cyc} | probability={r['prediction']['probability']:.1%} | "
          f"symptoms={r['symptoms']['findings']} | "
          f"SHAP row value={row['value']} | shap={row['shap_value']:+.3f}")