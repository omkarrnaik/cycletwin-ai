"""
Quick command-line smoke test for the CycleTwin AI pipeline.
Carried over from Phase 3, adapted to the Phase 4-8 orchestrator
(SHAP explainability instead of the old ExplainabilityAgent).

Run:
    python run_demo.py
"""

from orchestrator import CycleTwinOrchestrator

# Sample patient record. Swap in real values or wire this up to a form.
user_input = {
    'Age (yrs)': 27.0, 'Weight (Kg)': 52.0, 'Height(Cm)': 150.0, 'BMI': 23.1,
    'Blood Group': 11.0, 'Pulse rate(bpm)': 82.0, 'RR (breaths/min)': 20.0,
    'Hb(g/dl)': 11.1, 'Cycle(R/I)': 0.0, 'Cycle length(days)': 5.0,
    'Marraige Status (Yrs)': 5.0, 'Pregnant(Y/N)': 0.0, 'No. of aborptions': 0.0,
    'I   beta-HCG(mIU/mL)': 1.99, 'II    beta-HCG(mIU/mL)': 1.99,
    'FSH(mIU/mL)': 4.97, 'LH(mIU/mL)': 2.23, 'FSH/LH': 2.228699552,
    'Hip(inch)': 36.0, 'Waist(inch)': 32.0, 'Waist:Hip Ratio': 0.8888888889,
    'TSH (mIU/L)': 1.94, 'AMH(ng/mL)': 16.9, 'PRL(ng/mL)': 9.12,
    'Vit D3 (ng/mL)': 23.6, 'PRG(ng/mL)': 0.25, 'RBS(mg/dl)': 93.0,
    'Weight gain(Y/N)': 0.0, 'hair growth(Y/N)': 0.0, 'Skin darkening (Y/N)': 0.0,
    'Hair loss(Y/N)': 0.0, 'Pimples(Y/N)': 0.0, 'Fast food (Y/N)': 1.0,
    'Reg.Exercise(Y/N)': 1.0, 'BP _Systolic (mmHg)': 120.0, 'BP _Diastolic (mmHg)': 80.0,
    'Follicle No. (L)': 1.0, 'Follicle No. (R)': 1.0, 'Avg. F size (L) (mm)': 16.0,
    'Avg. F size (R) (mm)': 17.0, 'Endometrium (mm)': 8.55
}

orchestrator = CycleTwinOrchestrator()
result = orchestrator.run(user_input)

print("\nAGENT TRACE:")
print(" -> ".join(result["trace"]))

print("\nPREDICTION:")
print(result["prediction"])

print("\nSYMPTOM SIGNALS:")
print(result["symptoms"]["findings"])

print("\nTOP SHAP DRIVERS (increasing risk):")
for row in result["shap_top_positive"][:5]:
    print(f"  {row['feature']}: {row['shap_value']:.3f}")

print("\nRECOMMENDATIONS:")
for item in result["recommendations"]:
    print("-", item)

report_path = orchestrator.generate_report(result, "reports/cycletwin_report.pdf")
print("\nREPORT SAVED TO:")
print(report_path)
