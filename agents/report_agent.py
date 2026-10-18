from .base import Agent
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

class ReportAgent(Agent):
    name = "Report Agent"

    def run(self, state):
        path = Path(state.get("report_path", "reports/cycletwin_report.pdf"))
        path.parent.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(path), pagesize=A4)

        story = [
            Paragraph("CycleTwin AI — Risk Awareness Report", styles["Title"]),
            Spacer(1, 12),
            Paragraph(
                f"Assessment: {state['prediction']['risk_band']}",
                styles["Heading2"]
            ),
            Paragraph(
                f"Model score: {state['prediction']['probability']:.1%}",
                styles["BodyText"]
            ),
            Spacer(1, 12),
            Paragraph("Patient-specific input signals", styles["Heading2"])
        ]

        signals = state.get("symptoms", {}).get("findings", [])
        if signals:
            for item in signals:
                story.append(Paragraph("• " + item, styles["BodyText"]))
        else:
            story.append(Paragraph("No flagged symptom signals.", styles["BodyText"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Top SHAP drivers (increasing risk)", styles["Heading2"]))
        for row in state.get("shap_top_positive", [])[:6]:
            story.append(Paragraph(
                f"• {row['feature']} (value: {row['value']}, SHAP: {row['shap_value']:.3f})",
                styles["BodyText"]
            ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Top SHAP drivers (decreasing risk)", styles["Heading2"]))
        for row in state.get("shap_top_negative", [])[:6]:
            story.append(Paragraph(
                f"• {row['feature']} (value: {row['value']}, SHAP: {row['shap_value']:.3f})",
                styles["BodyText"]
            ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("General wellness suggestions", styles["Heading2"]))
        for item in state["recommendations"]:
            story.append(Paragraph("• " + item, styles["BodyText"]))

        story.append(Spacer(1, 16))
        story.append(Paragraph(
            "SHAP values above describe each feature's contribution to this individual "
            "model prediction relative to the model's baseline output. They reflect "
            "prediction influence, not causation.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Educational prototype only. This output is not a medical diagnosis or treatment recommendation.",
            styles["BodyText"]
        ))
        doc.build(story)

        state["report"] = {"path": str(path)}
        state["trace"].append(self.name)
        return state

    def generate(self, state, path):
        """Alias used by CycleTwinOrchestrator.generate_report()."""
        state["report_path"] = path
        state = self.run(state)
        return state["report"]["path"]
