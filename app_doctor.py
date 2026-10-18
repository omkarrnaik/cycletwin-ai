
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from doctor_database import (
    init_db, ensure_demo_doctor, authenticate, add_patient,
    get_patients, get_patient, save_assessment, get_assessments,
    get_latest_assessment, assessment_dataframe,
)
from orchestrator import CycleTwinOrchestrator
from what_if import simulate


st.set_page_config(
    page_title="CycleTwin AI — Doctor Portal",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()
ensure_demo_doctor()

with open("models/rf_pcos_model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("models/feature_schema.pkl", "rb") as f:
    FEATURES = pickle.load(f)
with open("models/feature_medians.json", "r") as f:
    MEDIANS = json.load(f)

DATA_PATH = Path("data/run_final/pcos_clean.csv")
DEMO_DF = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()

st.markdown("""
<style>
:root {
    --ct-bg:#0B0D14;
    --ct-panel:#151824;
    --ct-panel-2:#1B1E2B;
    --ct-border:#2A2E3D;
    --ct-text:#F7F7FB;
    --ct-muted:#A7ACBC;
    --ct-purple:#8B5CF6;
    --ct-purple-2:#A78BFA;
    --ct-pink:#F472B6;
    --ct-green:#39D98A;
}
#MainMenu, footer {visibility:hidden;}
.block-container {max-width:none !important;width:100% !important;padding:1.2rem 2.2rem 3.5rem !important;}
[data-testid="stAppViewContainer"] {background:var(--ct-bg);}
[data-testid="stHeader"] {background:rgba(11,13,20,.86);}
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#11131D 0%,#171521 100%);
    border-right:1px solid var(--ct-border);
}
section[data-testid="stSidebar"] * {color:#ECECF4;}
section[data-testid="stSidebar"] .stCaption {color:#A7ACBC !important;}
section[data-testid="stSidebar"] hr {border-color:#2A2E3D;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {color:#E8E8F0 !important;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {color:#C4B5FD !important;}
section[data-testid="stSidebar"] button {
    background:#222534 !important;
    color:#F7F7FB !important;
    border:1px solid #34384A !important;
}
section[data-testid="stSidebar"] button:hover {
    border-color:#8B5CF6 !important;
    color:#FFFFFF !important;
}
section[data-testid="stSidebar"] .stAlert {
    background:#241820 !important;
    border:1px solid #5B3145 !important;
    color:#FFD6E5 !important;
}
h1,h2,h3,h4,h5,h6,p,label {color:var(--ct-text);}
[data-testid="stMarkdownContainer"] p {color:var(--ct-text);}
[data-testid="stCaptionContainer"] p {color:var(--ct-muted) !important;}
.topbar {display:flex;justify-content:space-between;align-items:center;padding:10px 4px 18px;}
.brand-mini {font-size:18px;font-weight:850;color:#A78BFA;}
.live-pill {display:inline-flex;align-items:center;gap:7px;padding:7px 12px;border-radius:999px;background:#241B3B;color:#C4B5FD;font-size:12px;font-weight:750;border:1px solid #45316E;}
.live-dot {width:8px;height:8px;border-radius:50%;background:#39D98A;box-shadow:0 0 10px #39D98A;}
.dashboard-hero {
    position:relative;overflow:hidden;
    background:
      radial-gradient(circle at 88% 15%,rgba(244,114,182,.18),transparent 25%),
      radial-gradient(circle at 14% 0%,rgba(139,92,246,.24),transparent 31%),
      linear-gradient(135deg,#17132A 0%,#151824 58%,#211722 100%);
    border:1px solid #343044;border-radius:28px;padding:30px;
    box-shadow:0 20px 70px rgba(0,0,0,.30);
}
.dashboard-hero h1 {font-size:34px;margin:0 0 6px;color:#FFFFFF;}
.dashboard-hero p {color:#B7BAC8;margin:0;font-size:15px;}
.hero-chip-row {margin-top:17px;}
.kpi {
    background:linear-gradient(145deg,#171A27,#131620);
    border:1px solid #2D3140;border-radius:20px;padding:20px;min-height:135px;
    box-shadow:0 12px 32px rgba(0,0,0,.20);
}
.kpi-icon {font-size:22px;margin-bottom:8px;}
.kpi-label {font-size:12px;color:#AEB3C3;font-weight:750;}
.kpi-value {font-size:32px;line-height:1.1;font-weight:900;color:#A78BFA;margin-top:5px;}
.kpi-sub {font-size:11px;color:#858B9D;margin-top:8px;}
.section-head {display:flex;justify-content:space-between;align-items:end;margin:26px 0 11px;}
.section-head h2 {font-size:21px;margin:0;color:#F7F7FB;}
.section-head span {font-size:12px;color:#858B9D;}
.action-card {
    background:linear-gradient(145deg,#171A27,#141720);
    border:1px solid #2D3140;border-radius:18px;padding:18px;min-height:128px;
    box-shadow:0 9px 26px rgba(0,0,0,.18);
}
.action-card:hover {border-color:#57418A;transform:translateY(-1px);}
.action-icon {font-size:26px;}
.action-title {font-weight:850;color:#F7F7FB;margin-top:9px;}
.action-text {font-size:12px;color:#A7ACBC;margin-top:5px;}
.status-card {
    background:linear-gradient(145deg,#151824,#11141D);
    border:1px solid #2D3140;border-radius:18px;padding:17px;
}
.status-row {display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #282C39;}
.status-row:last-child {border-bottom:0;}
.status-green {width:9px;height:9px;border-radius:50%;background:#39D98A;box-shadow:0 0 8px rgba(57,217,138,.55);}
.timeline-card {
    background:linear-gradient(145deg,#171A27,#141720);
    border:1px solid #2D3140;border-radius:18px;padding:18px;
    box-shadow:0 9px 26px rgba(0,0,0,.18);
}
.timeline-card b,.status-card b {color:#F5F5FA;}
.hero {
    background:
      radial-gradient(circle at 12% 18%,rgba(139,92,246,.20),transparent 30%),
      radial-gradient(circle at 86% 20%,rgba(244,114,182,.16),transparent 28%),
      linear-gradient(135deg,#17132A 0%,#151824 58%,#211722 100%);
    border:1px solid #343044;border-radius:26px;padding:30px;margin-bottom:20px;
}
.login-hero {min-height:300px;display:flex;align-items:center;}
.login-hero h1 {color:#FFFFFF !important;}
.login-hero p {color:#B7BAC8 !important;}
.login-card {
    background:linear-gradient(145deg,#171A27,#11141D);
    border:1px solid #34384A;border-radius:26px;padding:30px;
    box-shadow:0 20px 65px rgba(0,0,0,.35);
}
.login-card h2,.login-card h3 {color:#FFFFFF !important;}
.card {
    background:#171A27;border:1px solid #2D3140;border-radius:18px;
    padding:18px;min-height:115px;box-shadow:0 8px 28px rgba(0,0,0,.20);
}
.card-title {font-size:13px;font-weight:700;color:#B4B8C7;}
.big {font-size:31px;font-weight:850;color:#A78BFA;margin-top:5px;}
.small {font-size:12px;color:#8F95A7;}
.section {font-size:21px;font-weight:800;color:#F7F7FB;margin:24px 0 11px;}
.badge {
    display:inline-block;padding:6px 11px;border-radius:999px;
    background:#241B3B;color:#C4B5FD;font-size:12px;
    font-weight:750;margin:3px;border:1px solid #45316E;
}
.warning {
    background:#261921;border:1px solid #613247;color:#FFD5E4;
    padding:12px 14px;border-radius:14px;font-size:12px;
}
.agent {
    border:1px solid #2D3140;border-radius:14px;padding:12px;
    background:#171A27;margin-bottom:8px;
}
.reco {
    border:1px solid #2D3140;border-radius:14px;padding:14px;
    background:#171A27;margin-bottom:9px;
}
.login-feature {
    border:1px solid #2D3140;border-radius:15px;padding:14px;
    background:#151824;box-shadow:0 8px 24px rgba(0,0,0,.15);
}
.login-feature b {color:#F7F7FB;}
.login-feature .small {color:#9298AA !important;}
div[data-testid="stMetric"] {
    background:#171A27;border:1px solid #2D3140;border-radius:16px;padding:14px;
}
div[data-testid="stMetricLabel"] {color:#A7ACBC !important;}
div[data-testid="stMetricValue"] {color:#A78BFA !important;}
div[data-testid="stDataFrame"] {
    border:1px solid #2D3140;border-radius:14px;overflow:hidden;
}
.stTextInput input,.stNumberInput input,.stDateInput input,.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background:#171A27 !important;color:#F7F7FB !important;
    border-color:#34384A !important;
}
.stTextInput input::placeholder,.stTextArea textarea::placeholder {color:#73798A !important;}
[data-baseweb="popover"] {background:#171A27 !important;}
[data-baseweb="menu"] {background:#171A27 !important;}
[data-baseweb="option"] {color:#F7F7FB !important;}
[data-baseweb="option"]:hover {background:#29233C !important;}
.stButton button {
    border-radius:12px !important;
    font-weight:750 !important;
}
.stButton button[kind="primary"] {
    background:linear-gradient(135deg,#7C3AED,#A855F7) !important;
    border:0 !important;color:white !important;
    box-shadow:0 8px 22px rgba(124,58,237,.28);
}
.stButton button[kind="secondary"] {
    background:#1B1E2B !important;color:#F7F7FB !important;border:1px solid #34384A !important;
}
div[data-testid="stAlert"] {background:#171A27;border-color:#34384A;}
[data-testid="stExpander"] {background:#151824;border:1px solid #2D3140;border-radius:14px;}
[data-testid="stExpander"] summary {color:#F7F7FB !important;}
.stTabs [data-baseweb="tab-list"] {gap:8px;background:#11141D;}
.stTabs [data-baseweb="tab"] {color:#9EA4B5 !important;}
.stTabs [aria-selected="true"] {color:#C4B5FD !important;border-bottom-color:#8B5CF6 !important;}

/* Interactive navigation controls */
.agent-nav-card {margin:8px 0;}
.agent-nav-card + .agent-nav-card {margin-top:9px;}
div[data-testid="stButton"] > button {
    border-radius:14px !important;
    border:1px solid #30364A !important;
    background:linear-gradient(135deg,#171B29,#121521) !important;
    color:#F7F7FB !important;
    min-height:48px !important;
    font-weight:750 !important;
    text-align:left !important;
    padding:0.65rem 1rem !important;
    transition:all .18s ease !important;
    box-shadow:0 8px 20px rgba(0,0,0,.16) !important;
}
div[data-testid="stButton"] > button:hover {
    border-color:#8B5CF6 !important;
    background:linear-gradient(135deg,#211B39,#171927) !important;
    transform:translateY(-1px) !important;
    color:#FFFFFF !important;
}
.agent-focus {
    border:1px solid #5B3AA5;
    background:linear-gradient(135deg,#17142A,#21172A);
    border-radius:18px;
    padding:18px 20px;
    margin:12px 0 22px;
}
.agent-focus .focus-title {font-size:18px;font-weight:850;color:#FFFFFF;}
.agent-focus .focus-sub {font-size:12px;color:#AEB3C3;margin-top:4px;}

</style>
""", unsafe_allow_html=True)


for key, default in {
    "doctor": None, "patient_id": None, "result": None,
    "demo_loaded": False, "nav_page": "Dashboard", "agent_focus": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def go_to(page_name, focus=None):
    # Keep sidebar navigation synchronized with programmatic buttons.
    st.session_state.nav_page = page_name
    st.session_state.agent_focus = focus
    st.rerun()


def metric(title, value, subtitle=""):
    st.markdown(
        f"""<div class="card">
        <div class="card-title">{title}</div>
        <div class="big">{value}</div>
        <div class="small">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True
    )


# ---------- SHAP visualization ----------
SHAP_RED =  "#DC2626"    # negative SHAP -> lower predicted PCOS risk
SHAP_GREEN =  "#22A06B"    # positive SHAP -> higher predicted PCOS risk

# Swapped color gradients
SHAP_RED_LIGHT = (200, 245, 222)    # weak positive contribution
SHAP_RED_DARK = (10, 84, 55)        # strong positive contribution

SHAP_GREEN_LIGHT = (253, 214, 214)  # weak negative contribution
SHAP_GREEN_DARK = (127, 15, 15)     # strong negative contribution


def _lerp_color(light_rgb, dark_rgb, t):
    t = max(0.0, min(1.0, t))
    r = round(light_rgb[0] + (dark_rgb[0] - light_rgb[0]) * t)
    g = round(light_rgb[1] + (dark_rgb[1] - light_rgb[1]) * t)
    b = round(light_rgb[2] + (dark_rgb[2] - light_rgb[2]) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def shap_bar_figure(rows, top_n=6, height=360):
    """Horizontal bar chart of SHAP contributions.

    Positive SHAP contributions are shown in red and indicate
    higher predicted PCOS risk.

    Negative SHAP contributions are shown in green and indicate
    lower predicted PCOS risk.

    Color intensity increases with the magnitude of the SHAP value.
    """
    d = pd.DataFrame(rows).copy()

    if d.empty:
        return None

    if "abs_shap" not in d.columns:
        d["abs_shap"] = d["shap_value"].abs()

    d = d.sort_values("abs_shap", ascending=False).head(top_n)

    # Ascending order so the most influential feature
    # renders at the top of the horizontal bar chart.
    d = d.sort_values("abs_shap", ascending=True)

    max_abs = d["abs_shap"].max() or 1.0

    bar_colors = []
    line_colors = []

    for v, a in zip(d["shap_value"], d["abs_shap"]):
        strength = a / max_abs

        # POSITIVE SHAP -> RED
        if v > 0:
            bar_colors.append(
                _lerp_color(
                    SHAP_GREEN_LIGHT,
                    SHAP_GREEN_DARK,
                    strength
                )
            )
            line_colors.append("#0A5437")

        # NEGATIVE SHAP -> GREEN
        else:
            bar_colors.append(
                _lerp_color(
                    SHAP_RED_LIGHT,
                    SHAP_RED_DARK,
                    strength
                )
            )
            line_colors.append("#7F0F0F")

    fig = go.Figure(go.Bar(
        x=d["shap_value"],
        y=d["feature"],
        orientation="h",

        marker=dict(
            color=bar_colors,
            line=dict(
                color=line_colors,
                width=1.2
            )
        ),

        text=[f"{v:+.3f}" for v in d["shap_value"]],
        textposition="outside",
        textfont=dict(
            size=12,
            color="#202A43"
        ),

        customdata=d["value"] if "value" in d.columns else None,

        hovertemplate=(
            "<b>%{y}</b><br>"
            "SHAP: %{x:+.4f}<br>"
            "Value: %{customdata}"
            "<extra></extra>"
        )
    ))

    fig.add_vline(
        x=0,
        line_width=1.5,
        line_color="#6B7280"
    )

    fig.update_layout(
        height=height,
        margin=dict(
            l=0,
            r=40,
            t=15,
            b=10
        ),
        showlegend=False,
        xaxis_title="SHAP contribution",
        plot_bgcolor="#FBFAFF",
        paper_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            showgrid=True,
            gridcolor="#E8E5F4",
            zeroline=False
        ),

        yaxis=dict(
            showgrid=False
        ),

        font=dict(size=12)
    )

    return fig

def shap_legend():
    st.markdown(
        f"""<div style="font-size:13px;line-height:1.7;color:#303A52;">
        <div style="margin-bottom:6px;">
        <span style="display:inline-block;width:70px;height:10px;border-radius:5px;
        background:linear-gradient(90deg, rgb{SHAP_RED_LIGHT}, rgb{SHAP_RED_DARK});
        vertical-align:middle;margin-right:8px;"></span>
        🟢 <b>Green</b> (deeper = stronger) = pushes prediction toward <b>lower</b> PCOS risk
        </div>
        <div>
        <span style="display:inline-block;width:70px;height:10px;border-radius:5px;
        background:linear-gradient(90deg, rgb{SHAP_GREEN_LIGHT}, rgb{SHAP_GREEN_DARK});
        vertical-align:middle;margin-right:8px;"></span>
        🔴 <b>Red</b> (deeper = stronger) = pushes prediction toward <b>higher</b> PCOS risk
        </div>
        </div>""",
        unsafe_allow_html=True
    )
    st.caption(
        "SHAP values explain how each feature influenced this individual prediction "
        "relative to the model's baseline output. They describe model behavior and "
        "do not prove causation."
    )


def is_binary(feature):
    return "(Y/N)" in feature or feature in {
        "Cycle(R/I)", "Reg.Exercise(Y/N)", "Fast food (Y/N)"
    }


def pretty(feature):
    return feature.replace("(Y/N)","").replace("(R/I)","").strip()


def binary_label(feature, value):
    value = int(value)
    if feature == "Cycle(R/I)":
        return "Irregular" if value else "Regular"
    return "Yes" if value else "No"


def number_value(v, default=0):
    try:
        if pd.isna(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def run_ai(values):
    clean = {}
    for f in FEATURES:
        v = values.get(f, MEDIANS.get(f, 0))
        clean[f] = int(number_value(v, 0)) if is_binary(f) else number_value(
            v, MEDIANS.get(f, 0)
        )
    for f in FEATURES:
        if is_binary(f) and clean[f] not in (0, 1):
            raise ValueError(f"{f} must be 0 or 1, got {clean[f]}")
    return CycleTwinOrchestrator().run(clean)


def selected_patient():
    pid = st.session_state.patient_id
    if not pid or not st.session_state.doctor:
        return None
    return get_patient(pid, st.session_state.doctor["doctor_id"])


def latest_as_result(patient):
    latest = get_latest_assessment(patient["patient_id"])
    if not latest:
        return None
    return {
        "input": latest["input"],
        "prediction": {
            "probability": latest["probability"],
            "risk_band": latest["risk_band"],
            "label": int(latest["probability"] >= .5)
        },
        "shap_rows": latest["shap_rows"],
        "recommendations": latest["recommendations"],
        "trace": latest["trace"],
    }


def render_agent_focus(result, focus):
    if not focus:
        return

    meta = {
        "Lifestyle Agent": ("🌿", "Lifestyle Agent", "Health Assessment"),
        "Symptom Agent": ("🩺", "Symptom Agent", "Health Assessment"),
        "Prediction Agent": ("🎯", "Prediction Agent", "AI Analysis"),
        "SHAP Explainability Agent": ("🧠", "SHAP Explainability Agent", "AI Analysis"),
        "Recommendation Agent": ("💡", "Recommendation Agent", "Recommendations"),
        "Report Engine": ("📄", "Report Engine", "Reports"),
    }.get(focus)

    if not meta:
        return

    icon, name, section = meta
    st.markdown(
        f'''<div class="agent-focus">
        <div class="focus-title">{icon} {name}</div>
        <div class="focus-sub">Opened from the pipeline • destination: {section}</div>
        </div>''',
        unsafe_allow_html=True)

    if not result:
        st.info("Run an assessment first to see this agent's patient-specific output.")
        return

    if focus == "Lifestyle Agent":
        findings = result.get("lifestyle", {}).get("findings", [])
        st.subheader("Lifestyle output")
        if findings:
            for x in findings:
                st.markdown(f"- {x}")
        else:
            st.success("No lifestyle flags detected from the selected inputs.")

    elif focus == "Symptom Agent":
        symptoms = result.get("symptoms", {})
        st.subheader("Symptom output")
        st.write(f"Mapped symptom indicators detected: **{symptoms.get('count', 0)}**")
        for x in symptoms.get("findings", []):
            st.markdown(f"- {x}")
        if not symptoms.get("findings"):
            st.success("No mapped symptom indicators detected.")

    elif focus == "Prediction Agent":
        pred = result.get("prediction", {})
        st.subheader("Prediction output")
        a, b = st.columns(2)
        with a:
            metric("Model Probability", f"{float(pred.get('probability', 0)):.1%}", "Random Forest")
        with b:
            metric("Risk Pattern", pred.get("risk_band", "—"))

    elif focus == "SHAP Explainability Agent":
        st.subheader("SHAP output")
        rows = result.get("shap_rows", [])
        if rows:
            fig = shap_bar_figure(rows, top_n=8)
            st.plotly_chart(fig, use_container_width=True)
            shap_legend()
            sdf = pd.DataFrame(rows).copy()
            sdf["abs_shap"] = sdf["shap_value"].abs()
            sdf = sdf.sort_values("abs_shap", ascending=False).head(8)
            st.dataframe(
                sdf[["feature", "value", "shap_value"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No SHAP output is available.")

    elif focus == "Recommendation Agent":
        st.subheader("Recommendation output")
        for x in result.get("recommendations", []):
            st.markdown(f"- {x}")

    elif focus == "Report Engine":
        st.subheader("Report output")
        st.info("Open Reports to generate and download the longitudinal PDF.")

    # Prevent the focus panel from appearing forever.
    st.session_state.agent_focus = None


def generate_pdf(patient, rows):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    Path("reports").mkdir(exist_ok=True)
    path = Path("reports") / (
        f"{patient['patient_code']}_CycleTwin_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "CTTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=22, leading=26, spaceAfter=8
    )
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    latest = rows[-1]
    baseline = rows[0]
    story = [
        Paragraph("CycleTwin AI", title),
        Paragraph("Doctor-Assisted PCOS Risk & Wellness Assessment",
                  styles["Heading2"]),
        Spacer(1,8),
        Paragraph(
            f"<b>Patient:</b> {patient['name']} "
            f"({patient['patient_code']})", styles["Normal"]
        ),
        Paragraph(f"<b>Source:</b> {patient['source']}", styles["Normal"]),
        Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y %H:%M')}",
            styles["Normal"]
        ),
        Spacer(1,12),
        Paragraph("Current Assessment", styles["Heading2"]),
        Paragraph(
            f"Latest model probability: <b>{latest['probability']:.1%}</b>",
            styles["Normal"]
        ),
        Paragraph(
            f"Risk-awareness pattern: <b>{latest['risk_band']}</b>",
            styles["Normal"]
        ),
        Spacer(1,10),
    ]

    if len(rows) > 1:
        delta = latest["probability"] - baseline["probability"]
        story += [
            Paragraph("Longitudinal Follow-up", styles["Heading2"]),
            Paragraph(
                f"Baseline {baseline['probability']:.1%} → "
                f"Latest {latest['probability']:.1%} "
                f"({delta:+.1%} percentage points).",
                styles["Normal"]
            ),
            Spacer(1,8)
        ]
        data = [["Visit","Date","Type","Probability","Risk"]]
        for i, r in enumerate(rows, 1):
            data.append([
                i, str(r["created_at"])[:16], r["assessment_type"],
                f"{r['probability']:.1%}", r["risk_band"]
            ])
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#6C3CEB")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#DDDDDD")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.white,colors.HexColor("#F8F6FF")])
        ]))
        story += [t, Spacer(1,12)]

        changes = []
        keys = sorted(set(baseline["input"]) | set(latest["input"]))
        for f in keys:
            old, new = baseline["input"].get(f), latest["input"].get(f)
            if str(old) != str(new):
                if is_binary(f):
                    old, new = binary_label(f, old), binary_label(f, new)
                changes.append([pretty(f), str(old), str(new)])
        if changes:
            story.append(Paragraph("What Changed?", styles["Heading2"]))
            t2 = Table([["Feature","Baseline","Latest"]] + changes,
                       repeatRows=1)
            t2.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#8B5CF6")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#DDDDDD"))
            ]))
            story += [t2, Spacer(1,12)]

    story.append(Paragraph("Top SHAP Drivers", styles["Heading2"]))
    shap_rows = latest.get("shap_rows", [])
    if shap_rows:
        data = [["Feature","Value","SHAP"]]
        for r in shap_rows[:10]:
            data.append([
                str(r.get("feature","")),
                str(r.get("value","")),
                f"{float(r.get('shap_value',0)):+.4f}"
            ])
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#6C3CEB")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#DDDDDD"))
        ]))
        story += [t, Spacer(1,12)]

    story.append(Paragraph("AI Recommendations", styles["Heading2"]))
    for i, rec in enumerate(latest.get("recommendations", []), 1):
        story += [
            Paragraph(f"<b>{i}.</b> {str(rec)}", styles["Normal"]),
            Spacer(1,4)
        ]

    story += [
        Spacer(1,12),
        Paragraph("Important", styles["Heading2"]),
        Paragraph(
            "Educational prototype only. SHAP explains model behavior and "
            "does not establish causation. Model output is risk awareness "
            "support, not a medical diagnosis or treatment recommendation.",
            styles["Normal"]
        )
    ]
    doc.build(story)
    return path


# ----------------------------- LOGIN -----------------------------

if st.session_state.doctor is None:
    left, right = st.columns([1.12, .88], gap="large")

    with left:
        st.markdown("""
        <div class="hero login-hero">
          <div>
            <div style="font-size:52px">🌸</div>
            <h1 style="font-size:46px;margin:0;color:#172033">
              CycleTwin AI
            </h1>
            <p style="font-size:20px;color:#5F6472">
              Doctor-assisted PCOS risk & wellness intelligence
            </p>
            <span class="badge">Random Forest</span>
            <span class="badge">SHAP / XAI</span>
            <span class="badge">Agent Pipeline</span>
            <span class="badge">Follow-up Tracking</span>
            <p style="margin-top:22px;color:#6B7280;font-size:15px">
              A single portal for patient assessment, explainable ML,
              personalized wellness suggestions, longitudinal comparison,
              what-if simulation and PDF reporting.
            </p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        a,b,c = st.columns(3)
        with a:
            st.markdown(
                '<div class="login-feature"><b>🧠 Explain</b><br>'
                '<span class="small">Patient-specific SHAP drivers</span></div>',
                unsafe_allow_html=True)
        with b:
            st.markdown(
                '<div class="login-feature"><b>📈 Track</b><br>'
                '<span class="small">Baseline to follow-up trends</span></div>',
                unsafe_allow_html=True)
        with c:
            st.markdown(
                '<div class="login-feature"><b>📄 Report</b><br>'
                '<span class="small">Downloadable PDF summary</span></div>',
                unsafe_allow_html=True)

    with right:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("## 👩‍⚕️ Doctor Login")
        st.caption("Sign in to the CycleTwin clinical-demo portal.")

        email = st.text_input("Doctor email",
                              value="doctor@cycletwin.demo")
        password = st.text_input("Password", value="1234",
                                 type="password")

        if st.button("Enter Doctor Portal  →",
                     type="primary", use_container_width=True):
            doctor = authenticate(email, password)
            if doctor:
                st.session_state.doctor = doctor
                st.rerun()
            st.error("Invalid doctor credentials.")

        st.divider()
        st.caption("Demo credentials")
        st.code("doctor@cycletwin.demo   |   1234")
        st.markdown(
            '<div class="warning">⚠️ Educational prototype only — '
            'not a medical diagnosis.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# ----------------------------- PORTAL -----------------------------

doctor = st.session_state.doctor
patients = get_patients(doctor["doctor_id"])

with st.sidebar:
    st.markdown("## 🌸 CycleTwin AI")
    st.caption(f"Doctor Portal • {doctor['name']}")
    st.divider()
    nav_options = [
        "Dashboard","Patients","Kaggle Demo","New Patient",
        "Health Assessment","AI Analysis","Trends & History",
        "Recommendations","What-if Simulator","Reports"
    ]
    if st.session_state.nav_page not in nav_options:
        st.session_state.nav_page = "Dashboard"
    # The selected page is stored independently from the radio widget. This is
    # important because Streamlit forbids changing a widget's session_state key
    # after that widget has been instantiated. Programmatic navigation changes
    # nav_page, reruns, and the radio derives its initial selection from nav_page.
    page = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state.nav_page),
        label_visibility="collapsed"
    )
    st.session_state.nav_page = page
    st.divider()

    patient = selected_patient()
    if patient:
        st.success(f"Patient: {patient['name']}")
        st.caption(patient["patient_code"])

    if st.button("Logout", use_container_width=True):
        st.session_state.doctor = None
        st.session_state.patient_id = None
        st.session_state.result = None
        st.session_state.nav_page = "Dashboard"
        st.session_state.agent_focus = None
        st.rerun()

    st.markdown(
        '<div class="warning">⚠️ Risk-awareness prototype. '
        'Not a medical diagnosis.</div>',
        unsafe_allow_html=True
    )


# ----------------------------- DASHBOARD -----------------------------


if page == "Dashboard":
    total = sum(len(get_assessments(p["patient_id"])) for p in patients)
    followups = sum(
        max(0, len(get_assessments(p["patient_id"])) - 1)
        for p in patients
    )
    assessed = sum(
        1 for p in patients if get_latest_assessment(p["patient_id"])
    )

    st.markdown("""
    <div class="topbar">
      <div class="brand-mini">🌸 CycleTwin AI</div>
      <div class="live-pill"><span class="live-dot"></span> Doctor Portal • Demo Mode</div>
    </div>
    <div class="dashboard-hero">
      <h1>Good to see you, Doctor 👋</h1>
      <p>One workspace for PCOS risk awareness, explainable AI and longitudinal patient follow-up.</p>
      <div class="hero-chip-row">
        <span class="badge">🧠 Explainable AI</span>
        <span class="badge">🤖 Multi-Agent</span>
        <span class="badge">📈 Follow-up Intelligence</span>
        <span class="badge">📄 Smart Reports</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-head"><h2>Practice Snapshot</h2>'
        '<span>Live from your local database</span></div>',
        unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    kpis = [
        ("👥","TOTAL PATIENTS",len(patients),"Registered in this portal"),
        ("🩺","ASSESSMENTS",total,"Baseline + follow-up visits"),
        ("🔄","FOLLOW-UPS",followups,"Longitudinal observations"),
        ("🧠","ASSESSED",assessed,"Patients with an AI result")
    ]
    for col,(icon,label,value,sub) in zip([a,b,c,d],kpis):
        with col:
            st.markdown(
                f'''<div class="kpi"><div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div></div>''',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="section-head"><h2>Quick Actions</h2>'
        '<span>Start your next workflow</span></div>',
        unsafe_allow_html=True)
    q1,q2,q3,q4 = st.columns(4)
    quick_actions = [
        ("🧪 Kaggle Demo","Load a cleaned dataset patient.","Kaggle Demo"),
        ("➕ New Patient","Create a persistent patient profile.","New Patient"),
        ("🩺 Health Assessment","Capture and save a patient visit.","Health Assessment"),
        ("📄 Reports","Generate a longitudinal PDF report.","Reports")
    ]
    for col,(label,desc,target) in zip([q1,q2,q3,q4],quick_actions):
        with col:
            st.markdown(
                f'''<div class="action-card">
                <div class="action-title">{label}</div>
                <div class="action-text">{desc}</div>
                </div>''',
                unsafe_allow_html=True)
            if st.button(f"Open {target} →", key=f"quick_{target}", use_container_width=True):
                go_to(target)


    st.markdown(
        '<div class="section-head"><h2>AI Pipeline — interactive</h2>'
        '<span>Click an agent to open its working section</span></div>',
        unsafe_allow_html=True)

    agent_routes = [
        ("🌿", "Lifestyle Agent", "Lifestyle signals → Health Assessment", "Health Assessment"),
        ("🩺", "Symptom Agent", "Symptom findings → Health Assessment", "Health Assessment"),
        ("🎯", "Prediction Agent", "Random Forest output → AI Analysis", "AI Analysis"),
        ("🧠", "SHAP Explainability Agent", "Patient-specific drivers → AI Analysis", "AI Analysis"),
        ("💡", "Recommendation Agent", "Wellness recommendations → Recommendations", "Recommendations"),
        ("📄", "Report Engine", "Longitudinal report → Reports", "Reports"),
    ]

    left, right = st.columns(2, gap="large")
    for i, (icon, name, desc, target) in enumerate(agent_routes):
        col = left if i % 2 == 0 else right
        with col:
            st.markdown('<div class="agent-nav-card">', unsafe_allow_html=True)
            if st.button(f"{icon}  {name}   ✓", key=f"dashboard_agent_{i}", use_container_width=True):
                go_to(target, name)
            st.caption(desc)
            st.markdown('</div>', unsafe_allow_html=True)


    st.markdown(
        '<div class="section-head"><h2>Patient Overview</h2>'
        '<span>Most recently added patients</span></div>',
        unsafe_allow_html=True)
    if not patients:
        st.info("No patients yet. Start with Kaggle Demo or New Patient.")
    else:
        rows=[]
        for p in patients[:10]:
            h=get_assessments(p["patient_id"])
            latest=h[-1] if h else None
            rows.append({
                "Patient":p["name"], "Patient ID":p["patient_code"],
                "Source":p["source"], "Visits":len(h),
                "Latest Probability":
                    f"{latest['probability']:.1%}" if latest else "—",
                "Risk Pattern":
                    latest["risk_band"] if latest else "Awaiting assessment"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    st.markdown("""
    <div style="margin-top:28px;padding:18px 20px;border:1px solid #2D3140;
         border-radius:18px;background:linear-gradient(135deg,#11141D,#17132A);">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:18px;">
        <div>
          <div style="font-weight:850;color:#F7F7FB;font-size:15px;">CycleTwin clinical-demo workflow</div>
          <div style="color:#8F95A7;font-size:12px;margin-top:5px;">
            Assess → Predict → Explain → Follow-up → Compare → Report
          </div>
        </div>
        <div style="color:#39D98A;font-size:12px;font-weight:800;">● SYSTEM READY</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------- PATIENTS -----------------------------

elif page == "Patients":
    st.title("👥 Patient Registry")
    q=st.text_input("Search by patient name or ID")
    shown=[p for p in patients if not q or
           q.lower() in p["name"].lower() or q.lower() in p["patient_code"].lower()]

    if not shown:
        st.info("No matching patients.")
    for p in shown:
        history=get_assessments(p["patient_id"])
        latest=history[-1] if history else None
        a,b,c,d=st.columns([2.2,1.1,1.4,1])
        with a:
            st.markdown(f"### {p['name']}")
            st.caption(f"{p['patient_code']} • {p['source']}")
        with b: st.write(f"Visits **{len(history)}**")
        with c: st.write(
            f"Risk **{latest['probability']:.0%}**" if latest else "Not assessed"
        )
        with d:
            if st.button("Open", key=f"open_{p['patient_id']}"):
                st.session_state.patient_id=p["patient_id"]
                st.session_state.result=None
                st.session_state.nav_page = "Health Assessment"
                st.session_state.agent_focus = None
                st.rerun()
        st.divider()


# ----------------------------- KAGGLE DEMO -----------------------------

elif page == "Kaggle Demo":
    st.markdown("""
    <div class="hero">
      <h1>🧪 Kaggle Demo</h1>
      <p>
        Load a real row from the cleaned Kaggle dataset into a persistent
        demo patient, then run the exact existing CycleTwin pipeline.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if DEMO_DF.empty:
        st.error("data/run_final/pcos_clean.csv not found.")
    else:
        st.info(f"Dataset available: {len(DEMO_DF):,} rows.")
        if st.button("🎲 Load Random Kaggle Patient",
                     type="primary", use_container_width=True):
            idx=random.randrange(len(DEMO_DF))
            row=DEMO_DF.iloc[idx]
            pid=add_patient(
                doctor["doctor_id"], f"Demo Patient {idx+1}",
                number_value(row.get("Age (yrs)",25),25),
                number_value(row.get("Height(Cm)",160),160),
                number_value(row.get("Weight (Kg)",60),60),
                "Kaggle Demo"
            )
            st.session_state.patient_id=pid
            st.session_state.result=None
            st.session_state.demo_loaded=True
            st.session_state.nav_page = "Health Assessment"
            st.session_state.agent_focus = None
            st.success(f"Loaded Demo Patient {idx+1}.")
            st.rerun()

        if selected_patient():
            p=selected_patient()
            st.success(f"Selected: {p['name']} • {p['patient_code']}")
            st.caption("Next: Health Assessment → Run AI Assessment.")


# ----------------------------- NEW PATIENT -----------------------------

elif page == "New Patient":
    st.title("➕ New Patient")
    with st.form("new_patient"):
        name=st.text_input("Patient name")
        age=st.number_input("Age",1,100,25)
        height=st.number_input("Height (cm)",80.0,220.0,160.0)
        weight=st.number_input("Weight (kg)",20.0,200.0,60.0)
        submit=st.form_submit_button("Create Patient",
                                      type="primary",
                                      use_container_width=True)
    if submit:
        if not name.strip():
            st.error("Patient name is required.")
        else:
            pid=add_patient(doctor["doctor_id"],name,age,height,weight)
            st.session_state.patient_id=pid
            st.session_state.result=None
            st.session_state.nav_page = "Health Assessment"
            st.session_state.agent_focus = None
            st.success("Patient created.")
            st.rerun()


# ----------------------------- ASSESSMENT -----------------------------

elif page == "Health Assessment":
    p=selected_patient()
    if not p:
        st.info("Select a patient from Patients or Kaggle Demo first.")
    else:
        history=get_assessments(p["patient_id"])
        kind="FOLLOW_UP" if history else "BASELINE"
        st.markdown(
            f"""<div class="hero"><h1>🩺 {kind.title()} Assessment</h1>
            <p>{p['name']} • {p['patient_code']}</p></div>""",
            unsafe_allow_html=True)
        if st.session_state.get("agent_focus") in {"Lifestyle Agent", "Symptom Agent"}:
            render_agent_focus(st.session_state.result or latest_as_result(p),
                               st.session_state.agent_focus)
        if history:
            st.success("Previous assessment found. This visit is a follow-up.")
        previous=history[-1]["input"] if history else {}

        groups={
            "Personal & Vitals":[],
            "Menstrual":[],
            "Hormones":[],
            "Body & Symptoms":[],
            "Lifestyle & Ultrasound":[]
        }
        for f in FEATURES:
            low=f.lower()
            if any(x in low for x in ["cycle","pregnant","marriage","abortion"]):
                groups["Menstrual"].append(f)
            elif any(x in low for x in ["hcg","fsh","lh","tsh","amh","prl",
                                        "vit d","prg","rbs"]):
                groups["Hormones"].append(f)
            elif any(x in low for x in ["waist","hip","weight gain","hair",
                                        "skin","pimple"]):
                groups["Body & Symptoms"].append(f)
            elif any(x in low for x in ["fast food","exercise","follicle",
                                        "f size","endometrium"]):
                groups["Lifestyle & Ultrasound"].append(f)
            else:
                groups["Personal & Vitals"].append(f)

        inputs={}
        for g,fs in groups.items():
            with st.expander(g, expanded=(g=="Personal & Vitals")):
                cols=st.columns(2)
                for i,f in enumerate(fs):
                    default=previous.get(f, MEDIANS.get(f,0))
                    with cols[i%2]:
                        if is_binary(f):
                            binary_value = int(number_value(default, 0))
                            binary_value = 1 if binary_value == 1 else 0

                            inputs[f] = st.selectbox(
                                pretty(f), [0, 1],
                                index=binary_value,
                                format_func=lambda x, ff=f: binary_label(ff, x),
                                key=f"assess_{p['patient_id']}_{kind}_{f}"
                            )
                        else:
                            inputs[f]=st.number_input(
                                pretty(f),
                                value=number_value(default,MEDIANS.get(f,0)),
                                key=f"assess_{p['patient_id']}_{kind}_{f}"
                            )


        if st.button("🚀 Run AI Assessment & Save",
                     type="primary", use_container_width=True):
            with st.spinner("Running existing agent pipeline + SHAP..."):
                try:
                    result=run_ai(inputs)
                    save_assessment(p["patient_id"],result,kind)
                    st.session_state.result=result
                    st.session_state.nav_page = "AI Analysis"
                    st.session_state.agent_focus = None
                    st.success(f"{kind.title()} assessment saved.")
                    st.rerun()
                except Exception as e:
                    st.error("Assessment pipeline failed.")
                    st.exception(e)


# ----------------------------- AI ANALYSIS -----------------------------

elif page == "AI Analysis":
    p=selected_patient()
    if not p:
        st.info("Select a patient first.")
    else:
        result=st.session_state.result or latest_as_result(p)
        if not result:
            st.info("Run an assessment first.")
        else:
            prob=result["prediction"]["probability"]
            st.markdown(
                f"""<div class="hero"><h1>🧠 Explainable AI Analysis</h1>
                <p>Patient-specific model explanation for {p['name']}.</p>
                </div>""", unsafe_allow_html=True)
            a,b,c=st.columns(3)
            with a: metric("Probability",f"{prob:.1%}","Random Forest output")
            with b: metric("Risk Pattern",result["prediction"]["risk_band"])
            with c: metric("Visits",len(get_assessments(p["patient_id"])))
            shap_rows=result.get("shap_rows",[])
            if shap_rows:
                fig=shap_bar_figure(shap_rows, top_n=12)
                fig.update_layout(title="Patient-specific SHAP Contributions")
                st.plotly_chart(fig,use_container_width=True)
                shap_legend()
            st.markdown('<div class="section">Agent Trace — click an agent to inspect its work</div>',
                        unsafe_allow_html=True)

            # The trace is interactive: each agent expands to show the actual
            # information produced by that agent for THIS patient.
            trace_details = {
                "Lifestyle Agent": {
                    "icon": "🌿",
                    "title": "Lifestyle Agent",
                    "summary": "Analyzes lifestyle-related inputs from the current assessment.",
                },
                "Symptom Agent": {
                    "icon": "🩺",
                    "title": "Symptom Agent",
                    "summary": "Checks the patient's symptom-related indicators.",
                },
                "Prediction Agent": {
                    "icon": "🎯",
                    "title": "Prediction Agent",
                    "summary": "Runs the trained Random Forest model and produces the risk probability.",
                },
                "SHAP Explainability Agent": {
                    "icon": "🧠",
                    "title": "SHAP Explainability Agent",
                    "summary": "Explains which patient features contributed most to the model output.",
                },
                "Recommendation Agent": {
                    "icon": "💡",
                    "title": "Recommendation Agent",
                    "summary": "Generates wellness-oriented recommendations from the assessment.",
                },
            }

            agent_targets = {
                "Lifestyle Agent": "Health Assessment",
                "Symptom Agent": "Health Assessment",
                "Prediction Agent": "AI Analysis",
                "SHAP Explainability Agent": "AI Analysis",
                "Recommendation Agent": "Recommendations",
            }
            agent_icons = {
                "Lifestyle Agent": "🌿",
                "Symptom Agent": "🩺",
                "Prediction Agent": "🎯",
                "SHAP Explainability Agent": "🧠",
                "Recommendation Agent": "💡",
            }

            for i, item in enumerate(result.get("trace", [])):
                target = agent_targets.get(item, "AI Analysis")
                icon = agent_icons.get(item, "⚙️")
                st.markdown('<div class="agent-nav-card">', unsafe_allow_html=True)
                if st.button(
                    f"{icon}  {item}   ✓   → Open {target}",
                    key=f"trace_agent_{i}",
                    use_container_width=True
                ):
                    go_to(target, item)
                st.caption(
                    "Click to open this agent's working section and inspect its "
                    "patient-specific output."
                )
                st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------- TRENDS -----------------------------

elif page == "Trends & History":
    p=selected_patient()
    if not p:
        st.info("Select a patient first.")
    else:
        rows=get_assessments(p["patient_id"])
        st.title(f"📈 {p['name']} — Trends & History")
        if not rows:
            st.info("No assessments yet.")
        else:
            df=assessment_dataframe(p["patient_id"])
            df["date"]=pd.to_datetime(df["date"])
            fig=px.line(df,x="date",y="probability",markers=True,
                        title="Model Probability Over Visits")
            fig.update_yaxes(range=[0,1],tickformat=".0%")
            st.plotly_chart(fig,use_container_width=True)

            base,latest=rows[0],rows[-1]
            delta=latest["probability"]-base["probability"]
            a,b,c=st.columns(3)
            with a: metric("Baseline",f"{base['probability']:.1%}")
            with b: metric("Latest",f"{latest['probability']:.1%}")
            with c: metric("Change",f"{delta:+.1%}","Percentage points")

            st.markdown('<div class="section">Interval-wise Trend</div>',
                        unsafe_allow_html=True)
            table=[]
            for i,r in enumerate(rows):
                prev=rows[i-1] if i else None
                d=r["probability"]-prev["probability"] if prev else 0
                table.append({
                    "Visit":i+1,"Date":r["created_at"],
                    "Type":r["assessment_type"],
                    "Probability":f"{r['probability']:.1%}",
                    "Interval Change":"—" if not prev else f"{d:+.1%}",
                    "Risk":r["risk_band"]
                })
            st.dataframe(pd.DataFrame(table),use_container_width=True,
                         hide_index=True)

            st.markdown('<div class="section">What Changed?</div>',
                        unsafe_allow_html=True)
            changes=[]
            keys=sorted(set(base["input"])|set(latest["input"]))
            for f in keys:
                old,new=base["input"].get(f),latest["input"].get(f)
                if str(old)!=str(new):
                    if is_binary(f):
                        old,new=binary_label(f,old),binary_label(f,new)
                    changes.append({"Feature":pretty(f),
                                    "Baseline":old,"Latest":new})
            if changes:
                st.dataframe(pd.DataFrame(changes),
                             use_container_width=True,hide_index=True)
            else:
                st.info("No input values changed between visits.")


# ----------------------------- RECOMMENDATIONS -----------------------------

elif page == "Recommendations":
    if st.session_state.get("agent_focus") == "Recommendation Agent":
        render_agent_focus(st.session_state.result or latest_as_result(selected_patient()),
                           "Recommendation Agent")

    p=selected_patient()
    if not p:
        st.info("Select a patient first.")
    else:
        result=st.session_state.result or latest_as_result(p)
        if not result:
            st.info("Run an assessment first.")
        else:
            st.title("💡 AI Recommendations")
            for i,rec in enumerate(result.get("recommendations",[]),1):
                st.markdown(
                    f'<div class="reco"><b>Recommendation {i}</b>'
                    f'<p style="margin:6px 0 0">{rec}</p></div>',
                    unsafe_allow_html=True
                )


# ----------------------------- WHAT-IF -----------------------------

elif page == "What-if Simulator":
    p=selected_patient()
    if not p:
        st.info("Select a patient first.")
    else:
        result=st.session_state.result or latest_as_result(p)
        if not result:
            st.info("Run an assessment first.")
        else:
            st.markdown(
                """<div class="hero"><h1>🔬 What-if Simulator</h1>
                <p>Explore how the trained model responds to selected
                binary input changes. This is simulation, not causality.</p>
                </div>""",unsafe_allow_html=True)
            base=result["input"]
            old=result["prediction"]["probability"]
            changes={}
            for f,label,fmt in [
                ("Fast food (Y/N)","Fast food",
                 lambda x:"Yes" if x else "No"),
                ("Reg.Exercise(Y/N)","Regular exercise",
                 lambda x:"Yes" if x else "No"),
                ("Cycle(R/I)","Cycle",
                 lambda x:"Irregular" if x else "Regular")
            ]:
                if f in base:
                    cur=int(base[f])
                    changes[f]=st.selectbox(
                        label,[cur,1-cur],format_func=fmt,key=f"what_{f}"
                    )
            if st.button("Simulate Change",type="primary",
                         use_container_width=True):
                new=simulate(MODEL,base,changes)
                a,b,c=st.columns(3)
                with a: metric("Current",f"{old:.1%}")
                with b: metric("Simulated",f"{new:.1%}")
                with c: metric("Change",f"{new-old:+.1%}",
                               "Percentage points")
                st.info("Simulation shows model sensitivity; it does not establish causality.")


# ----------------------------- REPORTS -----------------------------

elif page == "Reports":
    if st.session_state.get("agent_focus") == "Report Engine":
        render_agent_focus(st.session_state.result or latest_as_result(selected_patient()),
                           "Report Engine")

    p=selected_patient()
    if not p:
        st.info("Select a patient first.")
    else:
        rows=get_assessments(p["patient_id"])
        st.markdown(
            f"""<div class="hero"><h1>📄 Reports</h1>
            <p>{p['name']} • {p['patient_code']}</p></div>""",
            unsafe_allow_html=True)
        if not rows:
            st.info("Run an assessment first.")
        else:
            latest=rows[-1]
            a,b,c=st.columns(3)
            with a: metric("Latest",f"{latest['probability']:.1%}")
            with b: metric("Risk Pattern",latest["risk_band"])
            with c: metric("Visits",len(rows))
            st.write(
                "PDF includes current assessment, baseline-to-latest trend, "
                "interval-wise comparison, changed inputs, SHAP drivers and "
                "recommendations."
            )
            if st.button("📄 Generate PDF Report",type="primary",
                         use_container_width=True):
                try:
                    path=generate_pdf(p,rows)
                    with open(path,"rb") as f:
                        st.download_button(
                            "⬇️ Download CycleTwin PDF",
                            f,file_name=path.name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    st.success("PDF generated.")
                except Exception as e:
                    st.error("PDF generation failed.")
                    st.exception(e)
