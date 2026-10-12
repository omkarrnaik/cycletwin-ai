import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import init_db, get_or_create_user, save_assessment, get_history
from orchestrator import CycleTwinOrchestrator
from what_if import simulate

st.set_page_config(
    page_title="CycleTwin AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
:root {
    --purple: #6C3CEB;
    --purple2: #8B5CF6;
    --ink: #172033;
    --muted: #6B7280;
    --soft: #F6F4FF;
    --line: #E8E5F4;
    --pink: #FF4F81;
    --green: #22A06B;
    --orange: #F59E0B;
}
.block-container {padding-top: 1.4rem; max-width: 1500px;}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#FBFAFF 0%,#F4F0FF 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stMarkdown h1 {
    color: var(--purple);
}
.hero {
    background: linear-gradient(135deg,#F6F2FF 0%,#FFFFFF 60%,#FFF7FB 100%);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 18px 20px;
    min-height: 145px;
    box-shadow: 0 5px 20px rgba(44,30,90,.05);
}
.card h4 {margin:0;color:#303A52;font-size:14px;}
.big {font-size:34px;font-weight:750;color:var(--purple);margin-top:7px;}
.sub {font-size:13px;color:var(--muted);}
.pill {display:inline-block;padding:5px 10px;border-radius:999px;background:#F1ECFF;color:#6331D7;font-weight:700;font-size:12px;}
.section-title {font-size:21px;font-weight:750;color:#202A43;margin:20px 0 10px;}
.agent {
    border:1px solid var(--line); border-radius:14px; padding:12px;
    background:#fff; margin-bottom:9px;
}
.agent-ok {color:var(--green);font-weight:700;}
.warning-box {
    background:#FFF5F7;border:1px solid #FFD7E2;border-radius:14px;
    padding:13px 15px;color:#7B2945;font-size:12px;
}
.reco {
    border:1px solid var(--line); border-radius:14px; padding:14px;
    margin-bottom:9px; background:#fff;
}
.footer {text-align:center;color:#8B91A1;font-size:12px;padding:24px;}
</style>
""", unsafe_allow_html=True)

init_db()

with open("models/rf_pcos_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/feature_schema.pkl", "rb") as f:
    FEATURES = pickle.load(f)
with open("models/feature_medians.json", "r") as f:
    MEDIANS = json.load(f)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌸 CycleTwin AI")
    st.caption("Women's Wellness Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Health Assessment", "AI Analysis", "Trends & History",
         "Recommendations", "What-if Simulator", "Reports"],
        label_visibility="collapsed"
    )

    st.divider()
    name = st.text_input("User name", value="Ananya")
    user_id = get_or_create_user(name)

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Not a medical diagnosis.</b><br>
    This prototype is for educational risk awareness and wellness support only.
    </div>
    """, unsafe_allow_html=True)

# ---------- Helpers ----------
def collect_input():
    user_input = {}
    with st.sidebar.expander("Quick assessment input", expanded=False):
        for feature in FEATURES:
            if "(Y/N)" in feature or feature in ["Cycle(R/I)", "Reg.Exercise(Y/N)", "Fast food (Y/N)"]:
                user_input[feature] = st.selectbox(
                    feature, [0,1],
                    format_func=lambda x: "No / Regular" if x == 0 else "Yes / Irregular",
                    key="side_"+feature
                )
            else:
                user_input[feature] = st.number_input(
                    feature,
                    value=float(MEDIANS.get(feature, 0.0)),
                    key="side_"+feature
                )
    return user_input

def run_assessment(user_input):
    result = CycleTwinOrchestrator().run(user_input)
    save_assessment(user_id, result)
    return result

def metric_card(title, value, subtitle="", accent="purple"):
    st.markdown(
        f"""<div class="card">
        <h4>{title}</h4>
        <div class="big">{value}</div>
        <div class="sub">{subtitle}</div>
        </div>""",
        unsafe_allow_html=True
    )

# ---------- SHAP visualization ----------
SHAP_GREEN = "#22A06B"   # positive SHAP -> higher predicted PCOS risk
SHAP_RED = "#DC2626"     # negative SHAP -> lower predicted PCOS risk
SHAP_GREEN_LIGHT = (200, 245, 222)   # weak positive contribution
SHAP_GREEN_DARK = (10, 84, 55)       # strong positive contribution
SHAP_RED_LIGHT = (253, 214, 214)     # weak negative contribution
SHAP_RED_DARK = (127, 15, 15)        # strong negative contribution

def _lerp_color(light_rgb, dark_rgb, t):
    t = max(0.0, min(1.0, t))
    r = round(light_rgb[0] + (dark_rgb[0] - light_rgb[0]) * t)
    g = round(light_rgb[1] + (dark_rgb[1] - light_rgb[1]) * t)
    b = round(light_rgb[2] + (dark_rgb[2] - light_rgb[2]) * t)
    return f"#{r:02X}{g:02X}{b:02X}"

def shap_bar_figure(rows, top_n=8, height=360):
    """Horizontal bar chart of SHAP contributions. Bars stay green for a
    positive contribution (higher predicted PCOS risk) and red for a
    negative one (lower predicted PCOS risk), but the shade deepens with
    the strength of the contribution so the chart reads at a glance,
    sorted by magnitude with the most influential feature first."""
    d = pd.DataFrame(rows).copy()
    if d.empty:
        return None
    if "abs_shap" not in d.columns:
        d["abs_shap"] = d["shap_value"].abs()
    d = d.sort_values("abs_shap", ascending=False).head(top_n)
    # Ascending order so the most influential feature renders at the top
    # of the horizontal bar chart.
    d = d.sort_values("abs_shap", ascending=True)

    max_abs = d["abs_shap"].max() or 1.0
    bar_colors = []
    line_colors = []
    for v, a in zip(d["shap_value"], d["abs_shap"]):
        strength = a / max_abs
        if v > 0:
            bar_colors.append(_lerp_color(SHAP_GREEN_LIGHT, SHAP_GREEN_DARK, strength))
            line_colors.append("#0A5437")
        else:
            bar_colors.append(_lerp_color(SHAP_RED_LIGHT, SHAP_RED_DARK, strength))
            line_colors.append("#7F0F0F")

    fig = go.Figure(go.Bar(
        x=d["shap_value"],
        y=d["feature"],
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color=line_colors, width=1.2)
        ),
        text=[f"{v:+.3f}" for v in d["shap_value"]],
        textposition="outside",
        textfont=dict(size=12, color="#202A43"),
        customdata=d["value"] if "value" in d.columns else None,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<br>Value: %{customdata}<extra></extra>"
    ))
    fig.add_vline(x=0, line_width=1.5, line_color="#6B7280")
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=40, t=15, b=10),
        showlegend=False,
        xaxis_title="SHAP contribution",
        plot_bgcolor="#FBFAFF",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#E8E5F4", zeroline=False),
        yaxis=dict(showgrid=False),
        font=dict(size=12)
    )
    return fig

def shap_legend():
    st.markdown(
        f"""<div style="font-size:13px;line-height:1.7;color:#303A52;">
        <div style="margin-bottom:6px;">
        <span style="display:inline-block;width:70px;height:10px;border-radius:5px;
        background:linear-gradient(90deg, rgb{SHAP_GREEN_LIGHT}, rgb{SHAP_GREEN_DARK});
        vertical-align:middle;margin-right:8px;"></span>
        🟢 <b>Green</b> (deeper = stronger) = pushes prediction toward <b>higher</b> PCOS risk
        </div>
        <div>
        <span style="display:inline-block;width:70px;height:10px;border-radius:5px;
        background:linear-gradient(90deg, rgb{SHAP_RED_LIGHT}, rgb{SHAP_RED_DARK});
        vertical-align:middle;margin-right:8px;"></span>
        🔴 <b>Red</b> (deeper = stronger) = pushes prediction toward <b>lower</b> PCOS risk
        </div>
        </div>""",
        unsafe_allow_html=True
    )
    st.caption(
        "SHAP values explain how each feature influenced this individual prediction "
        "relative to the model's baseline output. They describe model behavior and "
        "do not prove causation."
    )

# ---------- Health Assessment ----------
if page == "Health Assessment":
    st.markdown('<div class="hero"><h1>Health Assessment</h1><p>Enter the structured health information used by the Phase-2 Random Forest model.</p></div>', unsafe_allow_html=True)
    inputs = collect_input()
    if st.button("Run AI Assessment", type="primary"):
        st.session_state["result"] = run_assessment(inputs)
        st.success("Assessment saved to your local CycleTwin database.")
        st.rerun()

# ---------- Dashboard ----------
elif page == "Dashboard":
    history = get_history(user_id)

    st.markdown(
        f"""<div class="hero">
        <h1>Welcome, {name} 👋</h1>
        <p>Here's your AI-powered PCOS wellness overview.</p>
        <span class="pill">Explainable AI</span>
        <span class="pill">Multi-Agent</span>
        <span class="pill">Personal History</span>
        </div>""",
        unsafe_allow_html=True
    )

    if history.empty and "result" not in st.session_state:
        st.info("No assessment yet. Open **Health Assessment** and run your first analysis.")
    else:
        result = st.session_state.get("result")
        if result is None:
            latest = history.iloc[-1]
            probability = float(latest["probability"])
            risk_band = latest["risk_band"]
            top_factors = []
            recs = []
        else:
            probability = result["prediction"]["probability"]
            risk_band = result["prediction"]["risk_band"]
            top_factors = result.get("shap_rows", [])[:6]
            recs = result.get("recommendations", [])

        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("PCOS Risk Score", f"{probability*100:.0f}/100", risk_band)
        with c2: metric_card("Risk Probability", f"{probability:.2f}", f"{probability:.0%} model probability")
        with c3: metric_card("Cycle Status", "Irregular" if result and result["input"].get("Cycle(R/I)") == 1 else "Tracked", "Based on current input")
        with c4:
            wellness = max(0, min(100, int((1-probability)*100)))
            metric_card("Overall Wellness", f"{wellness}/100", "Model-derived awareness score")

        st.markdown('<div class="section-title">AI Analysis</div>', unsafe_allow_html=True)
        left, mid, right = st.columns([1.2,1.4,0.8])

        with left:
            st.markdown('<div class="card"><h4>Top Factors Influencing Your Risk</h4></div>', unsafe_allow_html=True)
            if top_factors:
                fig = shap_bar_figure(top_factors, top_n=8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Run a new assessment to see patient-specific SHAP drivers.")

        with mid:
            st.markdown('<div class="card"><h4>SHAP Summary — Model Explanation</h4></div>', unsafe_allow_html=True)
            if top_factors:
                shap_legend()
            else:
                st.caption("SHAP explanation appears after an assessment.")

        with right:
            st.markdown('<div class="card"><h4>AI Recommendations</h4></div>', unsafe_allow_html=True)
            for rec in recs[:5]:
                st.markdown(f'<div class="reco">💡 {rec}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Trends & Agent Status</div>', unsafe_allow_html=True)
        a,b = st.columns([1.6,1])
        with a:
            if not history.empty:
                chart = history.copy()
                chart["created_at"] = pd.to_datetime(chart["created_at"])
                fig = px.line(chart, x="created_at", y="probability", markers=True,
                              labels={"created_at":"Assessment date","probability":"Risk probability"})
                fig.update_yaxes(range=[0,1], tickformat=".0%")
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)
        with b:
            agents = ["Lifestyle Agent","Symptom Agent","Prediction Agent","Explainability Agent","Recommendation Agent","Report Agent"]
            for ag in agents:
                st.markdown(f'<div class="agent">🟢 <span class="agent-ok">{ag}</span> <span style="float:right">✓</span></div>', unsafe_allow_html=True)

        st.markdown("<div class=\"footer\">CycleTwin AI • Built for women's wellness research • Educational prototype</div>", unsafe_allow_html=True)

# ---------- AI Analysis ----------
elif page == "AI Analysis":
    st.markdown('<div class="hero"><h1>AI Analysis</h1><p>See the reasoning pipeline behind your latest assessment.</p></div>', unsafe_allow_html=True)
    result = st.session_state.get("result")
    if not result:
        st.info("Run an assessment first.")
    else:
        st.success(f"Assessment: {result['prediction']['risk_band']} • Probability {result['prediction']['probability']:.1%}")
        st.subheader("Agent Trace")
        st.code(" → ".join(result["trace"]))
        st.subheader("Patient-specific SHAP drivers")
        fig = shap_bar_figure(result["shap_rows"], top_n=12)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        shap_legend()

# ---------- Trends ----------
elif page == "Trends & History":
    st.markdown('<div class="hero"><h1>Trends & History</h1><p>Your saved assessment history from the local SQLite database.</p></div>', unsafe_allow_html=True)
    history = get_history(user_id)
    if history.empty:
        st.info("No history yet.")
    else:
        history["created_at"] = pd.to_datetime(history["created_at"])
        fig = px.line(history, x="created_at", y="probability", markers=True)
        fig.update_yaxes(range=[0,1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(history.sort_values("created_at", ascending=False), use_container_width=True, hide_index=True)

# ---------- Recommendations ----------
elif page == "Recommendations":
    st.markdown('<div class="hero"><h1>Recommendations</h1><p>General wellness suggestions generated from your latest structured assessment.</p></div>', unsafe_allow_html=True)
    result = st.session_state.get("result")
    if not result:
        st.info("Run an assessment first.")
    else:
        for i, rec in enumerate(result["recommendations"], 1):
            st.markdown(f'<div class="reco"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)

# ---------- What-if ----------
elif page == "What-if Simulator":
    st.markdown('<div class="hero"><h1>What-if Simulator</h1><p>Explore how changing selected inputs changes the model output. This is a simulation, not a causal prediction.</p></div>', unsafe_allow_html=True)
    result = st.session_state.get("result")
    if not result:
        st.info("Run an assessment first.")
    else:
        base = result["input"]
        st.write("Current probability:", f"**{result['prediction']['probability']:.1%}**")

        changes = {}
        # Only simulate safe, interpretable binary lifestyle inputs in this college prototype.
        changes["Fast food (Y/N)"] = st.selectbox("Fast food", [base["Fast food (Y/N)"], 1-base["Fast food (Y/N)"]],
                                                  format_func=lambda x: "Yes" if x else "No")
        changes["Reg.Exercise(Y/N)"] = st.selectbox("Regular exercise", [base["Reg.Exercise(Y/N)"], 1-base["Reg.Exercise(Y/N)"]],
                                                     format_func=lambda x: "Yes" if x else "No")
        changes["Cycle(R/I)"] = st.selectbox("Cycle", [base["Cycle(R/I)"], 1-base["Cycle(R/I)"]],
                                             format_func=lambda x: "Irregular" if x else "Regular")

        if st.button("Simulate change", type="primary"):
            new_p = simulate(model, base, changes)
            delta = new_p - result["prediction"]["probability"]
            c1,c2,c3 = st.columns(3)
            c1.metric("Current", f"{result['prediction']['probability']:.1%}")
            c2.metric("Simulated", f"{new_p:.1%}")
            c3.metric("Change", f"{delta:+.1%}")
            st.info("The simulator shows how the trained model responds to changed inputs; it does not prove that changing a habit will cause the predicted change.")

# ---------- Reports ----------
elif page == "Reports":
    st.markdown('<div class="hero"><h1>Reports</h1><p>Generate a concise AI assessment report for review.</p></div>', unsafe_allow_html=True)
    result = st.session_state.get("result")
    if not result:
        st.info("Run an assessment first.")
    else:
        if st.button("Generate PDF Report", type="primary"):
            path = CycleTwinOrchestrator().generate_report(result, "reports/cycletwin_report.pdf")
            st.success("Report generated.")
            with open(path, "rb") as f:
                st.download_button("Download Report (PDF)", f, file_name="CycleTwin_AI_Report.pdf")
