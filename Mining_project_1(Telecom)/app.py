"""
╔══════════════════════════════════════════════════════════════════╗
║              TELCO CUSTOMER CHURN PREDICTION AI                 ║
║         Advanced Machine Learning Analytics Platform            ║
║                                                                  ║
║  A professional Streamlit application for predicting customer    ║
║  churn using KNN, SVM, Random Forest, Gradient Boosting, and    ║
║  Deep Neural Networks. Features interactive EDA, model          ║
║  comparison, and real-time prediction capabilities.             ║
╚══════════════════════════════════════════════════════════════════╝

Author  : Mining Project Team
Course  : Data Mining
Tech    : Python · Streamlit · Scikit-learn · TensorFlow · Plotly
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data


# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn AI — Intelligent Prediction Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Premium CSS Theme ──────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Initialize Session State ───────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None


# ── Sidebar Navigation ─────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <span style="font-size: 2.5rem;">⚡</span>
    <h2 style="margin:0.5rem 0 0; font-size:1.3rem; 
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;">
        Telco Churn AI
    </h2>
    <p style="color: #64748b; font-size: 0.8rem; margin-top:0.3rem;">
        Intelligent Prediction Platform
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navigation")
st.sidebar.info("Use the sidebar pages to explore **Data Analysis**, **Model Training**, and **Predictions**.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 0.8rem; background: rgba(99,102,241,0.08); 
     border-radius: 12px; border: 1px solid rgba(99,102,241,0.15);">
    <p style="margin:0; font-size:0.78rem; color:#94a3b8;">
        <strong style="color:#a78bfa;">Tech Stack</strong><br>
        Python · Streamlit · Scikit-learn<br>
        TensorFlow · Plotly · Pandas
    </p>
</div>
""", unsafe_allow_html=True)


# ── Hero Section ────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <div style="display:inline-flex; align-items:center; gap:8px; padding:6px 18px;
         background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.2);
         border-radius:20px; color:#a78bfa; font-size:0.82rem; font-weight:600;
         letter-spacing:0.04em; margin-bottom:1rem;">
        🔬 ADVANCED DATA MINING PROJECT
    </div>
</div>
""", unsafe_allow_html=True)

st.title("⚡ Telco Customer Churn Prediction AI")

st.markdown("""
<p style="text-align:center; font-size:1.15rem; color:#94a3b8; max-width:700px; margin:0 auto 2rem;
   line-height:1.8;">
    Harness the power of <strong style="color:#a78bfa;">Machine Learning</strong> and 
    <strong style="color:#ec4899;">Deep Neural Networks</strong> to predict customer churn 
    with precision. Explore, train, and deploy — all in one platform.
</p>
""", unsafe_allow_html=True)


# ── Feature Cards ───────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card" style="text-align:center; min-height:220px;">
        <span class="feature-icon">📊</span>
        <h3 style="margin:0 0 0.5rem; color:#a78bfa !important;">Explore & Analyze</h3>
        <p style="font-size:0.9rem; color:#94a3b8 !important;">
            Interactive visualizations with correlation heatmaps, 
            distribution analysis, and churn breakdowns.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card" style="text-align:center; min-height:220px;">
        <span class="feature-icon" style="animation-delay:0.5s;">🧠</span>
        <h3 style="margin:0 0 0.5rem; color:#a78bfa !important;">Train & Compare</h3>
        <p style="font-size:0.9rem; color:#94a3b8 !important;">
            Build KNN, SVM, Random Forest, Gradient Boosting, 
            and Neural Network models — compare performance instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card" style="text-align:center; min-height:220px;">
        <span class="feature-icon" style="animation-delay:1s;">🔮</span>
        <h3 style="margin:0 0 0.5rem; color:#a78bfa !important;">Predict & Retain</h3>
        <p style="font-size:0.9rem; color:#94a3b8 !important;">
            Real-time churn probability predictions with risk 
            assessment and personalized retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")


# ── Dataset Upload Section ──────────────────────────────────────
st.markdown("""
<h2 style="border:none !important; padding-bottom:0 !important;">
    📂 Get Started
</h2>
<p style="color:#64748b; margin-top:0.3rem;">
    Upload the Telco Customer Churn dataset to unlock all features.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop your CSV file here or click to browse",
    type="csv",
    help="Upload the 'Telco-Customer-Churn.csv' dataset to begin analysis"
)

if uploaded_file is not None:
    df, error = load_data(uploaded_file)

    if error:
        st.error(f"❌ Error loading data: {error}")
    else:
        st.session_state.df = df
        st.success("✅ Dataset loaded successfully! Navigate to **Data Analysis** to begin exploring.")

        st.markdown("---")

        # ── Live KPI Dashboard ──────────────────────────────────
        st.markdown("""
        <h2 style="border:none !important; padding-bottom:0 !important;">
            ⚡ Dataset at a Glance
        </h2>
        """, unsafe_allow_html=True)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        total_customers = len(df)
        total_features = df.shape[1]
        churn_rate = df["Churn"].mean() * 100 if "Churn" in df.columns else 0
        avg_tenure = df["tenure"].mean() if "tenure" in df.columns else 0

        kpi1.metric("👥 Total Customers", f"{total_customers:,}")
        kpi2.metric("📋 Features", f"{total_features}")
        kpi3.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
        kpi4.metric("📅 Avg Tenure", f"{avg_tenure:.0f} mo")

        # ── Quick Churn Donut Chart ─────────────────────────────
        if "Churn" in df.columns:
            st.markdown("<br>", unsafe_allow_html=True)

            chart_col1, chart_col2 = st.columns([1, 1])

            with chart_col1:
                churn_counts = df["Churn"].value_counts()
                fig_donut = go.Figure(data=[go.Pie(
                    labels=["Retained", "Churned"],
                    values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                    hole=0.65,
                    marker=dict(
                        colors=["#6366f1", "#ec4899"],
                        line=dict(color="#0a0e1a", width=3)
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=14, color="white"),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>"
                )])
                fig_donut.update_layout(
                    title=dict(text="Customer Retention Overview", font=dict(color="#f1f5f9", size=16)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    showlegend=True,
                    legend=dict(font=dict(color="#94a3b8")),
                    height=350,
                    margin=dict(t=50, b=20, l=20, r=20),
                    annotations=[dict(
                        text=f"<b>{churn_rate:.1f}%</b><br>Churn",
                        x=0.5, y=0.5,
                        font=dict(size=18, color="#a78bfa"),
                        showarrow=False
                    )]
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with chart_col2:
                # Monthly charges distribution by churn
                fig_dist = px.histogram(
                    df, x="MonthlyCharges", color="Churn",
                    barmode="overlay",
                    color_discrete_map={0: "#6366f1", 1: "#ec4899"},
                    labels={"Churn": "Churned", "MonthlyCharges": "Monthly Charges ($)"},
                    title="Monthly Charges by Churn Status",
                    opacity=0.7
                )
                fig_dist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    title_font=dict(color="#f1f5f9", size=16),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    height=350,
                    margin=dict(t=50, b=20, l=20, r=20),
                    legend=dict(font=dict(color="#94a3b8"))
                )
                st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem 2rem; background:rgba(255,255,255,0.02);
         border:1px dashed rgba(99,102,241,0.2); border-radius:16px; margin-top:1rem;">
        <span style="font-size:3rem; display:block; margin-bottom:1rem;">📁</span>
        <p style="color:#94a3b8; font-size:1.05rem; margin:0;">
            Upload the <strong style="color:#a78bfa;">Telco-Customer-Churn.csv</strong> dataset to get started
        </p>
        <p style="color:#64748b; font-size:0.85rem; margin-top:0.5rem;">
            The dataset should contain customer demographics, services, and churn labels
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="app-footer">
    <p>⚡ <strong>Telco Churn AI</strong> — Advanced Data Mining Project</p>
    <p>Built with Python · Streamlit · Scikit-learn · TensorFlow · Plotly</p>
</div>
""", unsafe_allow_html=True)
