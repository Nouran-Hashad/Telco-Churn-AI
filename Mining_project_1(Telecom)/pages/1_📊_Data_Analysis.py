"""
Page 1 — Interactive Data Analysis
Rich EDA with Plotly visualizations, KPI cards, and churn insights.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Analysis", page_icon="📊", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Interactive Data Analysis")
st.markdown('<p style="color:#94a3b8;font-size:1.05rem;margin-top:-0.5rem;">Deep-dive into customer data with interactive visualizations.</p>', unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ Please upload a dataset on the **Home** page first.")
    st.stop()

df = st.session_state.df
st.markdown("---")

# ── KPI Cards ───────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
churn_rate = df["Churn"].mean() * 100 if "Churn" in df.columns else 0
k1.metric("👥 Customers", f"{len(df):,}")
k2.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
k3.metric("📅 Avg Tenure", f"{df['tenure'].mean():.0f} mo" if "tenure" in df.columns else "N/A")
k4.metric("💳 Avg Monthly", f"${df['MonthlyCharges'].mean():.0f}" if "MonthlyCharges" in df.columns else "N/A")
k5.metric("💰 Avg Total", f"${df['TotalCharges'].mean():,.0f}" if "TotalCharges" in df.columns else "N/A")

st.markdown("---")
num_cols = list(df.select_dtypes(include=["float64", "int64", "int32"]).columns)
cat_cols = list(df.select_dtypes(include=["object"]).columns)

# Chart theme helper
def dark_layout(fig, title="", h=400):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), title=dict(text=title, font=dict(color="#f1f5f9", size=16)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#94a3b8")), height=h,
        coloraxis_showscale=False,
    )
    return fig

tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📈 Distributions", "🔥 Correlations", "🎯 Churn Insights"])

# ── Tab 1: Dataset ──────────────────────────────────────
with tab1:
    st.markdown("### 📋 Dataset Viewer")
    st.dataframe(df, use_container_width=True, height=400)
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><h4 style="color:#6366f1!important;margin-top:0;">🔢 Numerical</h4></div>', unsafe_allow_html=True)
        for c in num_cols:
            st.markdown(f"- `{c}` — {df[c].nunique()} unique")
    with c2:
        st.markdown('<div class="glass-card"><h4 style="color:#ec4899!important;margin-top:0;">🏷️ Categorical</h4></div>', unsafe_allow_html=True)
        for c in cat_cols:
            st.markdown(f"- `{c}` — {df[c].nunique()} unique")

# ── Tab 2: Distributions ───────────────────────────────
with tab2:
    col_type = st.radio("Feature Type", ["Numerical", "Categorical"], horizontal=True)
    if col_type == "Numerical":
        sel = st.selectbox("Column", num_cols, key="ndist")
        fig = px.histogram(df, x=sel, color="Churn", barmode="overlay", opacity=0.75,
                           color_discrete_map={0: "#6366f1", 1: "#ec4899"})
        st.plotly_chart(dark_layout(fig, f"Distribution of {sel} by Churn"), use_container_width=True)
        fig2 = px.box(df, x="Churn", y=sel, color="Churn", points="outliers",
                      color_discrete_map={0: "#6366f1", 1: "#ec4899"})
        st.plotly_chart(dark_layout(fig2, f"Box Plot — {sel}"), use_container_width=True)
    else:
        sel = st.selectbox("Column", cat_cols, key="cdist")
        fig = px.histogram(df, x=sel, color="Churn", barmode="group",
                           color_discrete_map={0: "#6366f1", 1: "#ec4899"})
        st.plotly_chart(dark_layout(fig, f"Distribution of {sel} by Churn"), use_container_width=True)
        if "Churn" in df.columns:
            cr = df.groupby(sel)["Churn"].mean().sort_values(ascending=False) * 100
            fig2 = px.bar(x=cr.index, y=cr.values, color=cr.values,
                          color_continuous_scale=["#6366f1", "#ec4899"],
                          labels={"x": sel, "y": "Churn Rate (%)"})
            st.plotly_chart(dark_layout(fig2, f"Churn Rate by {sel}"), use_container_width=True)

# ── Tab 3: Correlations ────────────────────────────────
with tab3:
    corr = df.select_dtypes(include=["number"]).corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#94a3b8"), title=dict(text="Correlation Matrix", font=dict(color="#f1f5f9", size=16)),
                      height=550, coloraxis_showscale=True)
    st.plotly_chart(fig, use_container_width=True)
    if "Churn" in corr.columns:
        st.markdown("### 🎯 Top Correlations with Churn")
        cc = corr["Churn"].drop("Churn").abs().sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(y=cc.index, x=cc.values, orientation="h",
                                marker=dict(color=cc.values, colorscale=[[0,"#6366f1"],[1,"#ec4899"]]),
                                text=[f"{v:.3f}" for v in cc.values], textposition="outside",
                                textfont=dict(color="#94a3b8")))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#94a3b8"), title=dict(text="Absolute Correlation with Churn", font=dict(color="#f1f5f9")),
                           xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           height=max(300, len(cc)*35), margin=dict(l=150))
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 4: Churn Insights ──────────────────────────────
with tab4:
    if "Churn" not in df.columns:
        st.warning("Churn column not found.")
        st.stop()
    c1, c2 = st.columns(2)
    with c1:
        if "Contract" in df.columns:
            ct = df.groupby("Contract")["Churn"].mean().reset_index()
            ct["Churn"] = ct["Churn"] * 100
            fig = px.bar(ct, x="Contract", y="Churn", color="Churn", text_auto=".1f",
                         color_continuous_scale=["#6366f1", "#ec4899"], labels={"Churn": "Churn Rate (%)"})
            st.plotly_chart(dark_layout(fig, "Churn by Contract Type", 350), use_container_width=True)
    with c2:
        if "InternetService" in df.columns:
            inet = df.groupby("InternetService")["Churn"].mean().reset_index()
            inet["Churn"] = inet["Churn"] * 100
            fig = px.bar(inet, x="InternetService", y="Churn", color="Churn", text_auto=".1f",
                         color_continuous_scale=["#6366f1", "#ec4899"], labels={"Churn": "Churn Rate (%)"})
            st.plotly_chart(dark_layout(fig, "Churn by Internet Service", 350), use_container_width=True)
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        st.markdown("### 🔬 Tenure vs Monthly Charges")
        fig = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn", opacity=0.5,
                         color_discrete_map={0: "#6366f1", 1: "#ec4899"},
                         labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)"})
        st.plotly_chart(dark_layout(fig, "Customer Landscape", 450), use_container_width=True)
    if "PaymentMethod" in df.columns:
        st.markdown("### 💳 Churn by Payment Method")
        pm = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        pm["Churn"] = pm["Churn"] * 100
        fig = px.bar(pm, x="PaymentMethod", y="Churn", color="Churn", text_auto=".1f",
                     color_continuous_scale=["#6366f1", "#ec4899"], labels={"Churn": "Churn Rate (%)"})
        st.plotly_chart(dark_layout(fig, "Churn Rate by Payment Method", 400), use_container_width=True)

st.markdown("---")
st.markdown('<div class="app-footer"><p>📊 Data Analysis — Telco Churn AI</p></div>', unsafe_allow_html=True)
