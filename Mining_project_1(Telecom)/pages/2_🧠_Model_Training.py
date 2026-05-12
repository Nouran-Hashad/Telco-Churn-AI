"""
Page 2 — Model Training & Evaluation
Train, compare, and evaluate ML models with rich visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from utils.data_loader import prepare_data, feature_engineering
from utils.model_utils import (
    train_knn, train_svm, train_nn, train_rf, train_gb,
    evaluate_model, get_feature_importance,
)

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 Model Training & Evaluation")
st.markdown('<p style="color:#94a3b8;font-size:1.05rem;margin-top:-0.5rem;">Train multiple ML models, compare performance, and select the best one.</p>', unsafe_allow_html=True)

if st.session_state.get("df") is None:
    st.warning("⚠️ Please upload a dataset on the **Home** page first.")
    st.stop()

df = st.session_state.df

# ── Sidebar Config ──────────────────────────────────────
st.sidebar.markdown("### ⚙️ Training Configuration")
train_mode = st.sidebar.radio("Training Mode", ["🎯 Single Model", "🏆 Compare All"], horizontal=False)
split_size = st.sidebar.slider("Test Split", 0.1, 0.5, 0.2, step=0.05)

if train_mode == "🎯 Single Model":
    model_choice = st.sidebar.selectbox("Model", ["KNN", "SVM", "Random Forest", "Gradient Boosting", "Neural Network"])

# ── Hyperparameters ─────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Hyperparameters")
knn_k = st.sidebar.slider("KNN — Neighbors (K)", 1, 20, 5)
svm_c = st.sidebar.slider("SVM — Regularization (C)", 0.01, 10.0, 1.0)
rf_trees = st.sidebar.slider("RF — Trees", 10, 200, 100)
rf_depth = st.sidebar.slider("RF — Max Depth", 2, 30, 10)
gb_lr = st.sidebar.slider("GB — Learning Rate", 0.01, 1.0, 0.1)
gb_est = st.sidebar.slider("GB — Estimators", 10, 200, 100)
nn_epochs = st.sidebar.slider("NN — Epochs", 5, 50, 20)

# ── Data Prep ───────────────────────────────────────────
X, y, error = prepare_data(df)
if error:
    st.error(error)
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42, stratify=y)
X_train_s, X_test_s, scaler = feature_engineering(X_train, X_test)
st.session_state["scaler"] = scaler
st.session_state["X_columns"] = X.columns

def dark_layout(fig, title="", h=400):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), title=dict(text=title, font=dict(color="#f1f5f9", size=16)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#94a3b8")), height=h,
    )
    return fig

def train_single(name):
    if name == "KNN":
        m = train_knn(X_train_s, y_train, knn_k)
        return m, evaluate_model(m, X_test_s, y_test, "sklearn"), "sklearn"
    elif name == "SVM":
        m = train_svm(X_train_s, y_train, C=svm_c)
        return m, evaluate_model(m, X_test_s, y_test, "sklearn"), "sklearn"
    elif name == "Random Forest":
        m = train_rf(X_train_s, y_train, rf_trees, rf_depth)
        return m, evaluate_model(m, X_test_s, y_test, "sklearn"), "sklearn"
    elif name == "Gradient Boosting":
        m = train_gb(X_train_s, y_train, gb_lr, gb_est)
        return m, evaluate_model(m, X_test_s, y_test, "sklearn"), "sklearn"
    else:
        m, hist = train_nn(X_train_s, y_train, nn_epochs)
        return m, evaluate_model(m, X_test_s, y_test, "keras"), "keras"

def show_results(name, metrics, model, mtype):
    st.markdown(f'<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"### 📊 {name} — Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("📏 Precision", f"{metrics['precision']:.4f}")
    c3.metric("🔍 Recall", f"{metrics['recall']:.4f}")
    c4.metric("⚖️ F1 Score", f"{metrics['f1']:.4f}")
    st.metric("📈 ROC AUC", f"{metrics['auc']:.4f}")

    rc, cc = st.columns(2)
    with rc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics["fpr"], y=metrics["tpr"], mode="lines",
                                 name=name, line=dict(color="#6366f1", width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance",
                                 line=dict(color="#64748b", dash="dash")))
        fig.add_annotation(x=0.6, y=0.3, text=f"AUC = {metrics['auc']:.4f}",
                          font=dict(color="#a78bfa", size=16), showarrow=False)
        st.plotly_chart(dark_layout(fig, "ROC Curve", 380), use_container_width=True)
    with cc:
        fig = px.imshow(metrics["cm"], text_auto=True, color_continuous_scale=["#1e1b4b","#6366f1","#a78bfa"],
                        x=["No Churn","Churn"], y=["No Churn","Churn"],
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(dark_layout(fig, "Confusion Matrix", 380), use_container_width=True)

    # Feature importance for tree models
    if name in ["Random Forest", "Gradient Boosting"]:
        imp = get_feature_importance(model, list(X.columns), top_n=15)
        if imp:
            st.markdown("### 🌳 Feature Importance")
            fig = go.Figure(go.Bar(
                y=list(imp.keys())[::-1], x=list(imp.values())[::-1],
                orientation="h", marker=dict(color=list(imp.values())[::-1],
                colorscale=[[0,"#6366f1"],[1,"#ec4899"]])))
            st.plotly_chart(dark_layout(fig, f"Top 15 Features — {name}", 450), use_container_width=True)

st.markdown("---")

# ── Single Model Mode ───────────────────────────────────
if train_mode == "🎯 Single Model":
    if st.button(f"🚀 Train {model_choice}", use_container_width=True):
        with st.spinner(f"Training {model_choice}..."):
            model, metrics, mtype = train_single(model_choice)
            st.session_state["trained_model"] = model
            st.session_state["model_type"] = model_choice
        st.success(f"✅ {model_choice} trained successfully!")
        show_results(model_choice, metrics, model, mtype)

# ── Compare All Mode ────────────────────────────────────
else:
    if st.button("🏆 Train & Compare All Models", use_container_width=True):
        models_list = ["KNN", "SVM", "Random Forest", "Gradient Boosting", "Neural Network"]
        all_metrics = {}
        best_auc, best_name, best_model = 0, "", None
        progress = st.progress(0)
        status = st.empty()

        for i, name in enumerate(models_list):
            status.markdown(f"⏳ Training **{name}**... ({i+1}/{len(models_list)})")
            model, metrics, mtype = train_single(name)
            all_metrics[name] = metrics
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_name = name
                best_model = model
            progress.progress((i + 1) / len(models_list))

        status.empty()
        progress.empty()

        st.session_state["trained_model"] = best_model
        st.session_state["model_type"] = best_name

        # Winner card
        st.markdown(f"""
        <div class="model-winner">
            <span style="font-size:2.5rem;">🏆</span>
            <h2 style="color:#10b981!important;border:none!important;margin:0.5rem 0;">
                Best Model: {best_name}
            </h2>
            <p style="color:#94a3b8;font-size:1.1rem;">
                ROC AUC = <strong style="color:#a78bfa;">{best_auc:.4f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Comparison table
        st.markdown("### 📊 Model Comparison")
        comp_data = []
        for name, m in all_metrics.items():
            comp_data.append({
                "Model": name,
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1 Score": f"{m['f1']:.4f}",
                "ROC AUC": f"{m['auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(comp_data).set_index("Model"), use_container_width=True)

        # All ROC curves
        st.markdown("### 📈 ROC Curves — All Models")
        colors = {"KNN":"#6366f1","SVM":"#8b5cf6","Random Forest":"#a78bfa",
                  "Gradient Boosting":"#ec4899","Neural Network":"#06b6d4"}
        fig = go.Figure()
        for name, m in all_metrics.items():
            fig.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines",
                                     name=f"{name} (AUC={m['auc']:.3f})",
                                     line=dict(color=colors.get(name,"#fff"), width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance",
                                 line=dict(color="#64748b", dash="dash")))
        st.plotly_chart(dark_layout(fig, "ROC Comparison", 450), use_container_width=True)

        # Radar chart
        st.markdown("### 🎯 Performance Radar")
        categories = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        fig = go.Figure()
        for name, m in all_metrics.items():
            vals = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]]
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]],
                                          fill="toself", name=name, line=dict(color=colors.get(name,"#fff")),
                                          opacity=0.7))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                     radialaxis=dict(range=[0,1], gridcolor="rgba(255,255,255,0.1)"),
                                     angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                          title=dict(text="Model Performance Radar", font=dict(color="#f1f5f9", size=16)),
                          legend=dict(font=dict(color="#94a3b8")), height=500)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown('<div class="app-footer"><p>🧠 Model Training — Telco Churn AI</p></div>', unsafe_allow_html=True)
