"""
Page 3 — Prediction System
Real-time churn prediction with full customer input form and visual gauge.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Prediction System", page_icon="🔮", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🔮 Real-Time Prediction System")
st.markdown('<p style="color:#94a3b8;font-size:1.05rem;margin-top:-0.5rem;">Enter customer details to predict churn probability with AI-powered analysis.</p>', unsafe_allow_html=True)

if "trained_model" not in st.session_state or "scaler" not in st.session_state:
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;background:rgba(255,255,255,0.02);
         border:1px dashed rgba(245,158,11,0.3);border-radius:16px;margin-top:1rem;">
        <span style="font-size:3rem;display:block;margin-bottom:1rem;">🧠</span>
        <h3 style="color:#f59e0b!important;">Model Not Trained Yet</h3>
        <p style="color:#94a3b8;">Please train a model on the <strong>Model Training</strong> page first.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

model = st.session_state["trained_model"]
scaler = st.session_state["scaler"]
feature_names = st.session_state["X_columns"]
model_type = st.session_state["model_type"]

st.markdown(f"""
<div class="glass-card" style="text-align:center;padding:1rem;">
    <p style="margin:0;color:#94a3b8;">Active Model: <strong style="color:#a78bfa;">{model_type}</strong></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Customer Input Form ─────────────────────────────────
st.markdown("### 📋 Customer Information")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
with col2:
    tenure = st.number_input("Tenure (Months)", 0, 100, 12)
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0, step=50.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with col3:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])

st.markdown("### 🔒 Additional Services")
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
with sc2:
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
with sc3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
with sc4:
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("---")

# ── Predict ─────────────────────────────────────────────
if st.button("🔮 Predict Churn Probability", use_container_width=True):
    # Build raw customer record
    customer = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple_lines,
        "InternetService": internet, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }

    input_df = pd.DataFrame([customer])
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training features
    input_aligned = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    for col in input_encoded.columns:
        if col in input_aligned.columns:
            input_aligned[col] = input_encoded[col].values[0]

    input_scaled = scaler.transform(input_aligned)

    # Predict
    if model_type == "Neural Network":
        prob = float(model.predict(input_scaled, verbose=0).ravel()[0])
    else:
        prob = float(model.predict_proba(input_scaled)[:, 1][0])

    # Risk level
    if prob < 0.3:
        risk, risk_color, risk_icon = "LOW RISK", "#10b981", "🟢"
    elif prob < 0.6:
        risk, risk_color, risk_icon = "MEDIUM RISK", "#f59e0b", "🟡"
    elif prob < 0.8:
        risk, risk_color, risk_icon = "HIGH RISK", "#f97316", "🟠"
    else:
        risk, risk_color, risk_icon = "CRITICAL RISK", "#ef4444", "🔴"

    st.markdown("---")

    # Results
    r1, r2 = st.columns([1, 1])

    with r1:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number=dict(suffix="%", font=dict(size=40, color="#f1f5f9")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#94a3b8", tickfont=dict(color="#94a3b8")),
                bar=dict(color=risk_color, thickness=0.3),
                bgcolor="rgba(255,255,255,0.03)",
                borderwidth=2, bordercolor="rgba(255,255,255,0.08)",
                steps=[
                    dict(range=[0, 30], color="rgba(16,185,129,0.15)"),
                    dict(range=[30, 60], color="rgba(245,158,11,0.15)"),
                    dict(range=[60, 80], color="rgba(249,115,22,0.15)"),
                    dict(range=[80, 100], color="rgba(239,68,68,0.15)"),
                ],
                threshold=dict(line=dict(color=risk_color, width=4), thickness=0.8, value=prob*100),
            ),
            title=dict(text="Churn Probability", font=dict(color="#94a3b8", size=16)),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"), height=350,
            margin=dict(t=80, b=20, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:2rem;min-height:280px;
             display:flex;flex-direction:column;justify-content:center;">
            <span style="font-size:3.5rem;display:block;">{risk_icon}</span>
            <h2 style="color:{risk_color}!important;border:none!important;margin:1rem 0 0.5rem;
                font-size:1.8rem!important;">{risk}</h2>
            <p class="kpi-number" style="font-size:2.5rem;">{prob*100:.1f}%</p>
            <p style="color:#94a3b8;margin-top:0.5rem;">Probability of churning</p>
        </div>
        """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("### 💡 Retention Recommendations")
    if prob >= 0.6:
        recs = []
        if contract == "Month-to-month":
            recs.append("📝 **Offer a discounted annual contract** — month-to-month customers churn at 3x the rate.")
        if internet == "Fiber optic":
            recs.append("🌐 **Review fiber optic pricing** — high costs drive churn in fiber customers.")
        if tenure < 12:
            recs.append("🎁 **Activate new-customer loyalty program** — first-year retention is critical.")
        if online_security == "No" or tech_support == "No":
            recs.append("🔒 **Bundle free security & tech support for 3 months** — service add-ons reduce churn.")
        if payment == "Electronic check":
            recs.append("💳 **Incentivize auto-pay enrollment** — electronic check users have highest churn.")
        if monthly > 70:
            recs.append("💰 **Offer a personalized discount** — high monthly charges are a key churn driver.")
        if not recs:
            recs.append("📞 **Proactive outreach** — contact customer to understand pain points.")
        for r in recs:
            st.markdown(r)
    elif prob >= 0.3:
        st.info("⚡ **Moderate risk** — Consider a loyalty reward or small upgrade to strengthen retention.")
    else:
        st.success("🎉 **Low risk!** This customer appears satisfied. Maintain current service quality.")
        st.balloons()

st.markdown("---")
st.markdown('<div class="app-footer"><p>🔮 Prediction System — Telco Churn AI</p></div>', unsafe_allow_html=True)
