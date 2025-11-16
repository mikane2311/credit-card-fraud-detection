import streamlit as st
import joblib
import numpy as np

# ===== Load models =====
lr_model = joblib.load("model_lr.pkl")
lgb_model = joblib.load("model_lgb.pkl")
if_model = joblib.load("model_isoforest.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Use one of the machine learning models to predict whether a transaction is fraudulent.")

# ===== User model selector ====
model_choice = st.selectbox(
    "Choose which model to use:",
    ("Logistic Regression", "LightGBM", "Isolation Forest")
)

# ===== Inputs =====
st.subheader("Transaction Inputs")

amount = st.number_input("Transaction Amount", min_value=0.0, max_value=50000.0, value=100.0)
time = st.number_input("Time (seconds)", min_value=0.0, max_value=200000.0, value=10000.0)

components = []
for i in range(1, 29):
    val = st.slider(f"V{i}", -20.0, 20.0, 0.0)
    components.append(val)

# Build single input row
features = np.array([time, amount] + components).reshape(1, -1)
features_scaled = scaler.transform(features)

# ===== Prediction logic =====
st.subheader("🔍 Prediction")

if st.button("Predict"):

    if model_choice == "Logistic Regression":
        proba = lr_model.predict_proba(features_scaled)[0, 1]
        y_pred = int(proba >= 0.5)
        st.write(f"Probability of Fraud: **{proba:.4f}**")

    elif model_choice == "LightGBM":
        proba = lgb_model.predict_proba(features_scaled)[0, 1]
        y_pred = int(proba >= 0.5)
        st.write(f"Probability of Fraud: **{proba:.4f}**")

    else:  # Isolation Forest
        # Isolation Forest predicts +1 (normal) and -1 (fraud)
        raw_pred = if_model.predict(features_scaled)[0]
        y_pred = 0 if raw_pred == 1 else 1  
        score = if_model.decision_function(features_scaled)[0]
        st.write(f"Anomaly Score (lower = more anomalous): **{score:.4f}**")

    # Output prediction text
    if y_pred == 1:
        st.error("🚨 **Fraud Detected**")
    else:
        st.success("✅ **Normal Transaction**")

