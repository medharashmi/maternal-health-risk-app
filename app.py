# app.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# =========================
# CONFIG / PATHS
# =========================
MODEL_PATH = Path("model.json")          # XGBoost Booster file (converted from your .pkl)
LABELS_PATH = Path("class_labels.json")  # ["Low", "Moderate", "High"]

# Order must match training
FEATURE_ORDER = [
    "Age", "Systolic_BP", "Diastolic_BP", "Hemoglobin", "BMI",
    "Prenatal_Visits", "Past_Complications", "Blood_Sugar", "Heart_Rate",
    "Fetal_Movement_Score", "BodyTemp", "Trimester_Num",
    "BP_Diff", "Anemia_Flag", "Obesity_Flag"
]

CLASS_LABELS_FALLBACK = ["Low", "Moderate", "High"]

st.set_page_config(page_title="Maternal Health Risk", page_icon="🩺", layout="centered")
st.title("🩺 Maternal Health Risk Prediction")
st.caption("Enter vitals → predict risk (Low / Moderate / High)")

# =========================
# LOAD MODEL + LABELS
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    if not LABELS_PATH.exists():
        st.warning("class_labels.json not found; falling back to ['Low','Moderate','High'].")

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))

    try:
        with open(LABELS_PATH, "r") as f:
            classes = json.load(f)
        # normalize labels (e.g., "medium" -> "Moderate")
        norm = {"low": "Low", "moderate": "Moderate", "medium": "Moderate", "high": "High"}
        classes = [norm.get(str(c).strip().lower(), str(c)) for c in classes]
        if len(classes) == 0:
            classes = CLASS_LABELS_FALLBACK
    except Exception:
        classes = CLASS_LABELS_FALLBACK

    return booster, classes

model, class_labels = load_model()

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Enter Patient Vitals")

age = st.sidebar.number_input("Age (years)", min_value=15, max_value=50, value=28, step=1)
systolic = st.sidebar.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=120, step=1)
diastolic = st.sidebar.number_input("Diastolic BP (mmHg)", min_value=50, max_value=140, value=80, step=1)
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=18.0, value=11.5, step=0.1, format="%.1f")
bmi = st.sidebar.number_input("BMI", min_value=15.0, max_value=50.0, value=24.0, step=0.1, format="%.1f")
prenatal_visits = st.sidebar.number_input("Prenatal Visits", min_value=0, max_value=40, value=4, step=1)
past_comp = st.sidebar.selectbox("Past Complications (0=No, 1=Yes)", options=[0, 1], index=0)
blood_sugar = st.sidebar.number_input("Blood Sugar (mmol/L)", min_value=3.0, max_value=30.0, value=6.0, step=0.1, format="%.1f")
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=50, max_value=180, value=80, step=1)
fetal_move = st.sidebar.number_input("Fetal Movement Score (1-10)", min_value=1, max_value=10, value=7, step=1)
body_temp = st.sidebar.number_input("Body Temperature (°F)", min_value=95.0, max_value=104.0, value=98.0, step=0.1, format="%.1f")
trimester = st.sidebar.selectbox("Trimester", options=["First", "Second", "Third"], index=1)

# =========================
# FEATURE ENGINEERING (same as training)
# =========================
trimester_num = {"First": 1, "Second": 2, "Third": 3}[trimester]
bp_diff = int(systolic - diastolic)
anemia_flag = 1 if hemoglobin < 10.5 else 0
obesity_flag = 1 if bmi >= 30 else 0

row = {
    "Age": age,
    "Systolic_BP": systolic,
    "Diastolic_BP": diastolic,
    "Hemoglobin": hemoglobin,
    "BMI": bmi,
    "Prenatal_Visits": prenatal_visits,
    "Past_Complications": int(past_comp),
    "Blood_Sugar": blood_sugar,
    "Heart_Rate": heart_rate,
    "Fetal_Movement_Score": fetal_move,
    "BodyTemp": body_temp,
    "Trimester_Num": trimester_num,
    "BP_Diff": bp_diff,
    "Anemia_Flag": anemia_flag,
    "Obesity_Flag": obesity_flag,
}

# DataFrame in the precise feature order
X = pd.DataFrame([row], columns=FEATURE_ORDER)

# =========================
# PREDICT
# =========================
if st.button("Predict Risk"):
    try:
        # XGBoost Booster expects DMatrix
        dmat = xgb.DMatrix(X)
        # For multi-class softprob, predict returns [ [p0, p1, p2] ]
        probs = model.predict(dmat)[0]

        # Safety: align length
        if len(probs) != len(class_labels):
            # If mismatch, pad or trim to avoid crashes (rare)
            min_len = min(len(probs), len(class_labels))
            probs = probs[:min_len]
            class_labels = class_labels[:min_len]

        pred_idx = int(np.argmax(probs))
        pred_label = class_labels[pred_idx]
        badge = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(pred_label, "🟣")

        st.subheader(f"{badge} Predicted Risk: **{pred_label}**")
        st.dataframe(
            pd.DataFrame({"Risk Class": class_labels, "Probability": np.round(probs, 3)}),
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("See input features"):
            st.json(row)

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Model: XGBoost Booster (converted from sklearn) • Inference-only (no scikit-learn needed).")
