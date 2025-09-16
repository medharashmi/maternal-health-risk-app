import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# ---- CONFIG ----
MODEL_PATH = Path("xgboost.pkl")
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

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    try:
        classes = list(model.classes_)
    except Exception:
        classes = CLASS_LABELS_FALLBACK
    norm = {"low":"Low","moderate":"Moderate","medium":"Moderate","high":"High"}
    classes = [norm.get(str(c).strip().lower(), str(c)) for c in classes]
    return model, classes

model, class_labels = load_model()

# ---- Inputs (sidebar) ----
st.sidebar.header("Enter Patient Vitals")
age = st.sidebar.number_input("Age (years)", 15, 50, 28)
systolic = st.sidebar.number_input("Systolic BP (mmHg)", 80, 200, 120)
diastolic = st.sidebar.number_input("Diastolic BP (mmHg)", 50, 140, 80)
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", 5.0, 18.0, 11.5, step=0.1)
bmi = st.sidebar.number_input("BMI", 15.0, 45.0, 24.0, step=0.1)
prenatal_visits = st.sidebar.number_input("Prenatal Visits", 0, 20, 4)
past_comp = st.sidebar.selectbox("Past Complications (0=No, 1=Yes)", [0,1], index=0)
blood_sugar = st.sidebar.number_input("Blood Sugar (mmol/L)", 3.0, 25.0, 6.0, step=0.1)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", 50, 160, 80)
fetal_move = st.sidebar.number_input("Fetal Movement Score (1-10)", 1, 10, 7)
body_temp = st.sidebar.number_input("Body Temperature (°F)", 95.0, 104.0, 98.0, step=0.1)
trimester = st.sidebar.selectbox("Trimester", ["First","Second","Third"], index=1)

# Engineered features (same as training pipeline)
trimester_num = {"First":1, "Second":2, "Third":3}[trimester]
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
    "Obesity_Flag": obesity_flag
}
X = pd.DataFrame([row], columns=FEATURE_ORDER)

if st.button("Predict Risk"):
    try:
        probs = model.predict_proba(X)[0]
        # align probs to class_labels (just in case)
        try:
            idx_map = {c:i for i,c in enumerate(model.classes_)}
            probs = np.array([probs[idx_map[c]] for c in class_labels])
        except Exception:
            pass

        pred_idx = int(np.argmax(probs))
        pred_label = class_labels[pred_idx]
        color = {"Low":"🟢","Moderate":"🟡","High":"🔴"}.get(pred_label, "🟣")
        st.subheader(f"{color} Predicted Risk: **{pred_label}**")

        st.dataframe(pd.DataFrame({
            "Risk Class": class_labels,
            "Probability": np.round(probs, 3)
        }), hide_index=True)

        with st.expander("See input features"):
            st.json(row)

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Model: XGBoost • Trained in Phase 3 • SHAP/LIME used offline for validation.")
