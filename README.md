# Maternal Health Risk Predictor with Explainable AI (XAI)

### [🚀 View Live Deployed Application](https://maternal-health-risk-app-cgbssqnsxjrfz4je2n3pbc.streamlit.app/)

## Project Overview
Maternal mortality is a critical global health challenge. This project provides a **Predictive Analytics platform** to identify high-risk pregnancies using machine learning. By integrating **Explainable AI (SHAP & LIME)**, the system moves beyond "black-box" predictions, giving clinicians transparent, data-backed justifications for risk assessments.

## Technical Stack
- **Modeling:** XGBoost (Primary), Random Forest, Logistic Regression.
- **Explainability:** SHAP (Global Importance), LIME (Local Instance Explanations).
- **Deployment:** Streamlit Cloud.
- **Engineering:** Python (Pandas, Scikit-Learn), Feature Engineering (BMI, Anemia Flags, BP Differentials).

## Database Schema & Feature Engineering
The model is trained on an enhanced clinical dataset. Beyond raw vitals, I engineered several features to improve predictive accuracy for maternal complications.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| **Patient_ID** | `VARCHAR` | Unique identifier for the patient record. |
| **Age** | `INTEGER` | Age of the mother (Key risk factor for advanced maternal age). |
| **Trimester** | `VARCHAR` | Current stage of pregnancy (First, Second, Third). |
| **Systolic_BP** | `INTEGER` | Upper value of Blood Pressure (High values indicate Preeclampsia risk). |
| **Diastolic_BP** | `INTEGER` | Lower value of Blood Pressure. |
| **Blood_Sugar** | `FLOAT` | Glucose levels (mmol/L) used to detect Gestational Diabetes. |
| **Hemoglobin** | `FLOAT` | Iron levels used to assess anemia. |
| **BMI** | `FLOAT` | **Engineered:** Body Mass Index to assess obesity-related risks. |
| **BP_Diff** | `INTEGER` | **Engineered:** Difference between Systolic and Diastolic pressure. |
| **Anemia_Flag** | `BOOLEAN` | **Engineered:** Binary indicator based on Hemoglobin thresholds (<11g/dL). |
| **Obesity_Flag**| `BOOLEAN` | **Engineered:** Binary indicator for high-risk BMI (>30). |
| **RiskLevel** | `VARCHAR` | **Target Variable:** Categorized as Low, Mid, or High Risk. |



## Explainable AI (XAI) Implementation
In healthcare, transparency is a requirement, not a feature.

### Global Interpretability (SHAP)
![SHAP Plot]("C:\Users\Medha Rashmi\OneDrive - O. P. Jindal Global University\Desktop\MateranlHealthProject\SHAP_outputs_averagedforallclasses\shap_waterfall_XGBoost_idx2.png")
* **Insight:** The SHAP summary plot confirms that **Blood Glucose** and **Systolic BP** are the primary drivers of "High Risk" classifications, aligning with clinical standards for gestational health.

### Local Interpretability (LIME)
![LIME Plot]("C:\Users\Medha Rashmi\OneDrive - O. P. Jindal Global University\Desktop\MateranlHealthProject\Old_LIME_outputs\lime_explanation_XGBoost_idx2.png")
* **Insight:** For individual patient triage, the LIME plot identifies exactly which vital sign (e.g., a sudden spike in Blood Sugar) triggered the risk alert, even if other vitals remained stable.

## Key Results & Impact
- **Prioritizing Recall:** The XGBoost model was optimized for **Recall**, ensuring that the system minimizes "False Negatives"—crucial in a clinical setting where missing a high-risk case can be fatal.
- **Transparency:** SHAP/LIME integration builds trust with medical professionals by justifying every automated alert.

## Ethical Disclaimer
This tool is a **Supportive AI System** intended for triage and awareness. It is **not a standalone diagnostic engine** and must be used in conjunction with professional clinical judgment.
