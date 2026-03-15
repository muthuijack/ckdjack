import streamlit as st
import joblib
from keras.models import load_model
import numpy as np
from fpdf import FPDF

# 1. Configuration & Translation
st.set_page_config(page_title="CKD Diagnostic System", page_icon="🏥", layout="wide")

translations = {
    "English": {
        "title": "🏥 CKD Comprehensive Diagnostic Tool",
        "tab1": "Clinical Measurements",
        "tab2": "History & Lifestyle",
        "btn": "Generate Diagnostic Report",
        "risk_label": "Risk Probability",
        "download": "📥 Download Official Patient Report (PDF)",
        "error_msg": "⚠️ Please fill in all Primary Clinical Data fields."
    },
    "Español": {
        "title": "🏥 Herramienta de Diagnóstico Integral de ERC",
        "tab1": "Mediciones Clínicas",
        "tab2": "Historial y Estilo de Vida",
        "btn": "Generar Informe de Diagnóstico",
        "risk_label": "Probabilidad de Riesgo",
        "download": "📥 Descargar Informe Oficial del Paciente (PDF)",
        "error_msg": "⚠️ Por favor, complete todos los campos de Datos Clínicos Principales."
    }
}

lang = st.sidebar.selectbox("Language / Idioma", ["English", "Español"])
t = translations[lang]

# 2. Load Models
# Change this:
# model = tf.keras.models.load_model("C:\\Users\\HP\\Downloads\\ckd_prediction_model.keras")

# To this (Relative Path):
model = tf.keras.models.load_model("models/ckd_prediction_model.keras")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load('models/num_imputer.pkl')

feature_map = {
    'serum_creatinine': 0, 'gfr': 1, 'bun': 2, 'serum_calcium': 3, 'ana': 4, 
    'c3_c4': 5, 'hematuria': 6, 'oxalate_levels': 7, 'urine_ph': 8, 'blood_pressure': 9, 
    'water_intake': 10, 'months': 11, 'cluster': 12, 'physical_activity_rarely': 13, 
    'physical_activity_weekly': 14, 'diet_high protein': 15, 'diet_low salt': 16, 
    'smoking_yes': 17, 'alcohol_never': 18, 'alcohol_occasionally': 19, 
    'painkiller_usage_yes': 20, 'family_history_yes': 21, 'weight_changes_loss': 22, 
    'weight_changes_stable': 23, 'stress_level_low': 24, 'stress_level_moderate': 25
}

binary_features = ['ana', 'hematuria', 'smoking_yes', 'physical_activity_rarely', 'physical_activity_weekly', 'diet_high protein', 'diet_low salt', 'alcohol_never', 'alcohol_occasionally', 'painkiller_usage_yes', 'family_history_yes', 'weight_changes_loss', 'weight_changes_stable', 'stress_level_low', 'stress_level_moderate']

# 3. Main Interface
st.title(t["title"])
user_inputs = {}

tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

with tab1:
    st.info("Please enter the primary laboratory results.")
    for feat in ['serum_creatinine', 'gfr', 'bun', 'serum_calcium', 'urine_ph', 'blood_pressure']:
        user_inputs[feat] = st.number_input(feat.replace('_', ' ').title(), value=0.0)

with tab2:
    st.info("Please enter the patient's medical and lifestyle history.")
    for feat in [f for f in feature_map if f not in ['serum_creatinine', 'gfr', 'bun', 'serum_calcium', 'urine_ph', 'blood_pressure']]:
        label = feat.replace('_', ' ').title()
        if feat in binary_features:
            user_inputs[feat] = 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0
        else:
            user_inputs[feat] = st.number_input(label, value=0.0)

# 4. Processing
if st.button(t["btn"]):
    if any(user_inputs[f] <= 0 for f in ['serum_creatinine', 'gfr', 'bun', 'blood_pressure']):
        st.error(t["error_msg"])
    else:
        input_data = imputer.statistics_.reshape(1, -1)
        for feat, value in user_inputs.items():
            if value != 0: input_data[0, feature_map[feat]] = value
                
        scaled_data = scaler.transform(imputer.transform(input_data))
        risk = model.predict(scaled_data)[0][0]
        
        st.markdown("---")
        st.metric(t["risk_label"], f"{risk:.2%}")
        
        if risk > 0.5:
            stage, col = "High Risk (Stage 3+)", "error"
        elif risk > 0.2:
            stage, col = "Early CKD (Stage 1-2)", "warning"
        else:
            stage, col = "Low Risk (Normal)", "success"
            
        getattr(st, col)(stage)

        # PDF Generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Official CKD Diagnostic Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Diagnosis: {stage}", ln=True)
        pdf.cell(200, 10, txt=f"Calculated Risk: {risk:.2%}", ln=True)
        pdf_bytes = pdf.output(dest='S').encode('latin-1')

        st.download_button(t["download"], data=pdf_bytes, file_name="CKD_Report.pdf")


