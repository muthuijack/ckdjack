import gradio as gr
import numpy as np
import joblib
from keras.models import load_model

# Load models
model = load_model("models/ckd_prediction_model.keras")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/num_imputer.pkl")

# Feature order (MUST match training)
feature_order = [
    'serum_creatinine','gfr','bun','serum_calcium','ana','c3_c4',
    'hematuria','oxalate_levels','urine_ph','blood_pressure',
    'water_intake','months','cluster','physical_activity_rarely',
    'physical_activity_weekly','diet_high_protein','diet_low_salt',
    'smoking_yes','alcohol_never','alcohol_occasionally',
    'painkiller_usage_yes','family_history_yes',
    'weight_changes_loss','weight_changes_stable',
    'stress_level_low','stress_level_moderate'
]

def predict_ckd(
    serum_creatinine, gfr, bun, serum_calcium,
    ana, c3_c4, hematuria, oxalate_levels,
    urine_ph, blood_pressure, water_intake,
    months, cluster,
    physical_activity_rarely, physical_activity_weekly,
    diet_high_protein, diet_low_salt,
    smoking_yes, alcohol_never, alcohol_occasionally,
    painkiller_usage_yes, family_history_yes,
    weight_changes_loss, weight_changes_stable,
    stress_level_low, stress_level_moderate
):

    data = np.array([[
        serum_creatinine,gfr,bun,serum_calcium,
        ana,c3_c4,hematuria,oxalate_levels,
        urine_ph,blood_pressure,water_intake,
        months,cluster,physical_activity_rarely,
        physical_activity_weekly,diet_high_protein,
        diet_low_salt,smoking_yes,alcohol_never,
        alcohol_occasionally,painkiller_usage_yes,
        family_history_yes,weight_changes_loss,
        weight_changes_stable,stress_level_low,
        stress_level_moderate
    ]])

    # preprocess
    data = imputer.transform(data)
    data = scaler.transform(data)

    risk = model.predict(data)[0][0]

    if risk > 0.5:
        stage = "⚠️ High Risk CKD (Stage 3+)"
    elif risk > 0.2:
        stage = "⚠️ Early CKD (Stage 1-2)"
    else:
        stage = "✅ Low Risk (Normal)"

    return f"Risk Probability: {risk:.2%}\nDiagnosis: {stage}"


demo = gr.Interface(

    fn=predict_ckd,

    inputs=[

        gr.Number(label="Serum Creatinine"),
        gr.Number(label="GFR"),
        gr.Number(label="Blood Urea Nitrogen (BUN)"),
        gr.Number(label="Serum Calcium"),

        gr.Dropdown([0,1], label="ANA"),
        gr.Number(label="C3/C4"),
        gr.Dropdown([0,1], label="Hematuria"),
        gr.Number(label="Oxalate Levels"),

        gr.Number(label="Urine PH"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Daily Water Intake"),
        gr.Number(label="Months Since Symptoms"),
        gr.Number(label="Cluster Group"),

        gr.Dropdown([0,1], label="Physical Activity Rarely"),
        gr.Dropdown([0,1], label="Physical Activity Weekly"),

        gr.Dropdown([0,1], label="High Protein Diet"),
        gr.Dropdown([0,1], label="Low Salt Diet"),

        gr.Dropdown([0,1], label="Smoking"),

        gr.Dropdown([0,1], label="Alcohol Never"),
        gr.Dropdown([0,1], label="Alcohol Occasionally"),

        gr.Dropdown([0,1], label="Painkiller Usage"),

        gr.Dropdown([0,1], label="Family History CKD"),

        gr.Dropdown([0,1], label="Weight Loss"),
        gr.Dropdown([0,1], label="Weight Stable"),

        gr.Dropdown([0,1], label="Low Stress Level"),
        gr.Dropdown([0,1], label="Moderate Stress Level")

    ],

    outputs="text",

    title="🏥 CKD AI Diagnostic System",

    description="AI-powered Chronic Kidney Disease risk detection using 26 clinical and lifestyle features."
)

demo.launch()
