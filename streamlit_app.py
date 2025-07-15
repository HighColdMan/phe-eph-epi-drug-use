import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn

# ===================
# Basic Settings
st.set_page_config(page_title="Intraoperative Drug Prediction System", layout="wide")
st.title("🎯 Intelligent Intraoperative Drug Prediction & Model Visualization")
st.markdown("---")

RESULTS_DIR = "results"
APPROACH_DEFAULT = 'Open'

# Feature order saved during training
feature_order = ['age', 'sex', 'bmi', 'asa', 'preop_htn', 'preop_dm', 'preop_arrhythmia',
                 'preop_pft', 'preop_hb', 'preop_plt', 'preop_pt', 'preop_aptt',
                 'preop_na', 'preop_k', 'preop_glucose', 'preop_alb', 'preop_got',
                 'preop_gpt', 'preop_bun', 'preop_cr', 'optype_Biliary/Pancreas',
                 'optype_Breast', 'optype_Colorectal', 'optype_Hepatic',
                 'optype_Major resection', 'optype_Minor resection', 'optype_Others',
                 'optype_Stomach', 'optype_Thyroid', 'optype_Transplantation',
                 'optype_Vascular', 'approach_Open', 'approach_Robotic',
                 'approach_Videoscopic', 'age_missing', 'bmi_missing', 'asa_missing',
                 'preop_htn_missing', 'preop_dm_missing', 'preop_arrhythmia_missing',
                 'preop_pft_missing', 'preop_hb_missing', 'preop_plt_missing',
                 'preop_pt_missing', 'preop_aptt_missing', 'preop_na_missing',
                 'preop_k_missing', 'preop_glucose_missing', 'preop_alb_missing',
                 'preop_got_missing', 'preop_gpt_missing', 'preop_bun_missing',
                 'preop_cr_missing']

# MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# Approach Encoding
def handle_approach(selected_approach):
    if selected_approach in ['Open', 'Videoscopic', 'Robotic']:
        return {
            'approach_Open': 1 if selected_approach == 'Open' else 0,
            'approach_Videoscopic': 1 if selected_approach == 'Videoscopic' else 0,
            'approach_Robotic': 1 if selected_approach == 'Robotic' else 0,
        }
    else:
        missing_fields.append("approach")
        return {
            'approach_Open': 1 if APPROACH_DEFAULT == 'Open' else 0,
            'approach_Videoscopic': 1 if APPROACH_DEFAULT == 'Videoscopic' else 0,
            'approach_Robotic': 1 if APPROACH_DEFAULT == 'Robotic' else 0,
        }

# Default Values for Missing Inputs
DEFAULT_VALUES = {
    'age': 59,
    'bmi': 23.1,
    'asa': 2,
    'preop_htn': 0,
    'preop_dm': 0,
    'preop_arrhythmia': 0,
    'preop_pft': 0,
    'preop_hb': 13,
    'preop_plt': 236,
    'preop_pt': 101,
    'preop_aptt': 32.1,
    'preop_na': 140,
    'preop_k': 4.2,
    'preop_glucose': 103,
    'preop_alb': 4.2,
    'preop_got': 21,
    'preop_gpt': 18,
    'preop_bun': 14,
    'preop_cr': 0.78,
}

# ===================
# Section 1: Model Performance Visualization
st.header("📊 Model Performance Evaluation")

drug_list = ['intraop_eph', 'intraop_phe', 'intraop_epi']
selected_drug = st.selectbox("Select Drug Category", drug_list)

metrics_file = os.path.join(RESULTS_DIR, selected_drug, "metrics.csv")
roc_curve_path = os.path.join(RESULTS_DIR, selected_drug, "roc_curves.png")
feature_heatmap_path = os.path.join(RESULTS_DIR, selected_drug, "feature_pbc_heatmap.png")

if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
    st.subheader("Evaluation Metrics (All Models)")
    st.dataframe(metrics_df.style.format(precision=3))
else:
    st.warning("Metrics file not found. Please train the model first.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curves")
    if os.path.exists(roc_curve_path):
        st.image(roc_curve_path)
    else:
        st.warning("ROC curve image not found. Please train the model first.")

with col2:
    st.subheader("Feature-Label Correlation")
    if os.path.exists(feature_heatmap_path):
        st.image(feature_heatmap_path)
    else:
        st.warning("Feature heatmap not found. Please train the model first.")

# ===================
# Section 2: Single Patient Drug Prediction
st.header("🧑‍⚕️ Intraoperative Drug Prediction for Single Patient")
st.markdown("Please input patient data in the sidebar for real-time prediction:")

st.sidebar.header("🩺 Patient Data Input")

with st.sidebar.form("patient_data_form"):
    sex = st.selectbox("Sex", ["M", "F"])
    optype = st.selectbox("Surgical Type", ["Colorectal", "Stomach", "Biliary/Pancreas",
                                            "Vascular", "Major resection", "Breast",
                                            "Minor resection", "Transplantation",
                                            "Hepatic", "Thyroid", "Others"])
    age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="Optional")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=None, placeholder="Optional")
    asa = st.number_input("ASA Grade", min_value=1, max_value=5, value=None, placeholder="Optional")
    approach = st.selectbox("Surgical Approach", ["Open", "Videoscopic", "Robotic"])
    preop_htn = st.selectbox("Preoperative Hypertension", ["Missing", "Y", "N"])
    preop_dm = st.selectbox("Preoperative Diabetes", ["Missing", "Y", "N"])
    preop_arrhythmia = st.selectbox("Preoperative Arrhythmia", ["Missing", "Y", "N"])
    preop_pft = st.selectbox("Preoperative Pulmonary Function", ["Missing", "normal", "abnormal"])

    preop_hb = st.number_input("Preoperative Hemoglobin (g/L)", value=None, placeholder="Optional")
    preop_plt = st.number_input("Preoperative Platelet Count (K/μL)", value=None, placeholder="Optional")
    preop_pt = st.number_input("Preoperative PT (s)", value=None, placeholder="Optional")
    preop_aptt = st.number_input("Preoperative APTT (s)", value=None, placeholder="Optional")
    preop_na = st.number_input("Preoperative Sodium (mmol/L)", value=None, placeholder="Optional")
    preop_k = st.number_input("Preoperative Potassium (mmol/L)", value=None, placeholder="Optional")
    preop_glucose = st.number_input("Preoperative Glucose (mmol/L)", value=None, placeholder="Optional")
    preop_alb = st.number_input("Preoperative Albumin (g/L)", value=None, placeholder="Optional")
    preop_got = st.number_input("Preoperative GOT (U/L)", value=None, placeholder="Optional")
    preop_gpt = st.number_input("Preoperative GPT (U/L)", value=None, placeholder="Optional")
    preop_bun = st.number_input("Preoperative BUN (mmol/L)", value=None, placeholder="Optional")
    preop_cr = st.number_input("Preoperative Creatinine (μmol/L)", value=None, placeholder="Optional")

    submit_btn = st.form_submit_button("Predict")

# ===================
# After Submission
if submit_btn:
    st.subheader(f"💊 [{selected_drug}] Prediction Result for Single Patient")

    missing_fields = []

    def fill_value(val, key):
        if val is None or val == "Missing":
            missing_fields.append(key)
            return DEFAULT_VALUES.get(key, 0)
        return val

    approach_dict = handle_approach(approach)

    input_dict = {
        'sex': 0 if sex == "M" else 1,
        'optype_Colorectal': 1 if optype == 'Colorectal' else 0,
        'optype_Stomach': 1 if optype == 'Stomach' else 0,
        'optype_Biliary/Pancreas': 1 if optype == 'Biliary/Pancreas' else 0,
        'optype_Vascular': 1 if optype == 'Vascular' else 0,
        'optype_Major resection': 1 if optype == 'Major resection' else 0,
        'optype_Breast': 1 if optype == 'Breast' else 0,
        'optype_Minor resection': 1 if optype == 'Minor resection' else 0,
        'optype_Transplantation': 1 if optype == 'Transplantation' else 0,
        'optype_Hepatic': 1 if optype == 'Hepatic' else 0,
        'optype_Thyroid': 1 if optype == 'Thyroid' else 0,
        'optype_Others': 1 if optype == 'Others' else 0,
        'age': fill_value(age, "age"),
        'bmi': fill_value(bmi, "bmi"),
        'asa': fill_value(asa, "asa"),
        'preop_htn': 1 if preop_htn == "Y" else (0 if preop_htn == "N" else fill_value(None, "preop_htn")),
        'preop_dm': 1 if preop_dm == "Y" else (0 if preop_dm == "N" else fill_value(None, "preop_dm")),
        'preop_arrhythmia': 0 if preop_arrhythmia == "N" else (1 if preop_arrhythmia == "Y" else fill_value(None, "preop_arrhythmia")),
        'preop_pft': 0 if preop_pft == "normal" else (1 if preop_pft == "abnormal" else fill_value(None, "preop_pft")),
        'preop_hb': fill_value(preop_hb, "preop_hb"),
        'preop_plt': fill_value(preop_plt, "preop_plt"),
        'preop_pt': fill_value(preop_pt, "preop_pt"),
        'preop_aptt': fill_value(preop_aptt, "preop_aptt"),
        'preop_na': fill_value(preop_na, "preop_na"),
        'preop_k': fill_value(preop_k, "preop_k"),
        'preop_glucose': fill_value(preop_glucose, "preop_glucose"),
        'preop_alb': fill_value(preop_alb, "preop_alb"),
        'preop_got': fill_value(preop_got, "preop_got"),
        'preop_gpt': fill_value(preop_gpt, "preop_gpt"),
        'preop_bun': fill_value(preop_bun, "preop_bun"),
        'preop_cr': fill_value(preop_cr, "preop_cr"),
    }

    input_dict.update(approach_dict)

    for col in [col for col in feature_order if col.endswith('_missing')]:
        input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])
    input_array = input_df[feature_order].values

    # Load models and scaler
    mlp_model = MLP(53).to('cpu')
    mlp_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, selected_drug, "MLP_weights.pth"), map_location='cpu'))
    mlp_model.eval()

    lgbm_model = joblib.load(os.path.join(RESULTS_DIR, selected_drug, "LGBM_weights.joblib"))
    rf_model = joblib.load(os.path.join(RESULTS_DIR, selected_drug, "RandomForest_weights.joblib"))
    stacking_model = joblib.load(os.path.join(RESULTS_DIR, selected_drug, "Stacking_weights.joblib"))
    scaler = joblib.load(os.path.join(RESULTS_DIR, selected_drug, "scaler.joblib"))

    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        mlp_proba = torch.sigmoid(mlp_model(input_tensor)).cpu().numpy()

    lgbm_proba = lgbm_model.predict_proba(input_scaled)[:, 1][0]
    rf_proba = rf_model.predict_proba(input_scaled)[:, 1][0]

    meta_feature = np.array([[float(mlp_proba), float(lgbm_proba), float(rf_proba)]])
    pred_proba = stacking_model.predict_proba(meta_feature)[:, 1][0]

    prediction = "Drug Required" if pred_proba >= 0.5 else "Drug Not Required"

    st.success(f"Prediction: **{prediction}**")
    st.metric("Predicted Probability", f"{pred_proba:.2%}")
    if missing_fields:
        st.warning(f"{len(missing_fields)} missing values filled with defaults: {', '.join(missing_fields)}. Please interpret carefully.")
    st.progress(min(max(pred_proba, 0.0), 1.0))
