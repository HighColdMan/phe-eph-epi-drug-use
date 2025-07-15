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
# 基础设置
st.set_page_config(page_title="术中用药预测系统", layout="wide")
st.title("🎯 术中用药智能预测与模型可视化")
st.markdown("---")

RESULTS_DIR = "results"

APPROACH_DEFAULT = 'Open'  

# 保存的特征顺序
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

# ==== 简单 MLP ====
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1) 


# approach专用处理
def handle_approach(selected_approach):
    if selected_approach in ['Open', 'Videoscopic', 'Robotic']:
        return {
            'approach_Open': 1 if selected_approach == 'Open' else 0,
            'approach_Videoscopic': 1 if selected_approach == 'Videoscopic' else 0,
            'approach_Robotic': 1 if selected_approach == 'Robotic' else 0,
        }
    else:
        missing_fields.append("approach")
        # 使用统计的众数作为默认
        return {
            'approach_Open': 1 if APPROACH_DEFAULT == 'Open' else 0,
            'approach_Videoscopic': 1 if APPROACH_DEFAULT == 'Videoscopic' else 0,
            'approach_Robotic': 1 if APPROACH_DEFAULT == 'Robotic' else 0,
        }
    
# 用于缺失特征的默认均值（示例，真实应来自训练集统计）
DEFAULT_VALUES = {
    'age': 59,
    'bmi': 23.1,
    'asa': 2,
    'preop_htn': 0,# 'N',
    'preop_dm': 0, # 'N',
    'preop_arrhythmia': 0, # 'N',
    'preop_pft': 0, # 'Normal',
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
# 功能 1：展示模型性能
st.header("📊 模型性能指标与评估结果")


drug_list = ['intraop_eph', 'intraop_phe', 'intraop_epi']
selected_drug = st.selectbox("请选择药物类别进行查看", drug_list)

metrics_file = os.path.join(RESULTS_DIR, selected_drug, "metrics.csv")
roc_curve_path = os.path.join(RESULTS_DIR, selected_drug, "roc_curves.png")
feature_heatmap_path = os.path.join(RESULTS_DIR, selected_drug, "feature_pbc_heatmap.png")

if os.path.exists(metrics_file):
    metrics_df = pd.read_csv(metrics_file)
    st.subheader("各模型评估指标（最新）")
    st.dataframe(metrics_df.style.format(precision=3))
else:
    st.warning("未找到模型指标文件，请先训练模型。")

col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC 曲线")
    if os.path.exists(roc_curve_path):
        st.image(roc_curve_path)
    else:
        st.warning("未找到 ROC 曲线图，请先训练模型。")

with col2:
    st.subheader("特征与标签的相关性")
    if os.path.exists(feature_heatmap_path):
        st.image(feature_heatmap_path)
    else:
        st.warning("未找到特征热力图，请先训练模型。")

# ===================
# 功能 2：单病人数据输入与预测
st.header("🧑‍⚕️ 单位病人术中用药预测")

st.markdown("请在左侧输入病人信息进行模型实时预测：")

# 侧边栏输入
st.sidebar.header("🩺 输入病人信息（仅必填信息必需）")
with st.sidebar.form("病人数据输入"):
    sex = st.selectbox("性别", ["M", "F"])
    optype = st.selectbox("手术方式 (optype)*", ["Colorectal", "Stomach", 
                                             "Biliary/Pancreas", 'Vascular', 'Major resection', 
                                             'Breast', 'Minor resection', 'Transplantation', 
                                             'Hepatic', 'Thyroid', 'Others'])
        
    age = st.number_input("年龄", min_value=0, max_value=120, value=None, placeholder="可选")
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=None, placeholder="可选")
    asa = st.number_input("ASA 分级", min_value=1, max_value=5, value=None, placeholder="可选")

    # 手术方式
    approach = st.selectbox("手术方式", ["Open", "Videoscopic", "Robotic"])

    # 术前合并症
    preop_htn = st.selectbox("术前高血压", ["缺失", "Y", "N"])
    preop_dm = st.selectbox("术前糖尿病", ["缺失", "Y", "N"])
    preop_arrhythmia = st.selectbox("术前心律失常", ["缺失", "Y", "N"])
    preop_pft = st.selectbox("术前肺功能", ["缺失", "normal", "abnormal"])

    # 血液指标
    preop_hb = st.number_input("术前血红蛋白 (g/L)", value=None, placeholder="可选")
    preop_plt = st.number_input("术前血小板计数 (K/μL)", value=None, placeholder="可选")
    preop_pt = st.number_input("术前凝血酶原时间 PT (秒)", value=None, placeholder="可选")
    preop_aptt = st.number_input("术前活化部分凝血活酶时间 APTT (秒)", value=None, placeholder="可选")

    # 电解质与血糖
    preop_na = st.number_input("术前钠浓度 (mmol/L)", value=None, placeholder="可选")
    preop_k = st.number_input("术前钾浓度 (mmol/L)", value=None, placeholder="可选")
    preop_glucose = st.number_input("术前血糖 (mmol/L)", value=None, placeholder="可选")

    # 肝功能 & 营养状态
    preop_alb = st.number_input("术前白蛋白 (g/L)", value=None, placeholder="可选")
    preop_got = st.number_input("术前谷草转氨酶 GOT (U/L)", value=None, placeholder="可选")
    preop_gpt = st.number_input("术前谷丙转氨酶 GPT (U/L)", value=None, placeholder="可选")

    # 肾功能
    preop_bun = st.number_input("术前尿素氮 BUN (mmol/L)", value=None, placeholder="可选")
    preop_cr = st.number_input("术前肌酐 Creatinine (μmol/L)", value=None, placeholder="可选")

    submit_btn = st.form_submit_button("提交预测")

if submit_btn:
    st.subheader(f"💊 [{selected_drug}] 单病人预测结果")

    missing_fields = []
    # 构造输入
    def fill_value(val, key):
        if val is None or val == "缺失":
            missing_fields.append(key)
            return DEFAULT_VALUES.get(key, 0)
        return val
    
    # approach编码
    approach_dict = handle_approach(approach)        
    # === 构造输入特征 ===
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

        # 数值与类别型（按optional列表完整覆盖）
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
    # 加入approach独热
    input_dict.update(approach_dict)

    # == 新增：缺失指示器特征 ==
    missing_indicator_columns = [
        'age_missing', 'bmi_missing', 'asa_missing', 'preop_htn_missing',
        'preop_dm_missing', 'preop_arrhythmia_missing', 'preop_pft_missing',
        'preop_hb_missing', 'preop_plt_missing', 'preop_pt_missing',
        'preop_aptt_missing', 'preop_na_missing', 'preop_k_missing',
        'preop_glucose_missing', 'preop_alb_missing', 'preop_got_missing',
        'preop_gpt_missing', 'preop_bun_missing', 'preop_cr_missing'
    ]

    # 默认都填 0（假设输入时未专门处理缺失指示器）
    for col in missing_indicator_columns:
        input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])

    # == 加载模型 ==
    stacking_model_path = os.path.join(RESULTS_DIR, selected_drug, "Stacking_weights.joblib")

    mlp_model = MLP(53).to('cpu')

    # 加载权重
    mlp_model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, selected_drug, "MLP_weights.pth"), map_location='cpu'))
    mlp_model.eval()
    # 加载 LightGBM 模型
    lgbm_model = joblib.load(os.path.join(RESULTS_DIR, selected_drug, "LGBM_weights.joblib"))

    # 加载 RandomForest 模型
    rf_model = joblib.load(os.path.join(RESULTS_DIR, selected_drug, "RandomForest_weights.joblib"))

    # 加载训练时保存的scaler
    scaler_path = os.path.join(RESULTS_DIR, selected_drug, "scaler.joblib")
    scaler = joblib.load(scaler_path)

    if not os.path.exists(stacking_model_path):
        st.error("找不到Stacking模型，请先训练模型。")
    else:
        stacking_model = joblib.load(stacking_model_path)

        # 保证输入顺序与训练时一致
        trained_features = feature_order
        input_array = input_df[trained_features].values
        # === 归一化输入 ===
        input_array_scaled = scaler.transform(input_array)  # ⬅️ 核心        
        # MLP 输入需要转换为 torch 张量
        input_tensor = torch.tensor(input_array_scaled, dtype=torch.float32).to('cpu')
        # === Step 3. 各个底层模型输出概率 ===

        # MLP 预测概率
        with torch.no_grad():
            mlp_proba = torch.sigmoid(mlp_model(input_tensor)).cpu().numpy()# 取出单个病人的概率

        # LightGBM 预测概率
        lgbm_proba = lgbm_model.predict_proba(input_array_scaled)[:, 1][0] # 取出概率

        # RandomForest 预测概率
        rf_proba = rf_model.predict_proba(input_array_scaled)[:, 1][0]
        meta_feature = np.array([[float(mlp_proba), float(lgbm_proba), float(rf_proba)]])  # shape: (1, 3)

        # Stacking 元模型预测
        pred_proba = stacking_model.predict_proba(meta_feature)[:, 1][0] 

        prediction = "需要用药" if pred_proba >= 0.5 else "不需要用药"

        # 显示结果
        st.success(f"预测结果：**{prediction}**")
        st.metric("预测概率", f"{pred_proba:.2%}")
        if missing_fields:
            st.warning(f"⚠️ 本次预测中 {len(missing_fields)} 项指标缺失：{', '.join(missing_fields)}。\n已使用默认值填充，结果仅供参考。")
        

        # 进度条展示
        st.progress(min(max(pred_proba, 0.0), 1.0))


# ===================
# 底部
st.markdown("---")
st.caption("Powered by Streamlit · 术中用药智能预测系统")
