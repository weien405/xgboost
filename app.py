
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="AMI预测工具", layout="wide")
st.title("🏥 急性心肌梗死 (AMI) 预后分组预测系统")

# 加载模型
@st.cache_resource
def load_assets():
    # 使用相对路径读取 deploy_files 下的文件
    model = joblib.load("deploy_files/best_xgb_model.pkl")
    le = joblib.load("deploy_files/label_encoder.pkl")
    median = joblib.load("deploy_files/train_median.pkl")
    return model, le, median

model, le, median = load_assets()

st.sidebar.header("患者指标输入")
inputs = {}
# 自动生成24个输入框
cols = st.columns(3)
for i, col_name in enumerate(median.index):
    with cols[i % 3]:
        inputs[col_name] = st.number_input(f"{col_name}", value=float(median[col_name]))

if st.button("🚀 点击进行预测"):
    input_df = pd.DataFrame([inputs])
    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)[0]
    res_label = le.inverse_transform(pred)[0]

    st.success(f"预测结果为: **Group {res_label}**")
    st.bar_chart(pd.DataFrame({"概率": proba}, index=le.classes_))
