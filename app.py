import numpy as np
import pandas as pd
import streamlit as st

from ml_model import MachineLearningModels

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="卧室空气质量 CO₂ 预测",
    page_icon="🏠",
    layout="centered",
)

# ── Dataset medians used as defaults for non-input features ──────────────────
DEFAULTS = {
    "voc": 108.6,
    "pm25": 3.6,
    "pm10": 4.6,
    "co2_median": 667.8,
    "co2_rolling_std": 58.6,
    "day_of_week": 3,
}

FEATURE_ORDER = [
    "temp", "humid", "voc", "pm25", "pm10",
    "hour", "day_of_week",
    "hour_sin", "hour_cos",
    "co2_lag_1", "co2_lag_2", "co2_lag_3",
    "co2_rolling_mean", "co2_rolling_std",
    "temp_humid_interaction",
]


# ── Model loader (cached so it only trains once per session) ─────────────────
@st.cache_resource(show_spinner="正在加载并训练模型，请稍候…")
def load_model():
    ml = MachineLearningModels()
    ml.load_and_preprocess()
    ml.prepare_model_data()
    ml.train_random_forest_model()
    return ml


# ── CO₂ level helper ─────────────────────────────────────────────────────────
def co2_status(ppm: float):
    if ppm < 600:
        return "优秀 ✅", "green"
    if ppm < 800:
        return "良好 🟢", "green"
    if ppm < 1000:
        return "一般 🟡", "orange"
    if ppm < 1500:
        return "较差 🔴", "red"
    return "很差 ⛔", "red"


# ── Build feature row from 3 user inputs + dataset defaults ──────────────────
def build_features(temp: float, humid: float, hour: int) -> pd.DataFrame:
    c = DEFAULTS["co2_median"]
    row = {
        "temp": temp,
        "humid": humid,
        "voc": DEFAULTS["voc"],
        "pm25": DEFAULTS["pm25"],
        "pm10": DEFAULTS["pm10"],
        "hour": hour,
        "day_of_week": DEFAULTS["day_of_week"],
        "hour_sin": np.sin(hour * (2 * np.pi / 24)),
        "hour_cos": np.cos(hour * (2 * np.pi / 24)),
        "co2_lag_1": c,
        "co2_lag_2": c,
        "co2_lag_3": c,
        "co2_rolling_mean": c,
        "co2_rolling_std": DEFAULTS["co2_rolling_std"],
        "temp_humid_interaction": temp * humid,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏠 卧室空气质量 CO₂ 预测")
st.caption("基于 Random Forest 模型，输入当前环境参数，预测室内 CO₂ 浓度。")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    temp = st.slider(
        "🌡️ 温度 (°C)",
        min_value=20.0,
        max_value=27.0,
        value=22.7,
        step=0.1,
        help="卧室实测温度，范围 20–27 °C",
    )

with col2:
    humid = st.slider(
        "💧 湿度 (%)",
        min_value=40.0,
        max_value=75.0,
        value=53.8,
        step=0.5,
        help="相对湿度，范围 40–75 %",
    )

with col3:
    hour = st.slider(
        "🕐 时间 (小时)",
        min_value=0,
        max_value=23,
        value=12,
        step=1,
        format="%d:00",
        help="当前时刻（24 小时制）",
    )

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
ml = load_model()

input_df = build_features(temp, humid, hour)
input_scaled = ml.scaler.transform(input_df)
predicted_co2 = float(ml.model.predict(input_scaled)[0])

status_label, status_color = co2_status(predicted_co2)

# Result display
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    st.metric(
        label="预测 CO₂ 浓度",
        value=f"{predicted_co2:.0f} ppm",
        delta=f"{predicted_co2 - 650:.0f} ppm vs 基准",
        delta_color="inverse",
    )

with res_col2:
    st.markdown(f"**空气质量评级**")
    st.markdown(
        f"<span style='font-size:1.6rem; color:{status_color}'>{status_label}</span>",
        unsafe_allow_html=True,
    )

# CO₂ reference table
with st.expander("CO₂ 浓度参考标准"):
    ref = pd.DataFrame({
        "范围 (ppm)": ["< 600", "600 – 800", "800 – 1000", "1000 – 1500", "> 1500"],
        "评级": ["优秀", "良好", "一般", "较差", "很差"],
        "说明": [
            "空气非常清新",
            "空气质量良好",
            "略有闷感，建议通风",
            "明显不适，需立即通风",
            "严重超标，有害健康",
        ],
    })
    st.table(ref)

st.divider()
st.caption("模型训练数据来源：卧室传感器（Bedroom.csv）｜预测结果仅供参考")
