import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit_lottie import st_lottie
import json
from datetime import datetime

# ========== MUST BE FIRST ==========
st.set_page_config(page_title="Air Quality App", layout="wide")

# ========== Functions ==========
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def set_background():
    st.markdown("""
        <style>
            .main {
                background-color: #f5f7fa;
            }
            .sidebar .sidebar-content {
                background-color: #e3f2fd;
            }
            .stButton>button {
                background-color: #4caf50;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

set_background()

# ========== Load Data and Models ==========
df = pd.read_csv("merged_data.csv")
model_lr = joblib.load("lr_model.pkl")
model_knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# ========== Lottie Animations ==========
lottie_home = load_lottie("animation_home.json")
lottie_eda = load_lottie("animation_eda.json")
lottie_predict = load_lottie("animation_predict.json")

# Add datetime column
if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# ========== Sidebar ==========
st.sidebar.title("🌍 Air Quality App")
menu = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Data Overview", "📈 EDA", "🤖 Predict"])
st.sidebar.markdown("---")

# ========== Home ==========
if menu == "🏠 Home":
    st.title("Air Quality Prediction Dashboard")
    st_lottie(lottie_home, height=300)
    st.markdown("""
    Welcome to the **Air Quality Prediction App**! This tool predicts **PM2.5** concentration levels based on various environmental features.

    ### 📌 About the Stations:
    - **Aotizhongxin**: Urban center, often experiences moderate to high pollution.
    - **Changping**: Suburban area with varying pollution based on wind and industrial activity.
    - **Dongsi**: Residential + commercial mix, sees peak traffic-related pollution.
    - **Guanyuan**: Near administrative zones, generally lower emission levels.

    ### 🚀 App Features:
    - Explore air quality data.
    - Run Linear Regression or KNN predictions.
    - Understand trends via interactive EDA.
    """)
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ========== Data Overview ==========
elif menu == "📊 Data Overview":
    st.title("📊 Dataset Overview")
    st.dataframe(df.head(10))
    st.markdown(f"**Shape:** `{df.shape}`")
    st.markdown("### Missing Values")
    st.write(df.isnull().sum())
    st.markdown("### Data Types")
    st.write(df.dtypes)

# ========== EDA ==========
elif menu == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    st_lottie(lottie_eda, height=200)

    plot_option = st.selectbox("Choose a plot", [
        "PM2.5 Distribution", "Correlation Heatmap", "PM2.5 vs Temperature",
        "Boxplot of PM2.5 by Month", "Trend Over Time"])

    if plot_option == "PM2.5 Distribution":
        fig, ax = plt.subplots()
        sns.histplot(df['PM2.5'].dropna(), kde=True, bins=40, ax=ax)
        st.pyplot(fig)

    elif plot_option == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif plot_option == "PM2.5 vs Temperature":
        fig, ax = plt.subplots()
        sns.scatterplot(x='TEMP', y='PM2.5', data=df, alpha=0.5, ax=ax)
        ax.set_title("PM2.5 vs TEMP")
        st.pyplot(fig)

    elif plot_option == "Boxplot of PM2.5 by Month":
        fig, ax = plt.subplots()
        sns.boxplot(x='month', y='PM2.5', data=df, ax=ax)
        st.pyplot(fig)

    elif plot_option == "Trend Over Time":
        fig, ax = plt.subplots(figsize=(10, 4))
        df_sorted = df.sort_values('datetime')
        ax.plot(df_sorted['datetime'], df_sorted['PM2.5'], alpha=0.5)
        ax.set_title("PM2.5 Trend Over Time")
        st.pyplot(fig)

# ========== Prediction ==========
elif menu == "🤖 Predict":
    st.title("🤖 PM2.5 Prediction")
    st_lottie(lottie_predict, height=200)

    st.subheader("Select Model")
    model_choice = st.radio("Choose model", ["Linear Regression", "KNN"])

    # Match the training feature order
    features = ['PM10', 'SO2', 'NO2', 'CO', 'O3',
                'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
                'year', 'month', 'day', 'hour']

    user_input = {}
    cols = st.columns(2)
    for i, feat in enumerate(features):
        with cols[i % 2]:
            default = 2020 if feat == 'year' else 1 if feat in ['month', 'day', 'hour'] else 50.0
            step = 1 if feat in ['year', 'month', 'day', 'hour'] else 0.1
            user_input[feat] = st.number_input(feat, value=default, step=step, format="%.2f")

    if st.button("🔍 Predict"):
        try:
            input_df = pd.DataFrame([user_input])
            X_input = input_df[features]  # Ensures column order
            X_scaled = scaler.transform(X_input)

            if model_choice == "Linear Regression":
                pred = model_lr.predict(X_scaled)[0]
            else:
                pred = model_knn.predict(X_scaled)[0]

            st.success(f"🌫️ Predicted PM2.5: **{pred:.2f} μg/m³**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
