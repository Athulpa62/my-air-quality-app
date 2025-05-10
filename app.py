import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit_lottie import st_lottie
import json
from datetime import datetime

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

# ========== Load Data ==========
df = pd.read_csv("merged_data.csv")

# ========== Load Models and Assets ==========
@st.cache_resource
def load_assets():
    model_lr = joblib.load("lr_model.pkl")
    model_knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model_lr, model_knn, scaler, feature_names

model_lr, model_knn, scaler, feature_names = load_assets()

# ========== Load Lottie Animations ==========
lottie_home = load_lottie("animation_home.json")
lottie_eda = load_lottie("animation_eda.json")
lottie_predict = load_lottie("animation_predict.json")

# ========== Preprocess Data ==========
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
    Welcome to the Air Quality Prediction App! This tool predicts **PM2.5** concentration levels based on environmental features.

    ### 📌 About the Monitoring Stations:

    - **🏙️ Aotizhongxin**
      - Located in Beijing's Olympic Sports Center area.
      - High population density and heavy traffic contribute to significant pollution levels.
      - Seasonal variation due to heating and construction activities.

    - **🌄 Changping**
      - Northern suburban district of Beijing.
      - Pollution levels vary with industrial emissions and regional wind patterns.
      - Nearby hills can trap pollutants during temperature inversions.

    - **🏘️ Dongsi**
      - A dense residential and commercial zone in downtown Beijing.
      - Experiences frequent PM2.5 spikes from vehicle emissions and local street activity.
      - Narrow streets reduce natural dispersion of pollutants.

    - **🏛️ Guanyuan**
      - Government district with administrative offices and open spaces.
      - Typically records cleaner air due to fewer traffic sources and strict regulations.
      - Serves as a useful reference for comparing central vs peripheral zones.

    ### 🚀 App Features:
    - Explore and visualize air quality data.
    - Perform predictions using Linear Regression or KNN models.
    - Gain insights through interactive EDA plots.
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

    user_input = {}
    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        with cols[i % 2]:
            default = 2020.0 if feat == 'year' else 1.0 if feat in ['month', 'day', 'hour'] else 50.0
            step = 1.0 if feat in ['year', 'month', 'day', 'hour'] else 0.1
            user_input[feat] = st.number_input(feat, value=default, step=step)

    if st.button("🔍 Predict"):
        try:
            X_ordered = np.array([[user_input[feat] for feat in feature_names]])
            X_scaled = scaler.transform(X_ordered)
            pred = model_lr.predict(X_scaled)[0] if model_choice == "Linear Regression" else model_knn.predict(X_scaled)[0]
            st.success(f"🌫️ Predicted PM2.5: **{pred:.2f} μg/m³**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
