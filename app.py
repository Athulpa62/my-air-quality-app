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
                padding: 1rem;
            }
            .stButton>button {
                background-color: #4caf50;
                color: white;
            }
            .stRadio > div {
                gap: 0.5rem;
            }
            .stImage > img {
                margin-bottom: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

# ========== Setup ==========
set_background()

# ========== Load Data and Models ==========
df = pd.read_csv("merged_data.csv")
model_lr = joblib.load("lr_model.pkl")
model_knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ========== Lottie Animations ==========
lottie_home = load_lottie("animation_home.json")
lottie_eda = load_lottie("animation_eda.json")
lottie_predict = load_lottie("animation_predict.json")
lottie_data = load_lottie("animation_data.json")
lottie_sidebar = load_lottie("animation_sidebar.json") 

# ========== Preprocess ==========
if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# ========== Sidebar ==========
with st.sidebar:
    st_lottie(lottie_sidebar, height=150)

    st.title("🌍 Air Quality App")
    menu = st.radio("📂 Navigate", ["🏠 Home", "📊 Data Overview", "📈 EDA", "🤖 Predict"])

    with st.expander("ℹ️ About"):
        st.write("Visualize air quality and predict PM2.5 using machine learning models.")

    st.markdown(f"🕒 **{datetime.now().strftime('%A, %B %d, %Y – %H:%M:%S')}**")
    st.markdown("---")

# ========== Pages ==========
if menu == "🏠 Home":
    st.title("Air Quality Prediction Dashboard")
    st_lottie(lottie_home, height=300)

    st.markdown("""
    Welcome to the Air Quality Prediction App! This tool predicts **PM2.5** concentration levels based on environmental features.

    ### 📌 Station Profiles:

    - **🏙️ Aotizhongxin**
        - Central urban district with dense traffic.
        - High pollution from vehicles and industry.
        - Frequent smog events, especially in winter.

    - **🌄 Changping**
        - Located in Beijing's north suburbs.
        - Pollution varies with wind and industrial activity.
        - Lower average PM2.5 than central stations.

    - **🏘️ Dongsi**
        - Mixed-use residential and commercial zone.
        - High vehicle flow, especially during rush hours.
        - PM2.5 peaks during weekdays.

    - **🏛️ Guanyuan**
        - Administrative center near government buildings.
        - Better regulation and lower industrial emissions.
        - Cleaner air, often used as a benchmark.

    ### 🚀 Features:
    - View raw and processed data.
    - Analyze PM2.5 trends and patterns.
    - Predict PM2.5 with ML models (LR & KNN).
    """)
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

elif menu == "📊 Data Overview":
    st.title("📊 Dataset Overview")
    st_lottie(lottie_data, height=200)

    st.dataframe(df.head(10))
    st.markdown(f"**Shape:** `{df.shape}`")
    st.markdown("### Missing Values")
    st.write(df.isnull().sum())
    st.markdown("### Data Types")
    st.write(df.dtypes)

elif menu == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    st_lottie(lottie_eda, height=200)

    plot_option = st.selectbox("Choose a plot", [
        "PM2.5 Distribution", "Correlation Heatmap", "PM2.5 vs Temperature",
        "Boxplot of PM2.5 by Month", "Trend Over Time",
        "Monthly Average PM2.5", "PM2.5 vs Wind Speed",
        "Hourly PM2.5 Pattern", "Yearly PM2.5 Trend", "Pairplot of Pollutants"])

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

    elif plot_option == "Monthly Average PM2.5":
        monthly_avg = df.groupby("month")["PM2.5"].mean()
        fig, ax = plt.subplots()
        ax.plot(monthly_avg.index, monthly_avg.values, marker='o')
        ax.set_title("Average PM2.5 by Month")
        st.pyplot(fig)

    elif plot_option == "PM2.5 vs Wind Speed":
        fig, ax = plt.subplots()
        sns.scatterplot(x='WSPM', y='PM2.5', data=df, alpha=0.5, ax=ax)
        ax.set_title("PM2.5 vs Wind Speed")
        st.pyplot(fig)

    elif plot_option == "Hourly PM2.5 Pattern":
        hourly_avg = df.groupby("hour")["PM2.5"].mean()
        fig, ax = plt.subplots()
        ax.plot(hourly_avg.index, hourly_avg.values, marker='o')
        ax.set_title("Hourly PM2.5 Pattern")
        st.pyplot(fig)

    elif plot_option == "Yearly PM2.5 Trend":
        if 'year' in df.columns:
            yearly_avg = df.groupby("year")["PM2.5"].mean()
            fig, ax = plt.subplots()
            ax.plot(yearly_avg.index, yearly_avg.values, marker='o')
            ax.set_title("Yearly PM2.5 Trend")
            st.pyplot(fig)

    elif plot_option == "Pairplot of Pollutants":
        selected = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        fig = sns.pairplot(df[selected].dropna())
        st.pyplot(fig)

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
            user_input[feat] = st.number_input(feat, value=default, step=step, format="%.2f")

    if st.button("🔍 Predict"):
        try:
            input_data = pd.DataFrame([user_input])[feature_names]
            input_scaled = scaler.transform(input_data)
            pred = model_lr.predict(input_scaled)[0] if model_choice == "Linear Regression" else model_knn.predict(input_scaled)[0]
            st.success(f"🌫️ Predicted PM2.5: **{pred:.2f} μg/m³**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
