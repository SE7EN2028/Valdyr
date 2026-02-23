import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "house_price_model.pkl")
model_columns = joblib.load(BASE_DIR / "model_columns.pkl")

# ---------------- HEADER ---------------- #
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🏡 House Price Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown("### Enter property details")

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.header("House Features")

area = st.sidebar.slider("Area (sq ft)", 500, 10000, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 4, 2)
parking = st.sidebar.slider("Parking", 0, 3, 1)

mainroad = st.sidebar.selectbox("Main Road", ["Yes", "No"])
guestroom = st.sidebar.selectbox("Guest Room", ["Yes", "No"])
basement = st.sidebar.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.sidebar.selectbox("Preferred Area", ["Yes", "No"])

furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# ---------------- DATAFRAME ---------------- #
input_dict = {
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "mainroad": 1 if mainroad == "Yes" else 0,
    "guestroom": 1 if guestroom == "Yes" else 0,
    "basement": 1 if basement == "Yes" else 0,
    "hotwaterheating": 1 if hotwaterheating == "Yes" else 0,
    "airconditioning": 1 if airconditioning == "Yes" else 0,
    "prefarea": 1 if prefarea == "Yes" else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ---------------- PREDICTION ---------------- #
st.write("")

if st.button("💰 Predict Price"):

    prediction = model.predict(input_df)[0]

    st.markdown("## 🏷️ Estimated Price")
    st.success(f"₹ {int(prediction):,}")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)