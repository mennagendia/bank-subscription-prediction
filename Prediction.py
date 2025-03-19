import pandas as pd
import numpy as np
import streamlit as st
import joblib

# ✅ Load Model
#@st.cache_resource
def load_model():
    try:
        return joblib.load("stacking_pipeline.pkl")
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please check if 'stacking_pipeline.pkl' exists.")
        return None

model = load_model()

# ✅ Set Page Layout
st.set_page_config(page_title="Bank Subscription Prediction", layout="centered")

# 🎨 Page Title
st.markdown("<h1 style='text-align: center;'>📊 Customer Subscription Prediction</h1>", unsafe_allow_html=True)
st.write("### Enter customer details below to predict if they will subscribe to a term deposit.")

# 📌 Input Layout Using Columns
col1, col2 = st.columns(2)

# ✅ Input Fields
with col1:
    duration = st.number_input("📞 Last Contact Duration (sec)", min_value=0, max_value=4918, value=180, step=10)
    previous = st.number_input("🔄 No. of Contacts in Past Campaigns", min_value=0, max_value=7, value=1, step=1)

with col2:
    poutcome = st.radio("📈 Outcome of Previous Campaign", options=[0, 1], format_func=lambda x: "Success" if x == 1 else "Failure")
    is_contacted_before = st.radio("📬 Contacted Before?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# 🎯 Predict Button
if st.button("🔍 Predict Subscription"):
    if model is None:
        st.warning("⚠️ Model is not loaded. Please check for missing files.")
    else:
        # ✅ Convert Input Data into a Pandas DataFrame with column names
        input_data = pd.DataFrame([[duration, previous, poutcome, is_contacted_before]], 
                                  columns=["duration", "previous", "poutcome", "is_contacted_before"])

        # 🔍 Perform Prediction
        prediction = model.predict(input_data)[0]  # Extract single value

        # 🎨 Display Result
        if prediction == 1:
            st.success("✅ The customer is **likely to subscribe** to the term deposit! 🎉")
        else:
            st.error("❌ The customer is **unlikely to subscribe** to the term deposit.")

# 📌 Footer
st.markdown("---")
st.write("🚀 Built with ❤️ using Streamlit")
