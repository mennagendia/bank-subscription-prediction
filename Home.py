import streamlit as st
import pandas as pd
import io
import joblib

# ✅ Set Page Configuration (Must be First)
st.set_page_config(page_title="Bank Subscription Prediction", layout="wide")

# 🎯 Sidebar Navigation
st.sidebar.title("🔹 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Prediction", "📈 Data Analysis"])

# ✅ Home Page
if page == "🏠 Home":
    st.title("🏠 Welcome to the Home Page")
    st.image("homepage.PNG", caption="Welcome to Customer Subscription Prediction Of Bank Marketing Campaign", use_container_width=True)
    # ✅ Load Dataset
    df_origin = pd.read_csv('first_5_rows.csv')

    # 📌 Display Dataset Sample
    st.markdown("### 📊 Dataset Sample:")
    st.dataframe(df_origin.style.background_gradient(cmap='OrRd_r'))

    # 📌 Display Column Descriptions
    st.markdown(
        """ 
        ### 🏷️ Columns Description:
        - **📌 Numeric Attributes**  
          - Age → Age of client  
          - Duration → Last contact duration (in sec)  
          - Campaign → No. of contacts in this campaign  
          - Pdays → Days since last contact (999 = never contacted)  
          - Previous → No. of contacts in past campaigns  
          - Emp.var.rate → Employment variation rate (quarterly)  
          - Cons.price.idx → Consumer price index (monthly)  
          - Cons.conf.idx → Consumer confidence index (monthly)  
          - Euribor3m → Euribor 3-month rate (daily)  
          - N.employed → Number of employees (quarterly)  

        - **📌 Categorical Attributes**  
          - Job → Type of job  
          - Marital → Marital status  
          - Education → Level of education  
          - Default → Has credit in default?  
          - Housing → Has housing loan?  
          - Loan → Has personal loan?  
          - Contact → Contact communication type  
          - Month → Last contact month  
          - Day_of_week → Last contact day  
          - Poutcome → Outcome of previous campaign  

        - **📌 Binary Attribute**  
          - Y → Subscribed to term deposit? (Target)
        """)
# ✅ Prediction Page
elif page == "📊 Prediction":
    st.title("📊 Prediction Page")
    st.write("Enter customer details to predict subscription.")

    # 🚀 Load Model Function
    @st.cache_resource
    def load_model():
        try:
            return joblib.load("stacking_pipeline.pkl")
        except FileNotFoundError:
            st.error("⚠️ Model file not found. Please check if 'stacking_pipeline.pkl' exists.")
            return None

    model = load_model()

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
            # ✅ Convert Input Data into DataFrame
            input_data = pd.DataFrame([[duration, previous, poutcome, is_contacted_before]], 
                                      columns=["duration", "previous", "poutcome", "is_contacted_before"])

            # 🔍 Perform Prediction
            prediction = model.predict(input_data)[0]  # Extract single value

            # 🎨 Display Result
            if prediction == 1:
                st.success("✅ The customer is **likely to subscribe** to the term deposit! 🎉")
            else:
                st.error("❌ The customer is **unlikely to subscribe** to the term deposit.")

# ✅ Data Analysis Page
elif page == "📈 Data Analysis":
    st.title("📈 Data Analysis")
    st.write("Explore visualizations and insights from the dataset.")

    

# 📌 Footer
st.markdown("---")
st.write("🚀 Built with ❤️ using Streamlit")
