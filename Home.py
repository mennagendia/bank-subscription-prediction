import streamlit as st
import pandas as pd
import io
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time  # For animations

# ✅ Set Page Configuration (Must be First)
st.set_page_config(page_title="Bank Subscription Prediction", layout="wide")

# 🔹 Sidebar Navigation with Animated Effect
st.sidebar.title("🔹 Navigation")
page = st.sidebar.radio("📍 Navigate to", ["🏠 Home", "📊 Prediction", "📈 Data Analysis"])

# ✅ Home Page
if page == "🏠 Home":
    st.title("🏠 Welcome to the Bank Subscription Prediction App")
    st.image("homepage.PNG", caption="🚀 AI-Powered Customer Subscription Prediction", use_container_width=True)

    with st.spinner("📊 Loading Dataset..."):
        time.sleep(1)  # Simulate loading
        df_origin = pd.read_csv('first_5_rows.csv')

    # 📌 Display Dataset Sample
    st.markdown("### 📊 Dataset Sample :")
    st.dataframe(df_origin.style.background_gradient(cmap='OrRd_r'))

    # 📌 Dataset Description
    st.markdown("📌 **Column Descriptions**")
    st.write(""" 
        - **The Dataset consists of** : 41,188 rows,and 21 columns.
        - **Numeric Features**: Age, Duration, Campaign, Pdays, etc.
        - **Categorical Features**: Job, Marital, Education, Contact Type, etc.
        - **Binary Target**: Subscribed to term deposit? (Yes/No)
        """ )
    with st.expander("📌 **View More Descriptions**"):
        st.write("""
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
    st.title("📊 Predict Customer Subscription")
    st.write("Fill in the customer details to predict if they will subscribe.")

    # 🚀 Load Model Function
    @st.cache_resource
    def load_model():
        try:
            return joblib.load("stacking_pipeline.pkl")
        except FileNotFoundError:
            st.error("⚠️ Model file not found. Please check if 'stacking_pipeline.pkl' exists.")
            return None

    model = load_model()

    # 📌 Input Fields with Animated Effect
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input("📞 Last Contact Duration (sec)", min_value=0, max_value=4918, value=180, step=10)
            previous = st.number_input("🔄 No. of Contacts in Past Campaigns", min_value=0, max_value=7, value=1, step=1)

        with col2:
            poutcome = st.radio("📈 Outcome of Previous Campaign", options=[0, 1], format_func=lambda x: "Success" if x == 1 else "Failure")
            is_contacted_before = st.radio("📬 Contacted Before?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # 🎯 Predict Button with Animation
    if st.button("🔍 Predict Subscription"):
        if model is None:
            st.warning("⚠️ Model is not loaded. Please check for missing files.")
        else:
            # ✅ Convert Input Data into DataFrame
            input_data = pd.DataFrame([[duration, previous, poutcome, is_contacted_before]], 
                                      columns=["duration", "previous", "poutcome", "is_contacted_before"])

            # 🎬 Show Progress Bar
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent + 1)

            # 🔍 Perform Prediction
            prediction = model.predict(input_data)[0]

            # 🎨 Display Result with Animation
            if prediction == 1:
                st.success("✅ The customer is **likely to subscribe** to the term deposit! 🎉")
            else:
                st.error("❌ The customer is **unlikely to subscribe** to the term deposit.")

# ✅ Data Analysis Page
elif page == "📈 Data Analysis":
    st.title("📈 Data Analysis & Insights")
    df = pd.read_csv("bank.csv", sep=";")
    
    tab1, tab2, tab3 = st.tabs(['📊 Univariate', '📉 Bivariate', '🔍 Multivariate'])

    # UNI-VARIATE ANALYSIS
    with tab1:
        st.subheader('🔍 Univariate Analysis', help="Select a column to analyze its distribution.")
        colu = list(df.columns)
        user_choice = st.selectbox('Select a column:', options=colu)

        if user_choice in df.select_dtypes(include='number').columns:
            fig_1 = px.histogram(df, x=user_choice, title=f'Distribution of {user_choice}')
            st.plotly_chart(fig_1)

            fig_2 = px.box(df, x=user_choice, title=f'Box Plot of {user_choice}')
            st.plotly_chart(fig_2)

        else:
            unique_count = df[user_choice].nunique()
            if unique_count <= 7:
                dff = df.groupby(user_choice).size().reset_index(name='Count').sort_values(by='Count', ascending=False)
                cat_fig = px.pie(dff, names=user_choice, values='Count', title=f'Distribution of {user_choice}')
            else:
                cat_fig = px.histogram(df, x=user_choice, title=f'Histogram of {user_choice}')
            st.plotly_chart(cat_fig)

    # BI-VARIATE ANALYSIS
    with tab2:
        st.header('📉 Bivariate Analysis', help="Explore relationships between two variables.")
        num_col = list(df.select_dtypes(include='number').columns)
        cat_colu = list(df.select_dtypes(include='O').columns)
        colu = list(df.columns)

        user_choice = st.selectbox('Select First Column:', options=colu)
        user_choice_2 = st.selectbox('Select Second Column:', options=colu)

        if user_choice in num_col and user_choice_2 in num_col:
            fig_p1 = px.scatter(df, x=user_choice, y=user_choice_2, title=f'Scatter Plot of {user_choice} vs {user_choice_2}')
            st.plotly_chart(fig_p1)

        elif user_choice in cat_colu and user_choice_2 in cat_colu:
            fig_1 = px.bar(df, x=user_choice, color=user_choice_2, title=f'Bar Chart of {user_choice} vs {user_choice_2}')
            st.plotly_chart(fig_1)

        else:
            fig_3 = px.box(df, x=user_choice, y=user_choice_2, title=f'Box Plot of {user_choice} vs {user_choice_2}')
            st.plotly_chart(fig_3)

    # MULTIVARIATE ANALYSIS
    with tab3:
        st.subheader('🔍 Multivariate Analysis')
        with st.spinner("Generating pairplot..."):
            sns.pairplot(df)
            st.pyplot(plt)

st.markdown("---")
st.markdown(
    """
    👨‍💻 **Developed by [Menna Gendia](https://www.linkedin.com/in/menna-gendia-59823a269/)**  
    📌 **GitHub:** [Menna Gendia](https://github.com/MennaGendia)  
    🛠️ **Powered by Deepnote, Python, Streamlit & Plotly**  
    🚀 _Making Data Science Accessible & Fun!_
    """
)
