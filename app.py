import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

# load model, scaler, and feature columns
model = joblib.load('churnmodel.pkl')
scaler = joblib.load('churn_scaler.pkl')
feature = joblib.load('feature_columns.pkl')

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", 
                   page_icon=":bar_chart:", layout="centered")

st.title("Customer Churn Predictor")
st.markdown("Fill in the customer's details below to predict whether a customer will churn or not.")

st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (in months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])


with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                             "Bank transfer (automatic)", "Credit card (automatic)"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    tv_service = st.selectbox("Streaming TV", ["Yes", "No"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No"])

st.divider()

if st.botton("Predict Churn"):
    raw = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,  # matches training encoding
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet_service,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": tv_service,
        "StreamingMovies": movies,
    }
    raw_df = pd.DataFrame([raw])

cat_cols = raw_df.select_dtypes(include=['object']).columns.tolist()
encoded_df = pd.get_dummies(raw_df, columns=cat_cols, drop_first=True)
input_df = encoded_df.reindex(columns=feature, fill_value=0)

# scale input
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaled_input = scaler.transform(input_df[numeric_cols])

# Predict
prob = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]   

# Display the result
st.subheader("Prediction Result")

if pred == 1:
    st.error(f"High Churn Risk - {prob*100:.1f} probability")
    st.markdown("**Recommended Actions:**")
    if contract == "Month-to-month":
        st.warning("- Offer discount to upgrade to annual contract")
    if internet_service == "Fiber optic":
        st.warning("- Check service quality and offer loyalty rewards")
    if payment == "Electronic check":
        st.warning("- Encourage switching to automatic payment methods")
    if tenure < 12:
        st.warning("- Offer incentives for longer tenure")
    if senior_citizen == "Yes":
        st.warning("- Provide senior citizen support and benefits")

else:
    st.success(f"LOW CHURN RISK - {prob*100:.1f} probability")
    st.markdown(f"Churn Probability: {prob*100:.1f}%")
    st.caption(f"Churn Probability: {prob*100:.1f}%")


st.divider()
