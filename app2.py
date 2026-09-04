import streamlit as st
import numpy as np
import pandas as pd
import joblib

# load model, scaler, and feature columns
model = joblib.load('churnmodel.pkl')
scaler = joblib.load('churn_scaler.pkl')
feature = joblib.load('feature_columns.pkl')  # list of columns model expects, in order

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction",
                    page_icon=":bar_chart:", layout="centered")

st.title("Customer Churn Predictor")
st.markdown("Fill in the customer's details below to predict whether a customer will churn or not.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (in months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    tv_service = st.selectbox("Streaming TV", ["Yes", "No"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                               "Bank transfer (automatic)", "Credit card (automatic)"])

st.divider()

if st.button("Predict Churn"):

    yes_no_map = {"No": 0, "Yes": 1}

    label_encoded = {
        "gender": 0 if gender == "Female" else 1,          # Female=0, Male=1
        "SeniorCitizen": yes_no_map[senior_citizen],
        "Partner": yes_no_map[partner],
        "Dependents": yes_no_map[dependents],
        "PhoneService": yes_no_map[phone_service],
        "MultipleLines": yes_no_map[multiple_lines],
        "OnlineSecurity": yes_no_map[online_security],
        "OnlineBackup": yes_no_map[online_backup],
        "DeviceProtection": yes_no_map[device_protection],
        "TechSupport": yes_no_map[tech_support],
        "StreamingTV": yes_no_map[tv_service],
        "StreamingMovies": yes_no_map[movies],
        "PaperlessBilling": yes_no_map[paperless_billing],
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # --- One-hot encoded columns (bool in training data) ---
    onehot_raw = {
        "InternetService": internet_service,
        "Contract": contract,
        "PaymentMethod": payment,
    }

    input_df = pd.DataFrame([label_encoded])
    onehot_df = pd.get_dummies(pd.DataFrame([onehot_raw]))
    input_df = pd.concat([input_df, onehot_df], axis=1)

  
    input_df = input_df.reindex(columns=feature, fill_value=0)

    # Scale numeric columns with the SAME scaler fit during training
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"High Churn Risk - {prob*100:.1f}% probability")
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
        st.success(f"Low Churn Risk - {prob*100:.1f}% probability")
        st.caption(f"Churn Probability: {prob*100:.1f}%")