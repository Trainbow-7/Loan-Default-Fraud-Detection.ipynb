import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_colab = True
st.set_page_config(
    page_title="Nigerian Bank Loan Default Predictor",
    page_icon="🏦",
    layout="centered"
)

# --- Load Model Package ---
@st.cache_resource
def load_model():
    return joblib.load("loan_default_model.joblib")

load_error = None
try:
    model_pack = load_model()
    pipeline = model_pack["pipeline"]
    threshold = model_pack["threshold"]
    features = model_pack["features"]
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

st.title("🏦 Nigerian Bank Loan Default Predictor")
st.write("This interactive dashboard evaluates credit default risk for Nigerian bank loan applicants across all regions using our optimized machine learning pipeline.")

if not model_loaded:
    st.error(f"⚠️ Could not load `loan_default_model.joblib`. Details: {load_error}")
else:
    st.success("✅ Model loaded successfully!")

    st.header("Applicant Details")

    # Nigerian States and FCT
    nigeria_states = [
        "Lagos", "Abuja (FCT)", "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi",
        "Bayelsa", "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti",
        "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi",
        "Kogi", "Kwara", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo",
        "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara"
    ]

    # Generate input forms based on dataset characteristics
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other/Prefer not to say"])
        region = st.selectbox("Nigerian State / Region", nigeria_states)
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Part-time", "Unemployed", "Retired"])
        employment_length = st.number_input("Employment Length (Years)", min_value=0, max_value=50, value=5)
        employer_tenure_years = st.number_input("Tenure with Current Employer (Years)", min_value=0.0, max_value=50.0, value=3.0)
        home_ownership = st.selectbox("Home Ownership", ["Rent", "Mortgage", "Own", "Other"])
        annual_income = st.number_input("Annual Income (₦)", min_value=0.0, value=6500000.0, step=100000.0)
        stated_income = st.number_input("Stated Income (₦)", min_value=0.0, value=6500000.0, step=100000.0)
        verified_income = st.number_input("Verified Income (₦)", min_value=0.0, value=6300000.0, step=100000.0)

    with col2:
        credit_score = st.slider("Credit Score (CRC / FirstCentral / CreditRegistry equivalent)", min_value=300, max_value=850, value=650)
        open_accounts = st.number_input("Open Credit Accounts", min_value=0, max_value=50, value=8)
        delinq_2yrs = st.number_input("Delinquencies (Last 2 Years)", min_value=0, max_value=20, value=0)
        inquiries_6m = st.number_input("Inquiries (Last 6 Months)", min_value=0, max_value=20, value=1)
        util = st.slider("Revolving Line Utilization (%)", min_value=0.0, max_value=100.0, value=30.0)
        revol_bal = st.number_input("Revolving Balance (₦)", min_value=0.0, value=500000.0, step=50000.0)
        dti = st.slider("Debt-to-Income Ratio (DTI %)", min_value=0.0, max_value=100.0, value=18.0)
        loan_amount = st.number_input("Requested Loan Amount (₦)", min_value=10000.0, value=1500000.0, step=100000.0)
        term = st.selectbox("Loan Term (Months)", [36, 60])
        purpose = st.selectbox("Loan Purpose", ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "medical", "car", "moving", "other"])
        application_type = st.selectbox("Application Type", ["individual", "joint"])
        grade = st.selectbox("Assigned Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = st.selectbox("Assigned Sub-Grade", [f"{g}{i}" for g in ["A","B","C","D","E","F","G"] for i in range(1,6)])
        interest_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.5) / 100.0
        ip_risk_score = st.slider("IP Risk Score", min_value=0.0, max_value=100.0, value=25.0)
        device_change_30d = st.number_input("Device Changes (Last 30 Days)", min_value=0, max_value=10, value=0)
        has_document_mismatch = st.selectbox("Document Mismatch?", [0, 1])
        income_mismatch_ratio = st.number_input("Income Mismatch Ratio", min_value=0.0, value=1.0)
        days_to_fund = st.number_input("Days to Fund", min_value=0, max_value=30, value=5)
        days_app_to_issue = st.number_input("Days from Application to Issue", min_value=0, max_value=100, value=3)
        days_issue_to_payment = st.number_input("Days from Issue to First Payment", min_value=0, max_value=100, value=30)

    # --- Compile inputs ---
    input_dict = {
        "age": age, "gender": gender, "region": region, "employment_status": employment_status,
        "employment_length": employment_length, "employer_tenure_years": employer_tenure_years,
        "home_ownership": home_ownership, "annual_income": annual_income, "stated_income": stated_income,
        "verified_income": verified_income, "credit_score": credit_score, "open_accounts": open_accounts,
        "delinq_2yrs": delinq_2yrs, "inquiries_6m": inquiries_6m, "util": util, "revol_bal": revol_bal,
        "dti": dti, "loan_amount": loan_amount, "term": term, "purpose": purpose,
        "application_type": application_type, "grade": grade, "sub_grade": sub_grade,
        "interest_rate": interest_rate, "ip_risk_score": ip_risk_score,
        "device_change_30d": device_change_30d, "has_document_mismatch": has_document_mismatch,
        "income_mismatch_ratio": income_mismatch_ratio, "days_to_fund": days_to_fund,
        "days_app_to_issue": days_app_to_issue, "days_issue_to_payment": days_issue_to_payment
    }

    input_df = pd.DataFrame([input_dict])[features]

    # --- Predict ---
    st.markdown("--- ")
    if st.button("🔍 Evaluate Credit Risk", use_container_width=True):
        proba = pipeline.predict_proba(input_df)[0, 1]

        st.subheader("Assessment Results")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric("Default Probability", f"{proba:.2%}")

        with col_res2:
            if proba >= threshold:
                st.error(f"🚨 Decision: **High Risk** (Threshold: {threshold:.3f})")
            else:
                st.success(f"✅ Decision: **Low Risk** (Threshold: {threshold:.3f})")
