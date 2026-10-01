import streamlit as st
import requests

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Advanced Loan Approval Predictor")
st.write("Enter the applicant's financial details below to assess loan eligibility.")

# Form Inputs
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=75000, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=5)
        
    with col2:
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=1000000, value=25000, step=1000)
        employment_years = st.number_input("Years of Employment", min_value=0, max_value=50, value=5, step=1)

    submit_button = st.form_submit_button("Assess Loan Eligibility")

if submit_button:
    payload = {
        "income": float(income),
        "credit_score": float(credit_score),
        "loan_amount": float(loan_amount),
        "employment_years": float(employment_years)
    }

    try:
        # Call FastAPI backend
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            status = result["status"]
            confidence = result["approval_confidence"]

            st.markdown("---")
            if status == "APPROVED":
                st.success(f"🎉 **Decision: LOAN APPROVED**")
                st.metric("Approval Confidence", f"{confidence}%")
            else:
                st.error(f"❌ **Decision: LOAN REJECTED**")
                st.metric("Approval Confidence", f"{confidence}%")
        else:
            st.error("Error communicating with prediction server.")
    except Exception as e:
        st.error(f"Could not connect to FastAPI backend. Ensure `main.py` is running on port 8000.\nDetails: {e}")