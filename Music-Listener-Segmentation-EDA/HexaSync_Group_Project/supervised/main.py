from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval Prediction API")

# 1. Load trained model & scaler artifacts
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model/scaler artifacts: {e}")

# 2. Define input schema using Pydantic
class ApplicantData(BaseModel):
    income: float
    credit_score: float
    loan_amount: float
    employment_years: float

@app.get("/")
def read_root():
    return {"message": "Loan Approval Prediction API is running."}

@app.post("/predict")
def predict_loan_status(applicant: ApplicantData):
    try:
        # Create DataFrame in the exact required feature order
        features = ["income", "credit_score", "loan_amount", "employment_years"]
        input_data = pd.DataFrame([[
            applicant.income,
            applicant.credit_score,
            applicant.loan_amount,
            applicant.employment_years
        ]], columns=features)

        # Scale features and predict
        scaled_data = scaler.transform(input_data)
        prediction = int(model.predict(scaled_data)[0])
        probability = float(model.predict_proba(scaled_data)[0][1] * 100)

        status = "APPROVED" if prediction == 1 else "REJECTED"

        return {
            "prediction": prediction,
            "status": status,
            "approval_confidence": round(probability, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))