import joblib
import pandas as pd

# Load saved model artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Expanded test suite with real-world edge cases
test_cases = [
    {"label": "High Risk / Bad Credit",        "data": [15000, 550, 100000, 0]},
    {"label": "Fresh Graduate / Early Career", "data": [32000, 680, 20000, 1]},
    {"label": "Mid-Career Safe Borrower",     "data": [85000, 740, 50000, 6]},
    {"label": "High Earner / Ruined Credit",   "data": [140000, 560, 40000, 8]},
    {"label": "Low Income / Extreme Debt",     "data": [28000, 610, 350000, 2]},
    {"label": "Moderate Salary / Top Credit",  "data": [60000, 820, 25000, 7]},
    {"label": "Borderline Boundary Case",      "data": [55000, 650, 60000, 3]},
    {"label": "Elite / High Earner",           "data": [150000, 850, 10000, 18]},
]

columns = ["income", "credit_score", "loan_amount", "employment_years"]

print("\n--- MODEL VALIDATION REPORT ---")
for case in test_cases:
    df_sample = pd.DataFrame([case["data"]], columns=columns)
    scaled_sample = scaler.transform(df_sample)
    
    # Prediction and confidence score
    pred = model.predict(scaled_sample)[0]
    prob_approved = model.predict_proba(scaled_sample)[0][1] * 100
    
    status = "APPROVED" if pred == 1 else "REJECTED"
    print(f"Profile: {case['label']}")
    print(f"  Input: {case['data']}")
    print(f"  Result: {status} (Approval Confidence: {prob_approved:.1f}%)\n")