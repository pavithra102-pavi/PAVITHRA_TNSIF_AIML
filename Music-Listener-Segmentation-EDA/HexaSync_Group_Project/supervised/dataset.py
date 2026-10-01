import pandas as pd
import numpy as np

def generate_loan_dataset(num_samples=1000):
    np.random.seed(42)
    
    # 1. Generate synthetic features matching project architecture
    income = np.random.randint(25000, 150000, num_samples)
    credit_score = np.random.randint(550, 850, num_samples)
    loan_amount = np.random.randint(10000, 500000, num_samples)
    employment_years = np.random.randint(0, 20, num_samples)
    
    # 2. Normalize features (0.0 to 1.0 scales)
    norm_credit = (credit_score - 550) / 300.0
    norm_income = income / 150000.0
    norm_emp = employment_years / 20.0
    
    # Loan burden ratio: High loan amount relative to income acts as a penalty
    loan_burden = loan_amount / (income * 4.0)
    loan_burden = np.clip(loan_burden, 0.0, 1.0)
    
    # 3. Weighted financial scoring formula
    approval_score = (
        0.40 * norm_credit +
        0.35 * norm_income +
        0.15 * norm_emp -
        0.20 * loan_burden
    )
    
    # 4. Balanced decision threshold
    loan_status = (approval_score > 0.35).astype(int)
    
    # 5. Build and save DataFrame
    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "loan_status": loan_status
    })
    
    df.to_csv("loans.csv", index=False)
    print("Successfully generated balanced 'loans.csv' with shape:", df.shape)
    print("Loan Status Distribution:\n", df["loan_status"].value_counts())

if __name__ == "__main__":
    generate_loan_dataset()