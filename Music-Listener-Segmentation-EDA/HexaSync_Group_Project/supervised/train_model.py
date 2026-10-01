import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def train_and_evaluate():
    # 1. Load Dataset
    print("Loading dataset...")
    df = pd.read_csv("loans.csv")
    print(f"Dataset shape: {df.shape}")

    # 2. Separate Features and Target
    features = ["income", "credit_score", "loan_amount", "employment_years"]
    target = "loan_status"

    X = df[features]
    y = df[target]

    # 3. Train/Test Split (80/20 with Stratifications)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Define Candidate Classification Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("\n--- Training & Evaluating Models ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

        print(f"\nModel: {name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")

    # 6. Select Best Model Based on F1 Score
    best_model_name = max(results, key=lambda m: results[m]["f1"])
    best_model = results[best_model_name]["model"]

    print(f"\n==========================================")
    print(f" Best Model Selected: {best_model_name}")
    print(f" F1 Score: {results[best_model_name]['f1']:.4f}")
    print(f"==========================================")

    # 7. Save Model & Scaler Artifacts
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSuccessfully saved 'model.pkl' and 'scaler.pkl'.")

if __name__ == "__main__":
    train_and_evaluate()