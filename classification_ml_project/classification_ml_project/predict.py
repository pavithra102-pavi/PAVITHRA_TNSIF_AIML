"""Predict a class for a row from sample_dataset.csv or a user-provided CSV."""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

with open(ROOT / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

model_name = metadata["best_model"]
model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
feature_names = metadata["feature_names"]
label_map = {"0": "malignant", "1": "benign"}

def predict_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    """Input must contain the same feature columns as the training dataset."""
    missing = [col for col in feature_names if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    X_new = input_df[feature_names].copy()
    predictions = model.predict(X_new)
    output = input_df.copy()
    output["predicted_class"] = [label_map[str(int(p))] for p in predictions]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_new)
        output["probability_malignant"] = probabilities[:, 0]
        output["probability_benign"] = probabilities[:, 1]
    return output

if __name__ == "__main__":
    print(f"Loaded model: {model_name}")
    print("1. Predict from a CSV file")
    print("2. Predict a sample row from sample_dataset.csv")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        file_path = input("Enter CSV path: ").strip()
        new_data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(ROOT / "sample_dataset.csv")
        # Exclude target so only input features are sent to the model.
        new_data = data.drop(columns=["target"]).iloc[[0]].copy()

    predictions = predict_dataframe(new_data)
    print(predictions[["predicted_class"]].to_string(index=False))
    if "probability_malignant" in predictions.columns:
        print(predictions[["probability_malignant", "probability_benign"]].to_string(index=False))
    predictions.to_csv(ROOT / "predictions.csv", index=False)
    print("Full predictions saved to predictions.csv")
