"""
Train and compare multiple classification algorithms.
Dataset: Breast Cancer Wisconsin dataset bundled with scikit-learn.
Educational use only; not for clinical diagnosis.
"""
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# 1. Load a built-in sample dataset (no separate download needed).
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target
target_names = list(dataset.target_names)  # 0 = malignant, 1 = benign

# Export a CSV so you can inspect and reuse the sample dataset.
sample = X.copy()
sample["target"] = y.map({0: "malignant", 1: "benign"})
sample.to_csv(ROOT / "sample_dataset.csv", index=False)
print(f"Dataset shape: {sample.shape}")
print("Target counts:")
print(sample["target"].value_counts())

# 2. Split into train/test sets with stratification to preserve class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Define algorithms. Pipelines scale features where it is helpful.
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": GaussianNB(),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}

# 4. Train, evaluate, and save each model.
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }
    results.append(metrics)

    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    print(f"\n{'=' * 60}\n{name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(classification_report(
        y_test, predictions, target_names=target_names, zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))

# 5. Save comparison and metadata.
results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
results_df.to_csv(ROOT / "model_comparison.csv", index=False)

best_name = str(results_df.iloc[0]["model"])
metadata = {
    "best_model": best_name,
    "feature_names": list(X.columns),
    "target_mapping": {"0": "malignant", "1": "benign"},
    "dataset": "scikit-learn Breast Cancer Wisconsin dataset",
    "note": "Educational demonstration only; not for medical diagnosis."
}
with open(ROOT / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nModel comparison (sorted by macro F1):")
print(results_df.to_string(index=False))
print(f"\nBest model by macro F1: {best_name}")
print(f"Models saved in: {MODEL_DIR}")
print("Sample dataset saved as sample_dataset.csv")
