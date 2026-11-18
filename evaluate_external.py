from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from features import extract_features

folder = Path(__file__).resolve().parent

data = pd.read_csv(
    folder / "data" / "external" / "external_urls.csv"
)

# Use only URLs whose HTTP/HTTPS protocol is explicitly recorded.
data = data[
    data["URL"].str.match(r"(?i)^https?://", na=False)
].copy()

print("External URLs with explicit protocols:", len(data))
print("Class counts: 0 = Phishing, 1 = Legitimate")
print(data["label"].value_counts())

print("\nExtracting features...", flush=True)
X = pd.DataFrame(
    data["URL"].map(extract_features).tolist()
)
y = data["label"].to_numpy()

# Evaluate existing models without retraining them.
models = {
    "Original Forest": "random_forest.joblib",
    "Augmented Forest": "random_forest_augmented.joblib",
    "Forest without HTTPS": "random_forest_no_https.joblib",
}

results = []

for name, filename in models.items():
    print(f"\n--- {name} ---", flush=True)
    model = joblib.load(folder / "models" / filename)

    # Match the exact feature order used during training.
    model_X = X[list(model.feature_names_in_)]
    predictions = model.predict(model_X)

    # Label 0 means phishing.
    phishing_index = list(model.classes_).index(0)
    scores = model.predict_proba(model_X)[:, phishing_index]

    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y == 0, scores)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Phishing ROC-AUC: {auc:.4f}")

    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix: Phishing, Legitimate")
    print(matrix)

    results.append({
        "model": name,
        "test_rows": len(data),
        "accuracy": accuracy,
        "phishing_roc_auc": auc,
        "phishing_recall": matrix[0, 0] / matrix[0].sum(),
        "legitimate_false_alarm_rate":
            matrix[1, 0] / matrix[1].sum(),
        "missed_phishing": int(matrix[0, 1]),
        "legitimate_flagged": int(matrix[1, 0]),
    })

# Save a comparison table for the report.
output = folder / "results"
output.mkdir(exist_ok=True)

pd.DataFrame(results).to_csv(
    output / "external_explicit_protocol_results.csv",
    index=False,
)

print("\nResults saved. No websites were opened.")