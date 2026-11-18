from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from features import extract_features

folder = Path(__file__).resolve().parent

# Load the original held-out URLs.
test = pd.read_csv(
    folder / "data" / "url_splits" / "test_urls.csv"
)

# Evaluate HTTPS URLs only.
https = test[
    test["URL"].str.lower().str.startswith("https://")
].copy()

X = pd.DataFrame(
    https["URL"].map(extract_features).tolist()
)
y = https["label"].to_numpy()

models = {
    "Original Forest": "random_forest.joblib",
    "Augmented Forest": "random_forest_augmented.joblib",
}

print("HTTPS test URLs:", len(https))

for name, filename in models.items():
    model = joblib.load(folder / "models" / filename)

    # Preserve the feature order expected by each model.
    ordered_X = X[list(model.feature_names_in_)]
    predictions = model.predict(ordered_X)

    print(f"\n--- {name} ---")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        digits=4,
        zero_division=0
    ))

    # Rows are actual classes; columns are predictions.
    # Class order: phishing, legitimate.
    print(confusion_matrix(y, predictions, labels=[0, 1]))