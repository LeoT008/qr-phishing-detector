from pathlib import Path
from urllib.parse import urlsplit
import pandas as pd
import joblib
from features import extract_features

folder = Path(__file__).resolve().parent

# Load both models once so we can compare their behavior.
models = {
    "Original Forest": joblib.load(
        folder / "models" / "random_forest.joblib"
    ),
    "Augmented Forest": joblib.load(
        folder / "models" / "random_forest_augmented.joblib"
    ),
}

print("Experimental URL checker. No website will be opened.")
print("Predictions do not guarantee safety. Type exit to stop.")

while True:
    url = input("\nEnter a full http:// or https:// URL: ").strip()

    if url.lower() == "exit":
        break

    # Validate the address locally before extracting features.
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or any(c.isspace() for c in url)
        ):
            raise ValueError("Invalid URL")
        parsed.port  # Check for an invalid port number.
    except ValueError:
        print("Please enter a valid HTTP or HTTPS URL.")
        continue

    # Use the same feature function used during training.
    features = pd.DataFrame([extract_features(url)])

    for name, model in models.items():
        X = features[list(model.feature_names_in_)]

        # Class 0 means phishing in this dataset.
        index = list(model.classes_).index(0)
        score = model.predict_proba(X)[0, index]
        prediction = int(model.predict(X)[0])

        label = "PHISHING" if prediction == 0 else "LEGITIMATE"
        print(f"{name}: {label} — phishing score {score:.3f}")