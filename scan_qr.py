from pathlib import Path
from urllib.parse import urlsplit
import sys
import pandas as pd
import joblib

from decode_qr import decode_qr
from features import extract_features

folder = Path(__file__).resolve().parent

if len(sys.argv) != 2:
    print("Usage: python scan_qr.py image.png")
    sys.exit(1)

image_path = Path(sys.argv[1]).expanduser()

if not image_path.is_file():
    print("Image file not found:", image_path)
    sys.exit(1)

try:
    # Decode locally. Never open the embedded destination.
    url = decode_qr(image_path).strip()
    print("Decoded content:", url)

    # Only classify valid HTTP or HTTPS addresses.
    parsed = urlsplit(url)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or any(c.isspace() for c in url)
    ):
        raise ValueError("QR content is not a valid HTTP/HTTPS URL.")

    parsed.port  # Check whether the port number is valid.

except ValueError as error:
    print("Error:", error)
    sys.exit(1)

# Extract the same features used during model training.
features = pd.DataFrame([extract_features(url)])

models = {
    "Mixed-source Gradient Boosting": "mixed_source_selected.joblib",
    "Domain-trained Forest": "random_forest_domain_reproducible.joblib",
    "Original Forest": "random_forest.joblib",
    "Augmented Forest": "random_forest_augmented.joblib",
}

print("\nExperimental predictions:")

for name, filename in models.items():
    model = joblib.load(folder / "models" / filename)
    X = features[list(model.feature_names_in_)]

    # Dataset label 0 represents phishing.
    index = list(model.classes_).index(0)
    score = model.predict_proba(X)[0, index]
    prediction = int(model.predict(X)[0])

    verdict = "PHISHING" if prediction == 0 else "LEGITIMATE"
    print(f"{name}: {verdict} — phishing score {score:.3f}")

print("\nPredictions do not guarantee safety.")
print("No website was opened.")
