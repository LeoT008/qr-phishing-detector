from pathlib import Path
import pandas as pd
import joblib
from features import extract_features

folder = Path(__file__).resolve().parent

models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Augmented Random Forest": "random_forest_augmented.joblib",
    "Forest without HTTPS": "random_forest_no_https.joblib",
}

urls = [
    "https://www.google.com",
    "https://google.com",
    "https://www.google.com/",
    "https://www.google.com/search?q=hello",
    "https://www.wikipedia.org",
    "https://wikipedia.org",
    "https://www.wikipedia.org/",
    "https://github.com",
    "https://github.com/login",
    "http://www.google.com",
    "http://google.com",
]

for name, filename in models.items():
    model = joblib.load(folder / "models" / filename)
    phishing_index = list(model.classes_).index(0)

    print(f"\n--- {name} ---")

    for url in urls:
        X = pd.DataFrame([extract_features(url)])
        X = X[list(model.feature_names_in_)]

        prediction = int(model.predict(X)[0])
        score = model.predict_proba(X)[0, phishing_index]
        verdict = "PHISHING" if prediction == 0 else "LEGITIMATE"

        print(f"{verdict:10} | score={score:.3f} | {url}")