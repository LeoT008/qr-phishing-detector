from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from time import perf_counter

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from features import extract_features


def extract_protocol_free_features(url):
    """Extract features after replacing every protocol with HTTPS."""
    url = str(url).strip()
    parsed = urlsplit(url if "://" in url else "//" + url)

    # Give HTTP and HTTPS addresses the same protocol.
    # Preserve the hostname, port, path, query, and fragment.
    normalized = urlunsplit((
        "https",
        parsed.netloc,
        parsed.path,
        parsed.query,
        parsed.fragment,
    ))

    values = extract_features(normalized)

    # This feature would now be constant, so remove it.
    values.pop("uses_https")
    return values


def main():
    folder = Path(__file__).resolve().parent

    # Reuse the previous augmented training URLs.
    train = pd.read_csv(
        folder / "data" / "url_splits" / "train_urls.csv"
    )

    # Reuse the same test URLs for a comparable experiment.
    test = pd.read_csv(
        folder / "data" / "url_splits" / "test_urls.csv"
    )

    # Verify that exact URLs do not overlap.
    overlap = set(train["URL"]) & set(test["URL"])
    if overlap:
        raise ValueError("Training and testing URLs overlap.")

    print("Extracting protocol-free features...", flush=True)

    X_train = pd.DataFrame(
        train["URL"].map(extract_protocol_free_features).tolist()
    )
    X_test = pd.DataFrame(
        test["URL"].map(extract_protocol_free_features).tolist()
    )
    X_test = X_test[X_train.columns]

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # Keep the previous augmented forest's settings.
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training protocol-free forest...", flush=True)
    start = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - start

    predictions = model.predict(X_test)

    # Label 0 represents phishing in this dataset.
    phishing_index = list(model.classes_).index(0)
    scores = model.predict_proba(X_test)[:, phishing_index]

    print(f"\nAccuracy: {(predictions == y_test).mean():.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(
        f"Phishing ROC-AUC: "
        f"{roc_auc_score(y_test == 0, scores):.4f}"
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix: Phishing, Legitimate")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Save separately, with the feature order for future predictions.
    output = folder / "models"
    output.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X_train.columns),
            "feature_mode": "protocol_free",
        },
        output / "random_forest_protocol_free_original.joblib",
    )

    print("\nExperimental model saved.")

    # Check familiar addresses without opening any website.
    examples = [
        "https://www.google.com",
        "http://www.google.com",
        "https://google.com",
        "http://google.com",
        "https://www.google.com/",
        "https://www.google.com/search?q=hello",
        "https://github.com",
        "https://github.com/login",
    ]

    print("\n--- Protocol-free URL checks ---")
    for url in examples:
        row = pd.DataFrame(
            [extract_protocol_free_features(url)],
            columns=X_train.columns,
        )

        prediction = model.predict(row)[0]
        score = model.predict_proba(row)[0, phishing_index]
        label = "PHISHING" if prediction == 0 else "LEGITIMATE"

        print(f"{label} | score={score:.3f} | {url}")


if __name__ == "__main__":
    main()