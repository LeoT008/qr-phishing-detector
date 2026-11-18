from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

folder = Path(__file__).resolve().parent

# Use the same augmented training data as the previous forest.
training = folder / "data" / "augmented"
testing = folder / "data" / "splits"

X_train = pd.read_csv(training / "X_train.csv")
y_train = pd.read_csv(training / "y_train.csv")["label"]

X_test = pd.read_csv(testing / "X_test.csv")
y_test = pd.read_csv(testing / "y_test.csv")["label"]

# Remove only HTTPS status for this comparison.
X_train = X_train.drop(columns=["uses_https"])
X_test = X_test[list(X_train.columns)]

# Keep the previous augmented forest's other settings.
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training forest without HTTPS...", flush=True)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    digits=4,
    zero_division=0
))

# Rows are actual classes; columns are predicted classes.
print("Confusion matrix: Phishing, Legitimate")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Preserve the existing models by saving a separate artifact.
joblib.dump(
    model,
    folder / "models" / "random_forest_no_https.joblib"
)

print("\nExperimental model saved.")