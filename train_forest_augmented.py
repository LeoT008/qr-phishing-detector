from pathlib import Path
from time import perf_counter
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

folder = Path(__file__).resolve().parent
augmented = folder / "data" / "augmented"
splits = folder / "data" / "splits"

X_train = pd.read_csv(augmented / "X_train.csv")
y_train = pd.read_csv(augmented / "y_train.csv")["label"]

# Evaluate on the unchanged original test set.
X_test = pd.read_csv(splits / "X_test.csv")
y_test = pd.read_csv(splits / "y_test.csv")["label"]
X_test = X_test[X_train.columns]

model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training experimental Random Forest...", flush=True)
start = perf_counter()
model.fit(X_train, y_train)
training_time = perf_counter() - start

predictions = model.predict(X_test)
phishing_index = list(model.classes_).index(0)
probabilities = model.predict_proba(X_test)[:, phishing_index]

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 4
))
print("Training time:", round(training_time, 2), "seconds")
print("Phishing ROC-AUC:", round(
    roc_auc_score((y_test == 0).astype(int), probabilities), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    digits=4,
    zero_division=0
))

print("Confusion matrix: Phishing, Legitimate")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

joblib.dump(
    model,
    folder / "models" / "random_forest_augmented.joblib"
)

print("\nExperimental model saved.")