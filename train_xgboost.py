from pathlib import Path
from time import perf_counter
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

folder = Path(__file__).resolve().parent
splits = folder / "data" / "splits"

X_train = pd.read_csv(splits / "X_train.csv")
X_test = pd.read_csv(splits / "X_test.csv")
y_train = pd.read_csv(splits / "y_train.csv")["label"]
y_test = pd.read_csv(splits / "y_test.csv")["label"]

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)

print("Training XGBoost...", flush=True)
start = perf_counter()
model.fit(X_train, y_train)
training_time = perf_counter() - start

predictions = model.predict(X_test)

phishing_index = list(model.classes_).index(0)
phishing_probability = model.predict_proba(X_test)[:, phishing_index]

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("Training time:", round(training_time, 2), "seconds")
print("Phishing ROC-AUC:", round(
    roc_auc_score((y_test == 0).astype(int), phishing_probability), 4
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

print("Confusion matrix: rows = actual, columns = predicted")
print("Class order: Phishing, Legitimate")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

models = folder / "models"
models.mkdir(exist_ok=True)
joblib.dump(model, models / "xgboost.joblib")

print("\nModel saved.")