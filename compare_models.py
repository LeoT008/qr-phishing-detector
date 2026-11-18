"""Compare saved original-data models on the original and external test sets."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from features import extract_features

ROOT = Path(__file__).resolve().parent
MODELS = {'Logistic regression': 'logistic_regression.joblib',
          'Random forest': 'random_forest.joblib',
          'XGBoost': 'xgboost.joblib'}


def main():
    x_test = pd.read_csv(ROOT / 'data/splits/X_test.csv')
    y_test = pd.read_csv(ROOT / 'data/splits/y_test.csv')['label']
    external = pd.read_csv(ROOT / 'data/external/external_urls.csv')
    external = external[external.URL.str.match(r'(?i)^https?://', na=False)].copy()
    external_x = pd.DataFrame(external.URL.map(extract_features).tolist())
    datasets = [('Original held-out test', x_test, y_test),
                ('External explicit-protocol test', external_x, external.label)]
    rows = []
    for model_name, filename in MODELS.items():
        print('Evaluating', model_name, flush=True)
        model = joblib.load(ROOT / 'models' / filename)
        for dataset_name, X, y in datasets:
            X = X[list(model.feature_names_in_)]
            pred = model.predict(X)
            scores = model.predict_proba(X)[:, list(model.classes_).index(0)]
            cm = confusion_matrix(y, pred, labels=[0, 1])
            rows.append({'dataset': dataset_name, 'model': model_name, 'rows': len(y),
                         'phishing_rows': int((y == 0).sum()), 'legitimate_rows': int((y == 1).sum()),
                         'accuracy': accuracy_score(y, pred),
                         'phishing_precision': precision_score(y, pred, pos_label=0, zero_division=0),
                         'phishing_recall': recall_score(y, pred, pos_label=0, zero_division=0),
                         'phishing_f1': f1_score(y, pred, pos_label=0, zero_division=0),
                         'legitimate_false_alarm_rate': cm[1, 0] / cm[1].sum(),
                         'phishing_roc_auc': roc_auc_score(y == 0, scores),
                         'phishing_detected': int(cm[0, 0]), 'phishing_missed': int(cm[0, 1]),
                         'legitimate_flagged': int(cm[1, 0]), 'legitimate_accepted': int(cm[1, 1])})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'results/algorithm_comparison.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
