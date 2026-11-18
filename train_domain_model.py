"""Train and evaluate a URL model on the existing domain-separated splits.

Labels: 0 = phishing, 1 = legitimate. The pilot test is never used to fit
or select the model. Input splits must have URL, label, and domain_group.
"""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from features import extract_features

ROOT = Path(__file__).resolve().parent


def load_split(name):
    data = pd.read_csv(ROOT / 'data' / 'domain_splits' / f'{name}.csv')
    if data[['URL', 'label', 'domain_group']].isna().any().any():
        raise ValueError(f'{name} contains missing values')
    if not set(data.label.unique()) <= {0, 1}:
        raise ValueError(f'{name} has unexpected labels')
    return data


def main():
    splits = {name: load_split(name) for name in ('train', 'validation', 'pilot_test')}
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if set(splits[name].domain_group) & set(splits[other].domain_group):
                raise ValueError(f'Domain groups overlap: {name} and {other}')
            if set(splits[name].URL) & set(splits[other].URL):
                raise ValueError(f'URLs overlap: {name} and {other}')

    def features(data):
        return pd.DataFrame(data.URL.map(extract_features).tolist())

    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    print('Training on domain-separated training data...', flush=True)
    model.fit(features(splits['train']), splits['train'].label)
    results = {}
    for name in ('validation', 'pilot_test'):
        data = splits[name]
        X = features(data)[list(model.feature_names_in_)]
        pred = model.predict(X)
        score = model.predict_proba(X)[:, list(model.classes_).index(0)]
        matrix = confusion_matrix(data.label, pred, labels=[0, 1])
        results[name] = {
            'rows': len(data), 'accuracy': accuracy_score(data.label, pred),
            'phishing_recall': matrix[0, 0] / matrix[0].sum(),
            'legitimate_false_alarm_rate': matrix[1, 0] / matrix[1].sum(),
            'phishing_roc_auc': roc_auc_score(data.label == 0, score),
            'confusion_matrix_actual_rows_predicted_columns_0_1': matrix.tolist(),
        }
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    joblib.dump(model, ROOT / 'models' / 'random_forest_domain_reproducible.joblib')
    (ROOT / 'results' / 'domain_model_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
