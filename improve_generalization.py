"""Compare mixed-source models on validation, then report existing test splits.

Both test sets were used in earlier experiments, so results are exploratory,
not a fresh independent final evaluation. Label 0 is phishing.
"""
from pathlib import Path
from urllib.parse import urlsplit
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from features import extract_features

ROOT = Path(__file__).resolve().parent


def group(url):
    try:
        host = (urlsplit(url if '://' in url else '//' + url).hostname or '').lower().rstrip('.')
        if not host:
            return None
        # Conservative approximation: group on the final two DNS labels. This
        # overgroups public suffixes like co.uk; it is not a public suffix parser.
        return '.'.join(host.split('.')[-2:])
    except ValueError:
        return None


def read(path):
    df = pd.read_csv(ROOT / path)[['URL', 'label']].dropna().copy()
    df['group'] = df.URL.map(group)
    return df.dropna(subset=['group'])


def xy(df):
    return pd.DataFrame(df.URL.map(extract_features).tolist()), df.label.to_numpy()


def metrics(model, X, y):
    pred = model.predict(X[list(model.feature_names_in_)])
    score = model.predict_proba(X[list(model.feature_names_in_)])[:, list(model.classes_).index(0)]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {'rows': len(y), 'accuracy': accuracy_score(y, pred),
            'phishing_precision': precision_score(y, pred, pos_label=0, zero_division=0),
            'phishing_recall': recall_score(y, pred, pos_label=0, zero_division=0),
            'phishing_f1': f1_score(y, pred, pos_label=0, zero_division=0),
            'legitimate_false_alarm_rate': cm[1, 0] / cm[1].sum(),
            'phishing_roc_auc': roc_auc_score(y == 0, score),
            'phishing_detected': int(cm[0, 0]), 'phishing_missed': int(cm[0, 1]),
            'legitimate_flagged': int(cm[1, 0]), 'legitimate_accepted': int(cm[1, 1])}


def main():
    original_train = read('data/url_splits/train_urls.csv')
    original_test = read('data/url_splits/test_urls.csv')
    external_train = read('data/domain_splits/train.csv')
    validation = read('data/domain_splits/validation.csv')
    pilot = read('data/domain_splits/pilot_test.csv')
    heldout_groups = set(pd.concat([original_test, validation, pilot]).group)
    # Keep both sources free of the conservative host groups in any evaluation.
    original_train = original_train[~original_train.group.isin(heldout_groups)]
    external_train = external_train[~external_train.group.isin(heldout_groups)]
    train = pd.concat([original_train, external_train], ignore_index=True)
    assert not (set(train.group) & heldout_groups)
    X_train, y_train = xy(train)
    X_val, y_val = xy(validation)
    # Give each source equal total weight despite different numbers of rows.
    weights = pd.Series([0.5 / len(original_train)] * len(original_train) +
                        [0.5 / len(external_train)] * len(external_train)).to_numpy() * len(train)
    candidates = {
        'Mixed random forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                                       class_weight='balanced', random_state=42, n_jobs=-1),
        'Mixed gradient boosting': HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31,
                                                                   l2_regularization=1, random_state=42),
    }
    models = {}
    validation_results = {}
    for name, model in candidates.items():
        print('Training', name, flush=True)
        model.fit(X_train, y_train, sample_weight=weights)
        models[name] = model
        validation_results[name] = metrics(model, X_val, y_val)
        print(name, validation_results[name], flush=True)
    # Fix selection based on validation phishing F1 only, before test scoring.
    winner = max(validation_results, key=lambda name: validation_results[name]['phishing_f1'])
    print('Validation-selected model:', winner, flush=True)
    model = models[winner]
    joblib.dump(model, ROOT / 'models' / 'mixed_source_selected.joblib')
    results = []
    original_model = joblib.load(ROOT / 'models/random_forest.joblib')
    domain_model = joblib.load(ROOT / 'models/random_forest_domain_reproducible.joblib')
    for dataset_name, data in [('Original URL test', original_test), ('External domain pilot', pilot)]:
        X, y = xy(data)
        for name, estimator in [('Original forest', original_model), ('Domain-trained forest', domain_model), (winner, model)]:
            # The domain-trained model has not been established as independent
            # of the original split. It remains an exploratory comparator.
            results.append({'dataset': dataset_name, 'model': name, **metrics(estimator, X, y)})
    pd.DataFrame(results).to_csv(ROOT / 'results/mixed_model_comparison.csv', index=False)
    metadata = {'training_rows': len(train), 'original_training_rows_after_group_filter': len(original_train),
                'external_training_rows_after_group_filter': len(external_train),
                'original_test_rows': len(original_test), 'external_validation_rows': len(validation),
                'external_pilot_rows': len(pilot), 'selection_metric': 'validation phishing F1',
                'selected_model': winner, 'validation': validation_results,
                'warning': 'Existing tests have been used previously; exploratory, not a fresh independent evaluation.'}
    (ROOT / 'results/mixed_model_experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
