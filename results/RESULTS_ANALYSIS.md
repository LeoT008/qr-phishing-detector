# Results analysis

## Experimental setup

Three previously trained algorithms were compared: logistic regression, random forest, and XGBoost. Each uses the project's 18 URL features. The saved models were evaluated without retraining, using their ordinary prediction thresholds. In this project, label 0 represents phishing and label 1 represents legitimate URLs. All phishing precision, recall, F1, and ROC-AUC values below treat phishing as the positive class.

Two test sets were used. The original held-out test contains 47,074 URLs (20,104 phishing and 26,970 legitimate) from the PhiUSIIL dataset. The original scripts used a stratified random URL-level split; it does not establish that the domains in training and test are different. The external test contains 54,982 URLs (19,917 phishing and 35,065 legitimate) from the separately supplied `malicious_phish.csv` dataset, restricted to addresses with explicit HTTP or HTTPS schemes. The external preparation script removes invalid addresses, conflicting and duplicate comparison keys, and matching URL keys from the original dataset. The external set is not a random sample of all external URLs: URLs without an explicit scheme were excluded, which can bias the results. The preparation does not establish that all domains are independent between sources.

Accuracy is the share of all URLs classified correctly. Phishing recall is the share of phishing URLs detected. Phishing precision is the share of phishing predictions that are actually phishing. The false-alarm rate is the share of legitimate URLs classified as phishing. ROC-AUC measures how well scores rank phishing URLs above legitimate ones across thresholds; 0.5 is chance-level ranking. High accuracy alone is not sufficient for this application, since both missed attacks and false alarms matter.

## Evaluation 1: original held-out test

| Algorithm | Accuracy | Phishing precision | Phishing recall | Phishing F1 | Legitimate false-alarm rate | Phishing ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 99.31% | 99.94% | 98.44% | 99.19% | 0.04% | 0.995 |
| Random forest | **99.57%** | 99.90% | **99.09%** | **99.49%** | 0.07% | 0.998 |
| XGBoost | 99.57% | 99.90% | 99.08% | 99.49% | 0.07% | **0.998** |

Random forest and XGBoost were almost tied. At greater precision, their accuracies were 99.5688% and 99.5666%, respectively. The random forest detected 19,921 phishing URLs and missed 183; XGBoost detected 19,919 and missed 185. Logistic regression missed 313 phishing URLs but produced only 12 false alarms, compared with 20 for random forest and 19 for XGBoost. Thus, random forest was narrowly best for accuracy and phishing recall on this test, while logistic regression had fewer false alarms. These small differences should not be treated as proof of a practically meaningful advantage without further statistical analysis.

## Evaluation 2: external explicit-protocol test

| Algorithm | Accuracy | Phishing precision | Phishing recall | Phishing F1 | Legitimate false-alarm rate | Phishing ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 35.97% | 36.06% | 99.31% | 52.91% | 100.00% | 0.377 |
| Random forest | **36.10%** | **36.15%** | **99.66%** | **53.05%** | 100.00% | **0.416** |
| XGBoost | **36.10%** | **36.15%** | **99.66%** | **53.05%** | 100.00% | 0.251 |

All three models classified all 35,065 legitimate URLs as phishing. The random forest and XGBoost each detected 19,849 of 19,917 phishing URLs, but their high recall came with a 100% false-alarm rate. Always predicting phishing would achieve approximately 36.22% accuracy on this particular test set, slightly higher than any of the three models. Consequently, the 36% accuracy results do not support using these models in practice on this dataset. The ROC-AUC values below 0.5 also indicate poor ranking on this subset; selecting a different threshold alone should not be assumed to solve the underlying problem.

## Comparison and interpretation

Random forest performed marginally best overall on the original test, with XGBoost nearly identical. It also achieved the highest external ROC-AUC of the three, but all three failed to distinguish legitimate URLs usefully at their default thresholds. The evidence therefore does **not** identify an algorithm that is reliable across both datasets. The original split supports a narrow in-dataset comparison, while the external evaluation reveals a major generalization problem.

The protocol distribution differs sharply between these test sets. In the original test, all 26,970 legitimate URLs use HTTPS. In the external subset, 33,258 of 35,065 legitimate URLs use HTTP. This difference could contribute to the false alarms because HTTPS is an input feature, but these observations do not establish it as the sole cause; datasets can also differ in URL structure, collection methods, and labeling. In addition, the external results apply only to the explicit-protocol subset.

A separate experimental domain-trained random forest achieved 89.46% accuracy, 98.21% phishing recall, and a 16.31% legitimate false-alarm rate on 6,478 URLs in the existing domain-separated pilot split. Those figures are recorded in `domain_model_results.json`. They are not directly comparable with the table above: this model uses different training data, and the pilot split comes from the external dataset. The split's creation and dataset provenance require further documentation before presenting it as a final independent evaluation. It should not be evaluated on the full external dataset as though that dataset were untouched, because its training data came from the same source.

A useful next experiment would train all candidate algorithms on the same documented training data, tune them on a separate validation set, and compare them on an untouched domain-separated test set from a different source. Report both phishing recall and legitimate false alarms, check for domain and URL overlap, and investigate the effect of protocol and other dataset differences.

## Reproducing the comparison

Run `python compare_models.py` from the project directory after installing the dependencies, including XGBoost. The script reads the saved models and writes `results/algorithm_comparison.csv`, including confusion-matrix counts. It does not retrain the models or open any websites. Results were calculated with scikit-learn 1.6.1, XGBoost 2.1.4, NumPy 2.0.2, and pandas 2.3.3. Saved-model compatibility can depend on library versions; the exact versions originally used to train all saved models are not documented.

## Follow-up experiment: training with two data sources

To improve performance across sources, a subsequent experiment trained two new candidates using both the original URL training split and the external domain training split. A conservative grouping rule based on the last two hostname labels removed training rows whose groups appeared in either test set or the external validation set. This rule is not a public-suffix parser: it can group unrelated domains such as those ending in `co.uk` together, and therefore its grouping is approximate. After filtering, training contained 128,611 original-source URLs and 34,841 external-source URLs; the sources received equal total training weight. The candidates were a random forest and histogram gradient boosting. The candidate with higher phishing F1 on the 7,618-row external validation split was selected, before computing the test results below. Gradient boosting won validation selection (F1 96.10%, versus 95.32% for the mixed random forest).

| Evaluation set | Model | Accuracy | Phishing precision | Phishing recall | Phishing F1 | Legitimate false-alarm rate |
|---|---|---:|---:|---:|---:|---:|
| Original URL test (47,074 URLs) | Original forest | 99.57% | 99.90% | 99.09% | 99.49% | 0.07% |
| Original URL test | Mixed gradient boosting | 99.40% | 99.84% | 98.74% | 99.29% | 0.11% |
| External domain pilot (6,478 URLs) | Original forest | 39.43% | 39.54% | 99.26% | 56.55% | 100.00% |
| External domain pilot | Domain-trained forest | 89.46% | 79.87% | 98.21% | 88.09% | 16.31% |
| External domain pilot | Mixed gradient boosting | 84.38% | 72.49% | 97.78% | 83.26% | 24.46% |

The mixed model improved on the original forest's external pilot accuracy by 44.95 percentage points and reduced its false-alarm rate from 100% to 24.46%, while retaining 99.40% accuracy on the original test. However, it was slightly worse than the original forest on that original test and worse than the domain-trained forest on the external pilot. On the pilot, it missed 57 of 2,573 phishing URLs and flagged 955 of 3,905 legitimate URLs. These errors remain substantial, so the model should not be presented as a dependable safety decision.

The 84.38% figure is for the **external pilot subset**, not the earlier 54,982-row external test. The full external dataset cannot serve as a test for the new model because its training uses data from that source. The pilot and original tests were also examined in previous experiments, so this follow-up is exploratory rather than a fresh, untouched final evaluation. Moreover, the original forest and domain-trained forest were trained under different procedures and do not all have the same domain-overlap controls; the table is useful for understanding observed behavior but is not a controlled comparison of algorithms alone. The earlier 89.46% result from the domain-trained forest was also observed before this experiment, so it is not an independent final confirmation.

To reproduce this experiment, run `python improve_generalization.py`. It saves the selected model to `models/mixed_source_selected.joblib`, the detailed counts to `results/mixed_model_comparison.csv`, and validation results and training counts to `results/mixed_model_experiment.json`. A final claim about generalization requires a new, independently collected test set, documented label definitions, and strict source and domain separation before model selection.
