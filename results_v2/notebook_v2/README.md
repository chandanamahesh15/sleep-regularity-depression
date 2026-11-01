# Results of the v2 notebook

Unzip `results_v2.zip` (downloaded by Cell 18 of the v2 notebook) into this folder:

| File | Contents |
|---|---|
| `subject_cv_summary.csv` | AUROC, balanced accuracy, sensitivity, specificity per model (subject-level CV) |
| `leaky_vs_subject.csv` | Same random forest: v1 random split vs v2 subject CV |
| `permutation_tests.csv` | Observed AUROC and permutation p-value for each test |
| `logistic_coefficients.csv` | Standardised coefficients, mean ± std across folds |
| `user_features.csv` | One row per student: term-level features, PHQ-9 score and label |
| `headline_metrics.json` | The headline numbers used by `scripts/make_figures.py` (already included) |
