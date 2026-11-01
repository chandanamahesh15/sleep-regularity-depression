# Audit of the original notebook (`notebooks/sleep_regularity_v1_leakage.ipynb`)

**Overall: 4/10 on organisation, 2/10 on methodological soundness.**
The notebook *reads* well (numbered sections, docstrings, a clear story arc),
but it is a single linear script where correctness depends on hidden state,
silent error handling and an evaluation protocol that leaks.

## Scorecard

| Dimension | Score | Notes |
|---|---|---|
| Narrative structure | 7/10 | Clear sections 1–7, markdown headers, logical order. |
| Modularity | 3/10 | Feature functions exist, but augmentation, splitting, training and evaluation are top-level code; nothing reusable or importable. |
| Reproducibility | 3/10 | `!wget` inside notebook, Kaggle-specific path, no pinned versions, no global seed for NumPy/PyTorch, no saved config. |
| Error handling | 2/10 | `except: continue` in every loader — failed files vanish silently. |
| Data validation | 1/10 | No checks for duplicate keys, label uniqueness per user, or date coverage. The duplicated rows are visible in Table 2 but not caught. |
| Evaluation design | 1/10 | Random window-level split → subject leakage; single split; no validation set; no CI; no trivial baseline. |
| Claims vs evidence | 2/10 | Printed "insight" contradicts the correlation matrix; "Innovation Validated" based on a leaky 0.04 F1 gap. |
| Testing | 0/10 | No tests. |
| Deployability | 0/10 | No package, CLI, config, container or CI. |

## Structural problems (in notebook order)

1. **Setup** – unused imports (`confusion_matrix`, `datetime`), config constants mixed with library imports, data download embedded in analysis.
2. **Pipeline** – three loaders with copy-pasted file-globbing logic; UTC used as local time; filenames parsed with string `replace`; a column name with a leading space (`' end_timestamp'`) hard-coded.
3. **Labels** – both `pre` and `post` PHQ-9 rows kept; missing answers silently scored as 0.
4. **Merge** – `inner` join on `uid` against a two-rows-per-user label table duplicates every day.
5. **EDA** – done on the merged (duplicated) table; conclusions drawn before any modelling evaluation.
6. **Augmentation** – windows built by row index, not calendar day, so a "7-day window" can span weeks when days are missing.
7. **Split** – `train_test_split` on windows; RF split kept in sync only by coincidence of identical `random_state`.
8. **LSTM** – fixed 100 epochs, no validation, no early stopping, unweighted loss, no seed.
9. **Comparison** – one bar chart of two leaky numbers with an arbitrary 0.70 "success threshold".

## Target structure (implemented in this repo)

```
configs/default.yaml        all knobs in one place
src/sleeprisk/
  data/loaders.py           raw StudentLife readers with logged failures
  data/validation.py        schema + invariant checks (no dup keys, 1 label/user)
  features/sleep.py         local-time episodes, circular midpoint, true SRI
  features/context.py       screen + conversation, missingness flags
  labels.py                 PHQ-9 scoring, single survey wave, completeness rule
  windows.py                calendar-day windows that carry their uid
  models/                   majority, logistic, RF, LSTM (early stopping)
  evaluation/               grouped CV, leaky-protocol replica, metrics, permutation test
  pipeline.py, cli.py       one command end to end
tests/                      leakage guards + unit tests
notebooks/                  thin narrative notebook that imports the package
```
