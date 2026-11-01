"""The tests that would have caught the original project's leakage."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from sleeprisk.data.dataset import build_dataset, merge_daily
from sleeprisk.data.validation import DataValidationError, assert_disjoint_groups, assert_unique_keys
from sleeprisk.evaluation.cv import leaky_replica, subject_cv
from sleeprisk.models.baselines import make_logistic, make_random_forest
from sleeprisk.windows import build_user_table, build_windows


def test_disjoint_guard_raises_on_overlap():
    with pytest.raises(DataValidationError):
        assert_disjoint_groups(["u00", "u01"], ["u01", "u02"])


def test_merge_rejects_duplicate_days():
    day = pd.Timestamp("2013-04-01")
    sleep = pd.DataFrame({"uid": ["u00", "u00"], "date": [day, day]})
    empty = pd.DataFrame({"uid": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
    with pytest.raises(DataValidationError):
        merge_daily(sleep, empty, empty)


def test_dataset_has_unique_days_and_one_label_per_user(cfg):
    ds = build_dataset(cfg)
    assert_unique_keys(ds.daily, ["uid", "date"], "daily")
    assert ds.labels["uid"].is_unique
    assert set(ds.daily["uid"]) == set(ds.labels["uid"])
    assert ds.load_report.failed.get("phonelock"), "corrupt synthetic file should be logged, not swallowed"


def test_every_window_belongs_to_exactly_one_user(cfg):
    ds = build_dataset(cfg)
    ws = build_windows(ds.daily, ds.labels, 7, 1, 4)
    label_of = ds.labels.set_index("uid")["label"]
    assert np.array_equal(ws.y, label_of.loc[ws.groups].to_numpy())
    assert ws.X_seq.shape[1:] == (7, 5)


def test_subject_cv_never_shares_users(cfg):
    ds = build_dataset(cfg)
    ws = build_windows(ds.daily, ds.labels, 7, 1, 4)
    users = build_user_table(ds.daily, ds.labels)
    seen = []

    def spy(model, r, f):
        seen.append((r, f))

    res = subject_cv("rf", lambda s: make_random_forest(50, 4, 2, s), "agg", users, ws, 3, 2, 0, 20, spy)
    assert len(seen) == 6
    assert res.oof_user["uid"].is_unique
    # Re-derive the folds and check disjointness independently of the library's own guard.
    for r in range(2):
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=r).split(users, users["label"]):
            assert not set(users["uid"].iloc[tr]) & set(users["uid"].iloc[te])


def test_leaky_replica_really_leaks(cfg):
    ds = build_dataset(cfg)
    ws = build_windows(ds.daily, ds.labels, 7, 1, 4)
    out = leaky_replica("rf", lambda s: make_random_forest(50, 4, 2, s), "agg", ws, 0.2, 0)
    assert out["test_windows_from_seen_users_pct"] > 90


def test_preprocessing_is_fit_inside_the_fold():
    # Imputer/scaler statistics must come from the training rows only.
    X_train = np.arange(10, dtype=float).reshape(-1, 1)
    y_train = np.r_[np.zeros(5), np.ones(5)]
    m = make_logistic().fit(X_train, y_train)
    assert m.named_steps["scale"].mean_[0] == pytest.approx(4.5)
    m.predict_proba(np.full((5, 1), 100.0))  # scoring test rows must not refit anything
    assert m.named_steps["scale"].mean_[0] == pytest.approx(4.5)
