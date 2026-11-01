"""Evaluation protocols.

``subject_cv``    – the corrected protocol: folds are formed over USERS, every
                    window of a user lives on one side of the split, and scores
                    are computed per user (mean predicted probability over that
                    user's windows). Repeated with different seeds.
``leaky_replica`` – a faithful re-run of the original notebook's protocol
                    (random 80/20 split over overlapping windows) so the size of
                    the leakage can be measured on the same data.
``permutation_test`` – label-permutation null distribution at the user level.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from sleeprisk.data.validation import assert_disjoint_groups
from sleeprisk.evaluation.metrics import binary_metrics, bootstrap_ci, summarise
from sleeprisk.windows import AGG_FEATURES, WindowSet

ModelFactory = Callable[[int], object]


@dataclass
class CVResult:
    name: str
    per_repeat: list[dict]
    summary: dict
    oof_user: pd.DataFrame  # uid, label, prob (mean over repeats)
    ci: dict
    window_level: dict | None = None  # window-level metrics under grouped CV
    extras: dict = field(default_factory=dict)


def _fit_predict(model, kind: str, X_tr, y_tr, g_tr, X_te):
    if kind == "seq":
        model.fit(X_tr, y_tr, g_tr)
    else:
        model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def subject_cv(
    name: str,
    factory: ModelFactory,
    kind: str,  # "user" (user table) | "agg" (window aggregates) | "seq" (window sequences)
    users: pd.DataFrame,  # uid, label (+ features when kind == "user")
    windows: WindowSet | None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
    n_boot: int = 1000,
    on_fold: Callable[[object, int, int], None] | None = None,
) -> CVResult:
    uids, y_user = users["uid"].to_numpy(), users["label"].to_numpy()
    n_splits = min(n_splits, int(np.bincount(y_user).min()))
    if n_splits < 2:
        raise ValueError("Need at least 2 users in each class for cross-validation")

    per_repeat, user_probs, win_runs = [], [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = pd.Series(np.nan, index=uids)
        win_true, win_prob = [], []
        for f, (tr_u, te_u) in enumerate(skf.split(uids, y_user)):
            train_users, test_users = uids[tr_u], uids[te_u]
            assert_disjoint_groups(train_users, test_users, f"{name} repeat {r} fold {f}")
            model = factory(seed + 1000 * r + f)

            if kind == "user":
                feats = [c for c in AGG_FEATURES if c in users.columns]
                Xu = users[feats].to_numpy(dtype=float)
                oof.iloc[te_u] = _fit_predict(model, kind, Xu[tr_u], y_user[tr_u], train_users, Xu[te_u])
            else:
                assert windows is not None
                tr_mask = np.isin(windows.groups, train_users)
                te_mask = np.isin(windows.groups, test_users)
                assert_disjoint_groups(windows.groups[tr_mask], windows.groups[te_mask], f"{name} windows")
                X = windows.X_seq if kind == "seq" else windows.X_agg.to_numpy(dtype=float)
                p = _fit_predict(model, kind, X[tr_mask], windows.y[tr_mask], windows.groups[tr_mask], X[te_mask])
                win_true.append(windows.y[te_mask])
                win_prob.append(p)
                per_user = pd.Series(p).groupby(windows.groups[te_mask]).mean()
                oof.loc[per_user.index] = per_user.to_numpy()
            if on_fold:
                on_fold(model, r, f)

        scored = oof.notna().to_numpy()
        per_repeat.append(binary_metrics(y_user[scored], oof.to_numpy()[scored]))
        user_probs.append(oof)
        if win_true:
            win_runs.append(binary_metrics(np.concatenate(win_true), np.concatenate(win_prob)))

    mean_prob = pd.concat(user_probs, axis=1).mean(axis=1)
    oof_user = pd.DataFrame({"uid": uids, "label": y_user, "prob": mean_prob.reindex(uids).to_numpy()})
    scored = oof_user["prob"].notna()
    ci = bootstrap_ci(oof_user.loc[scored, "label"], oof_user.loc[scored, "prob"], n_boot, seed)
    return CVResult(
        name=name,
        per_repeat=per_repeat,
        summary=summarise(per_repeat),
        oof_user=oof_user,
        ci=ci,
        window_level=summarise(win_runs) if win_runs else None,
    )


def leaky_replica(
    name: str, factory: ModelFactory, kind: str, windows: WindowSet, test_size: float = 0.2, seed: int = 42
) -> dict:
    """Re-create the original protocol: random stratified split over overlapping windows."""
    idx = np.arange(len(windows))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=windows.y)
    X = windows.X_seq if kind == "seq" else windows.X_agg.to_numpy(dtype=float)
    model = factory(seed)
    # Pass the window index as the "group" so the LSTM's inner split is also window-random,
    # exactly like the original (which had no grouping anywhere).
    p = _fit_predict(model, kind, X[tr], windows.y[tr], idx[tr], X[te])
    shared = set(windows.groups[tr]) & set(windows.groups[te])
    out = binary_metrics(windows.y[te], p)
    out.update(
        {
            "model": name,
            "test_users_also_in_train": len(set(windows.groups[te]) & shared),
            "test_users_total": int(len(set(windows.groups[te]))),
            "test_windows_from_seen_users_pct": float(np.isin(windows.groups[te], list(shared)).mean() * 100),
        }
    )
    return out


def permutation_test(
    factory: ModelFactory,
    users: pd.DataFrame,
    observed_auroc: float,
    n_permutations: int = 50,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """Shuffle user labels; refit the full CV each time. p = (1 + #null >= obs) / (1 + n)."""
    rng = np.random.default_rng(seed)
    null = []
    for i in range(n_permutations):
        shuffled = users.assign(label=rng.permutation(users["label"].to_numpy()))
        res = subject_cv("perm", factory, "user", shuffled, None, n_splits, n_repeats, seed + 7919 * (i + 1), n_boot=0)
        null.append(res.summary["auroc"]["mean"])
    null = np.array(null)
    return {
        "observed_auroc": float(observed_auroc),
        "null_mean": float(null.mean()),
        "null_95th": float(np.percentile(null, 95)),
        "p_value": float((1 + (null >= observed_auroc).sum()) / (1 + n_permutations)),
        "null_distribution": null.round(4).tolist(),
    }
