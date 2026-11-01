"""End-to-end experiment: data -> features -> subject-level CV -> report."""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sleeprisk import __version__
from sleeprisk.data.dataset import build_dataset, dataset_summary
from sleeprisk.evaluation import plots
from sleeprisk.evaluation.cv import CVResult, leaky_replica, permutation_test, subject_cv
from sleeprisk.models.baselines import make_logistic, make_majority, make_random_forest
from sleeprisk.windows import AGG_FEATURES, build_user_table, build_windows

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def _factories(cfg: dict) -> dict:
    m = cfg["models"]
    fac = {
        "majority": (lambda s: make_majority(s), "user"),
        "logistic_user": (lambda s: make_logistic(m["logistic_user"]["C"], s), "user"),
        "random_forest": (lambda s: make_random_forest(**m["random_forest"], seed=s), "agg"),
    }
    if "lstm" in m["run"]:
        from sleeprisk.models.lstm import LSTMClassifier

        fac["lstm"] = (lambda s: LSTMClassifier(**m["lstm"], seed=s), "seq")
    return {k: v for k, v in fac.items() if k in m["run"]}


def run(cfg: dict) -> dict:
    t0 = time.time()
    set_seed(cfg["seed"])
    out = Path(cfg["data"]["output_dir"])
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # 1. Data ---------------------------------------------------------------------
    ds = build_dataset(cfg)
    summary = dataset_summary(ds)
    w = cfg["windows"]
    windows = build_windows(ds.daily, ds.labels, w["length_days"], w["stride_days"], w["min_valid_days"])
    users = build_user_table(ds.daily, ds.labels)
    summary.update(
        {"n_windows": len(windows), "windows_per_user_median": float(pd.Series(windows.groups).value_counts().median())}
    )
    log.info("Dataset: %s", summary)

    # 2. Subject-level CV ----------------------------------------------------------
    cv = cfg["cv"]
    results: dict[str, CVResult] = {}
    lstm_histories: list = []
    lr_coefs: list = []
    for name, (factory, kind) in _factories(cfg).items():
        log.info("Subject-level CV: %s", name)
        hook = None
        if name == "lstm":
            hook = lambda model, r, f: lstm_histories.append(model.history_)  # noqa: E731
        elif name == "logistic_user":
            hook = lambda model, r, f: lr_coefs.append(model.named_steps["clf"].coef_[0])  # noqa: E731
        results[name] = subject_cv(
            name, factory, kind, users, windows, cv["n_splits"], cv["n_repeats"], cfg["seed"], cv["bootstrap"], hook
        )

    # 3. Leaky replica (measures how much the original protocol inflated scores) ---
    leaky = []
    if cfg["leaky_replica"]["enabled"]:
        facs = _factories(cfg)
        for name in ("random_forest", "lstm"):
            if name in facs:
                factory, kind = facs[name]
                log.info("Leaky replica: %s", name)
                leaky.append(
                    leaky_replica(name, factory, kind, windows, cfg["leaky_replica"]["test_size"], cfg["seed"])
                )

    # 4. Permutation test on the primary (user-level) model -------------------------
    perm = None
    if cv["n_permutations"] and "logistic_user" in results:
        log.info("Permutation test (%d permutations)", cv["n_permutations"])
        perm = permutation_test(
            _factories(cfg)["logistic_user"][0],
            users,
            results["logistic_user"].summary["auroc"]["mean"],
            cv["n_permutations"],
            cv["n_splits"],
            cv["n_repeats"],
            cfg["seed"],
        )

    # 5. Persist -------------------------------------------------------------------
    figs = {}
    honest = {k: v.summary for k, v in results.items()}
    if leaky:
        figs["leakage"] = str(plots.leakage_bar(leaky, honest, out / "figures" / "leaky_vs_subject_cv.png"))
    figs["roc"] = str(plots.roc_user_level(results, out / "figures" / "roc_user_level.png"))
    if perm:
        figs["permutation"] = str(plots.permutation_hist(perm, out / "figures" / "permutation_null.png"))
    if lstm_histories:
        figs["lstm"] = str(
            plots.lstm_curves(lstm_histories[: cv["n_splits"]], out / "figures" / "lstm_learning_curves.png")
        )

    coef = None
    if lr_coefs:
        c = np.vstack(lr_coefs)
        coef = (
            pd.DataFrame({"feature": AGG_FEATURES, "coef_mean": c.mean(0), "coef_std": c.std(0)})
            .assign(abs_mean=lambda d: d["coef_mean"].abs())
            .sort_values("abs_mean", ascending=False)
            .drop(columns="abs_mean")
        )
        coef.to_csv(out / "logistic_coefficients.csv", index=False)

    preds = pd.concat([r.oof_user.assign(model=k) for k, r in results.items()])
    preds.to_csv(out / "user_predictions.csv", index=False)
    users.to_csv(out / "user_features.csv", index=False)

    metrics = {
        "version": __version__,
        "runtime_s": round(time.time() - t0, 1),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "config": cfg,
        "dataset": summary,
        "subject_cv": {
            k: {"summary": r.summary, "ci": r.ci, "window_level": r.window_level} for k, r in results.items()
        },
        "leaky_replica": leaky,
        "permutation_test": {k: v for k, v in (perm or {}).items() if k != "null_distribution"} or None,
        "figures": figs,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "report.md").write_text(render_report(metrics, coef))
    log.info("Done in %.1fs -> %s", time.time() - t0, out)
    return metrics


def _fmt(s: dict, key: str) -> str:
    return f"{s[key]['mean']:.3f} ± {s[key]['std']:.3f}"


def render_report(m: dict, coef: pd.DataFrame | None) -> str:
    d = m["dataset"]
    lines = [
        "# Results report",
        "",
        f"_Auto-generated by `sleeprisk` v{m['version']} in {m['runtime_s']} s._",
        "",
        "## Dataset",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Users (labelled, with sensing) | {d['n_users']} ({d['n_positive_users']} positive / {d['n_negative_users']} negative) |",
        f"| Label | PHQ-9 ({m['config']['labels']['survey_wave']} survey) ≥ {m['config']['labels']['threshold']} |",
        f"| Calendar days / valid sleep days | {d['calendar_days']} / {d['valid_sleep_days']} |",
        f"| Windows ({m['config']['windows']['length_days']}-day) | {d['n_windows']} (median {d['windows_per_user_median']:.0f} per user) |",
        f"| Missing rate: screen / conversation | {d['screen_missing_rate']:.1%} / {d['conversation_missing_rate']:.1%} |",
        f"| Date range | {d['date_range'][0]} → {d['date_range'][1]} |",
        f"| Files loaded / failed | {d['files_loaded']} / {d['files_failed'] or 'none'} |",
        "",
        "## Subject-level cross-validation (corrected protocol)",
        "",
        f"{m['config']['cv']['n_repeats']}× repeated {m['config']['cv']['n_splits']}-fold CV over **users**; "
        "metrics computed per user (mean predicted probability over the user's windows). Mean ± std over repeats; "
        "95% CI from bootstrapping users on the repeat-averaged predictions.",
        "",
        "| Model | AUROC | 95% CI | Balanced acc. | Sensitivity | Specificity |",
        "|---|---|---|---|---|---|",
    ]
    for k, r in m["subject_cv"].items():
        s, ci = r["summary"], r["ci"]["auroc_ci"]
        lines.append(
            f"| {k} | {_fmt(s, 'auroc')} | [{ci[0]:.2f}, {ci[1]:.2f}] | {_fmt(s, 'balanced_accuracy')} | "
            f"{_fmt(s, 'sensitivity')} | {_fmt(s, 'specificity')} |"
        )
    lines += [
        "",
        "> With few users, pooled out-of-fold AUROC is noisy and can fall *below* 0.5 for models with no real "
        "signal (each held-out fold shifts the training-fold class balance the other way). The majority baseline "
        "shows the size of this effect; the permutation test below is the significance check.",
    ]
    if m["leaky_replica"]:
        lines += [
            "",
            "## Leakage audit: original protocol re-run on the same data",
            "",
            "| Model | Window AUROC (leaky) | Window F1-weighted (leaky) | Window AUROC (subject CV) | Test users also in train | Test windows from seen users |",
            "|---|---|---|---|---|---|",
        ]
        for r in m["leaky_replica"]:
            wl = m["subject_cv"].get(r["model"], {}).get("window_level") or {}
            honest = f"{wl['auroc']['mean']:.3f}" if wl else "—"
            lines.append(
                f"| {r['model']} | {r['auroc']:.3f} | {r['f1_weighted']:.3f} | {honest} | "
                f"{r['test_users_also_in_train']}/{r['test_users_total']} | {r['test_windows_from_seen_users_pct']:.0f}% |"
            )
    p = m["permutation_test"]
    if p:
        lines += [
            "",
            "## Permutation test (logistic_user)",
            "",
            f"Observed mean AUROC **{p['observed_auroc']:.3f}** vs. shuffled-label null mean {p['null_mean']:.3f} "
            f"(95th percentile {p['null_95th']:.3f}); **p = {p['p_value']:.3f}** "
            f"({m['config']['cv']['n_permutations']} permutations).",
        ]
    if coef is not None:
        lines += [
            "",
            "## Logistic regression coefficients (standardised, mean ± std across folds)",
            "",
            "| Feature | Coefficient |",
            "|---|---|",
        ]
        lines += [f"| {r.feature} | {r.coef_mean:+.3f} ± {r.coef_std:.3f} |" for r in coef.itertuples()]
    lines += ["", "## Figures", ""]
    lines += [
        f"![{k}]({Path(v).relative_to(Path(m['config']['data']['output_dir'])).as_posix()})"
        for k, v in m["figures"].items()
    ]
    return "\n".join(lines) + "\n"
