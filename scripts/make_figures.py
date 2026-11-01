"""Draw the README / case-study figures from results/notebook_v2/headline_metrics.json.

Usage: python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results/notebook_v2/headline_metrics.json").read_text())
OUT = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(colors=INK2, length=0)


def leakage_audit():
    cv = M["subject_cv"]
    bars = [
        ("Random forest, v1 random split", M["leakage_audit_rf"]["v1_random_window_split_auroc"], True),
        ("Random forest, subject CV", cv["random_forest"]["auroc"], False),
        ("LSTM, subject CV", cv["lstm"]["auroc"], False),
        ("Logistic, subject CV", cv["logistic_user"]["auroc"], False),
        ("Majority baseline, subject CV", cv["majority"]["auroc"], False),
    ]
    perm = next(t for t in M["permutation_tests"] if t["test"].startswith("random_forest"))["null_95th"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (_name, v, leaky) in enumerate(bars):
        ax.barh(i, v, color=BLUE if leaky else GREY, height=0.6)
        ax.text(1.02, i, f"{v:.2f}", va="center", color=INK, fontweight="bold" if leaky else "normal")
    ax.set_yticks(range(len(bars)), [b[0] for b in bars], color=INK)
    ax.invert_yaxis()
    for x, label, ha, dx in [
        (0.5, "chance 0.50", "right", -0.01),
        (perm, f"RF permutation 95th pct {perm:.2f}", "left", 0.01),
    ]:
        ax.axvline(x, ls="--", color=INK2, lw=1.2)
        ax.text(x + dx, -0.75, label, ha=ha, color=INK2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.suptitle(
        "Same model, same data: AUROC falls from 0.93 to 0.68 once students are kept apart",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "leakage_audit.png", dpi=160)
    plt.close(fig)


def feature_correlations():
    names = {
        "duration_mean": "Mean sleep duration",
        "screen_missing_frac": "Days without screen data",
        "duration_std": "Sleep duration variability",
        "sleep_valid_frac": "Nights with detected sleep",
        "screen_log_mean": "Screen use",
        "conversation_missing_frac": "Days without conversation data",
        "midpoint_std": "Sleep midpoint variability",
        "onset_std": "Bedtime variability",
        "conversation_log_mean": "Conversation time",
        "midpoint_mean": "Mean sleep midpoint",
        "sri": "Sleep Regularity Index (SRI)",
    }
    r = M["spearman_with_phq9"]
    n = M["dataset"]["students_labelled"]
    crit = 2 / n**0.5
    keys = list(r)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvspan(-crit, crit, color=GREY, alpha=0.25, lw=0)
    ax.text(0, -0.9, f"not significant: |r| < {crit:.2f} (n = {n})", ha="center", color=INK2, fontsize=9)
    for i, k in enumerate(keys):
        v = r[k]
        ax.barh(i, v, color=BLUE if k == "sri" else GREY, height=0.6)
        ax.text(
            v + (0.01 if v >= 0 else -0.01),
            i,
            f"{v:+.2f}".replace("-", "−"),
            va="center",
            ha="left" if v >= 0 else "right",
            color=INK,
            fontweight="bold" if (k == "sri" or abs(v) >= crit) else "normal",
        )
    ax.set_yticks(range(len(keys)), [names[k] for k in keys], color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-0.5, 0.5)
    fig.suptitle(
        "Sleep regularity shows no link to PHQ-9; only mean sleep duration passes the significance line",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "feature_correlations.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    leakage_audit()
    feature_correlations()
    print("Wrote", sorted(p.name for p in OUT.glob("*.png")))
