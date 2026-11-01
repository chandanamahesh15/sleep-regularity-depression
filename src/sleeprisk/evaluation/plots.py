"""Static figures for the results report (matplotlib, no seaborn dependency)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def leakage_bar(leaky: list[dict], honest: dict, path: Path) -> Path:
    """Same models, same data: leaky protocol vs subject-level protocol (AUROC)."""
    models = [r["model"] for r in leaky if r["model"] in honest]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 4))
    a = [r["auroc"] for r in leaky if r["model"] in honest]
    b = [honest[m]["auroc"]["mean"] for m in models]
    e = [honest[m]["auroc"]["std"] for m in models]
    ax.bar(x - 0.18, a, 0.34, color=ORANGE, label="Original protocol (random window split)")
    ax.bar(x + 0.18, b, 0.34, yerr=e, color=BLUE, capsize=3, ecolor=INK_2, label="Corrected (subject-level CV)")
    for xi, v in zip(x - 0.18, a):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=9)
    for xi, v in zip(x + 0.18, b):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=9)
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1)
    ax.text(len(models) - 0.5, 0.51, "chance", color=INK_2, fontsize=8, ha="right")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC", color=INK_2)
    ax.set_title("Same data, same models — only the split changes", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_user_level(results: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = [BLUE, ORANGE, "#1baf7a", "#4a3aa7"]
    for (name, res), c in zip(results.items(), colors):
        d = res.oof_user.dropna()
        if d["label"].nunique() < 2 or name == "majority":
            continue
        fpr, tpr, _ = roc_curve(d["label"], d["prob"])
        ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name} ({res.summary['auroc']['mean']:.2f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate", color=INK_2)
    ax.set_ylabel("True positive rate", color=INK_2)
    ax.set_title("User-level ROC (out-of-fold)", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def permutation_hist(perm: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(perm["null_distribution"], bins=15, color=GRAY, edgecolor="white")
    ax.axvline(perm["observed_auroc"], color=BLUE, linewidth=2)
    ax.text(
        perm["observed_auroc"],
        ax.get_ylim()[1] * 0.92,
        f"  observed {perm['observed_auroc']:.2f}\n  p = {perm['p_value']:.3f}",
        color=INK,
        fontsize=9,
        va="top",
    )
    ax.set_xlabel("Mean AUROC with shuffled labels", color=INK_2)
    ax.set_ylabel("Permutations", color=INK_2)
    ax.set_title("Is the signal distinguishable from chance?", color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def lstm_curves(histories: list[list[dict]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, h in enumerate(histories):
        ep = [r["epoch"] for r in h]
        ax.plot(
            ep, [r["train_loss"] for r in h], color=ORANGE, alpha=0.35, linewidth=1, label="train" if i == 0 else None
        )
        ax.plot(
            ep,
            [r["val_loss"] for r in h],
            color=BLUE,
            alpha=0.6,
            linewidth=1.5,
            label="validation (unseen users)" if i == 0 else None,
        )
    ax.set_xlabel("Epoch", color=INK_2)
    ax.set_ylabel("Weighted BCE loss", color=INK_2)
    ax.set_title("LSTM learning curves, one line per CV fold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
