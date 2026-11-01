"""Calendar-day windows and user-level aggregates.

Fixes vs. the original notebook:
* windows span *calendar* days (the original sliced rows, so a "7-day" window
  could cover weeks when days were missing);
* every window carries its ``uid`` so the splitter can group by subject;
* windows with too little sleep data are dropped instead of zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sleeprisk.features.sleep import sri_from_agreement

SEQ_FEATURES = ["duration_h", "midpoint_h", "sri_agree", "screen_log", "conversation_log"]


@dataclass
class WindowSet:
    X_agg: pd.DataFrame  # (n_windows, n_agg_features) for tree / linear models
    X_seq: np.ndarray  # (n_windows, length, n_seq_features) with NaN = missing
    groups: np.ndarray  # uid per window — THE grouping variable for every split
    y: np.ndarray
    end_dates: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def _with_logs(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.assign(
        screen_log=np.log1p(daily["screen_samples"]),
        conversation_log=np.log1p(daily["conversation_min"]),
    )


def aggregate(block: pd.DataFrame) -> dict:
    """Summary features for a block of days (a window, or a user's whole term)."""
    v = block[block["sleep_valid"]]
    return {
        "duration_mean": v["duration_h"].mean(),
        "duration_std": v["duration_h"].std(ddof=0),
        "midpoint_mean": v["midpoint_h"].mean(),
        "midpoint_std": v["midpoint_h"].std(ddof=0),
        "onset_std": v["onset_h"].std(ddof=0),
        "sri": sri_from_agreement(block["sri_agree"]),
        "screen_log_mean": block["screen_log"].mean(),
        "conversation_log_mean": block["conversation_log"].mean(),
        "sleep_valid_frac": block["sleep_valid"].mean(),
        "screen_missing_frac": block["screen_missing"].mean(),
        "conversation_missing_frac": block["conversation_missing"].mean(),
    }


AGG_FEATURES = [
    "duration_mean",
    "duration_std",
    "midpoint_mean",
    "midpoint_std",
    "onset_std",
    "sri",
    "screen_log_mean",
    "conversation_log_mean",
    "sleep_valid_frac",
    "screen_missing_frac",
    "conversation_missing_frac",
]


def build_windows(
    daily: pd.DataFrame, labels: pd.DataFrame, length: int = 7, stride: int = 1, min_valid_days: int = 4
) -> WindowSet:
    daily = _with_logs(daily).sort_values(["uid", "date"])
    label_of = labels.set_index("uid")["label"]
    aggs, seqs, groups, ys, ends = [], [], [], [], []
    for uid, g in daily.groupby("uid", sort=True):
        g = g.reset_index(drop=True)
        for start in range(0, len(g) - length + 1, stride):
            w = g.iloc[start : start + length]
            if w["sleep_valid"].sum() < min_valid_days:
                continue
            aggs.append(aggregate(w))
            seqs.append(w[SEQ_FEATURES].to_numpy(dtype=np.float32))
            groups.append(uid)
            ys.append(int(label_of[uid]))
            ends.append(w["date"].iloc[-1])
    if not aggs:
        raise ValueError("No windows satisfy min_valid_days; relax windows.min_valid_days")
    return WindowSet(
        X_agg=pd.DataFrame(aggs, columns=AGG_FEATURES),
        X_seq=np.stack(seqs),
        groups=np.array(groups),
        y=np.array(ys, dtype=np.int64),
        end_dates=np.array(ends),
    )


def build_user_table(daily: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per user: term-level aggregates + label (the honest n for this study)."""
    daily = _with_logs(daily)
    rows = [{"uid": uid, **aggregate(g)} for uid, g in daily.groupby("uid", sort=True)]
    return pd.DataFrame(rows).merge(labels[["uid", "label", "phq9_total"]], on="uid", how="inner")
