"""PHQ-9 scoring.

Fixes vs. the original notebook:
* keeps ONE survey wave (default ``post``) — the original kept pre *and* post,
  which duplicated every day of data in the inner merge;
* unanswered items are no longer scored as 0; users with too many gaps are dropped
  and users with a single gap are prorated (standard practice);
* the label threshold is configurable and recorded in the results.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SCORE_MAP = {
    "not at all": 0,
    "several days": 1,
    "more than half the days": 2,
    "nearly every day": 3,
}
N_ITEMS = 9


def score_phq9(
    phq: pd.DataFrame,
    survey_wave: str = "post",
    threshold: int = 10,
    max_missing_items: int = 1,
) -> pd.DataFrame:
    """Return one row per uid: ``uid, phq9_total, n_missing, label``."""
    wave = phq[phq["type"].astype(str).str.strip().str.lower() == survey_wave.lower()].copy()
    if wave.empty:
        raise ValueError(f"No PHQ-9 rows for survey wave {survey_wave!r}")

    item_cols = [c for c in wave.columns if c not in {"uid", "type"}][:N_ITEMS]
    if len(item_cols) != N_ITEMS:
        raise ValueError(f"Expected {N_ITEMS} PHQ-9 item columns, found {len(item_cols)}")

    items = wave[item_cols].apply(lambda s: s.astype(str).str.strip().str.lower().map(SCORE_MAP))
    wave["n_missing"] = items.isna().sum(axis=1)
    answered = N_ITEMS - wave["n_missing"]
    # Prorate: mean of answered items * 9 (only applied when <= max_missing_items are missing).
    wave["phq9_total"] = np.where(answered > 0, items.mean(axis=1) * N_ITEMS, np.nan)

    dropped = wave[wave["n_missing"] > max_missing_items]
    if len(dropped):
        log.warning("Dropping %d users with > %d unanswered PHQ-9 items", len(dropped), max_missing_items)
    wave = wave[wave["n_missing"] <= max_missing_items]

    dupes = wave["uid"].duplicated(keep="last")
    if dupes.any():
        log.warning("Found %d duplicate %s submissions; keeping the last one per uid", dupes.sum(), survey_wave)
    wave = wave[~dupes]

    wave["phq9_total"] = wave["phq9_total"].round(1)
    wave["label"] = (wave["phq9_total"] >= threshold).astype(int)
    return wave[["uid", "phq9_total", "n_missing", "label"]].reset_index(drop=True)
