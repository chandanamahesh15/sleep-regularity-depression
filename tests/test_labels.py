import numpy as np
import pandas as pd
import pytest

from sleeprisk.labels import score_phq9

A = ["Not at all", "Several days", "More than half the days", "Nearly every day"]


def _row(uid, wave, scores, missing_idx=None):
    items = [A[s] for s in scores]
    if missing_idx is not None:
        items[missing_idx] = np.nan
    return [uid, wave, *items, "x"]


def _phq(rows):
    return pd.DataFrame(rows, columns=["uid", "type", *[f"q{i}" for i in range(9)], "Response"])


def test_keeps_one_wave_and_one_row_per_user():
    # Regression test for the original bug: pre + post rows duplicated every day in the merge.
    phq = _phq([_row("u00", "pre", [0] * 9), _row("u00", "post", [2] * 9), _row("u01", "pre", [3] * 9)])
    out = score_phq9(phq, survey_wave="post", threshold=10)
    assert out["uid"].tolist() == ["u00"]
    assert out.loc[0, "phq9_total"] == 18
    assert out.loc[0, "label"] == 1


def test_scoring_ignores_response_column_and_case():
    phq = _phq([_row("u00", "POST ", [1, 1, 1, 1, 1, 0, 0, 0, 0])])
    assert score_phq9(phq, "post", threshold=5).loc[0, "phq9_total"] == 5


def test_missing_items_are_prorated_not_zeroed():
    phq = _phq([_row("u00", "post", [2] * 9, missing_idx=0)])
    out = score_phq9(phq, "post", threshold=10, max_missing_items=1)
    assert out.loc[0, "phq9_total"] == 18  # 8 answered items of 2 -> mean 2 * 9


def test_too_many_missing_items_drops_user():
    row = _row("u00", "post", [2] * 9, missing_idx=0)
    row[3] = np.nan
    assert score_phq9(_phq([row, _row("u01", "post", [0] * 9)]), "post", max_missing_items=1)["uid"].tolist() == ["u01"]


def test_unknown_wave_raises():
    with pytest.raises(ValueError):
        score_phq9(_phq([_row("u00", "pre", [0] * 9)]), "post")
