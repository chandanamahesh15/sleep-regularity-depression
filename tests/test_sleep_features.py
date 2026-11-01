import numpy as np
import pandas as pd
import pytest

from sleeprisk.features.sleep import build_daily_sleep, extract_main_episodes, sri_from_agreement

TZ = "America/New_York"


def _epoch(local: str) -> int:
    return int(pd.Timestamp(local).tz_localize(TZ).tz_convert("UTC").timestamp())


def _locks(pairs):
    return pd.DataFrame([{"uid": "u00", "start": _epoch(a), "end": _epoch(b)} for a, b in pairs])


def test_utc_is_converted_to_local_night():
    # 23:30 EDT on Apr 1 is 03:30 UTC on Apr 2. The night must be Apr 1, midpoint ~03:30 local.
    ep = extract_main_episodes(_locks([("2013-04-01 23:30", "2013-04-02 07:30")]), TZ)
    assert ep.loc[0, "date"] == pd.Timestamp("2013-04-01")
    daily = build_daily_sleep(_locks([("2013-04-01 23:30", "2013-04-02 07:30")]), TZ)
    assert daily.loc[0, "midpoint_h"] == pytest.approx(15.5)  # 03:30 = 15.5 h after noon


def test_short_locks_filtered_and_longest_kept():
    ep = extract_main_episodes(
        _locks(
            [
                ("2013-04-01 14:00", "2013-04-01 15:00"),  # 1 h — not sleep
                ("2013-04-01 23:00", "2013-04-02 06:00"),  # 7 h — main
                ("2013-04-02 08:00", "2013-04-02 11:30"),  # 3.5 h — shorter, same night
            ]
        ),
        TZ,
    )
    assert len(ep) == 1 and ep.loc[0, "duration_h"] == pytest.approx(7.0)


def test_calendar_is_complete_and_gaps_are_missing():
    daily = build_daily_sleep(
        _locks([("2013-04-01 23:00", "2013-04-02 07:00"), ("2013-04-04 23:00", "2013-04-05 07:00")]), TZ
    )
    assert len(daily) == 4
    assert daily["sleep_valid"].tolist() == [True, False, False, True]
    # The gap must NOT create a huge "irregularity" value (the original metric reached 48 h here).
    assert daily["sri_agree"].isna().all()


def test_perfectly_regular_sleep_has_sri_100():
    nights = [(f"2013-04-{d:02d} 23:00", f"2013-04-{d + 1:02d} 07:00") for d in range(1, 9)]
    daily = build_daily_sleep(_locks(nights), TZ)
    assert sri_from_agreement(daily["sri_agree"]) == pytest.approx(100.0)


def test_shifted_sleep_lowers_sri():
    regular = [(f"2013-04-{d:02d} 23:00", f"2013-04-{d + 1:02d} 07:00") for d in range(1, 9)]
    shifted = [
        (f"2013-04-{d:02d} {'23:00' if d % 2 else '20:00'}", f"2013-04-{d + 1:02d} {'07:00' if d % 2 else '04:00'}")
        for d in range(1, 9)
    ]
    sri_reg = sri_from_agreement(build_daily_sleep(_locks(regular), TZ)["sri_agree"])
    sri_shift = sri_from_agreement(build_daily_sleep(_locks(shifted), TZ)["sri_agree"])
    # 3 h shift = 12 fifteen-minute epochs mismatched at each end -> 24/96 differ -> agreement 72/96 -> SRI 50
    assert sri_shift == pytest.approx(200 * 72 / 96 - 100)
    assert sri_shift < sri_reg


def test_sri_of_empty_is_nan():
    assert np.isnan(sri_from_agreement(np.array([np.nan, np.nan])))
