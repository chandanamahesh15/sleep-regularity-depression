"""Sleep features from phone-lock episodes.

Fixes vs. the original notebook:
* timestamps are converted from UTC to Dartmouth local time before any date logic;
* the sleep midpoint is a *clock time* (hours since noon), not a raw Unix timestamp;
* the daily table is re-indexed to a full calendar so missing days are explicit;
* "SRI" is now the published Sleep Regularity Index (Phillips et al., 2017):
  the probability of being in the same sleep/wake state at two time points 24 h
  apart, rescaled to [-100, 100]. The original metric (rolling std of
  consecutive-midpoint *differences*) exploded to 100–200 h whenever a day was missing.

Limitation: phone-lock is a proxy for sleep. Only the longest lock period per
noon-to-noon day is treated as sleep; naps and phone-off periods are not modelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hours_since_noon(ts: pd.Series) -> pd.Series:
    """Wall-clock hours since 12:00 local, in [0, 24). Midnight -> 12, 03:00 -> 15."""
    h = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
    return (h - 12) % 24


def extract_main_episodes(
    phonelock: pd.DataFrame,
    tz: str,
    min_hours: float = 3.0,
    max_hours: float = 14.0,
) -> pd.DataFrame:
    """Return one row per (uid, sleep_date): the longest qualifying lock episode."""
    df = phonelock.dropna(subset=["start", "end"]).copy()
    df["duration_h"] = (df["end"] - df["start"]) / 3600.0
    df = df[(df["duration_h"] >= min_hours) & (df["duration_h"] <= max_hours)]
    if df.empty:
        return pd.DataFrame(columns=["uid", "date", "duration_h", "onset_local", "offset_local"])

    # tz-aware -> local wall clock (naive) so arithmetic is in local hours.
    df["onset_local"] = pd.to_datetime(df["start"], unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    df["offset_local"] = pd.to_datetime(df["end"], unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    # Noon-to-noon "sleep night": onset at 01:00 on the 5th belongs to the night of the 4th.
    df["date"] = (df["onset_local"] - pd.Timedelta(hours=12)).dt.normalize()

    idx = df.groupby(["uid", "date"])["duration_h"].idxmax()
    main = df.loc[idx, ["uid", "date", "duration_h", "onset_local", "offset_local"]]
    return main.sort_values(["uid", "date"]).reset_index(drop=True)


def _pairwise_sri_agreement(episodes: pd.DataFrame, calendar: pd.DatetimeIndex, epoch_minutes: int) -> np.ndarray:
    """For each calendar day d, fraction of epochs whose sleep/wake state matches day d-1.

    Days without a detected episode are unknown (NaN), so pairs touching them are NaN.
    """
    per_day = (24 * 60) // epoch_minutes
    n_days = len(calendar)
    origin = calendar[0] + pd.Timedelta(hours=12)  # noon of the first night
    timeline = np.zeros((n_days + 1) * per_day, dtype=np.int8)  # +1 day for overflow past noon

    for on, off in zip(episodes["onset_local"], episodes["offset_local"]):
        a = int((on - origin) / pd.Timedelta(minutes=epoch_minutes))
        b = int(np.ceil((off - origin) / pd.Timedelta(minutes=epoch_minutes)))
        timeline[max(a, 0) : max(min(b, len(timeline)), 0)] = 1

    grid = timeline[: n_days * per_day].reshape(n_days, per_day).astype(float)
    valid = calendar.isin(pd.DatetimeIndex(episodes["date"]))
    grid[~valid] = np.nan

    agree = np.full(n_days, np.nan)
    both = valid[1:] & valid[:-1]
    agree[1:][both] = (grid[1:][both] == grid[:-1][both]).mean(axis=1)
    return agree


def build_daily_sleep(
    phonelock: pd.DataFrame,
    tz: str,
    min_hours: float = 3.0,
    max_hours: float = 14.0,
    epoch_minutes: int = 15,
) -> pd.DataFrame:
    """Daily sleep table on a complete calendar per user.

    Columns: uid, date, sleep_valid, duration_h, midpoint_h, onset_h, sri_agree
    (``midpoint_h``/``onset_h`` are hours since local noon; ``sri_agree`` is the
    24-h state agreement with the previous night, the building block of the SRI).
    """
    episodes = extract_main_episodes(phonelock, tz, min_hours, max_hours)
    out = []
    for uid, ep in episodes.groupby("uid"):
        calendar = pd.date_range(ep["date"].min(), ep["date"].max(), freq="D")
        day = pd.DataFrame({"uid": uid, "date": calendar})
        ep = ep.assign(
            midpoint_h=_hours_since_noon(ep["onset_local"] + (ep["offset_local"] - ep["onset_local"]) / 2),
            onset_h=_hours_since_noon(ep["onset_local"]),
        )
        day = day.merge(ep[["date", "duration_h", "midpoint_h", "onset_h"]], on="date", how="left")
        day["sleep_valid"] = day["duration_h"].notna()
        day["sri_agree"] = _pairwise_sri_agreement(ep, calendar, epoch_minutes)
        out.append(day)
    cols = ["uid", "date", "sleep_valid", "duration_h", "midpoint_h", "onset_h", "sri_agree"]
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)[cols]


def sri_from_agreement(agree: np.ndarray | pd.Series) -> float:
    """Sleep Regularity Index in [-100, 100] from per-pair agreement fractions."""
    a = np.asarray(agree, dtype=float)
    a = a[~np.isnan(a)]
    return float(200 * a.mean() - 100) if a.size else np.nan
