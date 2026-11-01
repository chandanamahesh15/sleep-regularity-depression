"""Behavioural context features: phone use and conversation exposure.

Fixes vs. the original notebook:
* daily buckets use local time, not UTC;
* a day with no sensor rows is *missing* (NaN), not zero — the original
  ``fillna(0)`` told the model "no screen use / no talking" on days the sensor was off;
* negative or absurd conversation durations are dropped.
"""

from __future__ import annotations

import pandas as pd


def _local_date(epoch_s: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(epoch_s, unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None).dt.normalize()


def daily_screen(app_usage: pd.DataFrame, tz: str) -> pd.DataFrame:
    """uid, date, screen_samples — count of running-app samples per local day."""
    if app_usage.empty:
        return pd.DataFrame(columns=["uid", "date", "screen_samples"])
    df = app_usage.assign(date=_local_date(app_usage["timestamp"], tz))
    return df.groupby(["uid", "date"]).size().rename("screen_samples").reset_index()


def daily_conversation(conversation: pd.DataFrame, tz: str, max_hours: float = 12.0) -> pd.DataFrame:
    """uid, date, conversation_min — minutes of detected conversation per local day."""
    if conversation.empty:
        return pd.DataFrame(columns=["uid", "date", "conversation_min"])
    df = conversation.copy()
    df["dur_s"] = df["end_timestamp"] - df["start_timestamp"]
    df = df[(df["dur_s"] > 0) & (df["dur_s"] <= max_hours * 3600)]
    df["date"] = _local_date(df["start_timestamp"], tz)
    out = df.groupby(["uid", "date"])["dur_s"].sum().div(60).rename("conversation_min").reset_index()
    return out
