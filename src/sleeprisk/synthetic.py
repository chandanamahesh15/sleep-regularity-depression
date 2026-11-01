"""Synthetic data in the exact StudentLife file layout.

Used for tests, CI and demos when the real archive is not available. It is built
to reproduce the *structure* that made the original evaluation leak:

* every student has a strong behavioural fingerprint (habitual sleep length,
  bedtime, phone use), and
* the depression label depends only weakly on behaviour.

So a model that sees a student in training can "recognise" them in test and look
excellent, while true generalisation to new students is modest. It also mimics
pre/post PHQ-9 waves, UTC timestamps, sensor outages and non-sleep phone locks.
Numbers produced on this data say nothing about real-world depression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANSWERS = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
PHQ_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, hopeless.",
    "Trouble falling or staying asleep, or sleeping too much.",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself",
    "Trouble concentrating on things",
    "Moving or speaking so slowly",
    "Thoughts that you would be better off dead",
]
TZ = "America/New_York"


def _items_from_total(total: int, rng: np.random.Generator) -> list[str]:
    scores = np.zeros(9, dtype=int)
    for _ in range(int(total)):
        open_ = np.flatnonzero(scores < 3)
        scores[rng.choice(open_)] += 1
    return [ANSWERS[s] for s in scores]


def _epoch(local: pd.Timestamp) -> int:
    return int(local.tz_localize(TZ).tz_convert("UTC").timestamp())


def write_synthetic_studentlife(out_dir: str | Path, n_users: int = 46, days: int = 66, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for sub in ["sensing/phonelock", "sensing/conversation", "app_usage", "survey"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp("2013-03-27")
    phq_rows = []
    for u in range(n_users):
        uid = f"u{u:02d}"
        risk = rng.normal()
        # Fingerprint (large between-person spread) + weak dependence on risk.
        dur_mu = 7.0 - 0.25 * risk + rng.normal(0, 0.7)
        bed_mu = 0.5 + 0.20 * risk + rng.normal(0, 1.0)  # hours after midnight
        jitter = abs(0.6 + 0.15 * risk + rng.normal(0, 0.3))
        screen_mu = np.exp(5.0 + 0.08 * risk + rng.normal(0, 0.5))
        conv_mu = max(20.0, 120 - 12 * risk + rng.normal(0, 40))  # minutes/day
        p_missing = rng.uniform(0.05, 0.25)
        outage = rng.integers(0, days - 10)

        locks, apps, convs = [], [], []
        for d in range(days):
            day = start + pd.Timedelta(days=d)
            if rng.random() < p_missing:
                continue
            bed = day + pd.Timedelta(hours=24 + bed_mu + rng.normal(0, jitter))
            dur = float(np.clip(rng.normal(dur_mu, 0.9), 3.2, 12.0))
            locks.append((_epoch(bed), _epoch(bed + pd.Timedelta(hours=dur))))
            for _ in range(rng.integers(3, 9)):  # short daytime locks (filtered out)
                t = day + pd.Timedelta(hours=float(rng.uniform(9, 22)))
                locks.append((_epoch(t), _epoch(t + pd.Timedelta(minutes=float(rng.uniform(5, 90))))))
            if rng.random() < 0.08:  # a long daytime lock — the proxy's known weakness
                t = day + pd.Timedelta(hours=float(rng.uniform(13, 16)))
                locks.append((_epoch(t), _epoch(t + pd.Timedelta(hours=3.3))))

            if not (outage <= d < outage + 5):  # app logger outage block
                n = rng.poisson(screen_mu)
                secs = np.sort(rng.uniform(8 * 3600, 24 * 3600, n))
                base = _epoch(day)
                apps.extend(int(base + s) for s in secs)
            remaining = rng.gamma(4, conv_mu / 4) * 60
            while remaining > 60:
                ln = min(remaining, rng.uniform(60, 1800))
                t0 = _epoch(day) + int(rng.uniform(9 * 3600, 23 * 3600))
                convs.append((t0, t0 + int(ln)))
                remaining -= ln

        pd.DataFrame(locks, columns=["start", "end"]).to_csv(
            out / f"sensing/phonelock/phonelock_{uid}.csv", index=False
        )
        pd.DataFrame({"timestamp": apps, "device": "synthetic"}).to_csv(
            out / f"app_usage/running_app_{uid}.csv", index=False
        )
        # Leading space in the column name is deliberate: the real file has it too.
        pd.DataFrame(convs, columns=["start_timestamp", " end_timestamp"]).to_csv(
            out / f"sensing/conversation/conversation_{uid}.csv", index=False
        )

        post = int(np.clip(round(6 + 3.5 * risk + rng.normal(0, 3.0)), 0, 27))
        pre = int(np.clip(round(post + rng.normal(-1, 3)), 0, 27))
        for wave, total in (("pre", pre), ("post", post)):
            if wave == "post" and rng.random() < 0.12:  # some students skip the post survey
                continue
            items = _items_from_total(total, rng)
            if rng.random() < 0.05:
                items[rng.integers(9)] = np.nan  # an unanswered item
            phq_rows.append([uid, wave, *items, "Somewhat difficult"])

    pd.DataFrame(phq_rows, columns=["uid", "type", *PHQ_ITEMS, "Response"]).to_csv(
        out / "survey/PHQ-9.csv", index=False
    )
    # A corrupt file to exercise the logged-failure path.
    (out / "sensing/phonelock/phonelock_u99.csv").write_text("not,a,valid\nfile\n")
    return out
