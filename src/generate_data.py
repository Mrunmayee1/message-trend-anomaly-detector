"""Synthetic daily message-volume data with labelled anomalies.

Everything here is made up. No company data is used. Each campaign has a
weekday pattern, random noise, and a few injected anomalies so a detector can
be scored against known ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Monday..Sunday: weekdays are busier, weekends quieter.
WEEKDAY_FACTOR = np.array([1.00, 1.05, 1.05, 1.02, 0.98, 0.65, 0.55])

CAMPAIGNS = [
    {"name": "Order confirmations", "base": 52000, "noise": 0.04, "kind": "daily"},
    {"name": "Shipping updates", "base": 21000, "noise": 0.05, "kind": "daily"},
    {"name": "Weekly deals", "base": 6800, "noise": 0.07, "kind": "daily"},
    {"name": "Cart reminders", "base": 3200, "noise": 0.09, "kind": "daily"},
    {"name": "Loyalty nudges", "base": 640, "noise": 0.14, "kind": "daily"},
    {"name": "Win-back offers", "base": 240, "noise": 0.18, "kind": "daily"},
    {"name": "Quarterly surveys", "base": 60, "noise": 0.20, "kind": "cadence", "every": 3},
    {"name": "Partner digests", "base": 90, "noise": 0.20, "kind": "cadence", "every": 7},
]

N_DAYS = 90
START = "2026-01-05"  # a Monday
FIRST_EVENT_DAY = 40   # leave ~5 weeks of clean history for baselines


def _pick_starts(rng, n, lo, hi, spacing):
    """Pick n non-overlapping start days."""
    starts: list[int] = []
    tries = 0
    while len(starts) < n and tries < 1000:
        tries += 1
        s = int(rng.integers(lo, hi))
        if all(abs(s - t) >= spacing for t in starts):
            starts.append(s)
    return sorted(starts)


def generate(seed: int = 7, n_days: int = N_DAYS):
    """Return (data, events).

    data:   date, campaign, count
    events: campaign, type, start, end (inclusive dates) -- the ground truth
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=n_days, freq="D")
    dow = dates.dayofweek.to_numpy()
    rows, events = [], []

    for c in CAMPAIGNS:
        jitter = 1 + rng.normal(0, 0.03, 7)  # slight per-campaign weekday shape
        if c["kind"] == "daily":
            expected = c["base"] * WEEKDAY_FACTOR[dow] * jitter[dow]
            counts = expected * (1 + rng.normal(0, c["noise"], n_days))
            counts = np.clip(np.round(counts), 0, None)

            kinds = ["flatline", "spike", "dip", str(rng.choice(["spike", "dip", "flatline"]))]
            starts = _pick_starts(rng, len(kinds), FIRST_EVENT_DAY, n_days - 8, 9)
            rng.shuffle(kinds)
            for kind, s in zip(kinds, starts):
                if kind == "flatline":
                    length = int(rng.integers(3, 7))
                    counts[s:s + length] = 0
                elif kind == "spike":
                    length = int(rng.integers(1, 3))
                    counts[s:s + length] = np.round(counts[s:s + length] * rng.uniform(1.8, 2.6))
                else:
                    length = int(rng.integers(1, 4))
                    counts[s:s + length] = np.round(counts[s:s + length] * rng.uniform(0.15, 0.40))
                events.append((c["name"], kind, dates[s], dates[s + length - 1]))
        else:
            every = c["every"]
            send = np.zeros(n_days, dtype=bool)
            send[::every] = True
            counts = np.zeros(n_days)
            counts[send] = np.round(c["base"] * (1 + rng.normal(0, c["noise"], send.sum())))

            send_idx = np.flatnonzero(send)
            # Event 1: two scheduled sends are skipped (cadence gap)
            k = int(rng.integers(len(send_idx) // 2, len(send_idx) - 4))
            gap_start, gap_end = send_idx[k], send_idx[k + 2] - 1
            counts[gap_start:send_idx[k + 2]] = 0
            events.append((c["name"], "cadence_gap", dates[gap_start], dates[gap_end]))
            # Event 2: a burst of daily sends, well away from the gap
            candidates = [s for s in range(FIRST_EVENT_DAY - 14, n_days - 8)
                          if s + 5 < gap_start - 3 or s > gap_end + 4]
            b = int(rng.choice(candidates))
            counts[b:b + 5] = np.round(c["base"] * (1 + rng.normal(0, c["noise"], 5)))
            events.append((c["name"], "cadence_burst", dates[b], dates[b + 4]))

        rows.append(pd.DataFrame({"date": dates, "campaign": c["name"], "count": counts.astype(int)}))

    data = pd.concat(rows, ignore_index=True)
    ev = pd.DataFrame(events, columns=["campaign", "type", "start", "end"])
    return data, ev.sort_values(["campaign", "start"]).reset_index(drop=True)


if __name__ == "__main__":
    d, e = generate()
    print(d.head(8).to_string(index=False))
    print()
    print(e.to_string(index=False))
