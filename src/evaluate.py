"""Score the detectors against the injected ground truth.

An alert is a true positive if it overlaps a real event for the same campaign
(within `tolerance` days). An event counts as caught if any alert overlaps it.
"""
from __future__ import annotations

import pandas as pd

from .detector import alerts_from_flags, detect, naive_detect
from .generate_data import generate

TOLERANCE_DAYS = 1


def _overlap(a_start, a_end, b_start, b_end, tol=TOLERANCE_DAYS):
    pad = pd.Timedelta(days=tol)
    return a_start <= b_end + pad and a_end >= b_start - pad


def score(alerts: pd.DataFrame, events: pd.DataFrame) -> dict:
    tp_alerts = 0
    for a in alerts.itertuples():
        evs = events[events["campaign"] == a.campaign]
        if any(_overlap(a.start, a.end, e.start, e.end) for e in evs.itertuples()):
            tp_alerts += 1
    caught = 0
    caught_by_type: dict[str, list[int]] = {}
    for e in events.itertuples():
        als = alerts[alerts["campaign"] == e.campaign]
        hit = any(_overlap(a.start, a.end, e.start, e.end) for a in als.itertuples())
        caught += hit
        caught_by_type.setdefault(e.type, []).append(int(hit))
    n_alerts = len(alerts)
    return {
        "alerts": n_alerts,
        "false_alerts": n_alerts - tp_alerts,
        "precision": tp_alerts / n_alerts if n_alerts else 1.0,
        "recall": caught / len(events) if len(events) else 1.0,
        "events": len(events),
        "caught": caught,
        "recall_by_type": {k: sum(v) / len(v) for k, v in caught_by_type.items()},
    }


def run(seeds=range(1, 21)) -> pd.DataFrame:
    """Average both detectors over many random datasets."""
    rows = []
    for seed in seeds:
        data, events = generate(seed)
        for name, fn in (("naive", naive_detect), ("tiered rules", detect)):
            s = score(alerts_from_flags(fn(data)), events)
            s.pop("recall_by_type")
            rows.append({"detector": name, "seed": seed, **s})
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby("detector")[["alerts", "false_alerts", "precision", "recall"]].mean()
    return g.round(2).loc[["naive", "tiered rules"]]


if __name__ == "__main__":
    # Thresholds were tuned while looking at seeds 1-20, so report on fresh seeds.
    res = run(range(101, 201))
    print("Average over", res["seed"].nunique(), "unseen synthetic datasets (8 campaigns x 90 days each)\n")
    print(summary(res).to_string())
