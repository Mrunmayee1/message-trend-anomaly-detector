"""Rule-based anomaly detection for daily message volumes.

Two detectors live here:

* ``naive_detect``  - one global rule set for every campaign (the "before").
* ``detect``        - volume-tiered rules with weekday-aware baselines and
                      cadence rules for very low volume campaigns (the "after").

Rules (all thresholds are configurable per volume tier):
  flatline  zero volume for MORE than 2 days in a row
  spike     volume at least `spike_pct` above the usual for that weekday
  dip       volume at least `dip_sd` standard deviations below the usual
  cadence   (very low volume only) send frequency changed: a long gap since
            the last send, or a burst of unusually frequent sends
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TierRules:
    spike_pct: float = 0.50          # flag if >= 50% above the weekday average
    dip_sd: float = 3.0              # flag if >= 3 SD below the weekday average
    dip_min_pct: float = 0.20        # ...and at least this much lower (noise guard)
    flatline_days: int = 3           # zero for MORE than 2 days -> 3+ days
    use_cadence: bool = False        # cadence rules instead of volume rules
    cadence_gap_factor: float = 2.0  # gap > factor x typical gap -> flag
    cadence_burst_factor: float = 2.0


# Volume tiers: (name, upper cap on average daily volume). Each tier gets its
# own tolerance for peaks and dips, because noisy low-volume campaigns need
# looser limits than steady high-volume ones.
TIER_CAPS = [("very_low", 100), ("low", 1_000), ("medium", 10_000), ("high", math.inf)]

DEFAULT_TIERS = {
    "high": TierRules(spike_pct=0.30, dip_sd=3.0, dip_min_pct=0.15),
    "medium": TierRules(spike_pct=0.50, dip_sd=3.0, dip_min_pct=0.20),
    "low": TierRules(spike_pct=0.60, dip_sd=3.0, dip_min_pct=0.30),
    "very_low": TierRules(use_cadence=True),
}

HISTORY_DAYS = 28      # baseline window for volume rules
CADENCE_DAYS = 42      # longer window for cadence rules
MIN_SAME_WEEKDAY = 3   # need this many comparable days before alerting


def pooled_cv(counts: np.ndarray, dow: np.ndarray, idx: np.ndarray) -> float | None:
    """Day-to-day variation as a share of the weekday mean, pooled over all weekdays.

    Using only the 4 same-weekday values for the SD is very jumpy. Pooling the
    relative deviations from all 7 weekdays gives a much steadier estimate.
    """
    devs, groups = [], 0
    for d in range(7):
        vals = counts[idx[dow[idx] == d]]
        if len(vals) >= 2 and vals.mean() > 0:
            devs.extend(vals / vals.mean() - 1)
            groups += 1
    dof = len(devs) - groups
    if dof < 5:
        return None
    return float(np.sqrt(np.sum(np.square(devs)) / dof))


def assign_tier(avg_daily_volume: float) -> str:
    for name, cap in TIER_CAPS:
        if avg_daily_volume < cap:
            return name
    return "high"


def _detect_campaign(df: pd.DataFrame, tiers: dict[str, TierRules]) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    counts = df["count"].to_numpy(dtype=float)
    dow = df["date"].dt.dayofweek.to_numpy()
    n = len(df)

    flag = np.full(n, "", dtype=object)
    tier_col = np.full(n, "", dtype=object)
    expected = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    zero_run = 0

    for i in range(n):
        lo = max(0, i - HISTORY_DAYS)
        hist = np.arange(lo, i)
        clean = hist[flag[hist] == ""]
        if len(clean) < 7:
            zero_run = zero_run + 1 if counts[i] == 0 else 0
            continue  # warm-up: not enough history to judge

        tier = assign_tier(counts[clean].mean())
        rules = tiers[tier]
        tier_col[i] = tier
        zero_run = zero_run + 1 if counts[i] == 0 else 0

        if rules.use_cadence:
            c_lo = max(0, i - CADENCE_DAYS)
            window = np.arange(c_lo, i)
            sends = window[counts[window] > 0]
            # typical gap: only from consecutive sends with nothing flagged in between,
            # so earlier bursts and gaps do not skew what "normal" looks like
            gaps = [b - a for a, b in zip(sends[:-1], sends[1:])
                    if not (flag[a:b + 1] != "").any()]
            if len(sends) >= 3 and len(gaps) >= 2:
                typical_gap = float(np.median(gaps))
                if counts[i] == 0:
                    gap_now = i - sends[-1]
                    if gap_now > rules.cadence_gap_factor * typical_gap:
                        flag[i] = "cadence_gap"
                else:
                    early = (i - sends[-1]) < typical_gap  # this send came sooner than usual
                    recent = int((counts[max(0, i - 4):i + 1] > 0).sum())  # sends in last 5 days
                    baseline = 5.0 / typical_gap
                    if early and recent >= rules.cadence_burst_factor * baseline and recent >= baseline + 2:
                        flag[i] = "cadence_burst"
                        j = i                                    # mark the whole burst: walk back
                        hist_sends = list(sends)                 # through sends that came early
                        for k in range(len(hist_sends) - 1, 0, -1):
                            prev, before = hist_sends[k], hist_sends[k - 1]
                            if j - prev < typical_gap and prev - before < typical_gap:
                                flag[prev] = "cadence_burst"
                                j = prev
                            else:
                                break
            continue

        # --- volume rules: compare with the same weekday over the last 4 weeks
        same = clean[dow[clean] == dow[i]]
        if len(same) < MIN_SAME_WEEKDAY:
            continue
        cv = pooled_cv(counts, dow, clean)
        if cv is None:
            continue
        mean = counts[same].mean()
        sd = mean * cv
        expected[i] = mean
        upper[i] = mean * (1 + rules.spike_pct)
        lower[i] = max(0.0, min(mean - rules.dip_sd * sd, mean * (1 - rules.dip_min_pct)))

        if zero_run >= rules.flatline_days:
            flag[i - zero_run + 1:i + 1] = "flatline"   # mark the whole run
        elif counts[i] > upper[i]:
            flag[i] = "spike"
        elif counts[i] < mean - rules.dip_sd * sd and counts[i] < mean * (1 - rules.dip_min_pct):
            flag[i] = "dip"

    out = df.copy()
    out["tier"], out["expected"], out["lower"], out["upper"] = tier_col, expected, lower, upper
    out["flag"] = flag
    return out


def detect(data: pd.DataFrame, tiers: dict[str, TierRules] | None = None) -> pd.DataFrame:
    """Tiered, weekday-aware detector. Returns data plus tier/expected/flag columns."""
    tiers = tiers or DEFAULT_TIERS
    parts = [_detect_campaign(g, tiers) for _, g in data.groupby("campaign", sort=False)]
    return pd.concat(parts, ignore_index=True)


def naive_detect(data: pd.DataFrame, spike_pct: float = 0.5, dip_sd: float = 3.0) -> pd.DataFrame:
    """One rule set for everything: no tiers, no weekday baseline, no cadence.

    Flags any zero day, anything 50% above the trailing 28-day mean, and
    anything 3 SD below it. This is what a first quick version looks like.
    """
    parts = []
    for _, g in data.groupby("campaign", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        c = g["count"].astype(float)
        mean = c.shift(1).rolling(HISTORY_DAYS, min_periods=7).mean()
        sd = c.shift(1).rolling(HISTORY_DAYS, min_periods=7).std()
        flag = np.full(len(g), "", dtype=object)
        flag[(c > mean * (1 + spike_pct)).to_numpy()] = "spike"
        flag[(c < mean - dip_sd * sd).to_numpy()] = "dip"
        flag[((c == 0) & mean.notna()).to_numpy()] = "flatline"
        g["tier"], g["expected"], g["lower"], g["upper"] = "all", mean, mean - dip_sd * sd, mean * (1 + spike_pct)
        g["flag"] = flag
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def alerts_from_flags(flagged: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive flagged days of one campaign into a single alert."""
    rows = []
    for camp, g in flagged[flagged["flag"] != ""].groupby("campaign", sort=False):
        g = g.sort_values("date")
        start = prev = None
        kind = None
        for d, f in zip(g["date"], g["flag"]):
            if start is None:
                start, prev, kind = d, d, f
            elif (d - prev).days <= 1:
                prev = d
            else:
                rows.append((camp, kind, start, prev))
                start, prev, kind = d, d, f
        if start is not None:
            rows.append((camp, kind, start, prev))
    return pd.DataFrame(rows, columns=["campaign", "type", "start", "end"])
