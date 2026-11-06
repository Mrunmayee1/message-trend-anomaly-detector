"""Unit tests for the detection rules, using small hand-built series."""
import numpy as np
import pandas as pd

from src.detector import DEFAULT_TIERS, assign_tier, detect, naive_detect
from src.generate_data import generate

WEEKDAY = np.array([1.00, 1.05, 1.05, 1.02, 0.98, 0.65, 0.55])


def make(counts, name="c"):
    dates = pd.date_range("2026-01-05", periods=len(counts), freq="D")  # starts Monday
    return pd.DataFrame({"date": dates, "campaign": name, "count": counts})


def steady(n=60, base=5000, seed=0):
    rng = np.random.default_rng(seed)
    dow = np.arange(n) % 7
    return np.round(base * WEEKDAY[dow] * (1 + rng.normal(0, 0.04, n))).astype(int)


def flags(df):
    return detect(df).set_index("date")["flag"]


def test_tiers_follow_volume_caps():
    assert assign_tier(50) == "very_low"
    assert assign_tier(500) == "low"
    assert assign_tier(5_000) == "medium"
    assert assign_tier(50_000) == "high"


def test_normal_weekly_pattern_is_not_flagged():
    f = flags(make(steady()))
    assert (f == "").all()


def test_two_zero_days_are_not_a_flatline():
    c = steady()
    c[45:47] = 0
    f = flags(make(c))
    assert "flatline" not in set(f)


def test_three_zero_days_are_a_flatline():
    c = steady()
    c[45:48] = 0
    f = flags(make(c))
    assert (f.iloc[45:48] == "flatline").all()


def test_spike_over_fifty_percent_is_flagged():
    c = steady()
    c[46] = int(c[46] * 1.8)
    assert flags(make(c)).iloc[46] == "spike"


def test_small_bump_is_not_a_spike():
    c = steady()
    c[46] = int(c[46] * 1.15)
    assert flags(make(c)).iloc[46] == ""


def test_dip_is_flagged():
    c = steady()
    c[46] = int(c[46] * 0.4)
    assert flags(make(c)).iloc[46] == "dip"


def test_recovery_after_outages_is_not_a_spike():
    # Recent outages drag the naive 28-day average down, so a healthy day looks
    # like a spike. The tiered detector leaves flagged days out of its baseline.
    c = steady(n=70)
    c[34:40] = 0
    c[46:51] = 0
    df = make(c)
    after = slice(51, 60)
    assert (detect(df)["flag"].iloc[after] == "").all()
    assert (naive_detect(df)["flag"].iloc[after] == "spike").any()


def test_weekday_swings_hide_dips_from_naive_detector():
    c = steady()
    c[46] = int(c[46] * 0.5)
    df = make(c)
    assert detect(df)["flag"].iloc[46] == "dip"
    assert naive_detect(df)["flag"].iloc[46] == ""


def test_missed_sends_on_cadence_campaign_are_flagged():
    c = np.zeros(70, dtype=int)
    c[::3] = 60                      # sends every 3 days
    c[42:51] = 0                     # three scheduled sends skipped
    f = flags(make(list(c)))
    assert (f == "cadence_gap").any()


def test_regular_low_volume_cadence_is_quiet():
    c = np.zeros(70, dtype=int)
    c[::3] = 60
    assert (flags(make(list(c))) == "").all()


def test_burst_of_extra_sends_is_flagged():
    c = np.zeros(70, dtype=int)
    c[::3] = 60
    c[50:55] = 60                    # daily sends for five days
    assert (flags(make(list(c))) == "cadence_burst").any()


def test_generator_is_reproducible():
    a, ea = generate(3)
    b, eb = generate(3)
    assert a.equals(b) and ea.equals(eb)


def test_custom_tier_thresholds_change_results():
    c = steady()
    c[46] = int(c[46] * 1.4)
    df = make(c)
    default = detect(df)["flag"].iloc[46]
    loose = dict(DEFAULT_TIERS)
    from dataclasses import replace
    loose["medium"] = replace(DEFAULT_TIERS["medium"], spike_pct=0.30)
    strict = detect(df, loose)["flag"].iloc[46]
    assert default == "" and strict == "spike"
