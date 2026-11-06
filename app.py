"""Streamlit demo: explore the detector on synthetic campaigns.

Run with:  streamlit run app.py
"""
from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

from src.detector import DEFAULT_TIERS, alerts_from_flags, detect, naive_detect
from src.evaluate import score
from src.generate_data import generate

st.set_page_config(page_title="Message Trend Anomaly Detector", layout="wide")
st.title("Message Trend Anomaly Detector")
st.caption("Synthetic data only. Daily message volumes per campaign over 90 days, "
           "with injected flatlines, spikes, dips and cadence changes.")

with st.sidebar:
    st.header("Data")
    seed = st.number_input("Random seed", 1, 9999, 7, help="Generates a different synthetic dataset.")
    data, events = generate(int(seed))
    campaign = st.selectbox("Campaign", list(data["campaign"].unique()))

    st.header("Detector")
    mode = st.radio("Rules", ["Tiered rules", "Naive (one rule for all)"])
    tiers = dict(DEFAULT_TIERS)
    if mode == "Tiered rules":
        st.caption("Tune the thresholds for each volume tier:")
        for name in ("high", "medium", "low"):
            t = tiers[name]
            with st.expander(f"{name} volume", expanded=False):
                spike = st.slider("Spike: % above weekday average", 10, 150, int(t.spike_pct * 100), 5, key=f"s{name}")
                dip = st.slider("Dip: SD below weekday average", 1.5, 5.0, float(t.dip_sd), 0.5, key=f"d{name}")
                flat = st.slider("Flatline: days at zero", 2, 6, t.flatline_days, key=f"f{name}")
                tiers[name] = replace(t, spike_pct=spike / 100, dip_sd=dip, flatline_days=flat)

flagged = detect(data, tiers) if mode == "Tiered rules" else naive_detect(data)
alerts = alerts_from_flags(flagged)
result = score(alerts, events)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alerts raised", result["alerts"])
c2.metric("False alerts", result["false_alerts"])
c3.metric("Precision", f"{result['precision']:.0%}")
c4.metric("Events caught", f"{result['caught']} / {result['events']}")
st.caption("Scored against the injected ground truth across all 8 campaigns "
           "(an alert is false if no real event is near it).")

view = flagged[flagged["campaign"] == campaign].copy()
tier = view["tier"].replace("", pd.NA).dropna()
st.subheader(f"{campaign}" + (f"  ·  tier: {tier.mode().iat[0]}" if len(tier) else ""))

base = alt.Chart(view).encode(x=alt.X("date:T", title=None))
line = base.mark_line(color="#6b7280").encode(y=alt.Y("count:Q", title="Messages sent"))
band = base.mark_area(opacity=0.15, color="#2563eb").encode(y="lower:Q", y2="upper:Q")
truth_df = events[events["campaign"] == campaign].assign(end=lambda d: d["end"] + pd.Timedelta(days=1))
truth = alt.Chart(truth_df).mark_rect(opacity=0.18, color="#f59e0b").encode(x="start:T", x2="end:T")
pts = base.mark_point(size=90, filled=True).encode(
    y="count:Q",
    color=alt.Color("flag:N", title="Flag", scale=alt.Scale(
        domain=["spike", "dip", "flatline", "cadence_gap", "cadence_burst"],
        range=["#dc2626", "#7c3aed", "#111827", "#ea580c", "#0891b2"])),
    tooltip=["date:T", "count:Q", "flag:N"],
).transform_filter(alt.datum.flag != "")
st.altair_chart((truth + band + line + pts).properties(height=380), width="stretch")
st.caption("Orange bands = real injected events (ground truth). Blue band = expected range. "
           "Dots = days the detector flagged.")

left, right = st.columns(2)
with left:
    st.markdown("**Alerts for this campaign**")
    st.dataframe(alerts[alerts["campaign"] == campaign].assign(
        start=lambda d: d["start"].dt.date, end=lambda d: d["end"].dt.date), hide_index=True)
with right:
    st.markdown("**Real events for this campaign**")
    st.dataframe(events[events["campaign"] == campaign].assign(
        start=lambda d: d["start"].dt.date, end=lambda d: d["end"].dt.date), hide_index=True)
