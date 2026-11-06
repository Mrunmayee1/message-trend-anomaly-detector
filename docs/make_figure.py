"""Create docs/demo.png: one campaign, tiered detector vs naive, with ground truth."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.detector import alerts_from_flags, detect, naive_detect
from src.generate_data import generate

SEED, CAMPAIGN = 7, "Cart reminders"
COLORS = {"spike": "#dc2626", "dip": "#7c3aed", "flatline": "#111827"}

data, events = generate(SEED)
runs = [("Naive: one rule for every campaign", naive_detect(data)),
        ("Tiered rules: weekday-aware, per volume tier", detect(data))]

fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
for ax, (title, flagged) in zip(axes, runs):
    v = flagged[flagged["campaign"] == CAMPAIGN]
    for e in events[events["campaign"] == CAMPAIGN].itertuples():
        ax.axvspan(e.start, e.end + pd.Timedelta(days=1), color="#f59e0b", alpha=0.2, lw=0)
    ax.fill_between(v["date"], v["lower"], v["upper"], color="#2563eb", alpha=0.12, lw=0)
    ax.plot(v["date"], v["count"], color="#6b7280", lw=1.4)
    f = v[v["flag"] != ""]
    for kind, g in f.groupby("flag"):
        ax.scatter(g["date"], g["count"], s=38, color=COLORS.get(kind, "#000"), zorder=3, label=kind)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("messages / day")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, ncol=3, fontsize=8)
axes[0].text(1.0, 1.08, "orange = real injected events   blue = expected range   dots = flagged days",
             transform=axes[0].transAxes, ha="right", fontsize=8, color="#555")
fig.suptitle(f"{CAMPAIGN} (synthetic data)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig("docs/demo.png", dpi=130)
