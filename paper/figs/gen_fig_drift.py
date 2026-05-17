"""Figure 3 — Drift trajectory under injected bias.

Simulates Page-Hinkley cumulative sum over 80 prompts.
At t=40, demographic-biased prompts are injected.
Shows ARIA drift alarm firing; baseline tools have no drift detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

np.random.seed(42)
N = 80
INJECT_AT = 40
ALARM_THRESHOLD = 0.05

# Simulate DI axis values:
#   Phase 1 (t=0..39): DI close to 1.0 (fair), small noise
#   Phase 2 (t=40..79): DI drifts toward 0.65 (biased)
di_pre  = np.clip(np.random.normal(1.0, 0.05, INJECT_AT), 0, 1.5)
di_post = np.clip(np.random.normal(0.68, 0.07, N - INJECT_AT), 0, 1.5)
di_vals = np.concatenate([di_pre, di_post])

# Simulate faithfulness axis values: drops slightly at injection
f_pre  = np.clip(np.random.normal(0.85, 0.04, INJECT_AT), 0, 1)
f_post = np.clip(np.random.normal(0.78, 0.06, N - INJECT_AT), 0, 1)
f_vals = np.concatenate([f_pre, f_post])

# Run Page-Hinkley on DI
DELTA = 0.005
mean_running = di_vals[0]
cumsum = 0.0
min_cumsum = 0.0
cumsums = []
alarms = []
ALPHA = 0.999

for i, x in enumerate(di_vals):
    if i > 0:
        mean_running = ALPHA * mean_running + (1 - ALPHA) * x
    cumsum += x - mean_running - DELTA
    if cumsum < min_cumsum:
        min_cumsum = cumsum
    ph = cumsum - min_cumsum
    cumsums.append(ph)
    alarmed = ph > ALARM_THRESHOLD
    alarms.append(alarmed)
    if alarmed:
        cumsum = 0.0
        min_cumsum = 0.0

t = np.arange(N)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 3.8), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

# ── Top: per-prompt DI and faithfulness ─────────────────────────────────
ax1.plot(t, di_vals, color="#E76F51", linewidth=1.2, label="Equity DI", zorder=3)
ax1.plot(t, f_vals, color="#2A9D8F", linewidth=1.2, label="Faithfulness", zorder=3, linestyle="--")
ax1.axhline(1.0, color="#999", linewidth=0.7, linestyle=":", label="DI=1.0 (fair)")
ax1.axvline(INJECT_AT, color="#BF616A", linewidth=1.0, linestyle="--", alpha=0.7)
ax1.fill_betweenx([0, 1.5], INJECT_AT, N, color="#BF616A", alpha=0.04)
ax1.set_ylabel("Axis Score")
ax1.set_ylim(0.45, 1.25)
ax1.legend(loc="upper right", ncol=3, fontsize=7)
ax1.text(INJECT_AT + 0.5, 1.20, "Bias injection\n(t=40)", fontsize=7, color="#BF616A", va="top")
ax1.set_title("Page-Hinkley Drift Detection on Equity (DI) Axis", fontweight="bold")

# ── Bottom: PH cumsum + alarm markers ───────────────────────────────────
ax2.plot(t, cumsums, color="#264653", linewidth=1.5, label="PH cumsum", zorder=3)
ax2.axhline(ALARM_THRESHOLD, color="#E76F51", linewidth=1.0, linestyle="--", label=f"λ={ALARM_THRESHOLD}")
ax2.fill_betweenx([0, 0.3], INJECT_AT, N, color="#BF616A", alpha=0.04)
alarm_t = [i for i, a in enumerate(alarms) if a]
if alarm_t:
    ax2.scatter(alarm_t, [ALARM_THRESHOLD] * len(alarm_t), color="#BF616A",
                s=30, zorder=5, label="Alarm", marker="v")
ax2.set_ylabel("PH Stat.")
ax2.set_xlabel("Prompt Index")
ax2.set_ylim(0, 0.25)
ax2.legend(loc="upper left", ncol=3, fontsize=7)

fig.tight_layout(pad=1.2)
fig.savefig("fig_drift.pdf")
fig.savefig("fig_drift.png", dpi=300)
print("Saved fig_drift.pdf / .png")
