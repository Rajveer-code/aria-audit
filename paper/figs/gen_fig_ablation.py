"""Figure 5 — Ablation: composite score with vs without each axis.

Shows what is missed when each axis is disabled.
Two sub-plots:
  Left:  composite score (out of 100) per axis removed — grouped by model
  Right: fraction of failure cases detected per axis — shows unique value of each axis

Models evaluated: Qwen3-8B, Phi-4-mini, Llama-3-8B (placeholder data).
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

OUR_COLOR = "#E76F51"
COLORS = ["#264653", "#2A9D8F", "#5E81AC"]
MODELS = ["Qwen3-8B", "Phi-4-mini", "Llama-3-8B"]

# Composite score (0-100) when each axis is dropped vs full ARIA
# Format: {condition: [Qwen3, Phi4, Llama]}
CONDITIONS = [
    "Full ARIA",
    "−Calibration",
    "−Faithfulness",
    "−Consistency",
    "−Equity",
    "−Attribution",
]

# Full ARIA composite scores (baseline reference per model)
# Dropout impact is model-specific; equity hit is largest for Llama (more biased)
SCORES = {
    "Full ARIA":    [82.4, 78.1, 74.3],
    "−Calibration": [76.8, 73.5, 69.2],
    "−Faithfulness":[71.3, 67.0, 63.1],
    "−Consistency": [79.1, 75.8, 71.0],
    "−Equity":      [65.2, 60.4, 53.7],   # equity biggest hit
    "−Attribution": [80.1, 76.9, 72.8],
}

# Fraction of failure cases uniquely detected by each axis
# (= cases where that axis fired and no other axis fired)
UNIQUE_DETECTION = {
    "Calibration":  [0.142, 0.138, 0.155],
    "Faithfulness": [0.287, 0.271, 0.301],
    "Consistency":  [0.098, 0.103, 0.091],
    "Equity":       [0.331, 0.348, 0.389],  # highest unique value
    "Attribution":  [0.062, 0.058, 0.071],
}

x = np.arange(len(CONDITIONS))
n = len(MODELS)
width = 0.22

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.0))

# ── Left: ablation grouped bar ────────────────────────────────────────────
for i, (model, color) in enumerate(zip(MODELS, COLORS)):
    vals = [SCORES[c][i] for c in CONDITIONS]
    offset = (i - n / 2 + 0.5) * width
    bars = ax1.bar(x + offset, vals, width * 0.9,
                   label=model, color=color, alpha=0.82,
                   edgecolor="white", linewidth=0.4)

# Mark "Full ARIA" region
ax1.axvspan(-0.5, 0.5, color="#F0F0EE", alpha=0.5, zorder=0)
ax1.text(0, 89, "Full", ha="center", fontsize=6.5, color="#999")

ax1.set_xticks(x)
ax1.set_xticklabels(CONDITIONS, rotation=20, ha="right", fontsize=7.5)
ax1.set_ylabel("Composite Score (0–100)")
ax1.set_title("Axis Ablation", fontweight="bold")
ax1.set_ylim(45, 95)
ax1.legend(loc="upper right", ncol=1, fontsize=7)

# Annotate equity drop
eq_idx = CONDITIONS.index("−Equity")
ax1.annotate(
    "Equity: largest\ncomposite drop",
    xy=(eq_idx, SCORES["−Equity"][0]),
    xytext=(eq_idx - 1.3, 60),
    fontsize=6.5, color=OUR_COLOR,
    arrowprops=dict(arrowstyle="->", color=OUR_COLOR, lw=0.8),
)

# ── Right: unique detection rates ─────────────────────────────────────────
det_axes = list(UNIQUE_DETECTION.keys())
x2 = np.arange(len(det_axes))
for i, (model, color) in enumerate(zip(MODELS, COLORS)):
    vals = [UNIQUE_DETECTION[a][i] for a in det_axes]
    offset = (i - n / 2 + 0.5) * width
    ax2.bar(x2 + offset, vals, width * 0.9,
            label=model, color=color, alpha=0.82,
            edgecolor="white", linewidth=0.4)

ax2.set_xticks(x2)
ax2.set_xticklabels(det_axes, rotation=20, ha="right", fontsize=7.5)
ax2.set_ylabel("Unique Failure Detection Rate")
ax2.set_title("Axis Unique Value", fontweight="bold")
ax2.set_ylim(0, 0.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.legend(loc="upper right", fontsize=7)

# Annotate equity bar
eq2_idx = det_axes.index("Equity")
ax2.annotate(
    "33–39% unique\ndetection",
    xy=(eq2_idx, UNIQUE_DETECTION["Equity"][0]),
    xytext=(eq2_idx + 0.8, 0.42),
    fontsize=6.5, color=OUR_COLOR,
    arrowprops=dict(arrowstyle="->", color=OUR_COLOR, lw=0.8),
)

fig.tight_layout(pad=1.5)
fig.savefig("fig_ablation.pdf")
fig.savefig("fig_ablation.png", dpi=300)
print("Saved fig_ablation.pdf / .png")
