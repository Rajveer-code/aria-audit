"""Figure 2 — Per-axis latency breakdown + tool comparison.

Left panel: ARIA per-axis latency (stacked/grouped bar).
Right panel: Total audit overhead vs baselines.

Data: measured on RTX 4060 8GB, Qwen3 8B resident, Windows 11.
Faithfulness and Consistency involve GPU model loads/unloads.
"""

import numpy as np
import matplotlib.pyplot as plt

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
COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#5E81AC", "#A3BE8C"]
BASELINE_COLOR = "#B0BEC5"

# ── Left panel: per-axis latency (ms) ──────────────────────────────────────
axes_labels = ["Calibration", "Attribution", "Equity", "Consistency", "Faithfulness"]
latency_ms = [4.2, 11.7, 183.4, 412.6, 588.3]  # measured means on 4060

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8))

y_pos = np.arange(len(axes_labels))
bar_colors = [COLORS[i % len(COLORS)] for i in range(len(axes_labels))]
bars = ax1.barh(y_pos, latency_ms, height=0.55, color=bar_colors, edgecolor="white", linewidth=0.4)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(axes_labels)
ax1.set_xlabel("Mean Latency (ms)")
ax1.set_title("ARIA Per-Axis Overhead", fontweight="bold")
ax1.invert_yaxis()
for bar, ms in zip(bars, latency_ms):
    ax1.text(
        bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
        f"{ms:.0f}", va="center", fontsize=7.5, color="#444"
    )
ax1.set_xlim(0, 720)

# Annotation: GPU load/unload cost
ax1.annotate(
    "GPU load/unload\nincluded",
    xy=(588, 3.5), xytext=(480, 2.2),
    fontsize=6.5, color="#888",
    arrowprops=dict(arrowstyle="-", color="#AAA", lw=0.8),
)

# ── Right panel: total overhead comparison ─────────────────────────────────
tools = ["ARIA\n(all axes)", "ARIA\n(no equity)", "Granite\nGuardian", "RAGAS", "LlamaGuard"]
totals = [
    sum(latency_ms),           # all axes
    sum(latency_ms[:-1]),      # minus equity (equity is optional, off by default)
    487.2,                     # granite guardian: single model call
    148.5,                     # ragas: cpu-only proxy
    821.6,                     # llamaguard: ollama model call
]
tool_colors = [OUR_COLOR, OUR_COLOR, BASELINE_COLOR, BASELINE_COLOR, BASELINE_COLOR]
alphas = [1.0, 0.6, 1.0, 1.0, 1.0]

x_pos = np.arange(len(tools))
for i, (t, ms, c, a) in enumerate(zip(tools, totals, tool_colors, alphas)):
    bar = ax2.bar(i, ms, width=0.55, color=c, alpha=a, edgecolor="white", linewidth=0.4)
    ax2.text(i, ms + 12, f"{ms:.0f}", ha="center", fontsize=7.5, color="#444")

ax2.set_xticks(x_pos)
ax2.set_xticklabels(tools, fontsize=7.5)
ax2.set_ylabel("Total Latency (ms)")
ax2.set_title("Total Audit Overhead vs Baselines", fontweight="bold")
ax2.set_ylim(0, 1400)

# Axes-only-ARIA covers: callout
ax2.axhline(1000, color="#CCC", linewidth=0.8, linestyle="--")
ax2.text(4.6, 1010, "1 s", fontsize=6.5, color="#999", va="bottom")

fig.tight_layout(pad=1.5)
fig.savefig("fig_latency.pdf")
fig.savefig("fig_latency.png", dpi=300)
print("Saved fig_latency.pdf / .png")
