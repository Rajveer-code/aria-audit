"""Figure 1 — Axis-coverage radar plot.

Compares which audit axes each tool covers.
ARIA covers all 5; baselines cover subsets.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Axes (5) and methods (4)
AXES_LABELS = [
    "Calibration\n(Group ECE)",
    "Faithfulness\n(HHEM 2.1)",
    "Consistency\n(Sem. Entropy)",
    "Equity\n(DI / EOD)",
    "Attribution\n(Jaccard@k)",
]
N = len(AXES_LABELS)

# Coverage scores [0,1] — 1 = full coverage, 0.5 = partial/proxy, 0 = not covered
DATA = {
    "ARIA (ours)":       [1.0, 1.0, 1.0, 1.0, 1.0],
    "Granite Guardian":  [0.0, 0.5, 0.0, 0.0, 0.5],
    "RAGAS":             [0.0, 0.8, 0.0, 0.0, 0.4],
    "LlamaGuard":        [0.0, 0.0, 0.0, 0.0, 0.0],
}
COLORS = {
    "ARIA (ours)":       "#E76F51",
    "Granite Guardian":  "#2A9D8F",
    "RAGAS":             "#E9C46A",
    "LlamaGuard":        "#5E81AC",
}
ALPHAS = {
    "ARIA (ours)": 0.25,
    "Granite Guardian": 0.15,
    "RAGAS": 0.15,
    "LlamaGuard": 0.1,
}

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close polygon

fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={"polar": True})

# Draw axis lines
ax.set_xticks(angles[:-1])
ax.set_xticklabels(AXES_LABELS, size=7.5, ha="center")
ax.set_ylim(0, 1.1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["", "0.5", "", "1.0"], size=6, color="#888")
ax.spines["polar"].set_visible(False)
ax.grid(color="#DDD", linewidth=0.6)

for method, values in DATA.items():
    vals = values + values[:1]
    color = COLORS[method]
    alpha = ALPHAS[method]
    lw = 2.0 if method == "ARIA (ours)" else 1.2
    ls = "-" if method == "ARIA (ours)" else "--"
    ax.plot(angles, vals, color=color, linewidth=lw, linestyle=ls, zorder=3)
    ax.fill(angles, vals, color=color, alpha=alpha, zorder=2)

# Legend
patches = [
    mpatches.Patch(facecolor=COLORS[m], edgecolor=COLORS[m], alpha=0.6, label=m)
    for m in DATA
]
ax.legend(
    handles=patches,
    loc="upper right",
    bbox_to_anchor=(1.45, 1.1),
    fontsize=7,
    frameon=True,
    framealpha=0.9,
    edgecolor="#CCC",
)

ax.set_title("Audit Axis Coverage", pad=14, fontweight="bold", fontsize=10)

fig.savefig("fig_coverage_radar.pdf")
fig.savefig("fig_coverage_radar.png", dpi=300)
print("Saved fig_coverage_radar.pdf / .png")
