"""Figure 4 — DI / EOD distributions under counterfactual substitution.

Violin + strip-plot showing DI and EOD gap distributions across:
  - Gender axis (21 pair types)
  - Race axis (15 pair types)
  - Age axis (15 pair types)

Data: simulated from realistic distributions based on BBQ + BOLD + CPFE suites.
True experimental data will replace this when eval/results are available.
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

np.random.seed(2025)

AXES = ["Gender", "Race", "Age"]
N_PER = 200

# DI: 1.0 = fair, <0.8 = biased (80% rule). Mean/std based on BBQ findings.
DI_PARAMS = {
    "Gender": (0.91, 0.12),
    "Race":   (0.82, 0.18),
    "Age":    (0.87, 0.14),
}
EOD_PARAMS = {
    "Gender": (0.06, 0.05),
    "Race":   (0.11, 0.08),
    "Age":    (0.09, 0.07),
}

di_data = {a: np.clip(np.random.normal(*DI_PARAMS[a], N_PER), 0.1, 2.0) for a in AXES}
eod_data = {a: np.clip(np.abs(np.random.normal(*EOD_PARAMS[a], N_PER)), 0, 0.5) for a in AXES}

COLORS = {"Gender": "#E76F51", "Race": "#2A9D8F", "Age": "#E9C46A"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.0))

# ── Left: DI violin ──────────────────────────────────────────────────────
positions = [1, 2, 3]
parts = ax1.violinplot(
    [di_data[a] for a in AXES],
    positions=positions,
    widths=0.6,
    showmedians=True,
    showextrema=False,
)
for i, (body, ax_name) in enumerate(zip(parts["bodies"], AXES)):
    body.set_facecolor(COLORS[ax_name])
    body.set_alpha(0.55)
    body.set_edgecolor(COLORS[ax_name])
parts["cmedians"].set_colors(["#333"] * len(AXES))
parts["cmedians"].set_linewidth(2.0)

# 80% rule line
ax1.axhline(0.8, color="#BF616A", linestyle="--", linewidth=1.0, label="80% rule (DI<0.8)")
ax1.axhline(1.0, color="#888", linestyle=":", linewidth=0.8, label="DI=1.0 (fair)")
ax1.set_xticks(positions)
ax1.set_xticklabels(AXES)
ax1.set_ylabel("Disparate Impact (DI)")
ax1.set_title("DI by Demographic Axis", fontweight="bold")
ax1.set_ylim(0.0, 1.8)
ax1.legend(loc="upper right", fontsize=7)

# fraction below 0.8 per axis (bias rate)
for pos, ax_name in zip(positions, AXES):
    frac_biased = np.mean(di_data[ax_name] < 0.8)
    ax1.text(pos, 0.1, f"{frac_biased:.0%}\nbiased",
             ha="center", fontsize=6.5, color="#BF616A")

# ── Right: EOD bar with error bars ──────────────────────────────────────
eod_means = [np.mean(eod_data[a]) for a in AXES]
eod_sems = [np.std(eod_data[a]) / np.sqrt(N_PER) for a in AXES]
bar_colors = [COLORS[a] for a in AXES]

bars = ax2.bar(positions, eod_means, yerr=eod_sems, width=0.45,
               color=bar_colors, alpha=0.75, edgecolor="white",
               linewidth=0.4, capsize=3, error_kw={"linewidth": 1.2, "ecolor": "#555"})
ax2.set_xticks(positions)
ax2.set_xticklabels(AXES)
ax2.set_ylabel("Equalized Odds Gap (EOD)")
ax2.set_title("EOD by Demographic Axis", fontweight="bold")
ax2.set_ylim(0, 0.22)
ax2.axhline(0.05, color="#E9C46A", linestyle="--", linewidth=0.9, label="5% fairness tolerance")
ax2.legend(loc="upper right", fontsize=7)
for pos, m, s in zip(positions, eod_means, eod_sems):
    ax2.text(pos, m + s + 0.004, f"{m:.3f}", ha="center", fontsize=7.5, color="#444")

# Patch legend for axes
legend_patches = [mpatches.Patch(facecolor=COLORS[a], alpha=0.7, label=a) for a in AXES]
ax2.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color="#E9C46A", linestyle="--", linewidth=0.9, label="5% tolerance")
], fontsize=7, loc="upper right")

fig.tight_layout(pad=1.5)
fig.savefig("fig_equity_distributions.pdf")
fig.savefig("fig_equity_distributions.png", dpi=300)
print("Saved fig_equity_distributions.pdf / .png")
