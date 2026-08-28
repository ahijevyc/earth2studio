"""Plot Arm A ensemble spread growth: Global vs. CONUS, small multiples.

CONUS box matches the "HWT region" in ens_spread_skill.ipynb: latitude
20-60N, longitude 220-300E. Two series per subplot now (Global, CONUS), so
unlike the single-region version this needs a legend -- color assigned in
fixed categorical order (slot 1 = Global, slot 2 = CONUS), never cycled.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VARIABLES = ["t850", "t2m", "u850", "z500"]
UNITS = {"t850": "K", "t2m": "K", "u850": "m/s", "z500": "m²/s²"}
TITLES = {
    "t850": "850 hPa temperature",
    "t2m": "2 m temperature",
    "u850": "850 hPa zonal wind",
    "z500": "500 hPa geopotential",
}
REGION_COLOR = {"global": "#2a78d6", "conus": "#eb6834"}
REGION_LABEL = {"global": "Global", "conus": "CONUS"}
GRID_COLOR = "#d9d9d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

data = np.load("arm_a_spread_regions_v3.npz")
lead_hours = data["lead_hours"]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), facecolor="#fcfcfb")
fig.suptitle(
    "Arm A ensemble spread growth: Global vs. CONUS (50 members, mean across 14 inits)",
    fontsize=12.5,
    color=TEXT_PRIMARY,
    y=0.98,
)

for ax, v in zip(axes.flat, VARIABLES):
    ax.set_facecolor("#fcfcfb")
    endpoints = {
        region: data[f"{v}_{region}_mean"][-1] for region in ["global", "conus"]
    }
    # nudge end-labels apart vertically if the two lines converge at the right
    # edge, rather than letting them stack on top of each other. Threshold is
    # relative to the plotted y-range (what matters for visual separation at
    # a fixed font size), not the endpoint magnitude.
    all_vals = np.concatenate(
        [data[f"{v}_global_mean"], data[f"{v}_conus_mean"]]
    )
    yrange = all_vals.max() - all_vals.min()
    collide = abs(endpoints["global"] - endpoints["conus"]) < 0.06 * yrange
    label_dy = {"global": 6, "conus": -6} if collide else {"global": 0, "conus": 0}

    for region in ["global", "conus"]:
        y_mean = data[f"{v}_{region}_mean"]
        y_std = data[f"{v}_{region}_std_across_inits"]
        color = REGION_COLOR[region]
        ax.fill_between(
            lead_hours, y_mean - y_std, y_mean + y_std,
            color=color, alpha=0.1, linewidth=0,
        )
        ax.plot(lead_hours, y_mean, color=color, linewidth=2, solid_capstyle="round")
        ax.annotate(
            f"{y_mean[-1]:.2f}",
            xy=(lead_hours[-1], y_mean[-1]),
            xytext=(4, label_dy[region]),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT_PRIMARY,
            va="center",
        )

    ax.set_title(TITLES[v], fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("Lead time (h)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(f"Ensemble spread ({UNITS[v]})", fontsize=9, color=TEXT_SECONDARY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=1, alpha=0.7)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

legend_handles = [
    Line2D([0], [0], color=REGION_COLOR[r], linewidth=2, label=REGION_LABEL[r])
    for r in ["global", "conus"]
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
    fontsize=10,
    labelcolor=TEXT_PRIMARY,
)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("arm_a_spread_regions.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved arm_a_spread_regions.png")
