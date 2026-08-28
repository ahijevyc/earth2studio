"""Plot Arm A spread vs. skill (RMSE vs. IFS analysis truth), Global.

The core diagnostic this experiment was built to answer: does AIFS2ENS's
ensemble spread match its actual forecast error growth? Two series per
subplot (Spread, RMSE) -- a legend is required since there's more than one
series, per the dataviz categorical-color rule.
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
METRIC_COLOR = {"spread": "#2a78d6", "rmse": "#e34948"}
METRIC_LABEL = {"spread": "Ensemble spread", "rmse": "RMSE vs. IFS analysis"}
GRID_COLOR = "#d9d9d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

spread_data = np.load("arm_a_spread_regions_v3.npz")
rmse_data = np.load("arm_a_rmse.npz")
lead_hours = spread_data["lead_hours"]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), facecolor="#fcfcfb")
fig.suptitle(
    "Arm A: ensemble spread vs. skill (Global, 50 members, mean across 14 inits)",
    fontsize=12.5,
    color=TEXT_PRIMARY,
    y=0.98,
)

for ax, v in zip(axes.flat, VARIABLES):
    ax.set_facecolor("#fcfcfb")
    series = {
        "spread": spread_data[f"{v}_global_mean"],
        "rmse": rmse_data[f"{v}_global_rmse"],
    }

    endpoints = {m: series[m][-1] for m in series}
    yrange = max(series["spread"].max(), series["rmse"].max()) - min(
        series["spread"].min(), series["rmse"].min()
    )
    collide = abs(endpoints["spread"] - endpoints["rmse"]) < 0.06 * yrange
    label_dy = {"spread": -6, "rmse": 6} if collide else {"spread": 0, "rmse": 0}

    for metric in ["spread", "rmse"]:
        y = series[metric]
        color = METRIC_COLOR[metric]
        ax.plot(lead_hours, y, color=color, linewidth=2, solid_capstyle="round")
        ax.annotate(
            f"{y[-1]:.2f}",
            xy=(lead_hours[-1], y[-1]),
            xytext=(4, label_dy[metric]),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT_PRIMARY,
            va="center",
        )

    ax.set_title(TITLES[v], fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("Lead time (h)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(f"{UNITS[v]}", fontsize=9, color=TEXT_SECONDARY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=1, alpha=0.7)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

legend_handles = [
    Line2D([0], [0], color=METRIC_COLOR[m], linewidth=2, label=METRIC_LABEL[m])
    for m in ["spread", "rmse"]
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
fig.savefig("arm_a_spread_skill.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved arm_a_spread_skill.png")
