"""Plot Arm A ensemble spread growth, averaged across all 14 init dates.

Small multiples (one subplot per variable): mean spread across the 14 inits
as a solid line, with a shaded band showing +/-1 std across inits (i.e. how
much the spread-growth curve itself varies from one init date to another).
"""

import numpy as np
import matplotlib.pyplot as plt

VARIABLES = ["t850", "t2m", "u850", "z500"]
UNITS = {"t850": "K", "t2m": "K", "u850": "m/s", "z500": "m²/s²"}
TITLES = {
    "t850": "850 hPa temperature",
    "t2m": "2 m temperature",
    "u850": "850 hPa zonal wind",
    "z500": "500 hPa geopotential",
}
LINE_COLOR = "#2a78d6"
GRID_COLOR = "#d9d9d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

# arm_a_spread_all_inits.npz predates the mean(std)->sqrt(mean(variance)) fix
# and has no corrected counterpart; arm_a_spread_regions_v3.npz's "global"
# series carries the same per-variable mean/std_across_inits fields, correctly.
data = np.load("arm_a_spread_regions_v3.npz")
lead_hours = data["lead_hours"]

fig, axes = plt.subplots(2, 2, figsize=(9, 7), facecolor="#fcfcfb")
fig.suptitle(
    "Arm A ensemble spread growth (50 members, mean across 14 inits, 2026-06-01–14)",
    fontsize=12.5,
    color=TEXT_PRIMARY,
    y=0.98,
)

for ax, v in zip(axes.flat, VARIABLES):
    y_mean = data[f"{v}_global_mean"]
    y_std = data[f"{v}_global_std_across_inits"]
    ax.set_facecolor("#fcfcfb")

    ax.fill_between(
        lead_hours, y_mean - y_std, y_mean + y_std,
        color=LINE_COLOR, alpha=0.1, linewidth=0,
    )
    ax.plot(lead_hours, y_mean, color=LINE_COLOR, linewidth=2, solid_capstyle="round")

    ax.set_title(TITLES[v], fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("Lead time (h)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(f"Ensemble spread ({UNITS[v]})", fontsize=9, color=TEXT_SECONDARY)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=1, alpha=0.7)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    ax.annotate(
        f"{y_mean[-1]:.2f}",
        xy=(lead_hours[-1], y_mean[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=9,
        color=TEXT_PRIMARY,
        va="center",
    )

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("arm_a_spread_all_inits.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved arm_a_spread_all_inits.png")
