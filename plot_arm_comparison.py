"""Core comparison: Arm A (model perturbation) vs. Arm B (IC perturbation).

Two charts: spread-growth comparison, and RMSE (skill) comparison. Each is
small multiples (4 variables) x 2 series (Arm A, Arm B), Global region.
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
ARM_COLOR = {"a": "#2a78d6", "b": "#eb6834"}
ARM_LABEL = {"a": "Arm A (model perturbation)", "b": "Arm B (IC perturbation)"}
GRID_COLOR = "#d9d9d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def make_comparison_plot(arm_a_npz, arm_b_npz, value_suffix, region, title, out_path):
    data_a = np.load(arm_a_npz)
    data_b = np.load(arm_b_npz)
    lead_hours = data_a["lead_hours"]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5), facecolor="#fcfcfb")
    fig.suptitle(title, fontsize=12.5, color=TEXT_PRIMARY, y=0.98)

    for ax, v in zip(axes.flat, VARIABLES):
        ax.set_facecolor("#fcfcfb")
        series = {
            "a": data_a[f"{v}_{region}_{value_suffix}"],
            "b": data_b[f"{v}_{region}_{value_suffix}"],
        }

        endpoints = {arm: series[arm][-1] for arm in series}
        yrange = max(series["a"].max(), series["b"].max()) - min(
            series["a"].min(), series["b"].min()
        )
        yrange = max(yrange, 1e-9)
        collide = abs(endpoints["a"] - endpoints["b"]) < 0.09 * yrange
        if collide:
            higher = max(endpoints, key=endpoints.get)
            lower = "b" if higher == "a" else "a"
            label_dy = {higher: 6, lower: -6}
        else:
            label_dy = {"a": 0, "b": 0}

        for arm in ["a", "b"]:
            y = series[arm]
            color = ARM_COLOR[arm]
            ax.plot(lead_hours, y, color=color, linewidth=2, solid_capstyle="round")
            ax.annotate(
                f"{y[-1]:.2f}",
                xy=(lead_hours[-1], y[-1]),
                xytext=(4, label_dy[arm]),
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
        Line2D([0], [0], color=ARM_COLOR[arm], linewidth=2, label=ARM_LABEL[arm])
        for arm in ["a", "b"]
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
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved {out_path}")


REGION_LABEL = {"global": "Global", "conus": "CONUS"}

for region in ["global", "conus"]:
    make_comparison_plot(
        "arm_a_spread_regions_v3.npz",
        "arm_b_spread_regions_v3.npz",
        "mean",
        region,
        f"Ensemble spread: Arm A vs. Arm B ({REGION_LABEL[region]}, mean across 14 inits)",
        f"arm_comparison_spread_{region}.png",
    )

    make_comparison_plot(
        "arm_a_rmse.npz",
        "arm_b_rmse.npz",
        "rmse",
        region,
        f"RMSE vs. IFS analysis: Arm A vs. Arm B ({REGION_LABEL[region]}, mean across 14 inits)",
        f"arm_comparison_rmse_{region}.png",
    )
