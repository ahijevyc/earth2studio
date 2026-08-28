"""Spread/RMSE calibration ratio vs. lead time, Arm A vs. Arm B.

The actual core diagnostic of this experiment: not just "how much spread"
or "how much error" separately, but whether spread tracks error well
(ratio near 1.0 = well-calibrated; below 1.0 = underdispersive) across the
whole forecast, not just at one lead time -- a single endpoint can mask a
crossover (e.g. CONUS z500: Arm A leads early, Arm B leads late).

lead=0h is excluded for Arm A only: RMSE there is ~0 (Arm A's IC literally
is the truth source), making the ratio numerically meaningless/undefined at
that one point. Arm B's IC is a different (perturbed) ensemble member, not
the truth, so its lead=0h RMSE/ratio is meaningful and is plotted -- but
it's 2-6x the rest of the series (a real result: a single perturbed
analysis is already close to truth, so RMSE is tiny there while spread
across 50 such analyses is not). Each panel uses a broken y-axis (top strip
for the lead=0h spike, main strip for the 6-120h calibration story) so the
spike doesn't compress the rest of the panel into unreadability.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VARIABLES = ["t850", "t2m", "u850", "z500"]
TITLES = {
    "t850": "850 hPa temperature",
    "t2m": "2 m temperature",
    "u850": "850 hPa zonal wind",
    "z500": "500 hPa geopotential",
}
ARM_COLOR = {"a": "#2a78d6", "b": "#eb6834"}
ARM_LABEL = {"a": "Arm A (model perturbation)", "b": "Arm B (IC perturbation)"}
REF_COLOR = "#8a8975"
GRID_COLOR = "#d9d9d6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

REGION_LABEL = {"global": "Global", "conus": "CONUS"}

# (pair_row, col) placement of each variable in the 2x2 variable grid
GRID_POS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def draw_break_marks(ax_top, ax_bot):
    d = 0.5
    kwargs = dict(
        marker=[(-1, -d), (1, d)], markersize=10, linestyle="none",
        color=TEXT_SECONDARY, mec=TEXT_SECONDARY, mew=1, clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)


def make_ratio_plot(region, out_path):
    spread_a = np.load("arm_a_spread_regions_v3.npz")
    spread_b = np.load("arm_b_spread_regions_v3.npz")
    rmse_a = np.load("arm_a_rmse.npz")
    rmse_b = np.load("arm_b_rmse.npz")
    lead_hours_full = spread_a["lead_hours"]
    lead_hours = {"a": lead_hours_full[1:], "b": lead_hours_full}

    fig = plt.figure(figsize=(9.5, 8.5), facecolor="#fcfcfb")
    fig.suptitle(
        f"Spread/RMSE calibration ratio: Arm A vs. Arm B ({REGION_LABEL[region]})",
        fontsize=12.5, color=TEXT_PRIMARY, y=0.99,
    )
    outer = fig.add_gridspec(
        2, 2, hspace=0.45, wspace=0.28,
        left=0.08, right=0.97, top=0.93, bottom=0.09,
    )

    for (row, col), v in zip(GRID_POS, VARIABLES):
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.08)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

        ratio_a = (spread_a[f"{v}_{region}_mean"] / rmse_a[f"{v}_{region}_rmse"])[1:]
        ratio_b = spread_b[f"{v}_{region}_mean"] / rmse_b[f"{v}_{region}_rmse"]
        series = {"a": ratio_a, "b": ratio_b}

        # main strip covers the 6-120h range (both arms) plus the ref line;
        # top strip covers only the lead=0h Arm B spike
        main_max = max(ratio_a.max(), ratio_b[1:].max(), 1.0)
        main_min = min(ratio_a.min(), ratio_b[1:].min(), 1.0)
        main_pad = 0.08 * (main_max - main_min)
        bot_ylim = (main_min - main_pad, main_max + main_pad)
        spike = ratio_b[0]
        top_ylim = (bot_ylim[1], spike + 0.12 * (spike - bot_ylim[1]))

        for ax in (ax_top, ax_bot):
            ax.set_facecolor("#fcfcfb")
        ax_bot.axhline(1.0, color=REF_COLOR, linewidth=1, linestyle=(0, (4, 3)))

        endpoints = {arm: series[arm][-1] for arm in series}
        yrange = max(bot_ylim[1] - bot_ylim[0], 1e-9)
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
            for ax in (ax_top, ax_bot):
                ax.plot(lead_hours[arm], y, color=color, linewidth=2, solid_capstyle="round")
            ax_bot.annotate(
                f"{y[-1]:.2f}",
                xy=(lead_hours[arm][-1], y[-1]),
                xytext=(4, label_dy[arm]),
                textcoords="offset points",
                fontsize=8.5, color=TEXT_PRIMARY, va="center",
            )

        ax_top.annotate(
            f"{spike:.2f}",
            xy=(lead_hours["b"][0], spike),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8, color=TEXT_PRIMARY, va="center",
        )

        ax_top.set_ylim(*top_ylim)
        ax_bot.set_ylim(*bot_ylim)
        ax_top.set_title(TITLES[v], fontsize=11, color=TEXT_PRIMARY, loc="left")
        ax_bot.set_xlabel("Lead time (h)", fontsize=9, color=TEXT_SECONDARY)
        ax_bot.set_ylabel("Spread / RMSE", fontsize=9, color=TEXT_SECONDARY)

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        for spine in ["top", "right"]:
            ax_bot.spines[spine].set_visible(False)
            ax_top.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax_top.spines[spine].set_color(GRID_COLOR)
            ax_bot.spines[spine].set_color(GRID_COLOR)
        for ax in (ax_top, ax_bot):
            ax.grid(True, color=GRID_COLOR, linewidth=1, alpha=0.7)
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        ax_top.tick_params(labelsize=7.5)

        draw_break_marks(ax_top, ax_bot)

    legend_handles = [
        Line2D([0], [0], color=ARM_COLOR[arm], linewidth=2, label=ARM_LABEL[arm])
        for arm in ["a", "b"]
    ] + [Line2D([0], [0], color=REF_COLOR, linewidth=1, linestyle=(0, (4, 3)),
                label="Perfect calibration (ratio = 1)")]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.0), fontsize=9.5, labelcolor=TEXT_PRIMARY,
    )

    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved {out_path}")


for region in ["global", "conus"]:
    make_ratio_plot(region, f"arm_calibration_ratio_{region}.png")
