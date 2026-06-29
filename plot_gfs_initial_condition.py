from datetime import datetime

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from earth2studio.data import GFS


def main() -> None:
    init_times = [
        datetime(2025, 6, 24, 0, 0),
        datetime(2025, 6, 25, 0, 0),
    ]
    target_var = "z500"
    levels = [5400, 5700, 5880]

    gfs = GFS()
    fields = {}
    for init_time in init_times:
        print(f"Fetching GFS initial condition for {init_time:%Y-%m-%d %H:%M UTC}...")
        gfs_raw = gfs(init_time, target_var)
        # Convert geopotential to height in meters.
        fields[init_time] = (gfs_raw / 9.80665).squeeze()

    map_proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5)
    data_proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": map_proj})
    ax.set_extent([-125, -60, 15, 50], crs=data_proj)

    ax.add_feature(cfeature.LAND, facecolor="#f7f7f7")
    ax.add_feature(cfeature.OCEAN, facecolor="#e0f2f1")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="black")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="gray")

    first_time = init_times[0]
    second_time = init_times[1]

    cs_day1 = ax.contour(
        fields[first_time].lon,
        fields[first_time].lat,
        fields[first_time],
        levels=levels,
        transform=data_proj,
        colors="#111111",
        linewidths=2.6,
        linestyles="solid",
    )
    ax.clabel(cs_day1, fmt="%1.0fm", inline=True, fontsize=10, colors="#111111")

    cs_day2 = ax.contour(
        fields[second_time].lon,
        fields[second_time].lat,
        fields[second_time],
        levels=levels,
        transform=data_proj,
        colors="#b30000",
        linewidths=2.6,
        linestyles="dashed",
    )
    ax.clabel(cs_day2, fmt="%1.0fm", inline=True, fontsize=10, colors="#b30000")

    ax.legend(
        [
            plt.Line2D([0], [0], color="#111111", linewidth=2.6, linestyle="solid"),
            plt.Line2D([0], [0], color="#b30000", linewidth=2.6, linestyle="dashed"),
        ],
        [
            f"GFS Init {first_time:%Y-%m-%d %H:%M UTC}",
            f"GFS Init {second_time:%Y-%m-%d %H:%M UTC}",
        ],
        loc="lower left",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
    )

    ax.set_title(
        "GFS 500hPa Geopotential Height Initial Conditions Overlay\n"
        f"Valid: {first_time:%Y-%m-%d %H:%M UTC} and {second_time:%Y-%m-%d %H:%M UTC}",
        fontsize=13,
        weight="bold",
        pad=15,
    )

    output_png = "gfs_z500_initial_conditions_overlay_20250624_20250625.png"
    plt.savefig(output_png, bbox_inches="tight", dpi=150)
    print(f"Success! Plot saved to {output_png}")


if __name__ == "__main__":
    main()
