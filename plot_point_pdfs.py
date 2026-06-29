from __future__ import annotations

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_MEMBER_GLOB = "AIFS2ENS_ARCOWithGFSFallback_member_*.zarr"
DEFAULT_VARIABLES = ["t2m", "tcw", "tp06"]
DEFAULT_PRECIP_BOX_DEG = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ensemble PDFs at a single CONUS point from 500-member AIFS stores."
        )
    )
    parser.add_argument(
        "--member-glob",
        default=DEFAULT_MEMBER_GLOB,
        help="Glob for member Zarr stores.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=39.0,
        help="Target latitude for the nearest grid point.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-97.0,
        help="Target longitude for the nearest grid point.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to plot from each member store.",
    )
    parser.add_argument(
        "--lead-index",
        type=int,
        default=4,
        help="Lead-time index to plot from each member store.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="Variables to plot, for example: t2m tcw tp06.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to a name derived from the point and variables.",
    )
    parser.add_argument(
        "--precip-box-deg",
        type=float,
        default=DEFAULT_PRECIP_BOX_DEG,
        help="Half-width in degrees of the precipitation box around the target point.",
    )
    return parser.parse_args()


def _pretty_variable_label(var_name: str, values: np.ndarray) -> tuple[str, np.ndarray]:
    if var_name == "t2m":
        return "2m temperature (°C)", values - 273.15
    if var_name.startswith("q"):
        return f"{var_name} specific humidity (g/kg)", values * 1000.0
    if var_name.startswith("tp") or var_name == "tp":
        return f"{var_name} precipitation (mm)", values * 1000.0
    if var_name == "tcw":
        return "Total column water", values
    return var_name, values


def _wrap_lon_to_grid(lon: float, grid_lon: xr.DataArray) -> float:
    grid_min = float(grid_lon.min())
    grid_max = float(grid_lon.max())
    if grid_min >= 0.0 and lon < 0.0:
        return lon % 360.0
    if grid_max <= 180.0 and lon > 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _box_slice(coord: xr.DataArray, center: float, half_width: float) -> slice:
    lower = max(float(coord.min()), center - half_width)
    upper = min(float(coord.max()), center + half_width)
    coord_values = np.asarray(coord.values).reshape(-1)
    if coord_values[0] <= coord_values[-1]:
        return slice(lower, upper)
    return slice(upper, lower)


def _precip_box_max(
    data: xr.DataArray,
    nearest_lat: float,
    nearest_lon: float,
    precip_box_deg: float,
) -> np.ndarray:
    lat_slice = _box_slice(data.lat, nearest_lat, precip_box_deg)
    lon_slice = _box_slice(data.lon, nearest_lon, precip_box_deg)

    if float(data.lon.min()) >= 0.0:
        lon_min = (nearest_lon - precip_box_deg) % 360.0
        lon_max = (nearest_lon + precip_box_deg) % 360.0
        if lon_min <= lon_max:
            region = data.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
        else:
            region = xr.concat(
                [
                    data.sel(lat=lat_slice, lon=slice(lon_min, None)),
                    data.sel(lat=lat_slice, lon=slice(None, lon_max)),
                ],
                dim="lon",
            )
    else:
        region = data.sel(lat=lat_slice, lon=lon_slice)

    return np.asarray(region.max(dim=("lat", "lon"), skipna=True).load().values, dtype=np.float64).reshape(-1)


def _load_point_values(
    member_glob: str,
    variables: list[str],
    lat: float,
    lon: float,
    time_index: int,
    lead_index: int,
    precip_box_deg: float,
) -> tuple[dict[str, np.ndarray], float, float, xr.DataArray | None]:
    member_paths = sorted(glob.glob(member_glob, recursive=True))
    if not member_paths:
        raise FileNotFoundError(f"No ensemble member stores matched {member_glob!r}")

    ds = xr.open_mfdataset(
        [str(path) for path in member_paths],
        engine="zarr",
        combine="nested",
        concat_dim="ensemble",
        consolidated=False,
        parallel=True,
        compat="override",
        join="override",
        coords="minimal",
    )

    try:
        target_lon = _wrap_lon_to_grid(lon, ds.lon)
        point = ds[variables].sel(lat=lat, lon=target_lon, method="nearest")
        nearest_lat = float(point.lat.values)
        nearest_lon = float(point.lon.values)

        values_by_var: dict[str, np.ndarray] = {}
        for var_name in variables:
            if var_name not in point:
                raise KeyError(f"Variable {var_name!r} is not available in the ensemble stores")

            data = point[var_name]
            if time_index >= data.sizes.get("time", 1):
                raise IndexError(
                    f"time-index {time_index} is out of range for {var_name!r} with size {data.sizes.get('time', 1)}"
                )
            if lead_index >= data.sizes.get("lead_time", 1):
                raise IndexError(
                    f"lead-index {lead_index} is out of range for {var_name!r} with size {data.sizes.get('lead_time', 1)}"
                )

            if var_name.startswith("tp") or var_name == "tp":
                member_values = _precip_box_max(
                    ds[var_name].isel(time=time_index, lead_time=lead_index),
                    nearest_lat=nearest_lat,
                    nearest_lon=nearest_lon,
                    precip_box_deg=precip_box_deg,
                )
            else:
                member_values = data.isel(time=time_index, lead_time=lead_index).load().values
            values_by_var[var_name] = np.asarray(member_values, dtype=np.float64).reshape(-1)

        valid_time = None
        if "time" in point.coords and "lead_time" in point.coords:
            valid_time = point.time.isel(time=time_index) + point.lead_time.isel(lead_time=lead_index)

        return values_by_var, nearest_lat, nearest_lon, valid_time
    finally:
        ds.close()


def main() -> None:
    args = parse_args()

    values_by_var, nearest_lat, nearest_lon, valid_time = _load_point_values(
        member_glob=args.member_glob,
        variables=args.variables,
        lat=args.lat,
        lon=args.lon,
        time_index=args.time_index,
        lead_index=args.lead_index,
        precip_box_deg=args.precip_box_deg,
    )

    n_panels = len(args.variables)
    fig, axes = plt.subplots(
        nrows=n_panels,
        ncols=1,
        figsize=(9, 3 * n_panels),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    if valid_time is not None:
        valid_time_str = np.datetime_as_string(np.asarray(valid_time.values), unit="m")
    else:
        valid_time_str = "unknown valid time"

    for ax, var_name in zip(axes, args.variables, strict=True):
        raw_values = values_by_var[var_name]
        label, plot_values = _pretty_variable_label(var_name, raw_values)

        finite_values = plot_values[np.isfinite(plot_values)]
        if finite_values.size == 0:
            raise ValueError(f"No finite values found for {var_name!r}")

        bins = min(40, max(12, int(np.sqrt(finite_values.size) * 2)))
        ax.hist(
            finite_values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.25,
            color="#1f77b4",
            edgecolor="#1f77b4",
        )
        ax.hist(
            finite_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color="#0f3b66",
        )

        mean_value = float(np.nanmean(finite_values))
        median_value = float(np.nanmedian(finite_values))
        q25, q75 = np.nanquantile(finite_values, [0.25, 0.75])

        ax.axvline(mean_value, color="#b30000", linewidth=2.0, label="mean")
        ax.axvline(median_value, color="#111111", linewidth=1.8, linestyle="--", label="median")
        ax.axvspan(q25, q75, color="#ffbf00", alpha=0.18, label="IQR")

        ax.set_title(f"{label}", fontsize=12, weight="bold")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        summary = (
            f"n={finite_values.size} | mean={mean_value:.3f} | median={median_value:.3f} | "
            f"point=({nearest_lat:.2f}, {nearest_lon:.2f})"
        )
        ax.text(
            0.01,
            0.98,
            summary,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    axes[-1].set_xlabel("Value")
    fig.suptitle(
        "AIFS2ENS pointwise ensemble PDFs\n"
        f"Nearest grid point to ({args.lat:.2f}, {args.lon:.2f}) -> ({nearest_lat:.2f}, {nearest_lon:.2f}) | "
        f"time index {args.time_index}, lead index {args.lead_index} | valid time {valid_time_str} | "
        f"precip max in box +/- {args.precip_box_deg:.1f}°",
        fontsize=13,
        weight="bold",
    )

    if args.output is None:
        var_tag = "_".join(args.variables)
        output = (
            f"aifs_point_pdfs_{var_tag}_lat{args.lat:.2f}_lon{args.lon:.2f}"
            f"_t{args.time_index}_l{args.lead_index}.png"
        )
    else:
        output = args.output

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()