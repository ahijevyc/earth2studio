"""Ensemble spread, computed the way Fortin, Abaza, Anctil & Turcotte (2014)
show is actually justified: average the ensemble VARIANCE (across space AND
across init dates), then take the square root exactly once at the end --
not average-of-standard-deviations, which underestimates spread relative to
RMSE (Jensen's inequality: mean(sqrt(x)) <= sqrt(mean(x))) and is
inconsistent with how RMSE itself is computed (average MSE, sqrt once).

Two distinct finite-ensemble-size corrections are applied, not one:
  1. ddof=1 (Bessel's correction, R-1 denominator) -- unbiased estimate of
     the ensemble's own population variance from R samples.
  2. The exchangeability factor (R+1)/R -- under the assumption that truth
     behaves like an (R+1)th member exchangeable with the ensemble,
     E[(ensemble mean - truth)^2] = sigma^2 * (R+1)/R, not sigma^2. Without
     this, even a perfectly reliable ensemble would show spread/RMSE topping
     out at sqrt(R/(R+1)) < 1, not 1. Small for R=50 (~1%) but real, and
     distinct from (1) -- applied on top of it, not instead of it.

Shared script for both arms -- set CACHE_PATH/OUT_PATH via sys.argv.
"""

import sys

import numpy as np
import zarr

VARIABLES = ["t850", "t2m", "u850", "z500"]
LAT_MAX, LAT_MIN = 60, 20
LON_MIN, LON_MAX = 220, 300


def main(cache_path, out_path):
    root = zarr.open_group(cache_path, mode="r")

    lead_times = root["lead_time"][:]
    lat = root["lat"][:]
    lon = root["lon"][:]
    n_time = root["time"].shape[0]
    n_lead = len(lead_times)
    n_members = root["member"].shape[0]
    exchangeability_factor = (n_members + 1) / n_members
    print(
        f"R={n_members} members; exchangeability factor (R+1)/R="
        f"{exchangeability_factor:.4f}, sqrt={np.sqrt(exchangeability_factor):.4f} "
        "(the ~1% spread inflation this adds on top of the ddof=1 correction)"
    )

    lat_mask = (lat <= LAT_MAX) & (lat >= LAT_MIN)
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    conus_idx = np.ix_(lat_mask, lon_mask)

    regions = ["global", "conus"]
    # variance (not spread) per init/lead -- averaged across space only so far
    var_by_init = {(v, r): np.zeros((n_time, n_lead)) for v in VARIABLES for r in regions}
    var_by_init_uncorrected = {(v, r): np.zeros((n_time, n_lead)) for v in VARIABLES for r in regions}

    for ti in range(n_time):
        for li in range(n_lead):
            for v in VARIABLES:
                members = root[v][:, ti, li, :, :]  # (member, lat, lon)
                member_var = np.nanvar(members, axis=0, ddof=1)  # variance at each grid point
                var_by_init_uncorrected[(v, "global")][ti, li] = np.nanmean(member_var)
                var_by_init_uncorrected[(v, "conus")][ti, li] = np.nanmean(member_var[conus_idx])
                member_var = member_var * exchangeability_factor
                var_by_init[(v, "global")][ti, li] = np.nanmean(member_var)
                var_by_init[(v, "conus")][ti, li] = np.nanmean(member_var[conus_idx])
        print(f"done init {ti + 1}/{n_time}")

    lead_hours = lead_times / np.timedelta64(1, "h")
    save_kwargs = {"lead_hours": lead_hours}
    for v in VARIABLES:
        for r in regions:
            var_arr = var_by_init[(v, r)]  # (n_init, n_lead)
            # average variance across inits, THEN sqrt once -- matches RMSE's
            # own average-MSE-then-sqrt convention, so the ratio is meaningful
            save_kwargs[f"{v}_{r}_mean"] = np.sqrt(var_arr.mean(axis=0))
            # per-init spread (for reference/diagnostics only), each already
            # sqrt'd from its own spatial-mean variance -- fine at this level
            # since it's not being averaged further across anything after
            save_kwargs[f"{v}_{r}_by_init"] = np.sqrt(var_arr)
            save_kwargs[f"{v}_{r}_std_across_inits"] = np.sqrt(var_arr).std(axis=0, ddof=1)

    np.savez(out_path, **save_kwargs)
    print(f"Saved {out_path}")
    for v in VARIABLES:
        for r in regions:
            print(v, r, "mean:", save_kwargs[f"{v}_{r}_mean"])

    print(
        "\nEffect of the (R+1)/R exchangeability factor "
        f"(sqrt={np.sqrt(exchangeability_factor):.4f}) on spread, "
        "with vs. without, at lead=0h and lead=max:"
    )
    for v in VARIABLES:
        for r in regions:
            with_factor = save_kwargs[f"{v}_{r}_mean"]
            without_factor = np.sqrt(var_by_init_uncorrected[(v, r)].mean(axis=0))
            ratio = with_factor / without_factor
            print(
                f"  {v:5s} {r:6s}  lead0: {without_factor[0]:.4f} -> {with_factor[0]:.4f} "
                f"(x{ratio[0]:.4f})   lead_max: {without_factor[-1]:.4f} -> {with_factor[-1]:.4f} "
                f"(x{ratio[-1]:.4f})"
            )


if __name__ == "__main__":
    cache_path, out_path = sys.argv[1], sys.argv[2]
    main(cache_path, out_path)
