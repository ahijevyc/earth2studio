"""Ensemble-mean RMSE vs. IFS analysis truth for Arm B, Global and CONUS.

Reuses the same ifs_truth_cache.zarr built for Arm A -- same init dates,
same lead times, same variables, so no need to refetch truth data.
"""

import numpy as np
import zarr

VARIABLES = ["t850", "t2m", "u850", "z500"]
LAT_MAX, LAT_MIN = 60, 20
LON_MIN, LON_MAX = 220, 300


def main():
    fcst = zarr.open_group("arm_b_cache.zarr", mode="r")
    truth = zarr.open_group("ifs_truth_cache.zarr", mode="r")

    init_times = fcst["time"][:]
    lead_times = fcst["lead_time"][:]
    lat = fcst["lat"][:]
    lon = fcst["lon"][:]
    n_time = len(init_times)
    n_lead = len(lead_times)

    lat_mask = (lat <= LAT_MAX) & (lat >= LAT_MIN)
    lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    conus_idx = np.ix_(lat_mask, lon_mask)

    truth_valid_times = truth["valid_time"][:]
    vt_to_idx = {vt: i for i, vt in enumerate(truth_valid_times)}

    regions = ["global", "conus"]
    rmse = {(v, r): np.zeros(n_lead) for v in VARIABLES for r in regions}
    n_missing = 0

    for li in range(n_lead):
        lead_dt = lead_times[li]
        mse_accum = {(v, r): [] for v in VARIABLES for r in regions}
        for ti in range(n_time):
            valid_time = init_times[ti] + lead_dt
            if valid_time not in vt_to_idx:
                n_missing += 1
                continue
            vt_idx = vt_to_idx[valid_time]
            for v in VARIABLES:
                fcst_members = fcst[v][:, ti, li, :, :]
                fcst_mean = fcst_members.mean(axis=0)
                truth_field = truth[v][vt_idx, :, :]
                sq_err = (fcst_mean - truth_field) ** 2
                mse_accum[(v, "global")].append(np.nanmean(sq_err))
                mse_accum[(v, "conus")].append(np.nanmean(sq_err[conus_idx]))
        for v in VARIABLES:
            for r in regions:
                rmse[(v, r)][li] = np.sqrt(np.mean(mse_accum[(v, r)]))
        print(f"done lead {li + 1}/{n_lead}: {lead_dt}")

    if n_missing:
        print(f"WARNING: {n_missing} (init, lead) combos had no matching truth valid_time")

    lead_hours = lead_times / np.timedelta64(1, "h")
    save_kwargs = {"lead_hours": lead_hours}
    for v in VARIABLES:
        for r in regions:
            save_kwargs[f"{v}_{r}_rmse"] = rmse[(v, r)]

    np.savez("arm_b_rmse.npz", **save_kwargs)
    print("Saved arm_b_rmse.npz")
    for v in VARIABLES:
        for r in regions:
            print(v, r, "rmse:", save_kwargs[f"{v}_{r}_rmse"])


if __name__ == "__main__":
    main()
