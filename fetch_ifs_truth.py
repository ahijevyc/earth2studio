"""Fetch and cache IFS analysis truth for RMSE verification.

Valid times needed = union of (init + lead_time) over all 14 Arm A inits
(2026-06-01..14 daily 00Z) and 21 lead times (0..120h by 6h) -- a dense
6-hourly series from 2026-06-01 00Z to 2026-06-19 00Z (73 unique times),
since the 5-day forecast windows from consecutive daily inits overlap
heavily. Reuses ERA5WithGFSFallback(primary_source="IFS") for the same
hardened per-variable GFS fallback used throughout this project.
"""

import sys

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, "/glade/derecho/scratch/ahijevyc/earth2studio-aifs")
from run_aifs import ERA5WithGFSFallback

VARIABLES = ["t850", "t2m", "u850", "z500"]
CACHE_PATH = "ifs_truth_cache.zarr"

INIT_TIMES = pd.date_range("2026-06-01", "2026-06-14", freq="D")
LEAD_HOURS = range(0, 121, 6)


def main():
    valid_times = sorted(
        {init + pd.Timedelta(hours=h) for init in INIT_TIMES for h in LEAD_HOURS}
    )
    print(f"{len(valid_times)} unique valid times: {valid_times[0]} to {valid_times[-1]}")

    data = ERA5WithGFSFallback(primary_source="IFS")

    da0 = data(valid_times[0], VARIABLES)
    lat = da0.lat.values
    lon = da0.lon.values
    n_lat, n_lon = len(lat), len(lon)

    store = zarr.storage.LocalStore(CACHE_PATH)
    root = zarr.group(store, overwrite=True)
    valid_times_np = np.array([np.datetime64(t) for t in valid_times])

    root.create_array(
        "valid_time", shape=(len(valid_times),), chunks=(len(valid_times),),
        dtype=valid_times_np.dtype, dimension_names=["valid_time"],
    )
    root["valid_time"][:] = valid_times_np
    root.create_array("lat", shape=(n_lat,), chunks=(n_lat,),
                       dtype=lat.dtype, dimension_names=["lat"])
    root["lat"][:] = lat
    root.create_array("lon", shape=(n_lon,), chunks=(n_lon,),
                       dtype=lon.dtype, dimension_names=["lon"])
    root["lon"][:] = lon
    for v in VARIABLES:
        root.create_array(
            v, shape=(len(valid_times), n_lat, n_lon), chunks=(1, n_lat, n_lon),
            dtype=np.float32, dimension_names=["valid_time", "lat", "lon"],
        )

    for i, vt in enumerate(valid_times):
        da = da0 if i == 0 else data(vt, VARIABLES)
        for v in VARIABLES:
            root[v][i, :, :] = da.sel(variable=v).values.astype(np.float32)
        print(f"fetched {i + 1}/{len(valid_times)}: {vt}")

    print(f"Done. Truth cache written to {CACHE_PATH}")


if __name__ == "__main__":
    main()
