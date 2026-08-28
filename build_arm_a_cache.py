"""Consolidate the 50 separate Arm A member zarr stores into one compact
cache, so future analyses (spread-by-region, RMSE-vs-truth) don't each pay
the ~46 min cost of re-reading 50 stores individually.

Chunked as (member=all 50, time=1, lead_time=1, lat=full, lon=full) --
minimizes file count and matches the actual access pattern (every
downstream computation needs all 50 members at once for a given time/lead).

IMPORTANT: loop order is (time, lead_time) outer, member inner -- each
chunk spans all 50 members, so it must be written COMPLETE in one shot.
Writing member-by-member (chunk written 50 times, each requiring a read-
modify-write of the whole chunk since a partial update can't be encoded
independently) blew this up to >1 TB of I/O for a ~244 GB cache before
this was caught and fixed.
"""

import numpy as np
import xarray as xr
import zarr

VARIABLES = ["t850", "t2m", "u850", "z500"]
NUM_MEMBERS = 50
CACHE_PATH = "arm_a_cache.zarr"


def main():
    member_files = [
        f"AIFS2ENS_IFSControl_member_{m:03d}.zarr" for m in range(NUM_MEMBERS)
    ]
    handles = [xr.open_zarr(f, consolidated=False) for f in member_files]

    lead_times = handles[0].lead_time.values
    init_times = handles[0].time.values
    lat = handles[0].lat.values
    lon = handles[0].lon.values
    n_time, n_lead, n_lat, n_lon = len(init_times), len(lead_times), len(lat), len(lon)

    store = zarr.storage.LocalStore(CACHE_PATH)
    root = zarr.group(store, overwrite=True)

    root.create_array("member", shape=(NUM_MEMBERS,), chunks=(NUM_MEMBERS,),
                       dtype=np.int32, dimension_names=["member"])
    root["member"][:] = np.arange(NUM_MEMBERS, dtype=np.int32)
    root.create_array("time", shape=(n_time,), chunks=(n_time,),
                       dtype=init_times.dtype, dimension_names=["time"])
    root["time"][:] = init_times
    root.create_array("lead_time", shape=(n_lead,), chunks=(n_lead,),
                       dtype=lead_times.dtype, dimension_names=["lead_time"])
    root["lead_time"][:] = lead_times
    root.create_array("lat", shape=(n_lat,), chunks=(n_lat,),
                       dtype=lat.dtype, dimension_names=["lat"])
    root["lat"][:] = lat
    root.create_array("lon", shape=(n_lon,), chunks=(n_lon,),
                       dtype=lon.dtype, dimension_names=["lon"])
    root["lon"][:] = lon

    for v in VARIABLES:
        root.create_array(
            v,
            shape=(NUM_MEMBERS, n_time, n_lead, n_lat, n_lon),
            chunks=(NUM_MEMBERS, 1, 1, n_lat, n_lon),
            dtype=np.float32,
            dimension_names=["member", "time", "lead_time", "lat", "lon"],
        )

    for ti, it in enumerate(init_times):
        for li, lt in enumerate(lead_times):
            for v in VARIABLES:
                stacked = np.stack(
                    [h[v].sel(time=it, lead_time=lt).values for h in handles],
                    axis=0,
                ).astype(np.float32)
                root[v][:, ti, li, :, :] = stacked
        print(f"cached init {ti + 1}/{n_time}: {it}")

    for h in handles:
        h.close()

    print(f"Done. Cache written to {CACHE_PATH}")


if __name__ == "__main__":
    main()
