import os
import shutil

import numpy as np
import xarray as xr
import zarr

from earth2studio.models.px import AIFS2ENS
from earth2studio.data import GFS, ARCO, IFS
from earth2studio.io import ZarrBackend
from earth2studio.lexicon import ARCOLexicon, IFSLexicon
from earth2studio.run import deterministic

# Tried to write all members to the same zarr but it
# never saved ensemble dimension. deterministic didn't support it.
# Tried using chunks dictionary as in earth2studio documentation but still failed.
class ERA5WithGFSFallback:
    """
    ARCO = Analysis-Ready Cloud-Optimized ERA5
    Prefer a primary ERA5 source (ARCO or IFS) and fetch only missing variables from GFS.
    """

    def __init__(self, primary_source: str = "ARCO", verbose: bool = True) -> None:
        self.verbose = verbose
        self.gfs = GFS()

        primary_source = primary_source.upper()
        if primary_source == "IFS":
            self.primary = IFS()
            self.lexicon = IFSLexicon
            self.source_name = "IFS"
        else:
            self.primary = ARCO()
            self.lexicon = ARCOLexicon
            self.source_name = "ARCO"

    def __call__(self, time, variable):
        requested = [variable] if isinstance(variable, str) else list(variable)

        primary_vars = []
        gfs_vars = []
        for v in requested:
            try:
                _ = self.lexicon[v]
                primary_vars.append(v)
            except (KeyError, AttributeError):
                gfs_vars.append(v)

        if self.verbose and gfs_vars:
            print(
                f"{self.source_name} missing variables; falling back to GFS for: "
                + ", ".join(gfs_vars)
            )

        data_parts = []
        if primary_vars:
            data_parts.append(self.primary(time, primary_vars))
        if gfs_vars:
            data_parts.append(self.gfs(time, gfs_vars))

        if len(data_parts) == 1:
            return data_parts[0]

        merged = xr.concat(data_parts, dim="variable")
        return merged.sel(variable=requested)


def append_along_time(fout: str, tmp_out: str) -> None:
    """Append a single new init time (tmp_out) onto an existing zarr store (fout).

    Writes raw values directly via zarr instead of xr.Dataset.to_zarr(mode="a",
    append_dim="time"). ZarrBackend creates the "time" coordinate with a native
    zarr datetime64 dtype and no CF units/calendar attrs. xarray's append path
    re-derives CF encoding for the appended value from scratch (since the
    existing store carries no CF units to inherit), and a single-timestamp
    array always CF-encodes as 0 relative to itself -- silently corrupting the
    appended time coordinate to 1970-01-01 while leaving the data variables
    intact. Raw zarr writes avoid CF encoding entirely.
    """
    dst = zarr.open_group(fout, mode="a")
    src = zarr.open_group(tmp_out, mode="r")

    old_len = dst["time"].shape[0]
    new_len = old_len + 1

    for name in dst.array_keys():
        if name not in src:
            continue
        dims = dst[name].metadata.dimension_names
        if dims is None or "time" not in dims:
            continue
        axis = dims.index("time")
        new_shape = list(dst[name].shape)
        new_shape[axis] += 1
        dst[name].resize(new_shape)

        region = [slice(None)] * len(dims)
        region[axis] = slice(old_len, new_len)
        dst[name][tuple(region)] = src[name][:]


def main():
    package = AIFS2ENS.load_default_package()
    data_source = os.environ.get("AIFS_DATA_SOURCE", "ERA5_GFS").upper()
    if data_source == "GFS":
        data = GFS()
    else:
        data = ERA5WithGFSFallback() #primary_source="IFS")

    start_time_env = os.environ.get("AIFS_START_TIMES", "2025-06-24 00:00")
    start_times = [s.strip() for s in start_time_env.split(",") if s.strip()]
    overwrite = os.environ.get("AIFS_OVERWRITE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    print(f"Run init times: {start_times}")
    print(f"Overwrite existing members: {overwrite}")

    nsteps = 12
    num_members = 500
    for member in range(num_members):
        print(member)
        model = AIFS2ENS.load_model(package, seed=14+member)
        fout = f"{model.__class__.__name__}_{data.__class__.__name__}_member_{member:03d}.zarr"

        if overwrite and os.path.exists(fout):
            print(f"Overwriting existing member store: {fout}")
            if os.path.isdir(fout):
                shutil.rmtree(fout)
            else:
                os.remove(fout)

        for init_time in start_times:
            init_np = np.datetime64(init_time)

            if os.path.exists(fout):
                ds_existing = xr.open_zarr(fout, consolidated=False)
                already_present = init_np in ds_existing.time.values
                ds_existing.close()
                if already_present:
                    print(f"Skipping existing init time {init_time} in {fout}")
                    continue

            init_tag = init_time.replace("-", "").replace(":", "").replace(" ", "_")
            tmp_out = f"{fout}.tmp_{init_tag}"
            if os.path.exists(tmp_out):
                shutil.rmtree(tmp_out)

            io = ZarrBackend(tmp_out, backend_kwargs={"overwrite": True})
            print(
                "Starting AIFS2ENS multi-member inference via Earth2Studio ensemble workflow... "
                f"member={member:03d} init={init_time}"
            )
            deterministic(
                time=[init_time],
                nsteps=nsteps,
                prognostic=model,
                data=data,
                io=io,
            )

            if not os.path.exists(fout):
                os.rename(tmp_out, fout)
            else:
                append_along_time(fout, tmp_out)
                shutil.rmtree(tmp_out)

if __name__ == "__main__":
    main()
