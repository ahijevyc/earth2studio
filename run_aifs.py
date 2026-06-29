import os
import shutil

import numpy as np
import xarray as xr

from earth2studio.models.px import AIFS2ENS
from earth2studio.data import GFS, ARCO
from earth2studio.io import ZarrBackend
from earth2studio.lexicon import ARCOLexicon
from earth2studio.run import deterministic

# Tried to write all members to the same zarr but it
# never saved ensemble dimension. deterministic didn't support it.
# Tried using chunks dictionary as in earth2studio documentation but still failed.
class ARCOWithGFSFallback:
    """Prefer ARCO and fetch only missing variables from GFS."""

    def __init__(self, verbose: bool = True) -> None:
        self.arco = ARCO()
        self.gfs = GFS()
        self.verbose = verbose

    def __call__(self, time, variable):
        requested = [variable] if isinstance(variable, str) else list(variable)

        arco_vars = []
        gfs_vars = []
        for v in requested:
            try:
                ARCOLexicon[v]
                arco_vars.append(v)
            except KeyError:
                gfs_vars.append(v)

        if self.verbose and gfs_vars:
            print(
                "ARCO missing variables; falling back to GFS for: "
                + ", ".join(gfs_vars)
            )

        data_parts = []
        if arco_vars:
            data_parts.append(self.arco(time, arco_vars))
        if gfs_vars:
            data_parts.append(self.gfs(time, gfs_vars))

        if len(data_parts) == 1:
            return data_parts[0]

        merged = xr.concat(data_parts, dim="variable")
        return merged.sel(variable=requested)


def main():
    package = AIFS2ENS.load_default_package()
    data_source = os.environ.get("AIFS_DATA_SOURCE", "ARCO_GFS").upper()
    if data_source == "GFS":
        data = GFS()
    elif data_source == "ARCO":
        data = ARCO()
    else:
        data = ARCOWithGFSFallback()

    start_time_env = os.environ.get("AIFS_START_TIMES", "2025-06-24 00:00")
    start_times = [s.strip() for s in start_time_env.split(",") if s.strip()]
    overwrite = os.environ.get("AIFS_OVERWRITE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    print(f"Run init times: {start_times}")
    print(f"Overwrite existing members: {overwrite}")

    nsteps = 4
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
                already_present = bool(np.isin(init_np, ds_existing.time.values).any())
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
                ds_new = xr.open_zarr(tmp_out, consolidated=False)
                ds_new.to_zarr(fout, mode="a", append_dim="time", consolidated=False)
                ds_new.close()
                shutil.rmtree(tmp_out)

if __name__ == "__main__":
    main()
