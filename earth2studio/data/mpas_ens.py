# =============================================================================
# Imports
# =============================================================================
import datetime
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from earth2studio.data.mpas import _MPASBase
from earth2studio.data.utils import datasource_cache_root
from earth2studio.lexicon.mpas import MPASLexicon


class MPASEnsemble(_MPASBase):
    """
    A custom earth2studio data source for loading, processing, and regridding
    MPAS model *ensemble* output from HWT. This class handles the unstructured MPAS grid
    by regridding it to a regular lat-lon grid using a nearest-neighbor
    approach with a KDTree. It supports fetching data for multiple ensemble members
    at specific forecast hours.

    Attributes
    ----------
    data_path : str
        The *base directory* containing MPAS output files. The class expects
        a specific subdirectory structure, e.g.:
        `{data_path}/{YYYYMMDDHH}/post/mem_{mem}/diag.{YYYY-MM-DD_HH.MM.SS}.nc`
    grid_path : Path
        The path to the static MPAS grid definition file.
    cache_path : Path, optional
        The directory to store cached regridding indices.
    """

    cache_path: Path = Path(datasource_cache_root()) / "mpas_ensemble"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.lexicon = MPASLexicon

    @lru_cache(maxsize=16)
    def _load_and_regrid(
        self,
        itime: datetime.datetime,
        fhr: int,
        variables: tuple[str],
        members: tuple[int],
    ) -> xr.Dataset:
        """
        Loads data for a given initialization time, forecast hour, and member set,
        then regrids it. Caches the result to avoid redundant I/O and computation.
        """
        # Ensure itime is a pandas Timestamp for arithmetic and consistent formatting
        itime = pd.to_datetime(itime)
        valid_time = itime + pd.Timedelta(hours=fhr)

        data_paths = [
            f"{self.data_path}/mem_{mem}/diag.{valid_time.strftime('%Y-%m-%d_%H.%M.%S')}.nc"
            for mem in members
        ]

        existing_paths = [p for p in data_paths if os.path.exists(p)]
        self.existing_paths = existing_paths
        if not existing_paths:
            # Create a formatted string of attempted paths for the error message
            attempted_paths_str = "\n".join(map(str, data_paths))
            raise FileNotFoundError(
                f"No MPAS files found for itime={itime}, fhr={fhr}.\n"
                f"Attempted paths:\n{attempted_paths_str}"
            )

        # Determine the required source variables from the requested variables
        source_variables = self.lexicon.required_variables(list(variables))
        logger.info(f"Requesting source variables: {source_variables}")

        logger.info(f"Opening {len(existing_paths)} member files.")

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            """
            Preprocessor to add metadata, derive variables, and then immediately
            filter the dataset to only contain variables required for this request.
            This prevents carrying around unneeded data.
            """
            # Add member coordinate
            match = re.search(r"mem_?(\d+)", ds.encoding["source"])
            mem = int(match.group(1)) if match else -1
            ds = ds.expand_dims({"member": [mem]})

            # First, reduce the dataset to only the raw variables needed for derivation.
            # This prevents derive_variables from working on the entire file.
            raw_vars_to_load = [v for v in source_variables if v in ds.data_vars]
            ds_filtered = ds[raw_vars_to_load]

            # Derive the necessary variables from the filtered dataset
            ds_derived = self.lexicon.derive_variables(ds_filtered)

            # Now, filter again to the final list of source variables
            final_vars_to_keep = [
                v for v in source_variables if v in ds_derived.data_vars
            ]
            return ds_derived[final_vars_to_keep]

        # Open the dataset lazily
        ds_mpas = xr.open_mfdataset(
            existing_paths,
            combine="nested",
            concat_dim="member",
            preprocess=preprocess,
            parallel=True,
            engine="netcdf4",
        )

        ds_mpas = ds_mpas.squeeze("Time")

        # Validate that the data grid matches the regridding indices
        if ds_mpas.sizes.get("nCells") != self.grid_ncells:
            raise ValueError(
                f"Grid mismatch: The grid file has {self.grid_ncells} cells, "
                f"but data files have {ds_mpas.sizes['nCells']} cells."
            )

        # Regrid the dataset
        logger.info("Regridding dataset (lazy)...")
        # Do NOT load before regridding. Regrid first (reduces size), then compute.
        ds_regridded = self._regrid_dataset(ds_mpas)

        logger.info("Computing/Loading regridded data into memory...")
        # Now we trigger the computation. This pulls only the necessary data points
        # from the source files to fill the target grid, rather than reading everything.
        ds_regridded = ds_regridded.compute()

        logger.info("Regridding complete.")
        return ds_regridded

    def _finalize_dataset(
        self,
        ds_regridded: xr.Dataset,
        variables: list[str],
        time: datetime.datetime,
        fhr: int,
    ) -> xr.DataArray:
        """Builds the final DataArray from a processed, regridded Dataset."""
        # Map e2s variables to the source variable names from the lexicon
        rename_dict = {self.lexicon.get_item(var): var for var in variables}

        # Select only the needed variables and rename them to e2s standard names
        ds_final = ds_regridded[list(rename_dict.keys())].rename(rename_dict)

        # Add metadata for context
        ds_final.attrs["initialization_time"] = str(time)
        ds_final.attrs["forecast_hour"] = fhr

        # Convert to a single DataArray with a 'variable' dimension
        return ds_final.to_dataarray(dim="variable")

    def __call__(
        self,
        time: datetime.datetime | list[datetime.datetime] | np.ndarray,
        variables: list[str],
        *,
        fhr: int = 0,
        members: list[int] | None = None,
    ) -> xr.DataArray:
        """
        The main entry point for fetching data. It loads, processes, regrids,
        and returns the requested variables as an xarray.DataArray.

        Note: This method is async and designed to be called by earth2studio's
        async runners. It fetches data for a *single initialization time*.

        Parameters
        ----------
        time : datetime.datetime | list[datetime.datetime] | np.ndarray
            The initialization time(s) of the forecast.
        variables : List[str]
            A list of standardized variable names to fetch.
        fhr : int, optional
            Forecast hour to load, by default 0.
        members : List[int], optional
            List of ensemble members to load, by default list(range(1, 11)).

        Returns
        -------
        xr.DataArray
            A DataArray containing the requested data, regridded to a regular
            lat-lon grid.
        """
        if members is None:
            members_tuple = tuple(range(1, 11))
        else:
            members_tuple = tuple(sorted(members))

        # Pass variables and members as tuples so they can be hashed for the cache.
        sorted_variables = tuple(sorted(variables))

        if isinstance(time, (datetime.datetime, np.datetime64)):
            ds = self._load_and_regrid(time, fhr, sorted_variables, members_tuple)
            return self._finalize_dataset(ds, variables, time, fhr)
        else:
            results = []
            for t in time:
                ds = self._load_and_regrid(t, fhr, sorted_variables, members_tuple)
                da = self._finalize_dataset(ds, variables, t, fhr)
                da = da.assign_coords(time=t).expand_dims("time")
                results.append(da)

        if not results:
            return xr.DataArray()

        return xr.concat(results, dim="time")
