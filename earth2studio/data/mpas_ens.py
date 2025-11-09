# =============================================================================
# Imports
# =============================================================================
import asyncio
import dataclasses
import datetime
import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

from earth2studio.data import DataSource
from earth2studio.data.utils import datasource_cache_root
from earth2studio.lexicon.mpas import MPASLexicon

# =============================================================================
# Logging Configuration
# =============================================================================
# Use logging, consistent with the original mpas_ens.py
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Main Data Source Class
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class MPASEnsemble(DataSource):
    """
    A custom earth2studio data source for loading, processing, and regridding
    MPAS model *ensemble* output. This class handles the unstructured MPAS grid
    by regridding it to a regular lat-lon grid using a nearest-neighbor
    approach with a KDTree. It supports fetching data for multiple ensemble members
    at specific forecast hours.

    Attributes
    ----------
    data_path : Path
        The *base directory* containing MPAS output files. The class expects
        a specific subdirectory structure, e.g.:
        `{data_path}/{YYYYMMDDHH}/post/mem_{mem}/diag.{YYYY-MM-DD_HH.MM.SS}.nc`
    grid_path : Path
        Path to the MPAS grid definition file.
    d_lon : float, optional
        Target longitude spacing for regridding. Defaults to 0.25.
    d_lat : float, optional
        Target latitude spacing for regridding. Defaults to 0.25.
    cache_path : Path, optional
        Directory to store cached regridding indices.
    """

    data_path: Path
    grid_path: Path
    d_lon: float = 0.25
    d_lat: float = 0.25
    cache_path: Path = Path(datasource_cache_root()) / "mpas_ensemble"

    def __post_init__(self) -> None:
        """
        Post-initialization to prepare the target grid and compute or load the
        regridding indices from the cache.
        """
        self.lexicon = MPASLexicon
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Use np.linspace for robust grid generation (from mpas.py)
        n_lon = int(360 / self.d_lon)
        n_lat = int(180 / self.d_lat) + 1
        self.target_lon = np.linspace(0, 360, n_lon, endpoint=False)
        self.target_lat = np.linspace(90, -90, n_lat)

        # Compute or load regridding indices
        self.distance, self.indices = self._prepare_regridding_indices()

        # Create a target index for xarray's advanced indexing (from mpas.py)
        self.target_grid_index = xr.DataArray(
            self.indices,
            dims=["lat_lon"],
            coords={
                "lat": (
                    "lat_lon",
                    np.meshgrid(self.target_lon, self.target_lat)[1].ravel(),
                ),
                "lon": (
                    "lat_lon",
                    np.meshgrid(self.target_lon, self.target_lat)[0].ravel(),
                ),
            },
        ).set_index(lat_lon=["lat", "lon"])

        # Load grid cell count to validate against data file
        with xr.open_dataset(self.grid_path) as grid_ds:
            self.grid_ncells = grid_ds.sizes["nCells"]

    # Regridding and Data Loading Methods
    def _prepare_regridding_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates or loads cached nearest neighbor indices for regridding."""
        # Use cache file naming from mpas.py
        cache_file_name = f"{self.grid_path.stem}_{self.d_lon}x{self.d_lat}.npz"
        cached_file = self.cache_path / cache_file_name

        if cached_file.exists():
            logger.info(f"Loading cached regridding indices from {cached_file}")
            data = np.load(cached_file)
            return data["dists"], data["inds"]

        logger.info("Building KDTree from MPAS grid to compute regridding indices...")
        with xr.open_dataset(self.grid_path) as grid:
            lon_cell = grid["lonCell"]
            lat_cell = grid["latCell"]

            # Function to determine units and convert to radians if needed
            def process_coords(coord_da: xr.DataArray) -> np.ndarray:
                units = coord_da.attrs.get("units", "unknown").lower()
                values = coord_da.values

                if units in ["rad", "radians"]:
                    logger.info(
                        f"{coord_da.name} units are in radians. Using values directly."
                    )
                    return values
                elif units in ["deg", "degrees"]:
                    logger.info(
                        f"{coord_da.name} units are in degrees. Converting to radians."
                    )
                    return np.deg2rad(values)
                else:  # No units or unknown units
                    logger.info(
                        f"{coord_da.name} units are not specified. Inferring from value range."
                    )
                    # Check if any values fall outside the typical radian range
                    if np.any(np.abs(values) > 2 * np.pi):
                        logger.info(
                            "Values found outside [-2*pi, 2*pi]. Assuming degrees and converting."
                        )
                        return np.deg2rad(values)
                    else:
                        logger.info(
                            "Values are within [-2*pi, 2*pi]. Assuming radians."
                        )
                        return values

            mpas_lon_rad = process_coords(lon_cell)
            mpas_lat_rad = process_coords(lat_cell)
            mpas_xyz = self._lon_lat_to_cartesian(mpas_lon_rad, mpas_lat_rad)

        target_lon_grid, target_lat_grid = np.meshgrid(self.target_lon, self.target_lat)
        target_lon_rad = np.deg2rad(target_lon_grid.ravel())
        target_lat_rad = np.deg2rad(target_lat_grid.ravel())
        target_xyz = self._lon_lat_to_cartesian(target_lon_rad, target_lat_rad)

        kdtree = KDTree(mpas_xyz)
        logger.info("Querying tree to find nearest neighbors...")
        distance, indices = kdtree.query(target_xyz)

        logger.info(f"Saving new regridding indices to {cached_file}")
        np.savez_compressed(cached_file, dists=distance, inds=indices)
        return distance, indices

    @staticmethod
    def _lon_lat_to_cartesian(lon_rad: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
        """Converts lon/lat (radians) to 3D Cartesian coords for KDTree."""
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z]).T

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
        valid_time = itime + pd.Timedelta(hours=fhr)
        data_paths = [
            self.data_path
            / f"{itime.strftime('%Y%m%d%H')}/post/mem_{mem}/diag.{valid_time.strftime('%Y-%m-%d_%H.%M.%S')}.nc"
            for mem in members
        ]

        existing_paths = [p for p in data_paths if p.exists()]
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
            match = re.search(r"mem_(\d+)", ds.encoding["source"])
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

        # Validate that the data grid matches the regridding indices
        if ds_mpas.sizes.get("nCells") != self.grid_ncells:
            raise ValueError(
                f"Grid mismatch: The grid file has {self.grid_ncells} cells, "
                f"but data files have {ds_mpas.sizes['nCells']} cells."
            )

        # Regrid the dataset
        logger.info("Loading data into memory and regridding...")
        # Use the unstack method (from mpas.py)
        ds_regridded = self._regrid_dataset(ds_mpas.load())
        logger.info("Regridding complete.")
        return ds_regridded

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """Remaps from the unstructured grid to a regular lat-lon grid."""
        # Use the faster, cleaner isel/unstack method from mpas.py
        regridded_da = ds_mpas.isel(nCells=self.target_grid_index)
        return regridded_da.unstack("lat_lon")

    def _finalize_dataset(
        self,
        ds_regridded: xr.Dataset,
        variables: list[str],
        time: datetime.datetime,
        fhr: int,
    ) -> xr.DataArray:
        """Builds the final DataArray from a processed, regridded Dataset."""
        # Map e2s variables to the source variable names from the lexicon
        rename_dict = {self.lexicon[var]: var for var in variables}

        # Select only the needed variables and rename them to e2s standard names
        ds_final = ds_regridded[list(rename_dict.keys())].rename(rename_dict)

        # Add metadata for context
        ds_final.attrs["initialization_time"] = str(time)
        ds_final.attrs["forecast_hour"] = fhr

        # Convert to a single DataArray with a 'variable' dimension
        return ds_final.to_dataarray(dim="variable")

    async def __call__(
        self,
        time: datetime.datetime,
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
        time : datetime.datetime
            The initialization time of the forecast.
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

        # Run the synchronous, CPU/IO-bound load function in a separate thread
        # This prevents it from blocking the asyncio event loop.
        ds_regridded = await asyncio.to_thread(
            self._load_and_regrid, time, fhr, sorted_variables, members_tuple
        )

        # Finalize the dataset (renaming, attrs, to_dataarray)
        return self._finalize_dataset(ds_regridded, variables, time, fhr)
