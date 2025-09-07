# =============================================================================
# Imports
# =============================================================================
import dataclasses
import datetime
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import xarray as xr
from earth2studio.data import DataSource
from earth2studio.data.utils import datasource_cache_root, xtime
from earth2studio.lexicon.mpas import MPASLexicon
from scipy.spatial import KDTree

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# =============================================================================
# Main Data Source Class
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class MPAS(DataSource):
    """
    A custom earth2studio data source for loading, processing, and regridding
    MPAS model output. This class handles the unstructured MPAS grid by
    regridding it to a regular lat-lon grid using a nearest-neighbor approach
    with a KDTree.

    Attributes:
        data_path (Union[Path, List[Path]]): Direct path to the MPAS output file.
        grid_path (Path): Path to the MPAS grid definition file.
        cache_path (Path): Directory to store cached regridding indices.
    """

    data_path: Union[Path, List[Path]]
    grid_path: Path
    cache_path: Path = Path(datasource_cache_root()) / "mpas"

    # Regridding parameters
    d_lon: float = 0.25
    d_lat: float = 0.25

    def __post_init__(self):
        """
        Post-initialization to prepare the target grid and compute or load the
        regridding indices from the cache.
        """
        self.lexicon = MPASLexicon
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Convert data_path to tuple if it's a list to make it hashable for lru_cache
        if isinstance(self.data_path, list):
            self.data_path = tuple(self.data_path)

        # Define and store the target regular grid
        self.target_lon = np.arange(0, 360, self.d_lon)
        self.target_lat = np.arange(90, -90 - self.d_lat, -self.d_lat)

        # Compute or load regridding indices
        self.distance, self.indices = self._prepare_regridding_indices()

        # Load grid cell count to validate against data file
        with xr.open_dataset(self.grid_path) as grid_ds:
            self.grid_ncells = grid_ds.sizes["nCells"]

    # Regridding and Data Loading Methods
    def _prepare_regridding_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates or loads cached nearest neighbor indices for regridding."""
        cached_file = (self.cache_path / self.grid_path.name).with_suffix(".npz")
        if cached_file.exists():
            logging.info(f"Loading cached regridding indices from {cached_file}")
            data = np.load(cached_file)
            return data["dists"], data["inds"]

        logging.info("Building KDTree from MPAS grid to compute regridding indices...")
        with xr.open_dataset(self.grid_path) as grid:
            lon_cell = grid["lonCell"]
            lat_cell = grid["latCell"]

            # Function to determine units and convert to radians if needed
            def process_coords(coord_da: xr.DataArray) -> np.ndarray:
                units = coord_da.attrs.get("units", "unknown").lower()
                values = coord_da.values
                
                if units in ["rad", "radians"]:
                    logging.info(f"{coord_da.name} units are in radians. Using values directly.")
                    return values
                elif units in ["deg", "degrees"]:
                    logging.info(f"{coord_da.name} units are in degrees. Converting to radians.")
                    return np.deg2rad(values)
                else: # No units or unknown units
                    logging.info(f"{coord_da.name} units are not specified. Inferring from value range.")
                    # Check if any values fall outside the typical radian range
                    if np.any(np.abs(values) > 2 * np.pi):
                        logging.info(f"Values found outside [-2*pi, 2*pi]. Assuming degrees and converting.")
                        return np.deg2rad(values)
                    else:
                        logging.info(f"Values are within [-2*pi, 2*pi]. Assuming radians.")
                        return values

            mpas_lon_rad = process_coords(lon_cell)
            mpas_lat_rad = process_coords(lat_cell)
            mpas_xyz = self._lon_lat_to_cartesian(mpas_lon_rad, mpas_lat_rad)

        target_lon_grid, target_lat_grid = np.meshgrid(self.target_lon, self.target_lat)
        target_lon_rad = np.deg2rad(target_lon_grid.ravel())
        target_lat_rad = np.deg2rad(target_lat_grid.ravel())
        target_xyz = self._lon_lat_to_cartesian(target_lon_rad, target_lat_rad)

        kdtree = KDTree(mpas_xyz)
        logging.info("Querying tree to find nearest neighbors...")
        distance, indices = kdtree.query(target_xyz)

        logging.info(f"Saving new regridding indices to {cached_file}")
        np.savez_compressed(cached_file, dists=distance, inds=indices)
        return distance, indices

    def _open_data_dataset(self) -> xr.Dataset:
        """
        Opens the MPAS data file(s). If `data_path` is a list or tuple, it opens
        and combines them into a single xarray Dataset.
        """
        if isinstance(self.data_path, (list, tuple)):
            if not self.data_path:
                raise ValueError("data_path cannot be an empty list.")
            # Check for file existence first
            for p in self.data_path:
                if not p.exists():
                    raise FileNotFoundError(f"MPAS file not found at: {p}")
            logging.info(f"Opening and combining {len(self.data_path)} data files.")
            datasets = [xtime(xr.open_dataset(p)) for p in self.data_path]
            return xr.merge(datasets, combine_attrs="drop_conflicts")
        else:
            if not self.data_path.exists():
                raise FileNotFoundError(f"MPAS file not found at: {self.data_path}")
            return xtime(xr.open_dataset(self.data_path))

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
        time: datetime.datetime,
        variables: Tuple[str],
    ) -> xr.Dataset:
        """
        Loads data from the specified data_path, derives variables, and
        regrids it. Caches the result to avoid redundant I/O and computation.
        """

        # Determine the required source variables from the requested variables
        source_variables = self.lexicon.required_variables(list(variables))
        logging.info(f"Requesting source variables: {source_variables}")

        with self._open_data_dataset() as ds_mpas:
            # Select requested time from dataset
            ds_mpas = ds_mpas.sel(time=time)

            # Validate that the data grid matches the regridding indices
            if ds_mpas.sizes.get("nCells") != self.grid_ncells:
                raise ValueError(
                    f"Grid mismatch: The grid file has {self.grid_ncells} cells, "
                    f"but the data file has {ds_mpas.sizes['nCells']} cells."
                )

            # Filter to raw variables needed, derive, then filter to final source vars
            raw_vars_to_load = [v for v in source_variables if v in ds_mpas.data_vars]
            ds_filtered = ds_mpas[raw_vars_to_load]
            ds_derived = self.lexicon.derive_variables(ds_filtered)
            final_vars_to_keep = [v for v in source_variables if v in ds_derived.data_vars]
            ds_processed = ds_derived[final_vars_to_keep]

            # Regrid the dataset
            logging.info("Loading data into memory and regridding...")
            ds_regridded = self._regrid_dataset(ds_processed.load())
            logging.info("Regridding complete.")
            return ds_regridded

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """Remaps from the unstructured grid to a regular lat-lon grid."""
        # Select data at the nearest neighbor indices
        regridded_data = ds_mpas.isel(nCells=self.indices)

        # Determine the new shape for the lat/lon grid
        # Preserves any dimensions other than nCells (e.g., nVertLevels)
        new_shape = [
            size for dim, size in regridded_data.sizes.items() if dim != "nCells"
        ] + [len(self.target_lat), len(self.target_lon)]

        # Create new coordinates, preserving existing ones
        new_coords = {
            dim: coord for dim, coord in regridded_data.coords.items() if dim != "nCells"
        }
        new_coords["lat"] = self.target_lat
        new_coords["lon"] = self.target_lon

        # Build new dimensions, replacing nCells with lat/lon
        new_dims = [dim for dim in regridded_data.dims if dim != "nCells"] + [
            "lat",
            "lon",
        ]

        # Reshape each variable in the dataset
        regridded_vars = {}
        for var_name, da in regridded_data.data_vars.items():
            reshaped_values = da.values.reshape(new_shape)
            regridded_vars[var_name] = xr.DataArray(
                reshaped_values, coords=new_coords, dims=new_dims
            )

        return xr.Dataset(regridded_vars, attrs=ds_mpas.attrs)

    async def __call__(
        self,
        time: datetime.datetime,
        variables: List[str],
    ) -> xr.DataArray:
        """
        The main entry point for fetching data. It loads, processes, regrids,
        and returns the requested variables as an xarray.DataArray.

        Parameters
        ----------
        time : datetime.datetime
            The time to select from data file(s).
        variables : List[str]
            A list of standardized variable names to fetch.

        Returns
        -------
        xr.DataArray
            A DataArray containing the requested data, regridded to a regular
            lat-lon grid.
        """
        # Load and regrid the data. The result is cached by the helper method.
        # Pass variables as a tuple so it can be hashed for the cache.
        ds_regridded = self._load_and_regrid(time, tuple(sorted(variables)))

        # Map e2s variables to the source variable names from the lexicon
        rename_dict = {self.lexicon[var]: var for var in variables}

        # Select only the needed variables and rename them to e2s standard names
        ds_final = ds_regridded[list(rename_dict.keys())].rename(rename_dict)

        # Add metadata for context
        ds_final.attrs["time"] = str(time)
        ds_final.attrs["source_file"] = str(self.data_path)

        # Convert to a single DataArray with a 'variable' dimension
        return ds_final.to_dataarray(dim="variable")

