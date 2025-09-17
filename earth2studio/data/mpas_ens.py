# =============================================================================
# Imports
# =============================================================================
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
    with a KDTree. It supports fetching data for multiple ensemble members.

    Attributes:
        data_dir (Path): Directory containing MPAS output files.
        grid_path (Path): Path to the MPAS grid definition file.
        cache_path (Path): Directory to store cached regridding indices.
    """

    data_dir: Path
    grid_path: Path
    cache_path: Path = Path(datasource_cache_root()) / "mpas"

    # Regridding parameters
    d_lon: float = 0.25
    d_lat: float = 0.25

    def __post_init__(self) -> None:
        """
        Post-initialization to prepare the target grid and compute or load the
        regridding indices from the cache.
        """
        self.lexicon = MPASLexicon
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Define and store the target regular grid
        self.target_lon = np.arange(0, 360, self.d_lon)
        self.target_lat = np.arange(90, -90 - self.d_lat, -self.d_lat)

        # Compute or load regridding indices
        self.distance, self.indices = self._prepare_regridding_indices()

        # Load grid cell count to validate against data file
        with xr.open_dataset(self.grid_path) as grid_ds:
            self.grid_ncells = grid_ds.sizes["nCells"]

    # Regridding and Data Loading Methods
    def _prepare_regridding_indices(self) -> tuple[np.ndarray, np.ndarray]:
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
                    logging.info(
                        f"{coord_da.name} units are in radians. Using values directly."
                    )
                    return values
                elif units in ["deg", "degrees"]:
                    logging.info(
                        f"{coord_da.name} units are in degrees. Converting to radians."
                    )
                    return np.deg2rad(values)
                else:  # No units or unknown units
                    logging.info(
                        f"{coord_da.name} units are not specified. Inferring from value range."
                    )
                    # Check if any values fall outside the typical radian range
                    if np.any(np.abs(values) > 2 * np.pi):
                        logging.info(
                            "Values found outside [-2*pi, 2*pi]. Assuming degrees and converting."
                        )
                        return np.deg2rad(values)
                    else:
                        logging.info(
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
        logging.info("Querying tree to find nearest neighbors...")
        distance, indices = kdtree.query(target_xyz)

        logging.info(f"Saving new regridding indices to {cached_file}")
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
            self.data_dir
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
        logging.info(f"Requesting source variables: {source_variables}")

        logging.info(f"Opening {len(existing_paths)} member files.")

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

            # Now, filter again to the final list of source variables (including derived ones)
            final_vars_to_keep = [
                v for v in source_variables if v in ds_derived.data_vars
            ]
            return ds_derived[final_vars_to_keep]

        # Open the dataset lazily; the preprocess function will efficiently
        # derive and filter each file before they are combined.
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
        logging.info("Loading data into memory and regridding...")
        ds_regridded = self._regrid_dataset(ds_mpas.load())
        logging.info("Regridding complete.")
        return ds_regridded

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """Remaps from the unstructured grid to a regular lat-lon grid."""
        # Select data at the nearest neighbor indices and reshape
        regridded_data = ds_mpas.isel(nCells=self.indices)

        # Determine the new shape for the lat/lon grid
        new_shape = [
            regridded_data.sizes[dim] for dim in regridded_data.dims if dim != "nCells"
        ] + [len(self.target_lat), len(self.target_lon)]

        # Create new coordinates, preserving existing ones (time, member, etc.)
        new_coords = {
            dim: coord
            for dim, coord in regridded_data.coords.items()
            if dim != "nCells"
        }
        new_coords["lat"] = self.target_lat
        new_coords["lon"] = self.target_lon

        # Reshape each variable in the dataset
        regridded_vars = {}
        for var_name, da in regridded_data.data_vars.items():
            reshaped_values = da.values.reshape(new_shape)
            regridded_vars[var_name] = xr.DataArray(
                reshaped_values, coords=new_coords, dims=da.dims[:-1] + ("lat", "lon")
            )

        return xr.Dataset(regridded_vars, attrs=ds_mpas.attrs)

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
            members = list(range(1, 11))

        # Load and regrid the data for the given time/fhr/members
        # The result is cached by the helper method.
        # Pass variables and members as tuples so they can be hashed for the cache.
        ds_regridded = self._load_and_regrid(
            time, fhr, tuple(sorted(variables)), tuple(sorted(members))
        )

        # Map e2s variables to the source variable names from the lexicon
        rename_dict = {self.lexicon[var]: var for var in variables}

        # Select only the needed variables and rename them to e2s standard names
        ds_final = ds_regridded[list(rename_dict.keys())].rename(rename_dict)

        # Add metadata for context
        ds_final.attrs["initialization_time"] = str(time)
        ds_final.attrs["forecast_hour"] = fhr

        # Convert to a single DataArray with a 'variable' dimension, as expected by e2s
        return ds_final.to_dataarray(dim="variable")
