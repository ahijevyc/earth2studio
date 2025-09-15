# =============================================================================
# Imports
# =============================================================================
import dataclasses
import datetime
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import xarray as xr
from earth2studio.data import DataSource
from earth2studio.data.utils import datasource_cache_root, xtime
from earth2studio.lexicon.mpas import MPASHybridLexicon, MPASLexicon
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



@dataclasses.dataclass(unsafe_hash=True)
class MPASHybrid(DataSource):
    """
    A custom earth2studio data source for loading, processing, and regridding
    MPAS model output. This class can provide data on either the model's
    native hybrid vertical levels or interpolated to specified isobaric
    (pressure) levels.

    It handles the unstructured MPAS grid by regridding it to a regular lat-lon
    grid using a nearest-neighbor approach with a KDTree.

    Note: This implementation is designed for cell-centered MPAS variables (using
    the 'nCells' dimension).

    Attributes:
        data_path (Union[Path, List[Path]]): Path to the MPAS output file(s).
        grid_path (Path): Path to the MPAS grid definition file (e.g., mesh.nc).
        pressure_levels (Union[List[int], None]): Optional. A list of pressure
            levels (in hPa) to interpolate to. If None, data is returned on
            native hybrid levels. Defaults to None.
        d_lon (float): Longitude spacing for the target regular grid. Defaults to 0.25.
        d_lat (float): Latitude spacing for the target regular grid. Defaults to 0.25.
        cache_path (Path): Directory to store cached regridding indices.
    """

    data_path: Union[Path, List[Path]]
    grid_path: Path

    # Vertical interpolation parameters
    pressure_levels: Union[List[int], Tuple[int], None] = None

    # Regridding resolution is now configurable upon initialization
    d_lon: float = 0.25
    d_lat: float = 0.25
    cache_path: Path = Path(datasource_cache_root()) / "mpas_hybrid"

    def __post_init__(self):
        """
        Post-initialization to prepare the target grid and compute or load the
        regridding indices from the cache.
        """
        self.lexicon = MPASHybridLexicon
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Convert data_path to tuple if it's a list to make it hashable for lru_cache
        if isinstance(self.data_path, list):
            self.data_path = tuple(self.data_path)
        if isinstance(self.pressure_levels, list):
            self.pressure_levels = tuple(sorted(self.pressure_levels))

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
        # Create a unique filename based on grid file and target resolution
        cache_file_name = f"{self.grid_path.stem}_{self.d_lon}x{self.d_lat}.npz"
        cached_file = self.cache_path / cache_file_name

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
                    return values
                elif units in ["deg", "degrees"]:
                    return np.deg2rad(values)
                else:  # No units or unknown units
                    if np.any(np.abs(values) > 2 * np.pi):
                        return np.deg2rad(values)
                    else:
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
        paths = self.data_path if isinstance(self.data_path, tuple) else (self.data_path,)
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"MPAS file not found at: {p}")
        
        if len(paths) > 1:
            logging.info(f"Opening and combining {len(paths)} data files.")
            datasets = [xtime(xr.open_dataset(p)) for p in paths]
            return xr.merge(datasets, combine_attrs="drop_conflicts")
        else:
            return xtime(xr.open_dataset(paths[0]))


    @staticmethod
    def _lon_lat_to_cartesian(lon_rad: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
        """Converts lon/lat (radians) to 3D Cartesian coords for KDTree."""
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z]).T

    def _interpolate_to_pressure_levels(
        self,
        ds: xr.Dataset,
        target_levels_pa: List[float],
        requested_variables: Tuple[str],
    ) -> xr.Dataset:
        """
        Interpolates final derived data variables from native hybrid levels to
        specified isobaric pressure levels using a direct numpy approach.
        """
        
        vars_to_interp = {
            self.lexicon.get_derived_name(v)
            for v in requested_variables
            if self.lexicon.is_3d_variable(v)
        }

        # Vectorized interpolation function for a full 2D field
        def interp_field(data_field: np.ndarray, pres_field: np.ndarray, targets: np.ndarray) -> np.ndarray:
            # Setup output array
            output = np.full((data_field.shape[0], len(targets)), np.nan, dtype=data_field.dtype)
            # Loop over each cell (column) and interpolate
            for i in range(data_field.shape[0]):
                source_pres = pres_field[i, :]
                source_data = data_field[i, :]

                # Check if the pressure is decreasing and flip if necessary to ensure
                # it is monotonically increasing for numpy.interp.
                if source_pres[0] > source_pres[-1]:
                    source_pres = source_pres[::-1]
                    source_data = source_data[::-1]

                result = np.interp(targets, source_pres, source_data, left=np.nan, right=np.nan)
                output[i, :] = result
            return output

        interpolated_vars = {}

        for name, da in ds.data_vars.items():
            is_main_grid = "nVertLevels" in da.dims
            is_staggered_grid = "nVertLevelsP1" in da.dims
            
            if (is_main_grid or is_staggered_grid) and name in vars_to_interp:
                logging.info(f"Interpolating variable: {name}")

                if is_main_grid:
                    vert_dim = "nVertLevels"
                    pressure_field = ds["pressure"]
                else: # is_staggered_grid
                    vert_dim = "nVertLevelsP1"
                    pressure_field = ds["pressure_on_w"]

                data_np = da.transpose("nCells", vert_dim).values
                pressure_np = pressure_field.transpose("nCells", vert_dim).values
                
                interpolated_data = interp_field(data_np, pressure_np, np.array(target_levels_pa))
                
                interpolated_da = xr.DataArray(
                    interpolated_data,
                    dims=["nCells", "level"],
                    coords={"nCells": da.coords["nCells"], "level": [p / 100 for p in target_levels_pa]},
                )
                interpolated_vars[name] = interpolated_da
            
            elif not is_main_grid and not is_staggered_grid:
                 # This is a 2D (or other non-vertical) variable, so carry it over.
                interpolated_vars[name] = da

        interp_ds = xr.Dataset(interpolated_vars, attrs=ds.attrs)
        interp_ds.level.attrs["units"] = "hPa"
        interp_ds.level.attrs["long_name"] = "Isobaric pressure level"

        return interp_ds

    @lru_cache(maxsize=16)
    def _load_and_regrid(
        self,
        time: datetime.datetime,
        variables: Tuple[str],
    ) -> xr.Dataset:
        """
        Loads, processes (including vertical interpolation), and regrids data.
        Caches the result to avoid redundant I/O and computation.
        """
        source_variables = self.lexicon.required_variables(list(variables))
        logging.info(f"Requesting source variables: {source_variables}")

        # 1. Load time-varying data into a fully in-memory dataset
        with self._open_data_dataset() as ds_mpas:
            ds_mpas = ds_mpas.sel(time=time)
            ds_mpas = ds_mpas.squeeze()
            
            if ds_mpas.sizes.get("nCells") != self.grid_ncells:
                raise ValueError(
                    f"Grid mismatch: Grid file has {self.grid_ncells} cells, "
                    f"data file has {ds_mpas.sizes['nCells']} cells."
                )

            time_vars_to_load = [v for v in source_variables if v in ds_mpas.data_vars]
            ds_loaded = ds_mpas[time_vars_to_load].load()

        # 2. Load static grid data and add it to the main dataset
        with xr.open_dataset(self.grid_path) as grid_ds:
            grid_vars_to_load = [v for v in source_variables if v in grid_ds.data_vars]
            if grid_vars_to_load:
                for var_name in grid_vars_to_load:
                    # Assign loaded grid variables to the main dataset
                    ds_loaded[var_name] = grid_ds[var_name].load()
            
            # 3. Derive essential variables (e.g., pressure, temperature)
            ds_derived = self.lexicon.derive_variables(ds_loaded)

            # 4. Perform vertical interpolation only if necessary
            ds_processed = ds_derived
            if self.pressure_levels is not None:
                is_3d_request = any(
                    self.lexicon.is_3d_variable(v) for v in variables
                )

                if is_3d_request:
                    logging.info(f"3D variable requested. Performing vertical interpolation to {list(self.pressure_levels)} hPa.")
                    pressure_levels_pa = [p * 100 for p in self.pressure_levels]
                    ds_processed = self._interpolate_to_pressure_levels(
                        ds_derived, pressure_levels_pa, variables
                    )
            
            # 5. Perform horizontal regridding
            logging.info("Regridding data...")
            ds_regridded = self._regrid_dataset(ds_processed)
            logging.info("Regridding complete.")
            return ds_regridded

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """Remaps from the unstructured grid to a regular lat-lon grid."""
        regridded_data = ds_mpas.isel(nCells=self.indices)

        # Build a set of all possible coordinates from the input dataset
        base_coords = {
            dim: coord for dim, coord in regridded_data.coords.items() if dim != "nCells"
        }
        base_coords["lat"] = xr.DataArray(self.target_lat, dims=["lat"])
        base_coords["lon"] = xr.DataArray(self.target_lon, dims=["lon"])

        regridded_vars = {}
        for var_name, da in regridded_data.data_vars.items():
            other_dims = [dim for dim in da.dims if dim != "nCells"]
            new_dims = other_dims + ["lat", "lon"]
            
            # Filter the base coordinates to only include those relevant to the current variable's dimensions
            variable_coords = {
                k: v for k, v in base_coords.items() if all(dim in new_dims for dim in v.dims)
            }
            
            # Reshape the values. The logic differs for 2D (surface) and 3D (multi-level) vars.
            if "level" in new_dims:
                # For 3D vars, da.values is (points, level). Reshape to (lat, lon, level) then transpose.
                temp_shape = (len(self.target_lat), len(self.target_lon), da.sizes["level"])
                reshaped_values = da.values.reshape(temp_shape).transpose((2, 0, 1))
            else:
                # For 2D vars, da.values is (points,). Reshape is straightforward.
                new_shape = (len(self.target_lat), len(self.target_lon))
                reshaped_values = da.values.reshape(new_shape)

            regridded_vars[var_name] = xr.DataArray(
                reshaped_values, coords=variable_coords, dims=new_dims
            )

        return xr.Dataset(regridded_vars, attrs=ds_mpas.attrs)

    async def __call__(
        self,
        time: datetime.datetime,
        variables: List[str],
    ) -> xr.DataArray:
        """
        Main entry point for fetching data. Loads, processes, regrids, and
        returns the requested variables as an xarray.DataArray.
        """
        ds_regridded = self._load_and_regrid(time, tuple(sorted(variables)))

        vars_to_build = {}
        for var in variables:
            source_name = self.lexicon.get_derived_name(var)

            if source_name not in ds_regridded:
                raise KeyError(f"Requested variable '{var}' (mapped to '{source_name}') could not be found or derived.")
            
            da = ds_regridded[source_name]

            match = re.fullmatch(r"([a-zA-Z]+)([0-9]+)", var)
            if match and 'level' in da.coords:
                level = int(match.group(2))
                # Select the level and drop the now-scalar 'level' coordinate
                # to prevent merge conflicts when creating the final dataset.
                vars_to_build[var] = da.sel(level=level, method="nearest").drop_vars("level")
            else:
                vars_to_build[var] = da
        
        ds_final = xr.Dataset(vars_to_build)
        
        ds_final.attrs["time"] = str(time)
        ds_final.attrs["source_file"] = str(self.data_path)

        return ds_final.to_dataarray(dim="variable")


