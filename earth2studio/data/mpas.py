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
from metpy.constants import dry_air_gas_constant, g
from scipy.spatial import KDTree

from earth2studio.data import DataSource
from earth2studio.data.utils import datasource_cache_root
from earth2studio.lexicon.mpas import MPASHybridLexicon, MPASLexicon
from earth2studio.utils.time import xtime

# =============================================================================
# Constants
# =============================================================================
# Standard lapse rate in K/m for temperature extrapolation below ground.
STANDARD_LAPSE_RATE = 0.0065

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# =============================================================================
# Base Class for MPAS Data Sources
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class _MPASBase(DataSource):
    """
    A base class for MPAS data sources providing shared functionality for
    handling unstructured grids, caching, and file I/O.

    Attributes
    ----------
    data_path : str
        A string representing the path to your MPAS data files. This path
        should be a template that can be formatted using Python's `strftime`
        convention. For example: "/path/to/data/%Y%m%d/history_%H.nc"
    grid_path : Path
        The path to the static MPAS grid definition file.
    d_lon : float, optional
        Target longitude spacing for regridding. Defaults to 0.25.
    d_lat : float, optional
        Target latitude spacing for regridding. Defaults to 0.25.
    cache_path : Path, optional
        The directory to store cached regridding indices.
    """

    data_path: str
    grid_path: Path
    d_lon: float = 0.25
    d_lat: float = 0.25
    cache_path: Path = Path(datasource_cache_root()) / "mpas_base"

    def __post_init__(self) -> None:
        """
        Post-initialization to prepare the target grid and compute or load the
        regridding indices from the cache.
        """
        self.cache_path.mkdir(parents=True, exist_ok=True)

        if isinstance(self.data_path, list):
            self.data_path = tuple(self.data_path)

        # Use np.linspace for robust grid generation that avoids floating point
        # precision issues and guarantees endpoint inclusion.
        n_lon = int(360 / self.d_lon)
        n_lat = int(180 / self.d_lat) + 1
        self.target_lon = np.linspace(0, 360, n_lon, endpoint=False)
        self.target_lat = np.linspace(90, -90, n_lat)

        self.distance, self.indices = self._prepare_regridding_indices()

        with xr.open_dataset(self.grid_path) as grid_ds:
            self.grid_ncells = grid_ds.sizes["nCells"]

    def _prepare_regridding_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates or loads cached nearest neighbor indices for regridding."""
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

            def process_coords(coord_da: xr.DataArray) -> np.ndarray:
                units = coord_da.attrs.get("units", "unknown").lower()
                values = coord_da.values
                if units in ["rad", "radians"]:
                    return values
                elif units in ["deg", "degrees"]:
                    return np.deg2rad(values)
                else:
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

    @staticmethod
    def _lon_lat_to_cartesian(lon_rad: np.ndarray, lat_rad: np.ndarray) -> np.ndarray:
        """Converts lon/lat (radians) to 3D Cartesian coords for KDTree."""
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z]).T


# =============================================================================
# Pressure-Level Data Source
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class MPAS(_MPASBase):
    """
    Custom data source for MPAS model output on pressure levels.
    """

    cache_path: Path = Path(datasource_cache_root()) / "mpas_plev"

    def __post_init__(self) -> None:
        self.lexicon = MPASLexicon
        super().__post_init__()

    @lru_cache(maxsize=16)
    def _load_and_process(
        self,
        time: datetime.datetime | np.datetime64,
        variables: tuple[str],
    ) -> xr.Dataset:
        """
        Loads, derives variables, and regrids a single time slice of
        pressure-level data by dynamically finding the correct file.
        """
        source_variables = self.lexicon.required_variables(list(variables))
        logging.info(f"Requesting source variables for time {time}: {source_variables}")

        # Convert numpy.datetime64 to pandas Timestamp, which has strftime
        time_pd = pd.to_datetime(time)
        path_str = time_pd.strftime(self.data_path)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"MPAS file not found for time {time} at: {path}")

        with xtime(xr.open_dataset(path)) as ds_mpas:
            # Select exact time, will raise KeyError if not found.
            ds_slice = ds_mpas.sel(time=time)
            if "time" in ds_slice.coords:
                ds_slice = ds_slice.drop_vars("time")
            ds_slice = ds_slice.squeeze()

            if ds_slice.sizes.get("nCells") != self.grid_ncells:
                raise ValueError(
                    f"Grid mismatch: Grid file has {self.grid_ncells} cells, "
                    f"data file has {ds_slice.sizes['nCells']} cells."
                )

            raw_vars_to_load = [v for v in source_variables if v in ds_slice.data_vars]
            ds_filtered = ds_slice[raw_vars_to_load]
            ds_derived = self.lexicon.derive_variables(ds_filtered)
            final_vars_to_keep = [
                v for v in source_variables if v in ds_derived.data_vars
            ]
            ds_processed = ds_derived[final_vars_to_keep].load()

        logging.info("Regridding data...")
        ds_regridded = self._regrid_dataset(ds_processed)
        logging.info("Regridding complete.")
        return ds_regridded

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """Remaps from the unstructured grid to a regular lat-lon grid."""
        regridded_data = ds_mpas.isel(nCells=self.indices)

        new_shape = [
            size for dim, size in regridded_data.sizes.items() if dim != "nCells"
        ] + [len(self.target_lat), len(self.target_lon)]

        new_coords = {
            dim: coord
            for dim, coord in regridded_data.coords.items()
            if dim != "nCells"
        }
        new_coords["lat"] = self.target_lat
        new_coords["lon"] = self.target_lon

        new_dims = [dim for dim in regridded_data.dims if dim != "nCells"] + [
            "lat",
            "lon",
        ]

        regridded_vars = {}
        for var_name, da in regridded_data.data_vars.items():
            reshaped_values = da.values.reshape(new_shape)
            regridded_vars[var_name] = xr.DataArray(
                reshaped_values, coords=new_coords, dims=new_dims
            )
        return xr.Dataset(regridded_vars, attrs=ds_mpas.attrs)

    def _finalize_dataset(
        self, ds_regridded: xr.Dataset, variables: list[str]
    ) -> xr.DataArray:
        """Builds the final DataArray from a processed, regridded Dataset."""
        rename_dict = {self.lexicon.get_item(var): var for var in variables}
        ds_final = ds_regridded[list(rename_dict.keys())].rename(rename_dict)
        return ds_final.to_dataarray(dim="variable")

    def __call__(
        self,
        time: datetime.datetime | list[datetime.datetime] | np.ndarray,
        variables: list[str],
    ) -> xr.DataArray:
        """
        Main entry point for fetching data. Handles both single datetime requests
        (for framework runners) and lists of datetimes (for direct use).
        """
        sorted_variables = tuple(sorted(variables))

        if isinstance(time, (datetime.datetime, np.datetime64)):
            # Runner-compatible path: process a single time, return a time-unaware slice.
            ds_regridded = self._load_and_process(time, sorted_variables)
            return self._finalize_dataset(ds_regridded, variables)
        else:
            # Direct-use path: process a list of times, return a time-aware DataArray.
            results = []
            for t in time:
                ds_regridded = self._load_and_process(t, sorted_variables)
                da_slice = self._finalize_dataset(ds_regridded, variables)
                # Add time coordinate back for this slice
                da_slice = da_slice.assign_coords(time=t).expand_dims("time")
                results.append(da_slice)

            if not results:
                return xr.DataArray()
            return xr.concat(results, dim="time")


# =============================================================================
# Hybrid-Level Data Source
# =============================================================================
@dataclasses.dataclass(unsafe_hash=True)
class MPASHybrid(_MPASBase):
    """
    Custom data source for MPAS model output on native hybrid levels. Can also
    interpolate to pressure levels.
    """

    pressure_levels: list[int] | tuple[int, ...] | None = None
    cache_path: Path = Path(datasource_cache_root()) / "mpas_hybrid"

    def __post_init__(self) -> None:
        self.lexicon = MPASHybridLexicon
        if isinstance(self.pressure_levels, list):
            self.pressure_levels = tuple(sorted(self.pressure_levels))
        super().__post_init__()

    @lru_cache(maxsize=16)
    def _load_and_process(
        self,
        time: datetime.datetime | np.datetime64,
        variables: tuple[str],
    ) -> xr.Dataset:
        """
        Loads, processes (including vertical interpolation), and regrids data
        by dynamically finding the correct file for the requested time.
        """
        source_variables = self.lexicon.required_variables(list(variables))
        logging.info(f"Requesting source variables for time {time}: {source_variables}")

        # Convert numpy.datetime64 to pandas Timestamp, which has strftime
        time_pd = pd.to_datetime(time)
        path_str = time_pd.strftime(self.data_path)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"MPAS file not found for time {time} at: {path}")

        with xtime(xr.open_dataset(path)) as ds_mpas:
            # Select exact time, will raise KeyError if not found.
            ds_slice = ds_mpas.sel(time=time)

            if "time" in ds_slice.coords:
                ds_slice = ds_slice.drop_vars("time")
            ds_slice = ds_slice.squeeze()

            if ds_slice.sizes.get("nCells") != self.grid_ncells:
                raise ValueError(
                    f"Grid mismatch: Grid file has {self.grid_ncells} cells, "
                    f"data file has {ds_slice.sizes['nCells']} cells."
                )
            time_vars_to_load = [v for v in source_variables if v in ds_slice.data_vars]
            ds_loaded = ds_slice[time_vars_to_load].load()

        with xr.open_dataset(self.grid_path) as grid_ds:
            grid_vars_to_load = [v for v in source_variables if v in grid_ds.data_vars]
            for var_name in grid_vars_to_load:
                ds_loaded[var_name] = grid_ds[var_name].load()

        ds_derived = self.lexicon.derive_variables(ds_loaded)

        ds_processed = ds_derived
        if self.pressure_levels is not None:
            is_3d_request = any(self.lexicon.is_3d_variable(v) for v in variables)
            if is_3d_request:
                logging.info(
                    f"3D variable requested. Performing vertical interpolation to {list(self.pressure_levels)} hPa."
                )
                pressure_levels_pa = [p * 100 for p in self.pressure_levels]
                ds_processed = self._interpolate_to_pressure_levels(
                    ds_derived, pressure_levels_pa, variables
                )

        logging.info("Regridding data...")
        ds_regridded = self._regrid_dataset(ds_processed)
        logging.info("Regridding complete.")
        return ds_regridded

    def _interpolate_to_pressure_levels(
        self,
        ds: xr.Dataset,
        target_levels_pa: list[int],
        requested_variables: tuple[str],
    ) -> xr.Dataset:
        """
        Interpolates data from native hybrid levels to pressure levels, handling
        below-terrain points by either persisting surface values or extrapolating.
        """
        vars_to_interp = {
            self.lexicon.get_derived_name(v)
            for v in requested_variables
            if self.lexicon.is_3d_variable(v)
        }

        def interp_and_fill_field(
            data_field: np.ndarray,
            pres_field: np.ndarray,
            targets: np.ndarray,
            var_name: str,
            surface_temperature: np.ndarray,
            surface_pressure: np.ndarray,
            surface_geopotential: np.ndarray,
        ) -> np.ndarray:
            """

            Performs interpolation and fills/extrapolates below-ground NaNs
            for a single variable field across all horizontal cells.
            """
            output = np.full(
                (data_field.shape[0], len(targets)), np.nan, dtype=data_field.dtype
            )
            # Loop over each horizontal cell (atmospheric column)
            for i in range(data_field.shape[0]):
                source_pres = pres_field[i, :]
                source_data = data_field[i, :]

                # Ensure pressure is monotonically increasing for numpy.interp
                if source_pres[0] > source_pres[-1]:
                    source_pres, source_data = source_pres[::-1], source_data[::-1]

                # Perform standard interpolation, leaving NaNs for out-of-bounds points
                interp_results = np.interp(
                    targets, source_pres, source_data, left=np.nan, right=np.nan
                )

                # --- Fill/Extrapolate below-ground points ---
                p_sfc = surface_pressure[i]
                surface_value = source_data[-1]
                # Mask for target levels that are below last pressure level (not necessary the surface but half-level higher?)
                # Because these were the source_pres for np.interp vertical.
                # if you use targets > p_sfc, you still might have nans for targets between last pressure level and p_sfc.
                below_ground_mask = targets > source_pres[-1]

                # Extrapolate downwards using a standard lapse rate L.
                # T(p) = t_sfc * (p / p_sfc) ^ (R_d * L / g)
                # z(p) = z_sfc + t_sfc / L * (1 - p / p_sfc) ^ (R_d * L / g)
                exponent = (
                    dry_air_gas_constant.magnitude * STANDARD_LAPSE_RATE
                ) / g.magnitude
                t_sfc = surface_temperature[i]

                p_target = targets[below_ground_mask]
                # Apply special extrapolation for temperature and geopotential
                if var_name == "temperature":
                    extrap_values = t_sfc * (p_target / p_sfc) ** exponent
                    interp_results[below_ground_mask] = extrap_values

                elif var_name == "geopotential":
                    z_sfc = surface_geopotential[i]
                    extrap_values = (
                        z_sfc
                        + t_sfc
                        * g.magnitude
                        / STANDARD_LAPSE_RATE
                        * (1 - (p_target / p_sfc) ** exponent)
                    )
                    interp_results[below_ground_mask] = extrap_values

                else:
                    # For all other variables, and as a fallback, persist surface value
                    interp_results[below_ground_mask] = surface_value

                output[i, :] = interp_results
            return output

        interpolated_vars = {}
        for name, da in ds.data_vars.items():
            is_main = "nVertLevels" in da.dims
            is_staggered = "nVertLevelsP1" in da.dims
            if (is_main or is_staggered) and name in vars_to_interp:
                logging.info(f"Interpolating variable: {name}")
                pressure_field = ds["pressure"] if is_main else ds["pressure_on_w"]
                surface_temperature = ds["t2m"].values
                surface_pressure = ds["surface_pressure"].values
                surface_geopotential = ds["geopotential_at_surface"].values

                data_np = da.values
                pressure_np = pressure_field.values

                interp_data = interp_and_fill_field(
                    data_np,
                    pressure_np,
                    np.array(target_levels_pa),
                    name,
                    surface_temperature,
                    surface_pressure,
                    surface_geopotential,
                )

                interpolated_vars[name] = xr.DataArray(
                    interp_data,
                    dims=["nCells", "level"],
                    coords={
                        "nCells": da.coords["nCells"],
                        "level": [p / 100 for p in target_levels_pa],
                    },
                )
            elif not is_main and not is_staggered:
                interpolated_vars[name] = da

        interp_ds = xr.Dataset(interpolated_vars, attrs=ds.attrs)
        interp_ds.level.attrs["units"] = "hPa"
        return interp_ds

    def _regrid_dataset(self, ds_mpas: xr.Dataset) -> xr.Dataset:
        """Remaps from the unstructured grid to a regular lat-lon grid."""
        regridded_data = ds_mpas.isel(nCells=self.indices)

        base_coords = {
            dim: coord
            for dim, coord in regridded_data.coords.items()
            if dim != "nCells"
        }
        base_coords["lat"] = xr.DataArray(self.target_lat, dims=["lat"])
        base_coords["lon"] = xr.DataArray(self.target_lon, dims=["lon"])

        regridded_vars = {}
        for var_name, da in regridded_data.data_vars.items():
            other_dims = [dim for dim in da.dims if dim != "nCells"]
            new_dims = other_dims + ["lat", "lon"]

            variable_coords = {
                k: v
                for k, v in base_coords.items()
                if all(dim in new_dims for dim in v.dims)
            }

            if "level" in new_dims:
                temp_shape = (
                    len(self.target_lat),
                    len(self.target_lon),
                    da.sizes["level"],
                )
                reshaped_values = da.values.reshape(temp_shape).transpose((2, 0, 1))
                new_dims = ["level", "lat", "lon"]
            else:
                new_shape = (len(self.target_lat), len(self.target_lon))
                reshaped_values = da.values.reshape(new_shape)

            regridded_vars[var_name] = xr.DataArray(
                reshaped_values, coords=variable_coords, dims=new_dims
            )
        return xr.Dataset(regridded_vars, attrs=ds_mpas.attrs)

    def _finalize_dataset(
        self, ds_regridded: xr.Dataset, variables: list[str]
    ) -> xr.DataArray:
        """Builds the final DataArray from a processed, regridded Dataset."""
        vars_to_build = {}
        for var in variables:
            source_name = self.lexicon.get_derived_name(var)
            if source_name not in ds_regridded:
                raise KeyError(
                    f"Requested variable '{var}' (mapped to '{source_name}') could not be found or derived."
                )

            da = ds_regridded[source_name]
            match = re.fullmatch(r"([a-zA-Z]+)([0-9]+)", var)
            if match and "level" in da.coords:
                level = int(match.group(2))
                vars_to_build[var] = da.sel(level=level, method="nearest").drop_vars(
                    "level"
                )
            else:
                vars_to_build[var] = da

        ds_final = xr.Dataset(vars_to_build)
        return ds_final.to_dataarray(dim="variable")

    def __call__(
        self,
        time: datetime.datetime | list[datetime.datetime] | np.ndarray,
        variables: list[str],
    ) -> xr.DataArray:
        """
        Main entry point for fetching data. Handles both single datetime requests
        (for framework runners) and lists of datetimes (for direct use).
        """
        sorted_variables = tuple(sorted(variables))

        if isinstance(time, (datetime.datetime, np.datetime64)):
            # Runner-compatible path: process a single time, return a time-unaware slice.
            ds_regridded = self._load_and_process(time, sorted_variables)
            return self._finalize_dataset(ds_regridded, variables)
        else:
            # Direct-use path: process a list of times, return a time-aware DataArray.
            results = []
            for t in time:
                ds_regridded = self._load_and_process(t, sorted_variables)
                da_slice = self._finalize_dataset(ds_regridded, variables)
                # Add time coordinate back for this slice
                da_slice = da_slice.assign_coords(time=t).expand_dims("time")
                results.append(da_slice)

            if not results:
                return xr.DataArray()
            return xr.concat(results, dim="time")
