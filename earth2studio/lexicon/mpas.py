# =============================================================================
# Imports
# =============================================================================
import logging
import re
from typing import List

import numpy as np
import xarray as xr
from .base import LexiconType
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.constants import (
    dry_air_gas_constant,
    dry_air_spec_heat_press,
    g,
    pot_temp_ref_press,
)
from metpy.units import units
import metpy.xarray  # Activates .metpy accessor

# =============================================================================
# MPAS Lexicon Class
# =============================================================================
class MPASLexicon(metaclass=LexiconType):
    """
    Defines the lexicon for custom MPAS data. This class translates standard
    earth2studio variable names to their corresponding names in the MPAS NetCDF
    files. It also handles pressure level remapping and derived variable
    calculations as required by the source data.
    """

    # --- Class constants for prefixes and mappings ---
    _HEIGHT_PREFIX = "height_"
    _GPH_PREFIX = "geopotential_"
    _RELHUM_PREFIX = "relhum_"
    _MIX_RATIO_PREFIX = "mixing_ratio_"
    _TEMP_PREFIX = "temperature_"

    _PRESSURE_MAP = {
        300: 250,
        400: 500,
        600: 700,
        1000: 925,
    }

    # Define the native pressure levels for which source data exists.
    # Used for building variable names.
    PRESSURE_LEVELS: List[int] = [
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
    ]

    # --- Direct Mappings for Surface Variables ---
    # Maps e2s standard names to MPAS source variable names.
    surface_lexicon = {
        "t2m": "t2m",
        "u10m": "u10",
        "v10m": "v10",
        "msl": "surface_pressure",  # Assuming msl is surface pressure
    }

    @classmethod
    def get_item(cls, key: str) -> str:
        """Provides a direct lookup for variables."""
        if key in cls.surface_lexicon:
            return cls.surface_lexicon[key]

        # Handle pressure level variables dynamically using regex (e.g., 'z500')
        match = re.fullmatch(r"([a-zA-Z]+)([0-9]+)", key)
        if match:
            var_type = match.group(1)
            level = int(match.group(2))
            target_level = cls._remap_pressure(level)

            if var_type == "z":
                return f"geopotential_{target_level}hPa"
            elif var_type == "t":
                return f"temperature_{target_level}hPa"
            elif var_type == "u":
                return f"uzonal_{target_level}hPa"
            elif var_type == "v":
                return f"umeridional_{target_level}hPa"
            elif var_type in ["q", "r"]: # Support for humidity types
                return f"mixing_ratio_{target_level}hPa"

        raise KeyError(f"Lexicon key '{key}' not found in MPAS lexicon.")


    @staticmethod
    def _remap_pressure(pl: int) -> int:
        """
        Remaps input pressure levels to the target pressure levels used in the
        MPAS source file variable names, as required by the downstream model.
        """
        if pl <= 200:
            return 200
        # .get() provides a default value (pl) if the key is not in the map
        return MPASLexicon._PRESSURE_MAP.get(pl, pl)

    @classmethod
    def required_variables(cls, variables: List[str]) -> List[str]:
        """
        Determines the full set of source variables required to compute the
        requested e2s variables, including those needed for derivations.
        """
        required = set()
        for var in variables:
            # Attempt to parse pressure level variables, e.g. z500, t250
            match = re.fullmatch(r"([a-zA-Z]+)([0-9]+)", var)

            if match:
                var_type = match.group(1)
                level = int(match.group(2))
                target_level = cls._remap_pressure(level)

                if var_type == "z": # Geopotential
                    # If geopotential is not in the file, we need height to derive it.
                    required.add(f"geopotential_{target_level}hPa")
                    required.add(f"height_{target_level}hPa")
                elif var_type in ["q", "r"]: # Mixing Ratio
                    # We need RH and temperature to compute mixing ratio.
                    required.add(f"mixing_ratio_{target_level}hPa")
                    required.add(f"relhum_{target_level}hPa")
                    required.add(f"temperature_{target_level}hPa")
                else:
                    # For other pressure level variables (t, u, v), get the direct mapping
                    try:
                        required.add(cls[var])
                    except KeyError:
                        logging.warning(f"No mapping for '{var}' found. Adding directly.")
                        required.add(var)
            else:
                # Handle surface variables by looking them up in the lexicon
                try:
                    required.add(cls[var])
                except KeyError:
                    logging.warning(f"No mapping for '{var}' found. Adding directly.")
                    required.add(var)

        return list(required)

    @staticmethod
    def derive_variables(ds: xr.Dataset) -> xr.Dataset:
        """
        Derives new variables (e.g., geopotential, mixing ratio) if they are not
        present in the source dataset. This function is intended to be used as
        the `preprocess` function in `xr.open_mfdataset`.
        """
        # --- Derive Geopotential from Height ---
        height_vars = [v for v in ds.data_vars if v.startswith(MPASLexicon._HEIGHT_PREFIX)]
        for h_var in height_vars:
            g_var = h_var.replace(MPASLexicon._HEIGHT_PREFIX, MPASLexicon._GPH_PREFIX)
            if g_var not in ds:
                logging.info(f"Deriving '{g_var}' from '{h_var}'.")
                ds[g_var] = ds[h_var] * g.m

        # --- Derive Mixing Ratio from Relative Humidity ---
        rh_vars = [v for v in ds.data_vars if v.startswith(MPASLexicon._RELHUM_PREFIX)]
        for rh_var in rh_vars:
            q_var = rh_var.replace(MPASLexicon._RELHUM_PREFIX, MPASLexicon._MIX_RATIO_PREFIX)
            t_var = rh_var.replace(MPASLexicon._RELHUM_PREFIX, MPASLexicon._TEMP_PREFIX)
            if q_var not in ds and t_var in ds:
                logging.info(f"Deriving '{q_var}' from '{rh_var}' and '{t_var}'.")
                pressure_hpa = float(rh_var.split("_")[-1][:-3]) * units.hPa
                temperature_k = ds[t_var] * units.kelvin # Add units for metpy
                relative_humidity = ds[rh_var]

                mixing_ratio = mixing_ratio_from_relative_humidity(
                    pressure_hpa, temperature_k, relative_humidity
                )
                # Dequantify to get a plain numpy array for xarray
                ds[q_var] = mixing_ratio.metpy.dequantify()

        return ds


class MPASHybridLexicon(metaclass=LexiconType):
    """
    Defines the lexicon for native MPAS hybrid-level data. This class translates
    standard earth2studio variable names to their corresponding raw names in the
    MPAS NetCDF files (e.g., 't' -> 'theta'). It also provides the logic for
    deriving full pressure, temperature, and geopotential from the model's
    native variables.
    """

    # Maps general e2s variable identifiers to their native MPAS counterparts.
    # This is the single source of truth for variable names.
    _lexicon = {
        "t": "theta",  # Temperature is derived from Potential Temperature
        "z": "zgrid",  # Geopotential is derived from geometric height
        "q": "qv",
        "u": "uReconstructZonal",
        "v": "uReconstructMeridional",
        "w": "w",
        "msl": "surface_pressure",  # Mean sea level pressure (or surface pressure)
        "lsm": "landmask",  # Land-sea mask
        "t2m": "t2m",
        "u10m": "u10",
        "v10m": "v10",
        "tp": "tp",  # Base variable for total precipitation
    }

    @classmethod
    def get_item(cls, key: str) -> str:
        """Provides a direct lookup, stripping pressure levels for hybrid data."""
        # For a variable like 't850', we still need the base variable 'theta'
        # from the source file. The DataSource handles the interpolation.
        base_var = re.sub(r"\d+$", "", key)
        if base_var in cls._lexicon:
            return cls._lexicon[base_var]

        raise KeyError(f"Lexicon key '{key}' not found in MPASHybridLexicon.")

    @classmethod
    def required_variables(cls, variables: List[str]) -> List[str]:
        """
        Determines the full set of source variables required to compute the
        requested e2s variables, including dependencies for derivations. This logic
        is simplified based on the rule that all 3D variables must have a
        pressure level suffix (e.g., 't500').
        """
        required = set()
        for var in variables:
            # Handle 2D variables first
            if not cls.is_3d_variable(var):
                if var == "z":  # surface geopotential
                    required.add("ter")
                elif var.startswith("tp"):
                    required.update(["rainc", "rainnc"])
                else:  # Standard 2D vars: lsm, msl, t2m, u10m, v10m
                    try:
                        required.add(cls[var])
                    except KeyError:
                        logging.warning(f"No mapping for '{var}'. Adding directly.")
                        required.add(var)
                continue  # Done with this 2D variable

            # --- Everything below this point is a 3D variable (e.g., t500) ---
            base_var = re.sub(r"\d+$", "", var)

            # All 3D variables need pressure for vertical interpolation
            required.update(["pressure_p", "pressure_base"])

            # Add the source variable for the requested field (e.g., 'theta' for 't')
            required.add(cls[base_var])

            # Add any extra dependencies for specific 3D variable derivations
            if base_var == "w":
                # 'w' also needs 'zgrid' to derive pressure on the staggered grid
                required.add("zgrid")

        return list(required)

    @classmethod
    def is_3d_variable(cls, variable_name: str) -> bool:
        """
        Checks if a variable is a 3D field. A variable is considered 3D if it's
        not in a specific list of 2D variables and ends with a number (the
        pressure level).
        """
        # Explicitly define all known 2D variables.
        # This prevents 't2m' from being misinterpreted as 3D variable 't'.
        if variable_name in ["lsm", "msl", "t2m", "u10m", "v10m", "z"] or variable_name.startswith("tp"):
            return False

        # If it's not a known 2D variable and ends with digits, it's a 3D field.
        return bool(re.search(r"\d+$", variable_name))

    @classmethod
    def get_derived_name(cls, variable_name: str) -> str:
        """
        Gets the name of a variable after lexicon derivations have been applied.
        e.g., 't' or 't500' becomes 'temperature'.
        e.g., 'z' or 'z500' becomes 'geopotential'.
        """
        base_var = re.sub(r"\d+$", "", variable_name)
        if variable_name == "z":
            return "geopotential_at_surface"
        elif base_var == "t":
            return "temperature"
        elif base_var == "z":
            return "geopotential"
        elif base_var == "tp":
            return variable_name  # Return the full name, e.g., 'tp06'
        else:
            return cls[variable_name]

    @staticmethod
    def derive_variables(ds: xr.Dataset) -> xr.Dataset:
        """
        Derives standard meteorological variables from the raw MPAS output.
        - Calculates full pressure from base and perturbation pressure.
        - Calculates temperature from potential temperature (theta).
        - Calculates geopotential from geometric height (zgrid).
        - Calculates pressure on the staggered 'w' grid.
        - Calculates total precipitation from its components.
        """
        # --- 1. Derive Full Pressure (if needed) ---
        if "pressure_base" in ds and "pressure_p" in ds and "pressure" not in ds:
            logging.info("Deriving full pressure.")
            ds["pressure"] = ds["pressure_base"] + ds["pressure_p"]
            ds["pressure"].attrs = {
                "units": "Pa",
                "long_name": "Full atmospheric pressure",
            }

        # --- 2. Derive Temperature from Potential Temperature (if needed) ---
        if "theta" in ds and "pressure" in ds and "temperature" not in ds:
            logging.info(
                "Deriving temperature from potential temperature using unit-aware calculations."
            )
            # Quantify only the specific arrays needed for the calculation
            pressure_q = ds["pressure"].metpy.quantify()
            theta_q = ds["theta"].metpy.quantify()

            # Ensure reference pressure has the same units as the data
            ref_press_pa = pot_temp_ref_press.to(pressure_q.metpy.units)

            kappa = dry_air_gas_constant / dry_air_spec_heat_press
            exner = (pressure_q / ref_press_pa) ** kappa
            temperature_with_units = theta_q * exner
            ds["temperature"] = temperature_with_units.metpy.dequantify()
            ds["temperature"].attrs["long_name"] = "Temperature"

        # --- 3. Derive Geopotential from Geometric Height (zgrid) (if needed) ---
        if "zgrid" in ds and "pressure" in ds and "geopotential" not in ds:
            logging.info("Deriving geopotential from geometric height (zgrid).")

            # zgrid is on layer interfaces (nVertLevelsP1), but pressure/temp are on
            # layer centers (nVertLevels). We average zgrid to the centers.
            zgrid_vals = ds["zgrid"].values
            z_mid_level_vals = 0.5 * (zgrid_vals[..., :-1] + zgrid_vals[..., 1:])

            # Build coordinates explicitly to avoid carrying over unwanted ones
            # from a template array.
            height_coords = {
                "nCells": ds["pressure"].coords["nCells"],
                "nVertLevels": ds["pressure"].coords["nVertLevels"]
            }
            height_da = xr.DataArray(
                z_mid_level_vals,
                dims=("nCells", "nVertLevels"),
                coords=height_coords,
                attrs={"units": "m", "long_name": "Geometric height at layer center"}
            )
            ds["height"] = height_da

            # Quantify only the height array to calculate geopotential
            height_q = ds["height"].metpy.quantify()
            geopotential_with_units = height_q * g
            ds["geopotential"] = geopotential_with_units.metpy.dequantify()
            ds["geopotential"].attrs["long_name"] = "Geopotential"

        # --- 4. Derive Pressure on the staggered 'w' grid (if needed) ---
        # This routine uses log-linear interpolation to place pressure values onto the
        # vertical velocity levels (staggered grid). This is the standard method
        # used in MPAS post-processing to maintain consistency with the model's
        # vertical coordinate system.
        if "w" in ds and "zgrid" in ds and "pressure" in ds and "pressure_on_w" not in ds:
            logging.info("Deriving pressure on the staggered vertical grid for 'w'.")

            pressure = ds["pressure"]
            zgrid = ds["zgrid"]
            nVertLevels = ds.sizes["nVertLevels"]

            # Create an empty array for the staggered pressure
            pressure_on_w = xr.full_like(zgrid, np.nan)

            # Log-linear interpolation for interior levels
            # Corresponds to Fortran k = 2 to nVertLevels
            z_k = zgrid.isel(nVertLevelsP1=slice(1, nVertLevels))
            z_km1 = zgrid.isel(nVertLevelsP1=slice(0, nVertLevels - 1))
            z_kp1 = zgrid.isel(nVertLevelsP1=slice(2, nVertLevels + 1))

            p_k = pressure.isel(nVertLevels=slice(1, nVertLevels))
            p_km1 = pressure.isel(nVertLevels=slice(0, nVertLevels-1))

            w1 = (z_k.values - z_km1.values) / (z_kp1.values - z_km1.values)
            w2 = (z_kp1.values - z_k.values) / (z_kp1.values - z_km1.values)

            log_p_interp = w1 * np.log(p_k.values) + w2 * np.log(p_km1.values)
            pressure_on_w.values[:, 1:nVertLevels] = np.exp(log_p_interp)

            # Extrapolation for the bottom level (surface)
            # Corresponds to Fortran k = 1
            z0 = zgrid.isel(nVertLevelsP1=0)
            z1 = 0.5 * (zgrid.isel(nVertLevelsP1=0) + zgrid.isel(nVertLevelsP1=1))
            z2 = 0.5 * (zgrid.isel(nVertLevelsP1=1) + zgrid.isel(nVertLevelsP1=2))
            w1_bot = (z0 - z2) / (z1 - z2)
            w2_bot = 1.0 - w1_bot
            log_p_bot = w1_bot * np.log(pressure.isel(nVertLevels=0)) + w2_bot * np.log(pressure.isel(nVertLevels=1))
            pressure_on_w.values[:, 0] = np.exp(log_p_bot)

            # Extrapolation for the top level
            # Corresponds to Fortran k = nVertLevelsP1
            z0 = zgrid.isel(nVertLevelsP1=-1)
            z1 = 0.5 * (zgrid.isel(nVertLevelsP1=-1) + zgrid.isel(nVertLevelsP1=-2))
            z2 = 0.5 * (zgrid.isel(nVertLevelsP1=-2) + zgrid.isel(nVertLevelsP1=-3))
            w1_top = (z0 - z2) / (z1 - z2)
            w2_top = 1.0 - w1_top
            log_p_top = w1_top * np.log(pressure.isel(nVertLevels=-1)) + w2_top * np.log(pressure.isel(nVertLevels=-2))
            pressure_on_w.values[:, -1] = np.exp(log_p_top)

            ds["pressure_on_w"] = pressure_on_w
            ds["pressure_on_w"].attrs = {
                "units": "Pa",
                "long_name": "Pressure on vertical velocity grid",
            }

        # --- 5. Derive Total Precipitation (if needed) ---
        if "rainc" in ds and "rainnc" in ds and "tp06" not in ds:
            logging.info("Deriving 'tp06' from 'rainc' and 'rainnc'.")
            total_precip = ds["rainc"] + ds["rainnc"]
            # Preserve metadata and add a descriptive name
            total_precip.attrs = ds["rainc"].attrs.copy()
            total_precip.attrs["long_name"] = "Total precipitation"
            total_precip.attrs[
                "description"
            ] = "Sum of convective (rainc) and non-convective (rainnc) precipitation."
            ds["tp06"] = total_precip

        # --- 6. Derive Surface Geopotential from Terrain Height ---
        if "ter" in ds and "geopotential_at_surface" not in ds:
            logging.info("Deriving surface geopotential from terrain height ('ter').")
            ds["geopotential_at_surface"] = ds["ter"] * g
            ds["geopotential_at_surface"].attrs = {
                "units": "m^2 s^-2",
                "long_name": "Geopotential at Surface",
                "description": "Geopotential calculated from terrain height.",
            }

        return ds


