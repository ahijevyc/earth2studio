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
                # Geopotential might be derived from height
                return f"geopotential_{target_level}hPa"
            elif var_type == "t":
                return f"temperature_{target_level}hPa"
            elif var_type == "u":
                return f"uzonal_{target_level}hPa"
            elif var_type == "v":
                return f"umeridional_{target_level}hPa"
            elif var_type in ["q", "r"]: # Support for humidity types
                # Mixing ratio might be derived from relative humidity
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
        elif pl == 300:
            return 250
        elif pl == 400:
            return 500
        elif pl == 600:
            return 700
        elif pl == 1000:
            return 925
        else:
            return pl

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
        height_vars = [v for v in ds.data_vars if v.startswith("height_")]
        for h_var in height_vars:
            g_var = h_var.replace("height_", "geopotential_")
            if g_var not in ds:
                logging.info(f"Deriving '{g_var}' from '{h_var}'.")
                ds[g_var] = ds[h_var] * g.m

        # --- Derive Mixing Ratio from Relative Humidity ---
        rh_vars = [v for v in ds.data_vars if v.startswith("relhum_")]
        for rh_var in rh_vars:
            q_var = rh_var.replace("relhum_", "mixing_ratio_")
            t_var = rh_var.replace("relhum_", "temperature_")
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
        "msl": "surface_pressure",
        "t2m": "t2m",
        "u10m": "u10",
        "v10m": "v10",
    }

    # A set of base variables that are defined on 3D vertical levels.
    _3d_vars = {"t", "q", "u", "v", "w", "z"}

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
        requested e2s variables, including dependencies for derivations.
        """
        required = set()
        for var in variables:
            base_var = re.sub(r"\d+$", "", var)

            # Add dependencies based on the base variable type
            if base_var == "t":
                required.update(["theta", "pressure_p", "pressure_base"])
            elif base_var == "z":
                required.update(["zgrid", "pressure_p", "pressure_base"])
            elif base_var == "w":
                required.update(["w", "zgrid", "pressure_p", "pressure_base"])
            elif cls.is_3d_variable(var):
                required.add(cls[base_var])
                # Pressure is always needed for interpolation
                required.update(["pressure_p", "pressure_base"])
            else:
                # For surface variables or others without derivation needs.
                try:
                    required.add(cls[var])
                except KeyError:
                    logging.warning(f"No mapping for '{var}'. Adding directly.")
                    required.add(var)

        return list(required)

    @classmethod
    def is_3d_variable(cls, variable_name: str) -> bool:
        """Checks if a variable is a 3D field based on its base name."""
        base_var = re.sub(r"\d+$", "", variable_name)
        return base_var in cls._3d_vars

    @classmethod
    def get_derived_name(cls, variable_name: str) -> str:
        """
        Gets the name of a variable after lexicon derivations have been applied.
        e.g., 't' or 't500' becomes 'temperature'.
        e.g., 'z' or 'z500' becomes 'geopotential'.
        """
        base_var = re.sub(r"\d+$", "", variable_name)
        if base_var == "t":
            return "temperature"
        elif base_var == "z":
            return "geopotential"
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

        return ds


