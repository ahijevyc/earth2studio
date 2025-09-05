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
from metpy.constants import g
from metpy.units import units

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


