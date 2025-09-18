# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Imports
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
import logging
import re

import metpy.xarray  # noqa: F401 (activates .metpy accessor)
import numpy as np
import xarray as xr
from metpy.calc import mixing_ratio_from_relative_humidity
from metpy.constants import (
    dry_air_gas_constant,
    dry_air_spec_heat_press,
    g,
    pot_temp_ref_press,
)
from metpy.units import units

from earth2studio.lexicon.base import LexiconType

# =============================================================================
# Constants
# =============================================================================
# Standard lapse rate in K/m for temperature extrapolation below ground.
STANDARD_LAPSE_RATE = 0.0065


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# MPAS Lexicon Class
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
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
    PRESSURE_LEVELS: list[int] = [
        1000,
        925,
        850,
        700,
        600,
        500,
        400,
        300,
        250,
        200,
        150,
        100,
        50,
    ]

    # --- Direct Mappings for Surface Variables ---
    # Maps e2s standard names to MPAS source variable names.
    surface_lexicon = {
        "t2m": "t2m",
        "u10m": "u10",
        "v10m": "v10",
        "msl": "mslp",  # Mean sea level pressure
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
            elif var_type in ["q", "r"]:  # Support for humidity types
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
    def required_variables(cls, variables: list[str]) -> list[str]:
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

                if var_type == "z":  # Geopotential
                    # If geopotential is not in the file, we need height to derive it.
                    required.add(f"geopotential_{target_level}hPa")
                    required.add(f"height_{target_level}hPa")
                elif var_type in ["q", "r"]:  # Mixing Ratio
                    # We need RH and temperature to compute mixing ratio.
                    required.add(f"mixing_ratio_{target_level}hPa")
                    required.add(f"relhum_{target_level}hPa")
                    required.add(f"temperature_{target_level}hPa")
                else:
                    # For other pressure level variables (t, u, v), get the direct mapping
                    try:
                        required.add(cls.get_item(var))
                    except KeyError:
                        logging.warning(
                            f"No mapping for '{var}' found. Adding directly."
                        )
                        required.add(var)
            else:
                # Handle surface variables by looking them up in the lexicon
                try:
                    required.add(cls.get_item(var))
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
        height_vars = [
            v for v in ds.data_vars if v.startswith(MPASLexicon._HEIGHT_PREFIX)
        ]
        for h_var in height_vars:
            g_var = h_var.replace(MPASLexicon._HEIGHT_PREFIX, MPASLexicon._GPH_PREFIX)
            if g_var not in ds:
                logging.info(f"Deriving '{g_var}' from '{h_var}'.")
                ds[g_var] = ds[h_var] * g.m

        # --- Derive Mixing Ratio from Relative Humidity ---
        rh_vars = [v for v in ds.data_vars if v.startswith(MPASLexicon._RELHUM_PREFIX)]
        for rh_var in rh_vars:
            q_var = rh_var.replace(
                MPASLexicon._RELHUM_PREFIX, MPASLexicon._MIX_RATIO_PREFIX
            )
            t_var = rh_var.replace(MPASLexicon._RELHUM_PREFIX, MPASLexicon._TEMP_PREFIX)
            if q_var not in ds and t_var in ds:
                logging.info(f"Deriving '{q_var}' from '{rh_var}' and '{t_var}'.")
                pressure_hpa = float(rh_var.split("_")[-1][:-3]) * units.hPa
                temperature_k = ds[t_var] * units.kelvin
                relative_humidity = ds[rh_var]

                # Perform unit-aware calculation
                mixing_ratio = mixing_ratio_from_relative_humidity(
                    pressure_hpa, temperature_k, relative_humidity
                )
                # Keep the result as a unit-aware DataArray
                ds[q_var] = mixing_ratio

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
    _lexicon = {
        "t": "theta",  # Temperature is derived from Potential Temperature
        "z": "zgrid",  # Geopotential is derived from geometric height
        "q": "qv",
        "u": "uReconstructZonal",
        "v": "uReconstructMeridional",
        "w": "w",
        "lsm": "landmask",
        "t2m": "t2m",
        "u10m": "u10",
        "v10m": "v10",
        "tp": "tp",  # Base variable for total precipitation
    }

    @classmethod
    def get_item(cls, key: str) -> str:
        """Provides a direct lookup, stripping pressure levels for hybrid data."""
        base_var = re.sub(r"\d+$", "", key)
        if base_var in cls._lexicon:
            return cls._lexicon[base_var]

        raise KeyError(f"Lexicon key '{key}' not found in MPASHybridLexicon.")

    @classmethod
    def required_variables(cls, variables: list[str]) -> list[str]:
        """
        Determines the full set of source variables required to compute the
        requested e2s variables, including dependencies for derivations.
        """
        required = set()
        for var in variables:
            # Handle 2D variables first
            if not cls.is_3d_variable(var):
                if var == "z":  # surface geopotential
                    required.add("ter")
                elif var == "msl":  # mean sea level pressure
                    required.update(["surface_pressure", "ter", "t2m"])
                elif var.startswith("tp"):
                    required.update(["rainc", "rainnc"])
                else:  # Standard 2D vars: lsm, t2m, u10m, v10m
                    try:
                        required.add(cls.get_item(var))
                    except KeyError:
                        logging.warning(f"No mapping for '{var}'. Adding directly.")
                        required.add(var)
                continue

            # --- Everything below this point is a 3D variable ---
            base_var = re.sub(r"\d+$", "", var)
            required.update(["pressure_p", "pressure_base"])
            required.add(cls.get_item(base_var))

            if base_var == "w":
                required.add("zgrid")
                required.add("theta")
                required.add("qv")

        return list(required)

    @classmethod
    def is_3d_variable(cls, variable_name: str) -> bool:
        """
        Checks if a variable is a 3D field. A variable is considered 3D if it's
        not in a specific list of 2D variables and ends with a number.
        """
        if variable_name in [
            "lsm",
            "msl",
            "t2m",
            "u10m",
            "v10m",
            "z",
        ] or variable_name.startswith("tp"):
            return False
        return bool(re.search(r"\d+$", variable_name))

    @classmethod
    def get_derived_name(cls, variable_name: str) -> str:
        """
        Gets the name of a variable after lexicon derivations have been applied.
        """
        base_var = re.sub(r"\d+$", "", variable_name)
        if variable_name == "z":
            return "geopotential_at_surface"
        elif variable_name == "msl":
            return "mean_sea_level_pressure"
        elif base_var == "t":
            return "temperature"
        elif base_var == "z":
            return "geopotential"
        elif base_var == "w":
            return "pressure_vertical_velocity"
        elif base_var == "tp":
            return variable_name
        else:
            return cls.get_item(variable_name)

    @staticmethod
    def derive_variables(ds: xr.Dataset) -> xr.Dataset:
        """
        Derives standard meteorological variables from the raw MPAS output.
        """
        # --- 1. Derive Full Pressure ---
        if "pressure_base" in ds and "pressure_p" in ds and "pressure" not in ds:
            ds["pressure"] = ds["pressure_base"] + ds["pressure_p"]
            ds["pressure"].attrs["units"] = "Pa"

        # --- 2. Derive Temperature from Potential Temperature ---
        if "theta" in ds and "pressure" in ds and "temperature" not in ds:
            pressure_q = ds["pressure"].metpy.quantify()
            theta_q = ds["theta"].metpy.quantify()
            ref_press_pa = pot_temp_ref_press.to(pressure_q.metpy.units)
            kappa = dry_air_gas_constant / dry_air_spec_heat_press
            exner = (pressure_q / ref_press_pa) ** kappa
            ds["temperature"] = theta_q * exner

        # --- 3. Derive Geopotential from Geometric Height ---
        if "zgrid" in ds and "pressure" in ds and "geopotential" not in ds:
            zgrid_vals = ds["zgrid"].values
            z_mid_level_vals = 0.5 * (zgrid_vals[..., :-1] + zgrid_vals[..., 1:])
            height_coords = {
                "nCells": ds["pressure"].coords["nCells"],
                "nVertLevels": ds["pressure"].coords["nVertLevels"],
            }
            height_da = xr.DataArray(
                z_mid_level_vals,
                dims=("nCells", "nVertLevels"),
                coords=height_coords,
            )
            ds["height"] = height_da
            ds["geopotential"] = ds["height"] * g

        # --- 4. Derive Pressure on the staggered 'w' grid ---
        if (
            "w" in ds
            and "zgrid" in ds
            and "pressure" in ds
            and "pressure_on_w" not in ds
        ):
            pressure, zgrid = ds["pressure"], ds["zgrid"]
            nVertLevels = ds.sizes["nVertLevels"]
            pressure_on_w = xr.full_like(zgrid, np.nan)

            z_k = zgrid.isel(nVertLevelsP1=slice(1, nVertLevels))
            z_km1 = zgrid.isel(nVertLevelsP1=slice(0, nVertLevels - 1))
            z_kp1 = zgrid.isel(nVertLevelsP1=slice(2, nVertLevels + 1))
            p_k = pressure.isel(nVertLevels=slice(1, nVertLevels))
            p_km1 = pressure.isel(nVertLevels=slice(0, nVertLevels - 1))
            w1 = (z_k.values - z_km1.values) / (z_kp1.values - z_km1.values)
            log_p_interp = w1 * np.log(p_k.values) + (1 - w1) * np.log(p_km1.values)
            pressure_on_w.values[:, 1:nVertLevels] = np.exp(log_p_interp)

            for i, j, k, level_idx in [(0, 0, 1, 2), (-1, -1, -2, -3)]:
                z0 = zgrid.isel(nVertLevelsP1=i)
                z1 = 0.5 * (zgrid.isel(nVertLevelsP1=j) + zgrid.isel(nVertLevelsP1=k))
                z2 = 0.5 * (
                    zgrid.isel(nVertLevelsP1=k) + zgrid.isel(nVertLevelsP1=level_idx)
                )
                w1_bound = (z0 - z2) / (z1 - z2)
                log_p_bound = w1_bound * np.log(pressure.isel(nVertLevels=j)) + (
                    1 - w1_bound
                ) * np.log(pressure.isel(nVertLevels=k))
                pressure_on_w.values[:, i] = np.exp(log_p_bound)

            ds["pressure_on_w"] = pressure_on_w
            ds["pressure_on_w"].attrs["units"] = "Pa"

        # --- 5. Derive Total Precipitation ---
        if "rainc" in ds and "rainnc" in ds and "tp06" not in ds:
            ds["tp06"] = ds["rainc"] + ds["rainnc"]

        # --- 6. Derive Surface Geopotential ---
        if "ter" in ds and "geopotential_at_surface" not in ds:
            ds["geopotential_at_surface"] = ds["ter"] * g

        # --- 7. Convert Geometric to Pressure Vertical Velocity ---
        if (
            "w" in ds
            and "pressure_on_w" in ds
            and "temperature" in ds
            and "qv" in ds
            and "pressure_vertical_velocity" not in ds
        ):
            logging.info("Deriving pressure vertical velocity from geometric w.")
            w_geom_q = ds["w"].metpy.quantify()
            p_on_w_q = ds["pressure_on_w"].metpy.quantify()
            temp_mid_q = ds["temperature"].metpy.quantify()
            qv_mid_q = ds["qv"].metpy.quantify()

            # Robust interpolation from cell centers to interfaces
            temp_mid_vals = temp_mid_q.values
            qv_mid_vals = qv_mid_q.values
            temp_on_w_vals = np.zeros_like(w_geom_q.values)
            qv_on_w_vals = np.zeros_like(w_geom_q.values)
            temp_on_w_vals[:, 1:-1] = 0.5 * (
                temp_mid_vals[:, :-1] + temp_mid_vals[:, 1:]
            )
            qv_on_w_vals[:, 1:-1] = 0.5 * (qv_mid_vals[:, :-1] + qv_mid_vals[:, 1:])
            temp_on_w_vals[:, 0] = temp_mid_vals[:, 0]
            temp_on_w_vals[:, -1] = temp_mid_vals[:, -1]
            qv_on_w_vals[:, 0] = qv_mid_vals[:, 0]
            qv_on_w_vals[:, -1] = qv_mid_vals[:, -1]

            # Re-create DataArrays with correct coords/dims and attach units
            temp_on_w_da = (
                xr.DataArray(temp_on_w_vals, dims=w_geom_q.dims, coords=w_geom_q.coords)
                * units.kelvin
            )
            qv_on_w_da = (
                xr.DataArray(qv_on_w_vals, dims=w_geom_q.dims, coords=w_geom_q.coords)
                * units.dimensionless
            )

            tv_on_w = temp_on_w_da * (1 + 0.61 * qv_on_w_da)
            rho_on_w = p_on_w_q / (dry_air_gas_constant * tv_on_w)

            omega = -w_geom_q * g * rho_on_w
            ds["pressure_vertical_velocity"] = omega.metpy.convert_units("Pa/s")

        # --- 8. Derive Mean Sea Level Pressure ---
        if (
            "surface_pressure" in ds
            and "ter" in ds
            and "t2m" in ds
            and "mean_sea_level_pressure" not in ds
        ):
            logging.info("Deriving mean sea level pressure from surface pressure.")
            p_sfc_q = ds["surface_pressure"].metpy.quantify()
            h_sfc_q = ds["ter"].metpy.quantify()
            t_sfc_q = ds["t2m"].metpy.quantify()

            p_msl = p_sfc_q * np.exp((g * h_sfc_q) / (dry_air_gas_constant * t_sfc_q))
            ds["mean_sea_level_pressure"] = p_msl

        return ds
