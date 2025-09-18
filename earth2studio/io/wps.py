# =============================================================================
# Imports
# =============================================================================
import logging
import re
import struct
from collections import OrderedDict
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
import xarray as xr
from metpy.constants import g

from earth2studio.io import IOBackend


# =============================================================================
# Main IO Backend Class
# =============================================================================
class WPSBackend(IOBackend):
    """
    An IOBackend that writes the final time step of a forecast to a dynamically
    named WPS intermediate binary file. It can propagate static fields from the
    initial time step to the final output.
    """

    VARIABLE_MAP = {
        # Maps e2s name to (WPS Field Name, Units, Description, WPS Level Code)
        "t": ("TT", "K", "Temperature", -1),
        "z": ("GHT", "m", "Geopotential Height", -1),
        "u": ("UU", "m s-1", "Zonal Wind", -1),
        "v": ("VV", "m s-1", "Meridional Wind", -1),
        "q": ("SPECHUMD", "kg kg-1", "Specific Humidity", -1),
        "r": ("RH", "%", "Relative Humidity", -1),
        "w": ("WW", "Pa s-1", "Vertical Velocity", -1),
        "msl": ("PMSL", "Pa", "Sea Level Pressure", 201300.0),
        "lsm": ("LANDSEA", "proportion", "Land/Sea Mask", 200100.0),
        "t2m": ("T2", "K", "2-meter Temperature", 200100.0),
        "u10m": ("U10", "m s-1", "10-meter U-wind", 200100.0),
        "v10m": ("V10", "m s-1", "10-meter V-wind", 200100.0),
        "tp": ("PRECIP", "kg m-2", "Total Precipitation", 200100.0),
    }

    # Format strings based on the reference Fortran source.
    # Using '>' for big-endian byte order to match the WPS standard.
    HEADER_FORMAT = "> 24s f 32s 9s 25s 46s f i i i"
    PROJECTION_FORMAT_LATLON = "> 8s f f f f f"

    def __init__(
        self,
        path: Path,
        map_source: str = "earth2studio",
        static_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the WPSBackend.

        Parameters
        ----------
        path : Path
            The output directory where the final forecast file will be saved.
        map_source : str, optional
            The name of the source model, used for generating the output filename.
            Defaults to "earth2studio".
        static_fields : list[str], optional
            A list of variable names to treat as static. The data for these
            fields will be taken from the first forecast step and propagated to
            the final output.
        """
        super().__init__()
        self.output_dir = path
        self.map_source = map_source
        self.static_field_names = static_fields or []
        self.stored_static_data: xr.Dataset | None = None
        self.final_data_package: tuple | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_record(self, file_handle: IO[bytes], data: bytes) -> None:
        """Writes a Fortran-style record (marker, data, marker)."""
        marker = struct.pack(">i", len(data))
        file_handle.write(marker)
        file_handle.write(data)
        file_handle.write(marker)

    def write(
        self,
        data_list: list[np.ndarray],
        coords: OrderedDict[str, np.ndarray],
        array_name: str | list[str],
    ) -> None:
        """
        Stores the data from the current forecast step. Captures static fields
        from the first step.
        """
        # Capture static fields on the very first write call
        if self.stored_static_data is None and self.static_field_names:
            logging.info(
                f"Capturing static fields from initial time step: {self.static_field_names}"
            )

            temp_array_name = (
                [array_name] if isinstance(array_name, str) else array_name
            )

            temp_processed_data = [
                d.cpu().numpy() if hasattr(d, "cpu") else d for d in data_list
            ]
            temp_data_vars = {
                name: (list(coords.keys()), data)
                for name, data in zip(temp_array_name, temp_processed_data)
            }
            initial_ds = xr.Dataset(temp_data_vars, coords=coords)

            # Filter the dataset to only include the specified static fields
            vars_to_keep = [
                v for v in self.static_field_names if v in initial_ds.data_vars
            ]
            self.stored_static_data = initial_ds[vars_to_keep]

        # Always store the latest data package for the final forecast step
        self.final_data_package = (data_list, coords, array_name)

    def _write_complete_field(
        self,
        f: IO[bytes],
        da: xr.DataArray,
        hdate: str,
        field_name: str,
        units: str,
        desc: str,
        xlvl: float,
        nx: int,
        ny: int,
        coords: OrderedDict[str, np.ndarray],
        xfcst: float,
    ) -> None:
        """Writes all five Fortran records for a single 2D field."""
        # --- Record 1: Version ---
        self._write_record(f, struct.pack(">i", 5))

        # --- Record 2: Main Header ---
        header_data = struct.pack(
            self.HEADER_FORMAT,
            hdate.ljust(24).encode("utf-8"),
            xfcst,
            self.map_source.ljust(32).encode("utf-8"),
            field_name.ljust(9).encode("utf-8"),
            units.ljust(25).encode("utf-8"),
            desc.ljust(46).encode("utf-8"),
            xlvl,
            nx,
            ny,
            0,  # iproj = 0
        )
        self._write_record(f, header_data)

        # --- Record 3: Projection ---
        lat = coords["lat"]
        lon = coords["lon"]
        proj_data = struct.pack(
            self.PROJECTION_FORMAT_LATLON,
            "SWCORNER".ljust(8).encode("utf-8"),
            float(lat[0]),
            float(lon[0]),
            float(lat[1] - lat[0] if len(lat) > 1 else 0.0),
            float(lon[1] - lon[0] if len(lon) > 1 else 0.0),
            6371229.0,  # Earth radius in m
        )
        self._write_record(f, proj_data)

        # --- Record 4: Wind Flag ---
        # is_wind_grid_rel = .FALSE. -> 0 for Earth-relative winds
        self._write_record(f, struct.pack(">i", 0))

        # --- Record 5: Data Payload ---
        flat_data = da.values.astype(">f4")
        self._write_record(f, flat_data.tobytes())

    def close(self) -> None:
        """
        Writes the final stored forecast step, injecting static fields,
        to a dynamically named binary file.
        """
        if self.final_data_package is None:
            logging.warning("WPSBackend closing, but no data was provided to write.")
            return

        data_list, coords, array_name = self.final_data_package

        if isinstance(array_name, str):
            array_name = [array_name]

        valid_time = pd.to_datetime(coords["time"][0])

        # --- Create dynamic filename ---
        model_prefix = self.map_source.upper()[0:4]
        time_str = valid_time.strftime("%Y-%m-%d_%H")
        filename = f"{model_prefix}:{time_str}"
        output_path = self.output_dir / filename

        if output_path.exists():
            logging.warning(
                f"Output file {output_path} already exists and will be overwritten."
            )
            output_path.unlink()

        # --- Process and write the data ---
        processed_data = [
            d.cpu().numpy() if hasattr(d, "cpu") else d for d in data_list
        ]
        data_vars = {
            name: (list(coords.keys()), data)
            for name, data in zip(array_name, processed_data)
        }
        ds = xr.Dataset(data_vars, coords=coords).squeeze("lead_time")

        # --- Inject stored static fields ---
        if self.stored_static_data is not None:
            logging.info("Overriding variables with stored static fields.")
            # Squeeze lead_time from static data to match dynamic data
            static_ds_squeezed = self.stored_static_data.squeeze("lead_time", drop=True)
            ds.update(static_ds_squeezed)

        hdate = valid_time.strftime("%Y-%m-%d_%H:%M:%S")

        xfcst: float
        if "lead_time" in coords:
            lead_time_delta = coords["lead_time"][0]
            xfcst = lead_time_delta / np.timedelta64(1, "h")
        else:
            xfcst = 0.0

        logging.info(
            f"Writing final forecast step for time {hdate} (F{xfcst:03.0f}) to {output_path}"
        )

        with open(output_path, "ab") as f:
            for var_name_str in ds.data_vars:
                # (Logic to find base_var and write field is the same as before)
                base_var = None
                if str(var_name_str) in self.VARIABLE_MAP:
                    base_var = str(var_name_str)
                else:
                    match = re.match(r"([a-zA-Z]+)", str(var_name_str))
                    if match and match.group(1) in self.VARIABLE_MAP:
                        base_var = match.group(1)

                if not base_var:
                    logging.warning(f"No WPS mapping for '{var_name_str}', skipping.")
                    continue

                field_name, units, desc, xlvl_code = self.VARIABLE_MAP[base_var]
                da = ds[var_name_str]

                if "time" in da.dims:
                    da = da.squeeze("time")
                if "lat" in da.dims and "lon" in da.dims:
                    da = da.transpose(..., "lat", "lon")
                if base_var == "z":
                    da = da / g.magnitude

                nx, ny = da.sizes.get("lon", 1), da.sizes.get("lat", 1)

                final_xlvl: float
                if str(var_name_str) == "z":
                    final_xlvl = 200100.0
                elif xlvl_code == -1:
                    level_match = re.search(r"(\d+)$", str(var_name_str))
                    if level_match:
                        level_hpa = float(level_match.group(1))
                        final_xlvl = level_hpa * 100.0
                    else:
                        continue
                else:
                    final_xlvl = xlvl_code

                self._write_complete_field(
                    f,
                    da,
                    hdate,
                    field_name,
                    units,
                    desc,
                    final_xlvl,
                    nx,
                    ny,
                    coords,
                    xfcst,
                )

        logging.info(f"WPSBackend closed. Final output: {output_path}")
