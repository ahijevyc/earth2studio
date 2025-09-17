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
    An IOBackend that writes earth2studio data to the WPS intermediate binary
    format, suitable for initializing models like WRF or MPAS and readable
    by WPS utilities like rd_intermediate.exe.
    """

    VARIABLE_MAP = {
        # Maps e2s name to (WPS Field Name, Units, Description, WPS Level Code)
        # For 3D vars, -1 indicates the level is encoded in the variable name.
        "t": ("TT", "K", "Temperature", -1),
        "z": ("GHT", "m", "Geopotential Height", -1),
        "u": ("UU", "m s-1", "Zonal Wind", -1),
        "v": ("VV", "m s-1", "Meridional Wind", -1),
        "q": ("SPECHUMD", "kg kg-1", "Specific Humidity", -1),
        "r": ("RH", "%", "Relative Humidity", -1),
        # For 2D vars, the level code is a fixed value.
        "msl": ("PMSL", "Pa", "Sea Level Pressure", 201300.0),
        "lsm": ("LANDSEA", "proportion", "Land/Sea Mask", 200100.0),
        "t2m": ("T2", "K", "2-meter Temperature", 200100.0),
        "u10m": ("U10", "m s-1", "10-meter U-wind", 200100.0),
        "v10m": ("V10", "m s-1", "10-meter V-wind", 200100.0),
    }

    # Format strings based on the reference Fortran source.
    # Using '>' for big-endian byte order to match the WPS standard.
    HEADER_FORMAT = "> 24s f 32s 9s 25s 46s f i i i"
    PROJECTION_FORMAT_LATLON = "> 8s f f f f f"

    def __init__(
        self,
        path: Path,
        map_source: str = "earth2studio",
        **kwargs: Any,
    ) -> None:
        """Initialize the WPSBackend."""
        super().__init__()
        self.path = path
        self.map_source = map_source

        if self.path.exists():
            logging.warning(f"Output file {self.path} exists and will be overwritten.")
            self.path.unlink()

        self.path.touch()

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
        """Processes data and writes its contents to the binary file."""
        if isinstance(array_name, str):
            array_name = [array_name]

        processed_data = [
            d.cpu().numpy() if hasattr(d, "cpu") else d for d in data_list
        ]
        data_vars = {
            name: (list(coords.keys()), data)
            for name, data in zip(array_name, processed_data)
        }
        ds = xr.Dataset(data_vars, coords=coords).squeeze("lead_time")

        valid_time = pd.to_datetime(coords["time"][0])
        hdate = valid_time.strftime("%Y-%m-%d_%H:%M:%S")

        # Calculate forecast hour from lead_time coordinate
        xfcst: float
        if "lead_time" in coords:
            lead_time_delta = coords["lead_time"][0]
            # Convert numpy timedelta64 to floating point hours
            xfcst = lead_time_delta / np.timedelta64(1, "h")
        else:
            logging.warning(
                "'lead_time' not in coordinates, defaulting forecast hour to 0.0."
            )
            xfcst = 0.0

        logging.info(
            f"Writing binary data for time {hdate} (F{xfcst:03.0f}) to {self.path}"
        )

        with open(self.path, "ab") as f:
            for var_name_str in ds.data_vars:
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

                if base_var == "z":
                    da = da / g.magnitude

                nx, ny = da.sizes.get("lon", 1), da.sizes.get("lat", 1)

                # Determine the vertical level for the header
                final_xlvl: float
                if str(var_name_str) == "z":
                    # Special case for surface geopotential height
                    final_xlvl = 200100.0
                elif xlvl_code == -1:
                    # 3D var: parse level from name (e.g., z500 -> 500 hPa)
                    level_match = re.search(r"(\d+)$", str(var_name_str))
                    if level_match:
                        level_hpa = float(level_match.group(1))
                        final_xlvl = level_hpa * 100.0  # Convert hPa to Pa
                    else:
                        logging.warning(
                            f"Could not parse pressure level from '{var_name_str}'. Skipping."
                        )
                        continue
                else:
                    # Other 2D var: use the fixed code from the map
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
            6371229.0,  # Earth radius in m (ERA5)
        )
        self._write_record(f, proj_data)

        # --- Record 4: Wind Flag ---
        # is_wind_grid_rel = .FALSE. -> 0 for Earth-relative winds
        self._write_record(f, struct.pack(">i", 0))

        # --- Record 5: Data Payload ---
        flat_data = da.values.flatten("F").astype(">f4")
        self._write_record(f, flat_data.tobytes())

    def close(self) -> None:
        """Close the backend and perform any necessary cleanup."""
        logging.info("WPSBackend closed.")
        pass
