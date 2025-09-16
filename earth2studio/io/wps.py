# =============================================================================
# Imports
# =============================================================================
import logging
import re
import struct
from pathlib import Path
from typing import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from earth2studio.io import IOBackend
from metpy.constants import g

# =============================================================================
# Main IO Backend Class
# =============================================================================
class MPASInitBackend(IOBackend):
    """
    An IOBackend that writes earth2studio data to the WPS intermediate binary
    format, suitable for initializing the MPAS-Atmosphere model and readable
    by WPS utilities like rd_intermediate.exe.
    """

    VARIABLE_MAP = {
        # Maps e2s name to (WPS Field Name, Units, Description, WPS Level Code)
        "t": ("TT", "K", "Temperature", -1),
        "z": ("GHT", "m", "Geopotential Height", -1),
        "u": ("UU", "m s-1", "Zonal Wind", -1),
        "v": ("VV", "m s-1", "Meridional Wind", -1),
        "q": ("SPECHUMD", "kg kg-1", "Specific Humidity", -1),
        "r": ("RH", "%", "Relative Humidity", -1),
        "msl": ("PMSL", "Pa", "Sea Level Pressure", 200100.0),
        "lsm": ("LANDSEA", "proportion", "Land/Sea Mask", 200100.0),
        "t2m": ("T2", "K", "2-meter Temperature", 2.0),
        "u10m": ("U10", "m s-1", "10-meter U-wind", 10.0),
        "v10m": ("V10", "m s-1", "10-meter V-wind", 10.0),
    }
    
    # Format for the main header record (excluding the initial version number).
    # Using '>' for big-endian byte order to match the WPS standard.
    HEADER_FORMAT = "> 24s f 32s 9s 25s 46s f i i i"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


    def __init__(
        self,
        path: Path,
        map_source: str = "earth2studio",
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.map_source = map_source

        if self.path.exists():
            logging.warning(f"Output file {self.path} exists and will be overwritten.")
            self.path.unlink()
        
        self.path.touch()

    def _write_header(
        self,
        file_handle,
        hdate: str,
        field: str,
        units: str,
        desc: str,
        xlvl: float,
        nx: int,
        ny: int,
    ):
        """Writes the header as two separate Fortran records: version and main."""
        # --- Write Record 1: File Format Version ---
        version_marker = struct.pack(">i", 4) # Integer is 4 bytes
        version_data = struct.pack(">i", 5)
        file_handle.write(version_marker)
        file_handle.write(version_data)
        file_handle.write(version_marker)

        # --- Write Record 2: Main Header Data ---
        header_marker = struct.pack(">i", self.HEADER_SIZE)
        
        # Manually pad strings with spaces to match Fortran's behavior
        hdate_padded = hdate.ljust(24)
        map_source_padded = self.map_source.ljust(32)
        field_padded = field.ljust(9)
        units_padded = units.ljust(25)
        desc_padded = desc.ljust(46)

        header_data = struct.pack(
            self.HEADER_FORMAT,
            hdate_padded.encode("utf-8"),
            0.0,  # Forecast hour (xfcst)
            map_source_padded.encode("utf-8"),
            field_padded.encode("utf-8"),
            units_padded.encode("utf-8"),
            desc_padded.encode("utf-8"),
            xlvl,
            nx,
            ny,
            0,    # iproj (0 for lat-lon grid)
        )
        
        file_handle.write(header_marker)
        file_handle.write(header_data)
        file_handle.write(header_marker)

    def _write_data(self, file_handle, field_data: np.ndarray):
        """Writes the data payload record."""
        data_bytes = field_data.astype(">f4").tobytes()
        data_marker = struct.pack(">i", len(data_bytes))

        file_handle.write(data_marker)
        file_handle.write(data_bytes)
        file_handle.write(data_marker)

    def _write_wind_flag(self, file_handle):
        """Writes the special is_wind_earth_relative flag for wind components."""
        flag_marker = struct.pack(">i", 4) # Integer is 4 bytes
        flag_data = struct.pack(">i", 1)
        file_handle.write(flag_marker)
        file_handle.write(flag_data)
        file_handle.write(flag_marker)


    def write(
        self,
        data_list: list,
        coords: OrderedDict,
        array_name: str | list[str], 
    ) -> None:
        """
        Processes data and writes its contents to the binary file.
        """
        if isinstance(array_name, str):
            array_name = [array_name]

        processed_data = []
        for d in data_list:
            if hasattr(d, 'cpu') and hasattr(d, 'numpy'):
                processed_data.append(d.cpu().numpy())
            else:
                processed_data.append(d)

        data_vars = {
            name: (list(coords.keys()), data) 
            for name, data in zip(array_name, processed_data)
        }
        ds_latlon = xr.Dataset(data_vars, coords=coords)
        ds_latlon = ds_latlon.squeeze("lead_time")
        
        time_obj = pd.to_datetime(coords["time"][0])
        hdate = time_obj.strftime("%Y-%m-%d_%H:%M:%S")

        logging.info(f"Writing binary data for time {hdate} to {self.path}")

        with open(self.path, "ab") as f:
            for var_name_e2s in ds_latlon.data_vars:
                var_name_str = str(var_name_e2s)
                base_var = None

                if var_name_str in self.VARIABLE_MAP:
                    base_var = var_name_str
                else:
                    match = re.match(r"([a-zA-Z]+)", var_name_str)
                    if match:
                        candidate = match.group(1)
                        if candidate in self.VARIABLE_MAP:
                            base_var = candidate

                if base_var is None:
                    logging.warning(f"No WPS mapping for '{var_name_str}', skipping write.")
                    continue

                field_name, units, desc, xlvl_code = self.VARIABLE_MAP[base_var]
                da_latlon = ds_latlon[var_name_e2s]

                # Perform necessary unit conversions before writing
                if base_var == "z":
                    logging.info("Converting geopotential to geopotential height for GHT field.")
                    # Use the magnitude of the metpy constant for the calculation
                    da_latlon = da_latlon / g.magnitude

                nx = da_latlon.sizes.get("lon", 1)
                ny = da_latlon.sizes.get("lat", 1)

                if "level" in da_latlon.dims:
                    for level in da_latlon["level"].values:
                        da_level = da_latlon.sel(level=level)
                        flat_data = da_level.values.flatten('F')
                        
                        self._write_header(
                            f, hdate, field_name, units, desc, level * 100, nx, ny
                        )
                        self._write_data(f, flat_data)
                        
                        if field_name in ["UU", "VV"]:
                            self._write_wind_flag(f)
                else:
                    flat_data = da_latlon.values.flatten('F')
                    # Use the specific level code from the map for 2D fields
                    self._write_header(
                        f, hdate, field_name, units, desc, xlvl_code, nx, ny
                    )
                    self._write_data(f, flat_data)
                    
                    if field_name in ["U10", "V10"]:
                        self._write_wind_flag(f)

    def close(self):
        logging.info("MPASInitBackend closed.")
        pass


