# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.utils.type import LeadTimeArray, TimeArray


def timearray_to_datetime(time: TimeArray) -> list[datetime]:
    """Simple converter from numpy datetime64 array into a list of datetimes.

    Parameters
    ----------
    time : TimeArray
        Numpy datetime64 array

    Returns
    -------
    list[datetime]
        List of datetime object
    """
    _unix = np.datetime64(0, "s")
    _ds = np.timedelta64(1, "s")
    # TODO: Update to
    # time = [datetime.fromtimestamp((date - _unix) / _ds, UTC) for date in time]
    time = [datetime.utcfromtimestamp((date - _unix) / _ds) for date in time]

    return time


def leadtimearray_to_timedelta(lead_time: LeadTimeArray) -> list[timedelta]:
    """Simple converter from numpy timedelta64 array into a list of timedeltas

    Parameters
    ----------
    lead_time : TimeArray
        Numpy timedelta64 array

    Returns
    -------
    list[timedelta]
        List of timedelta object
    """
    # microsecond is smallest unit python timedelta supports
    return [
        timedelta(microseconds=int(time.astype("timedelta64[us]").astype(int)))
        for time in lead_time
    ]


def to_time_array(time: list[str] | list[datetime] | TimeArray) -> TimeArray:
    """A general converter for various time iterables into a numpy datetime64 array

    Parameters
    ----------
    time : list[str] | list[datetime] | TimeArray
        Time object iterable

    Returns
    -------
    TimeArray
        Numpy array of datetimes

    Raises
    ------
    TypeError
        If element in iterable is not a value time object
    """
    output = []

    for ts in time:
        if isinstance(ts, datetime):
            output.append(np.datetime64(ts))
        elif isinstance(ts, str):
            output.append(np.datetime64(ts))
        elif isinstance(ts, np.datetime64):
            output.append(ts)
        else:
            raise TypeError(
                f"Invalid time data type provided {ts}, should be datetime, string or np.datetime64"
            )

    return np.array(output).astype("datetime64[ns]")


def xtime(ds: xr.Dataset) -> xr.Dataset:
    """
    Decodes time variables from an MPAS file, calculates the forecast hour,
    extracts the member ID, and assigns them as new coordinates or variables
    in the xarray.Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset, typically from an MPAS output file. It is expected
        to have 'initial_time' and 'xtime' variables as byte strings.

    Returns
    -------
    xarray.Dataset
        The dataset with added coordinates for 'initial_time', 'valid_time',
        'mem' (if found), and a new data variable 'forecastHour'.
    """

    # --- Process Initial Time ---
    logger.info("Decoding 'initial_time' variable.")
    initial_time_str = ds["initial_time"].load().item().decode("utf-8").strip()
    initial_time = datetime.strptime(initial_time_str, "%Y-%m-%d_%H:%M:%S")
    # Assign as a scalar coordinate
    ds = ds.expand_dims("initial_time").assign_coords(initial_time=[initial_time])

    # --- Extract Member ID ---
    if "source" in ds.encoding:
        filename = Path(ds.encoding["source"])
        mem_parts = [p for p in filename.parts if p.startswith("mem")]
        if mem_parts:
            mem_str = mem_parts[0].lstrip("mem_")
            if mem_str.isdigit():
                mem = int(mem_str)
                # Assign member ID as a scalar coordinate
                ds = ds.assign_coords(mem=mem)
                logger.info(f"Found and assigned member ID: {mem}")

    # --- Process Valid Time ---
    logger.info("Decoding 'xtime' (valid time) variable.")

    # Define a function that parses a single byte-string time
    def parse_time(t_bytes: bytes) -> datetime:
        return datetime.strptime(t_bytes.decode("utf-8").strip(), "%Y-%m-%d_%H:%M:%S")

    parse_time_vec = np.vectorize(parse_time)
    time_values = parse_time_vec(ds["xtime"].values).flatten()

    # Assign the new datetime array/scalar as the 'time' coordinate.
    ds = ds.assign_coords(time=("Time", time_values))

    ds = ds.swap_dims({"Time": "time"})

    # Drop the original, now redundant, variable
    ds = ds.drop_vars(["xtime", "Time"], errors="ignore")

    return ds
