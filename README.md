# AIFS2ENS Project

This directory is a scratch workspace for AIFS2ENS analysis and plotting.
The long-term copy should live in a small project-only repository that keeps
the local wrapper scripts and excludes Earth2Studio source code and large
generated data.

## What belongs here

- `plot_point_pdfs.py`
- `plot_aifs.py`
- `plot_both_ic.py`
- `plot_gfs_initial_condition.py`
- `run_aifs.py`
- `pyproject.toml`
- `README.md`

## What should not be tracked

- `*.zarr/`
- `*.png`
- `.venv/`
- `__pycache__/`

## How to Run

1. Get on a GPU node (`gpu-type=a100_80gb`) and make sure the `gcc` module is
   loaded.
2. Activate the environment:
   ```csh
   source .venv/bin/activate.csh
   ```
3. Run the analysis wrapper script:
   ```csh
   uv run python run_aifs.py
   ```

### `run_aifs.py` environment variables

- `AIFS_DATA_SOURCE` -- `ERA5_GFS` (default) uses ARCO (ERA5) as the primary
  analysis source with GFS fallback for variables missing from ARCO; `GFS`
  uses GFS only.
- `AIFS_START_TIMES` -- comma-separated init times, e.g.
  `"2025-06-24 00:00,2025-07-01 00:00"` (default: `2025-06-24 00:00`).
- `AIFS_OVERWRITE` -- `1`/`true`/`yes` deletes and rebuilds existing member
  stores instead of appending new init times to them (default: off).

If the uv environment gets corrupted, see notes in the hand-written notebook,
such as:
- uv cache clean
- uv add "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git"
- uv add earth2studio --extra aifs2ens
- uv pip install wheel packaging torch ninja
