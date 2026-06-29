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

## Recommended long-term home

Keep this as a small project repo, for example `aifs2ens/`, and point the
Earth2Studio dependency at a dedicated branch in your `ahijevyc/earth2studio`
fork instead of vendoring the library here.

Example dependency line in `pyproject.toml`:

```toml
[tool.uv.sources]
earth2studio = { git = "https://github.com/ahijevyc/earth2studio.git", branch = "ahijevyc/aifs2ens" }
```

If you prefer a different branch name, update that line accordingly.

## Recreate the environment

```bash
uv sync
uv run python plot_point_pdfs.py
```

## Notes

- This project is separate from other Earth2Studio project directories such as
	`earth2studio-mpas` and `earth2studio-pangu`.
- Large ensemble member stores stay in scratch storage and are not committed.
