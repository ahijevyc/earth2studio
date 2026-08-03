import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Patch
from pathlib import Path

# Import both possible data sources
from earth2studio.data import ARCO, GFS

def main():
    ic = 'ERA5WithGFSFallback'
    ensemble_store = Path(f"AIFS2ENS_{ic}_ensemble.zarr")
    if False and ensemble_store.exists():
        print(f"Opening repacked ensemble store: {ensemble_store}...")
        ds = xr.open_zarr(ensemble_store, consolidated=True)
    else:
        print(f"Opening member stores...")
        ds = xr.open_mfdataset(
            f"AIFS2ENS_{ic}_member_*.zarr",
            engine="zarr",
            combine="nested",
            concat_dim="ensemble",
            consolidated=False,
            parallel=True,
            compat="no_conflicts",
            join="outer",
            coords='minimal',
        )
    print(f"Done.")
    num_members = ds.sizes["ensemble"]

    target_var = "z500"
    lead_idx = 4
    
    # Process model fields (Convert geopotential to height in meters)
    grid_at_lead = ds[target_var].isel(time=0, lead_time=lead_idx) / 9.80665
    ens_mean = grid_at_lead.mean(dim="ensemble")
    q25 = grid_at_lead.quantile(0.25, dim="ensemble")
    q75 = grid_at_lead.quantile(0.75, dim="ensemble")
    ens_min = grid_at_lead.min(dim="ensemble")
    ens_max = grid_at_lead.max(dim="ensemble")

    # 2. Compute the exact valid time via Pandas
    init_time_pd = pd.Timestamp(ds.time.values[0])
    lead_time_pd = pd.Timedelta(ds.lead_time.values[lead_idx])
    valid_time_pd = init_time_pd + lead_time_pd
    valid_datetime = valid_time_pd.to_pydatetime()
    valid_str = valid_time_pd.strftime("%Y-%m-%d %H:%M UTC")

    # 3. Dynamic Source Selection based on the year
    # ARCO (ERA5) cuts off on Dec 31, 2025.
    cutoff_date = pd.Timestamp("2025-12-31 23:59:59")
    
    if valid_time_pd <= cutoff_date:
        print(f"[{valid_str}] -> Before 2026: Fetching historical verification truth from ERA5 (ARCO)...")
        truth_source = ARCO()
        truth_label = "ERA5 Reanalysis Truth (ARCO)"
    else:
        print(f"[{valid_str}] -> 2026 or later: Fetching operational verification analysis from GFS...")
        truth_source = GFS()
        truth_label = "GFS Operational Analysis Truth"

    # Pull validation grid from selected source
    truth_raw = truth_source(valid_datetime, target_var)
    
    # Convert geopotential to height in meters and strip unused dimensions
    truth_field = truth_raw / 9.80665
    truth_field = truth_field.squeeze()

    # 4. Set up Map Layout over CONUS
    levels = [5400, 5700, 5880]
    map_proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=37.5)
    data_proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': map_proj})
    ax.set_extent([-125, -60, 15, 50], crs=data_proj)
    
    ax.add_feature(cfeature.LAND, facecolor='#f7f7f7')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f2f1')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray')

    # 5. Fill the contour-location envelopes for each level.
    # A gridpoint is shaded when a given contour level lies between the lower/upper fields.
    for level in levels:
        min_max_band = xr.where((ens_min <= level) & (ens_max >= level), 1.0, np.nan)
        q25_q75_band = xr.where((q25 <= level) & (q75 >= level), 1.0, np.nan)

        ax.contourf(
            ds.lon,
            ds.lat,
            min_max_band,
            levels=[0.5, 1.5],
            transform=data_proj,
            colors=['#d95f0e'],
            alpha=0.10,
            antialiased=True,
        )
        ax.contourf(
            ds.lon,
            ds.lat,
            q25_q75_band,
            levels=[0.5, 1.5],
            transform=data_proj,
            colors=['#08519c'],
            alpha=0.18,
            antialiased=True,
        )

    # 6. Overlay the Ensemble Mean in Bold Black
    cs_mean = ax.contour(
        ds.lon, ds.lat, ens_mean,
        levels=levels, transform=data_proj,
        colors='#111111', linewidths=3.0
    )
    ax.clabel(cs_mean, fmt='%1.0fm', inline=True, fontsize=10, colors='#111111')

    # 7. Overlay the validation observation profile in Dashed Bold Red.
    cs_truth = ax.contour(
        truth_field.lon, truth_field.lat, truth_field,
        levels=levels, transform=data_proj,
        colors='#b30000', linewidths=2.8, linestyles='dashed'
    )
    ax.clabel(cs_truth, fmt='%1.0fm', inline=True, fontsize=10, colors='#b30000')

    # 8. Legend and Titles.
    lines = [
        Patch(facecolor='#08519c', edgecolor='none', alpha=0.18),
        Patch(facecolor='#d95f0e', edgecolor='none', alpha=0.10),
        plt.Line2D([0], [0], color='#111111', linewidth=3.0),
        plt.Line2D([0], [0], color='#b30000', linewidth=2.8, linestyle='dashed')
    ]
    ax.legend(
        lines,
        ['25th-75th Envelope', 'Min-Max Envelope', f'Ensemble Mean (n={num_members})', truth_label],
        loc='lower left',
        ncol=1,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        handlelength=3.0,
        columnspacing=1.4,
        labelspacing=0.7
    )

    lead_hours = int(lead_time_pd.total_seconds() / 3600)
    ax.set_title(
        f"AIFS2ENS 500hPa Geopotential Height Verification Summary\n"
        f"Forecast Valid: {valid_str} (+{lead_hours}h Lead from {ic} Init)", 
        fontsize=13, weight='bold', pad=15
    )

    output_png = f"aifs_{ic}_z500_dynamic_verification.png"
    plt.savefig(output_png, bbox_inches='tight', dpi=150)
    print(f"Success! Verification plot saved to {output_png}")

if __name__ == "__main__":
    main()
