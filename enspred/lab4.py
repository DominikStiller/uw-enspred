from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np


obs_lat, obs_lon = 40, 200
sigma_obs = np.sqrt(10)

n_level = 50


def load_dataset(year):
    path = list(Path("/glade/work/chriss/ATMS544/lab4data").glob(f"ens_*{year}.nc"))[0]
    ds = xr.open_dataset(path)
    return ds


def get_exp_no(results):
    Ne = results[0]
    if Ne == 99:
        return 1
    elif Ne == 25:
        return 2
    else:
        raise "Invalid Ne"


def plot_field(
    axs,
    dss,
    field,
    colorbar=True,
    cbar_label=None,
    vmin=None,
    vmax=None,
    cmap="Blues",
    highlight_contour=None,
    rotate_cbar_ticks=False,
    **kwargs,
):
    if not isinstance(axs, list):
        axs = [axs]
    if not isinstance(dss, list):
        dss = [dss]

    vmin = vmin or min([ds[field].min() for ds in dss])
    vmax = vmax or max([ds[field].max() for ds in dss])

    for ax, ds in zip(axs, dss):
        # Use our own locator because the default locator does not respect vmin/vmax
        levels = mpl.ticker.MaxNLocator(n_level + 1).tick_values(vmin, vmax)
        cset = ax.contourf(
            ds.lon,
            ds.lat,
            ds[field],
            levels,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs,
        )
        for c in cset.collections:
            c.set_rasterized(True)

        if highlight_contour is not None:
            c_highlight = ax.contour(
                ds.lon,
                ds.lat,
                ds[field],
                [highlight_contour],
                transform=ccrs.PlateCarree(),
                colors="C1",
            )

    if colorbar:
        cb = plt.colorbar(cset, ax=ax, orientation="horizontal", label=cbar_label)
        if highlight_contour:
            cb.add_lines(c_highlight)
        if rotate_cbar_ticks:
            cb.ax.tick_params(rotation=15)


def make_H(idx):
    H = xr.DataArray(np.zeros((1, len(idx))), coords={"ob": [0], "elem": idx})
    H.loc[dict(lat=obs_lat, lon=obs_lon, field="z500")] = 1
    return H
