import os
from pathlib import Path
from typing import Union

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from cartopy.util import add_cyclic_point


def set_plotting_theme(force_light=False):
    dark = plt.rcParams["figure.facecolor"] == "black"
    if force_light:
        dark = False

    plt.style.use("default")
    sb.set(
        context="paper",
        style="white",
        palette=[
            "#0C2340",
            "#A50034",
            "#00A6D6",
            "#EF60A3",
            "#FFB81C",
            "#EC6842",
            "#6F1D77",
            "#009B77",
        ],
        font_scale=1.7,
        font="serif",
        rc={
            "font.family": "serif",
            "lines.linewidth": 1.2,
            "axes.titleweight": "bold",
            # "axes.labelweight": "light",
            # "font.weight": "light",
            # "mathtext.default": "regular",
            "figure.figsize": (12, 5),
            "figure.dpi": 450,
            "figure.constrained_layout.use": True,
            "axes.axisbelow": False,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.bottom": False,
            "ytick.minor.left": False,
            "xtick.minor.top": False,
            "ytick.minor.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
        },
    )

    if dark:
        # For example, due to VS Code dark theme
        plt.style.use("dark_background")


def save_plot(plots_folder: Union[Path, str], name: str, fig=None, type="pdf"):
    if isinstance(plots_folder, str):
        plots_folder = Path(plots_folder)

    plots_folder.mkdir(parents=True, exist_ok=True)

    if fig is None:
        fig = plt.gcf()
    fig.savefig(
        os.path.join(plots_folder, f"{name}.{type}"),
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def format_plot(
    x_major_locator=None,
    y_major_locator=None,
    x_minor_locator=None,
    y_minor_locator=None,
    tight_layout=False,
    zeroline=False,
    major_grid=False,
    minor_grid=False,
):
    fig = plt.gcf()
    for ax in fig.axes:
        if hasattr(ax, "_colorbar"):
            continue

        if zeroline:
            ax.axhline(0, linewidth=1.5, c="black")

        x_major_locator_ax = x_major_locator
        if not x_major_locator_ax:
            if ax.get_xscale() == "log":
                x_major_locator_ax = mpl.ticker.LogLocator()
            else:
                x_major_locator_ax = mpl.ticker.AutoLocator()

        y_major_locator_ax = y_major_locator
        if not y_major_locator_ax:
            if ax.get_yscale() == "log":
                y_major_locator_ax = mpl.ticker.LogLocator()
            else:
                y_major_locator_ax = mpl.ticker.AutoLocator()

        ax.get_xaxis().set_major_locator(x_major_locator_ax)
        ax.get_yaxis().set_major_locator(y_major_locator_ax)

        x_minor_locator_ax = x_minor_locator
        if not x_minor_locator_ax:
            if ax.get_xscale() == "log":
                x_minor_locator_ax = mpl.ticker.LogLocator(base=10, subs="auto", numticks=100)
            else:
                x_minor_locator_ax = mpl.ticker.AutoMinorLocator()

        y_minor_locator_ax = y_minor_locator
        if not y_minor_locator_ax:
            if ax.get_yscale() == "log":
                y_minor_locator_ax = mpl.ticker.LogLocator(base=10, subs="auto", numticks=100)
            else:
                y_minor_locator_ax = mpl.ticker.AutoMinorLocator()

        ax.get_xaxis().set_minor_locator(x_minor_locator_ax)
        ax.get_yaxis().set_minor_locator(y_minor_locator_ax)

        if major_grid:
            ax.grid(visible=True, which="major", linewidth=1.0, linestyle=":")
            ax.set_axisbelow(True)
        if minor_grid:
            ax.grid(
                visible=True,
                which="minor",
                linewidth=0.5,
                linestyle=(0, (2, 6)),
                alpha=0.8,
            )

    fig.align_ylabels()

    if tight_layout:
        fig.tight_layout(pad=0.1, h_pad=0.4, w_pad=0.4)


def plot_field(
    axs,
    das,
    colorbar=True,
    cbar_label=None,
    vmin=None,
    vmax=None,
    cmap="Blues",
    highlight_contour=None,
    rotate_cbar_ticks=False,
    n_level=50,
    same_limits=False,
    **kwargs,
):
    if not (isinstance(axs, list) or isinstance(axs, np.ndarray)):
        axs = [axs]
    if not (isinstance(axs, list) or isinstance(axs, np.ndarray)):
        das = [das]

    das = [da.load() for da in das]

    vmin = vmin or min([da.min() for da in das])
    vmax = vmax or max([da.max() for da in das])

    if same_limits:
        max_v = max(abs(vmin), abs(vmax))
        vmax = max_v
        vmin = -max_v

    for ax, da in zip(axs, das):
        lat = da.lat
        da, lon = add_cyclic_point(da.values, coord=da.lon)

        # Use our own locator because the default locator does not respect vmin/vmax
        levels = mpl.ticker.MaxNLocator(n_level + 1).tick_values(vmin, vmax)
        cset = ax.contourf(
            lon,
            lat,
            da,
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
                lon,
                lat,
                da,
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


set_plotting_theme()
