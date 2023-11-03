import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def plot_ensemble_mean(ax, ds, field):
    ax.contourf(
        ds.lon, ds.lat,
        ds[field].mean("ensemble"),
        5, transform=ccrs.PlateCarree())