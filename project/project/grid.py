import regionmask
import xarray as xr
import xesmf as xe

GLOBAL_GRID = xe.util.grid_global(2, 2, lon1=359, cf=True)


class Regridder:
    def __init__(self, target_grid):
        self.regridders = {}  # realm -> regridder
        self.target_grid = target_grid

    def regrid(self, realm, dataset):
        if realm not in self.regridders:
            self.regridders[realm] = xe.Regridder(
                dataset, self.target_grid, "bilinear", ignore_degenerate=True
            )
        return self.regridders[realm](dataset, keep_attrs=True).drop_vars(
            ["latitude_longitude"]
        )


def mask_greenland_and_antarctica(da: xr.DataArray) -> xr.DataArray:
    # Select Greenland and Antarctica regions from IPCC AR6
    # See https://regionmask.readthedocs.io/en/latest/defined_scientific.html#land
    mask = (
        regionmask.defined_regions.ar6.land.mask_3D(GLOBAL_GRID)
        .sel(region=[0, 44, 45])
        .any(dim="region")
    )
    return da.where(~mask)


def mask_poles(da: xr.DataArray) -> xr.DataArray:
    return da.where(abs(da.lat) <= 88)
