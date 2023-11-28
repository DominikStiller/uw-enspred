from typing import Optional

import dask
import numpy as np
import xarray as xr

from project.eof import EOF
from project.logger import get_logger
from project.util import stack_state

logger = get_logger(__name__)


class Detrend:
    def __init__(self):
        self.time_mean: Optional[xr.DataArray] = None
        self.state_mean: Optional[xr.DataArray] = None
        self.coeffs: Optional[xr.DataArray] = None

    def fit(self, da: xr.DataArray):
        self.time_mean = da.time.mean()
        self.state_mean = da.mean(dim="time")
        coeffs, _, _, _ = dask.array.linalg.lstsq(
            dask.array.from_array(np.atleast_2d(da.time - self.time_mean).T),
            (da - self.state_mean).data.T,
        )
        self.coeffs = xr.DataArray(
            coeffs.compute()[0, :], coords=dict(state=da.state)
        ).squeeze()

    def _linear_trend(self, da: xr.DataArray) -> xr.DataArray:
        return self.state_mean + (da.time - self.time_mean) @ self.coeffs

    def forward(self, da: xr.DataArray) -> xr.DataArray:
        # De-trend
        return da - self._linear_trend(da)

    def backward(self, da: xr.DataArray) -> xr.DataArray:
        # Add trend
        return da + self._linear_trend(da)


class PhysicalSpaceForecastSpaceMapper:
    def __init__(self, k, l, direct_fields: list[str] = None):
        self.k = k
        self.l = l
        self.detrend: Detrend = Detrend()
        self.eofs_individual: dict[str, EOF] = {}
        self.variances: Optional[xr.DataArray] = None
        self.eof_joint: EOF = EOF(self.l)

    def fit(self, ds: xr.Dataset):
        logger.info("Calculating field variances")
        self.variances = ds.var().compute()

        da = stack_state(ds)

        logger.info("Detrending data")
        self.detrend.fit(da)
        da = self.detrend.forward(da)
        da = da.fillna(0)

        ds_eof = {}

        for field in ds.keys():
            logger.info(f"Fitting EOF for {field}")

            da_var = da.sel(field=field)
            da_var *= np.sqrt(np.cos(np.radians(da_var.lat)))

            eof_individual = EOF(self.k)
            eof_individual.fit(da_var)
            ds_eof[field] = eof_individual.project_forwards(da_var)
            self.eofs_individual[str(field)] = eof_individual

        ds_eof = xr.Dataset(ds_eof)
        da_eof_normalized = stack_state(ds_eof / self.variances)

        logger.info("Fitting joint EOF")
        self.eof_joint.fit(da_eof_normalized)
        # TODO append whole OHC
