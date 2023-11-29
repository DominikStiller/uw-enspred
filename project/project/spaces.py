from typing import Optional

import dask
import numpy as np
import xarray as xr

from project.eof import EOF
from project.logger import get_logger
from project.util import stack_state, unstack_state, field_complement

logger = get_logger(__name__)


class Detrend:
    def __init__(self):
        self.time_mean: Optional[xr.DataArray] = None
        self.state_mean: Optional[xr.DataArray] = None
        self.coeffs: Optional[xr.DataArray] = None

    def fit(self, da: xr.DataArray):
        da = da.dropna("state")

        self.time_mean = da.time.mean()
        self.state_mean = da.mean(dim="time")

        time_demeaned = np.atleast_2d(da.time - self.time_mean).T
        state_demeaned = (da - self.state_mean).data.T

        if time_demeaned.dtype == np.dtype("timedelta64[ns]"):
            # timedeltas need to be converted to pure numbers
            time_demeaned = time_demeaned.astype("int64")

        coeffs, _, _, _ = dask.array.linalg.lstsq(
            dask.array.from_array(time_demeaned), state_demeaned
        )
        self.coeffs = xr.DataArray(
            coeffs.compute()[0, :], coords=dict(state=da.state)
        ).squeeze()

    def _linear_trend(self, da: xr.DataArray) -> xr.DataArray:
        time_demeaned = da.time - self.time_mean
        if time_demeaned.dtype == np.dtype("timedelta64[ns]"):
            time_demeaned = time_demeaned.astype("int64")

        return self.state_mean + time_demeaned @ self.coeffs

    def forward(self, da: xr.DataArray) -> xr.DataArray:
        # De-trend
        return da - self._linear_trend(da)

    def backward(self, da: xr.DataArray) -> xr.DataArray:
        # Add trend
        return da + self._linear_trend(da)


class PhysicalSpaceForecastSpaceMapper:
    def __init__(
        self,
        k,
        l,
        direct_fields: list[str] = None,
        standardized_initially_fields: list[str] = None,
    ):
        """

        Args:
            k: EOFs to retain in first step
            l: EOFs to retain in second step
            direct_fields: fields to directly append to state after step 1 instead of including them in step 2
                PH20 does this for OHC700m
            standardized_first_fields: fields to standardize before step 1
                PH21 does this for pr
        """
        self.k = k
        self.l = l
        self.direct_fields = direct_fields
        self.standardized_initially_fields = standardized_initially_fields or []

        self.detrend: Detrend = Detrend()
        self.eofs_individual: dict[str, EOF] = {}
        self.standard_deviations: Optional[xr.Dataset] = None
        self.eof_joint: EOF = EOF(self.l)

    def fit(self, ds: xr.Dataset):
        logger.info("PhysicalSpaceForecastSpaceMapper.fit()")
        logger.info("Calculating field variances")

        self.standard_deviations = ds.std().compute()

        da = stack_state(ds)

        logger.info("Detrending data")
        self.detrend.fit(da)
        da = self.detrend.forward(da)
        da = da.fillna(0)

        if self.standardized_initially_fields:
            da = unstack_state(da)
            for field in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} before individual EOF")
                da[field] /= self.standard_deviations[field]
            da = stack_state(da)

        ds_eof = {}

        for field in ds.keys():
            logger.info(f"Fitting EOF for {field}")

            da_field = da.sel(field=field)
            da_field *= np.sqrt(np.cos(np.radians(da_field.lat)))

            eof_individual = EOF(self.k)
            eof_individual.fit(da_field)
            ds_eof[field] = eof_individual.project_forwards(da_field)
            self.eofs_individual[str(field)] = eof_individual

        ds_eof_normalized = xr.Dataset(ds_eof)
        for field in ds_eof.keys():
            if field not in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} after individual EOF")
                ds_eof_normalized[field] /= self.standard_deviations[field]

        not_direct_fields = field_complement(ds_eof_normalized, self.direct_fields)
        ds_eof_normalized_for_second_eof = ds_eof_normalized[not_direct_fields]

        logger.info(f"Fitting joint EOF for {', '.join(not_direct_fields)}")
        self.eof_joint.fit(stack_state(ds_eof_normalized_for_second_eof))

    def forward(self, ds: xr.Dataset) -> xr.DataArray:
        logger.info("PhysicalSpaceForecastSpaceMapper.forward()")
        da = stack_state(ds)
        da = self.detrend.forward(da)
        da = da.fillna(0)

        if self.standardized_initially_fields:
            da = unstack_state(da)
            for field in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} before individual EOF")
                da[field] /= self.standard_deviations[field]
            da = stack_state(da)

        ds_eof_individual = {}

        for field in ds.keys():
            logger.info(f"Fitting EOF for {field}")

            da_field = da.sel(field=field)
            da_field *= np.sqrt(np.cos(np.radians(da_field.lat)))

            ds_eof_individual[field] = self.eofs_individual[
                str(field)
            ].project_forwards(da_field)

        ds_eof_normalized = xr.Dataset(ds_eof_individual)
        for field in ds_eof_individual.keys():
            if field not in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} after individual EOF")
                ds_eof_normalized[field] /= self.standard_deviations[field]

        not_direct_fields = field_complement(ds_eof_normalized, self.direct_fields)
        ds_eof_normalized_for_second_eof = ds_eof_normalized[not_direct_fields]

        logger.info(f"Fitting joint EOF for {', '.join(not_direct_fields)}")
        da_eof_joint = self.eof_joint.project_forwards(
            stack_state(ds_eof_normalized_for_second_eof)
        )
        da_eof_joint = da_eof_joint.expand_dims("field").assign_coords(field=["joint"])
        da_eof_joint = da_eof_joint.stack(dict(state=["field", "state_eof"])).T

        da_eof_joint_and_direct = xr.concat(
            [da_eof_joint, stack_state(ds_eof_normalized[self.direct_fields])],
            dim="state",
        )
        return da_eof_joint_and_direct

    def backward(self, ds: xr.Dataset) -> xr.DataArray:
        # TODO next
        pass
