import pickle
from pathlib import Path
from typing import Optional

import cftime
import dask
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from project.eof import EOF
from project.logger import get_logger
from project.util import stack_state, unstack_state, field_complement, get_timestamp

logger = get_logger(__name__)


class Detrend:
    def __init__(self):
        self.time_mean: Optional[NDArray] = None
        self.state_mean: Optional[dask.array.Array] = None
        self.coeffs: Optional[dask.array.Array] = None

    def _get_time(self, da: xr.DataArray) -> NDArray:
        time = da.time.data
        if isinstance(time[0], cftime.datetime):
            time = cftime.date2num(
                time,
                "days since 1970-01-01",
            )
        return time

    def fit(self, da: xr.DataArray):
        time = self._get_time(da)

        self.time_mean = time.mean()
        self.state_mean = da.mean(dim="time").data.persist()[:, np.newaxis]

        time_demeaned: NDArray = np.atleast_2d(time - self.time_mean).T
        state_demeaned: dask.array.Array = (da.data - self.state_mean).T

        coeffs, _, _, _ = dask.array.linalg.lstsq(
            dask.array.from_array(time_demeaned), state_demeaned
        )
        self.coeffs = coeffs.persist().squeeze()

    def _linear_trend(self, da: xr.DataArray) -> dask.array.Array:
        time = self._get_time(da)
        time_demeaned = time - self.time_mean

        return self.state_mean + self.coeffs[:, np.newaxis] @ time_demeaned[np.newaxis, :]

    def forward(self, da: xr.DataArray) -> xr.DataArray:
        # De-trend
        return da - self._linear_trend(da)

    def backward(self, da: xr.DataArray) -> xr.DataArray:
        # Add trend
        return da + self._linear_trend(da)


class NanMask:
    def __init__(self):
        self.nan_mask: Optional[dask.array.Array] = None

    def fit(self, da: dask.array.Array):
        self.nan_mask = np.isnan(da).all(axis=1).compute()

    def forward(self, da: dask.array.Array) -> dask.array.Array:
        return da[~self.nan_mask]

    def backward(self, da: dask.array.Array) -> dask.array.Array:
        decompressed = dask.array.empty((len(self.nan_mask), da.shape[1]))
        decompressed[~self.nan_mask] = da
        decompressed[self.nan_mask] = np.nan
        return decompressed


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

        self.original_field_order: Optional[list[str]] = None
        self.nan_mask = NanMask()
        self.detrend: Detrend = Detrend()
        self.eofs_individual: dict[str, EOF] = {}
        self.standard_deviations: Optional[xr.Dataset] = None
        self.eof_joint: EOF = EOF(self.l)

    def save(self, directory: Path):
        outfile = directory / f"mapper-{get_timestamp()}.pkl"
        pickle.dump(self, outfile.open("wb"))
        logger.info(f"Saved mapper to {outfile}")

    @classmethod
    def load(cls, file: Path):
        return pickle.load(file.open("rb"))

    def fit(self, ds: xr.Dataset):
        logger.info("PhysicalSpaceForecastSpaceMapper.fit()")
        self._fit(ds, also_forward=False)

    def fit_and_forward(self, ds: xr.Dataset):
        logger.info("PhysicalSpaceForecastSpaceMapper.fit_and_forward()")
        return self._fit(ds, also_forward=True)

    def _fit(self, ds: xr.Dataset, also_forward):
        logger.info("Calculating field variances")

        self.original_field_order = list(ds.keys())
        self.standard_deviations = ds.std().compute()

        da = stack_state(ds)

        self.nan_mask.fit(da)
        da = self.nan_mask.forward(da)

        logger.info("Detrending data")
        self.detrend.fit(da)
        da = self.detrend.forward(da)

        if self.standardized_initially_fields:
            for field in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} before individual EOF")
                # In-place assignment is very slow, not sure what to do about it
                da.loc[da.field == field] /= self.standard_deviations[field].item()

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
                ds_eof_normalized[field] /= self.standard_deviations[field].item()

        not_direct_fields = field_complement(ds_eof_normalized, self.direct_fields)
        ds_eof_normalized_for_second_eof = ds_eof_normalized[not_direct_fields]

        logger.info(f"Fitting joint EOF for {', '.join(not_direct_fields)}")
        self.eof_joint.fit(stack_state(ds_eof_normalized_for_second_eof))

        if also_forward:
            logger.info(f"Projecting joint EOF for {', '.join(not_direct_fields)}")
            da_eof_joint = self.eof_joint.project_forwards(
                stack_state(ds_eof_normalized_for_second_eof)
            )
            da_eof_joint = da_eof_joint.expand_dims("field").assign_coords(field=["joint"])
            da_eof_joint = da_eof_joint.stack(dict(state=["field", "state_eof"])).T

            logger.info(f"Appending direct fields for {', '.join(self.direct_fields)}")
            da_eof_joint_and_direct = xr.concat(
                [da_eof_joint, stack_state(ds_eof_normalized[self.direct_fields])],
                dim="state",
            )
            return da_eof_joint_and_direct

    def forward(self, ds: xr.Dataset) -> xr.DataArray:
        logger.info("PhysicalSpaceForecastSpaceMapper.forward()")

        da = stack_state(ds)
        da = self.nan_mask.forward(da)
        da = self.detrend.forward(da)

        if self.standardized_initially_fields:
            for field in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} before individual EOF")
                da.loc[da.field == field] /= self.standard_deviations[field].item()

        ds_eof_individual = {}

        for field in ds.keys():
            logger.info(f"Projecting EOF for {field}")

            da_field = da.sel(field=field)
            da_field *= np.sqrt(np.cos(np.radians(da_field.lat)))

            ds_eof_individual[field] = self.eofs_individual[str(field)].project_forwards(da_field)

        ds_eof_normalized = xr.Dataset(ds_eof_individual)
        for field in ds_eof_normalized.keys():
            if field not in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} after individual EOF")
                ds_eof_normalized[field] /= self.standard_deviations[field].item()

        not_direct_fields = field_complement(ds_eof_normalized, self.direct_fields)
        ds_eof_normalized_for_second_eof = ds_eof_normalized[not_direct_fields]

        logger.info(f"Projecting joint EOF for {', '.join(not_direct_fields)}")
        da_eof_joint = self.eof_joint.project_forwards(
            stack_state(ds_eof_normalized_for_second_eof)
        )
        da_eof_joint = da_eof_joint.expand_dims("field").assign_coords(field=["joint"])
        da_eof_joint = da_eof_joint.stack(dict(state=["field", "state_eof"])).T

        logger.info(f"Appending direct fields for {', '.join(self.direct_fields)}")
        da_eof_joint_and_direct = xr.concat(
            [da_eof_joint, stack_state(ds_eof_normalized[self.direct_fields])],
            dim="state",
        )
        return da_eof_joint_and_direct

    def backward(self, da: xr.DataArray) -> xr.Dataset:
        logger.info("PhysicalSpaceForecastSpaceMapper.backward()")

        da_eof_joint = da.sel(field="joint")
        ds_eof_direct = unstack_state(da)[self.direct_fields].isel(state_eof=slice(None, self.k))

        ds_eof_joint_backprojected = unstack_state(self.eof_joint.project_backwards(da_eof_joint))

        ds_eof_normalized = xr.merge([ds_eof_joint_backprojected, ds_eof_direct])
        for field in ds_eof_normalized.keys():
            if field not in self.standardized_initially_fields:
                logger.info(f"De-standardizing {field} after individual EOF")
                ds_eof_normalized[field] *= self.standard_deviations[field].item()

        da_physical_individual = {}

        for field in ds_eof_normalized.keys():
            logger.info(f"Back-projecting EOF for {field}")

            da_field = ds_eof_normalized[field]

            da_field_physical = self.eofs_individual[str(field)].project_backwards(da_field)

            da_field_physical /= np.sqrt(np.cos(np.radians(da_field_physical.lat)))
            da_field_physical = da_field_physical.expand_dims("field").assign_coords(field=[field])
            da_field_physical = da_field_physical.unstack("state").stack(
                state=["field", "lat", "lon"]
            )
            da_physical_individual[field] = da_field_physical

        # Ensure that we stack fields in the order that is expected by NanMask
        da_physical_individual = xr.concat(
            [da_physical_individual[field] for field in self.original_field_order],
            dim="state",
        ).T

        if self.standardized_initially_fields:
            for field in self.standardized_initially_fields:
                logger.info(f"De-standardizing {field} before individual EOF")
                da_physical_individual.loc[
                    da_physical_individual.field == field
                ] *= self.standard_deviations[field].item()

        da = self.detrend.backward(da_physical_individual)
        da = self.nan_mask.backward(da)

        ds = unstack_state(da)

        return ds
