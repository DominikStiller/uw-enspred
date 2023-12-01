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
from project.util import (
    stack_state,
    get_timestamp,
    list_complement,
)

logger = get_logger(__name__)


class Detrend:
    def __init__(self):
        self.time_mean: Optional[NDArray] = None
        self.data_mean: Optional[dask.array.Array] = None
        self.coeffs: Optional[dask.array.Array] = None

    def _get_time(self, time: dask.array.Array) -> dask.array.Array:
        if isinstance(time[0], cftime.datetime):
            time = cftime.date2num(
                time,
                "days since 1970-01-01",
            )
        return time

    def fit(self, data: dask.array.Array, time: dask.array.Array):
        time = self._get_time(time)

        self.time_mean = time.mean()
        self.data_mean = data.mean(axis=0).persist()[np.newaxis, :]

        time_demeaned: dask.array.Array = np.atleast_2d(time - self.time_mean).T
        state_demeaned: dask.array.Array = (data - self.data_mean).T

        coeffs, _, _, _ = dask.array.linalg.lstsq(
            dask.array.from_array(time_demeaned), state_demeaned
        )
        self.coeffs = coeffs.persist().squeeze()

    def _linear_trend(self, time: dask.array.Array) -> dask.array.Array:
        time = self._get_time(time)
        time_demeaned = time - self.time_mean

        return self.data_mean + self.coeffs[:, np.newaxis] @ time_demeaned[np.newaxis, :]

    def forward(self, data: dask.array.Array, time: dask.array.Array) -> dask.array.Array:
        # De-trend
        return data - self._linear_trend(time)

    def backward(self, days: dask.array.Array, time: dask.array.Array) -> dask.array.Array:
        # Add trend
        return days + self._linear_trend(time)


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

        self.fields: list[str] = None
        self.nan_masks: dict[str, NanMask] = {}
        self.detrends: dict[str, Detrend] = {}
        self.eofs_individual: dict[str, EOF] = {}
        self.standard_deviations: dict[str, float] = {}
        self.eof_joint: EOF = EOF(self.l)
        self.lats: dict[str, dask.array.Array] = {}
        self.state_coords: dict[str, xr.DataArray] = {}
        self.not_direct_fields: list[str] = None

    def save(self, directory: Path):
        directory.mkdir(parents=True)
        outfile = directory / f"mapper-{get_timestamp()}.pkl"
        pickle.dump(self, outfile.open("wb"))
        logger.info(f"Saved mapper to {outfile}")

    @classmethod
    def load(cls, file: Path):
        return pickle.load(file.open("rb"))

    def fit(self, ds: xr.Dataset):
        logger.info("PhysicalSpaceForecastSpaceMapper.fit()")
        self._fit(ds, also_forward=False)

    def fit_and_forward(self, ds: xr.Dataset) -> dask.array.Array:
        logger.info("PhysicalSpaceForecastSpaceMapper.fit_and_forward()")
        return self._fit(ds, also_forward=True)

    def _fit(self, ds: xr.Dataset, also_forward):
        self.fields = list(map(str, ds.keys()))
        self.not_direct_fields = list_complement(self.fields, self.direct_fields)

        logger.info("Splitting dataset into Dask arrays")
        data_raw: dict[str, dask.array.Array] = {}
        for field in self.fields:
            stacked_state = stack_state(ds[field])
            data_raw[field] = stacked_state.data
            self.state_coords[field] = stacked_state.state

        logger.info("Masking nans")
        data_nonan: dict[str, dask.array.Array] = {}
        for field in self.fields:
            self.nan_masks[field] = NanMask()
            self.nan_masks[field].fit(data_raw[field])
            data_nonan[field] = self.nan_masks[field].forward(data_raw[field])
            self.lats[field] = self.nan_masks[field].forward(self.state_coords[field].lat.data)[
                :, np.newaxis
            ]

        logger.info("Calculating field variances")
        self.standard_deviations = {
            field: data_nonan[field].std().compute() for field in self.fields
        }

        logger.info("Detrending data")
        data_detrended: dict[str, dask.array.Array] = {}
        for field in self.fields:
            self.detrends[field] = Detrend()
            self.detrends[field].fit(data_nonan[field], ds.time.data)
            data_detrended[field] = self.detrends[field].forward(data_nonan[field], ds.time.data)

            if field in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} before individual EOF")
                data_detrended[field] /= self.standard_deviations[field]

        data_eof_individual: dict[str, dask.array.Array] = {}

        for field in self.fields:
            data_field = data_detrended[field]
            data_field *= np.sqrt(np.cos(np.radians(self.lats[field])))

            self.eofs_individual[field] = EOF(self.k)

            logger.info(f"Fitting EOF for {field}")
            self.eofs_individual[field].fit(data_field)
            logger.info(f"Projecting EOF for {field}")
            data_eof_individual[field] = self.eofs_individual[field].project_forwards(data_field)

            if field not in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} after individual EOF")
                data_eof_individual[field] /= self.standard_deviations[field]

        data_stacked_for_joint_eof = dask.array.vstack(
            [data_eof_individual[field] for field in self.not_direct_fields]
        )

        logger.info(f"Fitting joint EOF for {', '.join(self.not_direct_fields)}")
        self.eof_joint.fit(data_stacked_for_joint_eof)

        if also_forward:
            logger.info(f"Projecting joint EOF for {', '.join(self.not_direct_fields)}")
            data_eof_joint = self.eof_joint.project_forwards(data_stacked_for_joint_eof)

            logger.info(f"Appending direct fields for {', '.join(self.direct_fields)}")
            data_eof_joint_and_direct = dask.array.vstack(
                [data_eof_joint] + [data_eof_individual[field] for field in self.direct_fields]
            )
            return data_eof_joint_and_direct

    def forward(self, ds: xr.Dataset) -> dask.array.Array:
        logger.info("PhysicalSpaceForecastSpaceMapper.forward()")

        logger.info("Splitting dataset into Dask arrays")
        data_raw: dict[str, dask.array.Array] = {}
        for field in self.fields:
            stacked_state = stack_state(ds[field])
            data_raw[field] = stacked_state.data

        logger.info("Masking nans")
        data_nonan: dict[str, dask.array.Array] = {}
        for field in self.fields:
            data_nonan[field] = self.nan_masks[field].forward(data_raw[field])

        logger.info("Detrending data")
        data_detrended: dict[str, dask.array.Array] = {}
        for field in self.fields:
            data_detrended[field] = self.detrends[field].forward(data_nonan[field], ds.time.data)

            if field in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} before individual EOF")
                data_detrended[field] /= self.standard_deviations[field]

        data_eof_individual: dict[str, dask.array.Array] = {}

        for field in self.fields:
            data_field = data_detrended[field]
            data_field *= np.sqrt(np.cos(np.radians(self.lats[field])))

            logger.info(f"Projecting EOF for {field}")
            data_eof_individual[field] = self.eofs_individual[field].project_forwards(data_field)

            if field not in self.standardized_initially_fields:
                logger.info(f"Standardizing {field} after individual EOF")
                data_eof_individual[field] /= self.standard_deviations[field]

        logger.info(f"Projecting joint EOF for {', '.join(self.not_direct_fields)}")
        data_stacked_for_joint_eof = dask.array.vstack(
            [data_eof_individual[field] for field in self.not_direct_fields]
        )
        data_eof_joint = self.eof_joint.project_forwards(data_stacked_for_joint_eof)

        logger.info(f"Appending direct fields for {', '.join(self.direct_fields)}")
        data_eof_joint_and_direct = dask.array.vstack(
            [data_eof_joint] + [data_eof_individual[field] for field in self.direct_fields]
        )
        return data_eof_joint_and_direct

    def backward(self, data: dask.array.Array, time: xr.DataArray) -> xr.Dataset:
        logger.info("PhysicalSpaceForecastSpaceMapper.backward()")

        data_eof_individual: dict[str, dask.array.Array] = {}

        logger.info(f"Splitting direct fields for {', '.join(self.direct_fields)}")
        start_row = self.eof_joint.rank
        for field in self.direct_fields:
            length = self.eofs_individual[field].rank
            data_eof_individual[field] = data[start_row : start_row + length]
            start_row += length

        logger.info(f"Back-projecting joint EOF for {', '.join(self.not_direct_fields)}")
        data_eof_joint = data[: self.eof_joint.rank]
        data_stacked_for_joint_eof = self.eof_joint.project_backwards(data_eof_joint)

        start_row = 0
        for field in self.not_direct_fields:
            length = self.eofs_individual[field].rank
            data_eof_individual[field] = data_stacked_for_joint_eof[start_row : start_row + length]
            start_row += length

        data_detrended: dict[str, dask.array.Array] = {}

        for field in self.fields:
            data_field = data_eof_individual[field]

            if field not in self.standardized_initially_fields:
                logger.info(f"De-standardizing {field} after individual EOF")
                data_field *= self.standard_deviations[field]

            logger.info(f"Back-projecting EOF for {field}")
            data_detrended[field] = self.eofs_individual[field].project_backwards(data_field)

            data_detrended[field] /= np.sqrt(np.cos(np.radians(self.lats[field])))

        logger.info("Re-trending data")
        data_nonan: dict[str, dask.array.Array] = {}
        for field in self.fields:
            if field in self.standardized_initially_fields:
                logger.info(f"De-standardizing {field} before individual EOF")
                data_detrended[field] *= self.standard_deviations[field]
            data_nonan[field] = self.detrends[field].backward(data_detrended[field], time.data)

        logger.info("Un-masking nans")
        data_raw: dict[str, dask.array.Array] = {}
        for field in self.fields:
            data_raw[field] = self.nan_masks[field].backward(data_nonan[field])

        logger.info("Merging Dask arrays into dataset")
        data_xarray: dict[str, xr.DataArray] = {}
        for field in self.fields:
            data_xarray[field] = (
                xr.DataArray(
                    data_raw[field], coords=dict(state=self.state_coords[field], time=time)
                )
                .unstack("state")
                .squeeze()
                .drop_vars("field")
            )

        ds = xr.Dataset(data_xarray)

        return ds
