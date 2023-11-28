from enum import Enum, auto
from typing import Optional

import dask
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from project.logger import get_logger
from project.util import is_dask_array

logger = get_logger(__name__)


class EOFMethod(Enum):
    DASK = auto()
    NUMPY = auto()


class EOF:
    def __init__(self, rank):
        self.rank = rank
        self.eof_idx = np.arange(self.rank)
        self.U: Optional[ArrayLike] = None
        self.S: Optional[ArrayLike] = None
        self.na_mask: Optional[ArrayLike] = None
        self.state_coords = None

    def _validate_input_vector(self, da: xr.DataArray):
        assert da.ndim <= 2, "Stack state vector before applying EOF"
        assert da.dims[0] == "state"
        if da.ndim == 2:
            assert da.dims[1] == "time"

    def _get_numpy_data(self, da: xr.DataArray) -> ArrayLike:
        if is_dask_array(da):
            return da.data.compute()
        else:
            return da.values

    def fit(self, da: xr.DataArray, method=EOFMethod.DASK):
        self._validate_input_vector(da)

        self.state_coords = da.state

        self.na_mask = da.isnull().any(dim="time")
        da = da.fillna(0)

        if method == EOFMethod.DASK and is_dask_array(da):
            U, S, V = dask.array.linalg.svd_compressed(da.data, self.rank)
            self.U = U.compute().T
            self.S = S.compute()
        else:
            U, S, V = np.linalg.svd(self._get_numpy_data(da), full_matrices=False)
            self.U = U[:, : self.rank].T
            self.S = U[: self.rank]

    def get_component(self, n):
        comp = np.where(self.na_mask, np.nan, self.U[n, :])
        return xr.DataArray(comp, coords=dict(state=self.state_coords))

    def project_forwards(self, da: xr.DataArray):
        self._validate_input_vector(da)

        projected = self.U @ self._get_numpy_data(da)
        if da.ndim == 2:
            # Multiple vectors (timesteps)
            return xr.DataArray(
                projected, coords=dict(state_eof=self.eof_idx, time=da.time)
            )
        else:
            # Single vector
            return xr.DataArray(projected, coords=dict(state_eof=self.eof_idx))

    def project_backwards(self, da: xr.DataArray):
        self._validate_input_vector(da)

        projected = self.U.T @ self._get_numpy_data(da)
        if da.ndim == 2:
            # Multiple vectors (timesteps)
            return xr.DataArray(
                projected, coords=dict(state=self.state_coords, time=da.time)
            )
        else:
            # Single vector
            return xr.DataArray(projected, coords=dict(eof=self.state_coords))
