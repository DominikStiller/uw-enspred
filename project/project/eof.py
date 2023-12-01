from enum import Enum, auto
from typing import Optional

import dask
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from project.logger import get_logger

logger = get_logger(__name__)


class EOFMethod(Enum):
    DASK = auto()
    NUMPY = auto()


class EOF:
    def __init__(self, rank):
        self.rank = rank
        self.eof_idx: Optional[NDArray] = None
        self.U: Optional[NDArray] = None
        self.S: Optional[NDArray] = None

    def _validate_input_vector(self, data: dask.array.Array):
        assert data.ndim <= 2, "Stack state vector before applying EOF"
        assert not np.isnan(data).any(), "nan is not allowed in EOF input data"

    def fit(self, data: dask.array.Array, method=EOFMethod.DASK):
        self._validate_input_vector(data)

        input_rank = min(data.shape)
        if input_rank < self.rank:
            logger.warn(
                f"Insufficient data for EOF with rank {self.rank}, output rank will be {input_rank}"
            )
            self.rank = input_rank
        self.eof_idx = np.arange(self.rank)

        if method == EOFMethod.DASK:
            logger.debug(f"Calculating EOFs using Dask (rank = {self.rank})")
            U, S, V = dask.array.linalg.svd_compressed(data, self.rank)
            self.U = U.T.persist()
            self.S = S.persist()
        else:
            logger.debug(f"Calculating EOFs using NumPy (rank = {self.rank})")
            U, S, V = np.linalg.svd(data.compute(), full_matrices=False)
            self.U = U[:, : self.rank].T
            self.S = U[: self.rank]

    def get_component(self, n):
        return self.U[n, :]

    def project_forwards(self, data: dask.array.Array) -> dask.array.Array:
        self._validate_input_vector(data)
        return self.U @ data

    def project_backwards(self, data: dask.array.Array) -> dask.array.Array:
        self._validate_input_vector(data)
        return self.U.T @ data


class EOFXArray(EOF):
    def __init__(self, rank):
        super().__init__(rank)
        self.state_coords = None

    def fit(self, da: xr.DataArray, method=EOFMethod.DASK):
        self.state_coords = da.state
        super().fit(da.data, method)

    def get_component(self, n):
        return xr.DataArray(super().get_component(n), coords=dict(state=self.state_coords))

    def project_forwards(self, da: xr.DataArray):
        projected = super().project_forwards(da.data)
        if da.ndim == 2:
            # Multiple vectors (timesteps)
            return xr.DataArray(projected, coords=dict(state_eof=self.eof_idx, time=da.time))
        else:
            # Single vector
            return xr.DataArray(projected, coords=dict(state_eof=self.eof_idx))

    def project_backwards(self, da: xr.DataArray):
        projected = super().project_backwards(da.data)
        if da.ndim == 2:
            # Multiple vectors (timesteps)
            return xr.DataArray(projected, coords=dict(state=self.state_coords, time=da.time))
        else:
            # Single vector
            return xr.DataArray(projected, coords=dict(state=self.state_coords))
