import numpy as np
import xarray as xr
from xarray import Dataset, DataArray
from typing import Optional


# def stack_state(ds: Dataset) -> DataArray:
#     da = ds.to_array("field")
#     dims_to_stack = set(da.dims)
#     dims_to_stack.remove("time")
#     return da.stack(state=dims_to_stack).transpose()


# def unstack_state(da: DataArray) -> Dataset:
#     return da.unstack("state").to_dataset("field")


def stack_state(ds: Dataset) -> DataArray:
    return ds.to_stacked_array(
        "state", sample_dims=["time"], variable_dim="field", name=""
    ).transpose()


def unstack_state(da: DataArray) -> Dataset:
    return data.to_unstacked_dataset("state")


def inverse(da: DataArray):
    return DataArray(np.linalg.inv(np.atleast_2d(da.values)), coords=da.coords)


def matrix_power(da: DataArray, n: int):
    return DataArray(np.linalg.matrix_power(np.atleast_2d(da.values), n), coords=da.coords)


def estimate_cov(
    da: DataArray, da2: Optional[DataArray] = None, sample_dim: str = "time"
):
    assert da.ndim == 2
    if da2 is not None:
        assert da2.ndim == 2
        assert da.dims == da2.dims
        assert da.shape == da2.shape
    else:
        da2 = da

    data_dims = list(da.coords)
    data_dims.remove(sample_dim)
    C = xr.cov(
        da.rename({data_dim: f"{data_dim}1" for data_dim in data_dims}),
        da2.rename({data_dim: f"{data_dim}2" for data_dim in data_dims}),
        dim=sample_dim,
    )

    return C
