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
    return da.to_unstacked_dataset("state")


def inverse(da: DataArray):
    return DataArray(np.linalg.inv(np.atleast_2d(da.values)), coords=da.coords)


def matrix_power(da: DataArray, n: int):
    return DataArray(
        np.linalg.matrix_power(np.atleast_2d(da.values), n), coords=da.coords
    )
