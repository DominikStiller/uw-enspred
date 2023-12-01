from datetime import datetime
from typing import Union

import numpy as np
import xarray as xr
from xarray import Dataset, DataArray


# def stack_state(ds: Dataset) -> DataArray:
#     da = ds.to_array("field")
#     dims_to_stack = set(da.dims)
#     dims_to_stack.remove("time")
#     return da.stack(state=dims_to_stack).transpose()


# def unstack_state(da: DataArray) -> Dataset:
#     return da.unstack("state").to_dataset("field")


def stack_state(ds: Union[Dataset, DataArray]) -> DataArray:
    if isinstance(ds, DataArray):
        ds = ds.to_dataset()
    return ds.to_stacked_array(
        "state", sample_dims=["time"], variable_dim="field", name=""
    ).transpose()


def unstack_state(da: DataArray) -> Dataset:
    ds = da.to_unstacked_dataset("state")
    if "state" in ds.dims:
        ds = ds.unstack("state")
    return ds


def inverse(da: DataArray):
    return DataArray(np.linalg.inv(np.atleast_2d(da.values)), coords=da.coords)


def matrix_power(da: DataArray, n: int):
    return DataArray(np.linalg.matrix_power(np.atleast_2d(da.values), n), coords=da.coords)


def is_dask_array(arr):
    return hasattr(arr.data, "dask")


def field_complement(ds: Union[xr.DataArray, xr.Dataset], other_fields: list[str]) -> list[str]:
    """
    Returns all field names that are in the dataset but not in other_fields

    Args:
        ds: the Dataset/DataArray with all fields
        other_fields: the fields to exclude

    Returns:
        the field name complement
    """
    if isinstance(ds, xr.DataArray):
        fields = np.unique(ds.field)
    else:
        fields = ds.keys()
    return list_complement(fields, other_fields)


def list_complement(elements: list, others: list) -> list:
    return list(set(elements) - set(others))


def get_timestamp():
    return datetime.now().replace(microsecond=0).isoformat().replace(":", "-")
