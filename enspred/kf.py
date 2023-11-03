import numpy as np
import xarray as xr


def stack_state(ds):
    return ds.to_array("field").stack(elem=("lat", "lon", "field")).transpose()

def unstack_state(ds):
    return ds.unstack("elem").to_dataset("field")

def inverse(X: xr.DataArray):
    return xr.DataArray(np.linalg.inv(X.values), coords=X.coords)