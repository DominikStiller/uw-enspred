from pathlib import Path

import xarray as xr
import numpy as np


obs_lat, obs_lon = 40, 200
sigma_obs = np.sqrt(10)


def load_dataset(year):
    path = list(Path("/glade/work/chriss/ATMS544/lab4data").glob(f"ens_*{year}.nc"))[0]
    ds = xr.open_dataset(path)
    return ds


def get_exp_no(results):
    Ne = results[0]
    if Ne == 99:
        return 1
    elif Ne == 25:
        return 2
    else:
        raise "Invalid Ne"


def make_H(idx):
    H = xr.DataArray(np.zeros((1, len(idx))), coords={"ob": [0], "elem": idx})
    H.loc[dict(lat=obs_lat, lon=obs_lon, field="z500")] = 1
    return H
