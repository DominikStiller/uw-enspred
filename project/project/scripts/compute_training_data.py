import os

import numpy as np
import xarray as xr
from distributed import Client, progress

from project.io import IntakeESMLoader, save_mfdataset
from project.logger import get_logger
from project.spaces import PhysicalSpaceForecastSpaceMapper
from project.util import get_data_path

logger = get_logger(__name__)

if __name__ == "__main__":
    client = Client(n_workesr=os.cpu_count() - 2, threads_per_worker=1)

    data_path = get_data_path()

    loader = IntakeESMLoader(
        "past2k",
        "MPI-ESM1-2-LR",
        [
            "zg500",
            "pr",
            "psl",
            "rsut",
            "rlut",
            "tas",
            "tos",
            "zos",
            "sos",
            "ohc700",
        ],
    )
    ds = loader.load_dataset()
    # ds = loader.load_dataset(["702101-704012"]).isel(time=slice(None, 20))

    mapper = PhysicalSpaceForecastSpaceMapper(400, 30, 20, ["ohc700"], ["pr"])
    array_eof = mapper.fit_and_forward(ds)
    ds_eof = xr.DataArray(array_eof, coords=dict(state=np.arange(array_eof.shape[0]), time=ds.time))
    mapper.save(data_path / "mapper")
    # with ProgressBar():
    #     save_mfdataset(ds_eof.to_dataset(name="data"), data_path / "training_data")
    task = save_mfdataset(
        ds_eof.to_dataset(name="data"), data_path / "training_data", compute=False
    )
    progress(task)
    task.compute()
    logger.info("Computing of training data completed")
