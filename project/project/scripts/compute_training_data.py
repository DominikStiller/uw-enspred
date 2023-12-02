import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from project.io import IntakeESMLoader, save_mfdataset
from project.spaces import PhysicalSpaceForecastSpaceMapper
from project.util import get_data_path

if __name__ == "__main__":
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

    mapper = PhysicalSpaceForecastSpaceMapper(400, 30, ["ohc700"], ["pr"])
    array_eof = mapper.fit_and_forward(ds)
    ds_eof = xr.DataArray(array_eof, coords=dict(state=np.arange(array_eof.shape[0]), time=ds.time))
    mapper.save(data_path / "mapper")
    with ProgressBar():
        save_mfdataset(ds_eof.to_dataset(name="data"), data_path / "training_data")
