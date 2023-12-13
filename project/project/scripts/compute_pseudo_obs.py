import dask
import numpy as np
import xarray as xr
from dask.distributed import Client
from pylipd import LiPD

from project.io import IntakeESMLoader, save_mfdataset
from project.logger import get_logger
from project.util import get_data_path, average_annually

logger = get_logger(__name__)

if __name__ == "__main__":
    sigma_obs = 1

    client = Client(n_workers=dask.system.CPU_COUNT // 2, threads_per_worker=1)

    data_path = get_data_path()

    loader = IntakeESMLoader(
        "past1000",
        "MRI-ESM2-0",
        ["tas"],
    )
    ds = loader.load_dataset()

    logger.info("Averaging annually")
    ds = average_annually(ds).chunk(chunks=dict(time=100))

    logger.info("Loading Pages2k")
    pages2k = LiPD()
    pages2k.load_from_dir("/home/enkf6/dstiller/obs/Pages2kTemperature2_1_2")

    locs = (
        pages2k.get_all_locations()
        .sample(frac=1, random_state=np.random.default_rng(46234))
        .rename(columns={"geo_meanLat": "lat", "geo_meanLon": "lon"})
        .reset_index(drop=True)
    )
    # Make negative lons positive
    locs["lon"].loc[locs["lon"] < 0] = 360 + locs["lon"].loc[locs["lon"] < 0]

    obs = xr.combine_by_coords(
        [
            ds.sel(lat=[locs.lat.iloc[i]], lon=[locs.lon.iloc[i]], method="nearest").stack(
                dict(location=["lon", "lat"])
            )
            for i in range(len(locs.index))
        ]
    )
    obs["tas"] += np.random.default_rng(52565641).normal(0, sigma_obs, obs["tas"].shape)

    save_mfdataset(obs.reset_index("location"), data_path / "obs")
    logger.info("Computing of pseudo-observations completed")
