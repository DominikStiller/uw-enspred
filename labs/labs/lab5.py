import xarray as xr
from pathlib import Path
from tqdm import tqdm
import pandas as pd


archive_path = Path("/glade/scratch/hakim/data/dl_enkf/archive")
verification_path = Path("/glade/scratch/tvonich/earth2mip/hwk_data/lab5data/dl_enkf/verification")
priors_path = Path("/glade/work/dstiller/enspred/lab5") / "priors.nc"
analyses_path = Path("/glade/work/dstiller/enspred/lab5") / "analyses.nc"


def load_results(type):
    results = []

    for step_folder in tqdm(sorted(archive_path.glob("*/"))):
        for member in range(100):
            ds_file = step_folder / type / f"member_{member}.h5"
            if not ds_file.exists():
                continue

            ds = xr.open_dataset(ds_file)
            ds = ds.sel(lat=50, lon=[188, 198, 208], channel_pl=["z", "v"], level=500)
            ds = xr.Dataset({
                "z500": (["lon"], ds.sel(channel_pl="z")["pl_data"].data / 9.81),
                "v500": (["lon"], ds.sel(channel_pl="v")["pl_data"].data)
            }, coords={
                "lon": ds.lon,
                "member": [member],
                "time": [pd.to_datetime(step_folder.name, format="%Y%m%d%H")]
            })
            
            results.append(ds)

    results = xr.combine_by_coords(results)
    return results


def load_full_field(type, step, n_members=100):
    results = []

    step_folder = list(sorted(archive_path.glob("*/")))[step]
    print(step_folder)
    for member in range(n_members):
        ds_file = step_folder / type / f"member_{member}.h5"
        if not ds_file.exists():
            continue

        ds = xr.open_dataset(ds_file)
        ds = ds.sel(channel_pl=["z", "v"], level=500)
        ds = xr.Dataset({
            "z500": (["lat", "lon"], ds.sel(channel_pl="z")["pl_data"].data / 9.81),
            "v500": (["lat", "lon"], ds.sel(channel_pl="v")["pl_data"].data)
        }, coords={
            "lat": ds.lat,
            "lon": ds.lon,
            "member": [member],
            "time": [pd.to_datetime(step_folder.name, format="%Y%m%d%H")]
        })
        
        results.append(ds)

    results = xr.combine_by_coords(results)
    return results


def load_verifications():
    verifications = xr.combine_by_coords([
        xr.open_dataset(verification_path / "188_verification.nc"),
        xr.open_dataset(verification_path / "198_verification.nc"),
        xr.open_dataset(verification_path / "208_verification.nc"),
    ])
    verifications = verifications.assign_coords({"lon": verifications.lon + 180}).squeeze()
    verifications["z500"] /= 9.81
    return verifications