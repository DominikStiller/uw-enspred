from pathlib import Path

from project.io import IntakeESMLoader, save_mfdataset
from project.spaces import PhysicalSpaceForecastSpaceMapper

if __name__ == "__main__":
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
    ds = loader.load_dataset()#.isel(time=slice(None, 20))

    mapper = PhysicalSpaceForecastSpaceMapper(20, 400, ["ohc700"], ["pr"])
    ds_eof = mapper.fit_and_forward(ds)
    mapper.save(Path("/home/enkf6/dstiller/enspred/mapper"))
    save_mfdataset(
        ds_eof.reset_index("state").to_dataset(name="data"),
        Path("/home/enkf6/dstiller/enspred/training_data"),
    )
