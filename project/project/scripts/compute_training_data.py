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
    ds = loader.load_dataset(
        [
            "700101-702012",
            "702101-704012",
        ]
    )

    mapper = PhysicalSpaceForecastSpaceMapper(20, 400, ["ohc700"], ["pr"])
    ds_eof = mapper.fit_and_forward(ds)
    mapper.save(data_path / "mapper")
    save_mfdataset(
        ds_eof.reset_index("state").to_dataset(name="data"),
        data_path / "training_data"
    )
