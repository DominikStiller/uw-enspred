"""
read RDA on glade, and select one lat,lon point in a field for scalar DA

Uses Zilu Meng's RDA reading code

Originator: G. Hakim
            University of Washington
            19 October 2023
"""

import sys

sys.path.append("/glade/work/hakim/ATMS544/earth2mip/earth2mip/initial_conditions")

from datetime import datetime, timedelta
import xarray as xr

# from rda import DataSource
from earth2mip.initial_conditions.cds import DataSource

# ---------------------------------------------------------------
# set parameters
grav = 9.81
# number of ensemble members
Nens = 100
# starting time for ensemble
year = 2013
month = 10
day = 6
# time increment between ensemble members, in hours
dt = 1

channels = ["z500", "u10"]

# create instance of DataSource object (has read fuction; takes list of channels)
ds = DataSource(channels)

date = datetime(year, month, day)
data = []
for n in range(Nens):
    print(f"n={n+1}/{Nens}, data={date}")

    # read
    data.append(ds[date])

    # increment to next time (ensemble member)
    date = date + timedelta(hours=dt)

    print()

# combine members into single dataset
data = xr.concat(data).to_dataset(dim="channel")

# change time coordinate to member index
data = data.assign_coords({"time": range(Nens)}).rename({"time": "ensemble"})

# Convert geopotential to geopotential height
data["z500"] /= grav

# save to a file
outfile = f"/glade/work/chriss/ATMS544/lab4data/ens_{day:02d}Oct{year}.hdf"
data.to_netcdf(outfile)

print("Saved data to", outfile)
