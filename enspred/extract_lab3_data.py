"""
read RDA on glade, and select one lat,lon point in a field for scalar DA

Uses Zilu Meng's RDA reading code

Originator: G. Hakim
            University of Washington
            19 October 2023
"""

import sys

sys.path.append('/glade/work/hakim/ATMS544/earth2mip/earth2mip/initial_conditions')

import numpy as np
import dataclasses
from datetime import datetime,timedelta
import h5py
import rda

#---------------------------------------------------------------
# set parameters
grav = 9.81
# number of ensemble members
Nens = 100
# starting time for ensemble
year = 2013
month = 10
day = 6
hour = 0
# time increment between ensemble members, in hours
dt = 1
# lat lon of points
lat1=40; lon1 = 200
lat2=20; lon2 = 200

#---------------------------------------------------------------
def read_rda_onetime_onepoint(ds,date):


    #print('got data for:',data['time'][0].dt.strftime("%Y-%m-%d %H:%M"))

    return ds

#--------------

# create instance of DataSource object (has read fuction; takes list of channels)
ds = rda.DataSource(['z500'])

point_one = np.zeros(Nens)
point_two = np.zeros(Nens)

date = datetime(year,month,day,hour)
dates = []
for n in range(Nens):
    print('n=',date)
    dates.append(date)
    
    # read rda
    data = ds[date]

    # one lat,lon
    point_one[n] = data.sel(channel='z500',lat=lat1,lon=lon1).to_numpy()[0]/grav
    point_two[n] = data.sel(channel='z500',lat=lat2,lon=lon2).to_numpy()[0]/grav

    # increment to next time (ensemble member)
    date = date + timedelta(hours=dt)

# Convert datetime objects to ISO formatted strings
dates_string = [dt.isoformat() for dt in dates]
print(dates_string)

# save to a file
outfile = 'z500_'+dates_string[0][:12]+'.h5'

h5f = h5py.File(outfile, 'w')
h5f.create_dataset('point_one',data=point_one)
h5f.create_dataset('point_two',data=point_two)
h5f.create_dataset('dates_string',data=dates_string)
h5f.create_dataset('latlon1',data=[lat1,lon1])
h5f.create_dataset('latlon2',data=[lat2,lon2])
# if needed:
#lat = data['lat'].to_numpy()
#lon = data['lon'].to_numpy()
#h5f.create_dataset('lat',data=lat)
#h5f.create_dataset('lon',data=lon)
h5f.close()

