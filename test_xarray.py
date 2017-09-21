# test to try and use xarray to load and save comptuation time and space
import xarray as xr
import os

gcm_file = os.path.dirname(__file__) + '/../Climate_data/cmip5/rcp85_r1i1p1_monNG/tas_mon_MPI-ESM-LR_rcp85_r1i1p1_native.nc'

ds = xr.open_mfdataset(gcm_file)
# temperature = ds['tas']
#
# print(temperature)
#
# print(temperature[0,0,0])

print(ds['tas'][0,0,0])
print('\n')
print(ds['tas'].loc['1870-01-15'])
print('\n')

print(ds['tas'].isel(time=0, lat=0, lon=0))

print(ds['tas'].sel(time='1870-01-15', lat=-86.72, lon=0.0))