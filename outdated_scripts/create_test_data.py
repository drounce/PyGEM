#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:13:38 2022

@author: drounce
"""

import numpy as np
import pandas as pd
import xarray as xr

import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup

#era5_fp = main_directory + '/../climate_data/ERA5/'
#era5_temp_fn = 'ERA5_temp_monthly.nc'
#era5_tempstd_fn = 'ERA5_tempstd_monthly.nc'
#era5_prec_fn = 'ERA5_totalprecip_monthly.nc'
#era5_elev_fn = 'ERA5_geopotential.nc'
#era5_pressureleveltemp_fn = 'ERA5_pressureleveltemp_monthly.nc'
#era5_lr_fn = 'ERA5_lapserates_monthly.nc'

era5_fp_subset = '/Users/drounce/Documents/HiMAT-PyGEM-test/climate_data/ERA5/'

def export_era5_subset(main_glac_rgi, data):
    """Output nearest lat/lon indices"""
    lat_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lat_colname].values[:,np.newaxis] - 
                          data.variables['latitude'][:].values).argmin(axis=1))
    lon_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lon_colname].values[:,np.newaxis] - 
                          data.variables['longitude'][:].values).argmin(axis=1))
    # Find unique latitude/longitudes
    latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
    
    data_subset = data.isel(latitude=[latlon_nearidx[0][0],latlon_nearidx[0][0]+1], 
                            longitude=[latlon_nearidx[0][1], latlon_nearidx[0][1]+1])
    
    return data_subset


# ===== LOAD GLACIERS =====
main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=pygem_prms.glac_no)


# ===== CLIMATE DATA =====
# Air temperature
era5_temp_fullfn = pygem_prms.era5_fp + pygem_prms.era5_temp_fn
ds_temp = xr.open_dataset(era5_temp_fullfn)
ds_temp_subset = export_era5_subset(main_glac_rgi, ds_temp)
ds_temp_subset.to_netcdf(era5_fp_subset + pygem_prms.era5_temp_fn)

# Air temperature daily stdev
era5_tempstd_fullfn = pygem_prms.era5_fp + pygem_prms.era5_tempstd_fn
ds_tempstd = xr.open_dataset(era5_tempstd_fullfn)
ds_tempstd_subset = export_era5_subset(main_glac_rgi, ds_tempstd)
ds_tempstd_subset.to_netcdf(era5_fp_subset + pygem_prms.era5_tempstd_fn)

# Precipitation
era5_prec_fullfn = pygem_prms.era5_fp + pygem_prms.era5_prec_fn
ds_prec = xr.open_dataset(era5_prec_fullfn)
ds_prec_subset = export_era5_subset(main_glac_rgi, ds_prec)
ds_prec_subset.to_netcdf(era5_fp_subset + pygem_prms.era5_prec_fn)

# Lapse rate
era5_lr_fullfn = pygem_prms.era5_fp + pygem_prms.era5_lr_fn
ds_lr = xr.open_dataset(era5_lr_fullfn)
ds_lr_subset = export_era5_subset(main_glac_rgi, ds_lr)
ds_lr_subset.to_netcdf(era5_fp_subset + pygem_prms.era5_lr_fn)

# Orography
era5_elev_fullfn = pygem_prms.era5_fp + pygem_prms.era5_elev_fn
ds_elev = xr.open_dataset(era5_elev_fullfn)
ds_elev_subset = export_era5_subset(main_glac_rgi, ds_elev)
ds_elev_subset.to_netcdf(era5_fp_subset + pygem_prms.era5_elev_fn)
