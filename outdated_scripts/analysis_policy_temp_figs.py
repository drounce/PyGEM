""" Analyze simulation output - mass change, runoff, etc. """

# Built-in libraries
import argparse
#from collections import OrderedDict
from collections import Counter
#import datetime
#import glob
import os
import pickle
import shutil
import time
#import zipfile
# External libraries
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
#from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
#import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#from matplotlib.ticker import EngFormatter
#from matplotlib.ticker import StrMethodFormatter
#from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#from mpl_toolkits.basemap import Basemap
import geopandas
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from scipy.stats import linregress
from scipy.ndimage import generic_filter
from scipy.ndimage import uniform_filter
#import scipy
import xarray as xr
# Local libraries
#import class_climate
#import class_mbdata
import pygem.pygem_input as pygem_prms
#import pygemfxns_gcmbiasadj as gcmbiasadj
import pygem.pygem_modelsetup as modelsetup

#from oggm import utils
from pygem.oggm_compat import single_flowline_glacier_directory
from pygem.shop import debris 
from oggm import tasks

time_start = time.time()

#%% ===== Input data =====
# Script options
option_policy_temp_figs = False                  # Policy figures based on temperature deviations 
option_calving_comparison_bydeg = False          # Multi-GCM comparison of including frontal ablation or not
option_glacier_cs_plots_calving_bydeg = False    # Cross sectional plots for calving glaciers based on degrees
option_tidewater_stats = False                   # Tidewater % by area for each region
option_tidewater_fa_err = False                  # Processes frontal ablation error for regional statistics
option_debris_comparison_bydeg = True           # Comparison of including debris or not
option_tidewater_volume_stats = False            # Statistics related to initial mass including and excluding frontal ablation
option_tidewater_landretreat = False             # Statistics related to number of tidewater glaciers that retreat onto land


#regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
regions = [1,3,4,5,7,9,17,19]

#deg_groups = [1.5,2,2.7,3,4]
#deg_groups_bnds = [0.25, 0.5, 0.5, 0.5, 0.5]
#deg_group_colors = ['#4575b4','#74add1', '#fee090', '#fdae61', '#f46d43', '#d73027']
deg_groups = [1.5,2,3,4]
deg_groups_bnds = [0.25, 0.5, 0.5, 0.5]
#deg_groups_bnds = [0.25, 0.25, 0.25, 0.25]
deg_group_colors = ['#4575b4', '#fee090', '#fdae61', '#f46d43', '#d73027']
temp_colordict = {}
for ngroup, deg_group in enumerate(deg_groups):
    temp_colordict[deg_group] = deg_group_colors[ngroup]

gcm_names_rcps = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                  'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR', 'NorESM1-M']
gcm_names_ssps = ['BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L', 
                  'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM']
gcm_names_ssp119 = ['EC-Earth3', 'EC-Earth3-Veg', 'GFDL-ESM4', 'MRI-ESM2-0']
#rcps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585', 'rcp26', 'rcp45', 'rcp85']
rcps = ['ssp119','ssp126', 'ssp245', 'ssp370', 'ssp585']
#rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

normyear = 2015

netcdf_fp_cmip5_land = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v4/'
netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-runoff_fixed/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-nodebris/'
#netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_ssp119/'

temp_dev_fn = 'Global_mean_temp_deviation_2081_2100_rel_1850_1900.csv'

ds_marzeion2020_fn = '/Users/drounce/Documents/HiMAT/spc_backup/analysis_calving_v3/Marzeion_etal_2020_results.nc'

analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
fig_fp = analysis_fp + '/figures/'
csv_fp = analysis_fp + '/csv/'
csv_fp_glacind = netcdf_fp_cmip5 + '_csv/'
csv_fp_glacind_land = netcdf_fp_cmip5_land + '_csv/'
pickle_fp = analysis_fp + '/pickle/'

rgi_shp_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_all_simplified2_robinson.shp'
rgi_regions_fn = '/Users/drounce/Documents/Papers/pygem_oggm_global/qgis/rgi60_regions_robinson-v2.shp'

    
degree_size = 0.1

vn_title_dict = {'massbal':'Mass\nBalance',                                                                      
                 'precfactor':'Precipitation\nFactor',                                                              
                 'tempchange':'Temperature\nBias',                                                               
                 'ddfsnow':'Degree-Day \nFactor of Snow'}
vn_label_dict = {'massbal':'Mass Balance\n[mwea]',                                                                      
                 'precfactor':'Precipitation Factor\n[-]',                                                              
                 'tempchange':'Temperature Bias\n[$^\circ$C]',                                                               
                 'ddfsnow':'Degree Day Factor of Snow\n[mwe d$^{-1}$ $^\circ$C$^{-1}$]',
                 'dif_masschange':'Mass Balance [mwea]\n(Observation - Model)'}
vn_label_units_dict = {'massbal':'[mwea]',                                                                      
                       'precfactor':'[-]',                                                              
                       'tempchange':'[$^\circ$C]',                                                               
                       'ddfsnow':'[mwe d$^{-1}$ $^\circ$C$^{-1}$]'}
rgi_reg_dict = {'all':'Global',
                'global':'Global',
                1:'Alaska',
                2:'W Canada & US',
                3:'Arctic Canada North',
                4:'Arctic Canada South',
                5:'Greenland Periphery',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus & Middle East',
                13:'Central Asia',
                14:'South Asia West',
                15:'South Asia East',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctic & Subantarctic'
                }
# Colors list
rcp_colordict = {'rcp26':'#3D52A4', 'rcp45':'#76B8E5', 'rcp60':'#F47A20', 'rcp85':'#ED2024', 
                 'ssp119':'blue', 'ssp126':'#3D52A4', 'ssp245':'#76B8E5', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
rcp_styledict = {'rcp26':':', 'rcp45':':', 'rcp85':':',
                 'ssp119':'-', 'ssp126':'-', 'ssp245':'-', 'ssp370':'-', 'ssp585':'-'}

#%% ===== FUNCTIONS =====
def slr_mmSLEyr(reg_vol, reg_vol_bsl):
    """ Calculate annual SLR accounting for the ice below sea level following Farinotti et al. (2019) """
    # Farinotti et al. (2019)
#    reg_vol_asl = reg_vol - reg_vol_bsl
#    return (-1*(reg_vol_asl[:,1:] - reg_vol_asl[:,0:-1]) * 
#            pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000)
    # OGGM new approach
    if len(reg_vol.shape) == 2:
        return (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                 (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
    elif len(reg_vol.shape) == 1:
        return (-1*(((reg_vol[1:] - reg_vol[0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                 (reg_vol_bsl[1:] - reg_vol_bsl[0:-1])) / pygem_prms.area_ocean * 1000))

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid defined by WGS84
    (From https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7)
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    (from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7)
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
#                    from numpy import meshgrid, deg2rad, gradient, cos
#                    from xarray import DataArray

    xlon, ylat = np.meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = np.deg2rad(np.gradient(ylat, axis=0))
    dlon = np.deg2rad(np.gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * np.cos(np.deg2rad(ylat))

    area = dy * dx

    xda = xr.DataArray(
        area,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda
        
#%% ===== TEMPERATURE INCREASE FOR THE VARIOUS CLIMATE SCENARIOS =====     
if option_policy_temp_figs:
    
    startyear = 1986
    endyear = 2100
    temps_plot_mad = [1.5,4]

    startyear_plot=2000
    endyear_plot=2100
    temp_colordict = {}
    for ngroup, deg_group in enumerate(deg_groups):
        temp_colordict[deg_group] = deg_group_colors[ngroup]

    fig_fp_multigcm = fig_fp + 'multi_gcm/'
    if not os.path.exists(fig_fp_multigcm):
        os.makedirs(fig_fp_multigcm, exist_ok=True)
    
    # Filenames
    fn_reg_temp_all = 'reg_temp_all.pkl'
    fn_reg_temp_all_monthly = 'reg_temp_all_monthly.pkl'
    fn_reg_prec_all = 'reg_prec_all.pkl'
    fn_reg_prec_all_monthly = 'reg_prec_all_monthly.pkl'
    fn_reg_datestable = 'policytemp_climate_dates_table_monthly.pkl'
            
    if os.path.exists(pickle_fp + fn_reg_temp_all):
        
        with open(pickle_fp + fn_reg_temp_all, 'rb') as f:
            reg_temp_all = pickle.load(f)
        with open(pickle_fp + fn_reg_temp_all_monthly, 'rb') as f:
            reg_temp_all_monthly = pickle.load(f)
        with open(pickle_fp + fn_reg_prec_all, 'rb') as f:
            reg_prec_all = pickle.load(f)
        with open(pickle_fp + fn_reg_prec_all_monthly, 'rb') as f:
            reg_prec_all_monthly = pickle.load(f)
        with open(pickle_fp + fn_reg_datestable, 'rb') as f:
            dates_table = pickle.load(f)
            
        years_climate = np.unique(dates_table.year)
        
    else:
    
        reg_temp_all = {}
        reg_temp_all_monthly = {}
        reg_prec_all = {}
        reg_prec_all_monthly = {}
        reg_main_glac_rgi = {}
        # Set up regions
        for reg in regions:
            reg_temp_all[reg] = {}
            reg_temp_all_monthly[reg] = {}
            reg_prec_all[reg] = {}
            reg_prec_all_monthly[reg] = {}
    
            for rcp in rcps:
                reg_temp_all[reg][rcp] = {}
                reg_temp_all_monthly[reg][rcp] = {}
                reg_prec_all[reg][rcp] = {}
                reg_prec_all_monthly[reg][rcp] = {}
            
            # Glaciers in each region rather than reload every time
            main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], rgi_regionsO2='all', 
                                                              rgi_glac_number='all')
            reg_main_glac_rgi[reg] = main_glac_rgi
            
        # Set up Global and All regions
        #  - global includes every pixel
        #  - all includes only glacierized pixels
        reg_temp_all['all'] = {}
        reg_temp_all_monthly['all'] = {}
        reg_prec_all['all'] = {}
        reg_prec_all_monthly['all'] = {}
        reg_temp_all['global'] = {}
        reg_temp_all_monthly['global'] = {}
        reg_prec_all['global'] = {}
        reg_prec_all_monthly['global'] = {}
    
        for rcp in rcps:
            reg_temp_all['all'][rcp] = {}
            reg_temp_all_monthly['all'][rcp] = {}
            reg_prec_all['all'][rcp] = {}
            reg_prec_all_monthly['all'][rcp] = {}
            reg_temp_all['global'][rcp] = {}
            reg_temp_all_monthly['global'][rcp] = {}
            reg_prec_all['global'][rcp] = {}
            reg_prec_all_monthly['global'][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                    
            dates_table = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0)
            
            for gcm_name in gcm_names:
    
                if rcp.startswith('ssp'):
                    # Variable names
                    temp_vn = 'tas'
                    prec_vn = 'pr'
                    elev_vn = 'orog'
                    lat_vn = 'lat'
                    lon_vn = 'lon'
                    time_vn = 'time'
                    # Variable filenames
                    temp_fn = gcm_name + '_' + rcp + '_r1i1p1f1_' + temp_vn + '.nc'
                    prec_fn = gcm_name + '_' + rcp + '_r1i1p1f1_' + prec_vn + '.nc'
                    elev_fn = gcm_name + '_' + elev_vn + '.nc'
                    # Variable filepaths
                    var_fp = pygem_prms.cmip6_fp_prefix + gcm_name + '/'
                    fx_fp = pygem_prms.cmip6_fp_prefix + gcm_name + '/'
                    # Extra information
                    timestep = pygem_prms.timestep
                        
                elif rcp.startswith('rcp'):
                    # Variable names
                    temp_vn = 'tas'
                    prec_vn = 'pr'
                    elev_vn = 'orog'
                    lat_vn = 'lat'
                    lon_vn = 'lon'
                    time_vn = 'time'
                    # Variable filenames
                    temp_fn = temp_vn + '_mon_' + gcm_name + '_' + rcp + '_r1i1p1_native.nc'
                    prec_fn = prec_vn + '_mon_' + gcm_name + '_' + rcp + '_r1i1p1_native.nc'
                    elev_fn = elev_vn + '_fx_' + gcm_name + '_' + rcp + '_r0i0p0.nc'
                    # Variable filepaths
                    var_fp = pygem_prms.cmip5_fp_var_prefix + rcp + pygem_prms.cmip5_fp_var_ending
                    fx_fp = pygem_prms.cmip5_fp_fx_prefix + rcp + pygem_prms.cmip5_fp_fx_ending
                    # Extra information
                    timestep = pygem_prms.timestep
                        
                if not os.path.exists(var_fp + temp_fn):
                    for realization in ['r4i1p1f1']:
                        fn_realization = temp_fn.replace('r1i1p1f1',realization)
                        if os.path.exists(var_fp + fn_realization):
                            temp_fn = fn_realization
                            prec_fn = prec_fn.replace('r1i1p1f1',realization)
                    
                ds_temp = xr.open_dataset(var_fp + temp_fn)
                ds_prec = xr.open_dataset(var_fp + prec_fn)
                
                start_idx = (np.where(pd.Series(ds_temp[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                                      dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
                end_idx = (np.where(pd.Series(ds_temp[time_vn]).apply(lambda x: x.strftime('%Y-%m')) == 
                                    dates_table['date'].apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]              
        
                time_series = pd.Series(ds_temp[time_vn][start_idx:end_idx+1])
                years = np.array([x.year for x in time_series][::12])
                
                # Global statistics
                if 'expver' in ds_temp.keys():
                    expver_idx = 0
                    temp_all = ds_temp[temp_vn][start_idx:end_idx+1, expver_idx, :, :].values
                    prec_all = ds_prec[prec_vn][start_idx:end_idx+1, expver_idx, :, :].values
                else:
                    temp_all = ds_temp[temp_vn][start_idx:end_idx+1, :, :].values
                    prec_all = ds_prec[prec_vn][start_idx:end_idx+1, :, :].values
                    
                # Correct precipitaiton to monthly
                if 'units' in ds_prec[prec_vn].attrs and ds_prec[prec_vn].attrs['units'] == 'kg m-2 s-1':  
                    # Convert from kg m-2 s-1 to m day-1
                    prec_all = prec_all/1000*3600*24
                    #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
                # Else check the variables units
                else:
                    print('Check units of precipitation from GCM is meters per day.')
                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                if 'daysinmonth' in dates_table.columns:
                    prec_all = prec_all * dates_table['daysinmonth'].values[:,np.newaxis,np.newaxis]

                # Global average must be weighted by area
                # area dataArray
                da_area = area_grid(ds_temp[lat_vn].values, ds_temp[lon_vn].values)
                latlon_areas = da_area.values
                # total area
                total_area = da_area.sum(['latitude','longitude']).values
                # temperature weighted by grid-cell area
                temp_global_mean_monthly = (temp_all*latlon_areas[np.newaxis,:,:]).sum((1,2)) / total_area
                temp_global_mean_annual = temp_global_mean_monthly.reshape(-1,12).mean(axis=1)
                
                # precipitation weighted by grid-cell area
                prec_global_mean_monthly = (prec_all*latlon_areas[np.newaxis,:,:]).sum((1,2)) / total_area
                prec_global_sum_annual = prec_global_mean_monthly.reshape(-1,12).sum(axis=1)
                
                reg_temp_all['global'][rcp][gcm_name] = temp_global_mean_annual
                reg_temp_all_monthly['global'][rcp][gcm_name] = temp_global_mean_monthly
                reg_prec_all['global'][rcp][gcm_name] = prec_global_sum_annual
                reg_prec_all_monthly['global'][rcp][gcm_name] = prec_global_mean_monthly
                

                # Regional statistics
                for nreg, reg in enumerate(regions):
                    
                    print(rcp, gcm_name, reg)
                    
                    main_glac_rgi = reg_main_glac_rgi[reg]
                        
                    #  argmin() finds the minimum distance between the glacier lat/lon and the GCM pixel; .values is used to 
                    #  extract the position's value as opposed to having an array
                    lat_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lat_colname].values[:,np.newaxis] - 
                                          ds_temp.variables[lat_vn][:].values).argmin(axis=1))
                    lon_nearidx = (np.abs(main_glac_rgi[pygem_prms.rgi_lon_colname].values[:,np.newaxis] - 
                                          ds_temp.variables[lon_vn][:].values).argmin(axis=1))
                    # Find unique latitude/longitudes
                    latlon_nearidx = list(zip(lat_nearidx, lon_nearidx))
                    latlon_nearidx_unique = list(set(latlon_nearidx))
                    # Create dictionary of time series for each unique latitude/longitude
                    temp_reg_latlon = {}
                    prec_reg_latlon = {}
                    area_reg_latlon = {}
                    for latlon in latlon_nearidx_unique:   
                        area_reg_latlon[latlon] = da_area[latlon[0],latlon[1]].values
                        if 'expver' in ds_temp.keys():
                            expver_idx = 0
                            temp_reg_latlon[latlon] = ds_temp[temp_vn][start_idx:end_idx+1, expver_idx, latlon[0], latlon[1]].values
                            prec_reg_latlon[latlon] = ds_prec[prec_vn][start_idx:end_idx+1, expver_idx, latlon[0], latlon[1]].values
                        else:
                            temp_reg_latlon[latlon] = ds_temp[temp_vn][start_idx:end_idx+1, latlon[0], latlon[1]].values
                            prec_reg_latlon[latlon] = ds_prec[prec_vn][start_idx:end_idx+1, latlon[0], latlon[1]].values
                    
                    # Convert to regional mean
                    area_reg_all = np.array([area_reg_latlon[x] for x in latlon_nearidx_unique])
                    
                    # Temperature mean
                    temp_reg_all = np.array([temp_reg_latlon[x] for x in latlon_nearidx_unique])
                    temp_reg_mean_monthly = (temp_reg_all * area_reg_all[:,np.newaxis]).sum(0) / area_reg_all.sum(0)
                    temp_reg_mean_annual = temp_reg_mean_monthly.reshape(-1,12).mean(axis=1)
                    
                    # Precipitation mean
                    prec_reg_all = np.array([prec_reg_latlon[x] for x in latlon_nearidx_unique])
                    prec_reg_mean_monthly = (prec_reg_all * area_reg_all[:,np.newaxis]).sum(0) / area_reg_all.sum()
                    # Correct precipitation to monthly
                    if 'units' in ds_prec[prec_vn].attrs and ds_prec[prec_vn].attrs['units'] == 'kg m-2 s-1':  
                        # Convert from kg m-2 s-1 to m day-1
                        prec_reg_mean_monthly = prec_reg_mean_monthly/1000*3600*24
                        #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
                    else:
                        print('Check units of precipitation from GCM is meters per day.')
                    # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                    if 'daysinmonth' in dates_table.columns:
                        prec_reg_mean_monthly = prec_reg_mean_monthly * dates_table['daysinmonth'].values
                    prec_reg_sum_annual = prec_reg_mean_monthly.reshape(-1,12).sum(axis=1)
                    
                    # Record data
                    reg_temp_all[reg][rcp][gcm_name] = temp_reg_mean_annual
                    reg_temp_all_monthly[reg][rcp][gcm_name] = temp_reg_mean_monthly
                    reg_prec_all[reg][rcp][gcm_name] = prec_reg_sum_annual
                    reg_prec_all_monthly[reg][rcp][gcm_name] = prec_reg_mean_monthly
                    
                    
                    if nreg == 0:
                        area_reg_all_raw = area_reg_all
                        temp_reg_all_raw = temp_reg_all
                        prec_reg_all_raw = prec_reg_all
                    else:
                        area_reg_all_raw = np.concatenate((area_reg_all_raw, area_reg_all), axis=0)
                        temp_reg_all_raw = np.concatenate((temp_reg_all_raw, temp_reg_all), axis=0)
                        prec_reg_all_raw = np.concatenate((prec_reg_all_raw, prec_reg_all), axis=0)
                    
                # All glacierized statistics
                area_reg_all = area_reg_all_raw
                temp_reg_all = temp_reg_all_raw
                temp_reg_mean_monthly = (temp_reg_all * area_reg_all[:,np.newaxis]).sum(0) / area_reg_all.sum(0)
                temp_reg_mean_annual = temp_reg_mean_monthly.reshape(-1,12).mean(axis=1)
                
                prec_reg_all = prec_reg_all_raw
                prec_reg_mean_monthly = (prec_reg_all * area_reg_all[:,np.newaxis]).sum(0) / area_reg_all.sum()
                # Correct precipitation to monthly
                if 'units' in ds_prec[prec_vn].attrs and ds_prec[prec_vn].attrs['units'] == 'kg m-2 s-1':  
                    # Convert from kg m-2 s-1 to m day-1
                    prec_reg_mean_monthly = prec_reg_mean_monthly/1000*3600*24
                    #   (1 kg m-2 s-1) * (1 m3/1000 kg) * (3600 s / hr) * (24 hr / day) = (m day-1)
                else:
                    print('Check units of precipitation from GCM is meters per day.')
                # Convert from meters per day to meters per month (COAWST data already 'monthly accumulated precipitation')
                if 'daysinmonth' in dates_table.columns:
                    prec_reg_mean_monthly = prec_reg_mean_monthly * dates_table['daysinmonth'].values
                prec_reg_sum_annual = prec_reg_mean_monthly.reshape(-1,12).sum(axis=1)
                

                # Record data
                reg_temp_all['all'][rcp][gcm_name] = temp_reg_mean_annual
                reg_temp_all_monthly['all'][rcp][gcm_name] = temp_reg_mean_monthly
                reg_prec_all['all'][rcp][gcm_name] = prec_reg_sum_annual
                reg_prec_all_monthly['all'][rcp][gcm_name] = prec_reg_mean_monthly
        
        with open(pickle_fp + fn_reg_datestable, 'wb') as f:
            pickle.dump(dates_table, f)        
        with open(pickle_fp + fn_reg_temp_all, 'wb') as f:
            pickle.dump(reg_temp_all, f)
        with open(pickle_fp + fn_reg_temp_all_monthly, 'wb') as f:
            pickle.dump(reg_temp_all_monthly, f)
        with open(pickle_fp + fn_reg_prec_all, 'wb') as f:
            pickle.dump(reg_prec_all, f)
        with open(pickle_fp + fn_reg_prec_all_monthly, 'wb') as f:
            pickle.dump(reg_prec_all_monthly, f)
            
        years_climate = years    

    #%% ----- FIGURE SHOWING DISTRIBUTION OF TEMPERATURES FOR VARIOUS RCP/SSP SCENARIOS -----
    normyear_idx = list(years_climate).index(normyear)
    
    gcm_colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#e6f598', '#abdda4',
                  '#66c2a5', '#3288bd', '#5e4fa2', '#253494'][::-1]
                  
    for rcp in rcps:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=6,ncols=8,wspace=1,hspace=0.4)
        ax1 = fig.add_subplot(gs[0:2,0:3])
        ax2 = fig.add_subplot(gs[0,6:])
        ax3 = fig.add_subplot(gs[1,4:6])
        ax4 = fig.add_subplot(gs[1,6:])
        ax5 = fig.add_subplot(gs[2,0:2])
        ax6 = fig.add_subplot(gs[2,2:4])
        ax7 = fig.add_subplot(gs[2,4:6])
        ax8 = fig.add_subplot(gs[2,6:])
        ax9 = fig.add_subplot(gs[3,0:2])
        ax10 = fig.add_subplot(gs[3,2:4])
        ax11 = fig.add_subplot(gs[3,4:6])
        ax12 = fig.add_subplot(gs[3,6:])
        ax13 = fig.add_subplot(gs[4,0:2])
        ax14 = fig.add_subplot(gs[4,2:4])
        ax15 = fig.add_subplot(gs[4,4:6])
        ax16 = fig.add_subplot(gs[4,6:])
        ax17 = fig.add_subplot(gs[5,0:2])
        ax18 = fig.add_subplot(gs[5,2:4])
        ax19 = fig.add_subplot(gs[5,4:6])
        ax20 = fig.add_subplot(gs[5,6:])
        
        regions_ordered = ['global',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
        for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        for nax, ax in enumerate([ax1]):
            
            reg = regions_ordered[nax]
            
            # Order GCMs based on global mean temperature increase for SSP245
            if rcp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
                rcp_key_order = list(reg_temp_all['global']['ssp245'].keys())
            if rcp in ['ssp119']:
                rcp_key_order = list(reg_temp_all['global']['ssp119'].keys())
            if rcp in ['rcp45']:
                rcp_key_order = list(reg_temp_all['global']['rcp45'].keys())
            gcm_name_list = []
            temp_annual_chg_list = []
            for gcm_name in rcp_key_order:
                
                temp_annual = reg_temp_all[reg][rcp][gcm_name][normyear_idx:] - 273.15
                temp_annual_chg = temp_annual[-1] - temp_annual[0]
                
                gcm_name_list.append(gcm_name)
                temp_annual_chg_list.append(temp_annual_chg)
                
            # Sort lists from coolest to warmest
            gcm_name_list_sorted = [x for _,x in sorted(zip(temp_annual_chg_list, gcm_name_list))]
            temp_annual_chg_list_sorted = sorted(temp_annual_chg_list)
            
            # Plot each one
            for ngcm, gcm_name in enumerate(gcm_name_list_sorted): 
                temp_annual = reg_temp_all[reg][rcp][gcm_name][normyear_idx:] - 273.15
                temp_annual_runningmean = uniform_filter(temp_annual, size=(11))
                
                ax.plot(years_climate[normyear_idx:], temp_annual_runningmean, linewidth=1, color=gcm_colors[ngcm], label=gcm_name)

            ax.set_xlim(2015,2100)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xticks([2050,2100])
#            ax.tick_params(axis='both', which='major', direction='inout', right=True)
#            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            if nax == 0:
                label_height=1.06
            else:
                label_height=1.14
            ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            if nax == 1:
                ax.legend(loc=(-2.2,0.0), fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                          handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                          )
#        fig.text(0.5,0.08,'Year', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
        fig.text(0.07,0.5,'Temperature ($^\circ$C)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
        
        # Save figure
        fig_fn = 'Temp_timeseries_regional_11yr_runningmean-' + rcp + '.png'
        fig.set_size_inches(8.5,11)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)   
    
    #%% ----- RELATIVE CHANGE IN TEMPERATURE FIGURE -----
    for rcp in rcps:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=6,ncols=8,wspace=1,hspace=0.4)
        ax1 = fig.add_subplot(gs[0:2,0:3])
        ax2 = fig.add_subplot(gs[0,6:])
        ax3 = fig.add_subplot(gs[1,4:6])
        ax4 = fig.add_subplot(gs[1,6:])
        ax5 = fig.add_subplot(gs[2,0:2])
        ax6 = fig.add_subplot(gs[2,2:4])
        ax7 = fig.add_subplot(gs[2,4:6])
        ax8 = fig.add_subplot(gs[2,6:])
        ax9 = fig.add_subplot(gs[3,0:2])
        ax10 = fig.add_subplot(gs[3,2:4])
        ax11 = fig.add_subplot(gs[3,4:6])
        ax12 = fig.add_subplot(gs[3,6:])
        ax13 = fig.add_subplot(gs[4,0:2])
        ax14 = fig.add_subplot(gs[4,2:4])
        ax15 = fig.add_subplot(gs[4,4:6])
        ax16 = fig.add_subplot(gs[4,6:])
        ax17 = fig.add_subplot(gs[5,0:2])
        ax18 = fig.add_subplot(gs[5,2:4])
        ax19 = fig.add_subplot(gs[5,4:6])
        ax20 = fig.add_subplot(gs[5,6:])
        
        regions_ordered = ['global',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
        for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        for nax, ax in enumerate([ax1]):
            
            reg = regions_ordered[nax]
            
            # Order GCMs based on global mean temperature increase for SSP245
            if rcp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
                rcp_key_order = list(reg_temp_all['global']['ssp245'].keys())
            if rcp in ['ssp119']:
                rcp_key_order = list(reg_temp_all['global']['ssp119'].keys())
            if rcp in ['rcp45']:
                rcp_key_order = list(reg_temp_all['global']['rcp45'].keys())
            gcm_name_list = []
            temp_annual_chg_list = []
            for gcm_name in rcp_key_order:
                temp_annual = reg_temp_all[reg][rcp][gcm_name][normyear_idx:] - 273.15
                temp_annual_chg = temp_annual[-1] - temp_annual[0]
                
                gcm_name_list.append(gcm_name)
                temp_annual_chg_list.append(temp_annual_chg)
                
            # Sort lists from coolest to warmest
            gcm_name_list_sorted = [x for _,x in sorted(zip(temp_annual_chg_list, gcm_name_list))]
            temp_annual_chg_list_sorted = sorted(temp_annual_chg_list)
            
            # Plot each one
            for ngcm, gcm_name in enumerate(gcm_name_list_sorted): 
                temp_annual = reg_temp_all[reg][rcp][gcm_name][normyear_idx:] - 273.15
                temp_annual_runningmean = uniform_filter(temp_annual, size=(11))
                temp_annual_runningmean_norm = temp_annual_runningmean - temp_annual_runningmean[0]
                
                ax.plot(years_climate[normyear_idx:], temp_annual_runningmean_norm, linewidth=1, color=gcm_colors[ngcm], label=gcm_name)

            ax.set_xlim(2015,2100)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xticks([2050,2100])

            ax.set_ylim(-1.5, 5)
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            

            if nax == 0:
                label_height=1.06
            else:
                label_height=1.14
            ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            if nax == 1:
                ax.legend(loc=(-2.2,0.0), fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                          handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                          )
#        fig.text(0.5,0.08,'Year', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
        fig.text(0.07,0.5,'$\Delta$Temperature rel. to 2015 ($^\circ$C)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
        
        # Save figure
        fig_fn = 'Temp_timeseries_regional_11yr_runningmean_relative-' + rcp + '.png'
        fig.set_size_inches(8.5,11)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)   
        
    #%% ----- PRECIPITATION FIGURE -----
    for rcp in rcps:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=6,ncols=8,wspace=1,hspace=0.4)
        ax1 = fig.add_subplot(gs[0:2,0:3])
        ax2 = fig.add_subplot(gs[0,6:])
        ax3 = fig.add_subplot(gs[1,4:6])
        ax4 = fig.add_subplot(gs[1,6:])
        ax5 = fig.add_subplot(gs[2,0:2])
        ax6 = fig.add_subplot(gs[2,2:4])
        ax7 = fig.add_subplot(gs[2,4:6])
        ax8 = fig.add_subplot(gs[2,6:])
        ax9 = fig.add_subplot(gs[3,0:2])
        ax10 = fig.add_subplot(gs[3,2:4])
        ax11 = fig.add_subplot(gs[3,4:6])
        ax12 = fig.add_subplot(gs[3,6:])
        ax13 = fig.add_subplot(gs[4,0:2])
        ax14 = fig.add_subplot(gs[4,2:4])
        ax15 = fig.add_subplot(gs[4,4:6])
        ax16 = fig.add_subplot(gs[4,6:])
        ax17 = fig.add_subplot(gs[5,0:2])
        ax18 = fig.add_subplot(gs[5,2:4])
        ax19 = fig.add_subplot(gs[5,4:6])
        ax20 = fig.add_subplot(gs[5,6:])
        
        regions_ordered = ['global',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
        for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        for nax, ax in enumerate([ax1]):
            
            reg = regions_ordered[nax]
            
            # Order GCMs based on global mean temperature increase for SSP245
            if rcp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
                rcp_key_order = list(reg_temp_all['global']['ssp245'].keys())
            if rcp in ['ssp119']:
                rcp_key_order = list(reg_temp_all['global']['ssp119'].keys())
            if rcp in ['rcp45']:
                rcp_key_order = list(reg_temp_all['global']['rcp45'].keys())
            gcm_name_list = []
            temp_annual_chg_list = []
            for gcm_name in rcp_key_order:
                temp_annual = reg_temp_all[reg][rcp][gcm_name][normyear_idx:] - 273.15
                temp_annual_chg = temp_annual[-1] - temp_annual[0]
                
                gcm_name_list.append(gcm_name)
                temp_annual_chg_list.append(temp_annual_chg)
                
            # Sort lists from coolest to warmest
            gcm_name_list_sorted = [x for _,x in sorted(zip(temp_annual_chg_list, gcm_name_list))]
            temp_annual_chg_list_sorted = sorted(temp_annual_chg_list)
                
            # Plot each one
            for ngcm, gcm_name in enumerate(gcm_name_list_sorted): 
                prec_annual = reg_prec_all[reg][rcp][gcm_name][normyear_idx:]
                prec_annual_runningmean = uniform_filter(prec_annual, size=(11))
                prec_annual_runningmean_norm = prec_annual_runningmean - prec_annual_runningmean[0]
                ax.plot(years_climate[normyear_idx:], prec_annual_runningmean, linewidth=0.5, color=gcm_colors[ngcm], label=gcm_name)
#                ax.plot(years_climate[normyear_idx:], prec_annual_runningmean_norm, linewidth=0.5, color=gcm_colors[ngcm], label=gcm_name)

            ax.set_xlim(2015,2100)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xticks([2050,2100])
            
#            ax.set_ylim(-0.35,0.35)
#            ax.yaxis.set_major_locator(MultipleLocator(0.2))
#            ax.yaxis.set_minor_locator(MultipleLocator(0.05))

            if nax == 0:
                label_height=1.06
            else:
                label_height=1.14
            ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            if nax == 1:
                ax.legend(loc=(-2.2,0.0), fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                          handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                          )
#        fig.text(0.5,0.08,'Year', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
        fig.text(0.07,0.5,'Precipitation (m)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
        
        # Save figure
        fig_fn = 'Prec_timeseries_regional_11yr_runningmean-' + rcp + '.png'
        fig.set_size_inches(8.5,11)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- RELATIVE PRECIPITATION CHANGE FIGURE -----
    for rcp in rcps:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=6,ncols=8,wspace=1,hspace=0.4)
        ax1 = fig.add_subplot(gs[0:2,0:3])
        ax2 = fig.add_subplot(gs[0,6:])
        ax3 = fig.add_subplot(gs[1,4:6])
        ax4 = fig.add_subplot(gs[1,6:])
        ax5 = fig.add_subplot(gs[2,0:2])
        ax6 = fig.add_subplot(gs[2,2:4])
        ax7 = fig.add_subplot(gs[2,4:6])
        ax8 = fig.add_subplot(gs[2,6:])
        ax9 = fig.add_subplot(gs[3,0:2])
        ax10 = fig.add_subplot(gs[3,2:4])
        ax11 = fig.add_subplot(gs[3,4:6])
        ax12 = fig.add_subplot(gs[3,6:])
        ax13 = fig.add_subplot(gs[4,0:2])
        ax14 = fig.add_subplot(gs[4,2:4])
        ax15 = fig.add_subplot(gs[4,4:6])
        ax16 = fig.add_subplot(gs[4,6:])
        ax17 = fig.add_subplot(gs[5,0:2])
        ax18 = fig.add_subplot(gs[5,2:4])
        ax19 = fig.add_subplot(gs[5,4:6])
        ax20 = fig.add_subplot(gs[5,6:])
        
        regions_ordered = ['global',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
        for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        for nax, ax in enumerate([ax1]):
            
            reg = regions_ordered[nax]
            
            # Order GCMs based on global mean temperature increase for SSP245
            if rcp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
                rcp_key_order = list(reg_temp_all['global']['ssp245'].keys())
            if rcp in ['ssp119']:
                rcp_key_order = list(reg_temp_all['global']['ssp119'].keys())
            if rcp in ['rcp45']:
                rcp_key_order = list(reg_temp_all['global']['rcp45'].keys())
            gcm_name_list = []
            temp_annual_chg_list = []
            for gcm_name in rcp_key_order:
                temp_annual = reg_temp_all[reg][rcp][gcm_name][normyear_idx:] - 273.15
                temp_annual_chg = temp_annual[-1] - temp_annual[0]
                
                gcm_name_list.append(gcm_name)
                temp_annual_chg_list.append(temp_annual_chg)
                
            # Sort lists from coolest to warmest
            gcm_name_list_sorted = [x for _,x in sorted(zip(temp_annual_chg_list, gcm_name_list))]
            temp_annual_chg_list_sorted = sorted(temp_annual_chg_list)
                
            # Plot each one
            for ngcm, gcm_name in enumerate(gcm_name_list_sorted): 
                prec_annual = reg_prec_all[reg][rcp][gcm_name][normyear_idx:]
                prec_annual_runningmean = uniform_filter(prec_annual, size=(11))
                prec_annual_runningmean_norm = prec_annual_runningmean - prec_annual_runningmean[0]
#                ax.plot(years_climate[normyear_idx:], prec_annual_runningmean, linewidth=0.5, color=gcm_colors[ngcm], label=gcm_name)
                ax.plot(years_climate[normyear_idx:], prec_annual_runningmean_norm, linewidth=0.5, color=gcm_colors[ngcm], label=gcm_name)

            ax.set_xlim(2015,2100)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xticks([2050,2100])
            
            ax.set_ylim(-0.35,0.35)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))

            if nax == 0:
                label_height=1.06
            else:
                label_height=1.14
            ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            if nax == 1:
                ax.legend(loc=(-2.2,0.0), fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                          handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                          )
#        fig.text(0.5,0.08,'Year', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
        fig.text(0.07,0.5,'$\Delta$Precipitation rel. to 2015 (m)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
        
        # Save figure
        fig_fn = 'Prec_timeseries_regional_11yr_runningmean_relative-' + rcp + '.png'
        fig.set_size_inches(8.5,11)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
        
        
    #%% ----- RELATIVE PRECIPITATION CHANGE (%) FIGURE -----
    for rcp in rcps:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=6,ncols=8,wspace=1,hspace=0.4)
        ax1 = fig.add_subplot(gs[0:2,0:3])
        ax2 = fig.add_subplot(gs[0,6:])
        ax3 = fig.add_subplot(gs[1,4:6])
        ax4 = fig.add_subplot(gs[1,6:])
        ax5 = fig.add_subplot(gs[2,0:2])
        ax6 = fig.add_subplot(gs[2,2:4])
        ax7 = fig.add_subplot(gs[2,4:6])
        ax8 = fig.add_subplot(gs[2,6:])
        ax9 = fig.add_subplot(gs[3,0:2])
        ax10 = fig.add_subplot(gs[3,2:4])
        ax11 = fig.add_subplot(gs[3,4:6])
        ax12 = fig.add_subplot(gs[3,6:])
        ax13 = fig.add_subplot(gs[4,0:2])
        ax14 = fig.add_subplot(gs[4,2:4])
        ax15 = fig.add_subplot(gs[4,4:6])
        ax16 = fig.add_subplot(gs[4,6:])
        ax17 = fig.add_subplot(gs[5,0:2])
        ax18 = fig.add_subplot(gs[5,2:4])
        ax19 = fig.add_subplot(gs[5,4:6])
        ax20 = fig.add_subplot(gs[5,6:])
        
        regions_ordered = ['global',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
        for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
#        for nax, ax in enumerate([ax1]):
            
            reg = regions_ordered[nax]
            
            # Order GCMs based on global mean temperature increase for SSP245
            if rcp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
                rcp_key_order = list(reg_temp_all['global']['ssp245'].keys())
            if rcp in ['ssp119']:
                rcp_key_order = list(reg_temp_all['global']['ssp119'].keys())
            if rcp in ['rcp45']:
                rcp_key_order = list(reg_temp_all['global']['rcp45'].keys())
            gcm_name_list = []
            temp_annual_chg_list = []
            for gcm_name in rcp_key_order:
                temp_annual = reg_temp_all[reg][rcp][gcm_name][normyear_idx:] - 273.15
                temp_annual_chg = temp_annual[-1] - temp_annual[0]
                
                gcm_name_list.append(gcm_name)
                temp_annual_chg_list.append(temp_annual_chg)
                
            # Sort lists from coolest to warmest
            gcm_name_list_sorted = [x for _,x in sorted(zip(temp_annual_chg_list, gcm_name_list))]
            temp_annual_chg_list_sorted = sorted(temp_annual_chg_list)
                
            # Plot each one
            prec_annual_runningmean_2015_list = []
            for ngcm, gcm_name in enumerate(gcm_name_list_sorted): 
                prec_annual = reg_prec_all[reg][rcp][gcm_name][normyear_idx:]
                prec_annual_runningmean = uniform_filter(prec_annual, size=(11))
                prec_annual_runningmean_norm = prec_annual_runningmean / prec_annual_runningmean[0]
                prec_annual_runningmean_2015_list.append(prec_annual_runningmean[0])
#                ax.plot(years_climate[normyear_idx:], prec_annual_runningmean, linewidth=0.5, color=gcm_colors[ngcm], label=gcm_name)
                ax.plot(years_climate[normyear_idx:], prec_annual_runningmean_norm, linewidth=0.5, color=gcm_colors[ngcm], label=gcm_name)
            

            ax.set_xlim(2015,2100)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xticks([2050,2100])
            
            ax.set_ylim(0.75,1.50)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05))

            if nax == 0:
                label_height=1.06
            else:
                label_height=1.14
            ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.text(0.05, 0.95, str(np.round(np.mean(prec_annual_runningmean_2015_list),2)) + 'm', size=10, horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            if nax == 1:
                ax.legend(loc=(-2.2,0.0), fontsize=10, ncol=2, columnspacing=0.5, labelspacing=0.25, 
                          handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                          )
#        fig.text(0.5,0.08,'Year', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
        fig.text(0.07,0.5,'Precipitation (rel. to 2015)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
        
        # Save figure
        fig_fn = 'Prec_timeseries_regional_11yr_runningmean_relative%-' + rcp + '.png'
        fig.set_size_inches(8.5,11)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)


    #%% ----- PROCESS DATA -----
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_bwl = {}
    reg_area_all = {} 
    reg_glac_vol_all = {}
    reg_glac_rgiids_all = {}
    reg_melt_all = {}
    reg_acc_all = {}
    reg_refreeze_all = {}
    reg_fa_all = {}
    reg_fa_err_all = {}
    
    # Set up Global region
    reg_vol_all['all'] = {}
    reg_vol_all_bwl['all'] = {}
    reg_area_all['all'] = {}
    reg_glac_vol_all['all'] = {}
    reg_glac_rgiids_all['all'] = {}
    reg_melt_all['all'] = {}
    reg_acc_all['all'] = {}
    reg_refreeze_all['all'] = {}
    reg_fa_all['all'] = {}
    reg_fa_err_all['all'] = {}
    for rcp in rcps:
        reg_vol_all['all'][rcp] = {}
        reg_vol_all_bwl['all'][rcp] = {}
        reg_area_all['all'][rcp] = {}
        reg_glac_vol_all['all'][rcp] = {}
        reg_glac_rgiids_all['all'][rcp] = {}
        reg_melt_all['all'][rcp] = {}
        reg_acc_all['all'][rcp] = {}
        reg_refreeze_all['all'][rcp] = {}
        reg_fa_all['all'][rcp] = {}
        reg_fa_err_all['all'][rcp] = {}
            
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        for gcm_name in gcm_names:
            reg_vol_all['all'][rcp][gcm_name] = None
            reg_vol_all_bwl['all'][rcp][gcm_name] = None
            reg_area_all['all'][rcp][gcm_name] = None
            reg_glac_vol_all['all'][rcp][gcm_name] = None
            reg_glac_rgiids_all['all'][rcp][gcm_name] = None
            reg_melt_all['all'][rcp][gcm_name] = None
            reg_acc_all['all'][rcp][gcm_name] = None
            reg_refreeze_all['all'][rcp][gcm_name] = None
            reg_fa_all['all'][rcp][gcm_name] = None
            reg_fa_err_all['all'][rcp][gcm_name] = None
            
    for reg in regions:
        reg_vol_all[reg] = {}
        reg_vol_all_bwl[reg] = {}
        reg_area_all[reg] = {}
        reg_glac_vol_all[reg] = {}
        reg_glac_rgiids_all[reg] = {}
        reg_melt_all[reg] = {}
        reg_acc_all[reg] = {}
        reg_refreeze_all[reg] = {}
        reg_fa_all[reg] = {}
        reg_fa_err_all[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_bwl[reg][rcp] = {}
            reg_area_all[reg][rcp] = {}
            reg_glac_vol_all[reg][rcp] = {}
            reg_glac_rgiids_all[reg][rcp] = {}
            reg_melt_all[reg][rcp] = {}
            reg_acc_all[reg][rcp] = {}
            reg_refreeze_all[reg][rcp] = {}
            reg_fa_all[reg][rcp] = {}
            reg_fa_err_all[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                print(reg, rcp, gcm_name)
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_land = (netcdf_fp_cmip5_land + '../analysis/pickle/' + str(reg).zfill(2) + 
                                  '/O1Regions/' + gcm_name + '/' + rcp + '/')
                pickle_fp_tw = pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                if '_calving' in netcdf_fp_cmip5 and not os.path.exists(pickle_fp_tw):
#                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
                    pickle_fp_reg =  pickle_fp_land
                    csv_fp_glacvol = csv_fp_glacind_land
                else:
                    pickle_fp_reg =  pickle_fp_tw
                    csv_fp_glacvol = csv_fp_glacind
                    
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                fn_reg_area_annual = reg_rcp_gcm_str + '_area_annual.pkl'
                fn_reg_area_annual_bd = reg_rcp_gcm_str + '_area_annual_bd.pkl'
                fn_reg_acc_monthly = reg_rcp_gcm_str + '_acc_monthly.pkl'
                fn_reg_refreeze_monthly = reg_rcp_gcm_str + '_refreeze_monthly.pkl'
                fn_reg_melt_monthly = reg_rcp_gcm_str + '_melt_monthly.pkl'
                fn_reg_frontalablation_monthly = reg_rcp_gcm_str + '_frontalablation_monthly.pkl'
                
                 # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                # Volume below sea level
                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                    reg_vol_annual_bwl = pickle.load(f)
                # Area 
                with open(pickle_fp_reg + fn_reg_area_annual, 'rb') as f:
                    reg_area_annual = pickle.load(f)   
                # Mass balance: accumulation
                with open(pickle_fp_reg + fn_reg_acc_monthly, 'rb') as f:
                    reg_acc_monthly = pickle.load(f)
                # Mass balance: refreeze
                with open(pickle_fp_reg + fn_reg_refreeze_monthly, 'rb') as f:
                    reg_refreeze_monthly = pickle.load(f)
                # Mass balance: melt
                with open(pickle_fp_reg + fn_reg_melt_monthly, 'rb') as f:
                    reg_melt_monthly = pickle.load(f)
                # Mass balance: frontal ablation
                with open(pickle_fp_reg + fn_reg_frontalablation_monthly, 'rb') as f:
                    reg_frontalablation_monthly = pickle.load(f)
                    
                # Regional glacier volume data -----
                vol_annual_fn = str(reg).zfill(2) + '_' + gcm_name + '_' + rcp + '_glac_vol_annual.csv'
                
                reg_glac_vol_annual_df = pd.read_csv(csv_fp_glacvol + vol_annual_fn)
                rgiids_gcm_raw = list(reg_glac_vol_annual_df.values[:,0])
                rgiids_gcm = [str(reg) + '.' + str(int(np.round((x-reg)*1e5))).zfill(5) for x in rgiids_gcm_raw]
                reg_glac_vol_annual_gcm = reg_glac_vol_annual_df.values[:,1:]
#                reg_lost_count = len(np.where(reg_glac_vol_annual_gcm[:,-1] == 0)[0])
                
                # Record data
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                if reg_vol_annual_bwl is None:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = np.zeros(reg_vol_annual.shape)
                else:
                    reg_vol_all_bwl[reg][rcp][gcm_name] = reg_vol_annual_bwl
                reg_area_all[reg][rcp][gcm_name] = reg_area_annual  
                reg_glac_vol_all[reg][rcp][gcm_name] = reg_glac_vol_annual_gcm
                reg_glac_rgiids_all[reg][rcp][gcm_name] = rgiids_gcm.copy()
                reg_melt_all[reg][rcp][gcm_name] = reg_melt_monthly
                reg_acc_all[reg][rcp][gcm_name] = reg_acc_monthly
                reg_refreeze_all[reg][rcp][gcm_name] = reg_refreeze_monthly
                reg_fa_all[reg][rcp][gcm_name] = reg_frontalablation_monthly
                
                if reg_vol_all['all'][rcp][gcm_name] is None:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all[reg][rcp][gcm_name]
                    reg_glac_vol_all['all'][rcp][gcm_name] = reg_glac_vol_all[reg][rcp][gcm_name]
                    reg_glac_rgiids_all['all'][rcp][gcm_name] = rgiids_gcm.copy()
                    reg_melt_all['all'][rcp][gcm_name] = reg_melt_all[reg][rcp][gcm_name]
                    reg_acc_all['all'][rcp][gcm_name] = reg_acc_all[reg][rcp][gcm_name]
                    reg_refreeze_all['all'][rcp][gcm_name] = reg_refreeze_all[reg][rcp][gcm_name]
                    reg_fa_all['all'][rcp][gcm_name] = reg_fa_all[reg][rcp][gcm_name]
                else:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all['all'][rcp][gcm_name] + reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl['all'][rcp][gcm_name] + reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_area_all['all'][rcp][gcm_name] = reg_area_all['all'][rcp][gcm_name] + reg_area_all[reg][rcp][gcm_name]
                    reg_glac_vol_all['all'][rcp][gcm_name] = np.concatenate((reg_glac_vol_all['all'][rcp][gcm_name], reg_glac_vol_all[reg][rcp][gcm_name]), axis=0)
                    reg_glac_rgiids_all['all'][rcp][gcm_name].extend(rgiids_gcm.copy())
                    reg_melt_all['all'][rcp][gcm_name] = reg_melt_all['all'][rcp][gcm_name] + reg_melt_all[reg][rcp][gcm_name]
                    reg_acc_all['all'][rcp][gcm_name] = reg_acc_all['all'][rcp][gcm_name] + reg_acc_all[reg][rcp][gcm_name]
                    reg_refreeze_all['all'][rcp][gcm_name] = reg_refreeze_all['all'][rcp][gcm_name] + reg_refreeze_all[reg][rcp][gcm_name]
                    reg_fa_all['all'][rcp][gcm_name] = reg_fa_all['all'][rcp][gcm_name] + reg_fa_all[reg][rcp][gcm_name]
                    
                    assert reg_glac_vol_all['all'][rcp][gcm_name].shape[0] == len(reg_glac_rgiids_all['all'][rcp][gcm_name]), 'check all regions lengths match'
                    assert reg_glac_vol_all[reg][rcp][gcm_name].shape[0] == len(reg_glac_rgiids_all[reg][rcp][gcm_name]), 'check region lengths match'
                    assert reg_glac_vol_all[1][rcp][gcm_name].shape[0] == len(reg_glac_rgiids_all[1][rcp][gcm_name]), 'check copying issues such that region lengths match'
                
    regions.append('all')
      
    #%% Global temperature increase
    temp_dev_cns = ['Scenario', 'GCM', 'global_mean_deviation_degC']
    for reg in regions:
        temp_dev_cns.append(str(reg) + '_dev_degC')
    ngcm_rcps = 0
    for rcp in rcps:
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        for gcm_name in gcm_names:
            ngcm_rcps += 1
    temp_dev_df = pd.DataFrame(np.zeros((ngcm_rcps,len(temp_dev_cns))), columns=temp_dev_cns)
    ncount = 0
    for rcp in rcps:
        
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        
        for gcm_name in gcm_names:

            idx_1986 = np.where(years_climate==1986)[0][0]
            idx_2005 = np.where(years_climate==2005)[0][0]
            
            temp_global_mean_annual = reg_temp_all['global'][rcp][gcm_name]
            temp_global_mean_1986_2005 = temp_global_mean_annual[idx_1986:idx_2005+1].mean()
            
            # SROCC Summary Policy Makers: future global mean surface air temperature relative to 1986-2005 
            #   + 0.63 to account fo changes from 1850-1986
            temp_global_mean_deviation = 0.63 + temp_global_mean_annual - temp_global_mean_1986_2005
            
            global_mean_deviation_degC = temp_global_mean_deviation[-20:].mean()
            
            for reg in regions:
                temp_reg_mean_annual = reg_temp_all[reg][rcp][gcm_name]
                temp_reg_mean_1986_2005 = temp_reg_mean_annual[idx_1986:idx_2005+1].mean()
                temp_reg_mean_deviation = 0.63 + temp_reg_mean_annual - temp_reg_mean_1986_2005
                reg_mean_deviation_degC = temp_reg_mean_deviation[-20:].mean()
            
                print(reg, rcp, gcm_name, reg_mean_deviation_degC)
                temp_dev_df.loc[ncount,str(reg) + '_dev_degC'] = reg_mean_deviation_degC
            
            temp_dev_df.loc[ncount,'GCM'] = gcm_name
            temp_dev_df.loc[ncount,'Scenario'] = rcp
            temp_dev_df.loc[ncount,'global_mean_deviation_degC'] = global_mean_deviation_degC
            
            ncount += 1
    
    temp_dev_df.to_csv(csv_fp + temp_dev_fn, index=False)
    
    #%% SLR for each region, gcm, and scenario
    years = np.arange(2000,2101+1)
    normyear_idx = np.where(years == normyear)[0][0]
    for reg in regions:
        temp_dev_df['SLR_mmSLE-' + str(reg)] = np.nan
        temp_dev_df['SLR_mmSLE_max-' + str(reg)] = np.nan
        temp_dev_df['Vol_2100_%-' + str(reg)] = np.nan
        temp_dev_df['Area_2100_%-' + str(reg)] = np.nan
        temp_dev_df['Glac_lost_2100-' + str(reg)] = np.nan
        temp_dev_df['Glac_count-' + str(reg)] = np.nan
        temp_dev_df['mb_mwea-' + str(reg)] = np.nan
    for rcp in rcps:
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        
        for gcm_name in gcm_names:
            
            ncount = temp_dev_df.loc[(temp_dev_df.Scenario==rcp) & (temp_dev_df.GCM==gcm_name)].index.values[0]
            for reg in regions:
                reg_vol = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_bsl = reg_vol_all_bwl[reg][rcp][gcm_name]
                reg_area = reg_area_all[reg][rcp][gcm_name]
                reg_glac_vol = reg_glac_vol_all[reg][rcp][gcm_name]
                
                # Cumulative Sea-level change [mm SLE]
                #  - accounts for water from glaciers replacing the ice that is below sea level as well
                #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
                reg_slr = (-1*(((reg_vol[1:] - reg_vol[0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                           (reg_vol_bsl[1:] - reg_vol_bsl[0:-1])) / pygem_prms.area_ocean * 1000))
                reg_slr_cum_raw = np.cumsum(reg_slr)
                reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[normyear_idx]
                
                reg_slr_max = reg_vol[normyear_idx] * pygem_prms.density_ice / pygem_prms.density_water / pygem_prms.area_ocean * 1000
                
                temp_dev_df.loc[ncount,'SLR_mmSLE-' + str(reg)] = reg_slr_cum[-1]
                temp_dev_df.loc[ncount,'SLR_mmSLE_max-' + str(reg)] = reg_slr_max
                
                temp_dev_df.loc[ncount,'Vol_2100_%-' + str(reg)] = reg_vol[-1] / reg_vol[normyear_idx] * 100
                temp_dev_df.loc[ncount,'Area_2100_%-' + str(reg)] = reg_area[-1] / reg_area[normyear_idx] * 100
                
                reg_glac_vol_2100 = reg_glac_vol[:,-1]
                reg_glac_lost = np.count_nonzero(reg_glac_vol_2100==0)

                temp_dev_df.loc[ncount,'Glac_lost_2100-' + str(reg)] = reg_glac_lost
                temp_dev_df.loc[ncount,'Glac_count-' + str(reg)] = reg_glac_vol.shape[0]                
                
                # Specific mass balance
                reg_mass = reg_vol * pygem_prms.density_ice    
                reg_mb = (reg_mass[1:] - reg_mass[0:-1]) / reg_area[0:-1]
                temp_dev_df.loc[ncount,'mb_mwea-' + str(reg)] = reg_mb[-1]
                

    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE -----
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.patch.set_facecolor('none')
        
    for rcp in rcps:
        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == rcp]
        
        if rcp.startswith('ssp'):
            marker = 'o'
        else:
            marker = 'd'
        ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-all'], 
                linewidth=0, marker=marker, mec=rcp_colordict[rcp], mew=1, mfc='none', label=rcp)
    
    ax.set_xlabel('Global mean temperature change ($^\circ$C)')
    ax.set_ylabel('Sea level rise (mm SLE)', size=12)
#    ax.set_xlim(startyear, endyear)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
#    ax.set_ylim(0,1.1)
#    ax.yaxis.set_major_locator(MultipleLocator(25))
#    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='both', which='major', direction='inout', right=True)
    ax.tick_params(axis='both', which='minor', direction='in', right=True)
    
    ax.legend(labels=['SSP1-1.9','SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
#              loc=(0.02,0.75), 
              fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25,
              borderpad=0.1, ncol=2, columnspacing=0.5, frameon=True)
    
    # Save figure
    fig_fn = 'Temp_vs_SLR-global.png'
    fig.set_size_inches(4,3)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE REGIONAL -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]            
    
        slr_max = 0
        for scenario in rcps:
            temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
            if scenario.startswith('ssp'):
                marker = 'o'
            else:
                marker = 'd'
            ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-' + str(reg)], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
            
            if temp_dev_df_subset['SLR_mmSLE_max-' + str(reg)].mean() > slr_max:
                slr_max = temp_dev_df_subset['SLR_mmSLE_max-' + str(reg)].mean()
        
        ax.hlines(slr_max, 0, 7, color='k', linewidth=0.5)
        
        ax.set_xlim(0,7)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
            
        if reg in ['all']:
            ax.set_ylim(0,270)
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
        if reg in [19, 3, 1]:
            ax.set_ylim(0,52)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5))    
        elif reg in [5, 9]:
            ax.set_ylim(0,33)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
        elif reg in [4, 7]:
            ax.set_ylim(0,23)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
        elif reg in [17, 13, 6, 14]:
            ax.set_ylim(0,13)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1))
        elif reg in [15, 2]:
            ax.set_ylim(0,2.7)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
        elif reg in [8]:
            ax.set_ylim(0,0.8)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [10, 11, 16]:
            ax.set_ylim(0,0.32)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [12, 18]:
            ax.set_ylim(0,0.22)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        
        if nax == 0:
            label_height=1.07
        else:
            label_height=1.16
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        # Add statistics of regional deviation vs. global deviation
        if reg == 'all':
            reg_dev_str = ('Temperature anomaly over glaciers: ' + str(np.round(np.median((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC'])),2)) + 
                           r'$\pm$' + 
                           str(np.round((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC']).std(),2)))
        else:
            reg_dev_str = (str(np.round(np.median((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC'])),2)) + 
                       r'$\pm$' + 
                       str(np.round((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC']).std(),2)))
        ax.text(0.97, 0.03, reg_dev_str, size=10, horizontalalignment='right', 
                verticalalignment='bottom', transform=ax.transAxes)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncol=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncol=1
            ax.legend(loc=(-1.5,0.2), labels=labels, fontsize=10, ncol=ncol, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    fig.text(0.5,0.08,'Global mean temperature change ($^\circ$C)', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    fig.text(0.07,0.5,'Sea level rise 2015-2100 (mm SLE)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
    
    # Save figure
    fig_fn = 'Temp_vs_SLR-regional.png'
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)


    #%% ----- FIGURE: VOLUME CHANGE GLOBAL -----
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.patch.set_facecolor('none')
        
    for scenario in rcps:
        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
        
        if scenario.startswith('ssp'):
            marker = 'o'
        else:
            marker = 'd'
        ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Vol_2100_%-' + str(reg)]/100, 
                linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
    
    ax.set_xlabel('Global mean temperature change ($^\circ$C)')
    ax.set_ylabel('Mass at 2100 (rel. to 2015)', size=12)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_ylim(0,1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='both', which='major', direction='inout', right=True)
    ax.tick_params(axis='both', which='minor', direction='in', right=True)
    
    ax.legend(labels=['SSP1-1.9','SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
#              loc=(0.02,0.75), 
              fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25,
              borderpad=0.1, ncol=2, columnspacing=0.5, frameon=True)
    
    # Save figure
    fig_fn = 'Temp_vs_VolChg-global.png'
    fig.set_size_inches(4,3)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: VOL LOST REGIONAL -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]            
    
        slr_max = 0
        r_data_temp = []
        r_data_vol = []
        for scenario in rcps:
            temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
            if scenario.startswith('ssp'):
                marker = 'o'
            else:
                marker = 'd'
                
            ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Vol_2100_%-' + str(reg)]/100, 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
            
            for nidx in temp_dev_df_subset.index.values:
                r_data_temp.append(temp_dev_df_subset.loc[nidx,'global_mean_deviation_degC'])
                r_data_vol.append(temp_dev_df_subset.loc[nidx,'Vol_2100_%-' + str(reg)])
        # Correlation
        slope, intercept, r_value, p_value, std_err = linregress(r_data_temp, r_data_vol)
        print(reg, '  r_value =', np.round(r_value**2,2))
#        print('slope = ', np.round(slope,2), 
#              'intercept = ', np.round(intercept,2), 'p_value = ', np.round(p_value,6))
            
        # Total mass
        reg_mass_init = []
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps    
            for gcm_name in gcm_names:
                reg_vol = reg_vol_all[reg][rcp][gcm_name]
                reg_mass = reg_vol * pygem_prms.density_ice
                reg_mass_init_Gt = reg_mass[normyear_idx] / 1e12
                reg_mass_init.append(reg_mass_init_Gt)
        reg_mass_init_med = np.median(reg_mass_init)

        ax.set_xlim(0,7)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
            
        ax.set_ylim(0,1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        
        if nax == 0:
            label_height=1.07
        else:
            label_height=1.16
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        # Add statistics of regional deviation vs. global deviation
        if reg == 'all':
            reg_dev_str = ('Glacier temp anomaly (\N{DEGREE SIGN}C): ' + 
                           str(np.round(np.median((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC'])),2)) + 
                           r'$\pm$' + 
                           str(np.round((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC']).std(),2)))
        else:
            reg_dev_str = (str(np.round(np.median((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC'])),2)) + 
                           r'$\pm$' + 
                           str(np.round((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC']).std(),2)))
        text1 = ax.text(0.98, 0.98, reg_dev_str, size=10, horizontalalignment='right', 
                       verticalalignment='top', transform=ax.transAxes)
        text1.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='None', pad=0))
        
        if reg == 'all':
            text2 = ax.text(0.98, 0.9, 'Mass at ' + str(normyear) + ' (10$^{3}$ Gt): ' + "{:.2f}".format(np.round(reg_mass_init_med/1000,2)), 
                            size=10, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, zorder=5)
        else:
            text2 = ax.text(0.98, 0.82, "{:.2f}".format(np.round(reg_mass_init_med/1000,2)), 
                        size=10, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, zorder=5)
        text2.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='None', pad=0))
        
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncol=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncol=1
            ax.legend(loc=(-1.35,0.2), labels=labels, fontsize=10, ncol=ncol, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    fig.text(0.5,0.08,'Global mean temperature change ($^\circ$C)', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    fig.text(0.07,0.5,'Mass at 2100 (rel. to 2015)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
    
    # Save figure
    fig_fn = 'Temp_vs_VolChg-regional.png'
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: AREA CHANGE GLOBAL -----
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.patch.set_facecolor('none')
        
    for scenario in rcps:
        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
        
        if scenario.startswith('ssp'):
            marker = 'o'
        else:
            marker = 'd'
        ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Area_2100_%-' + str(reg)]/100, 
                linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
    
    ax.set_xlabel('Global mean temperature change ($^\circ$C)')
    ax.set_ylabel('Area at 2100 (rel. to 2015)', size=12)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.set_ylim(0,1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis='both', which='major', direction='inout', right=True)
    ax.tick_params(axis='both', which='minor', direction='in', right=True)
    
    ax.legend(labels=['SSP1-1.9','SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
#              loc=(0.02,0.75), 
              fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25,
              borderpad=0.1, ncol=2, columnspacing=0.5, frameon=True)
    
    # Save figure
    fig_fn = 'Temp_vs_AreaChg-global.png'
    fig.set_size_inches(4,3)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: AREA LOST REGIONAL -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]            
    
        slr_max = 0
        for scenario in rcps:
            temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
            if scenario.startswith('ssp'):
                marker = 'o'
            else:
                marker = 'd'
                
            ax.plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Area_2100_%-' + str(reg)]/100, 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)

        ax.set_xlim(0,7)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
            
        ax.set_ylim(0,1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        
        if nax == 0:
            label_height=1.07
        else:
            label_height=1.16
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        # Add statistics of regional deviation vs. global deviation
        reg_dev_str = (str(np.round(np.median((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC'])),2)) + 
                       r'$\pm$' + 
                       str(np.round((temp_dev_df[str(reg) + '_dev_degC'] - temp_dev_df['global_mean_deviation_degC']).std(),2)))
        ax.text(0.98, 0.98, reg_dev_str, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        
        if nax == 1:
            if 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==7:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=2
            elif 'rcp26' in rcps and 'ssp126' in rcps and len(rcps)==8:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=2
            elif 'rcp26' in rcps and len(rcps) == 3:
                labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']
                ncol=1
            elif 'ssp126' in rcps and len(rcps) == 4:
                labels = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncol=1
            elif 'ssp126' in rcps and len(rcps) == 5:
                labels = ['SSP1-1.9', 'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
                ncol=1
            ax.legend(loc=(-1.5,0.2), labels=labels, fontsize=10, ncol=ncol, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False
                      )
    fig.text(0.5,0.08,'Global mean temperature change ($^\circ$C)', fontsize=12, horizontalalignment='center', verticalalignment='bottom')
    fig.text(0.07,0.5,'Area at 2100 (rel. to 2015)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)
    
    # Save figure
    fig_fn = 'Temp_vs_AreaChg-regional.png'
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- OVERVIEW FIGURE -----
    fig, ax = plt.subplots(2, 2, squeeze=False, sharex=False, sharey=False, 
                           gridspec_kw = {'wspace':0.6, 'hspace':0.15})
        
    glac_count_total = np.median(temp_dev_df['Glac_count-all'])
    for scenario in rcps:
        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
        
        if scenario.startswith('ssp'):
            marker = 'o'
        else:
            marker = 'd'
        # Volume
        ax[0,0].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Vol_2100_%-' + str(reg)]/100, 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
        # Area
        ax[0,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Area_2100_%-' + str(reg)]/100, 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
        # SLR
        ax[1,0].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-all'], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
        # Number
        ax[1,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Glac_lost_2100-all'], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=1, mfc='none', label=scenario)
    
    
    ax[0,0].set_ylabel('Mass at 2100 (rel. to 2015)', size=12)
    ax[0,1].set_ylabel('Area at 2100 (rel. to 2015)', size=12)
    ax[1,0].set_ylabel('Sea level rise (mm SLE)', size=12)
    ax[1,1].set_ylabel('Glaciers lost (count)', size=12)
        
    fig.text(0.5, 0.05, 'Global mean temperature change ($^\circ$C)', 
             horizontalalignment='center', verticalalignment='bottom', size=12)
   
    ax[0,0].set_xlim(0,7)
    ax[0,0].xaxis.set_major_locator(MultipleLocator(1))
    ax[0,0].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[0,1].set_xlim(0,7)
    ax[0,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[0,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[1,0].set_xlim(0,7)
    ax[1,0].xaxis.set_major_locator(MultipleLocator(1))
    ax[1,0].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[1,1].set_xlim(0,7)
    ax[1,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[1,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax[0,0].set_ylim(0,1)
    ax[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
    ax[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[0,1].set_ylim(0,1)
    ax[0,1].yaxis.set_major_locator(MultipleLocator(0.2))
    ax[0,1].yaxis.set_minor_locator(MultipleLocator(0.1))
    ax[1,0].set_ylim(0,300)
    ax[1,0].yaxis.set_major_locator(MultipleLocator(100))
    ax[1,0].yaxis.set_minor_locator(MultipleLocator(20))
    ax[1,1].set_ylim(0,glac_count_total)
    ax[1,1].yaxis.set_major_locator(MultipleLocator(50000))
    ax[1,1].yaxis.set_minor_locator(MultipleLocator(10000))
    
    ax[0,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[0,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[0,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[0,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[1,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[1,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[1,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[1,1].tick_params(axis='both', which='minor', direction='in', right=True)
    
    # Legend
    ax[0,0].legend(labels=['SSP1-1.9','SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
    #              loc=(0.02,0.75), 
                  fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25,
                  borderpad=0.1, ncol=1, columnspacing=0.5, frameon=True)
    
    # Save figure
    fig_fn = 'Temp_Overview-global.png'
    fig.set_size_inches(6,6)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    

    #%% MULTI-GCM PLOTS BY DEGREE
    # Set up processing
    reg_vol_all_bydeg = {}
    reg_vol_all_bydeg_bwl = {}
    reg_area_all_bydeg = {} 
    reg_glac_vol_all_bydeg = {}
    reg_glac_rgiids_all_bydeg = {}
    reg_melt_all_bydeg = {}
    reg_acc_all_bydeg = {}
    reg_refreeze_all_bydeg = {}
    reg_fa_all_bydeg = {}
    
    # Set up Global region
    reg_vol_all_bydeg['all'] = {}
    reg_vol_all_bydeg_bwl['all'] = {}
    reg_area_all_bydeg['all'] = {}
    reg_glac_vol_all_bydeg['all'] = {}
    reg_glac_rgiids_all_bydeg['all'] = {}
    reg_melt_all_bydeg['all'] = {}
    reg_acc_all_bydeg['all'] = {}
    reg_refreeze_all_bydeg['all'] = {}
    reg_fa_all_bydeg['all'] = {}
    temp_gcmrcp_dict = {}
    for deg_group in deg_groups:
        reg_vol_all_bydeg['all'][deg_group] = {}
        reg_vol_all_bydeg_bwl['all'][deg_group] = {}
        reg_area_all_bydeg['all'][deg_group] = {}
        reg_glac_vol_all_bydeg['all'][deg_group] = {}
        reg_glac_rgiids_all_bydeg['all'][deg_group] = {}
        reg_melt_all_bydeg['all'][deg_group] = {}
        reg_acc_all_bydeg['all'][deg_group] = {}
        reg_refreeze_all_bydeg['all'][deg_group] = {}
        reg_fa_all_bydeg['all'][deg_group] = {}
        temp_gcmrcp_dict[deg_group] = []
    temp_dev_df['rcp_gcm_name'] = [temp_dev_df.loc[x,'Scenario'] + '/' + temp_dev_df.loc[x,'GCM'] for x in temp_dev_df.index.values]
    
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        for rcp_gcm_name in temp_dev_df_subset['rcp_gcm_name']:
            temp_gcmrcp_dict[deg_group].append(rcp_gcm_name)
            reg_vol_all_bydeg['all'][deg_group][rcp_gcm_name] = None
            reg_vol_all_bydeg_bwl['all'][deg_group][rcp_gcm_name] = None
            reg_area_all_bydeg['all'][deg_group][rcp_gcm_name] = None
            reg_glac_vol_all_bydeg['all'][deg_group][rcp_gcm_name] = None
            reg_glac_rgiids_all_bydeg['all'][deg_group][rcp_gcm_name] = None
            reg_melt_all_bydeg['all'][deg_group][rcp_gcm_name] = None
            reg_acc_all_bydeg['all'][deg_group][rcp_gcm_name] = None
            reg_refreeze_all_bydeg['all'][deg_group][rcp_gcm_name] = None
            reg_fa_all_bydeg['all'][deg_group][rcp_gcm_name] = None
    
    regions.remove('all')     

    # Set up regions
    for reg in regions:
        reg_vol_all_bydeg[reg] = {}
        reg_vol_all_bydeg_bwl[reg] = {}
        reg_area_all_bydeg[reg] = {}  
        reg_glac_vol_all_bydeg[reg] = {}
        reg_glac_rgiids_all_bydeg[reg] = {}
        reg_melt_all_bydeg[reg] = {}
        reg_acc_all_bydeg[reg] = {}
        reg_refreeze_all_bydeg[reg] = {}
        reg_fa_all_bydeg[reg] = {}
        for deg_group in deg_groups:
            reg_vol_all_bydeg[reg][deg_group] = {}
            reg_vol_all_bydeg_bwl[reg][deg_group] = {}
            reg_area_all_bydeg[reg][deg_group] = {}
            reg_glac_vol_all_bydeg[reg][deg_group] = {}
            reg_glac_rgiids_all_bydeg[reg][deg_group] = {}
            reg_melt_all_bydeg[reg][deg_group] = {}
            reg_acc_all_bydeg[reg][deg_group] = {}
            reg_refreeze_all_bydeg[reg][deg_group] = {}
            reg_fa_all_bydeg[reg][deg_group] = {}
         
    for reg in regions:
        for ngroup, deg_group in enumerate(deg_groups):
            for rcp_gcm_name in temp_gcmrcp_dict[deg_group]:
                
                print('\n', reg, deg_group, rcp_gcm_name)
                
                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                reg_vol_annual = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_annual_bwl = reg_vol_all_bwl[reg][rcp][gcm_name]
                reg_area_annual = reg_area_all[reg][rcp][gcm_name]
                reg_glac_vol_annual = reg_glac_vol_all[reg][rcp][gcm_name]
                reg_glac_rgiids = reg_glac_rgiids_all[reg][rcp][gcm_name]
                reg_melt_monthly = reg_melt_all[reg][rcp][gcm_name]
                reg_acc_monthly = reg_acc_all[reg][rcp][gcm_name]
                reg_refreeze_monthly = reg_refreeze_all[reg][rcp][gcm_name]
                reg_frontalablation_monthly = reg_fa_all[reg][rcp][gcm_name]
                
                print('len(reg_glac_rgiids):', len(reg_glac_rgiids), 'glaciers:', reg_glac_vol_annual.shape[0])
        
                reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_vol_annual
                if reg_vol_annual_bwl is None:
                    reg_vol_all_bydeg_bwl[reg][deg_group][rcp_gcm_name] = np.zeros(reg_vol_annual.shape)
                else:
                    reg_vol_all_bydeg_bwl[reg][deg_group][rcp_gcm_name] = reg_vol_annual_bwl
                reg_area_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_area_annual  
                reg_glac_vol_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_glac_vol_annual  
                reg_glac_rgiids_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_glac_rgiids.copy()
                reg_melt_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_melt_monthly
                reg_acc_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_acc_monthly
                reg_refreeze_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_refreeze_monthly
                reg_fa_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_frontalablation_monthly
                
                if reg_vol_all_bydeg['all'][deg_group][rcp_gcm_name] is None:
                    reg_vol_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_vol_all_bydeg_bwl['all'][deg_group][rcp_gcm_name] = reg_vol_all_bydeg_bwl[reg][deg_group][rcp_gcm_name]
                    reg_area_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_area_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_glac_vol_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_glac_vol_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_glac_rgiids_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_glac_rgiids_all_bydeg[reg][deg_group][rcp_gcm_name].copy()
                    reg_melt_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_melt_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_acc_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_acc_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_refreeze_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_refreeze_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_fa_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_fa_all_bydeg[reg][deg_group][rcp_gcm_name]
                    print(reg_glac_vol_all_bydeg['all'][deg_group][rcp_gcm_name].shape[0], len(reg_glac_rgiids_all_bydeg['all'][deg_group][rcp_gcm_name]))
                else:
                    reg_vol_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_vol_all_bydeg['all'][deg_group][rcp_gcm_name] + reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_vol_all_bydeg_bwl['all'][deg_group][rcp_gcm_name] = reg_vol_all_bydeg_bwl['all'][deg_group][rcp_gcm_name] + reg_vol_all_bydeg_bwl[reg][deg_group][rcp_gcm_name]
                    reg_area_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_area_all_bydeg['all'][deg_group][rcp_gcm_name] + reg_area_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_glac_vol_all_bydeg['all'][deg_group][rcp_gcm_name] = np.concatenate((reg_glac_vol_all_bydeg['all'][deg_group][rcp_gcm_name], 
                                                                                             reg_glac_vol_all_bydeg[reg][deg_group][rcp_gcm_name]),axis=0)
                    reg_glac_rgiids_all_bydeg['all'][deg_group][rcp_gcm_name].extend(reg_glac_rgiids_all_bydeg[reg][deg_group][rcp_gcm_name].copy())
                    reg_melt_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_melt_all_bydeg['all'][deg_group][rcp_gcm_name] + reg_melt_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_acc_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_acc_all_bydeg['all'][deg_group][rcp_gcm_name] + reg_acc_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_refreeze_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_refreeze_all_bydeg['all'][deg_group][rcp_gcm_name] + reg_refreeze_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_fa_all_bydeg['all'][deg_group][rcp_gcm_name] = reg_fa_all_bydeg['all'][deg_group][rcp_gcm_name] + reg_fa_all_bydeg[reg][deg_group][rcp_gcm_name]
                    
                    print(reg_glac_vol_all_bydeg['all'][deg_group][rcp_gcm_name].shape[0], len(reg_glac_rgiids_all_bydeg['all'][deg_group][rcp_gcm_name]))
                
                assert reg_glac_vol_all_bydeg['all'][deg_group][rcp_gcm_name].shape[0] == len(reg_glac_rgiids_all_bydeg['all'][deg_group][rcp_gcm_name])
                assert reg_glac_vol_all_bydeg[reg][deg_group][rcp_gcm_name].shape[0] == len(reg_glac_rgiids_all_bydeg[reg][deg_group][rcp_gcm_name])
                assert reg_glac_vol_all_bydeg[1][deg_group][rcp_gcm_name].shape[0] == len(reg_glac_rgiids_all_bydeg[1][deg_group][rcp_gcm_name])
                    
    regions.append('all')
    
    
    #%%
    # MULTI-GCM STATISTICS
    ds_multigcm_vol_bydeg = {}
    ds_multigcm_vol_bydeg_bsl = {}
    ds_multigcm_area_bydeg = {}
    ds_multigcm_glac_lost_bydeg = {}
    ds_multigcm_glac_lost_slr_bydeg = {}
    ds_multigcm_glac_rgiids_bydeg = {}
    ds_multigcm_glac_lost2100_rgiids_bydeg = {}    
    ds_multigcm_melt_bydeg = {}
    ds_multigcm_acc_bydeg = {}
    ds_multigcm_refreeze_bydeg = {}
    ds_multigcm_fa_bydeg = {}
    
    for reg in regions:
        ds_multigcm_vol_bydeg[reg] = {}
        ds_multigcm_vol_bydeg_bsl[reg] = {}
        ds_multigcm_area_bydeg[reg] = {}
        ds_multigcm_glac_lost_bydeg[reg] = {}
        ds_multigcm_glac_lost_slr_bydeg[reg] = {}
        ds_multigcm_glac_rgiids_bydeg[reg] = {}
        ds_multigcm_glac_lost2100_rgiids_bydeg[reg] = {}
        ds_multigcm_melt_bydeg[reg] = {}
        ds_multigcm_acc_bydeg[reg] = {}
        ds_multigcm_refreeze_bydeg[reg] = {}
        ds_multigcm_fa_bydeg[reg] = {}
        for deg_group in deg_groups:
            
            gcm_rcps_list = temp_gcmrcp_dict[deg_group]
            
            for ngcm, rcp_gcm_name in enumerate(gcm_rcps_list):

                print(reg, deg_group, rcp_gcm_name)
                
                reg_vol_gcm = reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_vol_bsl_gcm = reg_vol_all_bydeg_bwl[reg][deg_group][rcp_gcm_name]
                reg_area_gcm = reg_area_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_glac_vol_gcm = reg_glac_vol_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_melt_monthly_gcm = reg_melt_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_acc_monthly_gcm = reg_acc_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_refreeze_monthly_gcm = reg_refreeze_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_frontalablation_monthly_gcm = reg_fa_all_bydeg[reg][deg_group][rcp_gcm_name]
                
                reg_glac_lost_gcm = np.count_nonzero(reg_glac_vol_gcm==0, axis=0)
                
                lost_idx = np.where(reg_glac_vol_gcm[:,-1] == 0)[0]
#                print(reg, deg_group, rcp_gcm_name, reg_glac_vol_gcm[lost_idx,normyear_idx].sum() / reg_glac_vol_gcm[:,normyear_idx].sum()*100)

                # Roughly estimate assuming no bsl for the lost glaciers since small glaciers unlikely to have significant fraction bsl
                #  - using this rough estimate as hack because don't have data
                reg_glac_lost_slr_gcm = slr_mmSLEyr(reg_glac_vol_gcm[lost_idx,:].sum(0), np.zeros((reg_glac_vol_gcm.shape[1])))
                
                # RGIIds for each simulation
                rgiids_gcm = reg_glac_rgiids_all_bydeg[reg][deg_group][rcp_gcm_name]
                rgiids_lost2100_gcm = [reg_glac_rgiids_all_bydeg[reg][deg_group][rcp_gcm_name][x] for x in lost_idx]

                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm   
                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm 
                    reg_area_gcm_all = reg_area_gcm
                    reg_glac_lost_gcm_all = reg_glac_lost_gcm
                    reg_glac_lost_slr_gcm_all = reg_glac_lost_slr_gcm
                    rgiids_gcm_all = rgiids_gcm.copy()
                    rgiids_lost2100_gcm_all = rgiids_lost2100_gcm.copy()
                    reg_melt_monthly_gcm_all = reg_melt_monthly_gcm
                    reg_acc_monthly_gcm_all = reg_acc_monthly_gcm
                    reg_refreeze_monthly_gcm_all = reg_refreeze_monthly_gcm
                    reg_frontalablation_monthly_gcm_all = reg_frontalablation_monthly_gcm
                    
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
                    reg_glac_lost_gcm_all = np.vstack((reg_glac_lost_gcm_all, reg_glac_lost_gcm))
                    reg_glac_lost_slr_gcm_all = np.vstack((reg_glac_lost_slr_gcm_all, reg_glac_lost_slr_gcm))
                    rgiids_gcm_all.extend(rgiids_gcm.copy())
                    rgiids_lost2100_gcm_all.extend(rgiids_lost2100_gcm.copy())
                    reg_melt_monthly_gcm_all = np.vstack((reg_melt_monthly_gcm_all, reg_melt_monthly_gcm))
                    reg_acc_monthly_gcm_all = np.vstack((reg_acc_monthly_gcm_all, reg_acc_monthly_gcm))
                    reg_refreeze_monthly_gcm_all = np.vstack((reg_refreeze_monthly_gcm_all, reg_refreeze_monthly_gcm))
                    reg_frontalablation_monthly_gcm_all = np.vstack((reg_frontalablation_monthly_gcm_all, reg_frontalablation_monthly_gcm))
                    

            ds_multigcm_vol_bydeg[reg][deg_group] = reg_vol_gcm_all
            ds_multigcm_vol_bydeg_bsl[reg][deg_group] = reg_vol_bsl_gcm_all
            ds_multigcm_area_bydeg[reg][deg_group] = reg_area_gcm_all
            ds_multigcm_glac_lost_bydeg[reg][deg_group] = reg_glac_lost_gcm_all
            ds_multigcm_glac_lost_slr_bydeg[reg][deg_group] = reg_glac_lost_slr_gcm_all
            ds_multigcm_glac_rgiids_bydeg[reg][deg_group] = rgiids_gcm_all.copy()
            ds_multigcm_glac_lost2100_rgiids_bydeg[reg][deg_group] = rgiids_lost2100_gcm_all.copy()
            ds_multigcm_melt_bydeg[reg][deg_group] = reg_melt_monthly_gcm_all
            ds_multigcm_acc_bydeg[reg][deg_group] = reg_acc_monthly_gcm_all
            ds_multigcm_refreeze_bydeg[reg][deg_group] = reg_refreeze_monthly_gcm_all
            ds_multigcm_fa_bydeg[reg][deg_group] = reg_frontalablation_monthly_gcm_all
           
            
    #%% ---- TEMPERATURE VARIATIONS ----
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        print(deg_group, temp_dev_df_subset.shape[0], temp_dev_df_subset.global_mean_deviation_degC.mean())
            
    #%% ---- TABLES OF VALUES -----
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms', 
#                          'Marzeion_slr_mmsle_mean', 'Edwards_slr_mmsle_mean',
                          'slr_mmSLE_med', 'slr_mmSLE_90', 'slr_mmSLE_95', 'slr_mmSLE_mean', 'slr_mmSLE_std', 'slr_mmSLE_mad',
                          'masschg_mmSLE_med', 'masschg_mmSLE_90', 'masschg_mmSLE_95', 'masschg_mmSLE_mean', 'masschg_mmSLE_std', 'masschg_mmSLE_mad',
                          'slr_2090-2100_mmSLEyr_med', 'slr_2090-2100_mmSLEyr_90', 'slr_2090-2100_mmSLEyr_95', 'slr_2090-2100_mmSLEyr_mean', 'slr_2090-2100_mmSLEyr_std', 'slr_2090-2100_mmSLEyr_mad', 
                          'slr_max_mmSLEyr_med', 'slr_max_mmSLEyr_90', 'slr_max_mmSLEyr_95', 'slr_max_mmSLEyr_mean', 'slr_max_mmSLEyr_std', 'slr_max_mmSLEyr_mad', 
                          'slr_mmSLE_fromlost_med', 'slr_mmSLE_fromlost_90', 'slr_mmSLE_fromlost_95', 'slr_mmSLE_fromlost_mean', 'slr_mmSLE_fromlost_std', 'slr_mmSLE_fromlost_mad',
                          'yr_max_slr_med', 'yr_max_slr_mean', 'yr_max_slr_std', 'yr_max_slr_mad',
                          'vol_lost_%_med', 'vol_lost_%_90', 'vol_lost_%_95', 'vol_lost_%_mean', 'vol_lost_%_std', 'vol_lost_%_mad',
#                          'Marzeion_vol_lost_%_mean',
                          'area_lost_%_med', 'area_lost_%_mean', 'area_lost_%_std', 'area_lost_%_mad',
                          'count_lost_med', 'count_lost_90', 'count_lost_95', 'count_lost_mean', 'count_lost_std', 'count_lost_mad',
                          'count_lost_%_med', 'count_lost_%_90', 'count_lost_%_95', 'count_lost_%_mean', 'count_lost_%_std', 'count_lost_%_mad',
                          'mb_2090-2100_mmwea_med',  'mb_2090-2100_mmwea_90',  'mb_2090-2100_mmwea_95', 'mb_2090-2100_mmwea_mean', 'mb_2090-2100_mmwea_std', 'mb_2090-2100_mmwea_mad',
                          'mb_max_mmwea_med', 'mb_max_mmwea_90', 'mb_max_mmwea_95', 'mb_max_mmwea_mean', 'mb_max_mmwea_std', 'mb_max_mmwea_mad']
    stats_overview_df = pd.DataFrame(np.zeros((len(regions)*len(deg_groups),len(stats_overview_cns))), columns=stats_overview_cns)
    
    ncount = 0
    regions_overview = regions
    if 'all' in regions:
        regions_overview = [regions[-1]] + regions[0:-1]
    for nreg, reg in enumerate(regions_overview):
        for deg_group in deg_groups:
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_bsl = ds_multigcm_vol_bydeg_bsl[reg][deg_group]
            reg_area = ds_multigcm_area_bydeg[reg][deg_group]

            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_med = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_mean = np.mean(reg_slr_cum, axis=0)
            reg_slr_cum_std = np.std(reg_slr_cum, axis=0)
            reg_slr_cum_mad = median_abs_deviation(reg_slr_cum, axis=0)
            reg_slr_cum_90 = 1.645*reg_slr_cum_std
            reg_slr_cum_95 = 1.96*reg_slr_cum_std
            

            # Cumulative Sea-level change from lost glaciers [mm SLE]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr_lost = ds_multigcm_glac_lost_slr_bydeg[reg][deg_group]
            reg_slr_lost_cum_raw = np.cumsum(reg_slr_lost, axis=1)
            reg_slr_lost_cum = reg_slr_lost_cum_raw - reg_slr_lost_cum_raw[:,normyear_idx][:,np.newaxis]
            
            # See how much not accounting for bsl affects results and correct as an approximation
            reg_slr_nobsl = slr_mmSLEyr(reg_vol, np.zeros(reg_vol.shape))
            reg_slr_nobsl_cum_raw = np.cumsum(reg_slr_nobsl, axis=1)
            reg_slr_nobsl_cum = reg_slr_nobsl_cum_raw - reg_slr_nobsl_cum_raw[:,normyear_idx][:,np.newaxis]
            reg_slr_nobsl_cum_med = np.median(reg_slr_nobsl_cum, axis=0)
            
            reg_slr_lost_cum = reg_slr_lost_cum * reg_slr_cum_med[-1] / reg_slr_nobsl_cum_med[-1]
            
            reg_slr_lost_cum_med = np.median(reg_slr_lost_cum, axis=0)
            reg_slr_lost_cum_mean = np.mean(reg_slr_lost_cum, axis=0)
            reg_slr_lost_cum_std = np.std(reg_slr_lost_cum, axis=0)
            reg_slr_lost_cum_mad = median_abs_deviation(reg_slr_lost_cum, axis=0)
            
            # Sea-level change rate [mm SLE yr-1]
            reg_slr_20902100_med = np.median(reg_slr[:,-10:], axis=(0,1))
            reg_slr_20902100_mean = np.mean(reg_slr[:,-10:], axis=(0,1))
            reg_slr_20902100_std = np.std(reg_slr[:,-10:], axis=(0,1))
            reg_slr_20902100_mad = median_abs_deviation(reg_slr[:,-10:], axis=(0,1))
            reg_slr_20902100_90 = 1.645*reg_slr_20902100_std
            reg_slr_20902100_95 = 1.96*reg_slr_20902100_std
            
            # Sea-level change max rate
            reg_slr_med_raw = np.median(reg_slr, axis=0)
            reg_slr_mean_raw = np.mean(reg_slr, axis=0)
            reg_slr_std_raw = np.std(reg_slr, axis=0)
            reg_slr_mad_raw = median_abs_deviation(reg_slr, axis=0) 
            reg_slr_med = uniform_filter(reg_slr_med_raw, size=(11))
            slr_max_idx = np.where(reg_slr_med == reg_slr_med.max())[0]
            reg_slr_mean = uniform_filter(reg_slr_mean_raw, size=(11))
            reg_slr_std = uniform_filter(reg_slr_std_raw, size=(11))
            reg_slr_mad = uniform_filter(reg_slr_mad_raw, size=(11))
            
            # Year of maximum sea-level change rate
            #  - use a median filter to sort through the peaks which otherwise don't get smoothed with mean
            reg_slr_uniformfilter = np.zeros(reg_slr.shape)
            for nrow in np.arange(reg_slr.shape[0]):
                reg_slr_uniformfilter[nrow,:] = generic_filter(reg_slr[nrow,:], np.median, size=(11))
            reg_yr_slr_max = years[np.argmax(reg_slr_uniformfilter, axis=1)]
            reg_yr_slr_max[reg_yr_slr_max < normyear] = normyear
            reg_yr_slr_max_med = np.median(reg_yr_slr_max)
            reg_yr_slr_max_mean = np.mean(reg_yr_slr_max)
            reg_yr_slr_max_std = np.std(reg_yr_slr_max)
            reg_yr_slr_max_mad = median_abs_deviation(reg_yr_slr_max)
#            if reg in ['all']:
#                print('\n ', deg_group)
#                print('  ', reg_yr_slr_max_med, reg_yr_slr_max_mean, reg_yr_slr_max_std, reg_yr_slr_max_mad)
            
            
            # Cumulative Mass Loss [mm SLE]
            #  - no bsl correction
            reg_vol_bsl_none = np.zeros(reg_vol.shape)
            reg_masschg_sle = slr_mmSLEyr(reg_vol, reg_vol_bsl_none)
            reg_masschg_sle_cum_raw = np.cumsum(reg_masschg_sle, axis=1)
            reg_masschg_sle_cum = reg_masschg_sle_cum_raw - reg_masschg_sle_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_masschg_sle_cum_med = np.median(reg_masschg_sle_cum, axis=0)
            reg_masschg_sle_cum_mean = np.mean(reg_masschg_sle_cum, axis=0)
            reg_masschg_sle_cum_std = np.std(reg_masschg_sle_cum, axis=0)
            reg_masschg_sle_cum_mad = median_abs_deviation(reg_masschg_sle_cum, axis=0)
            
            # Volume lost [%]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_std = np.std(reg_vol, axis=0)
            reg_vol_mean = np.mean(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            reg_vol_lost_med_norm = (1 - reg_vol_med / reg_vol_med[normyear_idx]) * 100
            reg_vol_lost_mean_norm = (1 - reg_vol_mean / reg_vol_mean[normyear_idx]) * 100
            reg_vol_lost_std_norm = reg_vol_std / reg_vol_med[normyear_idx] * 100
            reg_vol_lost_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx] * 100
            
            # Area lost [%]
            reg_area_med = np.median(reg_area, axis=0)
            reg_area_std = np.std(reg_area, axis=0)
            reg_area_mean = np.mean(reg_area, axis=0)
            reg_area_mad = median_abs_deviation(reg_area, axis=0)
            reg_area_lost_med_norm = (1 - reg_area_med / reg_area_med[normyear_idx]) * 100
            reg_area_lost_mean_norm = (1 - reg_area_mean / reg_area_mean[normyear_idx]) * 100
            reg_area_lost_std_norm = reg_area_std / reg_area_med[normyear_idx] * 100
            reg_area_lost_mad_norm = reg_area_mad / reg_area_med[normyear_idx] * 100
            
            # Glaciers lost [count]
            reg_glac_lost = ds_multigcm_glac_lost_bydeg[reg][deg_group]
            reg_glac_lost_med = np.median(reg_glac_lost, axis=0)
            reg_glac_lost_mean = np.mean(reg_glac_lost, axis=0)
            reg_glac_lost_std = np.std(reg_glac_lost, axis=0)
            reg_glac_lost_mad = median_abs_deviation(reg_glac_lost, axis=0)
            
            # Glaciers lost [%]
            glac_count_total = np.median(temp_dev_df['Glac_count-' + str(reg)])
            reg_glac_lost_med_norm = np.median(reg_glac_lost, axis=0) / glac_count_total * 100
            reg_glac_lost_mean_norm = np.mean(reg_glac_lost, axis=0) / glac_count_total * 100
            reg_glac_lost_std_norm = np.std(reg_glac_lost, axis=0) / glac_count_total * 100
            reg_glac_lost_mad_norm = median_abs_deviation(reg_glac_lost, axis=0) / glac_count_total * 100
            
            # Specific mass balance [kg m-2 yr-1 or mm w.e. yr-1]
            reg_mass = reg_vol * pygem_prms.density_ice
            reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
            reg_mb_20902100_med = np.median(reg_mb[:,-10:], axis=(0,1))
            reg_mb_20902100_mean = np.mean(reg_mb[:,-10:], axis=(0,1))
            reg_mb_20902100_std = np.std(reg_mb[:,-10:], axis=(0,1))
            reg_mb_20902100_mad = median_abs_deviation(reg_mb[:,-10:], axis=(0,1))
            
            # Mass balance max rate
            reg_mb_med_raw = np.median(reg_mb, axis=0)
            reg_mb_mean_raw = np.mean(reg_mb, axis=0)
            reg_mb_std_raw = np.std(reg_mb, axis=0)
            reg_mb_mad_raw = median_abs_deviation(reg_mb, axis=0) 
            reg_mb_med = uniform_filter(reg_mb_med_raw, size=(11))
            mb_max_idx = np.where(reg_mb_med == reg_mb_med.min())[0]
            reg_mb_mean = uniform_filter(reg_mb_mean_raw, size=(11))
            reg_mb_std = uniform_filter(reg_mb_std_raw, size=(11))
            reg_mb_mad = uniform_filter(reg_mb_mad_raw, size=(11))
            
#            if reg in ['all']:
#                print(reg_mb_med)
#                print(reg_mb_med.min(), reg_mb_med[mb_max_idx])
            
            # RECORD STATISTICS
            stats_overview_df.loc[ncount,'Region'] = reg
            stats_overview_df.loc[ncount,'Scenario'] = deg_group
            stats_overview_df.loc[ncount,'n_gcms'] = reg_vol.shape[0]
            stats_overview_df.loc[ncount,'slr_mmSLE_med'] = reg_slr_cum_med[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_mean'] = reg_slr_cum_mean[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_std'] = reg_slr_cum_std[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_mad'] = reg_slr_cum_mad[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_90'] = reg_slr_cum_90[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_95'] = reg_slr_cum_95[-1]
            
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_med'] = np.max(reg_slr_med)
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_mean'] = np.max(reg_slr_mean)
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_std'] = reg_slr_std[slr_max_idx]
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_mad'] = reg_slr_mad[slr_max_idx]
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_90'] = 1.645*stats_overview_df.loc[ncount,'slr_max_mmSLEyr_std']
            stats_overview_df.loc[ncount,'slr_max_mmSLEyr_95'] = 1.96*stats_overview_df.loc[ncount,'slr_max_mmSLEyr_std']
            
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_med'] = reg_slr_20902100_med
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_mean'] = reg_slr_20902100_mean
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_std'] = reg_slr_20902100_std
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_mad'] = reg_slr_20902100_mad
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_90'] = reg_slr_20902100_90
            stats_overview_df.loc[ncount,'slr_2090-2100_mmSLEyr_95'] = reg_slr_20902100_95
            
            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_med'] = reg_slr_lost_cum_med[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_mean'] = reg_slr_lost_cum_mean[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_std'] = reg_slr_lost_cum_std[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_mad'] = reg_slr_lost_cum_mad[-1]
            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_90'] = 1.645*stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_std']
            stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_95'] = 1.96*stats_overview_df.loc[ncount,'slr_mmSLE_fromlost_std']
            
            stats_overview_df.loc[ncount,'yr_max_slr_med'] = reg_yr_slr_max_med
            stats_overview_df.loc[ncount,'yr_max_slr_mean'] = reg_yr_slr_max_mean
            stats_overview_df.loc[ncount,'yr_max_slr_std'] = reg_yr_slr_max_std
            stats_overview_df.loc[ncount,'yr_max_slr_mad'] = reg_yr_slr_max_mad
            
            stats_overview_df.loc[ncount,'masschg_mmSLE_med'] = reg_masschg_sle_cum_med[-1]
            stats_overview_df.loc[ncount,'masschg_mmSLE_mean'] = reg_masschg_sle_cum_mean[-1]
            stats_overview_df.loc[ncount,'masschg_mmSLE_std'] = reg_masschg_sle_cum_std[-1]
            stats_overview_df.loc[ncount,'masschg_mmSLE_mad'] = reg_masschg_sle_cum_mad[-1]
            stats_overview_df.loc[ncount,'masschg_mmSLE_90'] = 1.645*stats_overview_df.loc[ncount,'masschg_mmSLE_std']
            stats_overview_df.loc[ncount,'masschg_mmSLE_95'] = 1.96*stats_overview_df.loc[ncount,'masschg_mmSLE_std']
            
            stats_overview_df.loc[ncount,'vol_lost_%_med'] = reg_vol_lost_med_norm[-1]
            stats_overview_df.loc[ncount,'vol_lost_%_mean'] = reg_vol_lost_mean_norm[-1]
            stats_overview_df.loc[ncount,'vol_lost_%_std'] = reg_vol_lost_std_norm[-1]
            stats_overview_df.loc[ncount,'vol_lost_%_mad'] = reg_vol_lost_mad_norm[-1]
            stats_overview_df.loc[ncount,'vol_lost_%_90'] = 1.645*stats_overview_df.loc[ncount,'vol_lost_%_std']
            stats_overview_df.loc[ncount,'vol_lost_%_95'] = 1.96*stats_overview_df.loc[ncount,'vol_lost_%_std']
            
            stats_overview_df.loc[ncount,'area_lost_%_med'] = reg_area_lost_med_norm[-1]
            stats_overview_df.loc[ncount,'area_lost_%_mean'] = reg_area_lost_mean_norm[-1]
            stats_overview_df.loc[ncount,'area_lost_%_std'] = reg_area_lost_std_norm[-1]
            stats_overview_df.loc[ncount,'area_lost_%_mad'] = reg_area_lost_mad_norm[-1]
            stats_overview_df.loc[ncount,'count_lost_med'] = reg_glac_lost_med[-1]
            stats_overview_df.loc[ncount,'count_lost_mean'] = reg_glac_lost_mean[-1]
            stats_overview_df.loc[ncount,'count_lost_std'] = reg_glac_lost_std[-1]
            stats_overview_df.loc[ncount,'count_lost_mad'] = reg_glac_lost_mad[-1]
            stats_overview_df.loc[ncount,'count_lost_90'] = 1.645*stats_overview_df.loc[ncount,'count_lost_std']
            stats_overview_df.loc[ncount,'count_lost_95'] = 1.96*stats_overview_df.loc[ncount,'count_lost_std']
            
            stats_overview_df.loc[ncount,'count_lost_%_med'] = reg_glac_lost_med_norm[-1]
            stats_overview_df.loc[ncount,'count_lost_%_mean'] = reg_glac_lost_mean_norm[-1]
            stats_overview_df.loc[ncount,'count_lost_%_std'] = reg_glac_lost_std_norm[-1]
            stats_overview_df.loc[ncount,'count_lost_%_mad'] = reg_glac_lost_mad_norm[-1]
            stats_overview_df.loc[ncount,'count_lost_%_90'] = 1.645*stats_overview_df.loc[ncount,'count_lost_%_std']
            stats_overview_df.loc[ncount,'count_lost_%_95'] = 1.96*stats_overview_df.loc[ncount,'count_lost_%_std']
            
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_med'] = reg_mb_20902100_med
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_mean'] = reg_mb_20902100_mean
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_std'] = reg_mb_20902100_std
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_mad'] = reg_mb_20902100_mad
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_90'] = 1.645*stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_std']
            stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_95'] = 1.96*stats_overview_df.loc[ncount,'mb_2090-2100_mmwea_std']
            
            stats_overview_df.loc[ncount,'mb_max_mmwea_med'] = np.min(reg_mb_med)
            stats_overview_df.loc[ncount,'mb_max_mmwea_mean'] = np.min(reg_mb_mean)
            stats_overview_df.loc[ncount,'mb_max_mmwea_std'] = reg_mb_std[mb_max_idx]
            stats_overview_df.loc[ncount,'mb_max_mmwea_mad'] = reg_mb_mad[mb_max_idx]
            stats_overview_df.loc[ncount,'mb_max_mmwea_90'] = 1.645*stats_overview_df.loc[ncount,'mb_max_mmwea_std']
            stats_overview_df.loc[ncount,'mb_max_mmwea_95'] = 1.96*stats_overview_df.loc[ncount,'mb_max_mmwea_std']
            
            ncount += 1
            
    if 2.7 in deg_groups:
        stats_overview_df.to_csv(csv_fp + 'stats_overview_bydeg_wcop26.csv', index=False)
        assert 1==0, 'stopping to avoid plots with 2.7 degC'
    else:
        stats_overview_df.to_csv(csv_fp + 'stats_overview_bydeg.csv', index=False)
    
    #%%
#    ds_glaciermip = xr.open_dataset(ds_marzeion2020_fn)
#    
#    # Scenarios: 1=2.6, 2=4.5, 4=8.5
#    # Region: 1-19
#    # Time: 2000-2100
#    for reg in regions[:-1]:
#        for scenario in [1,2,4]:
#            
#            print(ds_glaciermip.Mass.sel(Region=reg, Scenario=scenario).values.shape)
#            assert 1==0, 'here'
#    
#    #%%
#    
#    assert 1==0, 'check that it has what we want'
    
    #%% ---- TABLES OF VALUES -----
    mb_comp_cns = ['Region', 'Scenario', 'n_gcms',
                   'melt_2015-2100_gta_med', 'melt_2015-2100_gta_std', 
                   'acc_2015-2100_gta_med', 'acc_2015-2100_gta_std',
                   'refreeze_2015-2100_gta_med', 'refreeze_2015-2100_gta_std',
                   'fa_2015-2100_gta_med', 'fa_2015-2100_gta_std', 
                   'mbtot_2015-2100_gta_med', 'mbtot_2015-2100_gta_std', 
                   
                   'melt_2000-2020_gta_med', 'melt_2000-2020_gta_std', 
                   'acc_2000-2020_gta_med', 'acc_2000-2020_gta_std',
                   'refreeze_2000-2020_gta_med', 'refreeze_2000-2020_gta_std',
                   'fa_2000-2020_gta_med', 'fa_2000-2020_gta_std', 
                   'mbtot_2000-2020_gta_med', 'mbtot_2000-2020_gta_std', 
                   'melt_2020-2040_gta_med', 'melt_2020-2040_gta_std',
                   'acc_2020-2040_gta_med', 'acc_2020-2040_gta_std',
                   'refreeze_2020-2040_gta_med', 'refreeze_2020-2040_gta_std', 
                   'fa_2020-2040_gta_med', 'fa_2020-2040_gta_std',
                   'mbtot_2020-2040_gta_med', 'mbtot_2020-2040_gta_std',
                   'melt_2040-2060_gta_med', 'melt_2040-2060_gta_std', 
                   'acc_2040-2060_gta_med', 'acc_2040-2060_gta_std', 
                   'refreeze_2040-2060_gta_med', 'refreeze_2040-2060_gta_std', 
                   'fa_2040-2060_gta_med', 'fa_2040-2060_gta_std', 
                   'mbtot_2040-2060_gta_med', 'mbtot_2040-2060_gta_std', 
                   'melt_2060-2080_gta_med', 'melt_2060-2080_gta_std',
                   'acc_2060-2080_gta_med', 'acc_2060-2080_gta_std', 
                   'refreeze_2060-2080_gta_med', 'refreeze_2060-2080_gta_std', 
                   'fa_2060-2080_gta_med', 'fa_2060-2080_gta_std', 
                   'mbtot_2060-2080_gta_med', 'mbtot_2060-2080_gta_std', 
                   'melt_2080-2100_gta_med', 'melt_2080-2100_gta_std', 
                   'acc_2080-2100_gta_med', 'acc_2080-2100_gta_std', 
                   'refreeze_2080-2100_gta_med', 'refreeze_2080-2100_gta_std',
                   'fa_2080-2100_gta_med', 'fa_2080-2100_gta_std', 
                   'mbtot_2080-2100_gta_med', 'mbtot_2080-2100_gta_std', 
                   
                   'melt_2000-2020_mwea_med', 'melt_2000-2020_mwea_std', 
                   'acc_2000-2020_mwea_med', 'acc_2000-2020_mwea_std',
                   'refreeze_2000-2020_mwea_med', 'refreeze_2000-2020_mwea_std',
                   'fa_2000-2020_mwea_med', 'fa_2000-2020_mwea_std', 
                   'mbtot_2000-2020_mwea_med', 'mbtot_2000-2020_mwea_std', 
                   'melt_2020-2040_mwea_med', 'melt_2020-2040_mwea_std',
                   'acc_2020-2040_mwea_med', 'acc_2020-2040_mwea_std',
                   'refreeze_2020-2040_mwea_med', 'refreeze_2020-2040_mwea_std', 
                   'fa_2020-2040_mwea_med', 'fa_2020-2040_mwea_std',
                   'mbtot_2020-2040_mwea_med', 'mbtot_2020-2040_mwea_std',
                   'melt_2040-2060_mwea_med', 'melt_2040-2060_mwea_std', 
                   'acc_2040-2060_mwea_med', 'acc_2040-2060_mwea_std', 
                   'refreeze_2040-2060_mwea_med', 'refreeze_2040-2060_mwea_std', 
                   'fa_2040-2060_mwea_med', 'fa_2040-2060_mwea_std', 
                   'mbtot_2040-2060_mwea_med', 'mbtot_2040-2060_mwea_std', 
                   'melt_2060-2080_mwea_med', 'melt_2060-2080_mwea_std',
                   'acc_2060-2080_mwea_med', 'acc_2060-2080_mwea_std', 
                   'refreeze_2060-2080_mwea_med', 'refreeze_2060-2080_mwea_std', 
                   'fa_2060-2080_mwea_med', 'fa_2060-2080_mwea_std', 
                   'mbtot_2060-2080_mwea_med', 'mbtot_2060-2080_mwea_std', 
                   'melt_2080-2100_mwea_med', 'melt_2080-2100_mwea_std', 
                   'acc_2080-2100_mwea_med', 'acc_2080-2100_mwea_std', 
                   'refreeze_2080-2100_mwea_med', 'refreeze_2080-2100_mwea_std',
                   'fa_2080-2100_mwea_med', 'fa_2080-2100_mwea_std', 
                   'mbtot_2080-2100_mwea_med', 'mbtot_2080-2100_mwea_std', 
                        ]
    mb_comp_df = pd.DataFrame(np.zeros((len(regions)*len(deg_groups),len(mb_comp_cns))), columns=mb_comp_cns)
    ncount = 0
    regions_overview = regions
    if 'all' in regions:
        regions_overview = [regions[-1]] + regions[0:-1]
    for nreg, reg in enumerate(regions_overview):
        for deg_group in deg_groups:

            # MASS BALANCE COMPONENTS
            # - these are only meant for monthly and/or relative purposes 
            #   mass balance from volume change should be used for annual changes            
            reg_area_annual_all = ds_multigcm_area_bydeg[reg][deg_group]
            reg_melt_monthly_all = ds_multigcm_melt_bydeg[reg][deg_group]
            reg_acc_monthly_all = ds_multigcm_acc_bydeg[reg][deg_group]
            reg_refreeze_monthly_all = ds_multigcm_refreeze_bydeg[reg][deg_group]
            reg_frontalablation_monthly_all = ds_multigcm_fa_bydeg[reg][deg_group]
            reg_massbaltotal_monthly_all = (reg_acc_monthly_all + reg_refreeze_monthly_all - reg_melt_monthly_all - reg_frontalablation_monthly_all)

            # ----- 2015-2100 statistics
            def fulltime_stats_from_ds_monthly_gta(reg_var_monthly_all):
                reg_var_annual_all_gt = (reg_var_monthly_all * pygem_prms.density_water / 1e12).reshape(reg_var_monthly_all.shape[0],-1,12).sum(axis=2)
                reg_var_fulltime_all_gta = reg_var_annual_all_gt[:,normyear_idx:].mean(1)
                reg_var_fulltime_all_gta_med = np.median(reg_var_fulltime_all_gta)
                reg_var_fulltime_all_gta_std = np.std(reg_var_fulltime_all_gta)
                
                return reg_var_fulltime_all_gta_med, reg_var_fulltime_all_gta_std
        
        
            reg_melt_2015_2100_Gta_med, reg_melt_2015_2100_Gta_std = fulltime_stats_from_ds_monthly_gta(reg_melt_monthly_all)
            reg_acc_2015_2100_Gta_med, reg_acc_2015_2100_Gta_std = fulltime_stats_from_ds_monthly_gta(reg_acc_monthly_all)
            reg_refreeze_2015_2100_Gta_med, reg_refreeze_2015_2100_Gta_std = fulltime_stats_from_ds_monthly_gta(reg_refreeze_monthly_all)
            reg_frontalablation_2015_2100_Gta_med, reg_frontalablation_2015_2100_Gta_std = fulltime_stats_from_ds_monthly_gta(reg_frontalablation_monthly_all)
            reg_massbaltotal_2015_2100_Gta_med, reg_massbaltotal_2015_2100_Gta_std = fulltime_stats_from_ds_monthly_gta(reg_massbaltotal_monthly_all)
            
#            print(reg_melt_2015_2100_Gta_med, reg_melt_2015_2100_Gta_std)
#            print(reg_acc_2015_2100_Gta_med, reg_acc_2015_2100_Gta_std)
#            print(reg_refreeze_2015_2100_Gta_med, reg_refreeze_2015_2100_Gta_std)
#            print(reg_frontalablation_2015_2100_Gta_med, reg_frontalablation_2015_2100_Gta_std)
#            print(reg_massbaltotal_2015_2100_Gta_med, reg_massbaltotal_2015_2100_Gta_std)
            
            # ----- 20-yr period statistics -----
            def period_stats_from_ds_monthly(reg_var_monthly_all, reg_area_annual_all, period_yrs):
                reg_var_annual_all = reg_var_monthly_all.reshape(reg_var_monthly_all.shape[0],-1,12).sum(axis=2)
                reg_var_periods_all = reg_var_annual_all[:,0:100].reshape(reg_var_annual_all.shape[0],-1,period_yrs).sum(2)

                # Convert to mwea
                reg_area_periods_all = reg_area_annual_all[:,0:100].reshape(reg_area_annual_all.shape[0],-1,period_yrs).mean(2)
                reg_var_periods_all_mwea = reg_var_periods_all / reg_area_periods_all / period_yrs
                
                reg_var_periods_all_mwea_med = np.median(reg_var_periods_all_mwea, axis=0)
                reg_var_periods_all_mwea_std = np.std(reg_var_periods_all_mwea, axis=0)
                
                return reg_var_periods_all_mwea_med, reg_var_periods_all_mwea_std

            period_yrs = 20
            reg_melt_periods_mwea_med, reg_melt_periods_mwea_std = (
                    period_stats_from_ds_monthly(reg_melt_monthly_all, reg_area_annual_all, period_yrs))
            reg_acc_periods_mwea_med, reg_acc_periods_mwea_std = (
                    period_stats_from_ds_monthly(reg_acc_monthly_all, reg_area_annual_all, period_yrs))
            reg_refreeze_periods_mwea_med, reg_refreeze_periods_mwea_std = (
                    period_stats_from_ds_monthly(reg_refreeze_monthly_all, reg_area_annual_all, period_yrs))
            reg_frontalablation_periods_mwea_med, reg_frontalablation_periods_mwea_std = (
                    period_stats_from_ds_monthly(reg_frontalablation_monthly_all, reg_area_annual_all, period_yrs))
            reg_massbaltotal_periods_mwea_med, reg_massbaltotal_periods_mwea_std = (
                    period_stats_from_ds_monthly(reg_massbaltotal_monthly_all, reg_area_annual_all, period_yrs))
            
#            print('\n', reg, deg_group)
#            print(reg_massbaltotal_periods_mwea_med)
#            print(reg_acc_periods_mwea_med + reg_refreeze_periods_mwea_med - reg_frontalablation_periods_mwea_med - reg_melt_periods_mwea_med)
            
            # ----- 20-yr period statistics (Gta) -----
            def period_stats_from_ds_monthly_gta(reg_var_monthly_all, period_yrs):
                reg_var_annual_all_gta = (reg_var_monthly_all * pygem_prms.density_water / 1e12).reshape(reg_var_monthly_all.shape[0],-1,12).sum(axis=2)
                reg_var_periods_all_gta = reg_var_annual_all_gta[:,0:100].reshape(reg_var_annual_all_gta.shape[0],-1,period_yrs).sum(2) / period_yrs
    
                reg_var_periods_all_gta_med = np.median(reg_var_periods_all_gta, axis=0)
                reg_var_periods_all_gta_std = np.std(reg_var_periods_all_gta, axis=0)
                
                return reg_var_periods_all_gta_med, reg_var_periods_all_gta_std
            
            reg_melt_periods_gta_med, reg_melt_periods_gta_std = period_stats_from_ds_monthly_gta(reg_melt_monthly_all, period_yrs)
            reg_acc_periods_gta_med, reg_acc_periods_gta_std = period_stats_from_ds_monthly_gta(reg_acc_monthly_all, period_yrs)
            reg_refreeze_periods_gta_med, reg_refreeze_periods_gta_std = period_stats_from_ds_monthly_gta(reg_refreeze_monthly_all, period_yrs)
            reg_frontalablation_periods_gta_med, reg_frontalablation_periods_gta_std = period_stats_from_ds_monthly_gta(reg_frontalablation_monthly_all, period_yrs)
            reg_massbaltotal_periods_gta_med, reg_massbaltotal_periods_gta_std = period_stats_from_ds_monthly_gta(reg_massbaltotal_monthly_all, period_yrs)

            # RECORD STATISTICS
            mb_comp_df.loc[ncount,'Region'] = reg
            mb_comp_df.loc[ncount,'Scenario'] = deg_group
            mb_comp_df.loc[ncount,'n_gcms'] = reg_area_annual_all.shape[0]
            mb_comp_df.loc[ncount,'melt_2015-2100_gta_med'] = reg_melt_2015_2100_Gta_med
            mb_comp_df.loc[ncount,'melt_2015-2100_gta_std'] = reg_melt_2015_2100_Gta_std
            mb_comp_df.loc[ncount,'acc_2015-2100_gta_med'] = reg_acc_2015_2100_Gta_med
            mb_comp_df.loc[ncount,'acc_2015-2100_gta_std'] = reg_acc_2015_2100_Gta_std
            mb_comp_df.loc[ncount,'refreeze_2015-2100_gta_med'] = reg_refreeze_2015_2100_Gta_med
            mb_comp_df.loc[ncount,'refreeze_2015-2100_gta_std'] = reg_refreeze_2015_2100_Gta_std
            mb_comp_df.loc[ncount,'fa_2015-2100_gta_med'] = reg_frontalablation_2015_2100_Gta_med
            mb_comp_df.loc[ncount,'fa_2015-2100_gta_std'] = reg_frontalablation_2015_2100_Gta_std
            mb_comp_df.loc[ncount,'mbtot_2015-2100_gta_med'] = reg_massbaltotal_2015_2100_Gta_med
            mb_comp_df.loc[ncount,'mbtot_2015-2100_gta_std'] = reg_massbaltotal_2015_2100_Gta_std
            for nperiod, time_str in enumerate(['2000-2020', '2020-2040','2040-2060','2060-2080','2080-2100']):
                mb_comp_df.loc[ncount,'melt_' + time_str + '_mwea_med'] = reg_melt_periods_mwea_med[nperiod]
                mb_comp_df.loc[ncount,'acc_' + time_str + '_mwea_med'] = reg_acc_periods_mwea_med[nperiod]
                mb_comp_df.loc[ncount,'refreeze_' + time_str + '_mwea_med'] = reg_refreeze_periods_mwea_med[nperiod]
                mb_comp_df.loc[ncount,'fa_' + time_str + '_mwea_med'] = reg_frontalablation_periods_mwea_med[nperiod]
                mb_comp_df.loc[ncount,'mbtot_' + time_str + '_mwea_med'] = reg_massbaltotal_periods_mwea_med[nperiod]
                
                mb_comp_df.loc[ncount,'melt_' + time_str + '_mwea_std'] = reg_melt_periods_mwea_std[nperiod]
                mb_comp_df.loc[ncount,'acc_' + time_str + '_mwea_std'] = reg_acc_periods_mwea_std[nperiod]
                mb_comp_df.loc[ncount,'refreeze_' + time_str + '_mwea_std'] = reg_refreeze_periods_mwea_std[nperiod]
                mb_comp_df.loc[ncount,'fa_' + time_str + '_mwea_std'] = reg_frontalablation_periods_mwea_std[nperiod]
                mb_comp_df.loc[ncount,'mbtot_' + time_str + '_mwea_std'] = reg_massbaltotal_periods_mwea_std[nperiod]
                
                mb_comp_df.loc[ncount,'melt_' + time_str + '_gta_med'] = reg_melt_periods_gta_med[nperiod]
                mb_comp_df.loc[ncount,'acc_' + time_str + '_gta_med'] = reg_acc_periods_gta_med[nperiod]
                mb_comp_df.loc[ncount,'refreeze_' + time_str + '_gta_med'] = reg_refreeze_periods_gta_med[nperiod]
                mb_comp_df.loc[ncount,'fa_' + time_str + '_gta_med'] = reg_frontalablation_periods_gta_med[nperiod]
                mb_comp_df.loc[ncount,'mbtot_' + time_str + '_gta_med'] = reg_massbaltotal_periods_gta_med[nperiod]
                
                mb_comp_df.loc[ncount,'melt_' + time_str + '_gta_std'] = reg_melt_periods_gta_std[nperiod]
                mb_comp_df.loc[ncount,'acc_' + time_str + '_gta_std'] = reg_acc_periods_gta_std[nperiod]
                mb_comp_df.loc[ncount,'refreeze_' + time_str + '_gta_std'] = reg_refreeze_periods_gta_std[nperiod]
                mb_comp_df.loc[ncount,'fa_' + time_str + '_gta_std'] = reg_frontalablation_periods_gta_std[nperiod]
                mb_comp_df.loc[ncount,'mbtot_' + time_str + '_gta_std'] = reg_massbaltotal_periods_gta_std[nperiod]
            
            ncount += 1
            
    mb_comp_df['%_loss_FA'] = mb_comp_df['fa_2015-2100_gta_med'] / (mb_comp_df['melt_2015-2100_gta_med'] + mb_comp_df['fa_2015-2100_gta_med']) * 100
#    mb_comp_df['%_loss_FA_std'] = mb_comp_df['fa_2015-2100_gta_std'] / (mb_comp_df['melt_2015-2100_gta_med'] + mb_comp_df['fa_2015-2100_gta_med']) * 100
    mb_comp_df.to_csv(csv_fp + 'mb_comp_stats_bydeg.csv', index=False)
    
    #%% ----- FIGURE: GLOBAL COMBINED -----
    add_rgi_glaciers = True
    add_rgi_regions = True
    
    class MidpointNormalize(mpl.colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
    
        def __call__(self, value, clip=None):
            # Note that I'm ignoring clipping and other edge cases here.
            result, is_scalar = self.process_value(value)
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)

    rgi_reg_fig_dict = {'all':'Global',
                        1:'Alaska',
                        2:'W Canada\n& US',
                        3:'Arctic Canada\nNorth',
                        4:'Arctic Canada\nSouth',
                        5:'Greenland Periphery',
                        6:'Iceland',
                        7:'Svalbard',
                        8:'Scandinavia',
                        9:'Russian Arctic',
                        10:'North Asia',
                        11:'Central Europe',
                        12:'Caucasus &\nMiddle East',
                        13:'Central Asia',
                        14:'South Asia\nWest',
                        15:'South Asia\nEast',
                        16:'Low Latitudes',
                        17:'Southern\nAndes',
                        18:'New Zealand',
                        19:'Antarctic & Subantarctic'
                        }
    
#    rcp_colordict = {'ssp119':'#76B8E5', 'ssp126':'#76B8E5', 'ssp245':'#F1EA8A', 'ssp370':'#F47A20', 'ssp585':'#ED2024'}
#    rcp_namedict = {'ssp119':'SSP1-1.9',
#                    'ssp126':'SSP1-2.6',
#                    'ssp245':'SSP2-4.5',
#                    'ssp370':'SSP3-7.0',
#                    'ssp585':'SSP5-8.5'}

    pie_scenarios = [2]
    for pie_scenario in pie_scenarios:

        fig = plt.figure()
        
        # Add background image
        ax_background = fig.add_axes([0,0.15,1,0.7], projection=ccrs.Robinson())
        ax_background.patch.set_facecolor('lightblue')
        ax_background.get_yaxis().set_visible(False)
        ax_background.get_xaxis().set_visible(False)
    #    ax_background.coastlines(color='white')
        ax_background.add_feature(cartopy.feature.LAND, color='white')
        
        # Add global boundary
        ax_global_patch = fig.add_axes([0.08,0.145,0.19,0.38], facecolor='lightblue')
        ax_global_patch.get_yaxis().set_visible(False)
        ax_global_patch.get_xaxis().set_visible(False)
        
        # Add RGI glacier outlines
        if add_rgi_glaciers:
            shape_feature = ShapelyFeature(Reader(rgi_shp_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='indigo',linewidth=0.35,edgecolor='indigo')
            ax_background.add_feature(shape_feature)
            
        if add_rgi_regions:
            shape_feature = ShapelyFeature(Reader(rgi_regions_fn).geometries(), ccrs.Robinson(),alpha=1,facecolor='None',linewidth=0.35,edgecolor='k')
            ax_background.add_feature(shape_feature)
            
        
        
        regions_ordered = ['all',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        reg_pie_sizes = []
        for reg in regions_ordered:
            reg_slr_cum_pie = stats_overview_df.loc[(stats_overview_df.Region==reg) & (stats_overview_df.Scenario==pie_scenario),'slr_mmSLE_med'].values[0]
            
            print(reg, np.round(reg_slr_cum_pie,2))
            
            pie_size_min = 0.07
            if reg_slr_cum_pie > 80:
                pie_size = 0.33
            elif reg_slr_cum_pie > 25:
                pie_size = 0.2
            elif reg_slr_cum_pie < 1:
                pie_size = pie_size_min
            else:
                pie_size = pie_size_min + (reg_slr_cum_pie - 1) / (25-1) * (0.2 - pie_size_min)
            reg_pie_sizes.append(pie_size)
        
        ax0 = fig.add_axes([0.125,0.18,0.1,0.04], facecolor='none')
        ax1 = fig.add_axes([0.09,0.73,0.1,0.04], facecolor='none')
        ax2 = fig.add_axes([0.13,0.59,0.1,0.04], facecolor='none')
        ax3 = fig.add_axes([0.21,0.875,0.1,0.04], facecolor='none')
        ax4 = fig.add_axes([0.265,0.60,0.1,0.04], facecolor='none')
        ax5 = fig.add_axes([0.34,0.875,0.1,0.04], facecolor='none')
        ax6 = fig.add_axes([0.37,0.64,0.1,0.04], facecolor='none')
        ax7 = fig.add_axes([0.465,0.875,0.1,0.04], facecolor='none')
        ax8 = fig.add_axes([0.573,0.875,0.1,0.04], facecolor='none')
        ax9 = fig.add_axes([0.68,0.875,0.1,0.04], facecolor='none')
        ax10 = fig.add_axes([0.8,0.78,0.1,0.04], facecolor='none')
        ax11 = fig.add_axes([0.44,0.55,0.1,0.04], facecolor='none')
        ax12 = fig.add_axes([0.55,0.535,0.1,0.04], facecolor='none')
        ax13 = fig.add_axes([0.8,0.62,0.1,0.04], facecolor='none')
        ax14 = fig.add_axes([0.655,0.495,0.1,0.04], facecolor='none')
        ax15 = fig.add_axes([0.77,0.47,0.1,0.04], facecolor='none')
        ax16 = fig.add_axes([0.445,0.40,0.1,0.04], facecolor='none')
        ax17 = fig.add_axes([0.36,0.295,0.1,0.04], facecolor='none')
        ax18 = fig.add_axes([0.73,0.3,0.1,0.04], facecolor='none')
        ax19 = fig.add_axes([0.55,0.19,0.1,0.04], facecolor='none')

        # Pie charts
        ax0b = fig.add_axes([0.01,0.205,reg_pie_sizes[0],reg_pie_sizes[0]], facecolor='none')
        ax1b = fig.add_axes([0.052,0.762,reg_pie_sizes[1],reg_pie_sizes[1]], facecolor='none')
        ax2b = fig.add_axes([0.143,0.628,reg_pie_sizes[2],reg_pie_sizes[2]], facecolor='none')
        ax3b = fig.add_axes([0.205,0.91,reg_pie_sizes[3],reg_pie_sizes[3]], facecolor='none')
        ax4b = fig.add_axes([0.26,0.635,reg_pie_sizes[4],reg_pie_sizes[4]], facecolor='none')
        ax5b = fig.add_axes([0.325,0.91,reg_pie_sizes[5],reg_pie_sizes[5]], facecolor='none')
        ax6b = fig.add_axes([0.38,0.677,reg_pie_sizes[6],reg_pie_sizes[6]], facecolor='none')
        ax7b = fig.add_axes([0.47,0.91,reg_pie_sizes[7],reg_pie_sizes[7]], facecolor='none')
        ax8b = fig.add_axes([0.586,0.912,reg_pie_sizes[8],reg_pie_sizes[8]], facecolor='none')
        ax9b = fig.add_axes([0.68,0.91,reg_pie_sizes[9],reg_pie_sizes[9]], facecolor='none')
        ax10b = fig.add_axes([0.818,0.817,reg_pie_sizes[10],reg_pie_sizes[10]], facecolor='none')
        ax11b = fig.add_axes([0.455,0.587,reg_pie_sizes[11],reg_pie_sizes[11]], facecolor='none')
        ax12b = fig.add_axes([0.566,0.573,reg_pie_sizes[12],reg_pie_sizes[12]], facecolor='none')
        ax13b = fig.add_axes([0.81,0.656,reg_pie_sizes[13],reg_pie_sizes[13]], facecolor='none')
        ax14b = fig.add_axes([0.668,0.532,reg_pie_sizes[14],reg_pie_sizes[14]], facecolor='none')
        ax15b = fig.add_axes([0.783,0.508,reg_pie_sizes[15],reg_pie_sizes[15]], facecolor='none')
        ax16b = fig.add_axes([0.46,0.438,reg_pie_sizes[16],reg_pie_sizes[16]], facecolor='none')
        ax17b = fig.add_axes([0.365,0.331,reg_pie_sizes[17],reg_pie_sizes[17]], facecolor='none')
        ax18b = fig.add_axes([0.747,0.337,reg_pie_sizes[18],reg_pie_sizes[18]], facecolor='none')
        ax19b = fig.add_axes([0.54,0.225,reg_pie_sizes[19],reg_pie_sizes[19]], facecolor='none')
        
        # ----- Heat map of specific mass balance (2015 - 2100) -----
        for nax, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10,
                                  ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19]):
            
            reg = regions_ordered[nax]
            
    #        cmap = 'RdYlBu'
            cmap = 'Greys_r'
    #        cmap = 'YlOrRd'
            norm_values = [-2.5,-1.5,-0.25]
            norm = MidpointNormalize(midpoint=norm_values[1], vmin=norm_values[0], vmax=norm_values[2])
            
            mesh = None
            for deg_group in [1.5, 3]:
            
                # Median and absolute median deviation
                reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
                reg_mass = reg_vol * pygem_prms.density_ice
                reg_area = ds_multigcm_area_bydeg[reg][deg_group]
        
                # Specific mass change rate
                reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
                reg_mb_med = np.median(reg_mb, axis=0)
                reg_mb_mad = median_abs_deviation(reg_mb, axis=0)
                
                if mesh is None:
                    mesh = reg_mb_med[np.newaxis,normyear_idx:]
                else:
                    mesh = np.concatenate((mesh, reg_mb_med[np.newaxis,normyear_idx:]), axis=0)
                
            ax.imshow(mesh/1000, aspect='auto', cmap=cmap, norm=norm, interpolation='none')
            ax.hlines(0.5,0,mesh.shape[1]-1, color='k', linewidth=0.5, zorder=2)
            ax.get_yaxis().set_visible(False)
    #        ax.get_xaxis().set_visible(False)
            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.tick_params(axis='both', which='major', direction='inout', right=True, top=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True, top=True)
            ax.get_xaxis().set_ticks([])
            
            # Add region label
            ax.text(0.5, -0.14, rgi_reg_fig_dict[reg], size=8, horizontalalignment='center', 
                verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='k', pad=2))
            
        # ----- Pie Chart of Volume Remaining by end of century -----
        wedge_size = 0.15
        for nax, ax in enumerate([ax0b, ax1b, ax2b, ax3b, ax4b, ax5b, ax6b, ax7b, ax8b, ax9b, ax10b,
                                  ax11b, ax12b, ax13b, ax14b, ax15b, ax16b, ax17b, ax18b, ax19b]):
            
            reg = regions_ordered[nax]
            
            ssp_vol_remaining_pies = []
            ssp_pie_radius = 1
            for ndeg_group, deg_group in enumerate(deg_groups[::-1]):
                
                # Median and absolute median deviation
                reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
                reg_vol_med = np.median(reg_vol, axis=0)
                reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
                
                reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
                reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
    
                ssp_vol_remaining_pies.append(reg_vol_med_norm[-1])
                
                # Nested Pie Charts
#                ssp_pies = [1-ssp_vol_remaining_pies[ndeg_group], ssp_vol_remaining_pies[ndeg_group]]
#                ssp_pie_colors = ['lightgray',temp_colordict[deg_group]]
#                pie_slices, pie_labels = ax.pie(ssp_pies, radius=ssp_pie_radius, 
#                                                counterclock=False, startangle=90, colors=ssp_pie_colors,
#                                                wedgeprops=dict(width=wedge_size))
#                ssp_pie_radius = ssp_pie_radius - wedge_size
                
                # Plot the volume loss (don't plot if want to leave empty)
                ssp_pies = [1-ssp_vol_remaining_pies[ndeg_group]]
                ssp_pie_colors = ['lightgray']
                pie_slices, pie_labels = ax.pie(ssp_pies, radius=ssp_pie_radius,
                                                counterclock=False, startangle=90, colors=ssp_pie_colors,
                                                wedgeprops=dict(width=wedge_size))
                # Plot the volume remaining
                ssp_pies = [ssp_vol_remaining_pies[ndeg_group]]
                ssp_pie_colors = [temp_colordict[deg_group]]
                pie_slices, pie_labels = ax.pie(ssp_pies, radius=ssp_pie_radius, 
                                                counterclock=True, startangle=90, colors=ssp_pie_colors,
                                                wedgeprops=dict(width=wedge_size, linewidth=0.2, edgecolor='k'))
                
                ssp_pie_radius = ssp_pie_radius - wedge_size
                
            ssp_pie_radius_fill = 1 - wedge_size*len(deg_groups)
            wedge_size_fill = ssp_pie_radius_fill
            ssp_pies, ssp_pie_colors = [1], ['lightgray']
            pie_slices, pie_labels = ax.pie(ssp_pies, radius=ssp_pie_radius_fill,
                                            counterclock=False, startangle=90, colors=ssp_pie_colors,
                                            wedgeprops=dict(width=wedge_size_fill))
            ax.axis('equal')
            
            # SLR 
            reg_slr_cum_pie = stats_overview_df.loc[(stats_overview_df.Region==reg) & (stats_overview_df.Scenario==pie_scenario),'slr_mmSLE_med'].values[0]
            if reg_slr_cum_pie > 1:
                reg_slr_str = str(int(np.round(reg_slr_cum_pie)))
            else:
                reg_slr_str = ''
            if reg in ['all']:
                reg_slr_str += '\nmm SLE'
            ax.text(0.5, 0.5, reg_slr_str, size=10, color='k', horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes)
            
            # Add outer edge by adding new circle with desired properties
            center = pie_slices[0].center
            r = 1
            circle = mpl.patches.Circle(center, r, fill=False, edgecolor="k", linewidth=0.5)
            ax.add_patch(circle)
            
            
        # ----- LEGEND -----
        # Sized circles
#        ax_background.text(0.66,-0.2,'Sea level rise from\n2015-2100 for ' + rcp_namedict[pie_scenario] + '\n(mm SLE)', size=10, 
#                           horizontalalignment='center', verticalalignment='top', transform=ax_background.transAxes)
#        ax_circle1 = fig.add_axes([0.56,0.06,0.05,0.05], facecolor='none')
#        pie_slices, pie_labels = ax_circle1.pie([1], counterclock=False, startangle=90, colors=['white'],
#                                                wedgeprops=dict(edgecolor='k', linewidth=0.5))
#        ax_circle1.axis('equal')
#        ax_circle1.text(0.5,0.5,'1', size=8, horizontalalignment='center',  verticalalignment='center',
#                        transform=ax_circle1.transAxes)
#        
#        ax_circle2 = fig.add_axes([0.61,0.01,0.10625,0.10625], facecolor='none')
#        pie_slices, pie_labels = ax_circle2.pie([1], counterclock=False, startangle=90, colors=['white'],
#                                                wedgeprops=dict(edgecolor='k', linewidth=0.5))
#        ax_circle2.axis('equal')
#        ax_circle2.text(0.5,0.5,'10', size=8, horizontalalignment='center', verticalalignment='center', 
#                        transform=ax_circle2.transAxes)
    
        ax_circle3 = fig.add_axes([0.6,-0.02,0.13,0.13], facecolor='none')
#        ax_circle3 = fig.add_axes([0.68,-0.08,0.2,0.2], facecolor='none')
#        ax_circle3.text(0.5,0.5,'25', size=8, horizontalalignment='center', verticalalignment='center', transform=ax_circle3.transAxes)
        ssp_vol_remaining_pies = [0.6, 0.65, 0.7, 0.75, 0.8]
        ssp_pie_radius = 1
        for ndeg_group, deg_group in enumerate(deg_groups[::-1]):
            # Nested Pie Charts
            ssp_pies = [1-ssp_vol_remaining_pies[ndeg_group], ssp_vol_remaining_pies[ndeg_group]]
            ssp_pie_colors = ['lightgray',temp_colordict[deg_group]]
            pie_slices, pie_labels = ax_circle3.pie(ssp_pies, radius=ssp_pie_radius, 
                                                    counterclock=False, startangle=90, colors=ssp_pie_colors,
                                                    wedgeprops=dict(width=wedge_size))
            ssp_pie_radius = ssp_pie_radius - wedge_size
            
        ssp_pie_radius_fill = 1 - wedge_size*len(deg_groups)
        wedge_size_fill = ssp_pie_radius_fill
        ssp_pies, ssp_pie_colors = [1], ['lightgray']
        pie_slices, pie_labels = ax_circle3.pie(ssp_pies, radius=ssp_pie_radius_fill, 
                                                counterclock=False, startangle=90, colors=ssp_pie_colors,
                                                wedgeprops=dict(width=wedge_size_fill))
        ax_circle3.axis('equal')
    
        center = pie_slices[0].center
        r = 1
        circle = mpl.patches.Circle(center, r, fill=False, edgecolor="k", linewidth=1)
        ax_circle3.add_patch(circle)
#        ax_circle3.text(0.77,0.6,'1.5\u00B0C', color=temp_colordict[1.5], size=8, 
#                        horizontalalignment='left', transform=ax_circle3.transAxes)
#        ax_circle3.text(0.79,0.40,'2\u00B0C', color=temp_colordict[2], size=8, 
#                        horizontalalignment='left', transform=ax_circle3.transAxes)
#        ax_circle3.text(0.76,0.25,'3\u00B0C', color=temp_colordict[3], size=8, 
#                        horizontalalignment='left', transform=ax_circle3.transAxes)
#        ax_circle3.text(0.7,0.09,'4\u00B0C', color=temp_colordict[4], size=8, 
#                        horizontalalignment='left', transform=ax_circle3.transAxes)
##        ax_circle3.text(0.60,-0.09,'5\u00B0C', color=temp_colordict[5], size=8, 
##                        horizontalalignment='left', transform=ax_circle3.transAxes)
        
        ax_circle3.text(0.79,0.4,'1.5\u00B0C', color=temp_colordict[1.5], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        ax_circle3.text(0.76,0.25,'2\u00B0C', color=temp_colordict[2], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        ax_circle3.text(0.72,0.1,'3\u00B0C', color=temp_colordict[3], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        ax_circle3.text(0.62,-0.07,'4\u00B0C', color=temp_colordict[4], size=8, 
                        horizontalalignment='left', transform=ax_circle3.transAxes)
        
        
        ax_background.text(0.71,-0.05,'Mass at 2100 (rel. to 2015)', size=10, 
                           horizontalalignment='center', transform=ax_background.transAxes)
    
        # Heat maps
        ax_background.text(0.31,-0.05,'Annual mass balance (m w.e.)', size=10, 
                           horizontalalignment='center', transform=ax_background.transAxes)
        ax_heatmap = fig.add_axes([0.17,0.03,0.1,0.06], facecolor='none')
        ax_heatmap.hlines(0.5,2015,2100, color='k', linewidth=0.5, zorder=2)
        ax_heatmap.set_ylim(0,1)
        ax_heatmap.set_xlim(2015,2100)
        ax_heatmap.get_yaxis().set_visible(False)
        ax_heatmap.xaxis.set_major_locator(MultipleLocator(40))
        ax_heatmap.xaxis.set_minor_locator(MultipleLocator(10))
        ax_heatmap.set_xticks(ticks=[2050, 2100])
        ax_heatmap.tick_params(axis='both', which='major', direction='inout', right=True, top=True)
        ax_heatmap.tick_params(axis='both', which='minor', direction='in', right=True, top=True)
        ax_heatmap.text(0.5,0.71,'1.5\u00B0C', size=8, 
                        horizontalalignment='center', verticalalignment='center', transform=ax_heatmap.transAxes)
        ax_heatmap.text(0.5,0.21,'3\u00B0C', size=8, 
                        horizontalalignment='center', verticalalignment='center', transform=ax_heatmap.transAxes)
        
        # Heat map colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax = plt.axes([0.3, 0.06, 0.22, 0.015])
        cbar = plt.colorbar(sm, ax=ax, cax=cax, orientation='horizontal', extend='both')
        cax.xaxis.set_ticks_position('bottom')
        cax.xaxis.set_tick_params(pad=2)
        cbar.ax.tick_params(labelsize=8)
        
        labels = []
        for n,label in enumerate(cax.xaxis.get_ticklabels()):
            label_str = str(label.get_text())
            labels.append(label_str.split('.')[0] + '.' + label_str.split('.')[1][0])
        cbar.ax.set_xticklabels(labels)

        for n, label in enumerate(cax.xaxis.get_ticklabels()):
            print(n, label)
            if n%2 != 0:
                label.set_visible(False)
#        ax_background.text(0.5, -0.12, 'Mass balance (m w.e. yr$^{-1}$)', size=10, horizontalalignment='center', 
#                           verticalalignment='center', transform=ax_background.transAxes)

        # Save figure
        fig_fn = ('map_regional_mb_and_volremain_' + str(pie_scenario) + 'degC.png')
        fig.set_size_inches(8.5,5)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
        

    #%% ----- OVERVIEW FIGURE with Temporal variations -----
    reg = 'all'
    fig, ax = plt.subplots(5, 2, squeeze=False, sharex=False, sharey=False, 
                           gridspec_kw = {'wspace':0.2, 'hspace':0.25})
        
    glac_count_total = np.median(temp_dev_df['Glac_count-all'])
    ncount_by_deg = []
    for scenario in rcps:
        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
        
        if scenario.startswith('ssp'):
            marker = 'o'
        else:
            marker = '^'
        # ----- Temperature sensitivity plots -----
        marker_width = 0.5
        # Volume
        ax[0,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Vol_2100_%-' + str(reg)], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # Area
        ax[1,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['Area_2100_%-' + str(reg)], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # Number
        ax[2,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], 
                     100 - temp_dev_df_subset['Glac_lost_2100-all'] / temp_dev_df_subset['Glac_count-all'] * 100, 
                     linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # SLR
        ax[3,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-all'], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # Specific mass balance
        ax[4,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['mb_mwea-all'] / 1000, 
                     linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        
    # ----- Time series by degree plots -----
    for deg_group in deg_groups:
        
        # Volume
        reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
        reg_vol_med = np.median(reg_vol, axis=0)
        reg_vol_std = np.std(reg_vol, axis=0)
        
        reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx] * 100
        reg_vol_std_norm = reg_vol_std / reg_vol_med[normyear_idx] * 100
        
        ax[0,0].plot(years, reg_vol_med_norm, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        
        if deg_group in temps_plot_mad:
            ax[0,0].fill_between(years, 
                                 reg_vol_med_norm + 1.96*reg_vol_std_norm, 
                                 reg_vol_med_norm - 1.96*reg_vol_std_norm, 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
        # Area
        reg_area = ds_multigcm_area_bydeg[reg][deg_group]
        reg_area_med = np.median(reg_area, axis=0)
        reg_area_std = np.std(reg_area, axis=0)
        
        reg_area_med_norm = reg_area_med / reg_area_med[normyear_idx] * 100
        reg_area_std_norm = reg_area_std / reg_area_med[normyear_idx] * 100
        
        ax[1,0].plot(years, reg_area_med_norm, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        
        if deg_group in temps_plot_mad:
            ax[1,0].fill_between(years, 
                                 reg_area_med_norm + 1.96*reg_area_std_norm, 
                                 reg_area_med_norm - 1.96*reg_area_std_norm, 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
        # Glaciers remaining
        reg_glac_lost = ds_multigcm_glac_lost_bydeg[reg][deg_group]
        reg_glac_lost_med = np.median(reg_glac_lost, axis=0)
        reg_glac_lost_std = np.std(reg_glac_lost, axis=0)
        
        ax[2,0].plot(years, (glac_count_total - reg_glac_lost_med) / glac_count_total * 100, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        
        if deg_group in temps_plot_mad:
            ax[2,0].fill_between(years, 
                                 (glac_count_total - reg_glac_lost_med + 1.96*reg_glac_lost_std) / glac_count_total * 100, 
                                 (glac_count_total - reg_glac_lost_med - 1.96*reg_glac_lost_std) / glac_count_total * 100, 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
        # SLR
        reg_vol_bsl = ds_multigcm_vol_bydeg_bsl[reg][deg_group]

        # Cumulative Sea-level change [mm SLE]
        #  - accounts for water from glaciers replacing the ice that is below sea level as well
        #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
        reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                   (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
#            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water) / pygem_prms.area_ocean * 1000))
        reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
        
        reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

        reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
        reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
        
        ax[3,0].plot(years[0:-1], reg_slr_cum_avg, color=temp_colordict[deg_group], linestyle='-', 
                linewidth=1, zorder=4, label=deg_group)
        if deg_group in temps_plot_mad:
            ax[3,0].fill_between(years[0:-1], 
                                 (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
                                 (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
            
        # Mass balance
        reg_mass = reg_vol * pygem_prms.density_ice
        reg_area = ds_multigcm_area_bydeg[reg][deg_group]

        # Specific mass change rate
        reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
        reg_mb_med = np.median(reg_mb, axis=0)
        reg_mb_std = np.std(reg_mb, axis=0)
        
        ax[4,0].plot(years[0:-1], reg_mb_med / 1000, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        if deg_group in temps_plot_mad:
            ax[4,0].fill_between(years[0:-1], 
                                (reg_mb_med + 1.96*reg_mb_std) / 1000, 
                                (reg_mb_med - 1.96*reg_mb_std) / 1000,
                                alpha=0.2, facecolor=temp_colordict[deg_group], label=None)
            
        ncount_by_deg.append(reg_vol.shape[0])
                

        
#    ax[0,1].set_ylabel('Mass\nat 2100\n(rel. to 2015)', size=11)
#    ax[1,1].set_ylabel('Area\nat 2100\n(rel. to 2015)', size=11)
#    ax[2,1].set_ylabel('Sea level rise\nat 2100\n(mm SLE)', size=11)
#    ax[3,1].set_ylabel('Glaciers lost\nat 2100\n(10$^{3}$)', size=11)
    
    ax[0,0].set_ylabel('Mass\n(%)', size=12)
    ax[1,0].set_ylabel('Area\n(%)', size=12)
    ax[2,0].set_ylabel('Glaciers\n(%)', size=12)
    ax[3,0].set_ylabel('SLR\n(mm SLE)', size=12)
#    ax[4,0].set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)', size=12)
    ax[4,0].set_ylabel('$\Delta$M/$\Delta$t\n(m w.e. yr$^{-1}$)', size=12)
#    ax[4,0].set_ylabel('$\Delta$M/\n(m w.e. yr$^{-1}$)', size=12)
        
#    ax[4,1].set_xlabel('Global temperature change ($^\circ$C)', size=11)
    ax[4,1].set_xlabel('Global temperature change (' + u'\N{DEGREE SIGN}' + 'C' + ')', size=11)
    ax[4,0].set_xlabel('Year', size=11)
    
    
   
    # X axes
    ax[0,0].set_xlim(normyear, endyear_plot)
    ax[0,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[0,0].set_xticks([2050,2100])
    ax[1,0].set_xlim(normyear, endyear_plot)
    ax[1,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[1,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[1,0].set_xticks([2050,2100])
    ax[2,0].set_xlim(normyear, endyear_plot)
    ax[2,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[2,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[2,0].set_xticks([2050,2100])
    ax[3,0].set_xlim(normyear, endyear_plot)
    ax[3,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[3,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[3,0].set_xticks([2050,2100])
    ax[4,0].set_xlim(normyear, endyear_plot)
    ax[4,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[4,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[4,0].set_xticks([2050,2100])
    
    ax[0,1].set_xlim(0,5)
    ax[0,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[0,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[1,1].set_xlim(0,5)
    ax[1,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[1,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[2,1].set_xlim(0,5)
    ax[2,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[2,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[3,1].set_xlim(0,5)
    ax[3,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[3,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[4,1].set_xlim(0,5)
    ax[4,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[4,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax[0,0].axes.xaxis.set_ticklabels([])
    ax[1,0].axes.xaxis.set_ticklabels([])
    ax[2,0].axes.xaxis.set_ticklabels([])
    ax[3,0].axes.xaxis.set_ticklabels([])
    ax[0,1].axes.xaxis.set_ticklabels([])
    ax[1,1].axes.xaxis.set_ticklabels([])
    ax[2,1].axes.xaxis.set_ticklabels([])
    ax[3,1].axes.xaxis.set_ticklabels([])
    
    # Y axes
    ax[0,0].set_ylim(0,100)
    ax[0,0].yaxis.set_major_locator(MultipleLocator(50))
    ax[0,0].yaxis.set_minor_locator(MultipleLocator(10))
    ax[1,0].set_ylim(0,100)
    ax[1,0].yaxis.set_major_locator(MultipleLocator(50))
    ax[1,0].yaxis.set_minor_locator(MultipleLocator(10))
    ax[2,0].set_ylim(0,100)
    ax[2,0].yaxis.set_major_locator(MultipleLocator(50))
    ax[2,0].yaxis.set_minor_locator(MultipleLocator(10))
    ax[3,0].set_ylim(0,230)
    ax[3,0].yaxis.set_major_locator(MultipleLocator(100))
    ax[3,0].yaxis.set_minor_locator(MultipleLocator(20))
    ax[4,0].set_ylim(-4.5,0)
    ax[4,0].yaxis.set_major_locator(MultipleLocator(2))
    ax[4,0].yaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax[0,1].set_ylim(0,100)
    ax[0,1].yaxis.set_major_locator(MultipleLocator(50))
    ax[0,1].yaxis.set_minor_locator(MultipleLocator(10))
    ax[1,1].set_ylim(0,100)
    ax[1,1].yaxis.set_major_locator(MultipleLocator(50))
    ax[1,1].yaxis.set_minor_locator(MultipleLocator(10))
    ax[2,1].set_ylim(0,100)
    ax[2,1].yaxis.set_major_locator(MultipleLocator(50))
    ax[2,1].yaxis.set_minor_locator(MultipleLocator(10))
    ax[3,1].set_ylim(0,230)
    ax[3,1].yaxis.set_major_locator(MultipleLocator(100))
    ax[3,1].yaxis.set_minor_locator(MultipleLocator(20))
    ax[4,1].set_ylim(-4.5,0)
    ax[4,1].yaxis.set_major_locator(MultipleLocator(2))
    ax[4,1].yaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Tick parameters
#    ax[4,0].tick_params(axis='x', labelsize=12)
    ax[0,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[0,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[1,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[1,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[2,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[2,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[3,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[3,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[4,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[4,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[0,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[0,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[1,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[1,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[2,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[2,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[3,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[3,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[4,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[4,1].tick_params(axis='both', which='minor', direction='in', right=True)
    
    # Figure labels
    ax[0,0].text(0.02, 0.98, 'A', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[0,0].transAxes)
    ax[0,1].text(0.02, 0.98, 'B', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[0,1].transAxes)
    ax[1,0].text(0.02, 0.98, 'C', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[1,0].transAxes)
    ax[1,1].text(0.02, 0.98, 'D', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[1,1].transAxes)
    ax[2,0].text(0.02, 0.98, 'E', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[2,0].transAxes)
    ax[2,1].text(0.02, 0.98, 'F', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[2,1].transAxes)
    ax[3,0].text(0.02, 0.98, 'G', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[3,0].transAxes)
    ax[3,1].text(0.02, 0.98, 'H', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[3,1].transAxes)
    ax[4,0].text(0.02, 0.98, 'I', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[4,0].transAxes)
    ax[4,1].text(0.02, 0.98, 'J', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[4,1].transAxes)
    
#    ax[0,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[0,1].transAxes)
#    ax[1,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[1,1].transAxes)
#    ax[2,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[2,1].transAxes)
#    ax[3,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[3,1].transAxes)
    
    # Legends
    ax[0,1].legend(labels=['SSP1-1.9','SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
                  loc=(0.05,0.05), 
                  fontsize=6, labelspacing=0.25, handlelength=1, handletextpad=0.1,
                  borderpad=0.1, ncol=3, columnspacing=0.25, frameon=True)
    
#    labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (' + r'$\pm$' + format(deg_groups_bnds[x],'0.2f') + ')' 
#            for x in np.arange(len(deg_groups))]
    labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (n=' + str(ncount_by_deg[x]) + ')'
            for x in np.arange(len(deg_groups))]
    ax[0,0].legend(loc=(0.05, 0.05), labels=labels, fontsize=8, ncol=1, columnspacing=0.5, labelspacing=0.05, 
                   handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)
    
    # Save figure
    fig_fn = 'Temp_Overview_wtime-global-remaining.png'
    fig.set_size_inches(6,6)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- OVERVIEW FIGURE with Temporal variations -----
    reg = 'all'
    fig, ax = plt.subplots(5, 2, squeeze=False, sharex=False, sharey=False, 
                           gridspec_kw = {'wspace':0.2, 'hspace':0.2})
        
    glac_count_total = np.median(temp_dev_df['Glac_count-all'])
    ncount_by_deg = []
    for scenario in rcps:
        temp_dev_df_subset = temp_dev_df.loc[temp_dev_df['Scenario'] == scenario]
        
        if scenario.startswith('ssp'):
            marker = 'o'
        else:
            marker = '^'
        # ----- Temperature sensitivity plots -----
        marker_width = 0.5
        # Volume
        ax[0,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], 100-temp_dev_df_subset['Vol_2100_%-' + str(reg)], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # Area
        ax[1,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], 100-temp_dev_df_subset['Area_2100_%-' + str(reg)], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # SLR
        ax[2,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['SLR_mmSLE-all'], 
                    linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # Number
        ax[3,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], 
                     temp_dev_df_subset['Glac_lost_2100-all'] / temp_dev_df_subset['Glac_count-all'] * 100, 
                     linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        # Specific mass balance
        ax[4,1].plot(temp_dev_df_subset['global_mean_deviation_degC'], temp_dev_df_subset['mb_mwea-all'] / 1000, 
                     linewidth=0, marker=marker, mec=rcp_colordict[scenario], mew=marker_width, mfc='none', label=scenario)
        
    # ----- Time series by degree plots -----
    for deg_group in deg_groups[::-1]:
        
        # Volume
        reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
        reg_vol_med = np.median(reg_vol, axis=0)
        reg_vol_std = np.std(reg_vol, axis=0)
        
        reg_vol_med_norm = (1 - reg_vol_med / reg_vol_med[normyear_idx]) * 100
        reg_vol_std_norm = reg_vol_std / reg_vol_med[normyear_idx] * 100
        
        ax[0,0].plot(years, reg_vol_med_norm, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        
        if deg_group in temps_plot_mad:
            ax[0,0].fill_between(years, 
                                 reg_vol_med_norm + 1.96*reg_vol_std_norm, 
                                 reg_vol_med_norm - 1.96*reg_vol_std_norm, 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
        # Area
        reg_area = ds_multigcm_area_bydeg[reg][deg_group]
        reg_area_med = np.median(reg_area, axis=0)
        reg_area_std = np.std(reg_area, axis=0)
        
        reg_area_med_norm = (1 - reg_area_med / reg_area_med[normyear_idx]) * 100
        reg_area_std_norm = reg_area_std / reg_area_med[normyear_idx] * 100
        
        ax[1,0].plot(years, reg_area_med_norm, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        
        if deg_group in temps_plot_mad:
            ax[1,0].fill_between(years, 
                                 reg_area_med_norm + 1.96*reg_area_std_norm, 
                                 reg_area_med_norm - 1.96*reg_area_std_norm, 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
        # SLR
        reg_vol_bsl = ds_multigcm_vol_bydeg_bsl[reg][deg_group]

        # Cumulative Sea-level change [mm SLE]
        #  - accounts for water from glaciers replacing the ice that is below sea level as well
        #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
        reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                   (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
#            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water) / pygem_prms.area_ocean * 1000))
        reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
        
        reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

        reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
        reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
        
        ax[2,0].plot(years[0:-1], reg_slr_cum_avg, color=temp_colordict[deg_group], linestyle='-', 
                linewidth=1, zorder=4, label=deg_group)
        if deg_group in temps_plot_mad:
            ax[2,0].fill_between(years[0:-1], 
                                 (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
                                 (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
        # Glaciers lost
        reg_glac_lost = ds_multigcm_glac_lost_bydeg[reg][deg_group]
        reg_glac_lost_med = np.median(reg_glac_lost, axis=0)
        reg_glac_lost_std = np.std(reg_glac_lost, axis=0)
        
        ax[3,0].plot(years, reg_glac_lost_med / glac_count_total * 100, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        
        if deg_group in temps_plot_mad:
            ax[3,0].fill_between(years, 
                                 (reg_glac_lost_med + 1.96*reg_glac_lost_std) / glac_count_total * 100, 
                                 (reg_glac_lost_med - 1.96*reg_glac_lost_std) / glac_count_total * 100, 
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
        # Mass balance
        reg_mass = reg_vol * pygem_prms.density_ice
        reg_area = ds_multigcm_area_bydeg[reg][deg_group]

        # Specific mass change rate
        reg_mb = (reg_mass[:,1:] - reg_mass[:,0:-1]) / reg_area[:,0:-1]
        reg_mb_med = np.median(reg_mb, axis=0)
        reg_mb_std = np.std(reg_mb, axis=0)
        
        ax[4,0].plot(years[0:-1], reg_mb_med / 1000, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
        if deg_group in temps_plot_mad:
            ax[4,0].fill_between(years[0:-1], 
                                (reg_mb_med + 1.96*reg_mb_std) / 1000, 
                                (reg_mb_med - 1.96*reg_mb_std) / 1000,
                                alpha=0.2, facecolor=temp_colordict[deg_group], label=None)
            
        ncount_by_deg.append(reg_vol.shape[0])
                

        
#    ax[0,1].set_ylabel('Mass\nat 2100\n(rel. to 2015)', size=11)
#    ax[1,1].set_ylabel('Area\nat 2100\n(rel. to 2015)', size=11)
#    ax[2,1].set_ylabel('Sea level rise\nat 2100\n(mm SLE)', size=11)
#    ax[3,1].set_ylabel('Glaciers lost\nat 2100\n(10$^{3}$)', size=11)
    
    ax[0,0].set_ylabel('Mass loss \n(%)', size=12)
    ax[1,0].set_ylabel('Area loss \n(%)', size=12)
    ax[2,0].set_ylabel('SLR\n(mm SLE)', size=12)
#    ax[3,0].set_ylabel('Glaciers lost\n(10$^{3}$)', size=12)
    ax[3,0].set_ylabel('Glaciers lost\n(%)', size=12)
#    ax[4,0].set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)', size=12)
    ax[4,0].set_ylabel('$\Delta$M/$\Delta$t\n(m w.e. yr$^{-1}$)', size=12)
#    ax[4,0].set_ylabel('$\Delta$M/\n(m w.e. yr$^{-1}$)', size=12)
        
#    ax[4,1].set_xlabel('Global temperature change ($^\circ$C)', size=11)
    ax[4,1].set_xlabel('Global temperature change (' + u'\N{DEGREE SIGN}' + 'C' + ')', size=11)
    ax[4,0].set_xlabel('Year', size=11)
    
    
   
    # X axes
    ax[0,0].set_xlim(normyear, endyear_plot)
    ax[0,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[1,0].set_xlim(normyear, endyear_plot)
    ax[1,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[1,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[2,0].set_xlim(normyear, endyear_plot)
    ax[2,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[2,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[3,0].set_xlim(normyear, endyear_plot)
    ax[3,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[3,0].xaxis.set_minor_locator(MultipleLocator(10))
    ax[4,0].set_xlim(normyear, endyear_plot)
    ax[4,0].xaxis.set_major_locator(MultipleLocator(40))
    ax[4,0].xaxis.set_minor_locator(MultipleLocator(10))
    
    ax[0,1].set_xlim(0,5)
    ax[0,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[0,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[1,1].set_xlim(0,5)
    ax[1,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[1,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[2,1].set_xlim(0,5)
    ax[2,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[2,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[3,1].set_xlim(0,5)
    ax[3,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[3,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    ax[4,1].set_xlim(0,5)
    ax[4,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[4,1].xaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax[0,0].axes.xaxis.set_ticklabels([])
    ax[1,0].axes.xaxis.set_ticklabels([])
    ax[2,0].axes.xaxis.set_ticklabels([])
    ax[3,0].axes.xaxis.set_ticklabels([])
    ax[0,1].axes.xaxis.set_ticklabels([])
    ax[1,1].axes.xaxis.set_ticklabels([])
    ax[2,1].axes.xaxis.set_ticklabels([])
    ax[3,1].axes.xaxis.set_ticklabels([])
    
    # Y axes
    ax[0,0].set_ylim(0,100)
    ax[0,0].yaxis.set_major_locator(MultipleLocator(50))
    ax[0,0].yaxis.set_minor_locator(MultipleLocator(10))
    ax[1,0].set_ylim(0,100)
    ax[1,0].yaxis.set_major_locator(MultipleLocator(50))
    ax[1,0].yaxis.set_minor_locator(MultipleLocator(10))
    ax[2,0].set_ylim(0,280)
    ax[2,0].yaxis.set_major_locator(MultipleLocator(100))
    ax[2,0].yaxis.set_minor_locator(MultipleLocator(20))
    ax[3,0].set_ylim(0,glac_count_total/1000)
    ax[3,0].yaxis.set_major_locator(MultipleLocator(100))
    ax[3,0].yaxis.set_minor_locator(MultipleLocator(20))
    ax[3,0].set_ylim(0,100)
    ax[3,0].yaxis.set_major_locator(MultipleLocator(50))
    ax[3,0].yaxis.set_minor_locator(MultipleLocator(10))
    ax[4,0].set_ylim(-7,1)
    ax[4,0].yaxis.set_major_locator(MultipleLocator(2))
    ax[4,0].yaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax[0,1].set_ylim(0,100)
    ax[0,1].yaxis.set_major_locator(MultipleLocator(50))
    ax[0,1].yaxis.set_minor_locator(MultipleLocator(10))
    ax[1,1].set_ylim(0,100)
    ax[1,1].yaxis.set_major_locator(MultipleLocator(50))
    ax[1,1].yaxis.set_minor_locator(MultipleLocator(10))
    ax[2,1].set_ylim(0,300)
    ax[2,1].yaxis.set_major_locator(MultipleLocator(100))
    ax[2,1].yaxis.set_minor_locator(MultipleLocator(20))
#    ax[3,1].set_ylim(0,glac_count_total/1000)
#    ax[3,1].yaxis.set_major_locator(MultipleLocator(100))
#    ax[3,1].yaxis.set_minor_locator(MultipleLocator(20))
    ax[3,1].set_ylim(0,100)
    ax[3,1].yaxis.set_major_locator(MultipleLocator(50))
    ax[3,1].yaxis.set_minor_locator(MultipleLocator(10))
    ax[4,1].set_ylim(-7,1)
    ax[4,1].yaxis.set_major_locator(MultipleLocator(2))
    ax[4,1].yaxis.set_minor_locator(MultipleLocator(0.5))
    
    # Tick parameters
#    ax[4,0].tick_params(axis='x', labelsize=12)
    ax[0,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[0,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[1,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[1,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[2,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[2,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[3,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[3,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[4,0].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[4,0].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[0,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[0,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[1,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[1,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[2,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[2,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[3,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[3,1].tick_params(axis='both', which='minor', direction='in', right=True)
    ax[4,1].tick_params(axis='both', which='major', direction='inout', right=True)
    ax[4,1].tick_params(axis='both', which='minor', direction='in', right=True)
    
    # Figure labels
    ax[0,0].text(0.02, 0.98, 'A', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[0,0].transAxes)
    ax[0,1].text(0.02, 0.98, 'B', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[0,1].transAxes)
    ax[1,0].text(0.02, 0.98, 'C', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[1,0].transAxes)
    ax[1,1].text(0.02, 0.98, 'D', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[1,1].transAxes)
    ax[2,0].text(0.02, 0.98, 'E', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[2,0].transAxes)
    ax[2,1].text(0.02, 0.98, 'F', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[2,1].transAxes)
    ax[3,0].text(0.02, 0.98, 'G', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[3,0].transAxes)
    ax[3,1].text(0.02, 0.98, 'H', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[3,1].transAxes)
    ax[4,0].text(0.02, 0.98, 'I', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[4,0].transAxes)
    ax[4,1].text(0.02, 0.98, 'J', weight='bold', size=10, horizontalalignment='left', 
                 verticalalignment='top', transform=ax[4,1].transAxes)
    
#    ax[0,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[0,1].transAxes)
#    ax[1,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[1,1].transAxes)
#    ax[2,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[2,1].transAxes)
#    ax[3,1].text(0.5, 0.98, '2100', size=10, horizontalalignment='center', 
#                 verticalalignment='top', transform=ax[3,1].transAxes)
    
    # Legends
    ax[0,1].legend(labels=['SSP1-1.9','SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5', 'RCP2.6', 'RCP4.5', 'RCP8.5'],
                  loc=(0.15,0.70), 
                  fontsize=6, labelspacing=0.25, handlelength=1, handletextpad=0.1,
                  borderpad=0.1, ncol=3, columnspacing=0.25, frameon=True)
    
#    labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (' + r'$\pm$' + format(deg_groups_bnds[x],'0.2f') + ')' 
#            for x in np.arange(len(deg_groups))]
    labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (n=' + str(ncount_by_deg[x]) + ')'
            for x in np.arange(len(deg_groups))]
    ax[0,0].legend(loc=(0.15, 0.38), labels=labels[::-1], fontsize=8, ncol=1, columnspacing=0.5, labelspacing=0.05, 
                   handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)
    
    # Save figure
    fig_fn = 'Temp_Overview_wtime-global.png'
    fig.set_size_inches(6,6)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)


    #%% ----- FIGURE: MASS BALANCE COMOPNENTS -----
    
    # Mass balance data
    mb_comp_cns = ['reg']
    
    for deg_group in deg_groups:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.33,hspace=0.4)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        ax2 = fig.add_subplot(gs[0,3])
        ax3 = fig.add_subplot(gs[1,2])
        ax4 = fig.add_subplot(gs[1,3])
        ax5 = fig.add_subplot(gs[2,0])
        ax6 = fig.add_subplot(gs[2,1])
        ax7 = fig.add_subplot(gs[2,2])
        ax8 = fig.add_subplot(gs[2,3])
        ax9 = fig.add_subplot(gs[3,0])
        ax10 = fig.add_subplot(gs[3,1])
        ax11 = fig.add_subplot(gs[3,2])
        ax12 = fig.add_subplot(gs[3,3])
        ax13 = fig.add_subplot(gs[4,0])
        ax14 = fig.add_subplot(gs[4,1])
        ax15 = fig.add_subplot(gs[4,2])
        ax16 = fig.add_subplot(gs[4,3])
        ax17 = fig.add_subplot(gs[5,0])
        ax18 = fig.add_subplot(gs[5,1])
        ax19 = fig.add_subplot(gs[5,2])
        ax20 = fig.add_subplot(gs[5,3])
        
        regions_ordered = ['all',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
        for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
            
            reg = regions_ordered[nax]
            
            # MASS BALANCE COMPONENTS
            # - these are only meant for monthly and/or relative purposes 
            #   mass balance from volume change should be used for annual changes
            reg_area_annual = np.median(ds_multigcm_area_bydeg[reg][deg_group],axis=0)
            reg_melt_monthly = np.median(ds_multigcm_melt_bydeg[reg][deg_group],axis=0)
            reg_acc_monthly = np.median(ds_multigcm_acc_bydeg[reg][deg_group],axis=0)
            reg_refreeze_monthly = np.median(ds_multigcm_refreeze_bydeg[reg][deg_group],axis=0)
            reg_frontalablation_monthly = np.median(ds_multigcm_fa_bydeg[reg][deg_group],axis=0)            
            
            reg_acc_annual = reg_acc_monthly.reshape(-1,12).sum(axis=1)
            # Refreeze
            reg_refreeze_annual = reg_refreeze_monthly.reshape(-1,12).sum(axis=1)
            # Melt
            reg_melt_annual = reg_melt_monthly.reshape(-1,12).sum(axis=1)
            # Frontal Ablation
            reg_frontalablation_annual = reg_frontalablation_monthly.reshape(-1,12).sum(axis=1)
            # Periods
            if reg_acc_annual.shape[0] == 101:
                period_yrs = 20
                periods = (np.arange(years.min(), years[0:100].max(), period_yrs) + period_yrs/2).astype(int)
                reg_acc_periods = reg_acc_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_refreeze_periods = reg_refreeze_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_melt_periods = reg_melt_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_frontalablation_periods = reg_frontalablation_annual[0:100].reshape(-1,period_yrs).sum(1)
                reg_massbaltotal_periods = reg_acc_periods + reg_refreeze_periods - reg_melt_periods - reg_frontalablation_periods
                
                # Convert to mwea
                reg_area_periods = reg_area_annual[0:100].reshape(-1,period_yrs).mean(1)
                reg_acc_periods_mwea = reg_acc_periods / reg_area_periods / period_yrs
                reg_refreeze_periods_mwea = reg_refreeze_periods / reg_area_periods / period_yrs
                reg_melt_periods_mwea = reg_melt_periods / reg_area_periods / period_yrs
                reg_frontalablation_periods_mwea = reg_frontalablation_periods / reg_area_periods / period_yrs
                reg_massbaltotal_periods_mwea = reg_massbaltotal_periods / reg_area_periods / period_yrs
                
                print(reg, deg_group, reg_massbaltotal_periods_mwea)
                
            else:
                assert True==False, 'Set up for different time periods'

            # Plot
            ax.bar(periods, reg_acc_periods_mwea + reg_refreeze_periods_mwea, color='#3553A5', width=period_yrs/2-1, label='Refreeze', zorder=2)
            ax.bar(periods, reg_acc_periods_mwea, color='#3478BD', width=period_yrs/2-1, label='Accumulation', zorder=3)
            if not reg_frontalablation_periods_mwea.sum() == 0:
                ax.bar(periods, -reg_frontalablation_periods_mwea, color='#04D8B2', width=period_yrs/2-1, label='Frontal ablation', zorder=3)
            ax.bar(periods, -reg_melt_periods_mwea - reg_frontalablation_periods_mwea, color='#F47A20', width=period_yrs/2-1, label='Melt', zorder=2)
            ax.bar(periods, reg_massbaltotal_periods_mwea, color='#555654', width=period_yrs-2, label='Mass balance (total)', zorder=1)
            
            ax.set_xlim(years.min(), years[0:-1].max())
            ax.set_ylim(-5.5,3)
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(20))
            ax.yaxis.set_major_locator(MultipleLocator(2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            
            if ax in [ax1,ax5,ax9,ax13,ax17]:
#                ax.set_ylabel('$\Delta$M/$\Delta$t\n(10$^{3}$ kg m$^{-2}$ yr$^{-1}$)')
                ax.set_ylabel('$\Delta$M/$\Delta$t\n(m w.e. yr$^{-1}$)')
                
            if nax == 0:
                label_height=1.06
            else:
                label_height=1.14
            ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                    verticalalignment='top', transform=ax.transAxes)
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)
            
            if nax == 0:
                ax.legend(loc=(1.07,0.65), fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25,
                          handlelength=1, handletextpad=0.25, borderpad=0, frameon=False,
                          ) 
        # Save figure
        fig_fn = ('mbcomponents_allregions_multigcm-' + str(deg_group) + '_bydeg.png')
        fig.set_size_inches(8.5,11)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
            
    #%% ----- FIGURE: NORMALIZED VOLUME CHANGE MULTI-GCM -----
    for reg in regions:
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
    
        normyear_idx = np.where(years == normyear)[0][0]
    
        for deg_group in deg_groups:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_mad = median_abs_deviation(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_mad_norm = reg_vol_mad / reg_vol_med[normyear_idx]
            
            ax[0,0].plot(years, reg_vol_med_norm, color=temp_colordict[deg_group], linewidth=1, zorder=4, label=deg_group)
            
            if deg_group in temps_plot_mad:
                ax[0,0].fill_between(years, 
                                     reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
                                     reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
                                     alpha=0.2, facecolor=temp_colordict[deg_group], label=None)
        
        ax[0,0].set_ylabel('Volume (-)')
        ax[0,0].set_xlim(normyear, endyear_plot)
        ax[0,0].set_ylim(0,1)
        ax[0,0].xaxis.set_major_locator(MultipleLocator(20))
        ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
        ax[0,0].yaxis.set_major_locator(MultipleLocator(0.2))
        ax[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))
        ax[0,0].tick_params(direction='inout', right=True)
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].legend(
    #                rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
    #                handletextpad=0.25, borderpad=0, frameon=False
                )
        ax[0,0].tick_params(direction='inout', right=True)
        # Save figure
        fig_fn = ('deg_groups_' + str(reg) + '_volchangenorm_' + str(startyear_plot) + '-' + str(endyear_plot) + '_multigcm.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[1,3])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])
    ax7 = fig.add_subplot(gs[2,2])
    ax8 = fig.add_subplot(gs[2,3])
    ax9 = fig.add_subplot(gs[3,0])
    ax10 = fig.add_subplot(gs[3,1])
    ax11 = fig.add_subplot(gs[3,2])
    ax12 = fig.add_subplot(gs[3,3])
    ax13 = fig.add_subplot(gs[4,0])
    ax14 = fig.add_subplot(gs[4,1])
    ax15 = fig.add_subplot(gs[4,2])
    ax16 = fig.add_subplot(gs[4,3])
    ax17 = fig.add_subplot(gs[5,0])
    ax18 = fig.add_subplot(gs[5,1])
    ax19 = fig.add_subplot(gs[5,2])
    ax20 = fig.add_subplot(gs[5,3])
    
    regions_ordered = ['all',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]            
        
        for deg_group in deg_groups:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_std = np.std(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_med[normyear_idx]
            
            ax.plot(years, reg_vol_med_norm, color=temp_colordict[deg_group],
                    linewidth=1, zorder=4, label=deg_group)
            if deg_group in temps_plot_mad:
                ax.fill_between(years, 
                                reg_vol_med_norm + 1.96*reg_vol_std_norm, 
                                reg_vol_med_norm - 1.96*reg_vol_std_norm, 
                                alpha=0.2, facecolor=temp_colordict[deg_group], label=None)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('Mass (rel. to 2015)')
        ax.set_xlim(normyear, endyear_plot)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_ylim(0,1.05)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (n=' + str(ncount_by_deg[x]) + ')'
                    for x in np.arange(len(deg_groups))]
            ax.legend(loc=(-1.35,0.2), labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)
    # Save figure
    fig_fn = 'deg_groups_allregions_volchange_norm_' + str(normyear) + '-' + str(endyear_plot) + '_multigcm.png'
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: CUMULATIVE SEA-LEVEL RISE w BOX AND WHISKERS -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=8,wspace=0.66,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:3])
    ax1b = fig.add_subplot(gs[0:2,2:4])
    ax2 = fig.add_subplot(gs[0,6:])
    ax3 = fig.add_subplot(gs[1,4:6])
    ax4 = fig.add_subplot(gs[1,6:])
    ax5 = fig.add_subplot(gs[2,0:2])
    ax6 = fig.add_subplot(gs[2,2:4])
    ax7 = fig.add_subplot(gs[2,4:6])
    ax8 = fig.add_subplot(gs[2,6:])
    ax9 = fig.add_subplot(gs[3,0:2])
    ax10 = fig.add_subplot(gs[3,2:4])
    ax11 = fig.add_subplot(gs[3,4:6])
    ax12 = fig.add_subplot(gs[3,6:])
    ax13 = fig.add_subplot(gs[4,0:2])
    ax14 = fig.add_subplot(gs[4,2:4])
    ax15 = fig.add_subplot(gs[4,4:6])
    ax16 = fig.add_subplot(gs[4,6:])
    ax17 = fig.add_subplot(gs[5,0:2])
    ax18 = fig.add_subplot(gs[5,2:4])
    ax19 = fig.add_subplot(gs[5,4:6])
    ax20 = fig.add_subplot(gs[5,6:])
    
    data_boxplot = []
    regions_ordered = ['all',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]

        for deg_group in deg_groups[::-1]:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_bsl = ds_multigcm_vol_bydeg_bsl[reg][deg_group]

            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers replacing the ice that is below sea level as well
            #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
            
            ax.plot(years[0:-1], reg_slr_cum_avg, color=temp_colordict[deg_group], linestyle='-', 
                    linewidth=1, zorder=4, label=deg_group)
            if deg_group in temps_plot_mad:
                ax.fill_between(years[0:-1], 
                                (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
                                (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
                                alpha=0.35, facecolor=temp_colordict[deg_group], label=None)
            
            # Aggregate boxplot data
            if reg in ['all']:
                data_boxplot.append(reg_slr_cum[:,-1])
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
#            ax.set_ylabel('$\Delta$M (mm SLE)')
            ax.set_ylabel('SLR (mm SLE)')
        ax.set_xlim(normyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in ['all']:
            ax.set_ylim(0,210)
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
        if reg in [19, 1, 5]:
            ax.set_ylim(0,38)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5))    
        elif reg in [3, 9, 4, 7]:
            ax.set_ylim(0,22)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(5)) 
        elif reg in [17, 13, 6, 14]:
            ax.set_ylim(0,12)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(1)) 
        elif reg in [15, 2]:
            ax.set_ylim(0,2.8)
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.2)) 
        elif reg in [8]:
            ax.set_ylim(0,0.75)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [10, 11, 16, 12, 18]:
            ax.set_ylim(0,0.3)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (n=' + str(ncount_by_deg[x]) + ')'
                    for x in np.arange(len(deg_groups))]
#            labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (' + r'$\pm$' + format(deg_groups_bnds[x],'0.2f') + ')' for x in np.arange(len(deg_groups))]
            ax.legend(loc=(-1.35,0.2), labels=labels[::-1], fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)
    
    data_boxplot = data_boxplot[::-1]
    bp = ax1b.boxplot(data_boxplot, whis='range')
    ax1b.set_ylim(0,210)
    for nbox, box in enumerate(bp['boxes']):
        deg_group = deg_groups[nbox]
        # change outline color
        box.set(color=temp_colordict[deg_group], linewidth=1)
    for nitem, item in enumerate(bp['medians']):
        deg_group = deg_groups[nitem]
        # change outline color
        item.set(color=temp_colordict[deg_group], linewidth=1)
    for nitem, item in enumerate(bp['whiskers']):
        deg_group = deg_groups[int(np.floor(nitem/2))]
        # change outline color
        item.set(color=temp_colordict[deg_group], linewidth=1)
    for nitem, item in enumerate(bp['caps']):
        deg_group = deg_groups[int(np.floor(nitem/2))]
        # change outline color
        item.set(color=temp_colordict[deg_group], linewidth=1)
#    for nitem, item in enumerate(bp['fliers']):
#        deg_group = deg_groups[nitem]
#        # change outline color
#        item.set(mec=temp_colordict[deg_group], linewidth=1)
    # turn off axes
    ax1b.get_yaxis().set_visible(False)
    ax1b.get_xaxis().set_visible(False)
    ax1b.axis('off')
    ax1b.set_xlim(-5,8)
            
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('deg_groups_allregions_SLR-cum_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd-BoxWhisker.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    #%% ----- FIGURE: SEA-LEVEL RISE RATE -----
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=6,ncols=8,wspace=1,hspace=0.4)
    ax1 = fig.add_subplot(gs[0:2,0:4])
    ax2 = fig.add_subplot(gs[0,6:])
    ax3 = fig.add_subplot(gs[1,4:6])
    ax4 = fig.add_subplot(gs[1,6:])
    ax5 = fig.add_subplot(gs[2,0:2])
    ax6 = fig.add_subplot(gs[2,2:4])
    ax7 = fig.add_subplot(gs[2,4:6])
    ax8 = fig.add_subplot(gs[2,6:])
    ax9 = fig.add_subplot(gs[3,0:2])
    ax10 = fig.add_subplot(gs[3,2:4])
    ax11 = fig.add_subplot(gs[3,4:6])
    ax12 = fig.add_subplot(gs[3,6:])
    ax13 = fig.add_subplot(gs[4,0:2])
    ax14 = fig.add_subplot(gs[4,2:4])
    ax15 = fig.add_subplot(gs[4,4:6])
    ax16 = fig.add_subplot(gs[4,6:])
    ax17 = fig.add_subplot(gs[5,0:2])
    ax18 = fig.add_subplot(gs[5,2:4])
    ax19 = fig.add_subplot(gs[5,4:6])
    ax20 = fig.add_subplot(gs[5,6:])
    
    data_boxplot = []
    regions_ordered = ['all',1,5,19,3,4,9,7,17,6,13,14,2,15,8,10,11,16,18,12]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20]):
        
        reg = regions_ordered[nax]

        for deg_group in deg_groups[::-1]:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_bsl = ds_multigcm_vol_bydeg_bsl[reg][deg_group]

            # Sea-level change [mm SLE yr-1]
            #  - accounts for water from glaciers replacing the ice that is below sea level as well
            #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)
            
            reg_slr_med_raw = np.median(reg_slr, axis=0)
            reg_slr_std_raw = np.std(reg_slr, axis=0) 
            reg_slr_med = uniform_filter(reg_slr_med_raw, size=(11))
            reg_slr_std = uniform_filter(reg_slr_std_raw, size=(11))
            
            ax.plot(years[0:-1], reg_slr_med, color=temp_colordict[deg_group], linestyle='-', 
                    linewidth=1, zorder=4, label=deg_group)
            if deg_group in temps_plot_mad:
                ax.fill_between(years[0:-1], 
                                (reg_slr_med + 1.96*reg_slr_std), 
                                (reg_slr_med - 1.96*reg_slr_std), 
                                alpha=0.35, facecolor=temp_colordict[deg_group], label=None)
            
#            if reg in [19]:
#                print(deg_group)
#                print(reg_slr_med)
        
        if ax in [ax1,ax5,ax9,ax13,ax17]:
            ax.set_ylabel('$\Delta$M/$\Delta$t\n(mm SLE yr$^{-1}$)')
            ax.set_ylabel('SLR/$\Delta$t\n(mm SLE yr$^{-1}$)')
        ax.set_xlim(normyear, endyear)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
#        ax.plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        if reg in [19, 5, 1]:
            ax.set_ylim(-0,0.75)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))    
        elif reg in [3, 9, 4, 7]:
            ax.set_ylim(0,0.45)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1)) 
        elif reg in [17, 13, 6, 14]:
            ax.set_ylim(0,0.21)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.05)) 
        elif reg in [2, 15]:
            ax.set_ylim(0,0.11)
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.025)) 
        elif reg in [8, 10, 11, 16, 12, 18]:
            ax.set_ylim(0,0.025)
            ax.yaxis.set_major_locator(MultipleLocator(0.01))
            ax.yaxis.set_minor_locator(MultipleLocator(0.005)) 
            
        if nax == 0:
            label_height=1.06
        else:
            label_height=1.14
        ax.text(1, label_height, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        if nax == 1:
            labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (n=' + str(ncount_by_deg[x]) + ')'
                    for x in np.arange(len(deg_groups))]
#            labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (' + r'$\pm$' + format(deg_groups_bnds[x],'0.2f') + ')' for x in np.arange(len(deg_groups))]
            ax.legend(loc=(-1.35,0.2), labels=labels[::-1], fontsize=10, ncol=1, columnspacing=0.5, labelspacing=0.25, 
                      handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)
            
    # Save figure
    if 'rcp26' in rcps and 'ssp126' in rcps:
        scenario_str = 'rcps_ssps'
    elif 'rcp26' in rcps:
        scenario_str = 'rcps'
    elif 'ssp126' in rcps:
        scenario_str = 'ssps'
    fig_fn = ('deg_groups_allregions_SLR-rate_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd-BoxWhisker.png')
    fig.set_size_inches(8.5,11)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

    #%% ----- FIGURE: GLOBAL CUMULATIVE SEA LEVEL RISE -----
    data_boxplot = []
    for reg in ['all']:
        fig, ax = plt.subplots(1, 2, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
    
        normyear_idx = np.where(years == normyear)[0][0]
    
        for deg_group in deg_groups:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_bsl = ds_multigcm_vol_bydeg_bsl[reg][deg_group]

            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers replacing the ice that is below sea level as well
            #    from Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water - 
                       (reg_vol_bsl[:,1:] - reg_vol_bsl[:,0:-1])) / pygem_prms.area_ocean * 1000))
#            reg_slr = (-1*(((reg_vol[:,1:] - reg_vol[:,0:-1]) * pygem_prms.density_ice / pygem_prms.density_water) / pygem_prms.area_ocean * 1000))
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]

            reg_slr_cum_avg = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_var = np.std(reg_slr_cum, axis=0)
            
            ax[0,0].plot(years[0:-1], reg_slr_cum_avg, color=temp_colordict[deg_group], linestyle='-', 
                         linewidth=1, zorder=4, label=deg_group)
            if deg_group in temps_plot_mad:
                ax[0,0].fill_between(years[0:-1], 
                                     (reg_slr_cum_avg + 1.96*reg_slr_cum_var), 
                                     (reg_slr_cum_avg - 1.96*reg_slr_cum_var), 
                                     alpha=0.35, facecolor=temp_colordict[deg_group], label=None)
            
            # Aggregate boxplot data
            if reg in ['all']:
                data_boxplot.append(reg_slr_cum[:,-1])
                
        ax[0,0].set_ylabel('Sea Level Rise (mm SLE)')
        ax[0,0].set_xlim(normyear, endyear)
        ax[0,0].xaxis.set_major_locator(MultipleLocator(50))
        ax[0,0].xaxis.set_minor_locator(MultipleLocator(10))
        ax[0,0].plot(years,np.zeros(years.shape), color='k', linewidth=0.5)
            
        ax[0,0].set_ylim(0,250)
        ax[0,0].yaxis.set_major_locator(MultipleLocator(50))
        ax[0,0].yaxis.set_minor_locator(MultipleLocator(10))
        
        ax[0,0].text(0.98, 1.06, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax[0,0].transAxes)
#        handles, labels = ax[0,0].get_legend_handles_labels()
        labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (' + r'$\pm$' + format(deg_groups_bnds[x],'0.2f') + ')' for x in np.arange(len(deg_groups))]
        ax[0,0].legend(loc=(0.05,0.9), labels=labels, fontsize=10, ncol=1, columnspacing=0.5, labelspacing=-2.3, 
                       handlelength=1, handletextpad=0.25, borderpad=0, frameon=False)
    
        ax[0,0].tick_params(axis='both', which='major', direction='inout', right=True)
        ax[0,0].tick_params(axis='both', which='minor', direction='in', right=True)
        
        
        bp = ax[0,1].boxplot(data_boxplot, whis='range')
        ax[0,1].set_ylim(0,230)
        for nbox, box in enumerate(bp['boxes']):
            deg_group = deg_groups[nbox]
            # change outline color
            box.set(color=temp_colordict[deg_group], linewidth=1)
        for nitem, item in enumerate(bp['medians']):
            deg_group = deg_groups[nitem]
            # change outline color
            item.set(color=temp_colordict[deg_group], linewidth=1)
        for nitem, item in enumerate(bp['whiskers']):
            deg_group = deg_groups[int(np.floor(nitem/2))]
            # change outline color
            item.set(color=temp_colordict[deg_group], linewidth=1)
        for nitem, item in enumerate(bp['caps']):
            deg_group = deg_groups[int(np.floor(nitem/2))]
            # change outline color
            item.set(color=temp_colordict[deg_group], linewidth=1)
    #    for nitem, item in enumerate(bp['fliers']):
    #        deg_group = deg_groups[nitem]
    #        # change outline color
    #        item.set(mec=temp_colordict[deg_group], linewidth=1)
        # turn off axes
        ax[0,1].get_yaxis().set_visible(False)
        ax[0,1].get_xaxis().set_visible(False)
        ax[0,1].axis('off')
        ax[0,1].set_xlim(0,16)
        
        
        # Save figure
        fig_fn = ('deg_groups_all_SLR-cum_' + str(startyear) + '-' + str(endyear) + '_multigcm' + 
              '-' + scenario_str + '-medstd-BoxWhisker.png')
        fig.set_size_inches(6,3)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
    
    
    #%% ----- HISTOGRAM OF GLACIERS LOST BY SIZE -----
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                            rgi_regionsO1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 
                            rgi_regionsO2='all', rgi_glac_number='all')
    glac_area_dict = dict(zip(main_glac_rgi_all.RGIId.values, main_glac_rgi_all.Area.values))
    
    #%%
    # All glacier number statistics
    area_bins = [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 50, 1e5]
    area_bins_str = ['< 0.1', '0.1 - 0.25', '0.25 - 0.5', '0.5 - 1', '1 - 2', '2 - 5', '5 - 10', '10 - 50', '> 50']
    
    for reg in ['all']:
#    for reg in regions:

        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=True, sharey=False, 
                               gridspec_kw = {'wspace':0, 'hspace':0.1})
    
        hist_all = None
        for ngroup, deg_group in enumerate(deg_groups[::-1]):
            
            # Glaciers lost
            ngcms = ds_multigcm_glac_lost_bydeg[reg][deg_group].shape[0]
            rgiids_lost_reg = ds_multigcm_glac_lost2100_rgiids_bydeg[reg][deg_group]
            rgiids_lost_reg_count_dict = Counter(rgiids_lost_reg)
            # glacier lost if lost in more than half of them
            rgiids_lost_reg_unique_raw = [x for x in list(rgiids_lost_reg_count_dict.keys()) if rgiids_lost_reg_count_dict[x] > ngcms/2]
            
            # dictionary for areas
            rgiids_lost_reg_unique = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in rgiids_lost_reg_unique_raw]
            area_lost_reg_unique = [glac_area_dict[x] for x in rgiids_lost_reg_unique]
            
            rgiids_all_reg = ds_multigcm_glac_rgiids_bydeg[reg][deg_group]
            rgiids_all_reg_count_dict = Counter(rgiids_all_reg)
            rgiids_all_reg_unique_raw = [x for x in list(rgiids_all_reg_count_dict.keys()) if rgiids_all_reg_count_dict[x] > ngcms/2]
            rgiids_all_reg_unique = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in rgiids_all_reg_unique_raw]
            area_all_reg_unique = [glac_area_dict[x] for x in rgiids_all_reg_unique]
        
            # First, plot the all grey background
            if hist_all is None:
                hist_all, bins = np.histogram(area_all_reg_unique, bins=area_bins)
                hist_all_perc = hist_all / hist_all.sum() * 100
#                ax[0,0].bar(x=area_bins_str, height=hist_all_perc, width=1, 
#                           align='center', edgecolor='black', color='grey', zorder=0)
            
            # Plot lost glaciers by degree
            hist, bins = np.histogram(area_lost_reg_unique, bins=area_bins)
            hist_perc = hist / hist_all.sum() * 100
            ax[0,0].bar(x=area_bins_str, height=hist_perc, width=1, 
                        align='center', edgecolor=None, color=temp_colordict[deg_group], zorder=ngroup, 
                        label=deg_group)
            
            print(reg, deg_group, 'glaciers lost:', len(rgiids_lost_reg_unique))

        # ----- Overlay empty histogram ------
        ax[0,0].bar(x=area_bins_str, height=hist_all_perc, width=1, 
                    align='center', edgecolor='black', color='grey', fill=False, zorder=len(deg_groups)+1,
                    label=None)
        
        # ----- LABELS -----
#        # Region name
#        ax[0,0].text(0.98, 1.1, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                     verticalalignment='top', transform=ax[0,0].transAxes)
        

        # Histogram
        ax[0,0].set_xlabel('Area (km$^{2}$)', labelpad=0, fontsize=12)
        ax[0,0].set_ylabel('Glaciers lost (%)', fontsize=12)
        ax[0,0].yaxis.set_major_locator(MultipleLocator(5))
        ax[0,0].yaxis.set_minor_locator(MultipleLocator(1))
        ax[0,0].set_xticklabels(area_bins_str, ha='center', rotation=60)
#        ax[0,0].tick_params(pad=0.5)

        # Legend
        labels = ['+' + format(x,'.1f') + u'\N{DEGREE SIGN}' + 'C' for x in deg_groups[::-1]]
        ax[0,0].legend(loc='upper right', labels=labels,
                       fontsize=10, labelspacing=0.25, handlelength=1, 
                       handletextpad=0.25, borderpad=0, frameon=False)

        # Save figure
        fig_fn = ('Temp_vs_lost_histogram_' + str(reg) + '_bydeg.png')
        fig.set_size_inches(6,4)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
        
    #%% ----- SCALE TO 100% WITH BIGGER SIZES -----
    # All glacier number statistics
    area_bins = [0, 1, 10, 100, 1e5]
    area_bins_str = ['< 1', '1 - 10', '10 - 100', '> 100']
    
    for reg in ['all']:
#    for reg in regions:

        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=True, sharey=False, 
                               gridspec_kw = {'wspace':0, 'hspace':0.1})
    
        hist_all = None
        for ngroup, deg_group in enumerate(deg_groups[::-1]):
            
            # Glaciers lost
            ngcms = ds_multigcm_glac_lost_bydeg[reg][deg_group].shape[0]
            rgiids_lost_reg = ds_multigcm_glac_lost2100_rgiids_bydeg[reg][deg_group]
            rgiids_lost_reg_count_dict = Counter(rgiids_lost_reg)
            # glacier lost if lost in more than half of them
            rgiids_lost_reg_unique_raw = [x for x in list(rgiids_lost_reg_count_dict.keys()) if rgiids_lost_reg_count_dict[x] > ngcms/2]
            
            # dictionary for areas
            rgiids_lost_reg_unique = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in rgiids_lost_reg_unique_raw]
            area_lost_reg_unique = [glac_area_dict[x] for x in rgiids_lost_reg_unique]
            
            rgiids_all_reg = ds_multigcm_glac_rgiids_bydeg[reg][deg_group]
            rgiids_all_reg_count_dict = Counter(rgiids_all_reg)
            rgiids_all_reg_unique_raw = [x for x in list(rgiids_all_reg_count_dict.keys()) if rgiids_all_reg_count_dict[x] > ngcms/2]
            rgiids_all_reg_unique = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in rgiids_all_reg_unique_raw]
            area_all_reg_unique = [glac_area_dict[x] for x in rgiids_all_reg_unique]
        
            # First, plot the all grey background
            if hist_all is None:
                hist_all, bins = np.histogram(area_all_reg_unique, bins=area_bins)
                hist_all_perc = hist_all / hist_all.sum() * 100
#                ax[0,0].bar(x=area_bins_str, height=hist_all_perc, width=1, 
#                           align='center', edgecolor='black', color='grey', zorder=0)
            
            # Plot lost glaciers by degree
            hist, bins = np.histogram(area_lost_reg_unique, bins=area_bins)
            hist_perc = hist / hist_all * 100
            ax[0,0].bar(x=area_bins_str, height=hist_perc, width=1, 
                        align='center', edgecolor='k', color=temp_colordict[deg_group], zorder=ngroup, 
                        label=deg_group)
            
            print(reg, deg_group, 'glaciers lost:', len(rgiids_lost_reg_unique))

        # ----- Overlay empty histogram ------
#        ax[0,0].bar(x=area_bins_str, height=100, width=1, 
#                    align='center', edgecolor='black', color='grey', fill=False, zorder=len(deg_groups)+1,
#                    label=None)
        
        # ----- LABELS -----
#        # Region name
#        ax[0,0].text(0.98, 1.1, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                     verticalalignment='top', transform=ax[0,0].transAxes)
        
        ax[0,0].text(0.15, 1.01, str(hist_all[0]), size=10, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        ax[0,0].text(0.39, 1.01, str(hist_all[1]), size=10, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        ax[0,0].text(0.62, 1.01, str(hist_all[2]), size=10, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        ax[0,0].text(0.84, 1.01, str(hist_all[3]), size=10, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        

        # Histogram
        ax[0,0].set_xlabel('Area (km$^{2}$)', labelpad=0, fontsize=12)
        ax[0,0].set_ylabel('Glaciers lost (%)', fontsize=12)
        ax[0,0].set_ylim(0,100)
        ax[0,0].yaxis.set_major_locator(MultipleLocator(20))
        ax[0,0].yaxis.set_minor_locator(MultipleLocator(10))
        ax[0,0].set_xticklabels(area_bins_str, ha='center', rotation=0)
#        ax[0,0].tick_params(pad=0.5)

        # Legend
        labels = ['+' + format(x,'.1f') + u'\N{DEGREE SIGN}' + 'C' for x in deg_groups[::-1]]
        ax[0,0].legend(loc='upper right', labels=labels,
                       fontsize=10, labelspacing=0.25, handlelength=1, 
                       handletextpad=0.25, borderpad=0, frameon=False)

        # Save figure
        fig_fn = ('Temp_vs_lost_histogram_100scale_' + str(reg) + '_bydeg.png')
        fig.set_size_inches(6,4)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)
        
    #%% ----- SCALE TO 100% WITH EACH DEGREE ITS OWN COLUMN -----
    # All glacier number statistics
    area_bins = [0, 1, 10, 100, 1e5]
    area_bins_str = ['< 1', '1 - 10', '10 - 100', '> 100']
    
    for reg in ['all']:
#    for reg in regions:

        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=True, sharey=False, 
                               gridspec_kw = {'wspace':0, 'hspace':0.1})
    
        hist_all = None
        xpos = -2*1/(len(deg_groups)+1)
        for ngroup, deg_group in enumerate(deg_groups):
            
            # Glaciers lost
            ngcms = ds_multigcm_glac_lost_bydeg[reg][deg_group].shape[0]
            rgiids_lost_reg = ds_multigcm_glac_lost2100_rgiids_bydeg[reg][deg_group]
            rgiids_lost_reg_count_dict = Counter(rgiids_lost_reg)
            # glacier lost if lost in more than half of them
            rgiids_lost_reg_unique_raw = [x for x in list(rgiids_lost_reg_count_dict.keys()) if rgiids_lost_reg_count_dict[x] > ngcms/2]
            
            # dictionary for areas
            rgiids_lost_reg_unique = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in rgiids_lost_reg_unique_raw]
            area_lost_reg_unique = [glac_area_dict[x] for x in rgiids_lost_reg_unique]
            
            rgiids_all_reg = ds_multigcm_glac_rgiids_bydeg[reg][deg_group]
            rgiids_all_reg_count_dict = Counter(rgiids_all_reg)
            rgiids_all_reg_unique_raw = [x for x in list(rgiids_all_reg_count_dict.keys()) if rgiids_all_reg_count_dict[x] > ngcms/2]
            rgiids_all_reg_unique = ['RGI60-' + x.split('.')[0].zfill(2) + '.' + x.split('.')[1] for x in rgiids_all_reg_unique_raw]
            area_all_reg_unique = [glac_area_dict[x] for x in rgiids_all_reg_unique]
        
            # First, plot the all grey background
            if hist_all is None:
                hist_all, bins = np.histogram(area_all_reg_unique, bins=area_bins)
                hist_all_perc = hist_all / hist_all.sum() * 100
#                ax[0,0].bar(x=area_bins_str, height=hist_all_perc, width=1, 
#                           align='center', edgecolor='black', color='grey', zorder=0)
            
            # Plot lost glaciers by degree
            hist, bins = np.histogram(area_lost_reg_unique, bins=area_bins)
            hist_perc = hist / hist_all * 100
            
            ax[0,0].bar(x=np.arange(len(deg_groups)) + xpos, height=hist_perc, width=1/(len(deg_groups)+1), 
                        align='edge', edgecolor='k', color=temp_colordict[deg_group], zorder=ngroup, 
                        label=deg_group)
            
            xpos += 1/(len(deg_groups)+1)
            print(reg, deg_group, 'glaciers lost:', len(rgiids_lost_reg_unique))

        # ----- Overlay empty histogram ------
#        ax[0,0].bar(x=area_bins_str, height=100, width=1, 
#                    align='center', edgecolor='black', color='grey', fill=False, zorder=len(deg_groups)+1,
#                    label=None)
        
        # ----- LABELS -----
#        # Region name
#        ax[0,0].text(0.98, 1.1, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
#                     verticalalignment='top', transform=ax[0,0].transAxes)
        
#        ax[0,0].text(0.15, 1.01, str(hist_all[0]), size=10, horizontalalignment='center', 
#                     verticalalignment='bottom', transform=ax[0,0].transAxes)
#        ax[0,0].text(0.39, 1.01, str(hist_all[1]), size=10, horizontalalignment='center', 
#                     verticalalignment='bottom', transform=ax[0,0].transAxes)
#        ax[0,0].text(0.62, 1.01, str(hist_all[2]), size=10, horizontalalignment='center', 
#                     verticalalignment='bottom', transform=ax[0,0].transAxes)
#        ax[0,0].text(0.84, 1.01, str(hist_all[3]), size=10, horizontalalignment='center', 
#                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        
        ax[0,0].text(0.15, np.ceil(hist_perc[0])/100, str(hist_all[0]), size=12, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        ax[0,0].text(0.38, np.ceil(hist_perc[1])/100, str(hist_all[1]), size=12, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        ax[0,0].text(0.625, np.ceil(hist_perc[2])/100, str(hist_all[2]), size=12, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        ax[0,0].text(0.875, np.ceil(hist_perc[3])/100, str(hist_all[3]), size=12, horizontalalignment='center', 
                     verticalalignment='bottom', transform=ax[0,0].transAxes)
        

        # Histogram
        ax[0,0].set_xlabel('Area (km$^{2}$)', labelpad=0, fontsize=12)
        ax[0,0].set_ylabel('Glaciers lost (%)', fontsize=12)
        ax[0,0].set_ylim(0,100)
        ax[0,0].yaxis.set_major_locator(MultipleLocator(20))
        ax[0,0].yaxis.set_minor_locator(MultipleLocator(10))
        ax[0,0].set_xlim(-0.5,3.5)
        ax[0,0].xaxis.set_major_locator(MultipleLocator(1))
        area_bins_str_labels = ['']
        for area_label in area_bins_str:
            area_bins_str_labels.append(area_label)
        ax[0,0].set_xticklabels(area_bins_str_labels, ha='center', rotation=0)
#        ax[0,0].tick_params(pad=0.5)

        # Legend
        labels = ['+' + format(x,'.1f') + u'\N{DEGREE SIGN}' + 'C' for x in deg_groups]
        ax[0,0].legend(loc='upper right',
#                        loc=(0.78,0.68), 
                       labels=labels,
                       fontsize=12, labelspacing=0.25, handlelength=1.5, 
                       handletextpad=0.25, borderpad=0, frameon=False)

        # Save figure
        fig_fn = ('Temp_vs_lost_histogram_100scale_' + str(reg) + '_bydeg-separate.png')
        fig.set_size_inches(6,4)
        fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)

    
#%% ----- CALVING COMPARISON BY DEGREE ----- 
if option_calving_comparison_bydeg:
    
    regions = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    regions_calving = [1, 3, 4, 5, 7, 9, 17, 19]
    
    pickle_fp_nocalving = netcdf_fp_cmip5_land.replace('simulations','analysis') + 'pickle/'
    
    fig_fp = pickle_fp + '/../figures/'
    
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    fig_fp_calvingcompare = fig_fp + 'calving_compare/'
    if not os.path.exists(fig_fp_calvingcompare):
        os.makedirs(fig_fp_calvingcompare, exist_ok=True)
    
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_nocalving = {}
    reg_vol_all_bwl = {}
    reg_vol_all_bwl_nocalving = {}
#    reg_area_all = {} 
    
    reg_vol_all['all'] = {}
    reg_vol_all_bwl['all'] = {}
    reg_vol_all_nocalving['all'] = {}
    reg_vol_all_bwl_nocalving['all'] = {}
    for rcp in rcps:
        reg_vol_all['all'][rcp] = {}
        reg_vol_all_bwl['all'][rcp] = {}
        reg_vol_all_nocalving['all'][rcp] = {}
        reg_vol_all_bwl_nocalving['all'][rcp] = {}
        
        if 'rcp' in rcp:
            gcm_names = gcm_names_rcps
        elif 'ssp' in rcp:
            if rcp in ['ssp119']:
                gcm_names = gcm_names_ssp119
            else:
                gcm_names = gcm_names_ssps
        for gcm_name in gcm_names:
            reg_vol_all['all'][rcp][gcm_name] = None
            reg_vol_all_bwl['all'][rcp][gcm_name] = None
            reg_vol_all_nocalving['all'][rcp][gcm_name] = None
            reg_vol_all_bwl_nocalving['all'][rcp][gcm_name] = None
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_nocalving[reg] = {}
        reg_vol_all_bwl[reg] = {}
        reg_vol_all_bwl_nocalving[reg] = {}
#        reg_area_all[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_nocalving[reg][rcp] = {}
            reg_vol_all_bwl[reg][rcp] = {}
            reg_vol_all_bwl_nocalving[reg][rcp] = {}
#            reg_area_all[reg][rcp] = {}

            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----                    
                pickle_fp_land = (netcdf_fp_cmip5_land + '../analysis/pickle/' + str(reg).zfill(2) + 
                                  '/O1Regions/' + gcm_name + '/' + rcp + '/')
                pickle_fp_tw = pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                if '_calving' in netcdf_fp_cmip5 and not os.path.exists(pickle_fp_tw):
#                if '_calving' in netcdf_fp_cmip5 and reg in [2,6,8,10,11,12,13,14,15,16,18]:
                    pickle_fp_reg =  pickle_fp_land
                else:
                    pickle_fp_reg =  pickle_fp_tw
                    
                # Region string prefix
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl'
                fn_reg_vol_annual_bwl = reg_rcp_gcm_str + '_vol_annual_bwl.pkl'
                
                # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                # Volume below sea level
                with open(pickle_fp_reg + fn_reg_vol_annual_bwl, 'rb') as f:
                    reg_vol_annual_bwl = pickle.load(f)
                    
#                print(rcp, gcm_name, reg_vol_annual[0]/1e12)
                
                # ===== NO CALVING =====
                pickle_fp_reg_nocalving =  pickle_fp_nocalving + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                
#                print(os.path.exists(pickle_fp_reg_nocalving + fn_reg_vol_annual))
                
                if os.path.exists(pickle_fp_reg_nocalving + fn_reg_vol_annual):
                     # Volume
                    with open(pickle_fp_reg_nocalving + fn_reg_vol_annual, 'rb') as f:
                        reg_vol_annual_nocalving = pickle.load(f)
                    # Volume below sea level
                    with open(pickle_fp_reg_nocalving + fn_reg_vol_annual_bwl, 'rb') as f:
                        reg_vol_annual_bwl_nocalving = pickle.load(f)
                        
#                    print(reg, gcm_name, rcp, 'difference [%]', np.round((reg_vol_annual_nocalving[50] - reg_vol_annual[50])/ reg_vol_annual[0] * 100,2))
                
                else:
                    reg_vol_annual_nocalving = np.copy(reg_vol_annual)
                    reg_vol_annual_bwl_nocalving = np.copy(reg_vol_annual_bwl)
                
                if reg_vol_annual_bwl is None:
                    reg_vol_annual_bwl = np.zeros(reg_vol_annual.shape)
                    reg_vol_annual_bwl_nocalving = np.zeros(reg_vol_annual.shape)
                
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                reg_vol_all_nocalving[reg][rcp][gcm_name] = reg_vol_annual_nocalving
                
                reg_vol_all_bwl[reg][rcp][gcm_name] = reg_vol_annual_bwl
                reg_vol_all_bwl_nocalving[reg][rcp][gcm_name] = reg_vol_annual_bwl_nocalving
                
                # Global
                if reg_vol_all['all'][rcp][gcm_name] is None:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_vol_all_nocalving['all'][rcp][gcm_name] = reg_vol_all_nocalving[reg][rcp][gcm_name]
                    reg_vol_all_bwl_nocalving['all'][rcp][gcm_name] = reg_vol_all_bwl_nocalving[reg][rcp][gcm_name]
                else:
                    reg_vol_all['all'][rcp][gcm_name] = reg_vol_all['all'][rcp][gcm_name] + reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_all_bwl['all'][rcp][gcm_name] = reg_vol_all_bwl['all'][rcp][gcm_name] + reg_vol_all_bwl[reg][rcp][gcm_name]
                    reg_vol_all_nocalving['all'][rcp][gcm_name] = reg_vol_all_nocalving['all'][rcp][gcm_name] + reg_vol_all_nocalving[reg][rcp][gcm_name]
                    reg_vol_all_bwl_nocalving['all'][rcp][gcm_name] = reg_vol_all_bwl_nocalving['all'][rcp][gcm_name] + reg_vol_all_bwl_nocalving[reg][rcp][gcm_name]

    regions.append('all')

    # MULTI-GCM STATISTICS
    ds_multigcm_vol = {}
    ds_multigcm_vol_nocalving = {}
    ds_multigcm_vol_bsl = {}
    ds_multigcm_vol_bsl_nocalving = {}
#    ds_multigcm_area = {}
    for reg in regions:
        ds_multigcm_vol[reg] = {}
        ds_multigcm_vol_nocalving[reg] = {}
        ds_multigcm_vol_bsl[reg] = {}
        ds_multigcm_vol_bsl_nocalving[reg] = {}
#        ds_multigcm_area[reg] = {}
        for rcp in rcps: 
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for ngcm, gcm_name in enumerate(gcm_names):
                
#                print(rcp, gcm_name)
    
                reg_vol_gcm = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_gcm_nocalving = reg_vol_all_nocalving[reg][rcp][gcm_name]
                reg_vol_bsl_gcm = reg_vol_all_bwl[reg][rcp][gcm_name]
                reg_vol_bsl_gcm_nocalving = reg_vol_all_bwl_nocalving[reg][rcp][gcm_name]
#                reg_area_gcm = reg_area_all[reg][rcp][gcm_name]
    
                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm   
                    reg_vol_gcm_all_nocalving = reg_vol_gcm_nocalving
                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm
                    reg_vol_bsl_gcm_all_nocalving = reg_vol_bsl_gcm_nocalving  
#                    reg_area_gcm_all = reg_area_gcm    
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_vol_gcm_all_nocalving = np.vstack((reg_vol_gcm_all_nocalving, reg_vol_gcm_nocalving))
                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
                    reg_vol_bsl_gcm_all_nocalving = np.vstack((reg_vol_bsl_gcm_all_nocalving, reg_vol_bsl_gcm_nocalving))
#                    reg_area_gcm_all = np.vstack((reg_area_gcm_all, reg_area_gcm))
            
            ds_multigcm_vol[reg][rcp] = reg_vol_gcm_all
            ds_multigcm_vol_nocalving[reg][rcp] = reg_vol_gcm_all_nocalving
            ds_multigcm_vol_bsl[reg][rcp] = reg_vol_bsl_gcm_all
            ds_multigcm_vol_bsl_nocalving[reg][rcp] = reg_vol_bsl_gcm_all_nocalving
#            ds_multigcm_area[reg][rcp] = reg_area_gcm_all
            
    #%% MULTI-GCM PLOTS BY DEGREE
    # Set up temps
    temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
    temp_gcmrcp_dict = {}
    for deg_group in deg_groups:
        temp_gcmrcp_dict[deg_group] = []
    temp_dev_df['rcp_gcm_name'] = [temp_dev_df.loc[x,'Scenario'] + '/' + temp_dev_df.loc[x,'GCM'] for x in temp_dev_df.index.values]
    
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        for rcp_gcm_name in temp_dev_df_subset['rcp_gcm_name']:
            temp_gcmrcp_dict[deg_group].append(rcp_gcm_name)  

    # Set up regions
    reg_vol_all_bydeg = {}
    reg_vol_all_bydeg_nocalving = {}
    reg_vol_bsl_all_bydeg = {}
    reg_vol_bsl_all_bydeg_nocalving = {}
    for reg in regions:
        reg_vol_all_bydeg[reg] = {}
        reg_vol_all_bydeg_nocalving[reg] = {}
        reg_vol_bsl_all_bydeg[reg] = {}
        reg_vol_bsl_all_bydeg_nocalving[reg] = {}
        for deg_group in deg_groups:
            reg_vol_all_bydeg[reg][deg_group] = {}
            reg_vol_all_bydeg_nocalving[reg][deg_group] = {}
            reg_vol_bsl_all_bydeg[reg][deg_group] = {}
            reg_vol_bsl_all_bydeg_nocalving[reg][deg_group] = {}
         
    for reg in regions:
        for ngroup, deg_group in enumerate(deg_groups):
            for rcp_gcm_name in temp_gcmrcp_dict[deg_group]:
                
#                print('\n', reg, deg_group, rcp_gcm_name)
                
                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                reg_vol_annual = reg_vol_all[reg][rcp][gcm_name]
                reg_vol_annual_nocalving = reg_vol_all_nocalving[reg][rcp][gcm_name]
                reg_vol_bsl_annual = reg_vol_all_bwl[reg][rcp][gcm_name]
                reg_vol_bsl_annual_nocalving = reg_vol_all_bwl_nocalving[reg][rcp][gcm_name]

                reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_vol_annual
                reg_vol_all_bydeg_nocalving[reg][deg_group][rcp_gcm_name] = reg_vol_annual_nocalving
                reg_vol_bsl_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_vol_bsl_annual
                reg_vol_bsl_all_bydeg_nocalving[reg][deg_group][rcp_gcm_name] = reg_vol_bsl_annual_nocalving

    
    # MULTI-GCM STATISTICS
    ds_multigcm_vol_bydeg = {}
    ds_multigcm_vol_bydeg_nocalving = {}
    ds_multigcm_vol_bsl_bydeg = {}
    ds_multigcm_vol_bsl_bydeg_nocalving = {}
    for reg in regions:
        ds_multigcm_vol_bydeg[reg] = {}
        ds_multigcm_vol_bydeg_nocalving[reg] = {}
        ds_multigcm_vol_bsl_bydeg[reg] = {}
        ds_multigcm_vol_bsl_bydeg_nocalving[reg] = {}
        for deg_group in deg_groups:
            
            gcm_rcps_list = temp_gcmrcp_dict[deg_group]
            
            for ngcm, rcp_gcm_name in enumerate(gcm_rcps_list):

#                print(reg, deg_group, rcp_gcm_name)
                
                reg_vol_gcm = reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_vol_gcm_nocalving = reg_vol_all_bydeg_nocalving[reg][deg_group][rcp_gcm_name]
                reg_vol_bsl_gcm = reg_vol_bsl_all_bydeg[reg][deg_group][rcp_gcm_name]
                reg_vol_bsl_gcm_nocalving = reg_vol_bsl_all_bydeg_nocalving[reg][deg_group][rcp_gcm_name]


                if ngcm == 0:
                    reg_vol_gcm_all = reg_vol_gcm
                    reg_vol_gcm_all_nocalving = reg_vol_gcm_nocalving
                    reg_vol_bsl_gcm_all = reg_vol_bsl_gcm
                    reg_vol_bsl_gcm_all_nocalving = reg_vol_bsl_gcm_nocalving
                else:
                    reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                    reg_vol_gcm_all_nocalving = np.vstack((reg_vol_gcm_all_nocalving, reg_vol_gcm_nocalving))
                    reg_vol_bsl_gcm_all = np.vstack((reg_vol_bsl_gcm_all, reg_vol_bsl_gcm))
                    reg_vol_bsl_gcm_all_nocalving = np.vstack((reg_vol_bsl_gcm_all_nocalving, reg_vol_bsl_gcm_nocalving))
                    

            ds_multigcm_vol_bydeg[reg][deg_group] = reg_vol_gcm_all
            ds_multigcm_vol_bydeg_nocalving[reg][deg_group] = reg_vol_gcm_all_nocalving
            
            ds_multigcm_vol_bsl_bydeg[reg][deg_group] = reg_vol_bsl_gcm_all
            ds_multigcm_vol_bsl_bydeg_nocalving[reg][deg_group] = reg_vol_bsl_gcm_all_nocalving
            
#    print('check:', ds_multigcm_vol_bydeg['all'][1.5][0,0]/1e12)
#    print('check2:', ds_multigcm_vol_bsl_bydeg['all'][1.5][0,0]/1e12, ds_multigcm_vol_bsl_bydeg['all'][1.5][-1,-1]/1e12)
#    assert 1==0, 'here'
    
    #%% ----- STATISTICS OF INITIAL MASS AND DIFFERENCES -----
    normyear_idx = np.where(years == normyear)[0][0]
    
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms',
                          'vol_km3_2000_med_wfa', 'vol_km3_bsl_2000_med_wfa',
                          'vol_km3_2000_med_nofa', 'vol_km3_bsl_2000_med_nofa',
                          'vol_norm_%_2100_2015_med_wfa', 'vol_norm_%_2100_2015_med_nofa',
                          'vol_norm_2100_2015_dif%', 'vol_norm_dif%_max', 'vol_norm_dif%_max_yr',
                          'slr_cum_mmsle_2015_2100_med_wfa', 'slr_cum_mmsle_2015_2100_std_wfa',
                          'slr_cum_mmsle_2015_2100_med_nofa', 'slr_cum_mmsle_2015_2100_std_nofa',
                          'mass_gt_2000_med_wfa', 'mass_gt_2000_std_wfa',
                          'mass_gt_2000_med_nofa', 'mass_gt_2000_std_nofa',
                          'mass_2000_%dif_wfa',
                          'mass_bsl_gt_2000_med_wfa', 'mass_bsl_gt_2000_std_wfa',
                          'mass_bsl_gt_2000_med_nofa', 'mass_bsl_gt_2000_std_nofa',
                          'mass_bsl_2000_%dif_wfa',
                          'mass_gt_2015_med_wfa', 'mass_gt_2015_std_wfa',
                          'mass_gt_2015_med_nofa', 'mass_gt_2015_std_nofa',
                          'mass_2015_%dif_wfa',
                          'mass_bsl_gt_2015_med_wfa', 'mass_bsl_gt_2015_std_wfa',
                          'mass_bsl_gt_2015_med_nofa', 'mass_bsl_gt_2015_std_nofa',
                          'mass_bsl_2015_%dif_wfa']
    stats_overview_df = pd.DataFrame(np.zeros((len(regions)*len(deg_groups),len(stats_overview_cns))), columns=stats_overview_cns)
    regions_overview = regions
    if 'all' in regions:
        regions_overview = [regions[-1]] + regions[0:-1]
    ncount = 0
    for nreg, reg in enumerate(regions_overview):
        for ngroup, deg_group in enumerate(deg_groups):
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_bsl = ds_multigcm_vol_bsl_bydeg[reg][deg_group]
            
            reg_vol_nocalving = ds_multigcm_vol_bydeg_nocalving[reg][deg_group]
            reg_vol_bsl_nocalving = ds_multigcm_vol_bsl_bydeg_nocalving[reg][deg_group]
            
            reg_mass = reg_vol * pygem_prms.density_ice / 1e12
            reg_mass_bsl = reg_vol_bsl * pygem_prms.density_ice / 1e12
            reg_mass_nocalving = reg_vol_nocalving * pygem_prms.density_ice / 1e12
            reg_mass_bsl_nocalving = reg_vol_bsl_nocalving * pygem_prms.density_ice / 1e12
            
            # Cumulative Sea-level change [mm SLE]
            #  - accounts for water from glaciers below sea-level following Farinotti et al. (2019)
            #    for more detailed methods, see Fabien's blog post: https://nbviewer.jupyter.org/gist/jmalles/ca70090812e6499b34a22a3a7a7a8f2a
            reg_slr = slr_mmSLEyr(reg_vol, reg_vol_bsl)            
            reg_slr_cum_raw = np.cumsum(reg_slr, axis=1)
            reg_slr_cum = reg_slr_cum_raw - reg_slr_cum_raw[:,normyear_idx][:,np.newaxis]
            reg_slr_cum_med = np.median(reg_slr_cum, axis=0)
            reg_slr_cum_std = np.std(reg_slr_cum, axis=0)
            

            reg_slr_nocalving = slr_mmSLEyr(reg_vol_nocalving, reg_vol_bsl_nocalving)
            reg_slr_cum_raw_nocalving = np.cumsum(reg_slr_nocalving, axis=1)
            reg_slr_cum_nocalving = reg_slr_cum_raw_nocalving - reg_slr_cum_raw_nocalving[:,normyear_idx][:,np.newaxis]
            reg_slr_cum_med_nocalving = np.median(reg_slr_cum_nocalving, axis=0)
            reg_slr_cum_std_nocalving = np.std(reg_slr_cum_nocalving, axis=0)

            # Relative mass loss
            # Median and absolute median deviation
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_std = np.std(reg_vol, axis=0)
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_med[normyear_idx]
            
            # No calving
            reg_vol_med_nocalving = np.median(reg_vol_nocalving, axis=0)
            reg_vol_std_nocalving = np.std(reg_vol_nocalving, axis=0)
            reg_vol_med_norm_nocalving = reg_vol_med_nocalving / reg_vol_med_nocalving[normyear_idx]
            reg_vol_std_norm_nocalving = reg_vol_std_nocalving / reg_vol_med_nocalving[normyear_idx]
            
            # Difference
            reg_vol_norm_dif = (reg_vol_med_norm - reg_vol_med_norm_nocalving) * 100
            difmax_idx = np.where(np.abs(reg_vol_norm_dif[normyear_idx:]) == np.abs(reg_vol_norm_dif[normyear_idx:]).max())[0][0]
            reg_vol_norm_difmax = reg_vol_norm_dif[normyear_idx:][difmax_idx]
            reg_vol_norm_difmax_yr = years[normyear_idx:][difmax_idx]
            

            
            # RECORD STATISTICS
            stats_overview_df.loc[ncount,'Region'] = reg
            stats_overview_df.loc[ncount,'Scenario'] = deg_group
            stats_overview_df.loc[ncount,'n_gcms'] = reg_vol.shape[0]
            
            stats_overview_df.loc[ncount,'vol_km3_2000_med_wfa'] = np.median(reg_vol, axis=0)[0] / 1e9
            stats_overview_df.loc[ncount,'vol_km3_bsl_2000_med_wfa'] = np.median(reg_vol_bsl, axis=0)[0] / 1e9
            stats_overview_df.loc[ncount,'vol_km3_2000_med_nofa'] = np.median(reg_vol_nocalving, axis=0)[0] / 1e9
            stats_overview_df.loc[ncount,'vol_km3_bsl_2000_med_nofa'] = np.median(reg_vol_bsl_nocalving, axis=0)[0] / 1e9
            
            stats_overview_df.loc[ncount,'vol_norm_%_2100_2015_med_wfa'] = reg_vol_med_norm[-1] * 100
            stats_overview_df.loc[ncount,'vol_norm_%_2100_2015_med_nofa'] = reg_vol_med_norm_nocalving[-1] * 100
            
            stats_overview_df.loc[ncount,'vol_norm_dif%_max'] = reg_vol_norm_difmax
            stats_overview_df.loc[ncount,'vol_norm_dif%_max_yr'] = reg_vol_norm_difmax_yr
            
            
            stats_overview_df.loc[ncount,'slr_cum_mmsle_2015_2100_med_wfa'] = reg_slr_cum_med[-1]
            stats_overview_df.loc[ncount,'slr_cum_mmsle_2015_2100_std_wfa'] = reg_slr_cum_std[-1]
            stats_overview_df.loc[ncount,'slr_cum_mmsle_2015_2100_med_nofa'] = reg_slr_cum_med_nocalving[-1]
            stats_overview_df.loc[ncount,'slr_cum_mmsle_2015_2100_std_nofa'] = reg_slr_cum_std_nocalving[-1]
            
            
            stats_overview_df.loc[ncount,'mass_gt_2000_med_wfa'] = np.median(reg_mass, axis=0)[0]
            stats_overview_df.loc[ncount,'mass_gt_2000_std_wfa'] = np.std(reg_mass, axis=0)[0]
            stats_overview_df.loc[ncount,'mass_gt_2000_med_nofa'] = np.median(reg_mass_nocalving, axis=0)[0]
            stats_overview_df.loc[ncount,'mass_gt_2000_std_nofa'] = np.median(reg_mass_nocalving, axis=0)[0]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2000_med_wfa'] = np.median(reg_mass_bsl, axis=0)[0]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2000_std_wfa'] = np.std(reg_mass_bsl, axis=0)[0]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2000_med_nofa'] = np.median(reg_mass_bsl_nocalving, axis=0)[0]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2000_std_nofa'] = np.median(reg_mass_bsl_nocalving, axis=0)[0]
            
            stats_overview_df.loc[ncount,'mass_gt_2015_med_wfa'] = np.median(reg_mass, axis=0)[normyear_idx]
            stats_overview_df.loc[ncount,'mass_gt_2015_std_wfa'] = np.std(reg_mass, axis=0)[normyear_idx]
            stats_overview_df.loc[ncount,'mass_gt_2015_med_nofa'] = np.median(reg_mass_nocalving, axis=0)[normyear_idx]
            stats_overview_df.loc[ncount,'mass_gt_2015_std_nofa'] = np.median(reg_mass_nocalving, axis=0)[normyear_idx]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2015_med_wfa'] = np.median(reg_mass_bsl, axis=0)[normyear_idx]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2015_std_wfa'] = np.std(reg_mass_bsl, axis=0)[normyear_idx]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2015_med_nofa'] = np.median(reg_mass_bsl_nocalving, axis=0)[normyear_idx]
            stats_overview_df.loc[ncount,'mass_bsl_gt_2015_std_nofa'] = np.median(reg_mass_bsl_nocalving, axis=0)[normyear_idx]
            
            ncount += 1
        
    stats_overview_df['vol_km3_2000_added'] = stats_overview_df['vol_km3_2000_med_wfa'] - stats_overview_df['vol_km3_2000_med_nofa']
    stats_overview_df['vol_km3_bsl_2000_added'] = stats_overview_df['vol_km3_bsl_2000_med_wfa'] - stats_overview_df['vol_km3_bsl_2000_med_nofa']

    stats_overview_df['mass_2000_%dif_wfa'] = stats_overview_df['mass_gt_2000_med_wfa'] / stats_overview_df['mass_gt_2000_med_nofa'] * 100
    stats_overview_df['mass_bsl_2000_%dif_wfa'] = stats_overview_df['mass_bsl_gt_2000_med_wfa'] / stats_overview_df['mass_bsl_gt_2000_med_nofa'] * 100
    stats_overview_df['mass_2015_%dif_wfa'] = stats_overview_df['mass_gt_2015_med_wfa'] / stats_overview_df['mass_gt_2015_med_nofa'] * 100
    stats_overview_df['mass_bsl_2015_%dif_wfa'] = stats_overview_df['mass_bsl_gt_2015_med_wfa'] / stats_overview_df['mass_bsl_gt_2015_med_nofa'] * 100
    
    stats_overview_df['mass_bsl_%dif_of_change'] = (
            100 * (stats_overview_df['mass_bsl_gt_2000_med_wfa'] - stats_overview_df['mass_bsl_gt_2000_med_nofa']) / 
            (stats_overview_df['mass_gt_2000_med_wfa'] - stats_overview_df['mass_gt_2000_med_nofa']))
    
    
    # % below sea level
    stats_overview_df['bsl_%_2000_wfa'] = stats_overview_df['vol_km3_bsl_2000_med_wfa'] / stats_overview_df['vol_km3_2000_med_wfa'] * 100
    stats_overview_df['bsl_%_2000_nofa'] = stats_overview_df['vol_km3_bsl_2000_med_nofa'] / stats_overview_df['vol_km3_2000_med_nofa'] * 100
    
    # mm SLE statistics
    stats_overview_df['mm_sle_2000_wfa'] = ((stats_overview_df['vol_km3_2000_med_wfa'] * 1e9 * pygem_prms.density_ice / pygem_prms.density_water - 
                                             stats_overview_df['vol_km3_bsl_2000_med_wfa'] * 1e9) / pygem_prms.area_ocean * 1000)
    stats_overview_df['mm_sle_2000_nofa'] = ((stats_overview_df['vol_km3_2000_med_nofa'] * 1e9 * pygem_prms.density_ice / pygem_prms.density_water - 
                                             stats_overview_df['vol_km3_bsl_2000_med_nofa'] * 1e9) / pygem_prms.area_ocean * 1000)
    
    stats_overview_df['mm_sle_added'] = ((stats_overview_df['vol_km3_2000_added'] * 1e9 * pygem_prms.density_ice / pygem_prms.density_water - 
                                         stats_overview_df['vol_km3_bsl_2000_added'] * 1e9) / pygem_prms.area_ocean * 1000)

    stats_overview_df['slr_cum_mmsle_dif'] = (stats_overview_df['slr_cum_mmsle_2015_2100_med_wfa'] - 
                                              stats_overview_df['slr_cum_mmsle_2015_2100_med_nofa'])
    
    stats_overview_df['vol_norm_2100_2015_dif%'] = (stats_overview_df['vol_norm_%_2100_2015_med_wfa'] -
                                                    stats_overview_df['vol_norm_%_2100_2015_med_nofa'])

    stats_overview_df.to_csv(csv_fp + 'fa_stats_overview_bydeg.csv', index=False)
    

    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    for reg in regions_calving:
        fig = plt.figure()    
        gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        ax2 = fig.add_subplot(gs[2:3,0:2])
        ax3 = fig.add_subplot(gs[3:4,0:2])


        for deg_group in deg_groups:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_std = np.std(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_med[normyear_idx]

            # No calving
            reg_vol_nocalving = ds_multigcm_vol_bydeg_nocalving[reg][deg_group]
            reg_vol_med_nocalving = np.median(reg_vol_nocalving, axis=0)
            reg_vol_std_nocalving = np.std(reg_vol_nocalving, axis=0)
            
            reg_vol_med_norm_nocalving = reg_vol_med_nocalving / reg_vol_med_nocalving[normyear_idx]
            reg_vol_std_norm_nocalving = reg_vol_std_nocalving / reg_vol_med_nocalving[normyear_idx]
            
            # Delay in timing
            reg_vol_delay = np.zeros(reg_vol_med_norm_nocalving.shape)
            for nyear, year in enumerate(years):
#                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nocalving[nyear])
                year_idx = np.where(reg_vol_med_norm_nocalving[nyear] < reg_vol_med_norm)[0]
                if len(year_idx) > 0:
                    reg_vol_delay[nyear] = years[year_idx[-1]] - year

            # Plot
            ax1.plot(years, reg_vol_med_norm, color=temp_colordict[deg_group], linestyle='-', 
                    linewidth=1, zorder=4, label=deg_group)
            ax1.plot(years, reg_vol_med_norm_nocalving, color=temp_colordict[deg_group], linestyle=':', 
                    linewidth=1, zorder=3)
            ax2.plot(years, (reg_vol_med_norm - reg_vol_med_norm_nocalving)*100, color=temp_colordict[deg_group], 
                     linewidth=1, zorder=4)
            ax3.plot(years, reg_vol_delay, color=temp_colordict[deg_group], linewidth=1, zorder=4)
#            if deg_group in temps_plot_mad:
#                ax.fill_between(years, 
#                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
#                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
#                                alpha=0.2, facecolor=temp_colordict[deg_group], label=None)
        
        ax1.set_ylabel('Mass (rel. to 2015)')
        ax1.set_xlim(startyear, endyear)
        ax1.xaxis.set_major_locator(MultipleLocator(40))
        ax1.xaxis.set_minor_locator(MultipleLocator(10))
        ax1.set_ylim(0,1.1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.tick_params(axis='both', which='major', direction='inout', right=True)
        ax1.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Difference (%)')
        ax2.set_xlim(startyear, endyear)
        ax2.xaxis.set_major_locator(MultipleLocator(40))
        ax2.xaxis.set_minor_locator(MultipleLocator(10))
        if reg in [1,2,11]:
            ax2.set_ylim(0,7)
        elif reg in [18]:
            ax2.set_ylim(0,14)
        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax3.set_ylabel('Difference (yrs)')
        ax3.set_xlim(startyear, endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [1,2,11]:
#            ax2.set_ylim(0,7)
#        elif reg in [18]:
#            ax2.set_ylim(0,14)
#        ax2.yaxis.set_major_locator(MultipleLocator(5))
#        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax1.text(1, 1.16, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax1.transAxes)
        ax1.axes.xaxis.set_ticklabels([])
            
        # Save figure
        fig_fn = (str(reg).zfill(2) + '_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'calvingcompare_bydeg.png')
        fig.set_size_inches(4,4)
        fig.savefig(fig_fp_calvingcompare + fig_fn, bbox_inches='tight', dpi=300)
        
    #%%
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    ax7 = fig.add_subplot(gs[2,0])
    ax8 = fig.add_subplot(gs[2,1])
    
    regions_ordered = [1,3,4,5,7,9,17,19]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
        
        reg = regions_ordered[nax]  
    
        for deg_group in deg_groups:
            

            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_med = np.median(reg_vol, axis=0)
            reg_vol_std = np.std(reg_vol, axis=0)
            
            reg_vol_med_norm = reg_vol_med / reg_vol_med[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_med[normyear_idx]

            # No calving
            reg_vol_nocalving = ds_multigcm_vol_bydeg_nocalving[reg][deg_group]
            reg_vol_med_nocalving = np.median(reg_vol_nocalving, axis=0)
            reg_vol_std_nocalving = np.std(reg_vol_nocalving, axis=0)
            
            reg_vol_med_norm_nocalving = reg_vol_med_nocalving / reg_vol_med_nocalving[normyear_idx]
            reg_vol_std_norm_nocalving = reg_vol_std_nocalving / reg_vol_med_nocalving[normyear_idx]
            
            eoc_perc_chg = (reg_vol_med_norm_nocalving[-1] - reg_vol_med_norm[-1])*100
            if eoc_perc_chg > 0:
                eoc_perc_chg_str = '+' + str(np.round(eoc_perc_chg,1)) + '%'
            else:
                eoc_perc_chg_str = "\N{MINUS SIGN}" + str(np.round(np.abs(eoc_perc_chg),1)) + '%'
            print(reg, deg_group, eoc_perc_chg_str)
            
            # Delay in timing
            reg_vol_delay = np.zeros(reg_vol_med_norm_nocalving.shape)
            for nyear, year in enumerate(years):
#                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nocalving[nyear])
                year_idx = np.where(reg_vol_med_norm_nocalving[nyear] < reg_vol_med_norm)[0]
                if len(year_idx) > 0:
                    reg_vol_delay[nyear] = years[year_idx[-1]] - year
                    
            # Plot
            ax.plot(years, reg_vol_med_norm, color=temp_colordict[deg_group], linestyle='-', 
                    linewidth=1, zorder=4, label=deg_group)
            ax.plot(years, reg_vol_med_norm_nocalving, color=temp_colordict[deg_group], linestyle=':', 
                    linewidth=1, zorder=3, label=None)
            
            if ax in [ax1, ax4, ax7]:
                ax.set_ylabel('Mass (rel. to 2015)')
            ax.set_xlim(startyear, endyear)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xticks([2050,2100])
            ax.set_ylim(0,1.1)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            ax.text(1, 1.14, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax.transAxes)
            
            if deg_group in [1.5]:
                ypos = 0.46
            elif deg_group in [2]:
                ypos = 0.34
            elif deg_group in [3]:
                ypos = 0.22
            elif deg_group in [4]:
                ypos = 0.1
            ax.text(0.05,ypos, eoc_perc_chg_str, color=temp_colordict[deg_group], size=9, horizontalalignment='left', 
                    verticalalignment='center', transform=ax.transAxes)

    # Legend
    deg_labels = ['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C'
                  for x in np.arange(len(deg_groups))]
    
    leg_lines = []
    leg_labels = []
    line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
    leg_lines.append(line)
    leg_labels.append('Calving')
    line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=1)
    leg_lines.append(line)
    leg_labels.append('No calving')
    line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
    leg_lines.append(line)
    leg_labels.append(' ')
    line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
    leg_lines.append(line)
    leg_labels.append(' ')
    line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
    leg_lines.append(line)
    leg_labels.append(' ')
    
    for ngroup, deg_group in enumerate(deg_groups):
        line = Line2D([0,1],[0,1], color=temp_colordict[deg_group], linewidth=1)
        leg_lines.append(line)
        leg_labels.append(deg_labels[ngroup])
    
    ax8.legend(leg_lines, leg_labels, loc=(1.13,0.45), fontsize=9, labelspacing=0.2, handlelength=1, 
              handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.3, frameon=False)

    # Save figure
    fig_fn = ('allregions_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'calvingcompare_bydeg.png')
    fig.set_size_inches(6.5,5.5)
    fig.savefig(fig_fp_calvingcompare + fig_fn, bbox_inches='tight', dpi=300)
    
#%%
if option_glacier_cs_plots_calving_bydeg:
#    glac_nos = ['1.22193', '2.14297', '11.02739', '11.03005', '11.03643', '12.00080', 
#                '14.06794', '15.03733', '17.05076', '17.14140', '18.02342']
#    glac_nos = ['1.10689', '7.00238', '7.00240']
#    glac_nos = ['7.00242']
#    glac_nos = ['11.03005']
#    glac_nos = ['15.03733']
    glac_nos = ['18.02342']

    
    cleanice_fns = True
    debris_fns = True
    if debris_fns:
        cleanice_fns = False
    
#    # Clean ice filenames
#    if cleanice_fns:
#        for glac_no in glac_nos:
##            if int(glac_no.split('.')[0]) in [1,3,4,5,7,9,17,19]: 
#            netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/_calving/'
#            fa_label = ''
#    else:
#        # Calving filenames
#        netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v3/' # including calving
#        fa_label = 'Frontal ablation included'
#        
#    #%%
    
    # Clean ice filenames
    if cleanice_fns:
        for glac_no in glac_nos:
            if int(glac_no.split('.')[0]) in [1,3,4,5,7,9,17,19]: 
                netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations/_calving/' # treated as clean ice
                fa_label = ''
            else:
                netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/Output/simulations/' # treated as clean ice
                fa_label = ''
    elif debris_fns:
        netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations-ind/' # treated as clean ice
        fa_label = 'Debris included'
    else:
        # Calving filenames
        netcdf_fp_cmip5 = '/Users/drounce/Documents/HiMAT/spc_backup/simulations_calving_v5/' # including calving
        fa_label = 'Frontal ablation included'
    
    fig_fp_ind = fig_fp + 'ind_glaciers/'
    if not os.path.exists(fig_fp_ind):
        os.makedirs(fig_fp_ind)

    cs_year = 2000
    vol_norm_endyear = 2100
    startyear = 2015
    endyear = 2050
    years = np.arange(2000,2101+1)
    
    startyear_idx = np.where(years == startyear)[0][0]
    cs_idx = np.where(years == cs_year)[0][0]

    
    glac_name_dict = {'1.10689':'Columbia',
                      '1.22193':'Kahiltna (Denali)',
                      '2.14297':'Emmons (Rainier)',
                      '7.00027':'Basin-3 of Austfonna Ice Cap',
                      '7.00238': 'Storbreen',
                      '7.00240': 'Hansbreen',
                      '7.00242': 'Austre Torellbreen',
                      '11.02739':'Zmuttgletscher (Matterhorn)',
                      '11.03005':'Miage (Mont Blanc)',
                      '11.03643':'Mer de Glace (Mont Blanc)',
                      '12.00080':'Bolshoy Azau (Elbrus)',
                      '14.06794':'Baltoro (K2)',
                      '15.03733':'Khumbu',
                      '17.05076':'Viedma (Fitz Roy)',
                      '17.14140':'Horcones Inferior (Aconcagua)',
                      '18.02342':'Tasman'}
        

    # MULTI-GCM STATISTICS by degree
    # Set up temps
    temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
    temp_gcmrcp_dict = {}
    for deg_group in deg_groups:
        temp_gcmrcp_dict[deg_group] = []
    temp_dev_df['rcp_gcm_name'] = [temp_dev_df.loc[x,'Scenario'] + '/' + temp_dev_df.loc[x,'GCM'] for x in temp_dev_df.index.values]
    
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        for rcp_gcm_name in temp_dev_df_subset['rcp_gcm_name']:
            temp_gcmrcp_dict[deg_group].append(rcp_gcm_name)  

    # Set up processing
    glac_zbed_all = {}
    glac_thick_all = {}
    glac_zsurf_all = {}
    glac_vol_all = {}
    glac_multigcm_zbed = {}
    glac_multigcm_thick = {}
    glac_multigcm_zsurf = {}
    glac_multigcm_vol = {}
    for glac_no in glac_nos:
        
        print('\n\n', glac_no)

        gdir = single_flowline_glacier_directory(glac_no, logging_level='CRITICAL')
        
        tasks.init_present_time_glacier(gdir) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines')  # add debris enhancement factors to flowlines
        nfls = gdir.read_pickle('model_flowlines')
        
        
        x = np.arange(nfls[0].nx) * nfls[0].dx * nfls[0].map_dx
        
        glac_idx = np.nonzero(nfls[0].thick)[0]
        xmax = np.ceil(x[glac_idx].max()/1000+0.5)*1000
        
#        vol_m3_init = ds_binned.bin_volume_annual[0,:,0].values
#        thick_init = ds_binned.bin_thick_annual[0,:,0].values
#        widths_m = nfls[0].widths_m
#        lengths_m = vol_m3_init / thick_init / widths_m
                                
                                
        glac_zbed_all[glac_no] = {}
        glac_thick_all[glac_no] = {}
        glac_zsurf_all[glac_no] = {}
        glac_vol_all[glac_no] = {}
        
        for rcp in rcps:
            
            glac_zbed_all[glac_no][rcp] = {}
            glac_thick_all[glac_no][rcp] = {}
            glac_zsurf_all[glac_no][rcp] = {}
            glac_vol_all[glac_no][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                ds_binned_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                for i in os.listdir(ds_binned_fp):
                    if i.startswith(glac_no):
                        ds_binned_fn = i
                ds_stats_fp = netcdf_fp_cmip5 + glac_no.split('.')[0].zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                for i in os.listdir(ds_stats_fp):
                    if i.startswith(glac_no):
                        ds_stats_fn = i
                
                if glac_no in ds_stats_fn and rcp in ds_stats_fn and gcm_name in ds_stats_fn:
                    ds_binned = xr.open_dataset(ds_binned_fp + ds_binned_fn)
                    ds_stats = xr.open_dataset(ds_stats_fp + ds_stats_fn)
    
                    thick = ds_binned.bin_thick_annual[0,:,:].values
                    zsurf_init = ds_binned.bin_surface_h_initial[0].values
                    zbed = zsurf_init - thick[:,cs_idx]
                    vol = ds_stats.glac_volume_annual[0,:].values
                    
                    glac_thick_all[glac_no][rcp][gcm_name] = thick
                    glac_zbed_all[glac_no][rcp][gcm_name] = zbed
                    glac_zsurf_all[glac_no][rcp][gcm_name] = zbed[:,np.newaxis] + thick
                    glac_vol_all[glac_no][rcp][gcm_name] = vol

                else:
                    glac_thick_all[glac_no][rcp][gcm_name] = None
                    glac_zbed_all[glac_no][rcp][gcm_name] = None
                    glac_zsurf_all[glac_no][rcp][gcm_name] = None
                    glac_vol_all[glac_no][rcp][gcm_name] = None
         
            #%%
        # Set up regions
        glac_multigcm_zbed[glac_no] = {}
        glac_multigcm_thick[glac_no] = {}
        glac_multigcm_zsurf[glac_no] = {}
        glac_multigcm_vol[glac_no] = {}
         
        for deg_group in deg_groups:
            
            gcm_rcps_list = temp_gcmrcp_dict[deg_group]
            
            ngcm = 0
            for rcp_gcm_name in gcm_rcps_list:
                
#                print('\n', glac_no, deg_group, rcp_gcm_name)
                
                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                
                if rcp in rcps:
                
                    glac_zbed_gcm = glac_zbed_all[glac_no][rcp][gcm_name]
                    glac_thick_gcm = glac_thick_all[glac_no][rcp][gcm_name]
                    glac_zsurf_gcm = glac_zsurf_all[glac_no][rcp][gcm_name]
                    glac_vol_gcm = glac_vol_all[glac_no][rcp][gcm_name]
    
                    if not glac_vol_gcm is None:
                        if ngcm == 0:
                            glac_zbed_gcm_all = glac_zbed_gcm 
                            glac_thick_gcm_all = glac_thick_gcm[np.newaxis,:,:]
                            glac_zsurf_gcm_all = glac_zsurf_gcm[np.newaxis,:,:]
                            glac_vol_gcm_all = glac_vol_gcm[np.newaxis,:]
                        else:
                            glac_zbed_gcm_all = np.vstack((glac_zbed_gcm_all, glac_zbed_gcm))
                            glac_thick_gcm_all = np.vstack((glac_thick_gcm_all, glac_thick_gcm[np.newaxis,:,:]))
                            glac_zsurf_gcm_all = np.vstack((glac_zsurf_gcm_all, glac_zsurf_gcm[np.newaxis,:,:]))
                            glac_vol_gcm_all = np.vstack((glac_vol_gcm_all, glac_vol_gcm[np.newaxis,:]))
                        ngcm += 1

            glac_multigcm_zbed[glac_no][deg_group] = glac_zbed_gcm_all
            glac_multigcm_thick[glac_no][deg_group] = glac_thick_gcm_all
            glac_multigcm_zsurf[glac_no][deg_group] = glac_zsurf_gcm_all
            glac_multigcm_vol[glac_no][deg_group] = glac_vol_gcm_all
                
        #%% ----- FIGURE: VOLUME CHANGE MULTI-GCM -----
        temps_plot_mad = [1.5,4]
        zbed_deg_group = [deg_groups[0]]
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,0.65])
        ax.patch.set_facecolor('none')
#        ax2 = fig.add_axes([0,0.67,1,0.35])
#        ax2.patch.set_facecolor('none')
        ax3 = fig.add_axes([0.67,0.32,0.3,0.3])
        ax3.patch.set_facecolor('none')
        
        ymin, ymax, thick_max = None, None, None
        vol_med_all = []
        for ngroup, deg_group in enumerate(deg_groups):
            zbed_med = np.median(glac_multigcm_zbed[glac_no][deg_group],axis=0)
            zbed_std = np.std(glac_multigcm_zbed[glac_no][deg_group], axis=0)
            
            thick_med = np.median(glac_multigcm_thick[glac_no][deg_group],axis=0)
            thick_std = np.std(glac_multigcm_thick[glac_no][deg_group], axis=0)
            
            zsurf_med = np.median(glac_multigcm_zsurf[glac_no][deg_group],axis=0)
            zsurf_std = np.std(glac_multigcm_zsurf[glac_no][deg_group], axis=0)
            
            vol_med = np.median(glac_multigcm_vol[glac_no][deg_group],axis=0)
            vol_std = np.std(glac_multigcm_vol[glac_no][deg_group], axis=0)
            
            normyear_idx = np.where(years == normyear)[0][0]
            endyear_idx = np.where(years == endyear)[0][0]
            
            if deg_group in zbed_deg_group:
                ax.fill_between(x[1:]/1000, zbed_med[1:]-20, zbed_med[1:], color='white', zorder=5+len(deg_groups))
                ax.plot(x/1000, zbed_med[np.arange(len(x))],
                        color='k', linestyle='-', linewidth=1, zorder=5+len(deg_groups), label='zbed')
                ax.plot(x/1000, zsurf_med[np.arange(len(x)),normyear_idx], 
                             color='k', linestyle=':', linewidth=0.5, zorder=4+len(deg_groups), label=str(normyear))
#                ax2.plot(x/1000, thick_med[np.arange(len(x)),normyear_idx], 
#                         color='k', linestyle=':', linewidth=0.5, zorder=4, label=str(normyear))
                zbed_last = zbed_med
                add_zbed = False
                
            ax.plot(x/1000, zsurf_med[np.arange(len(x)),endyear_idx], 
                         color=temp_colordict[deg_group], linestyle='-', linewidth=0.5, zorder=4+(len(deg_groups)-ngroup), label=str(endyear))
            
#            ax2.plot(x/1000, thick_med[np.arange(len(x)),endyear_idx],
#                     color=temp_colordict[deg_group], linestyle='-', linewidth=0.5, zorder=4, label=str(endyear))
            
            ax3.plot(years, vol_med / vol_med[normyear_idx], color=temp_colordict[deg_group], 
                     linewidth=0.5, zorder=4, label=None)

            if deg_group in temps_plot_mad:
                ax3.fill_between(years, 
                                 (vol_med + vol_std)/vol_med[normyear_idx], 
                                 (vol_med - vol_std)/vol_med[normyear_idx],
                                 alpha=0.2, facecolor=temp_colordict[deg_group], label=None)
                
            # Record median value for printing
            vol_med_all.append(vol_med[normyear_idx])
            
            # ymin and ymax for bounds
            if ymin is None:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
                ymax = np.ceil(zsurf_med[:,endyear_idx].max()/100)*100
            if np.floor(zbed_med.min()/100)*100 < ymin:
                ymin = np.floor(zbed_med[glac_idx].min()/100)*100
            if np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100 > ymax:
                ymax = np.ceil(zsurf_med[glac_idx,endyear_idx].max()/100)*100
            # thickness max for bounds  
            if thick_max is None:
                thick_max = np.ceil(thick_med.max()/10)*10
            if np.ceil(thick_med.max()/10)*10 > thick_max:
                thick_max = np.ceil(thick_med.max()/10)*10
        
        if ymin < 0:
            water_idx = np.where(zbed_med < 0)[0]
            # Add water level
            ax.plot(x[water_idx]/1000, np.zeros(x[water_idx].shape), color='aquamarine', linewidth=1)
        
        if xmax/1000 > 25:
            x_major, x_minor = 10, 2
        elif xmax/1000 > 15:
            x_major, x_minor = 5, 1
        else:
            x_major, x_minor = 2, 0.5
        
        y_major, y_minor = 500,100
        
        if thick_max > 200:
            thick_major, thick_minor = 100, 20
        else:
            thick_major, thick_minor = 50, 10
            
            
        # ----- GLACIER SPECIFIC PLOTS -----
        plot_legend = True
        add_glac_name = True
        plot_sealevel = True
        leg_label = None
        if glac_no in ['1.10689']:
            thick_major, thick_minor = 200, 100
            ymin, ymax = -700, 3500
            y_major, y_minor = 1000, 200
            thick_max = 700
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
                plot_legend = False
#                add_glac_name = False
                leg_label = 'B'
            else:
                leg_label = 'A'
        elif glac_no in ['7.00238']:
            thick_max = 700
            thick_major, thick_minor = 200, 100
            ymin, ymax = -400, 1400
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
                leg_label = 'D'
            else:
                leg_label = 'C'
            plot_legend = False
#                add_glac_name = False
        elif glac_no in ['7.00240']:
            thick_max = 700
            thick_major, thick_minor = 200, 100
            ymin, ymax = -300, 650
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
                leg_label = 'F'
            else:
                leg_label = 'E'
            plot_legend = False
#                add_glac_name = False
        
        elif glac_no in ['7.00242']:
#            thick_max = 700
            thick_major, thick_minor = 200, 100
            ymin, ymax = -450, 1200
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
                leg_label = 'F'
            else:
                leg_label = 'E'
            plot_legend = False
#                add_glac_name = False
        
        elif glac_no in ['7.00027']:
            thick_max = 800
            thick_major, thick_minor = 200, 100
            ymin, ymax = -900, 1400
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Frontal ablation excluded'
            plot_legend = False
#                add_glac_name = False
        elif glac_no in ['15.03733']:
            ymin, ymax = 4600, 7800
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Debris excluded'
                plot_legend = False
                leg_label = 'B'
            else:
                fa_label = 'Debris included'
                plot_legend = True
                leg_label = 'A'
            plot_sealevel = False
        
        elif glac_no in ['18.02342']:
            ymin, ymax = 200, 3600
            y_major, y_minor = 500, 100
            if cleanice_fns:
                fa_label = 'Debris excluded'
                plot_legend = False
                leg_label = 'D'
            else:
                fa_label = 'Debris included'
                plot_legend = True
                leg_label = 'C'
            plot_sealevel = False
            
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0,xmax/1000)
#        ax2.set_xlim(0,xmax/1000)
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor)) 
#        ax2.set_ylim(0,thick_max)
#        ax2.yaxis.set_major_locator(MultipleLocator(thick_major))
##        ax2.yaxis.set_minor_locator(MultipleLocator(thick_minor))
#        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
#        ax2.get_xaxis().set_visible(False)
            
        ax.set_ylabel('Elevation (m a.s.l.)')
        ax.set_xlabel('Distance along flowline (km)')
#        ax2.set_ylabel('Ice thickness (m)', labelpad=10)
##        ax2.yaxis.set_label_position('right')
##        ax2.yaxis.tick_right()
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
#        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
#        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
#        ax.spines['top'].set_visible(False)
                
        if glac_no in glac_name_dict.keys():
            glac_name_text = glac_name_dict[glac_no]
        else:
             glac_name_text = glac_no
        
        if add_glac_name:
            ax.text(0.98, 1.02, glac_name_text, size=10, horizontalalignment='right', 
                    verticalalignment='bottom', transform=ax.transAxes)
        ax.text(0.02, 1.02, fa_label, size=10, horizontalalignment='left', 
                verticalalignment='bottom', transform=ax.transAxes)
        
        if not leg_label is None:
            ax.text(0.02, 0.98, leg_label, weight='bold', size=10, horizontalalignment='left', 
                    verticalalignment='top', transform=ax.transAxes)    
        
        ax3.set_ylabel('Mass (-)')
        ax3.set_xlim(normyear, vol_norm_endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
        ax3.set_ylim(0,1.1)
        ax3.yaxis.set_major_locator(MultipleLocator(0.5))
        ax3.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax3.tick_params(axis='both', which='major', direction='inout', right=True)
        ax3.tick_params(axis='both', which='minor', direction='in', right=True)
        vol_norm_gt = np.median(vol_med_all) * pygem_prms.density_ice / 1e12
        
        if vol_norm_endyear > endyear:
            ax3.axvline(endyear, color='k', linewidth=0.5, linestyle='--', zorder=4)
        
        if vol_norm_gt > 10:
            vol_norm_gt_str = str(int(np.round(vol_norm_gt,0))) + ' Gt'
        elif vol_norm_gt > 1:
            vol_norm_gt_str = str(np.round(vol_norm_gt,1)) + ' Gt'
        else:
            vol_norm_gt_str = str(np.round(vol_norm_gt,2)) + ' Gt'
        ax3.text(0.95, 0.95, vol_norm_gt_str, size=10, horizontalalignment='right', 
                verticalalignment='top', transform=ax3.transAxes)
        
        # Legend
        if plot_legend:
            leg_lines = []
            leg_labels = []
            deg_labels = ['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C'
                          for x in np.arange(len(deg_groups))]
            for ngroup, deg_group in enumerate(deg_groups):
                line = Line2D([0,1],[0,1], color=temp_colordict[deg_group], linewidth=1)
                leg_lines.append(line)
                leg_labels.append(deg_labels[ngroup])
            
            # add years
#            line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=0)
#            leg_lines.append(line)
#            leg_labels.append('')
            line = Line2D([0,1],[0,1], color='k', linestyle=':', linewidth=1)
            leg_lines.append(line)
            leg_labels.append(str(normyear))
            line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
            leg_lines.append(line)
            leg_labels.append(str(endyear))
            if plot_sealevel:
                line = Line2D([0,1],[0,1], color='aquamarine', linewidth=1)
                leg_lines.append(line)
                leg_labels.append('Sea level')
            line = Line2D([0,1],[0,1], color='k', linewidth=1)
            leg_lines.append(line)
            leg_labels.append('Bed')
            
            ax.legend(leg_lines, leg_labels, loc=(0.02,0.02), fontsize=8, labelspacing=0.25, handlelength=1, 
                      handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.5, frameon=False)
        
        if cleanice_fns:
            if int(glac_no.split('.')[0]) in [1,3,4,5,7,9,17,19]: 
                fn_str = '-nocalving'
            else:
                fn_str = '-nodebris'
        else:
            fn_str = ''
        
        # Save figure
        fig_fn = (glac_no + '_profile_' + str(endyear) + '_bydeg' + fn_str + '.png')
        fig.set_size_inches(4,3)
        fig.savefig(fig_fp_ind + fig_fn, bbox_inches='tight', dpi=300)
      
        
#%%
if option_tidewater_stats:
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                            rgi_regionsO1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], 
                            rgi_regionsO2='all', rgi_glac_number='all')
    
    termtype = main_glac_rgi_all.TermType.values
    tw_idx = [x for x in np.arange(0,main_glac_rgi_all.shape[0]) if main_glac_rgi_all.loc[x,'TermType'] in [1,5]]
    main_glac_rgi_tw = main_glac_rgi_all.loc[tw_idx, :]
    
    print('Globally', '% by area of tidewater glaciers:', main_glac_rgi_tw.Area.sum() / main_glac_rgi_all.Area.sum()*100)
    
    for reg in [1,3,4,5,7,9,17,19]:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                                rgi_regionsO1=[reg], 
                                rgi_regionsO2='all', rgi_glac_number='all')
    
        termtype = main_glac_rgi_all.TermType.values
        tw_idx = [x for x in np.arange(0,main_glac_rgi_all.shape[0]) if main_glac_rgi_all.loc[x,'TermType'] in [1,5]]
        main_glac_rgi_tw = main_glac_rgi_all.loc[tw_idx, :]
        
        print('Reg', reg, '% by area of tidewater glaciers:', np.round(main_glac_rgi_tw.Area.sum() / main_glac_rgi_all.Area.sum()*100,1))
        


#%%
if option_tidewater_fa_err:
    
#    regions = [19]
#    gcm_names_ssps = ['BCC-CSM2-MR']
#    rcps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    
    analysis_fp = netcdf_fp_cmip5.replace('simulations','analysis')
    fig_fp = analysis_fp + 'figures/'
    if not os.path.exists(fig_fp):
        os.makedirs(fig_fp, exist_ok=True)
    csv_fp = analysis_fp + 'csv/'
    if not os.path.exists(csv_fp):
        os.makedirs(csv_fp, exist_ok=True)
    pickle_fp = analysis_fp + 'pickle/'
    if not os.path.exists(pickle_fp):
        os.makedirs(pickle_fp, exist_ok=True)
    
    # Set up file
    fa_err_annual_gt_reg_dict = {}
    for reg in regions:
        fa_err_annual_gt_reg_dict[reg] = {}
    
    for reg in regions:
        # Load glaciers
        glacno_list = []
        for rcp in rcps:
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:
                
                # Filepath where glaciers are stored
                netcdf_fp = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                # Load the glaciers
                glacno_list_gcmrcp = []
                for i in os.listdir(netcdf_fp):
                    if i.endswith('.nc'):
                        glacno_list_gcmrcp.append(i.split('_')[0])
                glacno_list_gcmrcp = sorted(glacno_list_gcmrcp)
                
                print(gcm_name, rcp, 'simulated', len(glacno_list_gcmrcp), 'glaciers')
                
                # Only include the glaciers that were simulated by all GCM/RCP combinations
                if len(glacno_list) == 0:
                    glacno_list = glacno_list_gcmrcp
                else:
                    glacno_list = list(set(glacno_list).intersection(glacno_list_gcmrcp))
                glacno_list = sorted(glacno_list)
        
        # All glaciers for fraction
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[reg], 
                                                              rgi_regionsO2='all', rgi_glac_number='all', 
                                                              glac_no=None)
        # Glaciers with successful runs to process
        main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
        
        # Missing glaciers
        glacno_list_missing = sorted(np.setdiff1d(list(main_glac_rgi_all.glacno.values), glacno_list).tolist())
        if len(glacno_list_missing) > 0:
            main_glac_rgi_missing = modelsetup.selectglaciersrgitable(glac_no=glacno_list_missing)
        
        print('\nGCM/RCPs successfully simulated:\n  -', main_glac_rgi.shape[0], 'of', main_glac_rgi_all.shape[0], 'glaciers',
              '(', np.round(main_glac_rgi.shape[0]/main_glac_rgi_all.shape[0]*100,1),'%)')
        print('  -', np.round(main_glac_rgi.Area.sum(),0), 'km2 of', np.round(main_glac_rgi_all.Area.sum(),0), 'km2',
              '(', np.round(main_glac_rgi.Area.sum()/main_glac_rgi_all.Area.sum()*100,1),'%)')

        years = None
        for rcp in rcps:
            
            fa_err_annual_gt_reg_dict[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
            for gcm_name in gcm_names:

#                # Add Order 2 regions
#                regO2_list = list(np.unique(main_glac_rgi.O2Region))
#                for regO2 in regO2_list:
#                    fa_err_annual_gt_reg_dict[reg][rcp][gcm_name][regO2] = None
                
                # Tidewater glaciers
                # Filepath where glaciers are stored
                netcdf_fp_binned = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/binned/'
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                reg_fa_var_annual = None
                
                for nglac, glacno in enumerate(glacno_list):
                    if nglac%100 == 0:
                        print(gcm_name, rcp, glacno)
                            
                    # Load tidewater glacier ds
                    netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                    netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])
                    ds_stats = xr.open_dataset(netcdf_fp_stats + '/' + netcdf_fn_stats)
                    
                    # Years
                    if years is None:
                        years = ds_stats.year.values
                        normyear_idx = list(years).index(normyear)
                        
                    # Glacier frontal ablation error
                    #  Multiply by 1.4826 to account for nmad
                    glac_fa_nmad_annual = 1.4826*ds_stats.glac_frontalablation_monthly_mad.values[0,11::12]
                    glac_fa_var_annual = glac_fa_nmad_annual**2
                    
                    if reg_fa_var_annual is None:
                        reg_fa_var_annual = glac_fa_var_annual
                    else:
                        reg_fa_var_annual += glac_fa_var_annual
                    
                
                # Report the mean annual frontal ablation from 2015-2100 assuming perfect correlation in each region
                reg_fa_std_annual_gt = reg_fa_var_annual**0.5/1e9
                
                fa_err_annual_gt_reg_dict[reg][rcp][gcm_name] = reg_fa_std_annual_gt
                
                print('\n',gcm_name, rcp, 'mean fa gta:', np.round(np.mean(reg_fa_std_annual_gt),4),'\n')
                
                
    #%% MULTI-GCM PLOTS BY DEGREE
    # Set up temps
    temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
    temp_gcmrcp_dict = {}
    for deg_group in deg_groups:
        temp_gcmrcp_dict[deg_group] = []
    temp_dev_df['rcp_gcm_name'] = [temp_dev_df.loc[x,'Scenario'] + '/' + temp_dev_df.loc[x,'GCM'] for x in temp_dev_df.index.values]
    
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        for rcp_gcm_name in temp_dev_df_subset['rcp_gcm_name']:
            rcp = rcp = rcp_gcm_name.split('/')[0]
            if rcp in rcps:
                temp_gcmrcp_dict[deg_group].append(rcp_gcm_name)
    

    #%%
    # Set up regions
    ds_multigcm_fa_std_bydeg = {}
    for reg in regions:
        ds_multigcm_fa_std_bydeg[reg] = {}

        for ngroup, deg_group in enumerate(deg_groups):
            
            gcm_rcps_list = temp_gcmrcp_dict[deg_group]
            
            for ngcm, rcp_gcm_name in enumerate(gcm_rcps_list):
                
                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                
                if gcm_name in gcm_names:
                    reg_fa_std_annual = fa_err_annual_gt_reg_dict[reg][rcp][gcm_name]

                    if ngcm == 0:
                        reg_fa_std_all = reg_fa_std_annual
                    else:
                        reg_fa_std_all = np.vstack((reg_fa_std_all,reg_fa_std_annual))


            ds_multigcm_fa_std_bydeg[reg][deg_group] = reg_fa_std_all
    

    #%% Export data
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms',
                          'fa_gta_2015-2100_std', 'fa_gta_2000-2020_std', 'fa_gta_2080-2100_std']
    
    stats_overview_df = pd.DataFrame(np.zeros(((len(regions)+1)*len(deg_groups),len(stats_overview_cns))), columns=stats_overview_cns)
    
    ncount = 0
    for nreg, reg in enumerate(regions):
        for deg_group in deg_groups:
            
            reg_fa_gta_std_multigcm = ds_multigcm_fa_std_bydeg[reg][deg_group]
            reg_fa_gta_std_multigcm_med = np.median(reg_fa_gta_std_multigcm, axis=0)

            # RECORD STATISTICS
            stats_overview_df.loc[ncount,'Region'] = reg
            stats_overview_df.loc[ncount,'Scenario'] = deg_group
            stats_overview_df.loc[ncount,'n_gcms'] = reg_fa_gta_std_multigcm.shape[0]
            stats_overview_df.loc[ncount,'fa_gta_2015-2100_std'] = np.mean(reg_fa_gta_std_multigcm_med[normyear_idx:])
            stats_overview_df.loc[ncount,'fa_gta_2000-2020_std'] = np.mean(reg_fa_gta_std_multigcm_med[0:20])
            stats_overview_df.loc[ncount,'fa_gta_2080-2100_std'] = np.mean(reg_fa_gta_std_multigcm_med[-20:])
            
            ncount += 1
            
    for deg_group in deg_groups:
        fa_subset = stats_overview_df.loc[stats_overview_df['Scenario'] == deg_group]
        fa_all_std_2015_2100 = ((fa_subset['fa_gta_2015-2100_std']**2).sum())**0.5
        fa_all_std_2000_2020 = ((fa_subset['fa_gta_2000-2020_std']**2).sum())**0.5
        fa_all_std_2080_2100 = ((fa_subset['fa_gta_2080-2100_std']**2).sum())**0.5
        
        
        stats_overview_df.loc[ncount,'Region'] = 'all'
        stats_overview_df.loc[ncount,'Scenario'] = deg_group
        stats_overview_df.loc[ncount,'n_gcms'] = fa_subset.shape[0]
        stats_overview_df.loc[ncount,'fa_gta_2015-2100_std'] = fa_all_std_2015_2100
        stats_overview_df.loc[ncount,'fa_gta_2000-2020_std'] = fa_all_std_2000_2020
        stats_overview_df.loc[ncount,'fa_gta_2080-2100_std'] = fa_all_std_2080_2100
        
        ncount += 1

    stats_overview_df.to_csv(csv_fp + 'fa_annual_std_stats-bydeg.csv', index=False)


    

#%%
if option_debris_comparison_bydeg:
    
    regions = [1, 2, 11, 12, 13, 14, 15, 16, 18]
    
    pickle_fp = '/Users/drounce/Documents/HiMAT/spc_backup/analysis/pickle/'
    pickle_fp_nodebris = '/Users/drounce/Documents/HiMAT/spc_backup/analysis-nodebris/pickle-nodebris/'
    
    fig_fp = pickle_fp + '/../figures/'
    
    startyear = 2015
    endyear = 2100
    years = np.arange(2000,2101+1)
    
    temps_plot_mad = [1.5,4]
    
    fig_fp_debriscompare = fig_fp + 'debris_compare/'
    if not os.path.exists(fig_fp_debriscompare):
        os.makedirs(fig_fp_debriscompare, exist_ok=True)
    
    # Set up processing
    reg_vol_all = {}
    reg_vol_all_nodebris = {}
            
    for reg in regions:
    
        reg_vol_all[reg] = {}
        reg_vol_all_nodebris[reg] = {}
        
        for rcp in rcps:
            reg_vol_all[reg][rcp] = {}
            reg_vol_all_nodebris[reg][rcp] = {}
            
            if 'rcp' in rcp:
                gcm_names = gcm_names_rcps
            elif 'ssp' in rcp:
                if rcp in ['ssp119']:
                    gcm_names = gcm_names_ssp119
                else:
                    gcm_names = gcm_names_ssps
                
            for gcm_name in gcm_names:
                
                # ----- GCM/RCP PICKLE FILEPATHS AND FILENAMES -----
                pickle_fp_reg =  pickle_fp + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                reg_rcp_gcm_str = 'R' + str(reg) + '_' + rcp + '_' + gcm_name
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                
                 # Volume
                with open(pickle_fp_reg + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual = pickle.load(f)
                    
                # ===== NO DEBRIS =====
                pickle_fp_reg_nodebris =  pickle_fp_nodebris + str(reg).zfill(2) + '/O1Regions/' + gcm_name + '/' + rcp + '/'
                
                # Filenames
                fn_reg_vol_annual = reg_rcp_gcm_str + '_vol_annual.pkl' 
                
                 # Volume
                with open(pickle_fp_reg_nodebris + fn_reg_vol_annual, 'rb') as f:
                    reg_vol_annual_nodebris = pickle.load(f)
                    
                
                print(reg, gcm_name, rcp, 'difference [%]', np.round((reg_vol_annual_nodebris[50] - reg_vol_annual[50])/ reg_vol_annual[0] * 100,2))
                
                reg_vol_all[reg][rcp][gcm_name] = reg_vol_annual
                reg_vol_all_nodebris[reg][rcp][gcm_name] = reg_vol_annual_nodebris
                
    # ----- MULTI-GCM PLOTS BY DEGREE -----
    # Set up temps
    temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
    temp_gcmrcp_dict = {}
    for deg_group in deg_groups:
        temp_gcmrcp_dict[deg_group] = []
    temp_dev_df['rcp_gcm_name'] = [temp_dev_df.loc[x,'Scenario'] + '/' + temp_dev_df.loc[x,'GCM'] for x in temp_dev_df.index.values]
    
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        for rcp_gcm_name in temp_dev_df_subset['rcp_gcm_name']:
            temp_gcmrcp_dict[deg_group].append(rcp_gcm_name)  

    # Set up regions
    reg_vol_all_bydeg = {}
    reg_vol_all_bydeg_nodebris = {}
    for reg in regions:
        reg_vol_all_bydeg[reg] = {}
        reg_vol_all_bydeg_nodebris[reg] = {}
        for deg_group in deg_groups:
            reg_vol_all_bydeg[reg][deg_group] = {}
            reg_vol_all_bydeg_nodebris[reg][deg_group] = {}
         
    for reg in regions:
        for ngroup, deg_group in enumerate(deg_groups):
            for rcp_gcm_name in temp_gcmrcp_dict[deg_group]:
                
                print('\n', reg, deg_group, rcp_gcm_name)
                
                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                
                if rcp in rcps:    
                    reg_vol_annual = reg_vol_all[reg][rcp][gcm_name]
                    reg_vol_annual_nodebris = reg_vol_all_nodebris[reg][rcp][gcm_name]
    
                    reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name] = reg_vol_annual
                    reg_vol_all_bydeg_nodebris[reg][deg_group][rcp_gcm_name] = reg_vol_annual_nodebris

    
    # MULTI-GCM STATISTICS
    ds_multigcm_vol_bydeg = {}
    ds_multigcm_vol_bydeg_nodebris = {}
    for reg in regions:
        ds_multigcm_vol_bydeg[reg] = {}
        ds_multigcm_vol_bydeg_nodebris[reg] = {}
        for deg_group in deg_groups:
            
            print(deg_group)
            
            gcm_rcps_list = temp_gcmrcp_dict[deg_group]
            
            reg_vol_gcm_all = None
            for ngcm, rcp_gcm_name in enumerate(gcm_rcps_list):

#                print(reg, deg_group, rcp_gcm_name)
                
                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                                             
                if rcp in rcps:
                    
                    print(reg, deg_group, rcp_gcm_name)
                
                    reg_vol_gcm = reg_vol_all_bydeg[reg][deg_group][rcp_gcm_name]
                    reg_vol_gcm_nodebris = reg_vol_all_bydeg_nodebris[reg][deg_group][rcp_gcm_name]
    
    
                    if reg_vol_gcm_all is None:
                        reg_vol_gcm_all = reg_vol_gcm
                        reg_vol_gcm_all_nodebris = reg_vol_gcm_nodebris
                    else:
                        reg_vol_gcm_all = np.vstack((reg_vol_gcm_all, reg_vol_gcm))
                        reg_vol_gcm_all_nodebris = np.vstack((reg_vol_gcm_all_nodebris, reg_vol_gcm_nodebris))
                    

            ds_multigcm_vol_bydeg[reg][deg_group] = reg_vol_gcm_all
            ds_multigcm_vol_bydeg_nodebris[reg][deg_group] = reg_vol_gcm_all_nodebris
    
    #%% # Record statistics
    normyear_idx = np.where(years == normyear)[0][0]
    
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms', 
                          'dif_%_2100_med', 
#                          'dif_%_2100_std', 
                          'dif_%_max_med', 
#                          'dif_%_max_std'
                          'yr_max'
                            ]
    stats_overview_df = pd.DataFrame(np.zeros((len(regions)*len(deg_groups),len(stats_overview_cns))), columns=stats_overview_cns)
    
    ncount = 0
    for nreg, reg in enumerate(regions):
        for deg_group in deg_groups:
            
            # Load statistics
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_mean = np.median(reg_vol, axis=0)
            reg_vol_std = median_abs_deviation(reg_vol, axis=0)

            reg_vol_nodebris = ds_multigcm_vol_bydeg_nodebris[reg][deg_group]
            reg_vol_mean_nodebris = np.median(reg_vol_nodebris, axis=0)
            reg_vol_std_nodebris = median_abs_deviation(reg_vol_nodebris, axis=0)
            
            # Normalized
            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx] * 100
            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx] * 100
            
            reg_vol_mean_norm_nodebris = reg_vol_mean_nodebris / reg_vol_mean_nodebris[normyear_idx] * 100
            reg_vol_std_norm_nodebris = reg_vol_std_nodebris / reg_vol_mean_nodebris[normyear_idx] * 100
            
            reg_vol_mean_norm_dif = reg_vol_mean_norm - reg_vol_mean_norm_nodebris
            
            yr_max = np.where(reg_vol_mean_norm_dif == reg_vol_mean_norm_dif.max())[0][0]
            
#            reg_annual_twlr_all = reg_twlr_bydeg[reg][deg_group] 
#            reg_twcount_all = reg_twcount_bydeg[reg][deg_group]
#            
#            reg_annual_twlr_med = np.median(reg_annual_twlr_all, axis=0)
#            reg_annual_twlr_std = np.std(reg_annual_twlr_all, axis=0)
#            
#            reg_twcount_med = np.median(reg_twcount_all)
#            
            # RECORD STATISTICS
            stats_overview_df.loc[ncount,'Region'] = reg
            stats_overview_df.loc[ncount,'Scenario'] = deg_group
            stats_overview_df.loc[ncount,'n_gcms'] = reg_vol.shape[0]
            stats_overview_df.loc[ncount,'dif_%_2100_med'] = reg_vol_mean_norm_dif[-1]
#            stats_overview_df.loc[ncount,'dif_%_2100_std'] = 
            stats_overview_df.loc[ncount,'dif_%_max_med'] = reg_vol_mean_norm_dif[yr_max]
#            stats_overview_df.loc[ncount,'dif_%_max_std'] = 
            stats_overview_df.loc[ncount,'yr_max'] = years[yr_max]
            
            ncount += 1
    
    stats_overview_df.to_csv(csv_fp + 'debris_dif_stats.csv', index=False)
    
     
    #%% ----- FIGURE: ALL MULTI-GCM NORMALIZED VOLUME CHANGE -----
    
    for reg in regions:
        fig = plt.figure()    
        gs = fig.add_gridspec(nrows=6,ncols=4,wspace=0.3,hspace=0)
        ax1 = fig.add_subplot(gs[0:2,0:2])
        ax2 = fig.add_subplot(gs[2:3,0:2])
        ax3 = fig.add_subplot(gs[3:4,0:2])

        for deg_group in deg_groups:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_mean = np.median(reg_vol, axis=0)
            reg_vol_std = median_abs_deviation(reg_vol, axis=0)

            reg_vol_nodebris = ds_multigcm_vol_bydeg_nodebris[reg][deg_group]
            reg_vol_mean_nodebris = np.median(reg_vol_nodebris, axis=0)
            reg_vol_std_nodebris = median_abs_deviation(reg_vol_nodebris, axis=0)
            
            # Normalized
            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx]
            
            reg_vol_mean_norm_nodebris = reg_vol_mean_nodebris / reg_vol_mean_nodebris[normyear_idx]
            reg_vol_std_norm_nodebris = reg_vol_std_nodebris / reg_vol_mean_nodebris[normyear_idx]
            
            # Delay in timing
            reg_vol_delay = np.zeros(reg_vol_mean_norm_nodebris.shape)
            for nyear, year in enumerate(years):
#                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nodebris[nyear])
                year_idx = np.where(reg_vol_mean_norm_nodebris[nyear] < reg_vol_mean_norm)[0]
                if len(year_idx) > 0:
                    reg_vol_delay[nyear] = years[year_idx[-1]] - year

            # Plot
            ax1.plot(years, reg_vol_mean_norm, color=temp_colordict[deg_group], linestyle='-', 
                    linewidth=1, zorder=4, label=deg_group)
            ax1.plot(years, reg_vol_mean_norm_nodebris, color=temp_colordict[deg_group], linestyle=':', 
                    linewidth=1, zorder=3)
            ax2.plot(years, (reg_vol_mean_norm - reg_vol_mean_norm_nodebris)*100, color=temp_colordict[deg_group], 
                     linewidth=1, zorder=4)
            ax3.plot(years, reg_vol_delay, color=temp_colordict[deg_group], linewidth=1, zorder=4)
#            if rcp in rcps_plot_mad:
#                ax.fill_between(years, 
#                                reg_vol_med_norm + 1.96*reg_vol_mad_norm, 
#                                reg_vol_med_norm - 1.96*reg_vol_mad_norm, 
#                                alpha=0.2, facecolor=rcp_colordict[rcp], label=None)
        
        ax1.set_ylabel('Mass (rel. to 2015)')
        ax1.set_xlim(startyear, endyear)
        ax1.xaxis.set_major_locator(MultipleLocator(40))
        ax1.xaxis.set_minor_locator(MultipleLocator(10))
        ax1.set_ylim(0,1.1)
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.tick_params(axis='both', which='major', direction='inout', right=True)
        ax1.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Difference (%)')
        ax2.set_xlim(startyear, endyear)
        ax2.xaxis.set_major_locator(MultipleLocator(40))
        ax2.xaxis.set_minor_locator(MultipleLocator(10))
        if reg in [1,2,11]:
            ax2.set_ylim(0,7)
        elif reg in [18]:
            ax2.set_ylim(0,14)
        ax2.yaxis.set_major_locator(MultipleLocator(5))
        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax3.set_ylabel('Difference (yrs)')
        ax3.set_xlim(startyear, endyear)
        ax3.xaxis.set_major_locator(MultipleLocator(40))
        ax3.xaxis.set_minor_locator(MultipleLocator(10))
#        if reg in [1,2,11]:
#            ax2.set_ylim(0,7)
#        elif reg in [18]:
#            ax2.set_ylim(0,14)
#        ax2.yaxis.set_major_locator(MultipleLocator(5))
#        ax2.yaxis.set_minor_locator(MultipleLocator(1))
        ax2.tick_params(axis='both', which='major', direction='inout', right=True)
        ax2.tick_params(axis='both', which='minor', direction='in', right=True)
        
        ax1.text(1, 1.16, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax1.transAxes)
        ax1.axes.xaxis.set_ticklabels([])
            
        # Save figure
        fig_fn = (str(reg).zfill(2) + '_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'debriscompare_bydeg.png')
        fig.set_size_inches(4,4)
        fig.savefig(fig_fp_debriscompare + fig_fn, bbox_inches='tight', dpi=300)
        
    #%%
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    ax7 = fig.add_subplot(gs[2,0])
    ax8 = fig.add_subplot(gs[2,1])
    ax9 = fig.add_subplot(gs[2,2])
    
    reg_dc_area_perc_dict = {1:'9%',2:'5%',11:'10%',12:'26%',13:'6%',14:'10%',15:'17%',16:'16%',18:'18%'}
    reg_hd_dict = {1:'0.40 m',2:'0.28 m',11:'0.23 m',12:'0.32 m',13:'0.40 m',14:'0.36 m',15:'0.46 m',16:'0.24 m',18:'0.29 m'}
    
    regions_ordered = [1,2,11,13,14,15,16,12,18]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
        
        reg = regions_ordered[nax]            
    
        for deg_group in deg_groups:
            
            # Median and absolute median deviation
            reg_vol = ds_multigcm_vol_bydeg[reg][deg_group]
            reg_vol_mean = np.median(reg_vol, axis=0)
            reg_vol_std = median_abs_deviation(reg_vol, axis=0)

            reg_vol_nodebris = ds_multigcm_vol_bydeg_nodebris[reg][deg_group]
            reg_vol_mean_nodebris = np.median(reg_vol_nodebris, axis=0)
            reg_vol_std_nodebris = median_abs_deviation(reg_vol_nodebris, axis=0)
            
            # Normalized
            reg_vol_mean_norm = reg_vol_mean / reg_vol_mean[normyear_idx]
            reg_vol_std_norm = reg_vol_std / reg_vol_mean[normyear_idx]
            
            reg_vol_mean_norm_nodebris = reg_vol_mean_nodebris / reg_vol_mean_nodebris[normyear_idx]
            reg_vol_std_norm_nodebris = reg_vol_std_nodebris / reg_vol_mean_nodebris[normyear_idx]
            
            # Delay in timing
            reg_vol_delay = np.zeros(reg_vol_mean_norm_nodebris.shape)
            for nyear, year in enumerate(years):
#                print(nyear, year, reg_vol_mean_norm[nyear], reg_vol_mean_norm_nodebris[nyear])
                year_idx = np.where(reg_vol_mean_norm_nodebris[nyear] < reg_vol_mean_norm)[0]
                if len(year_idx) > 0:
                    reg_vol_delay[nyear] = years[year_idx[-1]] - year

            # Plot
            ax.plot(years, reg_vol_mean_norm, color=temp_colordict[deg_group], linestyle='-', 
                    linewidth=1, zorder=4, label=deg_group)
            ax.plot(years, reg_vol_mean_norm_nodebris, color=temp_colordict[deg_group], linestyle=':', 
                    linewidth=1, zorder=3, label=None)
        
        if ax in [ax1, ax4, ax7]:
            ax.set_ylabel('Mass (rel. to 2015)')
        ax.set_xlim(startyear, endyear)
#        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xticks([2050,2100])
        ax.set_ylim(0,1.1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='both', which='major', direction='inout', right=True)
        ax.tick_params(axis='both', which='minor', direction='in', right=True)
        
        # Text
        ax.text(0.05,0.15, reg_dc_area_perc_dict[reg], size=10, horizontalalignment='left', 
                 verticalalignment='bottom', transform=ax.transAxes)
        ax.text(0.05,0.02, reg_hd_dict[reg], size=10, horizontalalignment='left', 
                 verticalalignment='bottom', transform=ax.transAxes)
        
        # Legend
        if ax == ax3:            
            leg_lines = []
            leg_labels = []
            line = Line2D([0,1],[0,1], color='grey', linestyle='-', linewidth=1)
            leg_lines.append(line)
            leg_labels.append('Debris')
            line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=1)
            leg_lines.append(line)
            leg_labels.append('Clean')
            line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
            leg_lines.append(line)
            leg_labels.append(' ')
            line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
            leg_lines.append(line)
            leg_labels.append(' ')
            line = Line2D([0,1],[0,1], color='grey', linestyle=':', linewidth=0)
            leg_lines.append(line)
            leg_labels.append(' ')
            
            deg_labels = ['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C'
                          for x in np.arange(len(deg_groups))]
            for ngroup, deg_group in enumerate(deg_groups):
                line = Line2D([0,1],[0,1], color=temp_colordict[deg_group], linewidth=1)
                leg_lines.append(line)
                leg_labels.append(deg_labels[ngroup])
            
            ax.legend(leg_lines, leg_labels, loc=(0.13,0.38), fontsize=8, labelspacing=0.2, handlelength=1, 
                      handletextpad=0.25, borderpad=0, ncol=2, columnspacing=0.3, frameon=False)

        ax.text(1, 1.14, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                 verticalalignment='top', transform=ax.transAxes)
    
    # Save figure
    fig_fn = ('allregions_volchange_norm_' + str(startyear) + '-' + str(endyear) + '_' + 'debriscompare_bydeg.png')
    fig.set_size_inches(6.5,5.5)
    fig.savefig(fig_fp_debriscompare + fig_fn, bbox_inches='tight', dpi=300)
    
    
    
    
    
#%%
if option_tidewater_landretreat:
    
    # Global temperature increase dataframe
    temp_dev_df = pd.read_csv(csv_fp + temp_dev_fn)
    
    temps_plot_mad = [1.5,4]

    temp_colordict = {}
    for ngroup, deg_group in enumerate(deg_groups):
        temp_colordict[deg_group] = deg_group_colors[ngroup]

    fig_fp_multigcm = fig_fp + 'multi_gcm/'
    

    # ----- PROCESS DATA -----
    # Set up processing
    # Regional tidewater land retreat by degree dictionary
    # - 0s and 1s for when in water and when retreat onto land
    reg_twlr_bydeg = {}
    reg_twcount_bydeg = {}
    
    # Set up Global region
    reg_twlr_bydeg['all'] = {}
    reg_twcount_bydeg['all'] = {}
    temp_gcmrcp_dict = {}
    for deg_group in deg_groups:
        temp_gcmrcp_dict[deg_group] = []
    temp_dev_df['rcp_gcm_name'] = [temp_dev_df.loc[x,'Scenario'] + '/' + temp_dev_df.loc[x,'GCM'] for x in temp_dev_df.index.values]
    
    for ngroup, deg_group in enumerate(deg_groups):
        deg_group_bnd = deg_groups_bnds[ngroup]
        temp_dev_df_subset = temp_dev_df.loc[(temp_dev_df.global_mean_deviation_degC >= deg_group - deg_group_bnd) &
                                             (temp_dev_df.global_mean_deviation_degC < deg_group + deg_group_bnd),:]
        for rcp_gcm_name in temp_dev_df_subset['rcp_gcm_name']:
            temp_gcmrcp_dict[deg_group].append(rcp_gcm_name)
    
    # Set up regions
    for reg in regions:
        reg_twlr_bydeg[reg] = {}
        reg_twcount_bydeg[reg] = {}
         
    for reg in regions:
        for ngroup, deg_group in enumerate(deg_groups):
            
            reg_annual_twlr_all = None
            reg_twcount_all = []
            for rcp_gcm_name in temp_gcmrcp_dict[deg_group]:

                rcp = rcp_gcm_name.split('/')[0]
                gcm_name = rcp_gcm_name.split('/')[1]
                
                # Load glaciers and process
                print(reg, deg_group, rcp, gcm_name)
                
                # Filepath where glaciers are stored
                netcdf_fp_stats = netcdf_fp_cmip5 + str(reg).zfill(2) + '/' + gcm_name + '/' + rcp + '/stats/'
                
                glacno_list = []
                for i in os.listdir(netcdf_fp_stats):
                    if i.endswith('.nc'):
                        glacno_list.append(i.split('_')[0])
                glacno_list_sorted = sorted(glacno_list)
                
                ncount_tw_glaciers = 0
                reg_annual_twlr = None
                for glacno in glacno_list:

                    netcdf_fn_stats_ending = 'MCMC_ba1_50sets_2000_2100_all.nc'
                    netcdf_fn_stats = '_'.join([glacno, gcm_name, rcp, netcdf_fn_stats_ending])

                    ds_stats = xr.open_dataset(netcdf_fp_stats + netcdf_fn_stats)
                    
                    if ds_stats.glac_frontalablation_monthly.sum() > 0:
                        ncount_tw_glaciers += 1
                        
                        # Month when frontal ablation becomes zero
                        fa_annual = ds_stats.glac_frontalablation_monthly.values[0,11::12]
                        
                        fa_annual_twlr = -1*fa_annual
                        fa_annual_twlr[fa_annual == 0] = 1
                        fa_annual_twlr[fa_annual_twlr < 0] = 0
                        
                        if reg_annual_twlr is None:
                            reg_annual_twlr = fa_annual_twlr
                        else:
                            reg_annual_twlr += fa_annual_twlr
                   
                # Record annual tidewater land retreat binary 
                if reg_annual_twlr_all is None:
                    reg_annual_twlr_all = reg_annual_twlr[np.newaxis,:]
                else:
                    reg_annual_twlr_all = np.vstack((reg_annual_twlr_all, reg_annual_twlr[np.newaxis,:]))
                    
                # Record number of tidewater glaciers simulated
                reg_twcount_all.append(ncount_tw_glaciers)
            
            # Record array for all GCMs/scenarios 
            reg_twlr_bydeg[reg][deg_group] = np.copy(reg_annual_twlr_all)
            reg_twcount_bydeg[reg][deg_group] = np.copy(np.array(reg_twcount_all))
            

    # Output stats for all
    for ngroup, deg_group in enumerate(deg_groups):
        for nreg, reg in enumerate(regions):
            reg_annual_twlr_all = reg_twlr_bydeg[reg][deg_group] 
            reg_twcount_all = reg_twcount_bydeg[reg][deg_group]
            
            if nreg == 0:
                allregs_annual_twlr_all = np.copy(reg_annual_twlr_all)
                allregs_twcount = np.copy(reg_twcount_all)
            else:
                allregs_annual_twlr_all += np.copy(reg_annual_twlr_all)
                allregs_twcount += np.copy(reg_twcount_all)
        
        reg_twlr_bydeg['all'][deg_group] = allregs_annual_twlr_all
        reg_twcount_bydeg['all'][deg_group] = allregs_twcount
        
    
    regions.append('all')
    
    #%%
    # Record statistics
    stats_overview_cns = ['Region', 'Scenario', 'n_gcms', 
                          'tw_count',
                          'tw_count_landretreat_2100_med', 'tw_count_landretreat_2100_std']
    stats_overview_df = pd.DataFrame(np.zeros((len(regions)*len(deg_groups),len(stats_overview_cns))), columns=stats_overview_cns)
    
    ncount = 0
    for nreg, reg in enumerate(regions):
        for deg_group in deg_groups:
            
            # Load statistics
            reg_annual_twlr_all = reg_twlr_bydeg[reg][deg_group] 
            reg_twcount_all = reg_twcount_bydeg[reg][deg_group]
            
            reg_annual_twlr_med = np.median(reg_annual_twlr_all, axis=0)
            reg_annual_twlr_std = np.std(reg_annual_twlr_all, axis=0)
            
            reg_twcount_med = np.median(reg_twcount_all)
            
            # RECORD STATISTICS
            stats_overview_df.loc[ncount,'Region'] = reg
            stats_overview_df.loc[ncount,'Scenario'] = deg_group
            stats_overview_df.loc[ncount,'n_gcms'] = reg_twcount_all.shape[0]
            stats_overview_df.loc[ncount,'tw_count'] = reg_twcount_med
            stats_overview_df.loc[ncount,'tw_count_landretreat_2100_med'] = reg_annual_twlr_med[-1]
            stats_overview_df.loc[ncount,'tw_count_landretreat_2100_std'] = reg_annual_twlr_std[-1]
            
            ncount += 1
    
    stats_overview_df['tw_%_landretreat_2100_med'] = (100 *
            stats_overview_df['tw_count_landretreat_2100_med'] / stats_overview_df['tw_count'])
    stats_overview_df['tw_%_landretreat_2100_std'] = (100 *
            stats_overview_df['tw_count_landretreat_2100_std'] / stats_overview_df['tw_count'])
    
    stats_overview_df.to_csv(csv_fp + 'tidewater_landretreat_stats.csv', index=False)
    
    
    #%% # ----- Regional % of Tidewater Glaciers that retreated onto land -----
    years = np.arange(2000,2101)
    
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3,ncols=3,wspace=0.3,hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0])
    ax5 = fig.add_subplot(gs[1,1])
    ax6 = fig.add_subplot(gs[1,2])
    ax7 = fig.add_subplot(gs[2,0])
    ax8 = fig.add_subplot(gs[2,1])
    
    regions_ordered = [1,3,4,5,7,9,17,19]
    for nax, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]):
        
        reg = regions_ordered[nax]  
    
        ncount_by_deg = []
        for ngroup, deg_group in enumerate(deg_groups):
            

            # Median and absolute median deviation
            # Load statistics
            reg_annual_twlr_all = reg_twlr_bydeg[reg][deg_group] 
            reg_twcount_all = reg_twcount_bydeg[reg][deg_group]
            
            ncount_by_deg.append(reg_twlr_bydeg[reg][deg_group].shape[0])
            
            reg_annual_twlr_med = np.median(reg_annual_twlr_all, axis=0)
            reg_annual_twlr_std = np.std(reg_annual_twlr_all, axis=0)
            
            reg_twcount_med = np.median(reg_twcount_all)
            
            reg_annual_twlr_norm_med = reg_annual_twlr_med / reg_twcount_med * 100
            reg_annual_twlr_norm_std = reg_annual_twlr_std / reg_twcount_med * 100
            
                    
            # Plot
            if nax == 0:
                label=deg_group
            else:
                label=None
            ax.plot(years, reg_annual_twlr_norm_med, color=temp_colordict[deg_group], linestyle='-', 
                    linewidth=1, zorder=4, label=label)
            if deg_group in temps_plot_mad:
                ax.fill_between(years, 
                                reg_annual_twlr_norm_med + 1.96*reg_annual_twlr_norm_std, 
                                reg_annual_twlr_norm_med - 1.96*reg_annual_twlr_norm_std, 
                                alpha=0.2, facecolor=temp_colordict[deg_group], label=None, zorder=1)
            
            # Glacier count
            if ngroup == 0:
                ax.text(0.05, 0.95, str(int(np.round(reg_twcount_med))), size=10, horizontalalignment='left', 
                        verticalalignment='top', transform=ax.transAxes)
            
            ax.set_xlim(2015, 2100)
#            ax.xaxis.set_major_locator(MultipleLocator(40))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xticks([2050,2100])
            ax.set_ylim(0,100)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            ax.tick_params(axis='both', which='major', direction='inout', right=True)
            ax.tick_params(axis='both', which='minor', direction='in', right=True)

            ax.text(1, 1.14, rgi_reg_dict[reg], size=10, horizontalalignment='right', 
                     verticalalignment='top', transform=ax.transAxes)

    # Legend
    labels=['+' + format(deg_groups[x],'.1f') + u'\N{DEGREE SIGN}' + 'C' + ' (n=' + str(ncount_by_deg[x]) + ')'
            for x in np.arange(len(deg_groups))]
    ax8.legend(labels=labels, loc=(1.2,0.45), fontsize=10, labelspacing=0.2, handlelength=1, 
               handletextpad=0.25, borderpad=0, ncol=1, columnspacing=0.3, frameon=False)

    fig.text(0.05,0.5,'Tidewater glaciers (%)', size=12, horizontalalignment='center', verticalalignment='center', rotation=90)

    # Save figure
    fig_fn = ('allregions_tidewater_landretreat_2015-2100_bydeg.png')
    fig.set_size_inches(6.5,5.5)
    fig.savefig(fig_fp_multigcm + fig_fn, bbox_inches='tight', dpi=300)