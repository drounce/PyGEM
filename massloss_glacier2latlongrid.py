""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
from collections import OrderedDict
import datetime
import glob
import os
import pickle
# External libraries
import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import EngFormatter
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.ndimage import uniform_filter
import scipy
#from scipy import stats
#from scipy.stats.kde import gaussian_kde
#from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.stats import uniform
#from scipy.stats import linregress
#from scipy.stats import lognorm
#from scipy.optimize import minimize
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem.pygem_input as pygem_prms
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup
import run_calibration as calibration


option_mass_bydeg = 1

#%% ===== Input data =====
netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/'

regions = [13, 14, 15]

# GCMs and RCP scenarios
#gcm_names = ['bcc-csm1-1', 'CanESM2', 'CESM1-CAM5', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'FGOALS-g2', 'GFDL-CM3', 
#             'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'MIROC-ESM', 
#             'MIROC-ESM-CHEM', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MRI-CGCM3', 'NorESM1-M', 'NorESM1-ME']
rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']

# Grouping
grouping = 'degree'
degree_size = 0.1

#%% ===== FUNCTIONS =====
def pickle_data(fn, data):
    """Pickle data
    
    Parameters
    ----------
    fn : str
        filename including filepath
    data : list, etc.
        data to be pickled
    
    Returns
    -------
    .pkl file
        saves .pkl file of the data
    """
    with open(fn, 'wb') as f:
        pickle.dump(data, f)
        

def select_groups(grouping, main_glac_rgi_all):
    """
    Select groups based on grouping
    """
    if grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = sorted(groups)
    return groups, group_cn


def load_glacier_data(glac_no=None, rgi_regionsO1=None, rgi_regionsO2='all', rgi_glac_number='all',
                      load_caldata=0, startyear=2000, endyear=2018, option_wateryear=3):
    """
    Load glacier data (main_glac_rgi, hyps, and ice thickness)
    """
    # Load glaciers
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 =rgi_regionsO2, rgi_glac_number=rgi_glac_number,  
            glac_no=glac_no)

    # Glacier hypsometry [km**2], total area
    main_glac_hyps_all = modelsetup.import_Husstable(main_glac_rgi_all, pygem_prms.hyps_filepath, pygem_prms.hyps_filedict,
                                                     pygem_prms.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness_all = modelsetup.import_Husstable(main_glac_rgi_all, pygem_prms.thickness_filepath,
                                                             pygem_prms.thickness_filedict, pygem_prms.thickness_colsdrop)
    
    # Additional processing
    main_glac_hyps_all[main_glac_icethickness_all == 0] = 0
    main_glac_hyps_all = main_glac_hyps_all.fillna(0)
    main_glac_icethickness_all = main_glac_icethickness_all.fillna(0)
    
    # Add degree groups to main_glac_rgi_all
    # Degrees
    main_glac_rgi_all['CenLon_round'] = np.floor(main_glac_rgi_all.CenLon.values/degree_size) * degree_size
    main_glac_rgi_all['CenLat_round'] = np.floor(main_glac_rgi_all.CenLat.values/degree_size) * degree_size
    deg_groups = main_glac_rgi_all.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
    deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
    main_glac_rgi_all.reset_index(drop=True, inplace=True)
    cenlon_cenlat = [(main_glac_rgi_all.loc[x,'CenLon_round'], main_glac_rgi_all.loc[x,'CenLat_round']) 
                     for x in range(len(main_glac_rgi_all))]
    main_glac_rgi_all['CenLon_CenLat'] = cenlon_cenlat
    main_glac_rgi_all['deg_id'] = main_glac_rgi_all.CenLon_CenLat.map(deg_dict)
    
    if load_caldata == 1:
        cal_datasets = ['shean']
        startyear=2000
        dates_table = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0, 
                                               option_wateryear=option_wateryear)
        # Calibration data
        cal_data_all = pd.DataFrame()
        for dataset in cal_datasets:
            cal_subset = class_mbdata.MBData(name=dataset)
            cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_all, main_glac_hyps_all, dates_table)
            cal_data_all = cal_data_all.append(cal_subset_data, ignore_index=True)
        cal_data_all = cal_data_all.sort_values(['glacno', 't1_idx'])
        cal_data_all.reset_index(drop=True, inplace=True)
    
    if load_caldata == 0:
        return main_glac_rgi_all, main_glac_hyps_all, main_glac_icethickness_all
    else:
        return main_glac_rgi_all, main_glac_hyps_all, main_glac_icethickness_all, cal_data_all


#%%             
# ===== Time series of glacier mass grouped by degree ======
if option_mass_bydeg == 1:
    startyear = 2000
    endyear = 2100
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi) 
            
    #%%
    # Glacier and grouped annual specific mass balance and mass change
    ds_multi = {}
    for rcp in rcps:
#    for rcp in ['rcp85']:
        for region in regions:
            
            # Load datasets
            ds_fn = 'R' + str(region) + '_multimodel_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
            
            print(ds_fn)
            
            ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
            df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
            df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                           str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
            
            # Extract time variable
            time_values_annual = ds.coords['year_plus1'].values
            time_values_monthly = ds.coords['time'].values        
            
            # Convert mass balance to monthly mass change
            mb_monthly = ds['massbaltotal_glac_monthly'].values[:,:,0]
            area_annual = ds.area_glac_annual[:,:,0].values
            area_monthly = area_annual[:,0:-1].repeat(12,axis=1)
            masschg_monthly_Gt_raw = mb_monthly / 1000 * area_monthly
            masschg_annual_Gt_raw = (masschg_monthly_Gt_raw.reshape(-1,12).sum(1)
                                     .reshape(masschg_monthly_Gt_raw.shape[0], int(masschg_monthly_Gt_raw.shape[1]/12)))
            
            vol_annual_Gt = ds['volume_glac_annual'].values[:,:,0] * pygem_prms.density_ice / pygem_prms.density_water
            volchg_annual_Gt = vol_annual_Gt[:,1:] - vol_annual_Gt[:,0:-1]
            
            masschg_adjustment = masschg_annual_Gt_raw[:,0] / volchg_annual_Gt[:,0]
            
            # Correction factor to ensure propagation of mean mass balance * area doesn't cause different annual volume 
            #  change compared to the mean annual volume change
            correction_factor_annual = np.zeros(volchg_annual_Gt.shape)
            correction_factor_annual[np.nonzero(volchg_annual_Gt)] = (
                    volchg_annual_Gt[np.nonzero(volchg_annual_Gt)] / masschg_annual_Gt_raw[np.nonzero(volchg_annual_Gt)]
                    )
            correction_factor_monthly = correction_factor_annual.repeat(12,axis=1)
            
            masschg_monthly_Gt = masschg_monthly_Gt_raw * correction_factor_monthly            
            masschg_monthly_Gt_cumsum = np.cumsum(masschg_monthly_Gt, axis=1)
            mass_monthly_Gt = vol_annual_Gt[:,0][:,np.newaxis] + masschg_monthly_Gt_cumsum
            mass_monthly_Gt[mass_monthly_Gt < 0] = 0
            
            if region == regions[0]: 
                ds_multi[rcp] = mass_monthly_Gt
                df_all = df
            else:
                ds_multi[rcp] = np.concatenate((ds_multi[rcp], mass_monthly_Gt), axis=0)
                df_all = pd.concat([df_all, df], axis=0)
            
            ds.close()
            
    # Remove RGIIds from main_glac_rgi that are not in the model runs
    rgiid_df = list(df_all.RGIId.values)
    rgiid_all = list(main_glac_rgi.RGIId.values)
    rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
    main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
    main_glac_rgi.reset_index(inplace=True, drop=True)

    deg_dict = dict(zip(main_glac_rgi['deg_id'].values, main_glac_rgi['CenLon_CenLat']))
    ds_deg = {}
    for rcp in rcps:
#    for rcp in ['rcp85']:  
        ds_deg[rcp] = {}
        deg_groups_ordered = []
        mass_deg_output = pd.DataFrame(np.zeros((len(deg_dict), mass_monthly_Gt.shape[1])), 
                                                columns=time_values_monthly)
        for ngroup, group in enumerate(groups):
            deg_group_rounded = (np.round(deg_dict[group][0],1), np.round(deg_dict[group][1],1))
            deg_groups_ordered.append(deg_group_rounded)
            
            if ngroup%500 == 0: 
                print(group, deg_group_rounded)
                
                
            # Sum volume change for group
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
            vn_group = ds_multi[rcp][group_glac_indices,:].sum(axis=0)

            mass_deg_output.loc[ngroup, :] = vn_group    
            
        mass_deg_output.index = deg_groups_ordered
        
        mass_deg_output_fn = (('mass_Gt_monthly_' + rcp + '_' + str(np.round(degree_size,2)) + 'deg').replace('.','p') 
                              + '.csv')
        mass_deg_output.to_csv(pygem_prms.output_filepath + mass_deg_output_fn)
