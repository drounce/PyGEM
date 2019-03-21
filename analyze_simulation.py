""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
import glob
import os
import pickle
# External libraries
import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import pandas as pd
#import pymc
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
import pygem_input as input
import pygemfxns_gcmbiasadj as gcmbiasadj
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup
import run_calibration as calibration

# Script options
option_plot_cmip5_normalizedchange = 1
option_map_gcm_changes = 0


#%% ===== Input data =====
netcdf_fp_cmip5 = input.output_sim_fp + 'spc_subset/'
netcdf_fp_era = input.output_sim_fp + '/ERA-Interim/ERA-Interim_1980_2017_nochg'

#%%
regions = [13, 14, 15]


# GCMs and RCP scenarios
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0',  'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 
#             'MPI-ESM-LR', 'NorESM1-M']
#gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0',  'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR']
#gcm_names = ['CanESM2', 'CCSM4']
gcm_names = ['CanESM2']
#rcps = ['rcp26', 'rcp45', 'rcp85']
rcps = ['rcp26', 'rcp85']
#rcps = ['rcp85']

# Grouping
#grouping = 'all'
grouping = 'rgi_region'
#grouping = 'watershed'
#grouping = 'kaab'
#grouping = 'degree'

degree_size = 0.5

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

cal_datasets = ['shean']

# Bounds (90% bounds --> 95% above/below given threshold)
low_percentile = 5
high_percentile = 95

colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']

# Group dictionaries
watershed_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_watershed.csv'
watershed_csv = pd.read_csv(watershed_dict_fn)
watershed_dict = dict(zip(watershed_csv.RGIId, watershed_csv.watershed))
kaab_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_kaab.csv'
kaab_csv = pd.read_csv(kaab_dict_fn)
kaab_dict = dict(zip(kaab_csv.RGIId, kaab_csv.kaab_name))

# Shapefiles
rgiO1_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
rgi_glac_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA.shp'
watershed_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/HMA_basins_20181018_4plot.shp'
kaab_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/kaab2015_regions.shp'
srtm_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA.tif'
srtm_contour_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA_countours_2km_gt3000m_smooth.shp'

# GRACE mascons
mascon_fp = input.main_directory + '/../GRACE/GSFC.glb.200301_201607_v02.4/'
mascon_fn = 'mascon.txt'
mascon_cns = ['CenLat', 'CenLon', 'LatWidth', 'LonWidth', 'Area_arcdeg', 'Area_km2', 'location', 'basin', 
              'elevation_flag']
mascon_df = pd.read_csv(mascon_fp + mascon_fn, header=None, names=mascon_cns, skiprows=14, 
                        delim_whitespace=True)
mascon_df = mascon_df.sort_values(by=['CenLat', 'CenLon'])
mascon_df.reset_index(drop=True, inplace=True)

east = 104
west = 67
south = 25
north = 48


#%%
def select_groups(grouping, main_glac_rgi_all):
    """
    Select groups based on grouping
    """
    if grouping == 'rgi_region':
        groups = main_glac_rgi_all.O1Region.unique().tolist()
        group_cn = 'O1Region'
    elif grouping == 'watershed':
        groups = main_glac_rgi_all.watershed.unique().tolist()
        group_cn = 'watershed'
    elif grouping == 'kaab':
        groups = main_glac_rgi_all.kaab.unique().tolist()
        group_cn = 'kaab'
        groups = [x for x in groups if str(x) != 'nan']  
    elif grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    elif grouping == 'mascon':
        groups = main_glac_rgi_all.mascon_idx.unique().tolist()
        groups = [int(x) for x in groups]
        group_cn = 'mascon_idx'
    else:
        groups = ['all']
        group_cn = 'all_group'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = sorted(groups)
    return groups, group_cn


#%% LOAD ALL GLACIERS
def load_glacier_data(rgi_regions):
    """
    Load glacier data (main_glac_rgi, hyps, and ice thickness)
    """
    for rgi_region in rgi_regions:
        # Data on all glaciers
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(rgi_regionsO1=[rgi_region], rgi_regionsO2 = 'all', 
                                                                 rgi_glac_number='all')
         # Glacier hypsometry [km**2]
        main_glac_hyps_region = modelsetup.import_Husstable(main_glac_rgi_region, [rgi_region], input.hyps_filepath,
                                                            input.hyps_filedict, input.hyps_colsdrop)
        # Ice thickness [m], average
        main_glac_icethickness_region= modelsetup.import_Husstable(main_glac_rgi_region, [rgi_region], 
                                                                 input.thickness_filepath, input.thickness_filedict, 
                                                                 input.thickness_colsdrop)
        if rgi_region == rgi_regions[0]:
            main_glac_rgi_all = main_glac_rgi_region
            main_glac_hyps_all = main_glac_hyps_region
            main_glac_icethickness_all = main_glac_icethickness_region
        else:
            main_glac_rgi_all = pd.concat([main_glac_rgi_all, main_glac_rgi_region], sort=False)
            main_glac_hyps_all = pd.concat([main_glac_hyps_all, main_glac_hyps_region], sort=False)
            main_glac_icethickness_all = pd.concat([main_glac_icethickness_all, main_glac_icethickness_region], 
                                                   sort=False)
    main_glac_hyps_all = main_glac_hyps_all.fillna(0)
    main_glac_icethickness_all = main_glac_icethickness_all.fillna(0)
    
    # Add watersheds, regions, degrees, mascons, and all groups to main_glac_rgi_all
    # Watersheds
    main_glac_rgi_all['watershed'] = main_glac_rgi_all.RGIId.map(watershed_dict)
    # Regions
    main_glac_rgi_all['kaab'] = main_glac_rgi_all.RGIId.map(kaab_dict)
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
#    # Mascons
#    if grouping == 'mascon':
#        main_glac_rgi_all['mascon_idx'] = np.nan
#        for glac in range(main_glac_rgi_all.shape[0]):
#            latlon_dist = (((mascon_df.CenLat.values - main_glac_rgi_all.CenLat.values[glac])**2 + 
#                             (mascon_df.CenLon.values - main_glac_rgi_all.CenLon.values[glac])**2)**0.5)
#            main_glac_rgi_all.loc[glac,'mascon_idx'] = [x[0] for x in np.where(latlon_dist == latlon_dist.min())][0]
#        mascon_groups = main_glac_rgi_all.mascon_idx.unique().tolist()
#        mascon_groups = [int(x) for x in mascon_groups]
#        mascon_groups = sorted(mascon_groups)
#        mascon_latlondict = dict(zip(mascon_groups, mascon_df[['CenLat', 'CenLon']].values[mascon_groups].tolist()))
    # All
    main_glac_rgi_all['all_group'] = 'all'
    
    return main_glac_rgi_all, main_glac_hyps_all, main_glac_icethickness_all

#def load_gcm_data():
    


#%%
if option_plot_cmip5_normalizedchange == 1:
    vn = 'volume_glac_annual'
    vns_heatmap = ['massbaltotal_glac_monthly', 'temp_glac_monthly', 'prec_glac_monthly']
#    vns_heatmap = ['massbaltotal_glac_monthly', 'temp_glac_monthly']
    figure_fp = input.output_sim_fp + 'figures/'

    area_cutoffs = [20]
    nyears = 18

    multimodel_linewidth = 2
    alpha=0.2
    
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(regions)

    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=input.gcm_startyear, endyear=input.gcm_endyear, 
                                           spinupyears=input.gcm_spinupyears, option_wateryear=input.gcm_wateryear)
    
    #%%
    # Glacier and grouped annual specific mass balance and mass change
    for rcp in rcps:
        for ngcm, gcm_name in enumerate(gcm_names):
            
            #%%
            region = 15
            option_bias_adjustment=1
            # NEED TO SUBSET REGIONS AND THEN COMBINE!
            main_glac_rgi_region = main_glac_rgi.loc[main_glac_rgi['O1Region'] == region]
            main_glac_rgi_region.reset_index(drop=True, inplace=True)
            
            gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp)
        
            # Air temperature [degC], Precipitation [m], Elevation [m asl]
            gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi_region, 
                                                                         dates_table)
            # 
            gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi_region, 
                                                                         dates_table)
            # Elevation [m asl]
            gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi_region)          

#            # Adjust reference dates in event that reference is longer than GCM data
#            if input.startyear >= input.gcm_startyear:
#                ref_startyear = input.startyear
#            else:
#                ref_startyear = input.gcm_startyear
#            if input.endyear <= input.gcm_endyear:
#                ref_endyear = input.endyear
#            else:
#                ref_endyear = input.gcm_endyear
#            dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear, 
#                                                       spinupyears=input.spinupyears, 
#                                                       option_wateryear=input.option_wateryear)
#            # Monthly average from reference climate data
#            ref_gcm = class_climate.GCM(name=input.ref_gcm_name)
#               
#            # ===== BIAS CORRECTIONS =====
#            # No adjustments
#            if option_bias_adjustment == 0:
#                gcm_temp_adj = gcm_temp
#                gcm_prec_adj = gcm_prec
#                gcm_elev_adj = gcm_elev
#            # Bias correct based on reference climate data
#            else:
#                # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
#                ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn, 
#                                                                                 main_glac_rgi_region, dates_table_ref)
#                ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn, 
#                                                                                 main_glac_rgi_region, dates_table_ref)
#                ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, 
#                                                                     main_glac_rgi_region)
#                
#                # OPTION 1: Adjust temp using Huss and Hock (2015), prec similar but addresses for variance and outliers
#                if input.option_bias_adjustment == 1:
#                    # Temperature bias correction
#                    gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, 
#                                                                                dates_table_ref, dates_table)
#                    # Precipitation bias correction
#                    gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec, 
#                                                                              dates_table_ref, dates_table)
#                
#                # OPTION 2: Adjust temp and prec using Huss and Hock (2015)
#                elif input.option_bias_adjustment == 2:
#                    # Temperature bias correction
#                    gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, 
#                                                                                dates_table_ref, dates_table)
#                    # Precipitation bias correction
#                    gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec, 
#                                                                                dates_table_ref, dates_table)
            #%%
            
            # Extract data from netcdf
            group_glacidx = {}
            vol_reg_dict = {}
            temp_reg_dict = {}
            prec_reg_dict = {}
            acc_reg_dict = {}
            prectotal_reg_dict = {}
            
            for region in regions:
                       
                # Load datasets
                ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba' + str(input.option_bias_adjustment) +
                         '_100sets_2000_2100--subset.nc')
                ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                # Extract time variable
                time_values_annual = ds.coords['year_plus1'].values
                time_values_monthly = ds.coords['time'].values
                
                # Merge datasets
                if region == regions[0]:
                    # Volume
                    vol_glac_all = ds['volume_glac_annual'].values[:,:,0]
                    vol_glac_std_all = ds['volume_glac_annual'].values[:,:,1]
                    # Area
                    area_glac_all = ds['area_glac_annual'].values[:,:,0]
                    area_glac_std_all = ds['area_glac_annual'].values[:,:,1]
                    # Mass balance
                    mb_glac_all = ds['massbaltotal_glac_monthly'].values[:,:,0]
                    mb_glac_std_all = ds['massbaltotal_glac_monthly'].values[:,:,1]
                    # Temperature
                    temp_glac_all = ds['temp_glac_monthly'].values[:,:,0]
                    temp_glac_std_all = ds['temp_glac_monthly'].values[:,:,1]
                    # Precipitation
                    prec_glac_all = ds['prec_glac_monthly'].values[:,:,0]
                    prec_glac_std_all = ds['prec_glac_monthly'].values[:,:,1]
                    # Accumulation
                    acc_glac_all = ds['acc_glac_monthly'].values[:,:,0]
                    acc_glac_std_all = ds['acc_glac_monthly'].values[:,:,1]
                else:
                    # Volume
                    vol_glac_all = np.concatenate((vol_glac_all, ds['volume_glac_annual'].values[:,:,0]), axis=0)
                    vol_glac_std_all = np.concatenate((vol_glac_std_all, ds['volume_glac_annual'].values[:,:,1]),axis=0)
                    # Area
                    area_glac_all = np.concatenate((area_glac_all, ds['area_glac_annual'].values[:,:,0]), axis=0)
                    area_glac_std_all = np.concatenate((area_glac_std_all, ds['area_glac_annual'].values[:,:,1]),axis=0)
                    # Mass balance
                    mb_glac_all = np.concatenate((mb_glac_all, ds['massbaltotal_glac_monthly'].values[:,:,0]), axis=0)
                    mb_glac_std_all = np.concatenate((mb_glac_std_all, ds['massbaltotal_glac_monthly'].values[:,:,1]),axis=0)
                    # Temperature
                    temp_glac_all = np.concatenate((temp_glac_all, ds['temp_glac_monthly'].values[:,:,0]), axis=0)
                    temp_glac_std_all = np.concatenate((temp_glac_std_all, ds['temp_glac_monthly'].values[:,:,1]),axis=0)
                    # Precipitation
                    prec_glac_all = np.concatenate((prec_glac_all, ds['prec_glac_monthly'].values[:,:,0]), axis=0)
                    prec_glac_std_all = np.concatenate((prec_glac_std_all, ds['prec_glac_monthly'].values[:,:,1]),axis=0)
                    # Accumulation
                    acc_glac_all = np.concatenate((acc_glac_all, ds['acc_glac_monthly'].values[:,:,0]), axis=0)
                    acc_glac_std_all = np.concatenate((acc_glac_std_all, ds['acc_glac_monthly'].values[:,:,1]),axis=0)
    
                ds.close()
                
            # Cycle through groups
            for ngroup, group in enumerate(groups):
                # Select subset of data
                group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                group_glacidx[group] = group_glac_indices
                
                # Regional Volume, Temperature, Precipitation, and Accumulation
                vol_reg_dict[group] = vol_glac_all[group_glac_indices,:].sum(axis=0)
                temp_reg_dict[group] = (gcmbiasadj.annual_avg_2darray(temp_glac_all[group_glac_indices,:])).mean(axis=0)
                prec_reg_dict[group] = (gcmbiasadj.annual_sum_2darray(prec_glac_all[group_glac_indices,:])).mean(axis=0)
                acc_reg_dict[group] = (gcmbiasadj.annual_sum_2darray(acc_glac_all[group_glac_indices,:])).mean(axis=0)
                prectotal_reg_dict[group] = prec_reg_dict[group] + acc_reg_dict[group]
                
            # Annual Mass Balance
            mb_glac_all_annual = gcmbiasadj.annual_sum_2darray(mb_glac_all)
            # mask values where volume is zero
            mb_glac_all_annual[vol_glac_all[:,:-1] == 0] = np.nan
            
            # Annual Temperature, Precipitation, and Accumulation
            temp_glac_all_annual = gcmbiasadj.annual_avg_2darray(temp_glac_all)
            prec_glac_all_annual = gcmbiasadj.annual_sum_2darray(prec_glac_all)
            acc_glac_all_annual = gcmbiasadj.annual_sum_2darray(acc_glac_all)
            prectotal_glac_all_annual = prec_glac_all_annual + acc_glac_all_annual
            
            #%%
            
            cmap_dict = {'massbaltotal_glac_monthly':'RdYlBu',
                         'temp_glac_monthly':'RdYlBu_r',
                         'prec_glac_monthly':'RdYlBu'}
            norm_dict = {'massbaltotal_glac_monthly':plt.Normalize(-2,2),
                         'temp_glac_monthly':plt.Normalize(-2,2),
                         'prec_glac_monthly':plt.Normalize(0.5,1.5)}
            ylabel_dict = {'massbaltotal_glac_monthly':'Normalized Volume\n[-]',
                           'temp_glac_monthly':'Temperature Change\n[$^\circ$C]',
                           'prec_glac_monthly':'Precipitation Change\n[%]'}
            
            for area_cutoff in area_cutoffs:
                
                # Plot the normalized volume change for each region, along with the mass balances
                fig, ax = plt.subplots(len(groups), len(vns_heatmap), squeeze=False, sharex=True, sharey=False, 
                                       gridspec_kw = {'wspace':0.6, 'hspace':0.2})
                fig.subplots_adjust(top=0.9)
            
                for nregion, region in enumerate(regions):
                    
                    for nvar, vn_heatmap in enumerate(vns_heatmap):
                        
                        print(region, vn_heatmap)
                        
                        if vn_heatmap == 'massbaltotal_glac_monthly':
                            var_line = vol_reg_dict[region][:-1] / vol_reg_dict[region][0]
                            zmesh = mb_glac_all_annual
                        elif vn_heatmap == 'temp_glac_monthly':
                            var_line = temp_reg_dict[region] - temp_reg_dict[region][0:nyears].mean()
                            zmesh = temp_glac_all_annual - temp_glac_all_annual[:,0:nyears].mean(axis=1)[:,np.newaxis]
                        elif vn_heatmap == 'prec_glac_monthly':
                            var_line = prectotal_reg_dict[region] / prectotal_reg_dict[region][0:nyears].mean()
                            zmesh = (prectotal_glac_all_annual /
                                     prectotal_glac_all_annual[:,0:nyears].mean(axis=1)[:,np.newaxis])
                            
                        cmap = cmap_dict[vn_heatmap]
                        norm = norm_dict[vn_heatmap]
                                                 
                        ax[nregion,nvar].plot(time_values_annual[:-1], var_line, color='k', linewidth=2, 
                                              label=str(region))
                        ax[nregion,nvar].set_ylabel(ylabel_dict[vn_heatmap], size=10)
                    
                        # Plot mesh of mass balance values
                        # Only for glaciers greater than 1 km2
                        glac_idx4mesh = np.where(main_glac_rgi.loc[group_glacidx[region],'Area'] > area_cutoff)[0]
                        z = zmesh[glac_idx4mesh,:]
                        x = time_values_annual[:-1]
                        y = np.array(range(len(glac_idx4mesh)))
                        ax2 = ax[nregion,nvar].twinx()
                        ax[nregion,nvar].set_zorder(2)
                        ax[nregion,nvar].patch.set_visible(False)
                        ax2.pcolormesh(x, y, z, cmap=cmap, norm=norm)
#                        if nvar < len(vns_heatmap) - 1:
#                        ax2.yaxis.set_ticklabels([])
                        ax2.yaxis.set_visible(False)
                        
                
                # Mass balance colorbar
                fig.text(0.22, 1.02, 'Mass balance\n[mwea]', ha='center', va='center', size=12)
                cmap=cmap_dict['massbaltotal_glac_monthly']
                norm=norm_dict['massbaltotal_glac_monthly']
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax = plt.axes([0.13, 0.92, 0.18, 0.02])
                plt.colorbar(sm, cax=cax, orientation='horizontal')
                cax.xaxis.set_ticks_position('top')
                for n, label in enumerate(cax.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
                
                # Temperature colorbar
                fig.text(0.52, 1.02, 'Temperature Change\n[$^\circ$C]', ha='center', va='center', size=12)
                cmap=cmap_dict['temp_glac_monthly']
                norm=norm_dict['temp_glac_monthly']
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax = plt.axes([0.42, 0.92, 0.18, 0.02])
                plt.colorbar(sm, cax=cax, orientation='horizontal')
                cax.xaxis.set_ticks_position('top')
                for n, label in enumerate(cax.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
                
                # Precipitation colorbar
                fig.text(0.81, 1.02, 'Precipitation Change\n[%]', ha='center', va='center', size=12)
                cmap=cmap_dict['prec_glac_monthly']
                norm=norm_dict['prec_glac_monthly']
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax = plt.axes([0.71, 0.92, 0.18, 0.02])
                plt.colorbar(sm, cax=cax, orientation='horizontal')
                cax.xaxis.set_ticks_position('top')
                for n, label in enumerate(cax.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
                        
                    
                # Label y-axis
                fig.text(0, 0.80, 'Region 13', va='center', rotation='vertical', size=12) 
                fig.text(0, 0.52, 'Region 14', va='center', rotation='vertical', size=12) 
                fig.text(0, 0.25, 'Region 15', va='center', rotation='vertical', size=12)  
                
                print('NEED TO ADD SOMETHING TO STATE WHAT LINES ARE - REGIONAL AVERAGES, ALTHOUGH VOLUME CHANGE FOR FIRST')
                    
                # Save figure
                fig.set_size_inches(8,6)
                figure_fn = gcm_name + '_' + rcp + '_areagt' + str(area_cutoff) + 'km2.png'
                fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
                plt.show()
                plt.close()
        
#%%
if option_map_gcm_changes == 1:
    figure_fp = input.output_sim_fp + 'figures/gcm_changes/'
    if os.path.exists(figure_fp) == False:
            os.makedirs(figure_fp)
            
    nyears = 30
    
    temp_vn = 'tas'
    prec_vn = 'pr'
    elev_vn = 'orog'
    lat_vn = 'lat'
    lon_vn = 'lon'
    time_vn = 'time'
    
    xtick = 5
    ytick = 5
    xlabel = 'Longitude [$^\circ$]'
    ylabel = 'Latitude [$^\circ$]'
    labelsize = 12
    
#    # Extra information
#    self.timestep = input.timestep
#    self.rgi_lat_colname=input.rgi_lat_colname
#    self.rgi_lon_colname=input.rgi_lon_colname
#    self.rcp_scenario = rcp_scenario
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=1, option_wateryear=1)
    
    for rcp in rcps:
        for ngcm, gcm_name in enumerate(gcm_names):
            
            # Variable filepaths
            var_fp = input.cmip5_fp_var_prefix + rcp + input.cmip5_fp_var_ending
            fx_fp = input.cmip5_fp_fx_prefix + rcp + input.cmip5_fp_fx_ending
            # Variable filenames
            temp_fn = temp_vn + '_mon_' + gcm_name + '_' + rcp + '_r1i1p1_native.nc'
            prec_fn = prec_vn + '_mon_' + gcm_name + '_' + rcp + '_r1i1p1_native.nc'
            elev_fn = elev_vn + '_fx_' + gcm_name + '_' + rcp+ '_r0i0p0.nc'
            
            # Import netcdf file
            ds_temp = xr.open_dataset(var_fp + temp_fn)
            ds_prec = xr.open_dataset(var_fp + prec_fn)
            
            # Time, Latitude, Longitude
            start_idx = (np.where(pd.Series(ds_temp[time_vn])
                                  .apply(lambda x: x.strftime('%Y-%m')) == dates_table['date']
                                  .apply(lambda x: x.strftime('%Y-%m'))[0]))[0][0]
            end_idx = (np.where(pd.Series(ds_temp[time_vn])
                                .apply(lambda x: x.strftime('%Y-%m')) == dates_table['date']
                                .apply(lambda x: x.strftime('%Y-%m'))[dates_table.shape[0] - 1]))[0][0]
            north_idx = np.abs(north - ds_temp[lat_vn][:].values).argmin()
            south_idx = np.abs(south - ds_temp[lat_vn][:].values).argmin()
            west_idx = np.abs(west - ds_temp[lon_vn][:].values).argmin()
            east_idx = np.abs(east - ds_temp[lon_vn][:].values).argmin()
            
            lats = ds_temp[lat_vn][south_idx:north_idx+1].values
            lons = ds_temp[lon_vn][west_idx:east_idx+1].values
            temp = ds_temp[temp_vn][start_idx:end_idx+1, south_idx:north_idx+1, west_idx:east_idx+1].values - 273.15
            prec = ds_prec[prec_vn][start_idx:end_idx+1, south_idx:north_idx+1, west_idx:east_idx+1].values
            # Convert from kg m-2 s-1 to m day-1
            prec = prec/1000*3600*24
            
            temp_end = temp[-nyears*12:,:,:].mean(axis=0)
            temp_start = temp[:nyears*12,:,:].mean(axis=0)
            temp_change = temp_end - temp_start
            
            prec_end = prec[-nyears*12:,:,:].mean(axis=0)*365
            prec_start = prec[:nyears*12:,:,:].mean(axis=0)*365
            prec_change = prec_end / prec_start
            
            for vn in ['temp', 'prec']:
#            for vn in ['prec']:
                if vn == 'temp':
                    var_change = temp_change
                    cmap = 'RdYlBu_r'
                    norm = plt.Normalize(int(temp_change.min()), np.ceil(temp_change.max()))
                    var_label = 'Temperature Change [$^\circ$C]'
                elif vn == 'prec':
                    var_change = prec_change
                    cmap = 'RdYlBu'
                    norm = plt.Normalize(0.5, 1.5)
                    var_label = 'Precipitation Change [-]'
                    
                # Create the projection
                fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
                # Add country borders for reference
                ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, zorder=10)
                ax.add_feature(cartopy.feature.COASTLINE)
    
                # Set the extent
                ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
                # Label title, x, and y axes
                ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
                ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
                ax.set_xlabel(xlabel, size=labelsize)
                ax.set_ylabel(ylabel, size=labelsize)
                
                # Add contour lines
                srtm_contour_shp = cartopy.io.shapereader.Reader(srtm_contour_fn)
                srtm_contour_feature = cartopy.feature.ShapelyFeature(srtm_contour_shp.geometries(), cartopy.crs.PlateCarree(),
                                                                      edgecolor='black', facecolor='none', linewidth=0.15)
                ax.add_feature(srtm_contour_feature, zorder=9)      
                
                # Add regions
                group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
                group_feature = cartopy.feature.ShapelyFeature(group_shp.geometries(), cartopy.crs.PlateCarree(),
                                                               edgecolor='black', facecolor='none', linewidth=2)
                ax.add_feature(group_feature,zorder=10)
                
    #            # Add glaciers
    #            rgi_glac_shp = cartopy.io.shapereader.Reader(rgi_glac_shp_fn)
    #            rgi_glac_feature = cartopy.feature.ShapelyFeature(rgi_glac_shp.geometries(), cartopy.crs.PlateCarree(),
    #                                                              edgecolor='black', facecolor='none', linewidth=0.15)
    #            ax.add_feature(rgi_glac_feature, zorder=9)
    #            
                                                    
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
                fig.text(1, 0.5, var_label, va='center', ha='center', rotation='vertical', size=labelsize)
        
                ax.pcolormesh(lons, lats, var_change, cmap=cmap, norm=norm, zorder=2, alpha=0.8)            
                
                # Title
                ax.set_title(gcm_name + ' ' + rcp + ' (2070-2100 vs 2000-2030)')
                
                # Save figure
                fig.set_size_inches(6,4)
                fig_fn = gcm_name + '_' + rcp + '_' + vn + '.png'
                fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)
                plt.close()
            
            
            
            
    