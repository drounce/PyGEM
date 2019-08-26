""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
import collections
import decimal
import glob
import os
import pickle
# External libraries
import cartopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import pymc
from scipy import stats
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import uniform
#from scipy.stats import linregress
from scipy.stats import lognorm
from scipy.optimize import minimize
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem_input as input
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup
import pygemfxns_gcmbiasadj as gcmbiasadj
import run_calibration as calibration

#%%
option_observation_vs_calibration = 0
option_GRACE_2deg = 0
option_trishuli = 1


variables = ['massbal', 'precfactor', 'tempchange', 'ddfsnow']  
vn_title_dict = {'massbal':'Mass\nBalance',                                                                      
                 'precfactor':'Precipitation\nFactor',                                                              
                 'tempchange':'Temperature\nBias',                                                               
                 'ddfsnow':'Degree-Day \nFactor of Snow'}
vn_label_dict = {'massbal':'Mass Balance\n[mwea]',                                                                      
                 'precfactor':'Precipitation Factor\n[-]',                                                              
                 'tempchange':'Temperature Bias\n[$^\circ$C]',                                                               
                 'ddfsnow':'Degree Day Factor of Snow\n[mwe d$^{-1}$ $^\circ$C$^{-1}$]'}
vn_label_units_dict = {'massbal':'[mwea]',                                                                      
                       'precfactor':'[-]',                                                              
                       'tempchange':'[$^\circ$C]',                                                               
                       'ddfsnow':'[mwe d$^{-1}$ $^\circ$C$^{-1}$]'}

# Export option
sim_netcdf_fp = input.output_filepath + 'simulations/ERA-Interim/ERA-Interim_2000_2018_nochg/'
#sim_netcdf_fp = input.output_filepath + 'simulations/ERA-Interim/ERA-Interim_1980_2017_nochg/'
#sim_netcdf_fp = input.output_filepath + 'simulations/ERA-Interim_2000_2017wy_nobiasadj/'

figure_fp = sim_netcdf_fp + 'figures/'

regions = [13, 14, 15]
degree_size = 0.1

cal_datasets = ['shean']

burn=0

colors = ['#387ea0', '#fcb200', '#d20048']
linestyles = ['-', '--', ':']

east = 60
west = 110
south = 15
north = 50
xtick = 5
ytick = 5
xlabel = 'Longitude [$^\circ$]'
ylabel = 'Latitude [$^\circ$]'


#%%
# ===== FUNCTIONS ====
def plot_hist(df, cn, bins, xlabel=None, ylabel=None, fig_fn='hist.png', fig_fp=figure_fp):
    """
    Plot histogram for any bin size
    """           
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    
    data = df[cn].values
    hist, bin_edges = np.histogram(data,bins) # make the histogram
    fig,ax = plt.subplots()    
    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)),hist,width=1, edgecolor='k') 
    # Set the ticks to the middle of the bars
    ax.set_xticks([0.5+i for i,j in enumerate(hist)])
    # Set the xticklabels to a string that tells us what the bin edges were
    ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)], rotation=45, ha='right')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    # Save figure
    fig.set_size_inches(6,4)
    fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
    

def select_groups(grouping, main_glac_rgi_all):
    """
    Select groups based on grouping
    """
    if grouping == 'rgi_region':
        groups = regions
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


def load_masschange_monthly(regions, ds_ending, netcdf_fp=sim_netcdf_fp, option_add_caldata=0):
    """ Load monthly mass change data """
    count = 0
    for region in regions:
        count += 1
        
        # Load datasets
        ds_fn = 'R' + str(region) + ds_ending
        ds = xr.open_dataset(netcdf_fp + ds_fn)

        main_glac_rgi_region_ds = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
        glac_wide_massbaltotal_region = ds.massbaltotal_glac_monthly.values[:,:,0]
        glac_wide_area_annual_region = ds.area_glac_annual.values[:,:,0]
        time_values = pd.Series(ds.massbaltotal_glac_monthly.coords['time'].values)
        
        # ===== GLACIER DATA =====
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=[region], rgi_regionsO2 = 'all', rgi_glac_number='all')
        if (main_glac_rgi_region['glacno'] - main_glac_rgi_region_ds['glacno']).sum() == 0:
            print('Region', str(region),': number of glaciers match')
        # Glacier hypsometry
        main_glac_hyps_region = modelsetup.import_Husstable(
                main_glac_rgi_region, input.hyps_filepath,input.hyps_filedict, input.hyps_colsdrop)     
        # Ice thickness [m], average
        main_glac_icethickness_region = modelsetup.import_Husstable(
                main_glac_rgi_region, input.thickness_filepath, input.thickness_filedict, input.thickness_colsdrop)
        main_glac_hyps_region[main_glac_icethickness_region == 0] = 0
        # ===== CALIBRATION DATA =====
        if option_add_caldata == 1:
            dates_table_nospinup = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                                            spinupyears=0)
            cal_data_region = pd.DataFrame()
            for dataset in cal_datasets:
                cal_subset = class_mbdata.MBData(name=dataset)
                cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_region, main_glac_hyps_region, dates_table_nospinup)
                cal_data_region = cal_data_region.append(cal_subset_data, ignore_index=True)
            cal_data_region = cal_data_region.sort_values(['glacno', 't1_idx'])
            cal_data_region.reset_index(drop=True, inplace=True)
        
        # ===== APPEND DATASETS =====
        if count == 1:
            main_glac_rgi = main_glac_rgi_region
            main_glac_hyps = main_glac_hyps_region
            main_glac_icethickness = main_glac_icethickness_region
            glac_wide_massbaltotal = glac_wide_massbaltotal_region
            glac_wide_area_annual = glac_wide_area_annual_region
            
            if option_add_caldata == 1:
                cal_data = cal_data_region
            
        else:
            main_glac_rgi = main_glac_rgi.append(main_glac_rgi_region)
            
            glac_wide_massbaltotal = np.concatenate([glac_wide_massbaltotal, glac_wide_massbaltotal_region])
            glac_wide_area_annual = np.concatenate([glac_wide_area_annual, glac_wide_area_annual_region])
            
            if option_add_caldata == 1:
                cal_data = cal_data.append(cal_data_region)
            
            # If more columns in region, then need to expand existing dataset
            if main_glac_hyps_region.shape[1] > main_glac_hyps.shape[1]:
                all_col = list(main_glac_hyps.columns.values)
                reg_col = list(main_glac_hyps_region.columns.values)
                new_cols = [item for item in reg_col if item not in all_col]
                for new_col in new_cols:
                    main_glac_hyps[new_col] = 0
                    main_glac_icethickness[new_col] = 0
            elif main_glac_hyps_region.shape[1] < main_glac_hyps.shape[1]:
                all_col = list(main_glac_hyps.columns.values)
                reg_col = list(main_glac_hyps_region.columns.values)
                new_cols = [item for item in all_col if item not in reg_col]
                for new_col in new_cols:
                    main_glac_hyps_region[new_col] = 0
                    main_glac_icethickness_region[new_col] = 0
            main_glac_hyps = main_glac_hyps.append(main_glac_hyps_region)
            main_glac_icethickness = main_glac_icethickness.append(main_glac_icethickness_region)
        
    # reset index
    main_glac_rgi.reset_index(inplace=True, drop=True)
    main_glac_hyps.reset_index(inplace=True, drop=True)
    main_glac_icethickness.reset_index(inplace=True, drop=True)
    if option_add_caldata == 1:
        cal_data.reset_index(inplace=True, drop=True)
        
    # Volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
    

    # ===== MASS CHANGE CALCULATIONS =====
    # Compute glacier volume change for every time step and use this to compute mass balance
    glac_wide_area = np.repeat(glac_wide_area_annual[:,:-1], 12, axis=1)
    
    # Mass change [km3 mwe]
    #  mb [mwea] * (1 km / 1000 m) * area [km2]
    glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
    
    if option_add_caldata == 1:
        return main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values, cal_data
    else:
        return main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values
    

# ===== PLOT OPTIONS ==================================================================================================    
#%%        
def observation_vs_calibration(regions, netcdf_fp):
    """
    Compare mass balance observations with model calibration
    
    Parameters
    ----------
    regions : list of strings
        list of regions
    Returns
    -------
    .png files
        saves histogram of differences between observations and calibration
    .csv file
        saves .csv file of comparison
    """
#%% 
if option_observation_vs_calibration == 1:
#    observation_vs_calibration(regions, sim_netcdf_fp)
    
    t1_idx = 0
    t2_idx = 216
#    t1_idx = 240
#    t2_idx = 455
    t1 = 2000
    t2 = 2018
    
#    main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values = (
#            load_masschange_monthly(regions, ds_ending='_ERA-Interim_c2_ba1_100sets_1980_2017.nc'))
    main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values = (
            load_masschange_monthly(regions, ds_ending='_ERA-Interim_c2_ba1_100sets_2000_2018.nc'))
    
    # Mean annual mass balance [mwea]
    glac_wide_mb_mwea = glac_wide_masschange[:,t1_idx:t2_idx+1].sum(axis=1) / glac_wide_area[:,0] * 1000 / (t2 - t1)
    
    
    #%%
    # Total mass change [Gt/yr]
    total_masschange = glac_wide_masschange[:,t1_idx:t2_idx+1].sum(axis=1).sum() / (t2 - t1)
#    total_masschange_obs = (cal_data.mb_mwe.values / 1000 * glac_wide_area[:,0]).sum() / (t2 - t1)
    
    #%%
#    main_glac_rgi['mb_mwea_era_mean'] = glac_wide_mb_mwea
#    main_glac_rgi['mb_mwea_cal_mean'] = cal_data.mb_mwe.values / (t2 - t1)
#    main_glac_rgi['mb_mwea_cal_std'] = cal_data.mb_mwe_err.values / (t2 - t1)
    
#    #%% Total mass change accounting for truncation
#    # Maximum loss if entire glacier melted between 2000 and 2018
#    mb_max_loss = (-1 * (main_glac_hyps * main_glac_icethickness * input.density_ice / 
#                         input.density_water).sum(axis=1).values / main_glac_hyps.sum(axis=1).values / (t2 - t1))
#    main_glac_rgi['mb_max_loss'] = mb_max_loss
#    
#    # Truncated normal - updated means to remove portions of observations that are below max mass loss!
#    main_glac_rgi['mb_cal_dist_mean'] = np.nan
#    main_glac_rgi['mb_cal_dist_med'] = np.nan
#    main_glac_rgi['mb_cal_dist_std'] = np.nan
#    main_glac_rgi['mb_cal_dist_95low'] = np.nan
#    main_glac_rgi['mb_cal_dist_95high']  = np.nan
#    for glac in range(main_glac_rgi.shape[0]):
#        if glac%500 == 0:
#            print(glac)
#        cal_mu = main_glac_rgi.loc[glac, 'mb_mwea_cal_mean']
#        cal_sigma = main_glac_rgi.loc[glac, 'mb_mwea_cal_std']
#        cal_lowbnd = main_glac_rgi.loc[glac, 'mb_max_loss']
#        cal_rvs = stats.truncnorm.rvs((cal_lowbnd - cal_mu) / cal_sigma, np.inf, loc=cal_mu, scale=cal_sigma, 
#                                      size=int(1e5))
#        main_glac_rgi.loc[glac,'mb_cal_dist_mean'] = np.mean(cal_rvs)
#        main_glac_rgi.loc[glac,'mb_cal_dist_med'] = np.median(cal_rvs)
#        main_glac_rgi.loc[glac,'mb_cal_dist_std'] = np.std(cal_rvs)
#        main_glac_rgi.loc[glac,'mb_cal_dist_95low'] = np.percentile(cal_rvs, 2.5)
#        main_glac_rgi.loc[glac,'mb_cal_dist_95high']  = np.percentile(cal_rvs, 97.5) 
#        
#    total_masschange_obs_adjdist = np.nansum((main_glac_rgi.mb_cal_dist_mean.values / 1000 * glac_wide_area[:,0]))
#
#        
#        #%%
#
#    main_glac_rgi['dif_cal_era_mean'] = main_glac_rgi['mb_mwea_cal_mean'] - main_glac_rgi['mb_mwea_era_mean']
#
#    # remove nan values
#    main_glac_rgi_dropnan = (
#            main_glac_rgi.drop(np.where(np.isnan(main_glac_rgi['mb_mwea_era_mean'].values) == True)[0].tolist(), 
#                               axis=0))
#    main_glac_rgi_dropnan.reset_index(drop=True, inplace=True)
#    
#    # Degrees
#    main_glac_rgi_dropnan['CenLon_round'] = np.floor(main_glac_rgi_dropnan.CenLon.values/degree_size) * degree_size
#    main_glac_rgi_dropnan['CenLat_round'] = np.floor(main_glac_rgi_dropnan.CenLat.values/degree_size) * degree_size
#    deg_groups = main_glac_rgi_dropnan.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
#    deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
#    main_glac_rgi_dropnan.reset_index(drop=True, inplace=True)
#    cenlon_cenlat = [(main_glac_rgi_dropnan.loc[x,'CenLon_round'], main_glac_rgi_dropnan.loc[x,'CenLat_round']) 
#                     for x in range(len(main_glac_rgi_dropnan))]
#    main_glac_rgi_dropnan['CenLon_CenLat'] = cenlon_cenlat
#    main_glac_rgi_dropnan['deg_id'] = main_glac_rgi_dropnan.CenLon_CenLat.map(deg_dict)
#
##%%
#    # Histogram: Mass balance [mwea], Observation - ERA
#    hist_cn = 'dif_cal_era_mean'
#    low_bin = np.floor(main_glac_rgi_dropnan[hist_cn].min())
#    high_bin = np.ceil(main_glac_rgi_dropnan[hist_cn].max())
#    bins = [low_bin, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, high_bin]
#    plot_hist(main_glac_rgi_dropnan, hist_cn, bins, xlabel='Mass balance [mwea]\n(Calibration - MCMC_mean)', 
#              ylabel='# Glaciers', fig_fn='MB_cal_vs_mcmc_hist.png', fig_fp=figure_fp)
#    
#    #%%
#    
#    # Map: Mass change, difference between calibration data and median data
#    #  Area [km2] * mb [mwe] * (1 km / 1000 m) * density_water [kg/m3] * (1 Gt/km3  /  1000 kg/m3)
#    main_glac_rgi_dropnan['mb_cal_Gta'] = main_glac_rgi_dropnan['mb_mwea_cal_mean'] * main_glac_rgi_dropnan['Area'] / 1000
#    main_glac_rgi_dropnan['mb_cal_Gta_var'] = (main_glac_rgi_dropnan['mb_mwea_cal_std'] * main_glac_rgi_dropnan['Area'] / 1000)**2
#    main_glac_rgi_dropnan['mb_era_Gta'] = main_glac_rgi_dropnan['mb_mwea_era_mean'] * main_glac_rgi_dropnan['Area'] / 1000
##    main_glac_rgi_dropnan['mb_era_Gta_var'] = (main_glac_rgi_dropnan['mb_era_std'] * main_glac_rgi_dropnan['Area'] / 1000)**2
##    main_glac_rgi_dropnan['mb_era_Gta_med'] = main_glac_rgi_dropnan['mb_era_med'] * main_glac_rgi_dropnan['Area'] / 1000
##    print('All MB cal (mean +/- 1 std) [gt/yr]:', np.round(main_glac_rgi_dropnan['mb_cal_Gta'].sum(),3), 
##          '+/-', np.round(main_glac_rgi_dropnan['mb_cal_Gta_var'].sum()**0.5,3),
##          '\nAll MB ERA (mean +/- 1 std) [gt/yr]:', np.round(main_glac_rgi_dropnan['mb_era_Gta'].sum(),3), 
##          '+/-', np.round(main_glac_rgi_dropnan['mb_era_Gta_var'].sum()**0.5,3),
##          '\nAll MB ERA (med) [gt/yr]:', np.round(main_glac_rgi_dropnan['mb_era_Gta_med'].sum(),3))
#    
#    print('All MB cal (mean +/- 1 std) [gt/yr]:', np.round(main_glac_rgi_dropnan['mb_cal_Gta'].sum(),3),
#          '+/-', np.round(main_glac_rgi_dropnan['mb_cal_Gta_var'].sum()**0.5,3),
#          '\nAll MB ERA (mean) [gt/yr]:', np.round(main_glac_rgi_dropnan['mb_era_Gta'].sum(),3), 
#          )
#    
#    #%%
#
#    def partition_sum_groups(grouping, vn, main_glac_rgi_dropnan):
#        """Partition model parameters by each group
#        
#        Parameters
#        ----------
#        grouping : str
#            name of grouping to use
#        vn : str
#            variable name
#        main_glac_rgi_dropnan : pd.DataFrame
#            glacier table
#            
#        Output
#        ------
#        groups : list
#            list of group names
#        ds_group : list of lists
#            dataset containing the multimodel data for a given variable for all the GCMs
#        """
#        # Groups
#        groups, group_cn = select_groups(grouping, main_glac_rgi_dropnan)
#        
#        ds_group = [[] for group in groups]
#        
#        # Cycle through groups
#        for ngroup, group in enumerate(groups):
#            # Select subset of data
#            main_glac_rgi = main_glac_rgi_dropnan.loc[main_glac_rgi_dropnan[group_cn] == group]                        
#            vn_glac = main_glac_rgi_dropnan[vn].values[main_glac_rgi.index.values.tolist()]             
#            # Regional sum
#            vn_reg = vn_glac.sum(axis=0)                
#            
#            # Record data for each group
#            ds_group[ngroup] = [group, vn_reg]
#                
#        return groups, ds_group
#
#    grouping='degree'
#    
#    groups, ds_group_cal = partition_sum_groups(grouping, 'mb_cal_Gta', main_glac_rgi_dropnan)
#    groups, ds_group_era = partition_sum_groups(grouping, 'mb_era_Gta', main_glac_rgi_dropnan)
#    groups, ds_group_area = partition_sum_groups(grouping, 'Area', main_glac_rgi_dropnan)
#    
##    ds_group_dif = [[] for x in ds_group_cal ]
#
#    # Group difference [Gt/yr]
#    dif_cal_era_Gta = (np.array([x[1] for x in ds_group_cal]) - np.array([x[1] for x in ds_group_era])).tolist()
#    ds_group_dif_cal_era_Gta = [[x[0],dif_cal_era_Gta[n]] for n, x in enumerate(ds_group_cal)]
#    # Group difference [mwea]
#    area = [x[1] for x in ds_group_area]
#    ds_group_dif_cal_era_mwea = [[x[0], dif_cal_era_Gta[n] / area[n] * 1000] for n, x in enumerate(ds_group_cal)]
#    
#    east = 104
#    west = 67
#    south = 25
#    north = 48
#    
#    labelsize = 13
#    
#    norm = plt.Normalize(-0.1, 0.1)
#    
#    # Create the projection
#    fig, ax = plt.subplots(1, 1, figsize=(10,5), subplot_kw={'projection':cartopy.crs.PlateCarree()})
#    # Add country borders for reference
#    ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, zorder=10)
#    ax.add_feature(cartopy.feature.COASTLINE)
#    # Set the extent
#    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
#    # Label title, x, and y axes
#    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
#    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
#    ax.set_xlabel(xlabel, size=labelsize)
#    ax.set_ylabel(ylabel, size=labelsize)            
#    
#    cmap = 'RdYlBu_r'
#
#    # Add colorbar
##    sm = plt.cm.ScalarMappable(cmap=cmap)
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm._A = []
#    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
#    fig.text(1, 0.5, 'Mass balance [mwea]\n(Observation - MCMC)', va='center', rotation='vertical', size=labelsize)    
#
#    # Group by degree  
#    groups_deg = groups
#    ds_vn_deg = ds_group_dif_cal_era_mwea
#
#    z = [ds_vn_deg[ds_idx][1] for ds_idx in range(len(ds_vn_deg))]
#    x = np.array([x[0] for x in deg_groups]) 
#    y = np.array([x[1] for x in deg_groups])
#    lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
#    lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
#    x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
#    y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
#    z_array = np.zeros((len(y_adj), len(x_adj)))
#    z_array[z_array==0] = np.nan
#    for i in range(len(z)):
#        row_idx = int((y[i] - y.min()) / degree_size)
#        col_idx = int((x[i] - x.min()) / degree_size)
#        z_array[row_idx, col_idx] = z[i]    
#    ax.pcolormesh(lons, lats, z_array, cmap='RdYlBu_r', norm=norm, zorder=2, alpha=0.8)
#    
#    # Save figure
#    fig.set_size_inches(6,4)
#    fig_fn = 'MB_cal_minus_era_map_2000_2018_areachg.png'
#    fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)
#    
#    main_glac_rgi_dropnan.to_csv(figure_fp + 'main_glac_rgi_HMA_2000_2018_areachg.csv')
    
    #%%
    # Plot change in volume over time
    # Index time period
    time_values_year = np.array([x.to_pydatetime().year for x in time_values])
    ref_year = np.array([int(str(x)[0:4]) for x in main_glac_rgi['RefDate'].values])
    ref_month = np.array([int(str(x)[4:6]) for x in main_glac_rgi['RefDate'].values])
    ref_month[ref_month>12] = 6
    ref_day = np.array([int(str(x)[6:]) for x in main_glac_rgi['RefDate'].values])
    ref_day[ref_day>31] = 15
    ref_year_frac = ref_year + (ref_month + ref_day / (365/12)) / 12
    ref_year_avg = np.median(ref_year_frac)
    ref_year_4idx = int(ref_year_avg)
    ref_month_4idx = int((ref_year_avg%ref_year_4idx)*12)
    start_idx = np.where(time_values_year == ref_year_4idx)[0][ref_month_4idx-1]
    
    # Initial mass [Gt]
    region_initmass = main_glac_rgi.Volume.values.sum()
    # Monthly mass change [km3 w.e. == Gt]
    region_masschange = glac_wide_masschange.sum(axis=0)
    region_mass = np.zeros((region_masschange.shape))
    region_mass[start_idx] = region_initmass
    region_mass[start_idx+1:] = region_initmass + region_masschange[start_idx+1:].cumsum()
    region_mass[:start_idx] = region_initmass + region_masschange[:start_idx][::-1].cumsum()[::-1]
    
    # Normalized regional mass
    region_mass_norm = region_mass / region_initmass
    
    # ===== Plot =====
    fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=True,
                           figsize=(5,4), gridspec_kw = {'wspace':0, 'hspace':0})
    ax[0,0].plot(time_values, region_mass_norm, color='k', linewidth=1, label=None)
    ax[0,0].set_ylabel('Normalized Mass [-]')
                    
    print('ADD UNCERTAINTY TO PLOT!')
    
    # Save figure
    fig.set_size_inches(6.5,4)
    figure_fn = 'Normalized_Mass_1980-2017.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
    
#%%
if option_GRACE_2deg == 1:
    grouping = 'degree'
    
#    main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values = (
#            load_masschange_monthly(regions, ds_ending='_ERA-Interim_c2_ba1_100sets_1980_2017.nc'))
    main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values = (
            load_masschange_monthly(regions, ds_ending='_ERA-Interim_c2_ba1_100sets_2000_2018.nc'))
    
    # Add watersheds, regions, degrees, mascons, and all groups to main_glac_rgi_all
    # Degrees
    main_glac_rgi['CenLon_round'] = np.floor(main_glac_rgi.CenLon.values/degree_size) * degree_size
    main_glac_rgi['CenLat_round'] = np.floor(main_glac_rgi.CenLat.values/degree_size) * degree_size
    deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
    deg_dict = dict(zip(deg_groups, np.arange(0,len(deg_groups))))
    main_glac_rgi.reset_index(drop=True, inplace=True)
    cenlon_cenlat = [(main_glac_rgi.loc[x,'CenLon_round'], main_glac_rgi.loc[x,'CenLat_round']) 
                     for x in range(len(main_glac_rgi))]
    main_glac_rgi['CenLon_CenLat'] = cenlon_cenlat
    main_glac_rgi['deg_id'] = main_glac_rgi.CenLon_CenLat.map(deg_dict)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    
    ds_group_masschange = {}
    ds_group_area = {}
    
    # Cycle through groups
    for ngroup, group in enumerate(groups):
        if ngroup%50 == 0:
            print(group)
        # Select subset of data
        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]                        
        masschange_glac = glac_wide_masschange[main_glac_rgi_subset.index.values.tolist(),:]
        area_glac = glac_wide_area[main_glac_rgi_subset.index.values.tolist(),:]
        # Regional sum
        masschange_reg = masschange_glac.sum(axis=0)
        area_reg =  area_glac.sum(axis=0)            
        # Record data
        ds_group_masschange[ngroup] = masschange_reg
        ds_group_area[ngroup] = area_reg
        
    latitude = np.arange(main_glac_rgi.CenLat_round.min(), main_glac_rgi.CenLat_round.max() + degree_size, degree_size)
    longitude = np.arange(main_glac_rgi.CenLon_round.min(), main_glac_rgi.CenLon_round.max() + degree_size, degree_size)
    
    masschange_3d = np.zeros((len(longitude), len(latitude), len(time_values)))
    area_3d = np.zeros((len(longitude), len(latitude), len(time_values)))
    
#    #%%
#    def round_degsize(x, degree_size=degree_size):
#        """ Round scalar or array to nearest degree_size - avoids rounding errors with np.where """
#        ndigits = -1 * len(str(degree_size).split('.')[1])
#        if np.isscalar(x):
#            return np.around(int(np.floor(x/degree_size)) * degree_size, ndigits)
#        else:
#            return np.array([np.around(int(np.floor(i/degree_size)) * degree_size, ndigits) for i in x])
#    
##    longitude = round_degsize(longitude)
##    latitude = round_degsize(latitude)
    #%%
    for ngroup, group in enumerate(groups):        
        lon_idx = np.where(np.isclose(longitude, deg_groups[ngroup][0]))[0][0]
        lat_idx = np.where(np.isclose(latitude, deg_groups[ngroup][1]))[0][0]
        masschange_3d[lon_idx, lat_idx, :] = ds_group_masschange[ngroup]
        area_3d[lon_idx, lat_idx, :] = ds_group_area[ngroup]
    #%%
    
    # Create empty datasets for each variable and merge them
    # Coordinate values
    output_variables = ['masschange_monthly', 'area_monthly']    
    # Year type for attributes
    if time_values[0].to_pydatetime().month == 10:
        year_type = 'water year'
    elif time_values[0].to_pydatetime().month == 1:
        year_type = 'calendar year'
    else:
        year_type = 'custom year'

    # Variable coordinates dictionary
    output_coords_dict = {
            'masschange_monthly': collections.OrderedDict(
                    [('longitude', longitude), ('latitude', latitude), ('time', time_values)]),
            'area_monthly': collections.OrderedDict(
                    [('longitude', longitude), ('latitude', latitude), ('time', time_values)]),
            }
    # Attributes dictionary
    output_attrs_dict = {
            'longitude': {
                    'long_name': 'longitude',
                     'degree_size':degree_size},
            'latitude': {
                    'long_name': 'latitude',
                     'degree_size':degree_size},
            'time': {
                    'long_name': 'date',
                     'year_type':year_type},
            'masschange_monthly': {
                    'long_name': 'glacier mass change',
                    'units': 'Gt',
                    'temporal_resolution': 'monthly',
                    'comment': ('mass change of all glaciers with center in grid (glaciers spanning multiple grids' +
                                'are counted in grid where their center latitude/longitude is located)')},
            'area_monthly': {
                    'long_name': 'glacier area',
                    'units': 'km**2',
                    'temporal_resolution': 'monthly',
                    'comment': ('area of all glaciers with center in grid (glaciers spanning multiple grids' +
                                'are counted in grid where their center latitude/longitude is located')},
                }
    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_variables:
        count_vn += 1
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: xr.DataArray(empty_holder, 
                                                 dims=list(output_coords_dict[vn].keys()), 
                                                 coords=output_coords_dict[vn])})
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
        encoding[vn] = {'_FillValue': False}
        
    # Add values
    output_ds_all.masschange_monthly[:,:,:] = masschange_3d
    output_ds_all.area_monthly[:,:,:] = area_3d

    # Export netcdf
#    netcdf_fn = 'ERA-Interim_1980_2017_masschange_p' + str(int(degree_size*100)) + 'deg.nc'
    netcdf_fn = 'ERA-Interim_2000_2018_masschange_p' + str(int(degree_size*100)) + 'deg.nc'
    output_ds_all.to_netcdf(sim_netcdf_fp + netcdf_fn, encoding=encoding)   
    # Close datasets
    output_ds_all.close()
    
    print(np.round(output_ds_all.masschange_monthly[:,:,:].values.sum() / 18,2), 'Gt/yr')

#%%
if option_trishuli == 1:
    glac_no = input.glac_fromcsv(input.main_directory + '/../qgis_himat/trishuli_shp/trishuli_RGIIds.csv')
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    
#    ds_new = xr.open_dataset(input.output_sim_fp + 'ERA-Interim/Trishuli_ERA-Interim_c2_ba1_100sets_2000_2017.nc')
#    ds_old13 = xr.open_dataset(input.output_sim_fp + 'ERA-Interim/ERA-Interim_1980_2017_nochg/' + 
#                               'R13_ERA-Interim_c2_ba1_100sets_1980_2017.nc')
#    ds_old15 = xr.open_dataset(input.output_sim_fp + 'ERA-Interim/ERA-Interim_1980_2017_nochg/' + 
#                               'R15_ERA-Interim_c2_ba1_100sets_1980_2017.nc') 
#    time_old_idx_start = 12*20
#    time_new_idx_start = 0
#    years = np.arange(2000,2018)
#    option_components = 1
    
    ds_new = xr.open_dataset(input.output_sim_fp + 'IPSL-CM5A-LR/' + 
                             'Trishuli_IPSL-CM5A-LR_rcp85_c2_ba1_100sets_2000_2100.nc')
    ds_old13 = xr.open_dataset(input.output_sim_fp + 'spc_subset/' + 
                               'R13_IPSL-CM5A-LR_rcp26_c2_ba1_100sets_2000_2100--subset.nc')  
    ds_old15 = xr.open_dataset(input.output_sim_fp + 'spc_subset/' + 
                               'R15_IPSL-CM5A-LR_rcp26_c2_ba1_100sets_2000_2100--subset.nc')    
    time_old_idx_start = 16*12
    time_new_idx_start = 16*12
    years = np.arange(2016,2101)
    option_components = 0
    
    # Concatenate datasets
    ds_old = xr.concat([ds_old13, ds_old15], dim='glac')
    
    df_old = pd.DataFrame(ds_old.glacier_table.values, columns=ds_old.glac_attrs)
    df_old.reset_index(inplace=True, drop=True)
    df_old['rgino_str'] = [str(int(df_old.loc[x,'O1Region'])) + '.' + str(int(df_old.loc[x,'glacno'])).zfill(5)
                            for x in np.arange(df_old.shape[0])]
    
    # Find indices to select data from
    df_old_idx = np.where(df_old.rgino_str.isin(main_glac_rgi.rgino_str.values))[0]
    
    runoff_old_monthly = ds_old.runoff_glac_monthly.values[df_old_idx,time_old_idx_start:,0]
    offglac_runoff_old_monthly = ds_old.offglac_runoff_monthly.values[df_old_idx,time_old_idx_start:,0]
    totalrunoff_old_monthly = (runoff_old_monthly + offglac_runoff_old_monthly) / 10**9
    totalrunoff_old = gcmbiasadj.annual_sum_2darray(totalrunoff_old_monthly)
    totalrunoff_old_trishuli = totalrunoff_old.sum(axis=0)
    
    runoff_new_monthly = ds_new.runoff_glac_monthly.values[:,time_new_idx_start:,0]
    offglac_runoff_new_monthly = ds_new.offglac_runoff_monthly.values[:,time_new_idx_start:,0]
    totalrunoff_new_monthly = (runoff_new_monthly + offglac_runoff_new_monthly) / 10**9
    totalrunoff_new = gcmbiasadj.annual_sum_2darray(totalrunoff_new_monthly)
    totalrunoff_new_trishuli = totalrunoff_new.sum(axis=0)

    dif_runoff = totalrunoff_new_trishuli.sum() - totalrunoff_old_trishuli.sum()
    print('DIFFERENCE RUNOFF TOTAL:\n', np.round(dif_runoff,1), 'Gt',  
          np.round(dif_runoff / totalrunoff_old_trishuli.sum()*100,1), '%')


    if option_components == 1:
        area_annual = ds_old.area_glac_annual.values[df_old_idx,20:,0][:,:-1]
        # Compare precipitation
        prec_old_monthly = ds_old.prec_glac_monthly.values[df_old_idx,time_old_idx_start:,0]
        acc_old_monthly = ds_old.acc_glac_monthly.values[df_old_idx,time_old_idx_start:,0]
        totalprec_old_monthly = prec_old_monthly + acc_old_monthly
        totalprec_old = gcmbiasadj.annual_sum_2darray(totalprec_old_monthly)
        totalprec_old_Gt = totalprec_old / 1000 * area_annual
        totalprec_old_trishuli = totalprec_old_Gt.sum(axis=0)
        
        
        prec_new_monthly = ds_new.prec_glac_monthly.values[:,:,0]
        acc_new_monthly = ds_new.acc_glac_monthly.values[:,:,0]
        totalprec_new_monthly = prec_new_monthly + acc_new_monthly
        totalprec_new = gcmbiasadj.annual_sum_2darray(totalprec_new_monthly)
        totalprec_new_Gt = totalprec_new / 1000 * area_annual
        totalprec_new_trishuli = totalprec_new_Gt.sum(axis=0)   
        
        pf_dif = totalprec_new / totalprec_old
        dif_totalprec =  totalprec_new_trishuli.sum() - totalprec_old_trishuli.sum()
        print('DIFFERENCE PRECIPITATION TOTAL:\n', np.round(dif_totalprec,1), 'Gt',
              np.round(dif_totalprec / totalprec_old_trishuli.sum() * 100, 1), '%')
        
        # Compare melt
        melt_old_monthly = ds_old.melt_glac_monthly.values[df_old_idx,time_old_idx_start:,0]
        melt_old = gcmbiasadj.annual_sum_2darray(melt_old_monthly)
        melt_old_Gt = melt_old / 1000 * area_annual
        melt_old_trishuli = melt_old_Gt.sum(axis=0)
        
        melt_new_monthly = ds_new.melt_glac_monthly.values[:,:,0]
        melt_new = gcmbiasadj.annual_sum_2darray(melt_new_monthly)
        melt_new_Gt = melt_new / 1000 * area_annual
        melt_new_trishuli = melt_new_Gt.sum(axis=0) 
        
        dif_melt =  melt_new_trishuli.sum() - melt_old_trishuli.sum()
        print('DIFFERENCE Melt TOTAL:\n', np.round(dif_melt,1), 'Gt',
              np.round(dif_melt / melt_old_trishuli.sum() * 100, 1), '%')
        
        # Compare refreeze
        refreeze_old_monthly = ds_old.refreeze_glac_monthly.values[df_old_idx,time_old_idx_start:,0]
        refreeze_old = gcmbiasadj.annual_sum_2darray(refreeze_old_monthly)
        refreeze_old_Gt = refreeze_old / 1000 * area_annual
        refreeze_old_trishuli = refreeze_old_Gt.sum(axis=0)
        
        refreeze_new_monthly = ds_new.refreeze_glac_monthly.values[:,:,0]
        refreeze_new = gcmbiasadj.annual_sum_2darray(refreeze_new_monthly)
        refreeze_new_Gt = refreeze_new / 1000 * area_annual
        refreeze_new_trishuli = refreeze_new_Gt.sum(axis=0) 
        
        dif_refreeze =  refreeze_new_trishuli.sum() - refreeze_old_trishuli.sum()
        print('DIFFERENCE refreeze TOTAL:\n', np.round(dif_refreeze,1), 'Gt',
              np.round(dif_refreeze / refreeze_old_trishuli.sum() * 100, 1), '%')
    
    #%%
    # Set up your plot (and/or subplots)
    fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, gridspec_kw = {'wspace':0.4, 'hspace':0.15})
    ax[0,0].plot(years, totalrunoff_new_trishuli, color='k', linewidth=1, zorder=2, label='new')
    ax[0,0].plot(years, totalrunoff_old_trishuli, color='b', linewidth=1, zorder=2, label='old')

#    ax[0,0].text(0.5, 0.99, '[insert text]', size=10, horizontalalignment='center', verticalalignment='top', 
#                 transform=ax[0,0].transAxes)
    
    ax[0,0].set_ylabel('Glacier Runoff [Gt]', size=12)
    ax[0,0].legend(loc=(0.05, 0.05), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, 
                   frameon=False)

    # Save figure
    #  figures can be saved in any format (.jpg, .png, .pdf, etc.)
    fig.set_size_inches(4, 4)
    figure_fp = os.getcwd() + '/../Output/'
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    figure_fn = 'Trishuli_runoff_comparison_2016_2100.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
    
    
    
    
    
    
    
    
    
    
    
    