""" Analyze MCMC output - chain length, etc. """

# Built-in libraries
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
import run_calibration as calibration

#%%
option_observation_vs_calibration = 0
option_GRACE_2deg = 1


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
#sim_netcdf_fp = input.output_filepath + 'simulations/ERA-Interim/ERA-Interim_2000_2018_areachg/'
sim_netcdf_fp = input.output_filepath + 'simulations/ERA-Interim/ERA-Interim_1980_2017_nochg/'
#sim_netcdf_fp = input.output_filepath + 'simulations/ERA-Interim_2000_2017wy_nobiasadj/'

figure_fp = sim_netcdf_fp + 'figures/'

regions = [13, 14, 15]
degree_size = 0.5

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
                main_glac_rgi_region, [region], input.hyps_filepath,input.hyps_filedict, input.hyps_colsdrop)     
        # Ice thickness [m], average
        main_glac_icethickness_region = modelsetup.import_Husstable(
                main_glac_rgi_region, [region], input.thickness_filepath, input.thickness_filedict, 
                input.thickness_colsdrop)
        main_glac_hyps_region[main_glac_icethickness_region == 0] = 0
        # ===== CALIBRATION DATA =====
        if option_add_caldata == 1:
            dates_table_nospinup = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                                            spinupyears=0)
            cal_data_region = pd.DataFrame()
            for dataset in cal_datasets:
                cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=region)
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
    
#    t1_idx = 0
#    t2_idx = 216
    t1_idx = 240
    t2_idx = 455
    t1 = 2000
    t2 = 2018
    
    main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values = (
            load_masschange_monthly(regions, ds_ending='_ERA-Interim_c2_ba1_100sets_1980_2017.nc'))
    
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
    
    main_glac_rgi, glac_wide_masschange, glac_wide_area, time_values = (
            load_masschange_monthly(regions, ds_ending='_ERA-Interim_c2_ba1_100sets_1980_2017.nc'))
    
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
    
    #%%
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    
    ds_group = {}
    
    # Cycle through groups
    for ngroup, group in enumerate(groups):
        if ngroup%50 == 0:
            print(group)
        # Select subset of data
        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]                        
        vn_glac = glac_wide_masschange[main_glac_rgi_subset.index.values.tolist(),:]             
        # Regional sum
        vn_reg = vn_glac.sum(axis=0)                
        # Record data
        ds_group[ngroup] = vn_reg
        
#    #%%
#    lats = [x[1] for x in deg_groups]
#    lons = [x[0] for x in deg_groups]
#    values_list = [x[1] for x in ds_group]
#    values = np.vstack(values_list)


    # Convert monthly timestep to Year-Month
    if timestep == 'monthly':
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        time_values = [str(x.year) + '-' + str(x.month) for x in time_values_df]
    elif timestep == 'annual':
        time_values = time_values_annual[:-1]
    
    # Output column names
    mascon_output_cns = ['deg_idx', 'CenLat', 'CenLon']
#    for x in time_values:
#        mascon_output_cns.append(x)
#    # Mascon index
#    mascon_output_array = np.reshape(np.array(groups),(-1,1))
#    # Mascon center lat/lon
#    mascon_latlon = np.array([mascon_latlondict[x] for x in groups])
#    mascon_output_array = np.hstack([mascon_output_array, mascon_latlon])
#    # Mascon glacier mass [Gt]
#    mascon_values_list = [ds_mass_mascon_monthly[x][1] for x in range(len(groups))]
#    mascon_values_monthly = np.vstack(mascon_values_list)
#    if timestep == 'annual':
#        mascon_values = mascon_values_monthly[:,::12]
#    elif timestep == 'monthly':
#        mascon_values = mascon_values_monthly
#    mascon_output_array = np.hstack([mascon_output_array, mascon_values])
#    
#    # Output dataframe
#    mascon_output_df = pd.DataFrame(mascon_output_array, columns=mascon_output_cns)
#    mascon_output_df.to_csv(mascon_fp + output_fn, sep=' ', index=False)
#    
#    if timestep == 'annual':
#        headerline=('Annual glacier mass [Gt] for each mascon where at least 1 HMA glacier exists\n' + 
#                    'Glacier mass change modeled using the Python Glacier Evolution Model (PyGEM)\n' +
#                    'forced by ERA-Interim climate data\n'
#                    'Glaciers aggregated according to nearest center latitude and longitude\n' + 
#                    'Years refer to water years (e.g., 2000 is October 1999 - September 2000)\n' +
#                    'Glacier mass refers to mass at the start of the year (e.g., mass_change_2000 = mass_2001 - '
#                    'mass_2000\n' +
#                    'Mascons provided by Bryant Loomis (NASA GSFC mascons)\n' + 
#                    'Contact: drounce@alaska.edu\n' + 
#                    'Column 1: Mascon index used to aggregate glaciers\n' + 
#                    'Column 2: Mascon latitude center [deg]\n' +
#                    'Column 3: Mascon longitude center [deg]\n' +
#                    'Columns 4+: Water year\n' + 
#                    'END OF COMMENTS (first row is header of column names)\n')
#    elif timestep == 'monthly':
#        headerline=('Monthly glacier mass [Gt] for each mascon where at least 1 HMA glacier exists\n' + 
#                    'Glacier mass change modeled using the Python Glacier Evolution Model (PyGEM)\n' +
#                    'forced by ERA-Interim climate data\n'
#                    'Glaciers aggregated according to nearest center latitude and longitude\n' + 
#                    'Glacier mass refers to mass at the start of the month (e.g., mass_change_Sept = mass_Oct - '
#                    'mass_Sept)\n' +
#                    'Mascons provided by Bryant Loomis (NASA GSFC mascons)\n' + 
#                    'Contact: drounce@alaska.edu\n' + 
#                    'Column 1: Mascon index used to aggregate glaciers\n' + 
#                    'Column 2: Mascon latitude center [deg]\n' +
#                    'Column 3: Mascon longitude center [deg]\n' +
#                    'Columns 4+: Year-Month\n' + 
#                    'END OF COMMENTS (first row is header of column names)\n')
#            
#    with open(mascon_fp + output_fn, 'r+') as f:
#        content=f.read()
#        f.seek(0,0)
#        f.write(headerline.rstrip('\r\n') + '\n' + content)
        