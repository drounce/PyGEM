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

# Script options
option_plot_cmip5_normalizedchange = 0              # updated - 11/6/2019 (includes the runoff figure 5)
option_cmip5_heatmap_w_volchange = 0                # updated - 11/6/2019
option_cmip5_mb_vs_climate = 0
option_map_gcm_changes = 0
option_region_map_nodata = 0
option_peakwater_map = 0                            # updated - 11/7/2019
option_temp_and_prec_map = 0                        # updated - 11/18/2019
option_watersheds_colored = 0                       # still good 11/6/2019
option_runoff_monthlychange_and_components = 0      # updated - 11/20/2019
runoff_erainterim_bywatershed = 0                   # updated - better to export to table
option_excess_meltwater_diagram = 0

option_startdate = 0

option_plot_cmip5_normalizedchange_proposal = 0
option_runoff_components_proposal = 0

option_glaciermip_table = 0                         # updated - 11/12/2019
option_zemp_compare = 0                             # updated - 11/6/2019
option_gardelle_compare = 0                         # updated - 11/6/2019
option_wgms_compare = 0                             # updated - 11/6/2019
option_dehecq_compare = 0
option_uncertainty_fig = 0                          # updated - 11/12/2019
option_nick_snowline = 0
option_caldata_compare = 0

analyze_multimodel = 0
option_merge_multimodel_datasets = 0

option_regional_hyps = 0

#%% ===== Input data =====
#netcdf_fp_cmip5 = pygem_prms.output_sim_fp + 'spc_subset/'
#netcdf_fp_cmip5 = pygem_prms.output_sim_fp + 'spc_multimodel/'
netcdf_fp_cmip5 = pygem_prms.output_filepath + 'simulations/spc_20190914/merged/multimodel/'
#netcdf_fp_cmip5 = pygem_prms.output_filepath + 'simulations/CCSM4/'
netcdf_fp_era = pygem_prms.output_filepath + 'simulations/spc_20190914/merged/ERA-Interim/'


#%%
regions = [13, 14, 15]

# GCMs and RCP scenarios
gcm_names = ['bcc-csm1-1', 'CanESM2', 'CESM1-CAM5', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'FGOALS-g2', 'GFDL-CM3', 
             'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'MIROC-ESM', 
             'MIROC-ESM-CHEM', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MRI-CGCM3', 'NorESM1-M', 'NorESM1-ME']
#gcm_names = ['CCSM4']
#gcm_names = ['bcc-csm1-1', 'CESM1-CAM5', 'CCSM4', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
#             'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'MIROC-ESM', 
#             'MIROC-ESM-CHEM', 'MIROC5', 'MRI-CGCM3', 'NorESM1-ME']
rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
#rcps = ['rcp85']

# Grouping
#grouping = 'all'
grouping = 'rgi_region'
#grouping = 'watershed'
#grouping = 'kaab'
#grouping = 'himap'
#grouping = 'degree'


#subgrouping = 'hexagon'
subgrouping = 'hexagon55'

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
title_dict = {'Amu_Darya': 'Amu Darya',
              'Brahmaputra': 'Brahmaputra',
              'Ganges': 'Ganges',
              'Ili': 'Ili',
              'Indus': 'Indus',
              'Inner_Tibetan_Plateau': 'Inner TP',
              'Inner_Tibetan_Plateau_extended': 'Inner TP ext',
              'Irrawaddy': 'Irrawaddy',
              'Mekong': 'Mekong',
              'Salween': 'Salween',
              'Syr_Darya': 'Syr Darya',
              'Tarim': 'Tarim',
              'Yangtze': 'Yangtze',
              'inner_TP': 'Inner TP',
              'Karakoram': 'Karakoram',
              'Yigong': 'Yigong',
              'Yellow': 'Yellow',
              'Bhutan': 'Bhutan',
              'Everest': 'Everest',
              'West Nepal': 'West Nepal',
              'Spiti Lahaul': 'Spiti Lahaul',
              'tien_shan': 'Tien Shan',
              'Pamir': 'Pamir',
              'pamir_alai': 'Pamir Alai',
              'Kunlun': 'Kunlun',
              'Hindu Kush': 'Hindu Kush',
              13: 'Central Asia',
              14: 'South Asia West',
              15: 'South Asia East',
              'all': 'HMA',
              'Altun Shan':'Altun Shan',
              'Central Himalaya':'C Himalaya',
              'Central Tien Shan':'C Tien Shan',
              'Dzhungarsky Alatau':'Dzhungarsky Alatau',
              'Eastern Himalaya':'E Himalaya',
              'Eastern Hindu Kush':'E Hindu Kush',
              'Eastern Kunlun Shan':'E Kunlun Shan',
              'Eastern Pamir':'E Pamir',
              'Eastern Tibetan Mountains':'E Tibetan Mtns',
              'Eastern Tien Shan':'E Tien Shan',
              'Gangdise Mountains':'Gangdise Mtns',
              'Hengduan Shan':'Hengduan Shan',
              'Karakoram':'Karakoram',
              'Northern/Western Tien Shan':'N/W Tien Shan',
              'Nyainqentanglha':'Nyainqentanglha',
              'Pamir Alay':'Pamir Alay',
              'Qilian Shan':'Qilian Shan',
              'Tanggula Shan':'Tanggula Shan',
              'Tibetan Interior Mountains':'Tibetan Int Mtns',
              'Western Himalaya':'W Himalaya',
              'Western Kunlun Shan':'W Kunlun Shan',
              'Western Pamir':'W Pamir'
              }
title_location = {'Syr_Darya': [68, 46.1],
                  'Ili': [83.6, 45.5],
                  'Amu_Darya': [64.6, 36.9],
                  'Tarim': [83.0, 39.2],
                  'Inner_Tibetan_Plateau_extended': [100, 40],
                  'Indus': [70.7, 31.9],
                  'Inner_Tibetan_Plateau': [85, 32.4],
                  'Yangtze': [106.0, 29.8],
                  'Ganges': [81.3, 26.6],
                  'Brahmaputra': [92.0, 26],
                  'Irrawaddy': [96.2, 23.8],
                  'Salween': [98.5, 20.8],
                  'Mekong': [103.8, 17.5],
                  'Yellow': [106.0, 36],
                  13: [84,39],
                  14: [72, 33],
                  15: [84,26.8],
                  'inner_TP': [89, 33.5],
                  'Karakoram': [68.7, 33],
                  'Yigong': [97.5, 26.2],
                  'Bhutan': [92.1, 26],
                  'Everest': [85, 26.3],
                  'West Nepal': [76.5, 28],
                  'Spiti Lahaul': [70, 31.4],
                  'tien_shan': [80, 42],
                  'Pamir': [66, 36],
                  'pamir_alai': [65.2, 40.2],
                  'Kunlun': [79, 37.5],
                  'Hindu Kush': [64, 34.5]
                  }
rcp_dict = {'rcp26': '2.6',
            'rcp45': '4.5',
            'rcp60': '6.0',
            'rcp85': '8.5'}
# Colors list
colors_rgb = [(0.00, 0.57, 0.57), (0.71, 0.43, 1.00), (0.86, 0.82, 0.00), (0.00, 0.29, 0.29), (0.00, 0.43, 0.86), 
              (0.57, 0.29, 0.00), (1.00, 0.43, 0.71), (0.43, 0.71, 1.00), (0.14, 1.00, 0.14), (1.00, 0.71, 0.47), 
              (0.29, 0.00, 0.57), (0.57, 0.00, 0.00), (0.71, 0.47, 1.00), (1.00, 1.00, 0.47)]
gcm_colordict = dict(zip(gcm_names, colors_rgb[0:len(gcm_names)]))
rcp_colordict = {'rcp26':'b', 'rcp45':'k', 'rcp60':'m', 'rcp85':'r'}
rcp_styledict = {'rcp26':':', 'rcp45':'--', 'rcp85':'-.'}

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
himap_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_bolch.csv'
himap_csv = pd.read_csv(himap_dict_fn)
himap_dict = dict(zip(himap_csv.RGIId, himap_csv.bolch_name))
hex_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_hexbins.csv'
hex_csv = pd.read_csv(hex_dict_fn)
hex_dict = dict(zip(hex_csv.RGIId, hex_csv.hexid))
hex42_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_hexbins_42km.csv'
hex42_csv = pd.read_csv(hex42_dict_fn)
hex42_dict = dict(zip(hex42_csv.RGIId, hex42_csv.hexid42))
hex55_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA_dict_hexbins_55km.csv'
hex55_csv = pd.read_csv(hex55_dict_fn)
hex55_dict = dict(zip(hex55_csv.RGIId, hex55_csv.hexid55))

# Shapefiles
rgiO1_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/00_rgi60_regions/00_rgi60_O1Regions.shp'
rgi_glac_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/rgi60_HMA.shp'
watershed_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/HMA_basins_20181018_4plot.shp'
kaab_shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/kaab2015_regions.shp'
srtm_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA.tif'
srtm_contour_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/SRTM_HMA_countours_2km_gt3000m_smooth.shp'

# GRACE mascons
mascon_fp = pygem_prms.main_directory + '/../GRACE/GSFC.glb.200301_201607_v02.4/'
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


def plot_hist(df, cn, bins, xlabel=None, ylabel=None, fig_fn='hist.png', fig_fp=pygem_prms.output_filepath):
    """
    Plot histogram for any bin size
    """           
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
        

def peakwater(runoff, time_values, nyears):
    """Compute peak water based on the running mean of N years
    
    Parameters
    ----------
    runoff : np.array
        one-dimensional array of runoff for each timestep
    time_values : np.array
        time associated with each timestep
    nyears : int
        number of years to compute running mean used to smooth peakwater variations
        
    Output
    ------
    peakwater_yr : int
        peakwater year
    peakwater_chg : float
        percent change of peak water compared to first timestep (running means used)
    runoff_chg : float
        percent change in runoff at the last timestep compared to the first timestep (running means used)
    """
    runningmean = uniform_filter(runoff, size=(nyears))
    peakwater_idx = np.where(runningmean == runningmean.max())[-1][0]
    peakwater_yr = time_values[peakwater_idx]
    peakwater_chg = (runningmean[peakwater_idx] - runningmean[0]) / runningmean[0] * 100
    runoff_chg = (runningmean[-1] - runningmean[0]) / runningmean[0] * 100
    return peakwater_yr, peakwater_chg, runoff_chg


def excess_meltwater_m3(glac_vol, option_lastloss=1):
    """ Excess meltwater based on running minimum glacier volume 
    
    Parameters
    ----------
    glac_vol : np.array
        glacier volume [km3]
    option_lastloss : int
        1 - excess meltwater based on last time glacier volume is lost for good
        0 - excess meltwater based on first time glacier volume is lost (poorly accounts for gains)
    option_lastloss = 1 calculates excess meltwater from the last time the glacier volume is lost for good
    option_lastloss = 0 calculates excess meltwater from the first time the glacier volume is lost, but does
      not recognize when the glacier volume returns
    """
    glac_vol_m3 = glac_vol * pygem_prms.density_ice / pygem_prms.density_water * 1000**3
    if option_lastloss == 1:
        glac_vol_runningmin = np.maximum.accumulate(glac_vol_m3[:,::-1],axis=1)[:,::-1]
        # initial volume sets limit of loss (gaining and then losing ice does not contribute to excess melt)
        for ncol in range(0,glac_vol_m3.shape[1]):
            mask = glac_vol_runningmin[:,ncol] > glac_vol_m3[:,0]
            glac_vol_runningmin[mask,ncol] = glac_vol_m3[mask,0]
    else:
        # Running minimum volume up until that time period (so not beyond it!)
        glac_vol_runningmin = np.minimum.accumulate(glac_vol_m3, axis=1)
    glac_excess = glac_vol_runningmin[:,:-1] - glac_vol_runningmin[:,1:] 
    return glac_excess
        

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
    elif grouping == 'himap':
        groups = main_glac_rgi_all.himap.unique().tolist()
        group_cn = 'himap'
        groups = [x for x in groups if str(x) != 'nan']  
    elif grouping == 'degree':
        groups = main_glac_rgi_all.deg_id.unique().tolist()
        group_cn = 'deg_id'
    elif grouping == 'mascon':
        groups = main_glac_rgi_all.mascon_idx.unique().tolist()
        groups = [int(x) for x in groups]
        group_cn = 'mascon_idx'
    elif grouping == 'hexagon':
        groups = main_glac_rgi_all.hexid.unique().tolist()
        group_cn = 'hexid'
    elif grouping == 'hexagon42':
        groups = main_glac_rgi_all.hexid42.unique().tolist()
        group_cn = 'hexid42'
    elif grouping == 'hexagon55':
        groups = main_glac_rgi_all.hexid55.unique().tolist()
        group_cn = 'hexid55'
    else:
        groups = ['all']
        group_cn = 'all_group'
    try:
        groups = sorted(groups, key=str.lower)
    except:
        groups = sorted(groups)
    return groups, group_cn


def load_glacier_data(glac_no=None, rgi_regionsO1=None, rgi_regionsO2='all', rgi_glac_number='all',
                      load_caldata=0, startyear=2000, endyear=2018, option_wateryear=3):
#def load_glacier_data(rgi_regions, 
#                      load_caldata=0, startyear=2000, endyear=2018, option_wateryear=3):
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
    
    # Add watersheds, regions, degrees, mascons, and all groups to main_glac_rgi_all
    # Watersheds
    main_glac_rgi_all['watershed'] = main_glac_rgi_all.RGIId.map(watershed_dict)
    # Regions
    main_glac_rgi_all['kaab'] = main_glac_rgi_all.RGIId.map(kaab_dict)
    main_glac_rgi_all['himap'] = main_glac_rgi_all.RGIId.map(himap_dict)
    # Hexbins
    main_glac_rgi_all['hexid'] = main_glac_rgi_all.RGIId.map(hex_dict)
    main_glac_rgi_all['hexid42'] = main_glac_rgi_all.RGIId.map(hex42_dict)
    main_glac_rgi_all['hexid55'] = main_glac_rgi_all.RGIId.map(hex55_dict)
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


def retrieve_gcm_data(gcm_name, rcp, main_glac_rgi, option_bias_adjustment=pygem_prms.option_bias_adjustment):
    """ Load temperature, precipitation, and elevation data associated with GCM/RCP
    """
    regions = list(set(main_glac_rgi.O1Region.tolist()))
    for region in regions:
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

        # Adjust reference dates in event that reference is longer than GCM data
        if pygem_prms.startyear >= pygem_prms.gcm_startyear:
            ref_startyear = pygem_prms.startyear
        else:
            ref_startyear = pygem_prms.gcm_startyear
        if pygem_prms.endyear <= pygem_prms.gcm_endyear:
            ref_endyear = pygem_prms.endyear
        else:
            ref_endyear = pygem_prms.gcm_endyear
        dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear, 
                                                   spinupyears=pygem_prms.spinupyears, 
                                                   option_wateryear=pygem_prms.option_wateryear)
        # Monthly average from reference climate data
        ref_gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
           
        # ===== BIAS CORRECTIONS =====
        # No adjustments
        if option_bias_adjustment == 0:
            gcm_temp_adj = gcm_temp
            gcm_prec_adj = gcm_prec
            gcm_elev_adj = gcm_elev
        # Bias correct based on reference climate data
        else:
            # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
            ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn, 
                                                                             main_glac_rgi_region, dates_table_ref)
            ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn, 
                                                                             main_glac_rgi_region, dates_table_ref)
            ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, 
                                                                 main_glac_rgi_region)
            
            # OPTION 1: Adjust temp using Huss and Hock (2015), prec similar but addresses for variance and outliers
            if pygem_prms.option_bias_adjustment == 1:
                # Temperature bias correction
                gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, 
                                                                            dates_table_ref, dates_table)
                # Precipitation bias correction
                gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_opt1(ref_prec, ref_elev, gcm_prec, 
                                                                          dates_table_ref, dates_table)
            
            # OPTION 2: Adjust temp and prec using Huss and Hock (2015)
            elif pygem_prms.option_bias_adjustment == 2:
                # Temperature bias correction
                gcm_temp_adj, gcm_elev_adj = gcmbiasadj.temp_biasadj_HH2015(ref_temp, ref_elev, gcm_temp, 
                                                                            dates_table_ref, dates_table)
                # Precipitation bias correction
                gcm_prec_adj, gcm_elev_adj = gcmbiasadj.prec_biasadj_HH2015(ref_prec, ref_elev, gcm_prec, 
                                                                            dates_table_ref, dates_table)
        # Concatenate datasets
        if region == regions[0]:
            gcm_temp_adj_all = gcm_temp_adj
            gcm_prec_adj_all = gcm_prec_adj
            gcm_elev_adj_all = gcm_elev_adj
        else:
            gcm_temp_adj_all = np.vstack([gcm_temp_adj_all, gcm_temp_adj])
            gcm_prec_adj_all = np.vstack([gcm_prec_adj_all, gcm_prec_adj])
            gcm_elev_adj_all = np.concatenate([gcm_elev_adj_all, gcm_elev_adj])
    
    return gcm_temp_adj_all, gcm_prec_adj_all, gcm_elev_adj_all


if option_uncertainty_fig == 1:
    #%%
    netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_subset/'
    netcdf_fp_multi = pygem_prms.output_filepath + 'simulations/spc_20190914/merged/multimodel/'
    figure_fp = pygem_prms.output_sim_fp + 'figures/'
#    gcm_names = ['bcc-csm1-1', 'CanESM2', 'CESM1-CAM5', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'FGOALS-g2', 'GFDL-CM3', 
#                 'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'MIROC-ESM', 
#                 'MIROC-ESM-CHEM', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MRI-CGCM3', 'NorESM1-M', 'NorESM1-ME']
    gcm_names = ['GFDL-ESM2G']
    gcm_single = 'GFDL-ESM2G'
    rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    regions = [13,14,15]
    
    startyear = 2015
    endyear = 2100
    
    rgiid_small = 'RGI60-15.03854'
    rgiid_big = 'RGI60-15.03473'
    
    vn = 'volume_glac_annual'

    # Load glaciers    
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    cal_data = pd.read_csv(pygem_prms.shean_fp + pygem_prms.shean_fn)
    print('Check glacier indices are correct')
    
    main_glac_rgi['cal_mwea'] = cal_data['mb_mwea']
    main_glac_rgi['cal_mwea_sigma'] = cal_data['mb_mwea_sigma']
    main_glac_rgi['mass_Gt'] = ((main_glac_hyps.values * main_glac_icethickness.values / 1000).sum(axis=1) 
                                * pygem_prms.density_ice / pygem_prms.density_water)
    
    #%%
    # SINGLE GCM DATA
    region_single = int(rgiid_big.split('-')[1].split('.')[0])

    ds_single_all, ds_single_std_all = {}, {}
    for rcp in rcps:
        ds_single_all[rcp], ds_single_std_all[rcp] = {}, {}  
        for ngcm, gcm_name in enumerate(gcm_names):
            # Load datasets
            ds_fn = ('R' + str(region_single) + '--all--' + gcm_name + '_' + rcp + 
                     '_c2_ba1_100sets_2000_2100--subset.nc')
            ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
            
            # Extract time variable
            time_values_annual = ds.coords['year_plus1'].values
            time_values_monthly = ds.coords['time'].values
            time_idx_start = np.where(time_values_annual == startyear)[0][0]
            time_idx_end = np.where(time_values_annual == endyear)[0][0] + 1
            # Extract data
            ds_rgi_table = pd.DataFrame(ds['glacier_table'].values, columns=ds['glacier_table'].glac_attrs)
            ds_rgi_table['RGIId'] = ['RGI60-' + str(int(ds_rgi_table.loc[x, 'O1Region'])) + '.' +
                                     str(int(ds_rgi_table.loc[x,'glacno'])).zfill(5) 
                                     for x in range(ds_rgi_table.shape[0])]
            glac_idx = np.where(ds_rgi_table['RGIId'] == rgiid_big)[0][0]
            ds_single_all[rcp][gcm_name] = ds[vn].values[glac_idx,:,0]
            ds_single_std_all[rcp][gcm_name] = ds[vn].values[glac_idx,:,1]
            
            ds.close()
            
            #%%
    # Glacier and grouped annual specific mass balance and mass change
    ds_multi = {}
    ds_multi_std = {}
    for rcp in rcps:
        for region in regions:
            
            # Load datasets
            ds_fn = 'R' + str(region) + '_multimodel_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
            ds = xr.open_dataset(netcdf_fp_multi + ds_fn)
            df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
            df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                           str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
            
            # Extract time variable
            time_values_annual = ds.coords['year_plus1'].values
            time_values_monthly = ds.coords['time'].values        
            
            if region == regions[0]: 
                ds_multi[rcp] = ds[vn].values[:,:,0]
                ds_multi_std[rcp] = ds[vn].values[:,:,1]
                df_all = df
            else:
                ds_multi[rcp] = np.concatenate((ds_multi[rcp], ds[vn].values[:,:,0]), axis=0)
                ds_multi_std[rcp] = np.concatenate((ds_multi_std[rcp], ds[vn].values[:,:,1]), axis=0)
                df_all = pd.concat([df_all, df], axis=0)
            
            ds.close()
            
    # Remove RGIIds from main_glac_rgi that are not in the model runs
    rgiid_df = list(df_all.RGIId.values)
    rgiid_all = list(main_glac_rgi.RGIId.values)
    rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
    main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
    main_glac_rgi.reset_index(inplace=True, drop=True)
            
    #%%
    multimodel_linewidth = 2
    alpha=0.2
        
    fig, ax = plt.subplots(2, 2, squeeze=False, sharex=False, sharey=False, 
                           gridspec_kw = {'wspace':0.4, 'hspace':0.15})
    
    t1_idx = np.where(time_values_annual == startyear)[0][0]
    t2_idx = np.where(time_values_annual == endyear)[0][0] + 1
    
    time_values = time_values_annual[t1_idx:t2_idx]
    
    for rcp in rcps:
        
        rgi_idx_big = np.where(main_glac_rgi['RGIId'] == rgiid_big)[0][0]
        rgi_idx_small = np.where(main_glac_rgi['RGIId'] == rgiid_small)[0][0]
            
        # LARGE GLACIER, SINGLE GCM IN TOP LEFT
        vol = ds_single_all[rcp][gcm_single][t1_idx:t2_idx]
        vol_std = ds_single_std_all[rcp][gcm_single][t1_idx:t2_idx]
        vol_normalizer = vol[0]
        
        vol_norm = vol / vol_normalizer
        vol_low_norm = (vol - vol_std) / vol_normalizer
        vol_high_norm = (vol + vol_std) / vol_normalizer
        # remove values below zero
        vol_low_norm[vol_low_norm < 0] = 0        
        # Plot
        ax[0,0].plot(time_values, vol_norm, color=rcp_colordict[rcp], linewidth=1, zorder=4)
        if len(rcps) == 4 and rcp in ['rcp26', 'rcp85']:
            ax[0,0].plot(time_values, vol_low_norm, color=rcp_colordict[rcp], linewidth=1, linestyle=':', zorder=4)
            ax[0,0].plot(time_values, vol_high_norm, color=rcp_colordict[rcp], linewidth=1, linestyle=':', zorder=4)
            ax[0,0].fill_between(time_values, vol_low_norm, vol_high_norm, 
                                 facecolor=rcp_colordict[rcp], alpha=0.2, zorder=3)
        # Text
        ax[0,0].text(0.5, 0.99, rgiid_big + '\n(single GCM)', size=10, horizontalalignment='center', 
                      verticalalignment='top', transform=ax[0,0].transAxes)
        ax[0,0].text(0.05, 0.99, 'A', size=12, horizontalalignment='center', 
                      verticalalignment='top', transform=ax[0,0].transAxes, zorder=5)
#        ax[0,0].text(0.99, 0.99, '$\mathregular{Mass_{2015}}$: ' + 
#                                  str(np.round(main_glac_rgi.loc[rgi_idx_big,'mass_Gt'],1)) + ' Gt',
#                     size=10, horizontalalignment='right', verticalalignment='top', transform=ax[0,0].transAxes)
#        ax[0,0].text(0.99, 0.92, 'Initial mass: ' + str(np.round(main_glac_rgi.loc[rgi_idx_big,'mass_Gt'],1)) + 'Gt',
#                     size=10, horizontalalignment='right', verticalalignment='top', transform=ax[0,0].transAxes)
        # X-label
        ax[0,0].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                      time_values_annual[t1_idx:t2_idx].max())
        ax[0,0].xaxis.set_tick_params(labelsize=12)
        ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(50))
        ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
        ax[0,0].set_xticklabels(['2015','2050','2100'])        
        # Y-label
        ax[0,0].set_ylabel('Mass (-)', size=12)
        ax[0,0].set_ylim(0,1.1)
        ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        # Tick parameter
        ax[0,0].yaxis.set_ticks_position('both')
        ax[0,0].tick_params(axis='both', which='major', labelsize=12, direction='inout')
        ax[0,0].tick_params(axis='both', which='minor', labelsize=12, direction='inout')               
            
        # LARGE GLACIER, MULTI-MODEL IN TOP RIGHT
        vol_multi = ds_multi[rcp][rgi_idx_big][t1_idx:t2_idx]
        vol_multi_std = ds_multi_std[rcp][rgi_idx_big][t1_idx:t2_idx]
        vol_multi_normalizer = vol_multi[0]

        vol_multi_norm = vol_multi / vol_multi_normalizer
        vol_multi_low_norm = (vol_multi - vol_multi_std) / vol_multi_normalizer
        vol_multi_high_norm = (vol_multi + vol_multi_std) / vol_multi_normalizer
        # remove values below zero
        vol_multi_low_norm[vol_multi_low_norm < 0] = 0                    
        # Plot
        ax[0,1].plot(time_values, vol_multi_norm, color=rcp_colordict[rcp], linewidth=1, zorder=4)
        if len(rcps) == 4 and rcp in ['rcp26', 'rcp85']:
            ax[0,1].plot(time_values, vol_multi_low_norm, color=rcp_colordict[rcp], linewidth=1, linestyle=':', zorder=4)
            ax[0,1].plot(time_values, vol_multi_high_norm, color=rcp_colordict[rcp], linewidth=1, linestyle=':', zorder=4)
            ax[0,1].fill_between(time_values, vol_multi_low_norm, vol_multi_high_norm, 
                                 facecolor=rcp_colordict[rcp], alpha=0.2, zorder=3)
        # Text
        ax[0,1].text(0.5, 0.99, rgiid_big + '\n(multi-GCM mean)', size=10, horizontalalignment='center', 
                      verticalalignment='top', transform=ax[0,1].transAxes)
        ax[0,1].text(0.05, 0.99, 'B', size=12, horizontalalignment='center', 
                      verticalalignment='top', transform=ax[0,1].transAxes, zorder=5)
        # X-label
        ax[0,1].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                      time_values_annual[t1_idx:t2_idx].max())
        ax[0,1].xaxis.set_tick_params(labelsize=12)
        ax[0,1].xaxis.set_major_locator(plt.MultipleLocator(50))
        ax[0,1].xaxis.set_minor_locator(plt.MultipleLocator(10))
        ax[0,1].set_xticklabels(['2015','2050','2100'])        
        # Y-label
        ax[0,1].set_ylabel('Mass (-)', size=12)
        ax[0,1].set_ylim(0,1.1)
        ax[0,1].yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax[0,1].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
        # Tick parameters
        ax[0,1].yaxis.set_ticks_position('both')
        ax[0,1].tick_params(axis='both', which='major', labelsize=12, direction='inout')
        ax[0,1].tick_params(axis='both', which='minor', labelsize=12, direction='inout')   
        
        
        # SMALL GLACIER, MULTI-MODEL IN LOWER LEFT
        vol_multi = ds_multi[rcp][rgi_idx_small][t1_idx:t2_idx]
        vol_multi_std = ds_multi_std[rcp][rgi_idx_small][t1_idx:t2_idx]
        vol_multi_normalizer = vol_multi[0]

        vol_multi_norm = vol_multi / vol_multi_normalizer
        vol_multi_low_norm = (vol_multi - vol_multi_std) / vol_multi_normalizer
        vol_multi_high_norm = (vol_multi + vol_multi_std) / vol_multi_normalizer
        # remove values below zero
        vol_multi_low_norm[vol_multi_low_norm < 0] = 0                    
        # Plot
        ax[1,0].plot(time_values, vol_multi_norm, color=rcp_colordict[rcp], linewidth=1, zorder=4)
        if len(rcps) == 4 and rcp in ['rcp26', 'rcp85']:
            ax[1,0].plot(time_values, vol_multi_low_norm, color=rcp_colordict[rcp], linewidth=1, linestyle=':', zorder=4)
            ax[1,0].plot(time_values, vol_multi_high_norm, color=rcp_colordict[rcp], linewidth=1, linestyle=':', zorder=4)
            ax[1,0].fill_between(time_values, vol_multi_low_norm, vol_multi_high_norm, 
                                 facecolor=rcp_colordict[rcp], alpha=0.2, zorder=3)
        # Text
        ax[1,0].text(0.5, 0.99, rgiid_small + '\n(multi-GCM mean)', size=10, horizontalalignment='center', 
                      verticalalignment='top', transform=ax[1,0].transAxes)
        ax[1,0].text(0.05, 0.99, 'C', size=12, horizontalalignment='center', 
                      verticalalignment='top', transform=ax[1,0].transAxes, zorder=5)
        # X-label
        ax[1,0].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                      time_values_annual[t1_idx:t2_idx].max())
        ax[1,0].xaxis.set_tick_params(labelsize=12)
        ax[1,0].xaxis.set_major_locator(plt.MultipleLocator(50))
        ax[1,0].xaxis.set_minor_locator(plt.MultipleLocator(10))
        ax[1,0].set_xticklabels(['2015','2050','2100'])        
        # Y-label
        ax[1,0].set_ylabel('Mass (-)', size=12)
        ax[1,0].set_ylim(0,15)
        ax[1,0].yaxis.set_major_locator(plt.MultipleLocator(10))
        ax[1,0].yaxis.set_minor_locator(plt.MultipleLocator(2))
        # Tick parameters
        ax[1,0].yaxis.set_ticks_position('both')
        ax[1,0].tick_params(axis='both', which='major', labelsize=12, direction='inout')
        ax[1,0].tick_params(axis='both', which='minor', labelsize=12, direction='inout') 
        
        
        # GLACIER AREA VS MB_SIGMA                
        # Plot
        ax[1,1].plot(main_glac_rgi['Area'], main_glac_rgi['cal_mwea_sigma'], 'o', markersize=1, markeredgewidth=0.1,
                     markerfacecolor='none', markeredgecolor='k')
        ax[1,1].set_xscale('log')
        # X-label
        ax[1,1].set_xlabel('Glacier Area ($\mathregular{km^{2}}$)', size=12, labelpad=0)
        ax[1,1].set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3])
        locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
        ax[1,1].xaxis.set_minor_locator(locmin)
        ax[1,1].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax[1,1].tick_params(axis='both', which='major', labelsize=12)
        # Text
        ax[1,1].text(0.05, 0.99, 'D', size=12, horizontalalignment='center', 
                      verticalalignment='top', transform=ax[1,1].transAxes, zorder=5)
        # Y-label
        ax[1,1].set_ylabel('$\mathregular{\sigma_B  (m w.e. {yr^{-1}}}$)', size=12)
        ax[1,1].yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax[1,1].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
#        # Tick parameters
        ax[1,1].tick_params(axis='both', which='major', labelsize=12, direction='inout')
        ax[1,1].tick_params(axis='both', which='minor', labelsize=12, direction='inout') 
        
        
    # RCP Legend
    rcp_dict = {'rcp26': 'RCP 2.6',
                'rcp45': 'RCP 4.5',
                'rcp60': 'RCP 6.0',
                'rcp85': 'RCP 8.5'}
    rcp_lines = []
    for rcp in rcps:
        line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
        rcp_lines.append(line)
    rcp_labels = [rcp_dict[rcp] for rcp in rcps]
    ax[0,1].legend(rcp_lines, rcp_labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
                   handletextpad=0.25, borderpad=0, frameon=False)
    
#    # GCM Legend
#    gcm_lines = []
#    for gcm in [gcm_single]:
#        line = Line2D([0,1],[0,1], color='grey', linewidth=multimodel_linewidth)
#        gcm_lines.append(line)
#    ax[0,0].legend(gcm_lines, [gcm_single], loc=(0.07,0.01), fontsize=10, labelspacing=0.25, handlelength=1, 
#                   handletextpad=0.25, borderpad=0, frameon=False, title='Single GCM')

#    # Label
#    ylabel_str = 'Mass [-]'
#    fig.text(-0.01, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
    
    # Save figure
    fig.set_size_inches(6, 6)
    figure_fn = 'uncertainty_large_small_single_multi.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
    
                #%%

if analyze_multimodel == 1:
    # Find problematic GCMs
    netcdf_fp_cmip5 = '/Volumes/LaCie/PyGEM_simulations/2019_0317/multimodel/'
#    gcm_names = ['bcc-csm1-1', 'CanESM2', 'CESM1-CAM5', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'FGOALS-g2', 'GFDL-CM3', 
#             'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'MIROC-ESM', 
#             'MIROC-ESM-CHEM', 'MIROC5', 'MPI-ESM-LR', 'MPI-ESM-MR', 'MRI-CGCM3', 'NorESM1-M', 'NorESM1-ME']
    gcm_names = ['bcc-csm1-1', 'CESM1-CAM5', 'CCSM4', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 
                 'GFDL-ESM2G', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR', 'MIROC-ESM', 
                 'MIROC-ESM-CHEM', 'MIROC5', 'MRI-CGCM3', 'NorESM1-ME']
#    rcps = ['rcp26', 'rcp45']
    rcps = ['rcp60']
    regions = [13, 14, 15]
    
    ds_vns = ['temp_glac_monthly', 'prec_glac_monthly', 'acc_glac_monthly', 'refreeze_glac_monthly', 'melt_glac_monthly',
              'frontalablation_glac_monthly', 'massbaltotal_glac_monthly', 'runoff_glac_monthly', 'snowline_glac_monthly', 
              'area_glac_annual', 'volume_glac_annual', 'ELA_glac_annual', 'offglac_prec_monthly', 
              'offglac_refreeze_monthly', 'offglac_melt_monthly', 'offglac_snowpack_monthly', 'offglac_runoff_monthly']
#    ds_vns = ['frontalablation_glac_monthly', 'runoff_glac_monthly', 'offglac_refreeze_monthly']
    
    for region in regions:
        for rcp in rcps:
            
            def print_max(netcdf_fp, fn, ds_vns):
                print(fn)
                for vn in ds_vns:
                    print(vn)
                    ds = xr.open_dataset(netcdf_fp + fn)
                    A = ds[vn][:,:,0].values
                    if A.max() > 10e100:
                        print('Corrupt file:  ', A.max(), vn)
                    ds.close()
            
            # Check individual GCMs
#            list_fns = []
#            for i in os.listdir(netcdf_fp_cmip5):
#                if str(region) in i and rcp in i:
#                    list_fns.append(i)
#                    
##            for i in [list_fns[10]]:
#                    print_max(netcdf_fp_cmip5, i, ds_vns)
            
            # Check multimodel files
            netcdf_fp = pygem_prms.output_sim_fp + 'spc_multimodel/'
            ds_fn = 'R' + str(region) + '_multimodel_' + rcp + '_c2_ba1_100sets_2000_2100.nc'

            print_max(netcdf_fp, ds_fn, ds_vns)
    
##            # Check specific file
#            netcdf_fp = pygem_prms.output_sim_fp + '/spc_zipped/'
###            ds_fn = 'R13_GISS-E2-R_rcp45_c2_ba1_100sets_2000_2100.nc'
##            ds_fn = 'R13_CanESM2_rcp45_c2_ba1_100sets_2000_2100.nc'
#            ds_fn = 'R14_MPI-ESM-LR_rcp45_c2_ba1_100sets_2000_2100.nc'
##            ds_fn = 'R14_NorESM1-M_rcp45_c2_ba1_100sets_2000_2100.nc'
##
#            print_max(netcdf_fp, ds_fn, ds_vns)
            
#            ds = xr.open_dataset(netcdf_fp + ds_fn)


#%%
if option_runoff_monthlychange_and_components == 1:
    # Note: RECOMPUTE RUNOFF FROM RUNOFF / NOT COMPONENTS TO AVOID AGGREGATING UNCERTAINTIES
    rcps = ['rcp45']
    
    figure_fp = pygem_prms.output_sim_fp + 'figures/'
    
    grouping = 'watershed'    

    ref_startyear = 2000
    ref_endyear = 2015
    
    eoc_startyear = 2085
    eoc_endyear = 2100
    
    plt_startyear = 2015
    plt_endyear = 2100
    
    peakwater_startyear = 2015
    peakwater_endyear = 2100

    multimodel_linewidth = 2
    alpha=0.2
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)
    
    # Glacier and grouped annual specific mass balance and mass change
    for nrcp, rcp in enumerate(rcps):
        
        for region in regions:
            
            # Load datasets
            ds_fn = 'R' + str(region) + '_multimodel_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
            ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
            df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
            df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                           str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]

            # Extract time variable
            time_values_annual = ds.coords['year_plus1'].values
            time_values_monthly = ds.coords['time'].values
            time_values_df = pd.DatetimeIndex(time_values_monthly)
            time_values_months = np.array([x.month for x in time_values_df])
            months = list(time_values_months[0:12])
            refyear_idx1 = np.where(time_values_annual == ref_startyear)[0][0]
            refyear_idx2 = np.where(time_values_annual == ref_endyear)[0][0] + 1
            refmonth_idx1 = refyear_idx1 * 12
            refmonth_idx2 = refyear_idx2 * 12
            eocyear_idx1 = np.where(time_values_annual==eoc_startyear)[0][0]
            eocyear_idx2 = np.where(time_values_annual==eoc_endyear)[0][0] + 1
            eocmonth_idx1 = eocyear_idx1 * 12
            eocmonth_idx2 = eocyear_idx2 * 12
            
            # RUNOFF (Gt)
            ds_runoff_reg = ((ds['runoff_glac_monthly'].values[:,:,0] + ds['offglac_runoff_monthly'].values[:,:,0]) 
                             / 10**9)
            ds_runoff_reg_std = ((ds['runoff_glac_monthly'].values[:,:,1] + ds['offglac_runoff_monthly'].values[:,:,1]) 
                                 / 10**9)
            ds_runoff_onglac_reg = ds['runoff_glac_monthly'].values[:,:,0] / 10**9
            ds_runoff_offglac_reg = ds['offglac_runoff_monthly'].values[:,:,0] / 10**9
            
            # RUNOFF COMPONENTS (UNITS: Gt)
            ds_vol_reg = ds['volume_glac_annual'].values[:,:,0]
            ds_area_reg = ds['area_glac_annual'].values[:,:,0][:,:-1].repeat(12,axis=1)
            ds_prec_reg = ds['prec_glac_monthly'].values[:,:,0] * ds_area_reg * 10**6 / 10**9
            ds_melt_reg = ds['melt_glac_monthly'].values[:,:,0] * ds_area_reg * 10**6 / 10**9
            ds_refr_reg = ds['refreeze_glac_monthly'].values[:,:,0] * ds_area_reg * 10**6 / 10**9
            # Off-glacier
            ds_area_off_reg = ds_area_reg[:,0][:,np.newaxis] - ds_area_reg
            ds_area_off_reg[ds_area_off_reg < 0] = 0
            ds_prec_off_reg = ds['offglac_prec_monthly'].values[:,:,0] * ds_area_off_reg * 10**6 / 10**9
            ds_melt_off_reg = ds['offglac_melt_monthly'].values[:,:,0] * ds_area_off_reg * 10**6 / 10**9
            ds_refr_off_reg = ds['offglac_refreeze_monthly'].values[:,:,0] * ds_area_off_reg * 10**6 / 10**9
            
            ds.close()
            
            if region == regions[0]:
                df_all = df
                ds_runoff = ds_runoff_reg
                ds_runoff_std = ds_runoff_reg_std
                ds_runoff_onglac = ds_runoff_onglac_reg
                ds_runoff_offglac = ds_runoff_offglac_reg
                ds_vol = ds_vol_reg
                ds_prec = ds_prec_reg
                ds_melt = ds_melt_reg
                ds_refr = ds_refr_reg
                ds_area = ds_area_reg
                ds_prec_off = ds_prec_off_reg
                ds_melt_off = ds_melt_off_reg
                ds_refr_off = ds_refr_off_reg
                ds_area_off = ds_area_off_reg
            else:
                df_all = pd.concat([df_all, df], axis=0)
                ds_runoff = np.concatenate((ds_runoff, ds_runoff_reg), axis=0)
                ds_runoff_std = np.concatenate((ds_runoff_std, ds_runoff_reg_std), axis=0)
                ds_runoff_onglac = np.concatenate((ds_runoff_onglac, ds_runoff_onglac_reg), axis=0)
                ds_runoff_offglac = np.concatenate((ds_runoff_offglac, ds_runoff_offglac_reg), axis=0)
                ds_vol = np.concatenate((ds_vol, ds_vol_reg), axis=0)
                ds_area = np.concatenate((ds_area, ds_area_reg), axis=0)
                ds_prec = np.concatenate((ds_prec, ds_prec_reg), axis=0)
                ds_melt = np.concatenate((ds_melt, ds_melt_reg), axis=0)
                ds_refr = np.concatenate((ds_refr, ds_refr_reg), axis=0)
                ds_area_off = np.concatenate((ds_area_off, ds_area_off_reg), axis=0)
                ds_prec_off = np.concatenate((ds_prec_off, ds_prec_off_reg), axis=0)
                ds_melt_off = np.concatenate((ds_melt_off, ds_melt_off_reg), axis=0)
                ds_refr_off = np.concatenate((ds_refr_off, ds_refr_off_reg), axis=0)


        # RUNOFF FROM COMPONENTS AND RELATIVE FRACTION OF EACH COMPONENT
        #  note: this significantly differs from runoff values due to the propagation of errors associated with
        #        the averaging of each of the components and the area, which in part result from using the mean values
        #        since this is biased towards higher values (see Figure Uncertainty in Projections Paper)
        ds_runoff2 = ds_prec + ds_melt - ds_refr + ds_prec_off + ds_melt_off - ds_refr_off    
        
        # ANNUAL RUNOFF AND COMPONENTS
        ds_runoff_annual = gcmbiasadj.annual_sum_2darray(ds_runoff)
        ds_runoff_onglac_annual = gcmbiasadj.annual_sum_2darray(ds_runoff_onglac)
        ds_runoff_offglac_annual = gcmbiasadj.annual_sum_2darray(ds_runoff_offglac)
        ds_runoff2_annual = gcmbiasadj.annual_sum_2darray(ds_runoff2)
        ds_prec_annual = gcmbiasadj.annual_sum_2darray(ds_prec)
        ds_melt_annual = gcmbiasadj.annual_sum_2darray(ds_melt)
        ds_refr_annual = gcmbiasadj.annual_sum_2darray(ds_refr)
        ds_prec_off_annual = gcmbiasadj.annual_sum_2darray(ds_prec_off)
        ds_melt_off_annual = gcmbiasadj.annual_sum_2darray(ds_melt_off)
        ds_refr_off_annual = gcmbiasadj.annual_sum_2darray(ds_refr_off)               
        # excess glacier meltwater based on volume changen
        ds_melt_excess_annual = excess_meltwater_m3(ds_vol) / 1e9
   
        # Remove RGIIds from main_glac_rgi that are not in the model runs
        if nrcp == 0:
            rgiid_df = list(df_all.RGIId.values)
            rgiid_all = list(main_glac_rgi.RGIId.values)
            rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
            main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
            main_glac_rgi.reset_index(inplace=True, drop=True)

        #%%
        # GROUP PROCESSING (RUNOFF AND UNCERTAINTY)
        df_runoff_group_annual = pd.DataFrame(np.zeros((len(groups), len(time_values_annual[:-1]))), 
                                              columns=time_values_annual[:-1], index=groups)
        df_runoff2_group_annual = pd.DataFrame(np.zeros((len(groups), len(time_values_annual[:-1]))), 
                                              columns=time_values_annual[:-1], index=groups)
        df_runoff_onglac_group_annual = pd.DataFrame(np.zeros((len(groups), len(time_values_annual[:-1]))), 
                                                      columns=time_values_annual[:-1], index=groups)
        df_runoff_offglac_group_annual = pd.DataFrame(np.zeros((len(groups), len(time_values_annual[:-1]))), 
                                                      columns=time_values_annual[:-1], index=groups)
        df_runoff_group_ref_monthly = pd.DataFrame(np.zeros((len(groups),12)), columns=months, index=groups)
        df_runoff_group_eoc_monthly = pd.DataFrame(np.zeros((len(groups),12)), columns=months, index=groups)
        df_runoff_group_eoc_monthly_norm = pd.DataFrame(np.zeros((len(groups),12)), columns=months, index=groups)
        df_runoff_group_eoc_monthly_norm_std = pd.DataFrame(np.zeros((len(groups),12)), columns=months, index=groups)
        for ngroup, group in enumerate(groups):
            # Select subset of data
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()

            # MONTHLY GROUP RUNOFF
            runoff_group = ds_runoff[group_glac_indices,:].sum(axis=0)
                
            # Uncertainty associated with volume change based on subgroups
            #  sum standard deviations in each subgroup assuming that they are uncorrelated
            #  then use the root sum of squares using the uncertainty of each subgroup to get the 
            #  uncertainty of the group
            main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
            subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()
            
            subgroup_std = np.zeros((len(subgroups_subset), ds_runoff.shape[1]))
            for nsubgroup, subgroup in enumerate(subgroups_subset):
                main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
                subgroup_indices = (
                        main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist())
                # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
                subgroup_std[nsubgroup,:] = ds_runoff_std[subgroup_indices,:].sum(axis=0)
            runoff_group_std = (subgroup_std**2).sum(axis=0)**0.5    
            
            # ANNUAL GROUP RUNOFF
            df_runoff_group_annual.loc[group,:] = ds_runoff_annual[group_glac_indices,:].sum(axis=0)
            df_runoff2_group_annual.loc[group,:] = ds_runoff2_annual[group_glac_indices,:].sum(axis=0)
            # Total (glacier and off-glacier)
            df_runoff_onglac_group_annual.loc[group,:] = ds_runoff_onglac_annual[group_glac_indices,:].sum(axis=0)
            df_runoff_offglac_group_annual.loc[group,:] = ds_runoff_offglac_annual[group_glac_indices,:].sum(axis=0)
        
            # REFERENCE AND END OF CENTURY MONTHLY RUNOFF
            def monthly_mean(x, month_idx1, month_idx2):
                x_subset = x[month_idx1:month_idx2]
                monthly_mean = np.zeros((12))
                for nmonth in np.arange(0,12):
                    monthly_mean[nmonth] = x_subset[nmonth::12].mean()
                return monthly_mean
            
            runoff_group_ref_monthly = monthly_mean(runoff_group, refmonth_idx1, refmonth_idx2)
            runoff_group_ref_monthly_std = monthly_mean(runoff_group_std, refmonth_idx1, refmonth_idx2)
            runoff_group_eoc_monthly = monthly_mean(runoff_group, eocmonth_idx1, eocmonth_idx2)
            runoff_group_eoc_monthly_std = monthly_mean(runoff_group_std, eocmonth_idx1, eocmonth_idx2)
            df_runoff_group_ref_monthly.loc[group,:] = runoff_group_ref_monthly
            df_runoff_group_eoc_monthly.loc[group,:] = runoff_group_eoc_monthly
                                
            # NORMALIZED CHANGE IN RUNOFF BY END OF CENTURY RELATIVE TO REFERENCE PERIOD
            runoff_group_eoc_monthly_norm = ((runoff_group_eoc_monthly - runoff_group_ref_monthly) / 
                                             runoff_group_ref_monthly * 100)
            runoff_group_eoc_monthly_norm_std = runoff_group_eoc_monthly_std / runoff_group_ref_monthly * 100            
            df_runoff_group_eoc_monthly_norm.loc[group,:] = runoff_group_eoc_monthly_norm
            df_runoff_group_eoc_monthly_norm_std.loc[group,:] = runoff_group_eoc_monthly_norm_std

        # ===== EXPORT RUNOFF CHANGES FOR MAY - OCTOBER =====
        output_df = df_runoff_group_eoc_monthly_norm.copy()
        output_df_std = df_runoff_group_eoc_monthly_norm_std.copy()
        # Replace nan and infinity with 0
        output_df.replace({np.nan:0, np.inf:0},inplace=True)
        output_df_std.replace({np.nan:0, np.inf:0},inplace=True)
        for nrow in output_df.index.values:
            for ncol in output_df.columns.values:
                xmean = output_df.loc[nrow,ncol]
                xstd = output_df_std.loc[nrow,ncol]
                if xmean >= 0:
                    output_df.loc[nrow,ncol] = ('+' + str(int(np.round(xmean,0))) + u'\u00B1' + 
                                                str(int(np.round(xstd,0))) + '%')
                else:
                    output_df.loc[nrow,ncol] = (str(int(np.round(xmean,0))) + u'\u00B1' + 
                                                str(int(np.round(xstd,0))) + '%')
        output_df.index = [title_dict[group] for group in groups]
        
        cns_ordered = [1,2,3,4,5,6,7,8,9,10,11,12]
        
        output_df = output_df[cns_ordered]

        output_fn = grouping + '_monthly_chg_' + rcp + '.csv'
        output_df.to_csv(figure_fp + output_fn)
        
        #%%
        # ===== EXPORT PEAK WATER STATISTICS =====
        # Peakwater
        stats_cns = ['group', 'rcp', 'runoff_Gtyr_ref', 'peakwater_yr', 'peakwater_chg_perc', '2100_chg_perc']
        output_dfpw = pd.DataFrame(np.zeros((len(groups),len(stats_cns))), columns=stats_cns)
        output_dfpw['group'] = groups
        output_dfpw['rcp'] = rcp
        print('Peakwater by group for', rcp)
        nyears = 11
        group_peakwater = {}
        pw_idx1 = np.where(time_values_annual == peakwater_startyear)[0][0]
        pw_idx2 = np.where(time_values_annual == peakwater_endyear)[0][0]+1
        for ngroup, group in enumerate(groups):
            group_peakwater[group] = peakwater(df_runoff_group_annual.loc[group,:].values[pw_idx1:pw_idx2], 
                                               time_values_annual[pw_idx1:pw_idx2], nyears)
            print(group, group_peakwater[group][0], '\n  peakwater_chg[%]:', np.round(group_peakwater[group][1],0),
                  '\n  2100 chg[%]:', np.round(group_peakwater[group][2],0))
            output_dfpw.loc[ngroup,'runoff_Gtyr_ref'] = (
                    df_runoff_group_annual.loc[group,:].values[refyear_idx1:refyear_idx2].mean())
            output_dfpw.loc[ngroup,'peakwater_yr'] = group_peakwater[group][0]
            output_dfpw.loc[ngroup,'peakwater_chg_perc'] = group_peakwater[group][1]
            output_dfpw.loc[ngroup,'2100_chg_perc'] = group_peakwater[group][2]
        
        if grouping == 'watershed':
            # Add Aral Sea (Amu Darya + Syr Darya) for comparison with HH2019
            group = 'Aral_Sea'
            group_peakwater['Aral_Sea'] = peakwater(df_runoff_group_annual.loc['Amu_Darya',:].values[pw_idx1:pw_idx2] + 
                                                    df_runoff_group_annual.loc['Syr_Darya',:].values[pw_idx1:pw_idx2], 
                                                    time_values_annual[pw_idx1:pw_idx2], nyears)
            print(group, group_peakwater[group][0], '\n  peakwater_chg[%]:', np.round(group_peakwater[group][1],0),
                  '\n  2100 chg[%]:', np.round(group_peakwater[group][2],0))
            output_dfpw2 = pd.DataFrame(np.zeros((1,len(stats_cns))), columns=stats_cns)
            output_dfpw2.loc[0,'group'] = group
            output_dfpw2.loc[0,'rcp'] = rcp
            output_dfpw2.loc[0,'runoff_Gtyr_ref'] = (
                    (df_runoff_group_annual.loc['Amu_Darya',:].values + 
                     df_runoff_group_annual.loc['Syr_Darya',:].values)[refyear_idx1:refyear_idx2].mean())
            output_dfpw2.loc[0,'peakwater_yr'] = group_peakwater[group][0]
            output_dfpw2.loc[0,'peakwater_chg_perc'] = group_peakwater[group][1]
            output_dfpw2.loc[0,'2100_chg_perc'] = group_peakwater[group][2]
            output_dfpw = pd.concat([output_dfpw, output_dfpw2], axis=0)
            output_dfpw.reset_index(inplace=True, drop=True)
            
        output_dfpw.to_csv(figure_fp + grouping + '_peakwater_stats_' + rcp + '.csv', index=False)
        
        #%%
        groups2plot = groups.copy()
        if grouping == 'watershed':
            groups2plot.remove('Irrawaddy')
            groups2plot.remove('Yellow')
            
        t1_idx = np.where(time_values_annual == plt_startyear)[0][0]
        t2_idx = np.where(time_values_annual == plt_endyear)[0][0] + 1
        
        multimodel_linewidth = 2
        alpha=0.2
    
        reg_legend = []
        num_cols_max = 4
        if len(groups) < num_cols_max:
            num_cols = len(groups2plot)
        else:
            num_cols = num_cols_max
        num_rows = int(np.ceil(len(groups2plot)/num_cols))
            
        fig, ax = plt.subplots(num_rows, num_cols, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        add_group_label = 1
        
        # Cycle through groups  
        row_idx = 0
        col_idx = 0

        for ngroup, group in enumerate(groups2plot):
            # Set subplot position
            if (ngroup % num_cols == 0) and (ngroup != 0):
                row_idx += 1
                col_idx = 0
                
            # COMPONENTS OF ANNUAL RUNOFF ADJUSTED            
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
            group_annual_runoff2 = df_runoff2_group_annual.loc[group,:].values
            group_annual_prec = ds_prec_annual[group_glac_indices,:].sum(axis=0)
            group_annual_melt = ds_melt_annual[group_glac_indices,:].sum(axis=0)
            group_annual_melt_excess = ds_melt_excess_annual[group_glac_indices,:].sum(axis=0)
            group_annual_refr = ds_refr_annual[group_glac_indices,:].sum(axis=0)
            group_annual_prec_off = ds_prec_off_annual[group_glac_indices,:].sum(axis=0)
            group_annual_melt_off = ds_melt_off_annual[group_glac_indices,:].sum(axis=0)
            group_annual_refr_off = ds_refr_off_annual[group_glac_indices,:].sum(axis=0)
            
            # Runoff datasets (not from components)
            group_annual_runoff = df_runoff_group_annual.loc[group,:].values
            
            # Fraction of each component
            group_annual_prec_frac = group_annual_prec / group_annual_runoff2
            group_annual_melt_frac = group_annual_melt / group_annual_runoff2
            group_annual_melt_excess_frac = group_annual_melt_excess / group_annual_runoff2
            group_annual_refr_frac = group_annual_refr / group_annual_runoff2
            group_annual_prec_off_frac = group_annual_prec_off / group_annual_runoff2
            group_annual_melt_off_frac = group_annual_melt_off / group_annual_runoff2
            group_annual_refr_off_frac = group_annual_refr_off / group_annual_runoff2

            component_check = (group_annual_prec_frac + group_annual_melt_frac - group_annual_refr_frac + 
                               group_annual_prec_off_frac + group_annual_melt_off_frac - 
                               group_annual_refr_off_frac)
            
            # Each component adjusted
            group_annual_prec_adj = group_annual_prec_frac * group_annual_runoff
            group_annual_melt_adj = group_annual_melt_frac * group_annual_runoff
            group_annual_melt_excess_adj = group_annual_melt_excess_frac * group_annual_runoff
            group_annual_refr_adj = group_annual_refr_frac * group_annual_runoff
            group_annual_prec_off_adj = group_annual_prec_off_frac * group_annual_runoff
            group_annual_melt_off_adj = group_annual_melt_off_frac * group_annual_runoff
            group_annual_refr_off_adj = group_annual_refr_off_frac * group_annual_runoff
            
            
            # Normalize values for plot
            runoff_total_normalizer = group_annual_runoff[refyear_idx1:refyear_idx2].mean()
            
            runoff_total_norm = group_annual_runoff / runoff_total_normalizer
            runoff_glac_total_norm = df_runoff_onglac_group_annual.loc[group,:].values / runoff_total_normalizer 
            runoff_glac_prec_norm = group_annual_prec_adj / runoff_total_normalizer
            runoff_glac_melt_norm = group_annual_melt_adj / runoff_total_normalizer
            runoff_glac_excess_norm = group_annual_melt_excess_adj / runoff_total_normalizer
            runoff_glac_refreeze_norm = group_annual_refr_adj / runoff_total_normalizer
            runoff_offglac_prec_norm = group_annual_prec_off_adj / runoff_total_normalizer
            runoff_offglac_melt_norm = group_annual_melt_off_adj / runoff_total_normalizer
            runoff_offglac_refreeze_norm = group_annual_refr_off_adj / runoff_total_normalizer

            # Plot
            # Total runoff (line)
            ax[row_idx, col_idx].plot(time_values_annual[t1_idx:t2_idx], runoff_total_norm[t1_idx:t2_idx], 
                                      color='k', linewidth=1, zorder=4)
            # Glacier runoff (dotted line)
            ax[row_idx, col_idx].plot(time_values_annual[t1_idx:t2_idx], runoff_glac_total_norm[t1_idx:t2_idx], 
                                      color='k', linewidth=1, linestyle='--', zorder=3)
            
            # Components
            component_alpha = 0.5
            # Glacier melt on bottom (green fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    0, 
                    runoff_glac_melt_norm[t1_idx:t2_idx] - runoff_glac_excess_norm[t1_idx:t2_idx],
#                    facecolor='green', alpha=0.2, label='glac melt', zorder=3)
                    facecolor='maroon', alpha=component_alpha, label='glac melt', zorder=3)
            # Excess glacier melt (green fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx] - runoff_glac_excess_norm[t1_idx:t2_idx],
#                    facecolor='darkgreen', alpha=0.4, label='glac excess', zorder=3)
                    facecolor='orangered', alpha=component_alpha, label='glac excess', zorder=3)
            # Off-Glacier melt (blue fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx],
                    runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx],
                    facecolor='orange', alpha=component_alpha, label='offglac melt', zorder=3)
            # Glacier refreeze (grey fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx],
                    0,
                    runoff_glac_refreeze_norm[t1_idx:t2_idx],
                    facecolor='grey', alpha=component_alpha, label='glac refreeze', hatch='////', zorder=4)
            # Glacier precipitation (yellow fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx],
                    (runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx] + 
                     runoff_glac_prec_norm[t1_idx:t2_idx]),
#                    facecolor='yellow', alpha=0.2, label='glacier prec', zorder=3)
                    facecolor='mediumblue', alpha=component_alpha, label='glacier prec', zorder=3)
            # Off-glacier precipitation (red fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    (runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx] + 
                     runoff_glac_prec_norm[t1_idx:t2_idx]),
                    (runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx] + 
                     runoff_glac_prec_norm[t1_idx:t2_idx] + runoff_offglac_prec_norm[t1_idx:t2_idx]),
#                    facecolor='red', alpha=0.2, label='offglac prec', zorder=3)
                    facecolor='lightseagreen', alpha=component_alpha, label='offglac prec', zorder=3)
#            # Off-glacier refreeze (grey fill)
#            ax[row_idx, col_idx].fill_between(
#                    time_values_annual[t1_idx:t2_idx],
#                    runoff_glac_melt_norm[t1_idx:t2_idx],
#                    runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_refreeze_norm[t1_idx:t2_idx],
#                    facecolor='grey', alpha=0.2, label='offglac refreeze', hatch='....', zorder=4)
            
            # Group labels
            if add_group_label == 1:
                ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=10, horizontalalignment='center', 
                                          verticalalignment='top', transform=ax[row_idx, col_idx].transAxes)
    
            # X-label
            ax[row_idx, col_idx].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                          time_values_annual[t1_idx:t2_idx].max())
            ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=12)
            ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(50))
            ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
            if col_idx == 0 and row_idx == num_rows-1:
                ax[row_idx, col_idx].set_xticklabels(['2015','2050','2100'])
            elif row_idx == num_rows-1:
                ax[row_idx, col_idx].set_xticklabels(['','2050','2100'])
            else:
                ax[row_idx, col_idx].set_xticklabels(['','',''])
                
            # Y-label
            if rcp in ['rcp85']:
                ax[row_idx, col_idx].set_ylim(0,2.3)
                ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5', '2.0'])
            else:
                ax[row_idx, col_idx].set_ylim(0,2)
                ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5', ''])
            ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                

            # Tick parameters
            ax[row_idx, col_idx].yaxis.set_ticks_position('both')
            ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=12, direction='inout')
            ax[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=12, direction='inout')            
            
            # Add value to subplot
#            plot_str = '(' + str(int(np.round(runoff_total_normalizer,0))) + ' Gt $\mathregular{yr^{-1}}$)'
#            plot_str = '(' + str(np.round(runoff_total_normalizer,1)) + ' Gt $\mathregular{yr^{-1}}$)
            plot_str = str(np.round(runoff_total_normalizer,1))
            
            ax[row_idx, col_idx].text(0.5, 0.9, plot_str, size=10, horizontalalignment='center', 
                                      verticalalignment='top', transform=ax[row_idx, col_idx].transAxes, 
                                      color='k', zorder=5)
            # Count column index to plot
            col_idx += 1

        # Line legend
        leg_alpha = component_alpha
        leg_list = ['Fixed-gauge\nglacier runoff', 'Moving-gauge\nglacier runoff',
                    'Off-glacier\nprecipitation', 'Glacier\nprecipitation', 'Off-glacier\nmelt', 
#                    'Off-glacier\nrefreeze', 
                    'Glacier melt\n(excess)', 'Glacier melt\n(equilibrium)', 
                    'Glacier\nrefreeze']
        line_dict = {'Fixed-gauge\nglacier runoff':['black',1,'-',1,''], 
                     'Moving-gauge\nglacier runoff':['black',1,'--',1,''],
                     'Glacier melt\n(equilibrium)':['maroon',5,'-',leg_alpha,''], 
                     'Glacier melt\n(excess)':['orangered',5,'-',0.4,''], 
                     'Glacier\nprecipitation':['mediumblue',5,'-',leg_alpha,''],
                     'Glacier\nrefreeze':['grey',5,'-',leg_alpha,'////'],
                     'Off-glacier\nmelt':['orange',5,'-',leg_alpha,''], 
                     'Off-glacier\nprecipitation':['lightseagreen',5,'-',leg_alpha,''],
                     'Off-glacier\nrefreeze':['grey',5,'-',leg_alpha,'....']}
        leg_lines = []
        leg_labels = []
        for vn_label in leg_list:
            if 'refreeze' in vn_label:
                line = mpatches.Patch(facecolor=line_dict[vn_label][0], alpha=line_dict[vn_label][3], 
                                      hatch=line_dict[vn_label][4])
            else:
                line = Line2D([0,1],[0,1], color=line_dict[vn_label][0], linewidth=line_dict[vn_label][1], 
                              linestyle=line_dict[vn_label][2], alpha=line_dict[vn_label][3])
            leg_lines.append(line)
            leg_labels.append(vn_label)
        fig.subplots_adjust(right=0.83)
        fig.legend(leg_lines, leg_labels, loc=(0.83,0.30), fontsize=10, labelspacing=0.5, handlelength=1, ncol=1,
                   handletextpad=0.5, borderpad=0, frameon=False)

        # Label
        ylabel_str = 'Runoff (-)'
        fig.text(0.03, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
        
        # Save figure
        if len(groups) == 1:
            fig.set_size_inches(4, 4)
        else:
            fig.set_size_inches(7, num_rows*2)
        
        figure_fn = grouping + '_runoffcomponents_mulitmodel_' + rcp +  '.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
        #%%
        # ====== PLOT OF NORMALIZED CHANGE AND COMPONENTS =====
        groups2plot = groups.copy()
        if grouping == 'watershed':
            groups2plot.remove('Irrawaddy')
            groups2plot.remove('Yellow')
        
        multimodel_linewidth = 2
        alpha=0.5
    
        reg_legend = []
        num_cols_max = 4
        if len(groups) < num_cols_max:
            num_cols = len(groups2plot)
        else:
            num_cols = num_cols_max
        num_rows = int(np.ceil(len(groups2plot)/num_cols))
            
        fig, ax = plt.subplots(num_rows, num_cols, squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        add_group_label = 1
        
        # Cycle through groups  
        row_idx = 0
        col_idx = 0
        
        def shift_list(l,n):
            return l[n:] + l[:n]
        
        def norm_shift(values, norm_value, nshift):
            return np.array(shift_list(list(values / norm_value), nshift))
        n_shift = 3
        months_plot = shift_list(months, n_shift)
        
        for ngroup, group in enumerate(groups2plot):
            # COMPONENTS OF END OF CENTURY MONTHLY RUNOF
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
            group_runoff2 = ds_runoff2[group_glac_indices,:].sum(axis=0)
            group_prec = ds_prec[group_glac_indices,:].sum(axis=0)
            group_melt = ds_melt[group_glac_indices,:].sum(axis=0)
            group_refr = ds_refr[group_glac_indices,:].sum(axis=0)
            group_prec_off = ds_prec_off[group_glac_indices,:].sum(axis=0)
            group_melt_off = ds_melt_off[group_glac_indices,:].sum(axis=0)
            group_refr_off = ds_refr_off[group_glac_indices,:].sum(axis=0)
                        
            group_eoc_monthly_runoff2 = monthly_mean(group_runoff2, eocmonth_idx1, eocmonth_idx2)
            group_eoc_monthly_prec = monthly_mean(group_prec, eocmonth_idx1, eocmonth_idx2)
            group_eoc_monthly_melt = monthly_mean(group_melt, eocmonth_idx1, eocmonth_idx2)
            group_eoc_monthly_refr = monthly_mean(group_refr, eocmonth_idx1, eocmonth_idx2)
            group_eoc_monthly_prec_off = monthly_mean(group_prec_off, eocmonth_idx1, eocmonth_idx2)
            group_eoc_monthly_melt_off = monthly_mean(group_melt_off, eocmonth_idx1, eocmonth_idx2)
            group_eoc_monthly_refr_off = monthly_mean(group_refr_off, eocmonth_idx1, eocmonth_idx2)
            
            group_runoff2_check = (group_eoc_monthly_prec + group_eoc_monthly_melt - group_eoc_monthly_refr +
                                   group_eoc_monthly_prec_off + group_eoc_monthly_melt_off - group_eoc_monthly_refr_off)
            
            # Runoff datasets (not from components)
            runoff_group_eoc_monthly = df_runoff_group_eoc_monthly.loc[group,:].values
            runoff_group_ref_monthly = df_runoff_group_ref_monthly.loc[group,:].values
            
            runoff_dif = group_eoc_monthly_runoff2 - runoff_group_eoc_monthly
            runoff_dif_norm = np.round(runoff_dif / runoff_group_eoc_monthly * 100,1)
            
            print('\n', group, '\n', runoff_dif_norm)
            
            # Fraction of each component
            group_eoc_monthly_prec_frac = group_eoc_monthly_prec / group_eoc_monthly_runoff2
            group_eoc_monthly_melt_frac = group_eoc_monthly_melt / group_eoc_monthly_runoff2
            group_eoc_monthly_refr_frac = group_eoc_monthly_refr / group_eoc_monthly_runoff2
            group_eoc_monthly_prec_off_frac = group_eoc_monthly_prec_off / group_eoc_monthly_runoff2
            group_eoc_monthly_melt_off_frac = group_eoc_monthly_melt_off / group_eoc_monthly_runoff2
            group_eoc_monthly_refr_off_frac = group_eoc_monthly_refr_off / group_eoc_monthly_runoff2

            component_check = (group_eoc_monthly_prec_frac + group_eoc_monthly_melt_frac - group_eoc_monthly_refr_frac + 
                               group_eoc_monthly_prec_off_frac + group_eoc_monthly_melt_off_frac - 
                               group_eoc_monthly_refr_off_frac)
            
            # Each component adjusted
            group_eoc_monthly_prec_adj = group_eoc_monthly_prec_frac * runoff_group_eoc_monthly
            group_eoc_monthly_melt_adj = group_eoc_monthly_melt_frac * runoff_group_eoc_monthly
            group_eoc_monthly_refr_adj = group_eoc_monthly_refr_frac * runoff_group_eoc_monthly
            group_eoc_monthly_prec_off_adj = group_eoc_monthly_prec_off_frac * runoff_group_eoc_monthly
            group_eoc_monthly_melt_off_adj = group_eoc_monthly_melt_off_frac * runoff_group_eoc_monthly
            group_eoc_monthly_refr_off_adj = group_eoc_monthly_refr_off_frac * runoff_group_eoc_monthly
            
            
            # PLOT DETAILS
            # Set subplot position
            if (ngroup % num_cols == 0) and (ngroup != 0):
                row_idx += 1
                col_idx = 0

            # Normalize and shift values for plot            
            runoff_normalizer = runoff_group_ref_monthly.max()
            
            month_runoff_total_ref_plot = norm_shift(runoff_group_ref_monthly, runoff_normalizer, n_shift)
            month_runoff_total_plot = norm_shift(runoff_group_eoc_monthly, runoff_normalizer, n_shift)
            month_glac_prec_plot = norm_shift(group_eoc_monthly_prec_adj, runoff_normalizer, n_shift)
            month_glac_melt_plot = norm_shift(group_eoc_monthly_melt_adj, runoff_normalizer, n_shift)
            month_glac_refreeze_plot = norm_shift(group_eoc_monthly_refr_adj, runoff_normalizer, n_shift)
            month_offglac_prec_plot = norm_shift(group_eoc_monthly_prec_off_adj, runoff_normalizer, n_shift)
            month_offglac_melt_plot = norm_shift(group_eoc_monthly_melt_off_adj, runoff_normalizer, n_shift)
            month_offglac_refreeze_plot = norm_shift(group_eoc_monthly_refr_off_adj, runoff_normalizer, n_shift)
            
            ax[row_idx, col_idx].plot(months_plot, month_runoff_total_ref_plot, color='k', linewidth=1, linestyle='-',
                                      zorder=4)
            ax[row_idx, col_idx].plot(months_plot, month_runoff_total_plot, color='k', linewidth=1, linestyle='--',
                                      zorder=4)
            
            # Components
            # Glacier melt on bottom (green fill)
            ax[row_idx, col_idx].fill_between(months_plot, 0, 
                                              month_glac_melt_plot,
                                              facecolor='darkred', alpha=alpha, label='glac melt', zorder=3)
            # Off-Glacier melt (blue fill)
            ax[row_idx, col_idx].fill_between(months_plot, month_glac_melt_plot, 
                                              month_glac_melt_plot + month_offglac_melt_plot,
                                              facecolor='orange', alpha=alpha, label='offglac melt', zorder=3)
            # Glacier refreeze (grey fill)
            ax[row_idx, col_idx].fill_between(months_plot, 0, month_glac_refreeze_plot,
                                              facecolor='grey', alpha=alpha, label='glac refreeze', hatch='////', 
                                              zorder=4)
            # Glacier precipitation (yellow fill)
            ax[row_idx, col_idx].fill_between(months_plot, month_glac_melt_plot + month_offglac_melt_plot,
                                              month_glac_melt_plot + month_offglac_melt_plot + month_glac_prec_plot,
                                              facecolor='mediumblue', alpha=alpha, label='glacier prec', zorder=3)
            # Off-glacier precipitation (red fill)
            ax[row_idx, col_idx].fill_between(months_plot,
                                              month_glac_melt_plot + month_offglac_melt_plot + month_glac_prec_plot,
                                              (month_glac_melt_plot + month_offglac_melt_plot + month_glac_prec_plot
                                               + month_offglac_prec_plot),
                                               facecolor='lightseagreen', alpha=alpha, label='offglac prec', zorder=3)         
#            # Off-glacier refreeze (grey fill)
#            ax[row_idx, col_idx].fill_between(months, month_glac_melt_plot,
#                                              month_glac_melt_plot + month_offglac_refreeze_plot,
#                                              facecolor='grey', alpha=alpha, label='glac refreeze', hatch='....', 
#                                              zorder=4)
            # Group labels
            if add_group_label == 1:
                ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=12, horizontalalignment='center', 
                                          verticalalignment='top', transform=ax[row_idx, col_idx].transAxes)
    
            # X-label
            ax[row_idx, col_idx].set_xlim(3.5,10.5)
            ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=12)
            ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(1))
            ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(1))
            if row_idx == num_rows-1:
                ax[row_idx, col_idx].set_xticklabels(['','4','5','6','7','8','9','10'])
            else:
                ax[row_idx, col_idx].set_xticklabels(['','','','','','','',''])
                
            # Y-label
            if rcp in ['rcp26', 'rcp45']:
                ax[row_idx, col_idx].set_ylim(0,1.5)
            else:
                ax[row_idx, col_idx].set_ylim(0,1.8)
            ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5', ''])

            # Tick parameters
            ax[row_idx, col_idx].yaxis.set_ticks_position('both')
            ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=12, direction='inout')
            ax[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=12, direction='inout')            
            
            # Add value to subplot
            month_chg = ([np.round((month_runoff_total_plot[i] - month_runoff_total_ref_plot[i]) / 
                                   month_runoff_total_ref_plot[i] * 100,0) for i in range(0,12)])
            month_chg_subset = month_chg[5:9]
            maxchg_subset = np.nanmin(month_chg_subset)
            print(group, month_chg[5:9])
            month_dict = {5:'May', 6:'June', 7:'July', 8:'August', 9:'September'}
            if maxchg_subset < 0:
                txtcolor='k'
                month_of_maxchg = np.where(maxchg_subset == month_chg)[0][0] + 1
                plot_str = '(' + str(int(np.round(maxchg_subset,0))) + '%, ' + month_dict[month_of_maxchg] + ')'
            else:
                txtcolor='k'
                maxchg_subset = np.nanmax(month_chg_subset)
                month_of_maxchg = np.where(maxchg_subset == month_chg)[0][0] + 1
                plot_str = '(+' + str(int(np.round(maxchg_subset,0))) + '%, ' + month_dict[month_of_maxchg] + ')'
#            plot_str = '(' + str(int(np.round(maxchg,0))) + ' %)'

            ax[row_idx, col_idx].text(0.5, 0.88, plot_str, size=8, horizontalalignment='center', 
                                      verticalalignment='top', transform=ax[row_idx, col_idx].transAxes, 
                                      color=txtcolor, zorder=5)
            # Count column index to plot
            col_idx += 1

        # Line legend
        leg_alpha = alpha
        leg_list = ['Glacier runoff\n2000-2015', 'Glacier runoff\n2085-2100',
                    'Off-glacier\nprecipitation', 'Glacier\nprecipitation', 'Off-glacier\nmelt', 
#                    'Off-glacier\nrefreeze', 
                    'Glacier melt', 
                    'Glacier\nrefreeze']
        line_dict = {'Glacier runoff\n2000-2015':['black',1,'-',1,''], 
                     'Glacier runoff\n2085-2100':['black',1,'--',1,''],
                     'Glacier melt':['darkred',5,'-',leg_alpha,''],
                     'Glacier\nprecipitation':['mediumblue',5,'-',leg_alpha,''],
                     'Glacier\nrefreeze':['grey',5,'-',leg_alpha,'////'],
                     'Off-glacier\nmelt':['orange',5,'-',leg_alpha,''], 
                     'Off-glacier\nprecipitation':['lightseagreen',5,'-',leg_alpha,''],
                     'Off-glacier\nrefreeze':['grey',5,'-',leg_alpha,'....']}
        leg_lines = []
        leg_labels = []
        for vn_label in leg_list:
            if 'refreeze' in vn_label:
                line = mpatches.Patch(facecolor=line_dict[vn_label][0], alpha=line_dict[vn_label][3], 
                                      hatch=line_dict[vn_label][4])
            else:
                line = Line2D([0,1],[0,1], color=line_dict[vn_label][0], linewidth=line_dict[vn_label][1], 
                              linestyle=line_dict[vn_label][2], alpha=line_dict[vn_label][3])
            leg_lines.append(line)
            leg_labels.append(vn_label)
        fig.subplots_adjust(right=0.83)
        fig.legend(leg_lines, leg_labels, loc=(0.83,0.38), fontsize=10, labelspacing=0.5, handlelength=1, ncol=1,
                   handletextpad=0.5, borderpad=0, frameon=False)

        # Label
        ylabel_str = 'Runoff (-)'
        xlabel_str = 'Month'
        fig.text(0.03, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
        fig.text(0.5, 0.05, xlabel_str, va='bottom', ha='center', size=12)
        
        # Save figure
        if len(groups) == 1:
            fig.set_size_inches(4, 4)
        else:
            fig.set_size_inches(7, num_rows*2)
        
        figure_fn = grouping + '_runoffmonthlychange_mulitmodel_' + rcp +  '.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)


#%%
if option_peakwater_map == 1:
    figure_fp = netcdf_fp_cmip5 + 'figures/'
    if os.path.exists(figure_fp) == False:
        os.mkdir(figure_fp)
    
    rcps = ['rcp26', 'rcp45', 'rcp85']
    
    option_plot4paper_3rcps = 1
    
    startyear = 2015
    endyear = 2100
    
    vn = 'runoff_glac_monthly'
    grouping = 'watershed'
    peakwater_Nyears = 11
    
#    east = 60
#    west = 110
#    south = 15
#    north = 50
    east = 104
    west = 64
    south = 26
    north = 47
    xtick = 5
    ytick = 5
    xlabel = 'Longitude ($\mathregular{^{\circ}}$)'
    ylabel = 'Latitude ($\mathregular{^{\circ}}$)'
    
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    groups_deg, group_cn_deg = select_groups('degree', main_glac_rgi)
    deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
    deg_dict = dict(zip(np.arange(0,len(deg_groups)), deg_groups))

    #%%
    # Glacier and grouped annual runoff
    ds_multimodel, ds_multimodel_std = {}, {}
    ds_multimodel_deg, ds_multimodel_std_deg = {}, {}
    for rcp in rcps:
        
        ds_multimodel[rcp], ds_multimodel_std[rcp] = {}, {}
        ds_multimodel_deg[rcp], ds_multimodel_std_deg[rcp] = {}, {}
        for region in regions:
            
            # Load datasets
            ds_fn = 'R' + str(region) + '_multimodel_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
            ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
            df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
            df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                           str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
            
            # Extract time variable
            time_values_annual = ds.coords['year_plus1'].values
            time_values_monthly = ds.coords['time'].values
            
            vn_glac_region = ds[vn].values[:,:,0]
            vn_glac_std_region = ds[vn].values[:,:,1]
            
            # Add off-glacier and convert monthly values to annual
            if vn == 'runoff_glac_monthly':
                vn_offglac_region = ds['offglac_runoff_monthly'].values[:,:,0]
                vn_offglac_std_region = ds['offglac_runoff_monthly'].values[:,:,1]                                
                vn_glac_region += vn_offglac_region
                vn_glac_std_region += vn_offglac_std_region                            
                
                vn_glac_region = gcmbiasadj.annual_sum_2darray(vn_glac_region)
                time_values_annual = time_values_annual[:-1]                    
                vn_glac_std_region = gcmbiasadj.annual_sum_2darray(vn_glac_std_region)
                
            # Merge datasets
            if region == regions[0]:
                vn_glac_all = vn_glac_region
                vn_glac_std_all = vn_glac_std_region
                df_all = df
            else:
                vn_glac_all = np.concatenate((vn_glac_all, vn_glac_region), axis=0)
                vn_glac_std_all = np.concatenate((vn_glac_std_all, vn_glac_std_region), axis=0)
                df_all = pd.concat([df_all, df], axis=0)
            
            ds.close()
            
        # Remove RGIIds from main_glac_rgi that are not in the model runs
        rgiid_df = list(df_all.RGIId.values)
        rgiid_all = list(main_glac_rgi.RGIId.values)
        rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
        main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
        main_glac_rgi.reset_index(inplace=True, drop=True)

            
        for ngroup, group in enumerate(groups):
            # Select subset of data
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
            vn_glac = vn_glac_all[group_glac_indices,:]
            
            subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)

            # Sum volume change for group
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
            vn_group = vn_glac_all[group_glac_indices,:].sum(axis=0)
            
            # Uncertainty associated with volume change based on subgroups
            #  sum standard deviations in each subgroup assuming that they are uncorrelated
            #  then use the root sum of squares using the uncertainty of each subgroup to get the 
            #  uncertainty of the group
            main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
            subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()

            subgroup_std = np.zeros((len(subgroups_subset), vn_group.shape[0]))
            for nsubgroup, subgroup in enumerate(subgroups_subset):
                main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
                subgroup_indices = (
                        main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist())
                # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
                subgroup_std[nsubgroup,:] = vn_glac_std_all[subgroup_indices,:].sum(axis=0)
            vn_group_std = (subgroup_std**2).sum(axis=0)**0.5        
            
            ds_multimodel[rcp][group] = vn_group
            ds_multimodel_std[rcp][group] = vn_group_std
          
        for ngroup, group in enumerate(groups_deg):
            # Select subset of data
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
            vn_glac = vn_glac_all[group_glac_indices,:]
            
            subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)

            # Sum volume change for group
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi['deg_id'] == group].index.values.tolist()
            vn_group = vn_glac_all[group_glac_indices,:].sum(axis=0)
            
            # Uncertainty associated with volume change based on subgroups
            #  sum standard deviations in each subgroup assuming that they are uncorrelated
            #  then use the root sum of squares using the uncertainty of each subgroup to get the 
            #  uncertainty of the group
            main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi['deg_id'] == group]
            subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()

            subgroup_std = np.zeros((len(subgroups_subset), vn_group.shape[0]))
            for nsubgroup, subgroup in enumerate(subgroups_subset):
                main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
                subgroup_indices = (
                        main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist())
                # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
                subgroup_std[nsubgroup,:] = vn_glac_std_all[subgroup_indices,:].sum(axis=0)
            vn_group_std = (subgroup_std**2).sum(axis=0)**0.5        
            
            ds_multimodel_deg[rcp][group] = vn_group
            ds_multimodel_std_deg[rcp][group] = vn_group_std


    time_idx_start = np.where(time_values_annual == startyear)[0][0]
    time_idx_end = np.where(time_values_annual == endyear)[0][0] + 1
            
    for rcp in rcps:
        for group in groups:
            
            vn_multimodel_mean = ds_multimodel[rcp][group]
            peakwater_yr, peakwater_chg, runoff_chg = (
                    peakwater(vn_multimodel_mean[time_idx_start:time_idx_end], 
                              time_values_annual[time_idx_start:time_idx_end], peakwater_Nyears))
            
            print(rcp, group, peakwater_yr, np.round(peakwater_chg,0), np.round(runoff_chg,0)) 
#%%
        # Create the projection
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':cartopy.crs.PlateCarree()},
                               gridspec_kw = {'wspace':0, 'hspace':0})
        
        # Add group and attribute of interest
        if grouping == 'rgi_region':
            group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
            group_shp_attr = 'RGI_CODE'
        elif grouping == 'watershed':
            group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
            group_shp_attr = 'watershed'
        elif grouping == 'kaab':
            group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
            group_shp_attr = 'Name'
            
        title_location = {'Syr_Darya': [71, 42],
                          'Ili': [82, 44.5],
                          'Amu_Darya': [69, 36],
                          'Tarim': [82.5, 38.5],
                          'Inner_Tibetan_Plateau_extended': [98.5, 38.2],
                          'Indus': [72, 32],
                          'Inner_Tibetan_Plateau': [86.5, 33],
                          'Yangtze': [100.7, 30.5],
                          'Ganges': [81.3, 26.6],
                          'Brahmaputra': [92.5, 26.5],
                          'Irrawaddy': [96.2, 23.8],
                          'Salween': [93.2, 31.15],
                          'Mekong': [96, 31.8],
                          'Yellow': [106.0, 36]}
        title_dict = {'Amu_Darya': 'Amu\nDarya',
                      'Brahmaputra': 'Brahma-\nputra',
                      'Ganges': 'Ganges',
                      'Ili': 'Ili',
                      'Indus': 'Indus',
                      'Inner_Tibetan_Plateau': 'Inner TP',
                      'Inner_Tibetan_Plateau_extended': 'Inner TP ext',
                      'Irrawaddy': 'Irrawaddy',
                      'Mekong': 'Mk',
                      'Salween': 'Sw',
                      'Syr_Darya': 'Syr\nDarya',
                      'Tarim': 'Tarim',
                      'Yangtze': 'Yz'}
        group_colordict = {'Amu_Darya': 'mediumblue',
                           'Brahmaputra': 'salmon',
                           'Ganges': 'lightskyblue',
                           'Ili': 'royalblue',
                           'Indus': 'darkred',
                           'Inner_Tibetan_Plateau': 'gold',
                           'Inner_Tibetan_Plateau_extended': 'navy',
                           'Irrawaddy': 'white',
                           'Mekong': 'white',
                           'Salween': 'plum',
                           'Syr_Darya':'darkolivegreen',
                           'Tarim': 'olive',
                           'Yangtze': 'orange',
                           'Yellow':'white'}
        
        # Add country borders for reference
        if grouping == 'rgi_region':
            ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, linewidth=0.25)
#        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.25)
        # Set the extent
        ax.set_extent([east, 66, 24, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
#        ax.set_xlabel(xlabel, size=10, labelpad=0)
#        ax.set_ylabel(ylabel, size=10, labelpad=0)
        ax.xaxis.set_tick_params(pad=0, size=2, labelsize=8)
        ax.yaxis.set_tick_params(pad=0, size=2, labelsize=8)
        ax.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        
        for rec in group_shp.records():
            if rec.attributes[group_shp_attr] in groups:
                group = rec.attributes[group_shp_attr]
                ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', edgecolor='Black', 
                                  linewidth=1, zorder=3)
#                ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor=group_colordict[group], 
#                                  edgecolor='Black', linewidth=0.5, alpha=0.5, zorder=3)
                if group not in ['Yellow', 'Irrawaddy', 'Mekong']:
                    ax.text(title_location[rec.attributes[group_shp_attr]][0], 
                            title_location[rec.attributes[group_shp_attr]][1], 
                            title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', size=9, 
                            zorder=4)
#                if group == 'Brahmaputra':
#                    ax.text(92.5, 26, 'Brahma-', horizontalalignment='center', size=9, zorder=4)
#                    ax.text(92.5, 24.8, 'putra', horizontalalignment='center', size=9, zorder=4)
#        
        colorbar_dict = {'volume_norm':[0,1],
                         'runoff_glac_monthly':[2020,2080]}
        cmap = mpl.cm.RdYlBu
        norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])
          
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax = plt.axes([0.92, 0.38, 0.015, 0.25])
        cbar = plt.colorbar(sm, ax=ax, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_tick_params(pad=0)
        cbar.ax.tick_params(labelsize=8)
        for n, label in enumerate(cax.xaxis.get_ticklabels()):
            if n%2 != 0:
                label.set_visible(False)
#        fig.text(0.5, 0.89, 'Year', ha='center', va='center', size=10)
#        fig.text(0.5, 0.89, 'Peak water (year) - RCP ' + rcp[3:], ha='center', va='center', size=10)
            
        # Degree peakwater
        x, y, z = [], [], []
        for group in groups_deg:
            vn_multimodel_mean = ds_multimodel_deg[rcp][group]
            peakwater_yr, peakwater_chg, runoff_chg = (
                    peakwater(vn_multimodel_mean[time_idx_start:time_idx_end], 
                              time_values_annual[time_idx_start:time_idx_end], peakwater_Nyears))
            x.append(deg_dict[group][0])
            y.append(deg_dict[group][1])
            z.append(peakwater_yr)
        x = np.array(x)
        y = np.array(y)
        
        lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
        lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
        x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
        y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
        z_array = np.zeros((len(y_adj), len(x_adj)))
        z_array[z_array==0] = np.nan
        for i in range(len(z)):
            row_idx = int((y[i] - y.min()) / degree_size)
            col_idx = int((x[i] - x.min()) / degree_size)
            z_array[row_idx, col_idx] = z[i]
        ax.pcolormesh(lons, lats, z_array, cmap='RdYlBu', norm=norm, zorder=2)
    
        # Save figure
        fig.set_size_inches(3.5,6)
        figure_fn = 'peakwater_map_' + grouping + '_multimodel_' + rcp +  '.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)
        
        
        #%%
        # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====
        # Create the projection
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':cartopy.crs.PlateCarree()},
                               gridspec_kw = {'wspace':0, 'hspace':0})
        
        # Add group and attribute of interest
        if grouping == 'rgi_region':
            group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
            group_shp_attr = 'RGI_CODE'
        elif grouping == 'watershed':
            group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
            group_shp_attr = 'watershed'
        elif grouping == 'kaab':
            group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
            group_shp_attr = 'Name'
        
        title_location = {'Syr_Darya': [70.5, 42.7],
                          'Ili': [83, 45],
                          'Amu_Darya': [68.2, 36],
                          'Tarim': [82.5, 38.5],
                          'Inner_Tibetan_Plateau_extended': [98.7, 39.75],
                          'Indus': [72, 32],
                          'Inner_Tibetan_Plateau': [86.2, 34.3],
                          'Yangtze': [100.7, 31.5],
                          'Ganges': [81.3, 26.6],
                          'Brahmaputra': [91.9, 24.3],
                          'Irrawaddy': [96.2, 23.8],
                          'Salween': [92.6, 31.15],
                          'Mekong': [96, 31.8],
                          'Yellow': [106.0, 36]}
    
        # Set the extent
        ax.set_extent([east, 66, 24, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax.xaxis.set_tick_params(pad=0, size=2, labelsize=6)
        ax.yaxis.set_tick_params(pad=0, size=2, labelsize=6)
        ax.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        
        for rec in group_shp.records():
            if rec.attributes[group_shp_attr] in groups:
                group = rec.attributes[group_shp_attr]
                ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', edgecolor='Black', 
                                  linewidth=0.5, zorder=3)
                if group not in ['Yellow', 'Irrawaddy', 'Mekong']:
                    ax.text(title_location[rec.attributes[group_shp_attr]][0], 
                            title_location[rec.attributes[group_shp_attr]][1], 
                            title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', size=8, 
                            zorder=4)

        colorbar_dict = {'volume_norm':[0,1],
                         'runoff_glac_monthly':[2020,2080]}
        cmap = mpl.cm.RdYlBu
        norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])
          
        # Add colorbar
        cmap_alpha=1
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax = plt.axes([0.92, 0.38, 0.015, 0.23])
        cbar = plt.colorbar(sm, ax=ax, cax=cax, orientation='vertical', alpha=cmap_alpha)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_tick_params(pad=0)
        cbar.ax.tick_params(labelsize=6)
        for n, label in enumerate(cax.xaxis.get_ticklabels()):
            if n%2 != 0:
                label.set_visible(False)
        fig.text(0.965, 0.63, 'Year', ha='center', va='center', size=7)
            
        # Degree peakwater
        x, y, z, s = [], [], [], []
        for group in groups_deg:
            vn_multimodel_mean = ds_multimodel_deg[rcp][group]
            peakwater_yr, peakwater_chg, runoff_chg = (
                    peakwater(vn_multimodel_mean[time_idx_start:time_idx_end], 
                              time_values_annual[time_idx_start:time_idx_end], peakwater_Nyears))
            x.append(deg_dict[group][0])
            y.append(deg_dict[group][1])
            z.append(peakwater_yr)
            s.append(vn_multimodel_mean[0:16].mean())
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        s = np.array(s) / 1e9 # convert to Gt/yr
        
        # Size thresholds
        s_sizes = [1, 3, 9, 20]
        s_plot = np.array(s)
        s_plot[s <= 0.01] = s_sizes[0]
        s_plot[s > 0.01] = s_sizes[1]
        s_plot[s > 0.1] = s_sizes[2]
        s_plot[s > 1] = s_sizes[3]
        marker_linecolor='k'
        marker_linewidth=0.1
        a = ax.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, 
#                       s=5,
                       s=s_plot,
                       marker='o', edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
        
        # Add legend        
        circ1 = ax.scatter([10],[0], s=s_sizes[0], marker='o', color='grey', 
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ2 = ax.scatter([0],[0], s=s_sizes[1], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ3 = ax.scatter([0],[0], s=s_sizes[2], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ4 = ax.scatter([0],[0], s=s_sizes[3], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        legend=ax.legend([circ1,circ2,circ3,circ4], ['0.001', '0.01', '0.1', '1'], 
                  scatterpoints=1,
#                  scatteryoffsets=0.5,
                  ncol=5, loc='upper right', fontsize=6, 
                  labelspacing=0.3,
                  columnspacing=0,
                  handletextpad=0,
                  handlelength=1,
                  borderpad=0.2,
                  framealpha=1,
                  title='Runoff (Gt yr$^{-1}$)',
                  borderaxespad=0.2,
#                  titlefontsize=5,
#                  title_fontsize=5
                  )
        legend.get_title().set_fontsize('6')
        legend.get_frame().set_linewidth(0.5)
    
        # Save figure
        fig.set_size_inches(3.5,6)
        figure_fn = 'peakwater_map_' + grouping + '_multimodel_' + rcp +  '_circles_lowres.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=150, transparent=True)
        
    #%%
    # ===== THREE RCPS TOGETHER WITH CIRCLES =====
    if option_plot4paper_3rcps == 1 and len(rcps) >= 3:
        
        title_location = {'Syr_Darya': [70.5, 42.7],
                          'Ili': [83, 45],
                          'Amu_Darya': [68.2, 36],
                          'Tarim': [82.5, 38.5],
                          'Inner_Tibetan_Plateau_extended': [98.7, 39.75],
                          'Indus': [72, 32],
                          'Inner_Tibetan_Plateau': [86.2, 34.3],
                          'Yangtze': [100.7, 31.5],
                          'Ganges': [81.3, 26.6],
                          'Brahmaputra': [91.9, 24.3],
                          'Irrawaddy': [96.2, 23.8],
                          'Salween': [92.6, 31.15],
                          'Mekong': [96, 31.8],
                          'Yellow': [106.0, 36]}
        
        # Create the projection
        fig = plt.figure()
    
        # Custom subplots
        gs = mpl.gridspec.GridSpec(122, 1)
        ax1 = plt.subplot(gs[0:40,0], projection=cartopy.crs.PlateCarree())
        ax2 = plt.subplot(gs[41:81,0], projection=cartopy.crs.PlateCarree())
        ax3 = plt.subplot(gs[82:122,0], projection=cartopy.crs.PlateCarree())
        
        # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====
        marker_linecolor='k'
        marker_linewidth=0.1
        
        for group in groups:

            # Add group and attribute of interest
            if grouping == 'rgi_region':
                group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
                group_shp_attr = 'RGI_CODE'
            elif grouping == 'watershed':
                group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
                group_shp_attr = 'watershed'
            elif grouping == 'kaab':
                group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
                group_shp_attr = 'Name'

            group_fontsize = 10
            for rec in group_shp.records():
                if rec.attributes[group_shp_attr] in groups:
                    group = rec.attributes[group_shp_attr]
                    ax1.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax2.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax3.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    if group not in ['Yellow', 'Irrawaddy', 'Mekong']:
                        ax1.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
                        ax2.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
                        ax3.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
        
        for nrcp, rcp in enumerate(['rcp26', 'rcp45', 'rcp85']):
            for group in groups:
            
                vn_multimodel_mean = ds_multimodel[rcp][group]
                peakwater_yr, peakwater_chg, runoff_chg = (
                        peakwater(vn_multimodel_mean[time_idx_start:time_idx_end], 
                                  time_values_annual[time_idx_start:time_idx_end], peakwater_Nyears))
                
                print(rcp, group, peakwater_yr, np.round(peakwater_chg,0), np.round(runoff_chg,0)) 

                # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====  
                # Degree peakwater
                x, y, z, s = [], [], [], []
                for group in groups_deg:
                    vn_multimodel_mean = ds_multimodel_deg[rcp][group]
                    peakwater_yr, peakwater_chg, runoff_chg = (
                            peakwater(vn_multimodel_mean[time_idx_start:time_idx_end], 
                                      time_values_annual[time_idx_start:time_idx_end], peakwater_Nyears))
                    x.append(deg_dict[group][0])
                    y.append(deg_dict[group][1])
                    z.append(peakwater_yr)
                    s.append(vn_multimodel_mean[0:16].mean())
                x = np.array(x)
                y = np.array(y)
                z = np.array(z)
                s = np.array(s) / 1e9 # convert to Gt/yr
                
                # Size thresholds
#                s_sizes = [1, 3, 9, 20]
                s_sizes = [2, 6, 18, 40]
                s_plot = np.array(s)
                s_plot[s <= 0.01] = s_sizes[0]
                s_plot[s > 0.01] = s_sizes[1]
                s_plot[s > 0.1] = s_sizes[2]
                s_plot[s > 1] = s_sizes[3]
                
                if nrcp == 0:
                    a = ax1.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                    edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
                elif nrcp == 1:
                    a = ax2.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                    edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
                elif nrcp == 2:
                    a = ax3.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                    edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
                
        west = 66
        north = 47
        south = 24
        axis_fontsize = 8
        # Set the extent
        ax1.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax1.set_xticks(np.arange(70,east+1,10), cartopy.crs.PlateCarree())
        ax1.set_yticks(np.arange(30,north+1,5), cartopy.crs.PlateCarree())
        ax1.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax1.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax2.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax2.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax2.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax2.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax2.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax2.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax3.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax3.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax3.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax3.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax3.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax3.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
            
            
        # Add colorbar legend for peakwater
        leg_fontsize = 10
        colorbar_dict = {'volume_norm':[0,1],
                         'runoff_glac_monthly':[2020,2080]}
        cmap = mpl.cm.RdYlBu
        norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])
        cmap_alpha = 1
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax = fig.add_axes([0.68, 0.5, 0.01, 0.35])
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical', alpha=cmap_alpha)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_tick_params(pad=0)
        cbar.ax.tick_params(labelsize=leg_fontsize)
        for n, label in enumerate(cax.xaxis.get_ticklabels()):
            if n%2 != 0:
                label.set_visible(False)
        fig.text(0.7, 0.87, 'Year', ha='center', va='center', size=leg_fontsize)
        
        # Add circle size legend       
        circ1 = ax1.scatter([0],[0], s=s_sizes[0], marker='o', color='grey', 
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ2 = ax1.scatter([0],[0], s=s_sizes[1], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ3 = ax1.scatter([0],[0], s=s_sizes[2], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ4 = ax1.scatter([0],[0], s=s_sizes[3], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        legend = fig.legend([circ1,circ2,circ3,circ4], ['0.001', '0.01', '0.1', '1'], 
          scatterpoints=1,
          ncol=1, 
          loc='lower right', 
          bbox_to_anchor=(0.395,0.24),
          fontsize=leg_fontsize, 
          labelspacing=0.3,
          columnspacing=0,
          handletextpad=0,
          handlelength=1,
          borderpad=0.2,
          framealpha=0,
#          title='Runoff\n(Gt yr$^{-1}$)',
          borderaxespad=0.2,
          )
#        legend.get_title().set_fontsize(str(leg_fontsize + 1))
        legend.get_frame().set_linewidth(0)
        fig.text(0.7, 0.4425, 'Runoff', ha='center', va='center', size=leg_fontsize)
        fig.text(0.7, 0.425, '(Gt yr$^{-1}$)', ha='center', va='center', size=leg_fontsize)
        
        fig.text(0.37, 0.865, 'A', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.37, 0.61, 'B', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.37, 0.356, 'C', ha='center', va='center', size=12, fontweight='bold')
            
        # Save figure
        fig_height = 14
        fig_width = 10.5
        fig.set_size_inches(fig_height,fig_width)
        figure_fn = 'peakwater_map_' + grouping + '_multimodel_3rcps_circles.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)
        #%%

if option_temp_and_prec_map == 1:
    netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_subset/'  
    figure_fp = netcdf_fp_cmip5 + 'figures/'
    if os.path.exists(figure_fp) == False:
        os.mkdir(figure_fp)
    
    rcps = ['rcp26', 'rcp45', 'rcp85']
    
    option_plot4paper_3rcps = 1
    
    startyear = 2015
    endyear = 2100
    
    vn = 'temperature'
    grouping = 'watershed'
    
#    east = 60
#    west = 110
#    south = 15
#    north = 50
    east = 104
    west = 64
    south = 26
    north = 47
    xtick = 5
    ytick = 5
    xlabel = 'Longitude ($\mathregular{^{\circ}}$)'
    ylabel = 'Latitude ($\mathregular{^{\circ}}$)'

    startyear=2000
    endyear=2100
    

#    ref_startyear = 2000
#    ref_endyear = 2015
    
#    plt_startyear = 2015
#    plt_endyear = 2100
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0, option_wateryear=1)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    groups_deg, group_cn_deg = select_groups('degree', main_glac_rgi)
    deg_groups = main_glac_rgi.groupby(['CenLon_round', 'CenLat_round']).size().index.values.tolist()
    deg_dict = dict(zip(np.arange(0,len(deg_groups)), deg_groups))

    #%%
    # Glacier and grouped annual runoff
    ds_temp_multimodel_deg, ds_temp_multimodel_std_deg = {}, {}
    ds_prec_multimodel_deg, ds_prec_multimodel_std_deg = {}, {}
    ds_temp_multimodel_all, ds_prec_multimodel_all = {}, {}
    for rcp in rcps:
        
        ds_temp_multimodel_deg[rcp], ds_temp_multimodel_std_deg[rcp] = {}, {}
        ds_prec_multimodel_deg[rcp], ds_prec_multimodel_std_deg[rcp] = {}, {}
        
        for ngcm, gcm_name in enumerate(gcm_names):
            print(rcp, gcm_name)
            
            group_glacidx = {}
            vol_group_dict = {}
            temp_group_dict = {}
            prectotal_group_dict = {}
        
            # Extract data from netcdf
            for region in regions:
                try:
                    # Load datasets
                    ds_fn = ('R' + str(region) + '--all--' + gcm_name + '_' + rcp + 
                             '_c2_ba1_100sets_2000_2100--subset.nc')
                    ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                    df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
                    df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                                   str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
                    skip_gcm = 0
                except:
                    skip_gcm = 1
                    print('Skip', gcm_name, rcp, region)
                        
                if skip_gcm == 0:
                    # Extract time variable
                    time_values_annual = ds.coords['year_plus1'].values
                    time_values_monthly = ds.coords['time'].values
                    # Merge datasets
                    if region == regions[0]:
                        area_glac_all = ds['area_glac_annual'].values[:,:,0]
                        area_glac_std_all = ds['area_glac_annual'].values[:,:,1]
                        df_all = df
                    else:
                        area_glac_all = np.concatenate((area_glac_all, ds['area_glac_annual'].values[:,:,0]), axis=0)
                        area_glac_std_all = np.concatenate((area_glac_std_all, ds['area_glac_annual'].values[:,:,1]),axis=0)
                        df_all = pd.concat([df_all, df], axis=0)        
                    ds.close()
                
            # Remove RGIIds from main_glac_rgi that are not in the model runs
            if ngcm == 0:
                rgiid_df = list(df_all.RGIId.values)
                rgiid_all = list(main_glac_rgi.RGIId.values)
                rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
                main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
                main_glac_rgi.reset_index(inplace=True, drop=True)
    
            # Annual Temperature, Precipitation, and Accumulation
            temp_glac_all, prectotal_glac_all, elev_glac_all = retrieve_gcm_data(gcm_name, rcp, main_glac_rgi)
            temp_glac_all_annual = gcmbiasadj.annual_avg_2darray(temp_glac_all)
            prectotal_glac_all_annual = gcmbiasadj.annual_sum_2darray(prectotal_glac_all)
            
            # Groups for single GCM
            for ngroup, group in enumerate(groups_deg):
                # Select subset of data
                group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn_deg] == group].index.values.tolist()
                group_glacidx[group] = group_glac_indices
                
                # Regional Volume, and Area-weighted Temperature and Precipitation (SINGLE GCM)
                temp_group_all = ((temp_glac_all_annual[group_glac_indices,:] * 
                                   area_glac_all[group_glac_indices,:][:,0][:,np.newaxis]).sum(axis=0) / 
                                  area_glac_all[group_glac_indices,:][:,0].sum())
                prectotal_group_all = ((prectotal_glac_all_annual[group_glac_indices,:] * 
                                        area_glac_all[group_glac_indices,:][:,0][:,np.newaxis]).sum(axis=0) / 
                                       area_glac_all[group_glac_indices,:][:,0].sum())
                    
                # Expand dimensions for multi-model calculations
                temp_group_all = np.expand_dims(temp_group_all, axis=1)
                prectotal_group_all = np.expand_dims(prectotal_group_all, axis=1)
    
                temp_group_dict[group] = temp_group_all
                prectotal_group_dict[group] = prectotal_group_all
                
            # Expand dimensions for multi-model calculation
            temp_glac_all_annual = np.expand_dims(temp_glac_all_annual, axis=2)
            prectotal_glac_all_annual = np.expand_dims(prectotal_glac_all_annual, axis=2)
            
            # ===== MULTI-MODEL =====
            if ngcm == 0:
                temp_glac_all_annual_multimodel = temp_glac_all_annual
                prectotal_glac_all_annual_multimodel = prectotal_glac_all_annual
                
                temp_group_dict_multimodel = temp_group_dict
                prectotal_group_dict_multimodel = prectotal_group_dict
                
            else:
                temp_glac_all_annual_multimodel = np.append(temp_glac_all_annual_multimodel, 
                                                            temp_glac_all_annual, axis=2)
                prectotal_glac_all_annual_multimodel = np.append(prectotal_glac_all_annual_multimodel, 
                                                                 prectotal_glac_all_annual, axis=2)
                
                for ngroup, group in enumerate(groups_deg):
                    temp_group_dict_multimodel[group] = np.append(temp_group_dict_multimodel[group], 
                                                                  temp_group_dict[group], axis=1)
                    prectotal_group_dict_multimodel[group] = np.append(prectotal_group_dict_multimodel[group], 
                                                                       prectotal_group_dict[group], axis=1)

        ds_temp_multimodel_all[rcp] = temp_glac_all_annual_multimodel
        HMA_multimodel_gcm_temp = temp_glac_all_annual_multimodel.mean(axis=2).mean(axis=0)
        HMA_multimodel_gcm_temp_increase = HMA_multimodel_gcm_temp[90:101].mean() - HMA_multimodel_gcm_temp[0:16].mean()
        print('manually specifying range')
        print(rcp, 'HMA temp increase', HMA_multimodel_gcm_temp_increase)
        
        ds_prec_multimodel_all[rcp] = prectotal_glac_all_annual_multimodel
        HMA_multimodel_gcm_prec = prectotal_glac_all_annual_multimodel.mean(axis=2).mean(axis=0)
        HMA_multimodel_gcm_prec_increase = HMA_multimodel_gcm_prec[85:101].mean() / HMA_multimodel_gcm_prec[0:16].mean()
        print(rcp, 'HMA prec increase', HMA_multimodel_gcm_prec_increase)

        # Group multimodel mean for each RCP
        for ngroup, group in enumerate(groups_deg):
            ds_temp_multimodel_deg[rcp][group] = temp_group_dict_multimodel[group].mean(axis=1)
            ds_prec_multimodel_deg[rcp][group] = prectotal_group_dict_multimodel[group].mean(axis=1)
            
    # Area for size plotting
    area_init_group = {}
    for ngroup, group in enumerate(groups_deg):
        # Select subset of data
        area_init_group[group] = area_glac_all[group_glacidx[group],0].sum()

    #%%
    for rcp in rcps:
        temp_glac_all_annual_multimodel = ds_temp_multimodel_all[rcp]
        HMA_multimodel_gcm_temp = temp_glac_all_annual_multimodel.mean(axis=2).mean(axis=0)
        HMA_multimodel_gcm_temp_increase = HMA_multimodel_gcm_temp[85:101].mean() - HMA_multimodel_gcm_temp[0:16].mean()
        print(rcp, 'HMA temp increase', HMA_multimodel_gcm_temp_increase)
        prectotal_glac_all_annual_multimodel = ds_prec_multimodel_all[rcp]
        HMA_multimodel_gcm_prec = prectotal_glac_all_annual_multimodel.mean(axis=2).mean(axis=0)
        HMA_multimodel_gcm_prec_increase = HMA_multimodel_gcm_prec[85:101].mean() / HMA_multimodel_gcm_prec[0:16].mean()
        print(rcp, 'HMA prec increase', HMA_multimodel_gcm_prec_increase)

    time_idx_start = np.where(time_values_annual == startyear)[0][0]
    time_idx_end = np.where(time_values_annual == endyear)[0][0] + 1

    #%%
    # ===== THREE RCPS TOGETHER WITH CIRCLES FOR TEMPERATURE =====
    if option_plot4paper_3rcps == 1 and len(rcps) >= 3:
        
        title_location = {'Syr_Darya': [70.5, 42.7],
                          'Ili': [83, 45],
                          'Amu_Darya': [68.2, 36],
                          'Tarim': [82.5, 38.5],
                          'Inner_Tibetan_Plateau_extended': [98.7, 39.75],
                          'Indus': [72, 32],
                          'Inner_Tibetan_Plateau': [86.2, 34.3],
                          'Yangtze': [100.7, 31.5],
                          'Ganges': [81.3, 26.6],
                          'Brahmaputra': [91.9, 24.3],
                          'Irrawaddy': [96.2, 23.8],
                          'Salween': [92.6, 31.15],
                          'Mekong': [96, 31.8],
                          'Yellow': [106.0, 36]}
        title_dict = {'Amu_Darya': 'Amu\nDarya',
                      'Brahmaputra': 'Brahma-\nputra',
                      'Ganges': 'Ganges',
                      'Ili': 'Ili',
                      'Indus': 'Indus',
                      'Inner_Tibetan_Plateau': 'Inner TP',
                      'Inner_Tibetan_Plateau_extended': 'Inner TP ext',
                      'Irrawaddy': 'Irrawaddy',
                      'Mekong': 'Mk',
                      'Salween': 'Sw',
                      'Syr_Darya': 'Syr\nDarya',
                      'Tarim': 'Tarim',
                      'Yangtze': 'Yz'}
        group_colordict = {'Amu_Darya': 'mediumblue',
                           'Brahmaputra': 'salmon',
                           'Ganges': 'lightskyblue',
                           'Ili': 'royalblue',
                           'Indus': 'darkred',
                           'Inner_Tibetan_Plateau': 'gold',
                           'Inner_Tibetan_Plateau_extended': 'navy',
                           'Irrawaddy': 'white',
                           'Mekong': 'white',
                           'Salween': 'plum',
                           'Syr_Darya':'darkolivegreen',
                           'Tarim': 'olive',
                           'Yangtze': 'orange',
                           'Yellow':'white'}
        
        # Create the projection
        fig = plt.figure()
    
        # Custom subplots
        gs = mpl.gridspec.GridSpec(122, 1)
        ax1 = plt.subplot(gs[0:40,0], projection=cartopy.crs.PlateCarree())
        ax2 = plt.subplot(gs[41:81,0], projection=cartopy.crs.PlateCarree())
        ax3 = plt.subplot(gs[82:122,0], projection=cartopy.crs.PlateCarree())
        
        # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====
        marker_linecolor='k'
        marker_linewidth=0.1
        
        for group in groups:

            # Add group and attribute of interest
            if grouping == 'rgi_region':
                group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
                group_shp_attr = 'RGI_CODE'
            elif grouping == 'watershed':
                group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
                group_shp_attr = 'watershed'
            elif grouping == 'kaab':
                group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
                group_shp_attr = 'Name'

            group_fontsize = 10
            for rec in group_shp.records():
                if rec.attributes[group_shp_attr] in groups:
                    group = rec.attributes[group_shp_attr]
                    ax1.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax2.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax3.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    if group not in ['Yellow', 'Irrawaddy', 'Mekong']:
                        ax1.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
                        ax2.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
                        ax3.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
        
        for nrcp, rcp in enumerate(['rcp26', 'rcp45', 'rcp85']):
            # Add colorbar legend for peakwater
            leg_fontsize = 10
            cmap_alpha=1
            cmap = mpl.cm.RdYlBu_r
            if rcp == 'rcp26':
                norm = plt.Normalize(0.5, 1.5)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax1 = fig.add_axes([0.68, 0.65, 0.01, 0.19])
                cbar1 = fig.colorbar(sm, cax=cax1, orientation='vertical', alpha=cmap_alpha)
                cax1.xaxis.set_ticks_position('top')
                cax1.xaxis.set_tick_params(pad=0)
                cbar1.ax.tick_params(labelsize=leg_fontsize)
                for n, label in enumerate(cax1.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
                fig.text(0.705, 0.865, '$\Delta$ Temperature\n($^\circ$C)', ha='center', va='center', size=leg_fontsize)
            elif rcp == 'rcp45':
                norm = plt.Normalize(2, 3)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax1 = fig.add_axes([0.68, 0.41, 0.01, 0.19])
                cbar1 = fig.colorbar(sm, cax=cax1, orientation='vertical', alpha=cmap_alpha)
                cax1.xaxis.set_ticks_position('top')
                cax1.xaxis.set_tick_params(pad=0)
                cbar1.ax.tick_params(labelsize=leg_fontsize)
                for n, label in enumerate(cax1.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
            elif rcp == 'rcp85':
                norm = plt.Normalize(5, 7)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax1 = fig.add_axes([0.68, 0.15, 0.01, 0.19])
                cbar1 = fig.colorbar(sm, cax=cax1, orientation='vertical', alpha=cmap_alpha)
                cax1.xaxis.set_ticks_position('top')
                cax1.xaxis.set_tick_params(pad=0)
                cbar1.ax.tick_params(labelsize=leg_fontsize)
                for n, label in enumerate(cax1.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
            cmap_alpha = 1
            
            # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====  
            # Degree peakwater
            x, y, z, s = [], [], [], []
            for group in groups_deg:
                vn_multimodel_mean = ds_temp_multimodel_deg[rcp][group]
                vn_multimodel_mean_plot = (
                        vn_multimodel_mean[-1] - vn_multimodel_mean[0:16].mean())
                x.append(deg_dict[group][0])
                y.append(deg_dict[group][1])
                z.append(vn_multimodel_mean_plot)
                s.append(area_init_group[group])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            s = np.array(s)
            
            # Size thresholds
            s_sizes = [2, 6, 18, 40]
            s_plot = np.array(s)
            s_plot[s <= 10] = s_sizes[0]
            s_plot[s > 10] = s_sizes[1]
            s_plot[s > 100] = s_sizes[2]
            s_plot[s > 1000] = s_sizes[3]
            
            if nrcp == 0:
                a = ax1.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 1:
                a = ax2.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 2:
                a = ax3.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
                
        west = 66
        north = 47
        south = 24
        axis_fontsize = 8
        # Set the extent
        ax1.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax1.set_xticks(np.arange(70,east+1,10), cartopy.crs.PlateCarree())
        ax1.set_yticks(np.arange(30,north+1,5), cartopy.crs.PlateCarree())
        ax1.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax1.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax2.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax2.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax2.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax2.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax2.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax2.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax3.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax3.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax3.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax3.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax3.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax3.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        # Add circle size legend       
        circ1 = ax1.scatter([0],[0], s=s_sizes[0], marker='o', color='grey', 
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ2 = ax1.scatter([0],[0], s=s_sizes[1], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ3 = ax1.scatter([0],[0], s=s_sizes[2], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ4 = ax1.scatter([0],[0], s=s_sizes[3], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        legend=ax1.legend([circ1,circ2,circ3,circ4], ['0.001', '0.01', '0.1', '1'], 
                          scatterpoints=1,
        #                  scatteryoffsets=0.5,
                          ncol=5, loc='upper right', fontsize=leg_fontsize, 
                          labelspacing=0.3,
                          columnspacing=0,
                          handletextpad=0,
                          handlelength=1,
                          borderpad=0.2,
                          framealpha=1,
                          title='Initial area (km$^{2}$)',
                          borderaxespad=0.2,
                          )
        legend.get_title().set_fontsize(str(leg_fontsize))
        legend.get_frame().set_linewidth(0.5)
        
        fig.text(0.37, 0.865, 'A', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.37, 0.61, 'B', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.37, 0.356, 'C', ha='center', va='center', size=12, fontweight='bold')
            
        # Save figure
        fig_height = 14
        fig_width = 10.5
        fig.set_size_inches(fig_height,fig_width)
        figure_fn = 'temp_map_' + grouping + '_multimodel_3rcps_circles.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)
        #%%
        # Create the projection
        fig = plt.figure()
    
        # Custom subplots
        gs = mpl.gridspec.GridSpec(122, 1)
        ax1 = plt.subplot(gs[0:40,0], projection=cartopy.crs.PlateCarree())
        ax2 = plt.subplot(gs[41:81,0], projection=cartopy.crs.PlateCarree())
        ax3 = plt.subplot(gs[82:122,0], projection=cartopy.crs.PlateCarree())
        
        # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====
        marker_linecolor='k'
        marker_linewidth=0.1
        
        for group in groups:

            # Add group and attribute of interest
            if grouping == 'rgi_region':
                group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
                group_shp_attr = 'RGI_CODE'
            elif grouping == 'watershed':
                group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
                group_shp_attr = 'watershed'
            elif grouping == 'kaab':
                group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
                group_shp_attr = 'Name'

            group_fontsize = 10
            for rec in group_shp.records():
                if rec.attributes[group_shp_attr] in groups:
                    group = rec.attributes[group_shp_attr]
                    ax1.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax2.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax3.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    if group not in ['Yellow', 'Irrawaddy', 'Mekong']:
                        ax1.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
                        ax2.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
                        ax3.text(title_location[rec.attributes[group_shp_attr]][0], 
                                        title_location[rec.attributes[group_shp_attr]][1], 
                                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', 
                                        size=group_fontsize, zorder=4)
        
        # Add colorbar legend
        leg_fontsize = 10
        cmap = mpl.cm.RdYlBu
        norm = plt.Normalize(0.9, 1.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax1 = fig.add_axes([0.68, 0.3, 0.01, 0.4])
        cbar1 = fig.colorbar(sm, cax=cax1, orientation='vertical', alpha=cmap_alpha)
        cax1.xaxis.set_ticks_position('top')
        cax1.xaxis.set_tick_params(pad=0)
        cbar1.ax.tick_params(labelsize=leg_fontsize)
        for n, label in enumerate(cax1.xaxis.get_ticklabels()):
            if n%2 != 0:
                label.set_visible(False)
        fig.text(0.705, 0.73, '$\Delta$ Precipitation\n(-)', ha='center', va='center', size=leg_fontsize)
        cmap_alpha = 1
        
        for nrcp, rcp in enumerate(['rcp26', 'rcp45', 'rcp85']):
            # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====  
            # Degree peakwater
            x, y, z, s = [], [], [], []
            for group in groups_deg:
                vn_multimodel_mean = ds_prec_multimodel_deg[rcp][group]
                vn_multimodel_mean_plot = (
                        vn_multimodel_mean[-1] / vn_multimodel_mean[0:16].mean())
                x.append(deg_dict[group][0])
                y.append(deg_dict[group][1])
                z.append(vn_multimodel_mean_plot)
                s.append(area_init_group[group])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            s = np.array(s)
            
            # Size thresholds
            s_sizes = [2, 6, 18, 40]
            s_plot = np.array(s)
            s_plot[s <= 10] = s_sizes[0]
            s_plot[s > 10] = s_sizes[1]
            s_plot[s > 100] = s_sizes[2]
            s_plot[s > 1000] = s_sizes[3]
            
            if nrcp == 0:
                a = ax1.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 1:
                a = ax2.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 2:
                a = ax3.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
                
        west = 66
        north = 47
        south = 24
        axis_fontsize = 8
        # Set the extent
        ax1.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax1.set_xticks(np.arange(70,east+1,10), cartopy.crs.PlateCarree())
        ax1.set_yticks(np.arange(30,north+1,5), cartopy.crs.PlateCarree())
        ax1.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax1.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax2.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax2.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax2.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax2.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax2.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax2.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax3.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax3.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax3.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax3.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax3.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax3.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        # Add circle size legend       
        circ1 = ax1.scatter([0],[0], s=s_sizes[0], marker='o', color='grey', 
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ2 = ax1.scatter([0],[0], s=s_sizes[1], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ3 = ax1.scatter([0],[0], s=s_sizes[2], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ4 = ax1.scatter([0],[0], s=s_sizes[3], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        legend=ax1.legend([circ1,circ2,circ3,circ4], ['0.001', '0.01', '0.1', '1'], 
                          scatterpoints=1,
        #                  scatteryoffsets=0.5,
                          ncol=5, loc='upper right', fontsize=leg_fontsize, 
                          labelspacing=0.3,
                          columnspacing=0,
                          handletextpad=0,
                          handlelength=1,
                          borderpad=0.2,
                          framealpha=1,
                          title='Initial area (km$^{2}$)',
                          borderaxespad=0.2,
                          )
        legend.get_title().set_fontsize(str(leg_fontsize))
        legend.get_frame().set_linewidth(0.5)
        
        fig.text(0.37, 0.865, 'A', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.37, 0.61, 'B', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.37, 0.356, 'C', ha='center', va='center', size=12, fontweight='bold')
            
        # Save figure
        fig_height = 14
        fig_width = 10.5
        fig.set_size_inches(fig_height,fig_width)
        figure_fn = 'prec_map_' + grouping + '_multimodel_3rcps_circles.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)
        #%%
        # ===== BOTH PRECIPITATION AND TEMPERATURE ====        
        # Create the projection
        fig = plt.figure()
    
        # Custom subplots
        gs = mpl.gridspec.GridSpec(122, 101, wspace=0, hspace=0)
        ax1 = plt.subplot(gs[0:40,0:46], projection=cartopy.crs.PlateCarree())
        ax2 = plt.subplot(gs[41:81,0:46], projection=cartopy.crs.PlateCarree())
        ax3 = plt.subplot(gs[82:122,0:46], projection=cartopy.crs.PlateCarree())
        ax4 = plt.subplot(gs[0:40,55:101], projection=cartopy.crs.PlateCarree())
        ax5 = plt.subplot(gs[41:81,55:101], projection=cartopy.crs.PlateCarree())
        ax6 = plt.subplot(gs[82:122,55:101], projection=cartopy.crs.PlateCarree())
        
        # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====
        marker_linecolor='k'
        marker_linewidth=0.1
        leg_fontsize = 9
        
        for group in groups:

            # Add group and attribute of interest
            if grouping == 'rgi_region':
                group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
                group_shp_attr = 'RGI_CODE'
            elif grouping == 'watershed':
                group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
                group_shp_attr = 'watershed'
            elif grouping == 'kaab':
                group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
                group_shp_attr = 'Name'

            group_fontsize = 10
            for rec in group_shp.records():
                if rec.attributes[group_shp_attr] in groups:
                    group = rec.attributes[group_shp_attr]
                    ax1.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax2.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax3.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax4.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax5.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
                    ax6.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                                              edgecolor='Black', linewidth=0.5, zorder=3)
        
        for nrcp, rcp in enumerate(['rcp26', 'rcp45', 'rcp85']):
            # Add colorbar legend for peakwater
            cmap_alpha=1
            cmap = mpl.cm.RdYlBu_r
            if rcp == 'rcp26':
                norm = plt.Normalize(0.5, 1.5)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax1 = fig.add_axes([0.485, 0.64, 0.01, 0.18])
                cbar1 = fig.colorbar(sm, cax=cax1, orientation='vertical', alpha=cmap_alpha)
                cax1.xaxis.set_ticks_position('top')
                cax1.xaxis.set_tick_params(pad=0)
                cbar1.ax.tick_params(labelsize=leg_fontsize)
                for n, label in enumerate(cax1.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
                fig.text(0.51, 0.85, '$\Delta$ T\n($^\circ$C)', ha='center', va='center', size=leg_fontsize+1)
            elif rcp == 'rcp45':
                norm = plt.Normalize(2, 3)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax1 = fig.add_axes([0.485, 0.41, 0.01, 0.18])
                cbar1 = fig.colorbar(sm, cax=cax1, orientation='vertical', alpha=cmap_alpha)
                cax1.xaxis.set_ticks_position('top')
                cax1.xaxis.set_tick_params(pad=0)
                cbar1.ax.tick_params(labelsize=leg_fontsize)
                for n, label in enumerate(cax1.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
            elif rcp == 'rcp85':
                norm = plt.Normalize(5, 6)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cax1 = fig.add_axes([0.485, 0.16, 0.01, 0.18])
                cbar1 = fig.colorbar(sm, cax=cax1, orientation='vertical', alpha=cmap_alpha)
                cax1.xaxis.set_ticks_position('top')
                cax1.xaxis.set_tick_params(pad=0)
                cbar1.ax.tick_params(labelsize=leg_fontsize)
                for n, label in enumerate(cax1.xaxis.get_ticklabels()):
                    if n%2 != 0:
                        label.set_visible(False)
            cmap_alpha = 1
            
            # ===== PLOT WITH CIRCLES SIZED ACCORDING TO AREA =====  
            # Degree TEMPERATURE
            x, y, z, s = [], [], [], []
            for group in groups_deg:
                vn_multimodel_mean = ds_temp_multimodel_deg[rcp][group]
                vn_multimodel_mean_plot = (
                        vn_multimodel_mean[90:101].mean() - vn_multimodel_mean[0:16].mean())
                x.append(deg_dict[group][0])
                y.append(deg_dict[group][1])
                z.append(vn_multimodel_mean_plot)
                s.append(area_init_group[group])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            s = np.array(s)
            
            # Size thresholds
            s_sizes = [2, 4, 8, 16]
            s_plot = np.array(s)
            s_plot[s <= 10] = s_sizes[0]
            s_plot[s > 10] = s_sizes[1]
            s_plot[s > 100] = s_sizes[2]
            s_plot[s > 1000] = s_sizes[3]
            
            if nrcp == 0:
                a = ax1.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 1:
                a = ax2.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 2:
                a = ax3.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
                
            # PRECIPITATION
            cmap = mpl.cm.RdYlBu
            norm = plt.Normalize(0.9, 1.1)
            x, y, z, s = [], [], [], []
            for group in groups_deg:
                vn_multimodel_mean = ds_prec_multimodel_deg[rcp][group]
                vn_multimodel_mean_plot = (
                        vn_multimodel_mean[90:101].mean() / vn_multimodel_mean[0:16].mean())
                x.append(deg_dict[group][0])
                y.append(deg_dict[group][1])
                z.append(vn_multimodel_mean_plot)
                s.append(area_init_group[group])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            s = np.array(s)
            
            # Size thresholds
            s_plot = np.array(s)
            s_plot[s <= 10] = s_sizes[0]
            s_plot[s > 10] = s_sizes[1]
            s_plot[s > 100] = s_sizes[2]
            s_plot[s > 1000] = s_sizes[3]
            
            if nrcp == 0:
                a = ax4.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 1:
                a = ax5.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
            elif nrcp == 2:
                a = ax6.scatter(x, y, c=z, cmap=cmap, norm=norm, zorder=3, s=s_plot, marker='o', 
                                edgecolor=marker_linecolor, linewidth=marker_linewidth, alpha=cmap_alpha)
                
        west = 66
        north = 47
        south = 24
        axis_fontsize = 7
        # Set the extent
        ax1.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax1.set_xticks(np.arange(70,east+1,10), cartopy.crs.PlateCarree())
        ax1.set_yticks(np.arange(30,north+1,5), cartopy.crs.PlateCarree())
        ax1.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax1.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax1.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax2.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax2.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax2.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax2.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax2.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax2.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax2.xaxis.get_ticklabels():
            label.set_visible(False)
        # Set the extent
        ax3.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax3.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
        ax3.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
        ax3.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax3.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax3.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
#        for label in ax3.xaxis.get_ticklabels()[::2]:
#            label.set_visible(False)
        
        # Set the extent
        ax4.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax4.set_xticks(np.arange(70,east+1,10), cartopy.crs.PlateCarree())
        ax4.set_yticks(np.arange(30,north+1,5), cartopy.crs.PlateCarree())
        ax4.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax4.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax4.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax4.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax4.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax4.yaxis.get_ticklabels():
            label.set_visible(False)
        
        # Set the extent
        ax5.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax5.set_xticks(np.arange(70,east+1,10), cartopy.crs.PlateCarree())
        ax5.set_yticks(np.arange(30,north+1,5), cartopy.crs.PlateCarree())
        ax5.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax5.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax5.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax5.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax5.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax5.yaxis.get_ticklabels():
            label.set_visible(False)
        
        # Set the extent
        ax6.set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
        # Label title, x, and y axes
        ax6.set_xticks(np.arange(70,east+1,10), cartopy.crs.PlateCarree())
        ax6.set_yticks(np.arange(30,north+1,5), cartopy.crs.PlateCarree())
        ax6.xaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax6.yaxis.set_tick_params(pad=0, size=2, labelsize=axis_fontsize)
        ax6.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
        ax6.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
        for label in ax6.yaxis.get_ticklabels():
            label.set_visible(False)
        
        # Add circle size legend       
        circ1 = ax1.scatter([0],[0], s=s_sizes[0], marker='o', color='grey', 
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ2 = ax1.scatter([0],[0], s=s_sizes[1], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ3 = ax1.scatter([0],[0], s=s_sizes[2], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        circ4 = ax1.scatter([0],[0], s=s_sizes[3], marker='o', color='grey',
                           edgecolor=marker_linecolor, linewidth=marker_linewidth)
        legend=fig.legend([circ1,circ2,circ3,circ4], ['1', '10', '100', '1000'], 
                          scatterpoints=1, ncol=1, loc='lower right', bbox_to_anchor=(0.895,0.07),
                          fontsize=leg_fontsize, labelspacing=0.3,
                          columnspacing=0, handletextpad=0, handlelength=1, borderpad=0.2, framealpha=0,
                          title='Area\n(km$^{2}$)', borderaxespad=0.2)
        legend.get_title().set_fontsize(str(leg_fontsize+1))
#        legend.get_frame().set_linewidth(0.5)
        
        
        # Add colorbar legend
        cmap = mpl.cm.RdYlBu
        norm = plt.Normalize(0.9, 1.1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cax4 = fig.add_axes([0.91, 0.32, 0.01, 0.49])
        cbar4 = fig.colorbar(sm, cax=cax4, orientation='vertical', alpha=cmap_alpha)
        cax4.xaxis.set_ticks_position('top')
        cax4.xaxis.set_tick_params(pad=0)
        cbar4.ax.tick_params(labelsize=leg_fontsize)
        for label in cax4.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)
        fig.text(0.94, 0.85, '$\Delta$ P\n(-)', ha='center', va='center', size=leg_fontsize+1)
        cmap_alpha = 1
        
        fig.text(0.14, 0.86, 'A', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.14, 0.605, 'B', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.14, 0.35, 'C', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.565, 0.86, 'D', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.565, 0.605, 'E', ha='center', va='center', size=12, fontweight='bold')
        fig.text(0.565, 0.35, 'F', ha='center', va='center', size=12, fontweight='bold')
            
        # Save figure
        fig_height = 7
        fig_width = 7.8
        fig.set_size_inches(fig_width,fig_height)
        figure_fn = 'temp_prec_map_' + grouping + '_multimodel_3rcps_circles.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)
        #%%
        
if option_watersheds_colored == 1:
    grouping = 'watershed'
    groups = ['Amu_Darya','Brahmaputra','Ganges','Ili','Indus','Inner_Tibetan_Plateau','Inner_Tibetan_Plateau_extended',
              'Irrawaddy','Mekong','Salween','Syr_Darya','Tarim','Yangtze']
    
    figure_fp = pygem_prms.output_sim_fp + 'figures/'
    
#    east = 60
#    west = 110
#    south = 15
#    north = 50
    east = 104
    west = 64
    south = 26
    north = 47
    xtick = 5
    ytick = 5
    xlabel = 'Longitude ($\mathregular{^{\circ}}$)'
    ylabel = 'Latitude ($\mathregular{^{\circ}}$)'
        
    # Create the projection
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':cartopy.crs.PlateCarree()},
                           gridspec_kw = {'wspace':0, 'hspace':0})
    
    # Add group and attribute of interest
    if grouping == 'rgi_region':
        group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
        group_shp_attr = 'RGI_CODE'
    elif grouping == 'watershed':
        group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
        group_shp_attr = 'watershed'
    elif grouping == 'kaab':
        group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
        group_shp_attr = 'Name'
        
    title_location = {'Syr_Darya': [71, 42],
                      'Ili': [82, 44.5],
                      'Amu_Darya': [69, 36],
                      'Tarim': [82.5, 38.5],
                      'Inner_Tibetan_Plateau_extended': [98.5, 38.2],
                      'Indus': [72, 32],
                      'Inner_Tibetan_Plateau': [86.5, 33],
                      'Yangtze': [100.7, 30.5],
                      'Ganges': [81.3, 26.6],
                      'Brahmaputra': [92.5, 26.5],
                      'Irrawaddy': [96.2, 23.8],
                      'Salween': [93.2, 31.15],
                      'Mekong': [96, 31.8],
                      'Yellow': [106.0, 36]}
    title_dict = {'Amu_Darya': 'Amu\nDarya',
                  'Brahmaputra': 'Brahma-\nputra',
                  'Ganges': 'Ganges',
                  'Ili': 'Ili',
                  'Indus': 'Indus',
                  'Inner_Tibetan_Plateau': 'Inner TP',
                  'Inner_Tibetan_Plateau_extended': 'Inner TP ext',
                  'Irrawaddy': 'Irrawaddy',
                  'Mekong': 'Mk',
                  'Salween': 'Sw',
                  'Syr_Darya': 'Syr\nDarya',
                  'Tarim': 'Tarim',
                  'Yangtze': 'Yz'}
    group_colordict = {'Amu_Darya': 'mediumblue',
                       'Brahmaputra': 'salmon',
                       'Ganges': 'lightskyblue',
                       'Ili': 'royalblue',
                       'Indus': 'darkred',
                       'Inner_Tibetan_Plateau': 'gold',
                       'Inner_Tibetan_Plateau_extended': 'navy',
                       'Irrawaddy': 'white',
                       'Mekong': 'white',
                       'Salween': 'plum',
                       'Syr_Darya':'darkolivegreen',
                       'Tarim': 'olive',
                       'Yangtze': 'orange',
                       'Yellow':'white'}
    
    # Add country borders for reference
    if grouping == 'rgi_region':
        ax.add_feature(cartopy.feature.BORDERS, alpha=0.15, linewidth=0.25)
#        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.25)
    # Set the extent
    ax.set_extent([east, 66, 24, north], cartopy.crs.PlateCarree())    
    # Label title, x, and y axes
    ax.set_xticks(np.arange(70,100+1,10), cartopy.crs.PlateCarree())
    ax.set_yticks(np.arange(30,45+1,5), cartopy.crs.PlateCarree())
#        ax.set_xlabel(xlabel, size=10, labelpad=0)
#        ax.set_ylabel(ylabel, size=10, labelpad=0)
    ax.xaxis.set_tick_params(pad=0, size=2, labelsize=8)
    ax.yaxis.set_tick_params(pad=0, size=2, labelsize=8)
    ax.yaxis.set_major_formatter(EngFormatter(unit=u"°N"))
    ax.xaxis.set_major_formatter(EngFormatter(unit=u"°E"))
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    
    for rec in group_shp.records():
        if rec.attributes[group_shp_attr] in groups:
            group = rec.attributes[group_shp_attr]
#                ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', edgecolor='Black', 
#                                  linewidth=0.5, zorder=3)
            ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor=group_colordict[group], 
                              edgecolor='Black', linewidth=0.5, alpha=0.5, zorder=3)
            if group not in ['Yellow', 'Irrawaddy', 'Mekong']:
                ax.text(title_location[rec.attributes[group_shp_attr]][0], 
                        title_location[rec.attributes[group_shp_attr]][1], 
                        title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', size=9, 
                        zorder=4)
    # Save figure
    fig.set_size_inches(3.5,6)
    figure_fn = 'watershed_colored_map.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)
        
        #%%
#        # Create the projection
#        fig, ax = plt.subplots(2, 1, subplot_kw={'projection':cartopy.crs.PlateCarree()},
#                               gridspec_kw = {'wspace':0, 'hspace':0})
#        
#        # Add group and attribute of interest
#        if grouping == 'rgi_region':
#            group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
#            group_shp_attr = 'RGI_CODE'
#        elif grouping == 'watershed':
#            group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
#            group_shp_attr = 'watershed'
#        elif grouping == 'kaab':
#            group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
#            group_shp_attr = 'Name'
#            
#        title_location = {'Syr_Darya': [70.2, 43.1],
#                          'Ili': [83.6, 45.5],
#                          'Amu_Darya': [66.7, 35.5],
#                          'Tarim': [83.0, 39.2],
#                          'Inner_Tibetan_Plateau_extended': [100, 40],
#                          'Indus': [70.7, 31.9],
#                          'Inner_Tibetan_Plateau': [85, 32.4],
#                          'Yangtze': [106.0, 29.8],
#                          'Ganges': [81.3, 26.6],
#                          'Brahmaputra': [92.0, 26],
#                          'Irrawaddy': [96.2, 23.8],
#                          'Salween': [98.5, 20.8],
#                          'Mekong': [103.8, 17.5],
#                          'Yellow': [106.0, 36]}
#        title_dict = {'Amu_Darya': 'Amu\nDarya',
#                      'Brahmaputra': 'Brahmaputra',
#                      'Ganges': 'Ganges',
#                      'Ili': 'Ili',
#                      'Indus': 'Indus',
#                      'Inner_Tibetan_Plateau': 'Inner TP',
#                      'Inner_Tibetan_Plateau_extended': 'Inner TP ext',
#                      'Irrawaddy': 'Irrawaddy',
#                      'Mekong': 'Mekong',
#                      'Salween': 'Salween',
#                      'Syr_Darya': 'Syr\nDarya',
#                      'Tarim': 'Tarim',
#                      'Yangtze': 'Yangtze'}
#        
#        for n in [0,1]:
#            # Add country borders for reference
#            if grouping == 'rgi_region':
#                ax.add_feature(cartopy.feature.BORDERS, alpha=0.15)
#            ax[n].add_feature(cartopy.feature.COASTLINE)
#            # Set the extent
#            ax[n].set_extent([east, west, south, north], cartopy.crs.PlateCarree())    
#            # Label title, x, and y axes
#            ax[n].set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
#            ax[n].set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
#            if n == 1:
#                ax[n].set_xlabel(xlabel, size=12)
#            ax[n].set_ylabel(ylabel, size=12)
#            
#            for rec in group_shp.records():
#                if rec.attributes[group_shp_attr] in groups:
#                    group = rec.attributes[group_shp_attr]
#                    ax[n].add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', edgecolor='Black', 
#                                      zorder=3)
#                    if group in ['Indus', 'Ganges', 'Tarim', 'Syr_Darya', 'Amu_Darya']:
#                        ax[n].text(title_location[rec.attributes[group_shp_attr]][0], 
#                                title_location[rec.attributes[group_shp_attr]][1], 
#                                title_dict[rec.attributes[group_shp_attr]], horizontalalignment='center', size=10, 
#                                zorder=4)
#        
#            colorbar_dict = {'volume_norm':[0,1],
#                             'runoff_glac_monthly':[2015,2100]}
#            cmap = mpl.cm.RdYlBu
#            norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])
#              
#            # Add colorbar
#            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#            sm._A = []
#            cbar = plt.colorbar(sm, ax=ax[n], fraction=0.015, pad=0.02)
#            
##            if n == 0:
#            # Degree peakwater
#            x, y, z = [], [], []
#            for group in groups_deg:
#                if n == 0:
#                    vn_multimodel_mean = ds_multimodel_deg[rcp][group]
#                    peakwater_yr, peakwater_chg, runoff_chg = (
#                            peakwater(vn_multimodel_mean[time_idx_start:time_idx_end], 
#                                      time_values_annual[time_idx_start:time_idx_end], peakwater_Nyears))
#                    z.append(peakwater_yr)
#                elif n == 1:
#                    vn_multimodel_std = ds_multimodel_std_deg[rcp][group]
#                    z.append(vn_multimodel_std)
#                x.append(deg_dict[group][0])
#                y.append(deg_dict[group][1])
#            x = np.array(x)
#            y = np.array(y)
#            
#            lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
#            lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
#            x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
#            y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
#            z_array = np.zeros((len(y_adj), len(x_adj)))
#            z_array[z_array==0] = np.nan
#            for i in range(len(z)):
#                row_idx = int((y[i] - y.min()) / degree_size)
#                col_idx = int((x[i] - x.min()) / degree_size)
#                z_array[row_idx, col_idx] = z[i]
#            ax[n].pcolormesh(lons, lats, z_array, cmap='RdYlBu', norm=norm, zorder=2)
#                
##            if n == 0:
##                # Degree peakwater
##                x, y, z = [], [], []
##                for group in groups_deg:
##                    vn_multimodel_mean = ds_multimodel_deg[rcp][group]
##                    peakwater_yr, peakwater_chg, runoff_chg = (
##                            peakwater(vn_multimodel_mean[time_idx_start:time_idx_end], 
##                                      time_values_annual[time_idx_start:time_idx_end], peakwater_Nyears))
##                    x.append(deg_dict[group][0])
##                    y.append(deg_dict[group][1])
##                    z.append(peakwater_yr)
##                x = np.array(x)
##                y = np.array(y)
##                
##                lons = np.arange(x.min(), x.max() + 2 * degree_size, degree_size)
##                lats = np.arange(y.min(), y.max() + 2 * degree_size, degree_size)
##                x_adj = np.arange(x.min(), x.max() + 1 * degree_size, degree_size) - x.min()
##                y_adj = np.arange(y.min(), y.max() + 1 * degree_size, degree_size) - y.min()
##                z_array = np.zeros((len(y_adj), len(x_adj)))
##                z_array[z_array==0] = np.nan
##                for i in range(len(z)):
##                    row_idx = int((y[i] - y.min()) / degree_size)
##                    col_idx = int((x[i] - x.min()) / degree_size)
##                    z_array[row_idx, col_idx] = z[i]
##                ax[n].pcolormesh(lons, lats, z_array, cmap='RdYlBu', norm=norm, zorder=2)
#    
#        # Save figure
#        fig.set_size_inches(6,4)
#        figure_fn = 'peakwater_map_' + grouping + '_multimodel_' + rcp +  '.png'
#        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
        #%%
if option_cmip5_mb_vs_climate == 1:
    vns_heatmap = ['massbaltotal_glac_monthly', 'temp_glac_monthly', 'prec_glac_monthly']
    figure_fp = pygem_prms.output_sim_fp + 'figures/'
    
    startyear=2000
    endyear=2100
    
    option_plot_multimodel = 1
    option_plot_single = 0

    area_cutoffs = [1]
    ref_startyear = 2000
    ref_endyear = 2015
    
    plt_startyear = 2015
    plt_endyear = 2100

    multimodel_linewidth = 2
    alpha=0.2
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0, option_wateryear=1)

    # Glacier and grouped annual specific mass balance and mass change
    for rcp in rcps:

        for ngcm, gcm_name in enumerate(gcm_names):
            
            print(rcp, gcm_name)
            
            # Climate data
            temp_glac_all, prectotal_glac_all, elev_glac_all = retrieve_gcm_data(gcm_name, rcp, main_glac_rgi)
            
            group_glacidx = {}
            vol_group_dict = {}
            temp_group_dict = {}
            prectotal_group_dict = {}
            
            # Extract data from netcdf
            for region in regions:
                       
                # Load datasets
                ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba1_100sets_2000_2100--subset.nc')
                ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
                df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                               str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]

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
                else:
                    # Volume
                    vol_glac_all = np.concatenate((vol_glac_all, ds['volume_glac_annual'].values[:,:,0]), axis=0)
                    vol_glac_std_all = np.concatenate((vol_glac_std_all, ds['volume_glac_annual'].values[:,:,1]),axis=0)
                    # Area
                    area_glac_all = np.concatenate((area_glac_all, ds['area_glac_annual'].values[:,:,0]), axis=0)
                    area_glac_std_all = np.concatenate((area_glac_std_all, ds['area_glac_annual'].values[:,:,1]),axis=0)
                    # Mass balance
                    mb_glac_all = np.concatenate((mb_glac_all, ds['massbaltotal_glac_monthly'].values[:,:,0]), axis=0)
                    mb_glac_std_all = np.concatenate((mb_glac_std_all, ds['massbaltotal_glac_monthly'].values[:,:,1]),
                                                      axis=0)
    
                ds.close()
               
            # Annual Mass Balance
            mb_glac_all_annual = gcmbiasadj.annual_sum_2darray(mb_glac_all)
            # mask values where volume is zero
            mb_glac_all_annual[vol_glac_all[:,:-1] == 0] = np.nan

            # Annual Temperature, Precipitation, and Accumulation
            temp_glac_all_annual = gcmbiasadj.annual_avg_2darray(temp_glac_all)
            prectotal_glac_all_annual = gcmbiasadj.annual_sum_2darray(prectotal_glac_all)
            
            
            # Remove RGIIds from main_glac_rgi that are not in the model runs
            if ngcm == 0:
                rgiid_df = list(df_all.RGIId.values)
                rgiid_all = list(main_glac_rgi.RGIId.values)
                rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
                main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
                main_glac_rgi.reset_index(inplace=True, drop=True)
            
            # Groups
            for ngroup, group in enumerate(groups):
                # Select subset of data
                group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                group_glacidx[group] = group_glac_indices
                
                # Regional Volume, and Area-weighted Temperature and Precipitation
                vol_group_all = vol_glac_all[group_glac_indices,:].sum(axis=0)
                
                temp_group_all = ((temp_glac_all_annual[group_glac_indices,:] * 
                                   area_glac_all[group_glac_indices,:][:,0][:,np.newaxis]).sum(axis=0) / 
                                  area_glac_all[group_glac_indices,:][:,0].sum())
                prectotal_group_all = ((prectotal_glac_all_annual[group_glac_indices,:] * 
                                        area_glac_all[group_glac_indices,:][:,0][:,np.newaxis]).sum(axis=0) / 
                                       area_glac_all[group_glac_indices,:][:,0].sum())
                    
                # Expand dimensions for multi-model calculations
                vol_group_all = np.expand_dims(vol_group_all, axis=1)
                temp_group_all = np.expand_dims(temp_group_all, axis=1)
                prectotal_group_all = np.expand_dims(prectotal_group_all, axis=1)

                vol_group_dict[group] = vol_group_all
                temp_group_dict[group] = temp_group_all
                prectotal_group_dict[group] = prectotal_group_all
                
            # Expand dimensions for multi-model calculations
            vol_glac_all = np.expand_dims(vol_glac_all, axis=2)
            mb_glac_all_annual = np.expand_dims(mb_glac_all_annual, axis=2)
            temp_glac_all_annual = np.expand_dims(temp_glac_all_annual, axis=2)
            prectotal_glac_all_annual = np.expand_dims(prectotal_glac_all_annual, axis=2)
            
            # ===== MULTI-MODEL =====
            if ngcm == 0:
                vol_glac_all_annual_multimodel = vol_glac_all
                mb_glac_all_annual_multimodel = mb_glac_all_annual
                temp_glac_all_annual_multimodel = temp_glac_all_annual
                prectotal_glac_all_annual_multimodel = prectotal_glac_all_annual
                
                vol_group_dict_multimodel = vol_group_dict
                temp_group_dict_multimodel = temp_group_dict
                prectotal_group_dict_multimodel = prectotal_group_dict
                
            else:
                vol_glac_all_annual_multimodel = np.append(vol_glac_all_annual_multimodel, vol_glac_all, axis=2)
                mb_glac_all_annual_multimodel = np.append(mb_glac_all_annual_multimodel, 
                                                          mb_glac_all_annual, axis=2)
                temp_glac_all_annual_multimodel = np.append(temp_glac_all_annual_multimodel, 
                                                            temp_glac_all_annual, axis=2)
                prectotal_glac_all_annual_multimodel = np.append(prectotal_glac_all_annual_multimodel, 
                                                                 prectotal_glac_all_annual, axis=2)
                
                for ngroup, group in enumerate(groups):
                    vol_group_dict_multimodel[group] = np.append(vol_group_dict_multimodel[group], 
                                                                 vol_group_dict[group], axis=1)
                    temp_group_dict_multimodel[group] = np.append(temp_group_dict_multimodel[group], 
                                                                  temp_group_dict[group], axis=1)
                    prectotal_group_dict_multimodel[group] = np.append(prectotal_group_dict_multimodel[group], 
                                                                       prectotal_group_dict[group], axis=1)
    
    
        #%%
        # Pickle data
        vol_fn_pkl = figure_fp + grouping + '_vol_annual_' + rcp + '_' + str(len(gcm_names)) + 'gcms.pkl'
        temp_fn_pkl = figure_fp + grouping + '_temp_annual_' + rcp + '_' + str(len(gcm_names)) + 'gcms.pkl'
        prec_fn_pkl = figure_fp + grouping + '_prec_annual_' + rcp + '_' + str(len(gcm_names)) + 'gcms.pkl'
        
        pickle_data(vol_fn_pkl, vol_group_dict_multimodel)
        pickle_data(temp_fn_pkl, temp_group_dict_multimodel)
        pickle_data(prec_fn_pkl, prectotal_group_dict_multimodel)
        
        group_vol_remain = [vol_group_dict_multimodel[group].mean(axis=1)[-1] / 
                            vol_group_dict_multimodel[group].mean(axis=1)[0] for group in groups]
        group_tempchange = [temp_group_dict_multimodel[group].mean(axis=1)[-10:].mean() - 
                            temp_group_dict_multimodel[group].mean(axis=1)[:15].mean() for group in groups]
        group_precchange = [prectotal_group_dict_multimodel[group].mean(axis=1)[-10:].mean() / 
                            prectotal_group_dict_multimodel[group].mean(axis=1)[:15].mean() for group in groups]

        fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(3,3), gridspec_kw = {'wspace':0.3, 'hspace':0})

        # All glaciers
        x = group_tempchange
        y = group_vol_remain
        ax[0,0].scatter(x, y, s=15, marker='.', c='k')
        # Line of best fit
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax[0,0].plot(x,p(x),"k", linewidth=1)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        ax[0,0].text(0.95, 0.95, '$\mathregular{R^{2}} = $' + str(np.round(r_value**2,2)), va='center', ha='right', 
                     size=10, transform=ax[0,0].transAxes)
        print('Temp vs. Vol Remain', np.round(slope,2), np.round(intercept,2), np.round(r_value**2,2), np.round(p_value,2))
        
        x = group_precchange
        y = group_vol_remain
        ax[0,1].scatter(x, y, s=15, marker='.', c='k')
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax[0,1].plot(x,p(x),"k", linewidth=1)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        ax[0,1].text(0.95, 0.95, '$\mathregular{R^{2}} = $' + str(np.round(r_value**2,2)), va='center', ha='right', 
                     size=10, transform=ax[0,1].transAxes)
        print('Prec vs. Vol Remain', np.round(slope,2), np.round(intercept,2), np.round(r_value**2,2), np.round(p_value,2))
        
        x = group_tempchange
        y = group_precchange
        ax[0,2].scatter(x, y, s=15, marker='.', c='k')
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax[0,2].plot(x,p(x),"k", linewidth=1)
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        ax[0,2].text(0.95, 0.95, '$\mathregular{R^{2}} = $' + str(np.round(r_value**2,2)), va='center', ha='right', 
                     size=10, transform=ax[0,2].transAxes)
        print('Temp vs. Prec', np.round(slope,2), np.round(intercept,2), np.round(r_value**2,2), np.round(p_value,2))
        
#        a = ax[0,0].scatter(mb_glac_all_annual_multimodel_mean, temp_norm, 
#                                c=cal_data_plot['year'].values, cmap=cmap, norm=norm, zorder=3, s=15,
#                                marker='D')
#        a.set_facecolor('none')
#        b.set_facecolor('none')
#        ymin = -1.25
#        ymax = 0.6
#        xmin = -1.25
#        xmax = 0.6
#        ax[0,nplot].set_xlim(xmin,xmax)
#        ax[0,nplot].set_ylim(ymin,ymax)
#        ax[0,nplot].plot([np.min([xmin,ymin]),np.max([xmax,ymax])], [np.min([xmin,ymin]),np.max([xmax,ymax])], 
#                         color='k', linewidth=0.25, zorder=1)
#        ax[0,nplot].yaxis.set_ticks(np.arange(-1, ymax+0.1, 0.5))
#        ax[0,nplot].xaxis.set_ticks(np.arange(-1, xmax+0.11, 0.5))
#        
#        ax[0,nplot].set_ylabel('$\mathregular{B_{mod}}$ (m w.e. $\mathregular{a^{-1}}$)', labelpad=0, size=12)
#        ax[0,nplot].set_xlabel('$\mathregular{B_{geo}}$ (m w.e. $\mathregular{a^{-1}}$)\n(WGMS, 2017)', labelpad=0, size=12)
#        # Add text
#        ax[0,nplot].text(0.05, 0.95, 'E', va='center', size=12, fontweight='bold', transform=ax[0,nplot].transAxes)
#        ax[0,nplot].text(0.7, 0.1, 'n=' + str(cal_data_plot.shape[0]) + '\nglaciers=' + 
#                         str(cal_data_plot.glacno.unique().shape[0]), va='center', ha='center', size=12,
#                         transform=ax[0,nplot].transAxes)
        
        # Add title
        #ax[0,nplot].set_title('Mass Balance (m w.e.)', size=12)
    
#        # Add colorbar
#        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#        sm._A = []
#        fig.subplots_adjust(right=0.9)
#        cbar_ax = fig.add_axes([0.92, 0.16, 0.03, 0.67])
#        cbar = fig.colorbar(sm, cax=cbar_ax)
        
        # Save figure
        fig.set_size_inches(6.5,2)
        figure_fn = 'mb_vs_climate_' + rcp + '_' + str(len(gcm_names)) + 'gcms.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    #%%


if option_cmip5_heatmap_w_volchange == 1:
    netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_subset/'  
    grouping = 'himap'
    
    vns_heatmap = ['massbaltotal_glac_monthly', 'temp_glac_monthly', 'prec_glac_monthly']
    figure_fp = pygem_prms.output_sim_fp + 'figures/'
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    
    rcps = ['rcp60']
    
    startyear=2000
    endyear=2100
    
    option_plot_multimodel = 1
    option_plot_single = 0

    area_cutoffs = [1]
    ref_startyear = 2000
    ref_endyear = 2015
    
    plt_startyear = 2015
    plt_endyear = 2100

    multimodel_linewidth = 2
    alpha=0.2
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    if grouping == 'himap':
        group_order = [11,17,1,2,18,4,10,7,15,3,12,20,8,0,19,5,21,14,13,16,9,6]
        groups = [x for _,x in sorted(zip(group_order,groups))]
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0, option_wateryear=1)

#%%
    # Glacier and grouped annual specific mass balance and mass change
    for rcp in rcps:
        
        if rcp == 'rcp60':
            print('\nIF RCP6.0 MAKE SURE THAT ALL GCMS HAVE RCP60\n')

        for ngcm, gcm_name in enumerate(gcm_names):
            
            print(rcp, gcm_name)
            
            group_glacidx = {}
            vol_group_dict = {}
            temp_group_dict = {}
            prectotal_group_dict = {}
            
            # Extract data from netcdf
            for region in regions:
                       
                try:
                    # Load datasets
                    ds_fn = ('R' + str(region) + '--all--' + gcm_name + '_' + rcp + 
                             '_c2_ba1_100sets_2000_2100--subset.nc')
                    ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                    df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
                    df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                                   str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
                    skip_gcm = 0
                except:
                    skip_gcm = 1
                    print('Skip', gcm_name, rcp, region)
                        
                if skip_gcm == 0:
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
                        df_all = df
                    else:
                        # Volume
                        vol_glac_all = np.concatenate((vol_glac_all, ds['volume_glac_annual'].values[:,:,0]), axis=0)
                        vol_glac_std_all = np.concatenate((vol_glac_std_all, ds['volume_glac_annual'].values[:,:,1]),axis=0)
                        # Area
                        area_glac_all = np.concatenate((area_glac_all, ds['area_glac_annual'].values[:,:,0]), axis=0)
                        area_glac_std_all = np.concatenate((area_glac_std_all, ds['area_glac_annual'].values[:,:,1]),axis=0)
                        # Mass balance
                        mb_glac_all = np.concatenate((mb_glac_all, ds['massbaltotal_glac_monthly'].values[:,:,0]), axis=0)
                        mb_glac_std_all = np.concatenate((mb_glac_std_all, ds['massbaltotal_glac_monthly'].values[:,:,1]),
                                                          axis=0)
                        df_all = pd.concat([df_all, df], axis=0)
        
                    ds.close()
                
               
            # Remove RGIIds from main_glac_rgi that are not in the model runs
            if ngcm == 0:
                rgiid_df = list(df_all.RGIId.values)
                rgiid_all = list(main_glac_rgi.RGIId.values)
                rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
                main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
                main_glac_rgi.reset_index(inplace=True, drop=True)
            
            # Annual Mass Balance
            mb_glac_all_annual = gcmbiasadj.annual_sum_2darray(mb_glac_all)
            # mask values where volume is zero
            mb_glac_all_annual[vol_glac_all[:,:-1] == 0] = np.nan

            # Annual Temperature, Precipitation, and Accumulation
            temp_glac_all, prectotal_glac_all, elev_glac_all = retrieve_gcm_data(gcm_name, rcp, main_glac_rgi)
            temp_glac_all_annual = gcmbiasadj.annual_avg_2darray(temp_glac_all)
            prectotal_glac_all_annual = gcmbiasadj.annual_sum_2darray(prectotal_glac_all)
            
            # Groups
            for ngroup, group in enumerate(groups):
                # Select subset of data
                group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                group_glacidx[group] = group_glac_indices
                
                # Regional Volume, and Area-weighted Temperature and Precipitation
                vol_group_all = vol_glac_all[group_glac_indices,:].sum(axis=0)
                
                temp_group_all = ((temp_glac_all_annual[group_glac_indices,:] * 
                                   area_glac_all[group_glac_indices,:][:,0][:,np.newaxis]).sum(axis=0) / 
                                  area_glac_all[group_glac_indices,:][:,0].sum())
                prectotal_group_all = ((prectotal_glac_all_annual[group_glac_indices,:] * 
                                        area_glac_all[group_glac_indices,:][:,0][:,np.newaxis]).sum(axis=0) / 
                                       area_glac_all[group_glac_indices,:][:,0].sum())
                    
                # Expand dimensions for multi-model calculations
                vol_group_all = np.expand_dims(vol_group_all, axis=1)
                temp_group_all = np.expand_dims(temp_group_all, axis=1)
                prectotal_group_all = np.expand_dims(prectotal_group_all, axis=1)

                vol_group_dict[group] = vol_group_all
                temp_group_dict[group] = temp_group_all
                prectotal_group_dict[group] = prectotal_group_all
                
            # Expand dimensions for multi-model calculations
            mb_glac_all_annual = np.expand_dims(mb_glac_all_annual, axis=2)
            temp_glac_all_annual = np.expand_dims(temp_glac_all_annual, axis=2)
            prectotal_glac_all_annual = np.expand_dims(prectotal_glac_all_annual, axis=2)
            
            # ===== MULTI-MODEL =====
            if ngcm == 0:
                mb_glac_all_annual_multimodel = mb_glac_all_annual
                temp_glac_all_annual_multimodel = temp_glac_all_annual
                prectotal_glac_all_annual_multimodel = prectotal_glac_all_annual
                
                vol_group_dict_multimodel = vol_group_dict
                temp_group_dict_multimodel = temp_group_dict
                prectotal_group_dict_multimodel = prectotal_group_dict
                
            else:
                mb_glac_all_annual_multimodel = np.append(mb_glac_all_annual_multimodel, 
                                                          mb_glac_all_annual, axis=2)
                temp_glac_all_annual_multimodel = np.append(temp_glac_all_annual_multimodel, 
                                                            temp_glac_all_annual, axis=2)
                prectotal_glac_all_annual_multimodel = np.append(prectotal_glac_all_annual_multimodel, 
                                                                 prectotal_glac_all_annual, axis=2)
                
                for ngroup, group in enumerate(groups):
                    vol_group_dict_multimodel[group] = np.append(vol_group_dict_multimodel[group], 
                                                                 vol_group_dict[group], axis=1)
                    temp_group_dict_multimodel[group] = np.append(temp_group_dict_multimodel[group], 
                                                                  temp_group_dict[group], axis=1)
                    prectotal_group_dict_multimodel[group] = np.append(prectotal_group_dict_multimodel[group], 
                                                                       prectotal_group_dict[group], axis=1)
                    #%%
##%%        
#        def plot_heatmap(rcp, mb_glac_all_annual, temp_glac_all_annual, prectotal_glac_all_annual, 
#                         vol_group_dict, temp_group_dict, prectotal_group_dict, gcm_name=None):
        cmap_dict = {'massbaltotal_glac_monthly':'RdYlBu',
                     'temp_glac_monthly':'RdYlBu_r',
                     'prec_glac_monthly':'RdYlBu'}
#        norm_dict = {'massbaltotal_glac_monthly':[-2,0,1],
#                     'temp_glac_monthly':plt.Normalize(0,6),
#                     'prec_glac_monthly':plt.Normalize(0.9,1.15)}
#        norm_dict = {'massbaltotal_glac_monthly':[-1.5,0,0.5],
#                     'temp_glac_monthly':[0,1.5,4],
#                     'prec_glac_monthly':[0.9,1,1.15]}
        norm_dict = {'massbaltotal_glac_monthly':{'rcp26':[-1.5,-0.75,0],
                                                  'rcp45':[-1.5,-0.75,0],
                                                  'rcp60':[-1.5,-0.75,0],
                                                  'rcp85':[-1.5,-0.75,0],},
                     'temp_glac_monthly':{'rcp26':[0,1.5,2.5], 
                                          'rcp45':[0,1.5,2.5], 
                                          'rcp60':[0,2,3.5],
                                          'rcp85':[0,2.5,5.5]},
                     'prec_glac_monthly':{'rcp26':[0.9,1,1.1], 
                                          'rcp45':[0.9,1,1.1], 
                                          'rcp60':[0.9,1,1.1],
                                          'rcp85':[0.9,1,1.1]}}
        ylabel_dict = {'massbaltotal_glac_monthly':'Normalized Volume\n(-)',
                       'temp_glac_monthly':'Temperature Change\n($^\circ$C)',
                       'prec_glac_monthly':'Precipitation Change\n(%)'}
        line_label_dict = {'massbaltotal_glac_monthly':'Regional mass (-)',
                           'temp_glac_monthly':'Regional mean',
                           'prec_glac_monthly':'Regional mean'}
        line_loc_dict = {'massbaltotal_glac_monthly':(0.125,0.8825),
                         'temp_glac_monthly':(0.41,0.8825),
                         'prec_glac_monthly':(0.69,0.8825)}
        colorbar_dict = {'massbaltotal_glac_monthly':(0.23, 1.02, 'Mass Balance\n(mwea)', 
                                                      [0.14, 0.92, 0.215, 0.02]),
                         'temp_glac_monthly':(0.51, 1.02, 'Temperature Change\n($^\circ$C)',
                                             [0.125, 0.92, 0.215, 0.02]),
                         'prec_glac_monthly':(0.79, 1.02, 'Precipitation Change\n(-)',
                                             [0.405, 0.92, 0.215, 0.02])
                         }
        
        class MidpointNormalize(mpl.colors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                mpl.colors.Normalize.__init__(self, vmin, vmax, clip)
        
            def __call__(self, value, clip=None):
                # Note that I'm ignoring clipping and other edge cases here.
                result, is_scalar = self.process_value(value)
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)

        for area_cutoff in area_cutoffs:
            # Plot the normalized volume change for each region, along with the mass balances
            fig, ax = plt.subplots(len(groups), len(vns_heatmap), squeeze=False, sharex=True, sharey=False, 
                                   gridspec_kw = {'wspace':0.3, 'hspace':0.15})
            fig.subplots_adjust(top=0.94)
        
            for nvar, vn_heatmap in enumerate(vns_heatmap):
                
                cmap = cmap_dict[vn_heatmap]
#                norm = norm_dict[vn_heatmap]   
#                norm = plt.Normalize(norm_dict[vn_heatmap][0],norm_dict[vn_heatmap][2])
                norm = MidpointNormalize(midpoint=norm_dict[vn_heatmap][rcp][1], vmin=norm_dict[vn_heatmap][rcp][0], 
                                         vmax=norm_dict[vn_heatmap][rcp][2])
                
                for ngroup, group in enumerate(groups):
                    
                    ref_idx_start = np.where(time_values_annual == ref_startyear)[0][0]
                    ref_idx_end = np.where(time_values_annual == ref_endyear)[0][0] + 1
                    
                    plt_idx_start = np.where(time_values_annual == plt_startyear)[0][0]
                    plt_idx_end = np.where(time_values_annual == plt_endyear)[0][0] + 1

                    if vn_heatmap == 'massbaltotal_glac_monthly':
                        zmesh = mb_glac_all_annual_multimodel.mean(axis=2)
                        var_line = (vol_group_dict_multimodel[group][:-1].mean(axis=1) / 
                                    vol_group_dict_multimodel[group][16].mean())
                        if ngroup == 0:
                            print('\nhard coded 2015 start date\n')
                        
                    elif vn_heatmap == 'temp_glac_monthly':
                        zmesh_raw = temp_glac_all_annual_multimodel.mean(axis=2)
                        zmesh_norm = zmesh_raw[:,ref_idx_start:ref_idx_end].mean(axis=1)[:,np.newaxis]
                        zmesh = zmesh_raw - zmesh_norm
                        var_line = (
                                temp_group_dict_multimodel[group].mean(axis=1) - 
                                temp_group_dict_multimodel[group].mean(axis=1)[ref_idx_start:ref_idx_end].mean())
                        
                    elif vn_heatmap == 'prec_glac_monthly':
                        zmesh_raw = prectotal_glac_all_annual_multimodel.mean(axis=2)
                        zmesh_norm = zmesh_raw[:,ref_idx_start:ref_idx_end].mean(axis=1)[:,np.newaxis]
                        zmesh = zmesh_raw / zmesh_norm
                        var_line = (
                                prectotal_group_dict_multimodel[group].mean(axis=1) /
                                prectotal_group_dict_multimodel[group].mean(axis=1)[ref_idx_start:ref_idx_end].mean())
                        
                        
                    # HEATMAP (only glaciers greater than area threshold)
                    glac_idx4mesh = np.where(main_glac_rgi.loc[group_glacidx[group],'Area'] > area_cutoff)[0]
                    z = zmesh[glac_idx4mesh,plt_idx_start:plt_idx_end]
                    x = time_values_annual[plt_idx_start:plt_idx_end]
                    y = np.array(range(len(glac_idx4mesh)))
                    # plot
                    ax[ngroup,nvar].pcolormesh(x, y, z, cmap=cmap, norm=norm, zorder=1)
                    if nvar == 0:
#                        y_spacing = int((y.max()/2+20)/20)*20
#                        ax[ngroup,nvar].yaxis.set_ticks(np.arange(0,y.max(),y_spacing))
                        y_spacing = int(np.round((y.max() / 4)/10))*10
                        y_spacing_minor = int(np.ceil(y.max()/2))
                        ax[ngroup,nvar].yaxis.set_ticks(np.arange(int(y_spacing), y.max(), 2*y_spacing))
                        ax[ngroup,nvar].yaxis.set_minor_locator(MultipleLocator(y_spacing_minor))
                    else:
                        y_spacing = int(np.round((y.max() / 4)/10))*10
                        y_spacing_minor = int(np.ceil(y.max()/2))
                        ax[ngroup,nvar].yaxis.set_ticks(np.arange(int(y_spacing), y.max(), 2*y_spacing))
                        ax[ngroup,nvar].yaxis.set_ticklabels([])
                        ax[ngroup,nvar].yaxis.set_minor_locator(MultipleLocator(y_spacing_minor))
                    
                         
                    # LINE
                    ax2_line_color ='black'
                    ax2_line_width = 1
                    ax2 = ax[ngroup,nvar].twinx()
                    if vn_heatmap in ['temp_glac_monthly', 'prec_glac_monthly']:
                        ax2.plot(time_values_annual[plt_idx_start:plt_idx_end], var_line[plt_idx_start:plt_idx_end], 
                                 color=ax2_line_color, linewidth=ax2_line_width, label=str(group), zorder=2)
                    else:
                        ax2.plot(time_values_annual[plt_idx_start:plt_idx_end], var_line[plt_idx_start:plt_idx_end], 
                                 color=ax2_line_color, linewidth=1.5, label=str(group), zorder=2)
#                    ax2.plot(time_values_annual[plt_idx_start:plt_idx_end], var_line[plt_idx_start:plt_idx_end], 
#                             color='gray', linewidth=1, label=str(group), zorder=2)
                    ax2.tick_params(axis='y', colors=ax2_line_color)
                    ax2.set_zorder(2)
                    ax2.patch.set_visible(False)
                    ax2.set_xlim([time_values_annual[plt_idx_start:plt_idx_end][0], 
                                  time_values_annual[plt_idx_start:plt_idx_end][-1]])
                    ax2.xaxis.set_ticks(np.arange(2020,2101,30))
                    ax2.xaxis.set_minor_locator(MultipleLocator(10))
                    
                    if vn_heatmap == 'massbaltotal_glac_monthly':
                        ax2.set_ylim([0,1])
                        ax2.yaxis.set_ticks(np.arange(0,1.05,0.5))
                        ax2.set_yticklabels(['', '0.5', '1'])
                        ax2.yaxis.set_minor_locator(MultipleLocator(0.25))
                    elif vn_heatmap == 'temp_glac_monthly':
                        if rcp == 'rcp26':
                            ax2.set_ylim([0, 2])
                            ax2.yaxis.set_ticks([0.5,1.5])
                            ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
                        elif rcp in ['rcp45', 'rcp60']:
                            ax2.set_ylim([0, 4])
                            ax2.yaxis.set_ticks([1,3])
                            ax2.yaxis.set_minor_locator(MultipleLocator(1))
                        elif rcp == 'rcp85':
                            ax2.set_ylim([0, 7])
                            ax2.yaxis.set_ticks([2,5])
                            ax2.yaxis.set_minor_locator(MultipleLocator(1))
                    elif vn_heatmap == 'prec_glac_monthly':
                        ax2.set_ylim([0.9, 1.3])
                        ax2.yaxis.set_ticks([1,1.2])
                        ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
                        
#                    if nvar == 0:
#                        ax[ngroup,nvar].text(0.98,0.6, title_dict[group], horizontalalignment='right', 
#                          transform=ax[ngroup,nvar].transAxes, size=10, zorder=3)
                    if nvar == 1:
                        ax[ngroup,nvar].text(0.5,0.6, title_dict[group], horizontalalignment='center', 
                          transform=ax[ngroup,nvar].transAxes, size=10, zorder=3)
                        

#                # Line legend
#                line = Line2D([0,1],[0,1], linestyle='-', color=ax2_line_color, linewidth=ax2_line_width)
#                leg_line = [line]
#                leg_label = [line_label_dict[vn_heatmap]]
#                leg = fig.legend(leg_line, leg_label, loc='upper left', 
#                                 bbox_to_anchor=line_loc_dict[vn_heatmap], 
#                                 handlelength=1, handletextpad=0.5, borderpad=0, frameon=False)
#                for text in leg.get_texts():
#                    text.set_color(ax2_line_color)
            
            # Mass balance colorbar
#            fig.text(0.23, 1.02, 'Mass Balance\n(m w.e. yr$^{-1}$)', ha='center', va='center', size=12)
            fig.text(0.23, 1.015, 'Mass Balance\n', ha='center', va='center', size=12)
            fig.text(0.23, 1.005, '(m w.e. yr$^{-1}$)', ha='center', va='center', size=10)
            cmap=cmap_dict['massbaltotal_glac_monthly']
#            norm=norm_dict['massbaltotal_glac_monthly']
#            norm = MidpointNormalize(midpoint=0, vmin=-2, vmax=1)
            norm = MidpointNormalize(midpoint=norm_dict['massbaltotal_glac_monthly'][rcp][1], 
                                     vmin=norm_dict['massbaltotal_glac_monthly'][rcp][0], 
                                     vmax=norm_dict['massbaltotal_glac_monthly'][rcp][2])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cax = plt.axes([0.125, 0.95, 0.215, 0.015])
            plt.colorbar(sm, cax=cax, orientation='horizontal')
            cax.xaxis.set_ticks_position('top')
            for n, label in enumerate(cax.xaxis.get_ticklabels()):
                if n%2 != 0:
                    label.set_visible(False)
            
            # Temperature colorbar
#            fig.text(0.51, 1.02, '$\Delta$ Temperature\n($^\circ$C)', ha='center', va='center', size=12)
            fig.text(0.51, 1.025, '$\Delta$ Temperature', ha='center', va='center', size=12)
            fig.text(0.51, 1.005, '($^\circ$C)', ha='center', va='center', size=10)
            cmap=cmap_dict['temp_glac_monthly']
#            norm=norm_dict['temp_glac_monthly']
            norm = MidpointNormalize(midpoint=norm_dict['temp_glac_monthly'][rcp][1], 
                                     vmin=norm_dict['temp_glac_monthly'][rcp][0], 
                                     vmax=norm_dict['temp_glac_monthly'][rcp][2])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cax = plt.axes([0.405, 0.95, 0.215, 0.015])
            plt.colorbar(sm, cax=cax, orientation='horizontal')
            cax.xaxis.set_ticks_position('top')
            for n, label in enumerate(cax.xaxis.get_ticklabels()):
                if n%2 != 0:
                    label.set_visible(False)
            
            # Precipitation colorbar
#            fig.text(0.79, 1.02, '$\Delta$ Precipitation\n(-)', ha='center', va='center', size=12)
            fig.text(0.79, 1.025, '$\Delta$ Precipitation', ha='center', va='center', size=12)
            fig.text(0.79, 1.005, '(-)', ha='center', va='center', size=10)
            cmap=cmap_dict['prec_glac_monthly']
#            norm=norm_dict['prec_glac_monthly']
            norm = MidpointNormalize(midpoint=norm_dict['prec_glac_monthly'][rcp][1], 
                                     vmin=norm_dict['prec_glac_monthly'][rcp][0], 
                                     vmax=norm_dict['prec_glac_monthly'][rcp][2])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []
            cax = plt.axes([0.685, 0.95, 0.215, 0.015])
            plt.colorbar(sm, cax=cax, orientation='horizontal', format='%.2f')
            cax.xaxis.set_ticks_position('top')
            for n, label in enumerate(cax.xaxis.get_ticklabels()):
                if n%2 != 0:
                    label.set_visible(False)
                        
            # Label y-axis
            fig.text(0.03, 0.5, 'Glacier Number', 
                     va='center', ha='center', rotation='vertical', size=12)  
                
            # Save figure
            fig.set_size_inches(7,8)
            figure_fn = (grouping + '_heatmap_' + rcp + '_' + str(len(gcm_names)) + 'gcms_areagt' + str(area_cutoff) + 
                         'km2.png')
            fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

#%%
if option_map_gcm_changes == 1:
    figure_fp = pygem_prms.output_sim_fp + 'figures/gcm_changes/'
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
#    self.timestep = pygem_prms.timestep
#    self.rgi_lat_colname=pygem_prms.rgi_lat_colname
#    self.rgi_lon_colname=pygem_prms.rgi_lon_colname
#    self.rcp_scenario = rcp_scenario
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=2000, endyear=2100, spinupyears=1, option_wateryear=1)
    
    for rcp in rcps:
        for ngcm, gcm_name in enumerate(gcm_names):
            
            # Variable filepaths
            var_fp = pygem_prms.cmip5_fp_var_prefix + rcp + pygem_prms.cmip5_fp_var_ending
            fx_fp = pygem_prms.cmip5_fp_fx_prefix + rcp + pygem_prms.cmip5_fp_fx_ending
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

                
#%% TIME SERIES OF SUBPLOTS FOR EACH GROUP
if option_plot_cmip5_normalizedchange == 1:
    netcdf_fp_cmip5 = '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_subset/'  
    
    vn = 'volume_glac_annual'
#    vn = 'runoff_glac_monthly'
    
    if 'runoff' in vn:
        grouping = 'watershed'
#        grouping = 'all'
    
    # Peakwater running mean years
    nyears=11
    
    startyear = 2015
    endyear = 2100
    
#    startyear = 1961
#    endyear = 2260
    
    figure_fp = netcdf_fp_cmip5 + 'figures/'
    runoff_fn_pkl = figure_fp + 'watershed_runoff_annual_22gcms_4rcps-' + grouping + '.pkl'
    vol_fn_pkl = figure_fp + 'regional_vol_annual_22gcms_4rcps-' + grouping + '.pkl'
    option_plot_individual_gcms = 0
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    
    if grouping in ['rgi_region']:
        groups.append('all')

    #%%
    # Load data
    if vn == 'runoff_glac_monthly' and os.path.isfile(runoff_fn_pkl):
        with open(runoff_fn_pkl, 'rb') as f:
            ds_all = pickle.load(f)
        # Load single GCM to get time values needed for plot
        ds_fn = ('R' + str(regions[0]) + '--all--' + gcm_names[0] + '_' + rcps[0] + 
                 '_c2_ba1_100sets_2000_2100--subset.nc')
        ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
    elif vn == 'volume_glac_annual' and os.path.isfile(vol_fn_pkl):
        with open(vol_fn_pkl, 'rb') as f:
            ds_all = pickle.load(f)
        # Load single GCM to get time values needed for plot
        ds_fn = ('R' + str(regions[0]) + '--all--' + gcm_names[0] + '_' + rcps[0] + 
                 '_c2_ba1_100sets_2000_2100--subset.nc')
        ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
    else:
        # Load data
        ds_all = {}
        ds_std_all = {}
        for rcp in rcps:
            ds_all[rcp] = {}
            ds_std_all[rcp] = {}
            
            for ngcm, gcm_name in enumerate(gcm_names):
                ds_all[rcp][gcm_name] = {}
                ds_std_all[rcp][gcm_name] = {}
                
                print(rcp, gcm_name)
            
                # Merge all data, then select group data
                for region in regions:  
                    print(region)
    
                    # Load datasets
                    ds_fn = ('R' + str(region) + '--all--' + gcm_name + '_' + rcp + 
                             '_c2_ba1_100sets_2000_2100--subset.nc')
#                    ds_fn = ('R' + str(region) + '--all--' + gcm_name + '_' + rcp + 
#                             '_c2_ba1_100sets_1961_2260.nc')
      
                    # Bypass GCMs that are missing a rcp scenario
                    try:
                        print(netcdf_fp_cmip5)
                        print(ds_fn)
                        ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                        df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
                        df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                                       str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
                        skip_gcm = 0
                    except:
                        skip_gcm = 1
                        print('Skip', gcm_name, rcp, region)
                    
                    if skip_gcm == 0:
                        # Extract time variable
                        time_values_annual = ds.coords['year_plus1'].values
                        time_values_monthly = ds.coords['time'].values
                        # Extract data
                        vn_glac_region = ds[vn].values[:,:,0]
                        vn_glac_std_region = ds[vn].values[:,:,1]
                        
                        # Convert monthly values to annual
                        if vn == 'runoff_glac_monthly':
                            vn_offglac_region = ds['offglac_runoff_monthly'].values[:,:,0]
                            vn_offglac_std_region = ds['offglac_runoff_monthly'].values[:,:,1]                                
                            vn_glac_region += vn_offglac_region
                            vn_glac_std_region += vn_offglac_std_region                            
                            
                            vn_glac_region = gcmbiasadj.annual_sum_2darray(vn_glac_region)
                            time_values_annual = time_values_annual[:-1]                    
                            vn_glac_std_region = gcmbiasadj.annual_sum_2darray(vn_glac_std_region)
                            
                        # Merge datasets
                        if region == regions[0]:
                            vn_glac_all = vn_glac_region
                            vn_glac_std_all = vn_glac_std_region   
                            df_all = df
                        else:
                            vn_glac_all = np.concatenate((vn_glac_all, vn_glac_region), axis=0)
                            vn_glac_std_all = np.concatenate((vn_glac_std_all, vn_glac_std_region), axis=0)
                            df_all = pd.concat([df_all, df], axis=0)
                        
                        ds.close()
                    

                # Remove RGIIds from main_glac_rgi that are not in the model runs
                if ngcm == 0:
                    rgiid_df = list(df_all.RGIId.values)
                    rgiid_all = list(main_glac_rgi.RGIId.values)
                    rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
                    main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
                    main_glac_rgi.reset_index(inplace=True, drop=True)
                    
                    
                if skip_gcm == 0:
                    for ngroup, group in enumerate(groups):
                        # Select subset of data
                        if group in ['all']:
                            group_glac_indices = (
                                    main_glac_rgi.loc[main_glac_rgi['all_group'] == group].index.values.tolist())
                        else:
                            group_glac_indices = (
                                    main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist())
                        vn_glac = vn_glac_all[group_glac_indices,:]
                        
                        subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)
    
                        # Sum volume change for group
                        if group in ['all']:
                            group_glac_indices = (
                                    main_glac_rgi.loc[main_glac_rgi['all_group'] == group].index.values.tolist())
                        else:
                            group_glac_indices = (
                                    main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist())
                        vn_group = vn_glac_all[group_glac_indices,:].sum(axis=0)
    #                    area_group = area_glac_all[group_glac_indices,:].sum(axis=0)
                        
    #                    # Uncertainty associated with volume change based on subgroups
    #                    #  sum standard deviations in each subgroup assuming that they are uncorrelated
    #                    #  then use the root sum of squares using the uncertainty of each subgroup to get the 
    #                    #  uncertainty of the group
    #                    main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
    #                    subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()
    #
    #                    subgroup_std = np.zeros((len(subgroups_subset), vn_group.shape[0]))
    #                    for nsubgroup, subgroup in enumerate(subgroups_subset):
    #                        main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
    #                        subgroup_indices = (
    #                                main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist())
    #                        # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
    #                        subgroup_std[nsubgroup,:] = vn_glac_std_all[subgroup_indices,:].sum(axis=0)
    #                    vn_group_std = (subgroup_std**2).sum(axis=0)**0.5        
                        
                        ds_all[rcp][gcm_name][group] = vn_group
    #                    ds_std_all[rcp][gcm_name][group] = vn_group_std
        
        if vn == 'runoff_glac_monthly':
            pickle_data(runoff_fn_pkl, ds_all)
        elif vn == 'volume_glac_annual':
            pickle_data(vol_fn_pkl, ds_all)
    
    #%%
    # Select multimodel data
    ds_multimodel = {}
    for rcp in rcps:
        ds_multimodel[rcp] = {}
        
        for ngroup, group in enumerate(groups):
        
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, group, gcm_name)
                
                try:
                    vn_group = ds_all[rcp][gcm_name][group]
                    skip_gcm = 0
                    
                except:
                    skip_gcm = 1
                
                if skip_gcm == 0:
                    if ngcm == 0:
                        vn_multimodel = vn_group                
                    else:
                        vn_multimodel = np.vstack((vn_multimodel, vn_group))
            
            ds_multimodel[rcp][group] = vn_multimodel
     
    # Adjust groups or their order           
    if grouping == 'watershed':
        groups.remove('Irrawaddy')
        groups.remove('Yellow')
        
    if grouping == 'himap':
        group_order = [11,17,1,2,18,4,10,7,15,3,12,20,8,0,19,5,21,14,13,16,9,6]
        groups = [x for _,x in sorted(zip(group_order,groups))]
    elif grouping == 'watershed':
        group_order = [0,10,9,3,8,4,5,6,11,1,2,7]
        groups = [x for _,x in sorted(zip(group_order,groups))]
        
    #%%
    # ===== PLOT THE NORMALIZED CHANGE AS SUBPLOTS =====
    multimodel_linewidth = 2
    alpha=0.2

    reg_legend = []
    num_cols_max = 4
    if len(groups) < num_cols_max:
        num_cols = len(groups)
    else:
        num_cols = num_cols_max
    num_rows = int(np.ceil(len(groups)/num_cols))
        
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=False, sharex=False, sharey=True, 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    
    if vn == 'volume_glac_annual':
        stats_cns = ['group', 'rcp', 'Gt_init', '% remaining', '% remaining_std', 'mb_gt_loss', 'mb_gt_loss_std']
        output_df = pd.DataFrame(np.zeros((len(groups)*4, len(stats_cns))), columns=stats_cns)
        ncount = 0
    # Cycle through groups  
    row_idx = 0
    col_idx = 0
    for ngroup, group in enumerate(groups):
        # Set subplot position
        if (ngroup % num_cols == 0) and (ngroup != 0):
            row_idx += 1
            col_idx = 0

        for rcp in rcps:  

            # ===== Plot =====            
            # Multi-model statistics
            vn_multimodel = ds_multimodel[rcp][group]
#            vn_multimodel_mean = vn_multimodel
            vn_multimodel_mean = vn_multimodel.mean(axis=0)
            vn_multimodel_std = vn_multimodel.std(axis=0)
#            vn_multimodel_std = np.zeros(vn_multimodel_mean.shape)
            vn_multimodel_stdlow = vn_multimodel_mean - vn_multimodel_std
            vn_multimodel_stdhigh = vn_multimodel_mean + vn_multimodel_std
            
            # Normalize volume by initial volume
            if vn == 'volume_glac_annual':
#                vn_normalizer = vn_multimodel_mean[0]
                tnorm_idx = np.where(time_values_annual == 2006)[0][0]
                vn_normalizer = vn_multimodel_mean[tnorm_idx]
            # Normalize runoff by mean runoff from 2000-2015
            elif vn == 'runoff_glac_monthly':
                t1_idx = np.where(time_values_annual == 2000)[0][0]
                t2_idx = np.where(time_values_annual == 2015)[0][0] + 1
#                vn_normalizer = vn_multimodel.mean(axis=0)[t1_idx:t2_idx].mean()
                vn_normalizer = vn_multimodel[t1_idx:t2_idx].mean()
            vn_multimodel_mean_norm = vn_multimodel_mean / vn_normalizer
            vn_multimodel_std_norm = vn_multimodel_std / vn_normalizer
            vn_multimodel_stdlow_norm = vn_multimodel_mean_norm - vn_multimodel_std_norm
            vn_multimodel_stdhigh_norm = vn_multimodel_mean_norm + vn_multimodel_std_norm
            
            t1_idx = np.where(time_values_annual == startyear)[0][0]
            t2_idx = np.where(time_values_annual == endyear)[0][0] + 1
            
            ax[row_idx, col_idx].plot(
                    time_values_annual[t1_idx:t2_idx], vn_multimodel_mean_norm[t1_idx:t2_idx], color=rcp_colordict[rcp], 
                    linewidth=multimodel_linewidth, label=rcp, zorder=4)
            if len(rcps) == 4 and rcp in ['rcp26', 'rcp85']: 
                ax[row_idx, col_idx].plot(
                        time_values_annual[t1_idx:t2_idx], vn_multimodel_stdlow_norm[t1_idx:t2_idx], 
                        color=rcp_colordict[rcp], linewidth=0.25, linestyle='-', label=rcp, zorder=3)
                ax[row_idx, col_idx].plot(
                        time_values_annual[t1_idx:t2_idx], vn_multimodel_stdhigh_norm[t1_idx:t2_idx], 
                        color=rcp_colordict[rcp], linewidth=0.25, linestyle='-', label=rcp, zorder=3)
                ax[row_idx, col_idx].fill_between(
                        time_values_annual[t1_idx:t2_idx], vn_multimodel_stdlow_norm[t1_idx:t2_idx], 
                        vn_multimodel_stdhigh_norm[t1_idx:t2_idx], 
                        facecolor=rcp_colordict[rcp], alpha=0.2, label=None, zorder=2)
                
            # Group labels
            if vn == 'volume_glac_annual':
                group_labelsize = 10
            else:
                group_labelsize = 12
            ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=group_labelsize, 
                                      horizontalalignment='center', verticalalignment='top', 
                                      transform=ax[row_idx, col_idx].transAxes)

            # X-label
            ax[row_idx, col_idx].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                          time_values_annual[t1_idx:t2_idx].max())
            ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=12)
            ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(50))
            ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
            if col_idx == 0 and row_idx == num_rows-1:
                ax[row_idx, col_idx].set_xticklabels(['2015','2050','2100'])
            elif row_idx == num_rows-1:
                ax[row_idx, col_idx].set_xticklabels(['','2050','2100'])
            else:
                ax[row_idx, col_idx].set_xticklabels(['','',''])
                
            # Y-label

            if vn == 'volume_glac_annual':
                ax[row_idx, col_idx].set_ylim(0, 1.15)
                if option_plot_individual_gcms == 1:
                    ax[row_idx, col_idx].set_ylim(0,1.35)
                ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
                ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0',])
            elif vn == 'runoff_glac_monthly':
                ax[row_idx, col_idx].set_ylim(0,2.2)
                ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
                ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5','2.0',''])
            ax[row_idx, col_idx].yaxis.set_tick_params(labelsize=12)
#            ax[row_idx, col_idx].yaxis.set_major_locator(MaxNLocator(prune='both'))
                
            # Tick parameters
            ax[row_idx, col_idx].yaxis.set_ticks_position('both')
            ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=12, direction='inout')
            ax[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=12, direction='inout')            
            
            # Add value to subplot
            plot_str = ''
            if vn == 'volume_glac_annual' and rcp == rcps[-1]:
                volume_str = str(int(np.round(vn_multimodel_mean[0] * pygem_prms.density_ice / pygem_prms.density_water, 0)))
                plot_str = volume_str
                if grouping == 'himap':
                    plot_str_loc = 0.05
                else:
                    plot_str_loc = 0.9
                
            elif vn == 'runoff_glac_monthly' and rcp == rcps[-1]:
                group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                group_volume_Gt = ((main_glac_hyps.values[group_glac_indices,:] * 
                                    main_glac_icethickness.values[group_glac_indices,:] / 1000 * pygem_prms.density_ice / 
                                    pygem_prms.density_water).sum())
                group_runoff_Gta = ds_multimodel[rcp][group].mean(axis=0)[:15].mean() * (1/1000)**3
                plot_str = '(' + str(int(np.round(group_runoff_Gta,0))) + ' Gt $\mathregular{yr^{-1}}$)'
                plot_str_loc = 0.90

            if rcp == rcps[-1]:
                if grouping in ['all', 'rgi_region', 'kaab']:
                    ax[row_idx, col_idx].text(0.5, 0.9, plot_str, size=10, horizontalalignment='center', 
                                              verticalalignment='top', transform=ax[row_idx, col_idx].transAxes, 
                                              color='k', zorder=5)
                else:
                    ax[row_idx, col_idx].text(0.05, plot_str_loc, plot_str, size=10, horizontalalignment='left', 
                                              verticalalignment='bottom', transform=ax[row_idx, col_idx].transAxes, 
                                              color='k', zorder=5)
                   
            # Print relevant information
            print(group, rcp, int(np.round(vn_multimodel_mean_norm[-1],2)*100), 
                  '+/-', int(np.round(vn_multimodel_std_norm[-1],2)*100),'% remaining', 
                  np.round((1-vn_multimodel_mean_norm[-1]) * np.round(vn_multimodel_mean[0] * pygem_prms.density_ice
                            / pygem_prms.density_water,0),0), 
                  '+/-', np.round(vn_multimodel_std[-1]*pygem_prms.density_ice/pygem_prms.density_water,0), 'Gt total loss')
                 
            if vn == 'volume_glac_annual':
                output_df.loc[ncount,:] = (
                        [group, rcp, 
                         vn_multimodel_mean[0] * pygem_prms.density_ice / pygem_prms.density_water,
                         vn_multimodel_mean_norm[-1]*100, 
                         vn_multimodel_std_norm[-1]*100,
                        (1-vn_multimodel_mean_norm[-1]) * vn_multimodel_mean[tnorm_idx] * pygem_prms.density_ice
                            / pygem_prms.density_water,
                         vn_multimodel_std[-1]*pygem_prms.density_ice/pygem_prms.density_water])
                ncount += 1
            
            if grouping == 'rgi_region':
                print('  HH2015 comparison 2010-2100, vol loss [%]:', 
                      np.round((vn_multimodel_mean[-1] - vn_multimodel_mean[10]) / vn_multimodel_mean[10] * 100,0))
            
                
        # Count column index to plot
        col_idx += 1
        
    if grouping == 'himap' and vn == 'volume_glac_annual' and num_cols == 4:
        fig.delaxes(ax[5][2])
        fig.delaxes(ax[5][3])
        
        # X-label
        ax[4,2].set_xticklabels(['','2050','2100'])
        ax[4,3].set_xticklabels(['','2050','2100'])
            
        # RCP Legend
        rcp_lines = []
        rcp_number_dict = {'rcp26':' (22 GCMs)','rcp45':' (22 GCMs)','rcp60':' (15 GCMs)','rcp85':' (22 GCMs)',}
        for rcp in rcps:
            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
            rcp_lines.append(line)
        rcp_labels = ['RCP ' + rcp_dict[rcp] + rcp_number_dict[rcp] for rcp in rcps]
#        line = Line2D([0,1],[0,1], color='grey', linewidth=multimodel_linewidth)
#        rcp_lines.append(line)
#        rcp_labels.append('multi-model\nmean')
#        line = Line2D([0,1],[0,1], color='grey', linewidth=4*multimodel_linewidth, alpha=0.2)
#        rcp_lines.append(line)
#        rcp_labels.append('multi-model\nstandard deviation')
#        line = Line2D([0,1],[0,1], color='none', linewidth=multimodel_linewidth)
#        rcp_lines.append(line)
#        rcp_labels.append('')
        fig.legend(rcp_lines, rcp_labels, loc=(0.6, 0.075), fontsize=10, labelspacing=0.1, handlelength=1, 
                   handletextpad=0.5, borderpad=0, frameon=False, ncol=1)
    else:
        # RCP Legend
        rcp_lines = []
        for rcp in rcps:
            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
            rcp_lines.append(line)
        rcp_labels = ['RCP ' + rcp_dict[rcp] for rcp in rcps]
        if vn == 'temp_glac_annual' or vn == 'prec_glac_annual':
            legend_loc = 'upper left'
        else:
            legend_loc = 'lower left'
        ax[0,0].legend(rcp_lines, rcp_labels, loc=(0.05,0.01), fontsize=10, labelspacing=0, handlelength=1, 
                       handletextpad=0.25, borderpad=0, frameon=False, ncol=1)
#        if grouping != 'all':
#            # RCP Legend
#            rcp_lines = []
#            line = Line2D([0,1],[0,1], color='grey', linewidth=multimodel_linewidth)
#            rcp_lines.append(line)
#            line = Line2D([0,1],[0,1], color='grey', linewidth=4*multimodel_linewidth, alpha=0.2)
#            rcp_lines.append(line)
#            rcp_labels = ['multi-model\nmean', 'multi-model\nstandard deviation']
#            ax[0,1].legend(rcp_lines, rcp_labels, loc=(0.05,0.01), fontsize=8, labelspacing=0.2, handlelength=1, 
#                           handletextpad=0.6, borderpad=0, frameon=False, ncol=1)

    # Label
    if vn == 'runoff_glac_monthly':
        ylabel_str = 'Runoff (-)'
    elif vn == 'volume_glac_annual':
        ylabel_str = 'Mass (-)'
    # Y-Label
    if len(groups) == 1:
        fig.text(-0.01, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
    else:
        fig.text(0.03, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
    
    # Save figure
    if len(groups) == 1:
        fig.set_size_inches(4, 4)
    elif vn == 'runoff_glac_monthly':
        fig.set_size_inches(7.2, 7)
    elif vn == 'volume_glac_annual':
        if grouping == 'himap':
            fig.set_size_inches(7.2, 7)
        else:
            fig.set_size_inches(2*num_cols, 2*num_rows)
    
    figure_fn = grouping + '_' + vn + '_multimodel_' + str(len(rcps)) + 'rcps.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)

    if vn == 'volume_glac_annual':    
        output_df.to_csv(figure_fp + grouping + '_' + vn + '_multimodel_' + str(len(rcps)) + 'rcps_stats.csv',
                         index=False)
    
    #%%
    # ===== RUNOFF PLOT OF NORMALIZED CHANGE WITH SUBPLOTS AROUND A MAP =====
    if vn == 'runoff_glac_monthly':
        # External subplots
        multimodel_linewidth = 2
        alpha=0.2
        year_start_peakwater = 2015
        year_idx_start = np.where(time_values_annual == year_start_peakwater)[0][0]
        
        group_colordict = {'Amu_Darya': 'mediumblue',
                           'Brahmaputra': 'salmon',
                           'Ganges': 'lightskyblue',
                           'Ili': 'royalblue',
                           'Indus': 'darkred',
                           'Inner_Tibetan_Plateau': 'gold',
                           'Inner_Tibetan_Plateau_extended': 'navy',
                           'Irrawaddy': 'white',
                           'Mekong': 'white',
                           'Salween': 'plum',
                           'Syr_Darya':'darkolivegreen',
                           'Tarim': 'olive',
                           'Yangtze': 'orange',
                           'Yellow':'white'}
        group_rowcol_idx = {'Amu_Darya': [1,0],
                           'Brahmaputra': [3,1],
                           'Ganges': [3,0],
                           'Ili': [0,2],
                           'Indus': [2,0],
                           'Inner_Tibetan_Plateau': [1,3],
                           'Inner_Tibetan_Plateau_extended': [0,3],
                           'Irrawaddy': [0,0],
                           'Mekong': [0,0],
                           'Salween': [3,2],
                           'Syr_Darya':[0,0],
                           'Tarim': [0,1],
                           'Yangtze': [2,3],
                           'Yellow':[0,0]}
        
        # Cycle through groups  
        if 'Mekong' in groups:
            groups.remove('Mekong')
        
        fig, ax = plt.subplots(4, 4, squeeze=False, sharex=False, sharey=False, 
                                   gridspec_kw = {'wspace':0.25, 'hspace':0.25})
        
        for ngroup, group in enumerate(groups):
    #    for ngroup, group in enumerate(['Amu_Darya']):
            row_idx = group_rowcol_idx[group][0]
            col_idx = group_rowcol_idx[group][1]
    
            for rcp in rcps:  
    
                # ===== Plot =====            
                # Multi-model statistics
                vn_multimodel = ds_multimodel[rcp][group]
                vn_multimodel_mean = vn_multimodel.mean(axis=0)
                vn_multimodel_std = vn_multimodel.std(axis=0)
                vn_multimodel_stdlow = vn_multimodel_mean - vn_multimodel_std
                vn_multimodel_stdhigh = vn_multimodel_mean + vn_multimodel_std
                
                # Normalize volume by initial volume
                if vn == 'volume_glac_annual':
                    vn_normalizer = vn_multimodel_mean[0]
                # Normalize runoff by mean runoff from 2000-2015
                elif vn == 'runoff_glac_monthly':
                    t1_idx = np.where(time_values_annual == 2000)[0][0]
                    t2_idx = np.where(time_values_annual == 2015)[0][0] + 1
                    vn_normalizer = vn_multimodel.mean(axis=0)[t1_idx:t2_idx].mean()
                vn_multimodel_mean_norm = vn_multimodel_mean / vn_normalizer
                vn_multimodel_std_norm = vn_multimodel_std / vn_normalizer
                vn_multimodel_stdlow_norm = vn_multimodel_mean_norm - vn_multimodel_std_norm
                vn_multimodel_stdhigh_norm = vn_multimodel_mean_norm + vn_multimodel_std_norm
                
                t1_idx = np.where(time_values_annual == startyear)[0][0]
                t2_idx = np.where(time_values_annual == endyear)[0][0] + 1
                
                ax[row_idx, col_idx].plot(
                        time_values_annual[t1_idx:t2_idx], vn_multimodel_mean_norm[t1_idx:t2_idx], 
                        color=rcp_colordict[rcp], linewidth=multimodel_linewidth, label=rcp, zorder=4)
                if len(rcps) == 4 and rcp in ['rcp26', 'rcp85']:
                    ax[row_idx, col_idx].plot(
                            time_values_annual[t1_idx:t2_idx], vn_multimodel_stdlow_norm[t1_idx:t2_idx], 
                            color=rcp_colordict[rcp], linewidth=0.25, linestyle='-', label=rcp, zorder=3)
                    ax[row_idx, col_idx].plot(
                            time_values_annual[t1_idx:t2_idx], vn_multimodel_stdhigh_norm[t1_idx:t2_idx], 
                            color=rcp_colordict[rcp], linewidth=0.25, linestyle='-', label=rcp, zorder=3)
                    ax[row_idx, col_idx].fill_between(
                            time_values_annual[t1_idx:t2_idx], vn_multimodel_stdlow_norm[t1_idx:t2_idx], 
                            vn_multimodel_stdhigh_norm[t1_idx:t2_idx], 
                            facecolor=rcp_colordict[rcp], alpha=0.2, label=None, zorder=2)
    
                # Peakwater stats on plots (lower left: peakwater and increase, lower right: change by end of century)
                group_peakwater = peakwater(vn_multimodel_mean[year_idx_start:], 
                                            time_values_annual[year_idx_start:-1], nyears)
                
                # Add value to subplot
                plot_str = ''
                if vn == 'volume_glac_annual' and rcp == rcps[-1]:
                    volume_str = str(int(np.round(vn_multimodel_mean[0] * pygem_prms.density_ice / pygem_prms.density_water, 0)))
                    plot_str = '(' + volume_str + ' Gt)'
                    plot_str_loc = 0.83
                elif vn == 'runoff_glac_monthly' and rcp == rcps[-1]:
                    group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                    group_volume_Gt = ((main_glac_hyps.values[group_glac_indices,:] * 
                                        main_glac_icethickness.values[group_glac_indices,:] / 1000 * pygem_prms.density_ice / 
                                        pygem_prms.density_water).sum())
                    group_runoff_Gta = ds_multimodel[rcp][group].mean(axis=0)[:15].mean() * (1/1000)**3
#                    plot_str = str(int(np.round(group_runoff_Gta,0))) + ' Gt $\mathregular{yr^{-1}}$'
                    plot_str = str(np.round(group_runoff_Gta,1)) + ' Gt $\mathregular{yr^{-1}}$'
                    plot_str = str(np.round(group_runoff_Gta,1))
                    plot_str_loc = 0.90
                
                print(rcp, group, group_peakwater[0])
                ax[row_idx,col_idx].plot((group_peakwater[0], group_peakwater[0]), (0, 1+group_peakwater[1]/100), 
                                            color=rcp_colordict[rcp], linewidth=1, linestyle='--', zorder=5)
               
            # Group labels
            group_labelsize = 10
            tick_labelsize = 9
            title_dict['Salween'] = 'Salween (Sw)'
            title_dict['Yangtze'] = 'Yangtze (Yz)'
            ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=group_labelsize, color='black', 
                                      horizontalalignment='center', verticalalignment='top', 
                                      transform=ax[row_idx, col_idx].transAxes, zorder=10)
    
            # X-label
            ax[row_idx, col_idx].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                          time_values_annual[t1_idx:t2_idx].max())
            ax[row_idx, col_idx].xaxis.set_tick_params(pad=2, size=4, labelsize=tick_labelsize)
            ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
            ax[row_idx, col_idx].set_xticks([2020,2050,2080])
                
            # Y-label
            
            if vn == 'runoff_glac_monthly':
                ylabel_str = 'Runoff (-)'
                ax[row_idx, col_idx].set_ylim(0,2.2)
                ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax[row_idx, col_idx].set_yticks([0.5,1,1.5,2])
                ax[row_idx, col_idx].set_yticklabels(['0.5','1.0','1.5','2.0'])
                ax[row_idx, col_idx].yaxis.set_tick_params(pad=0, size=4, labelsize=tick_labelsize)
            elif vn == 'volume_glac_annual':
                ylabel_str = 'Mass (-)'
                ax[row_idx, col_idx].set_ylim(0, 1.15)
                if option_plot_individual_gcms == 1:
                    ax[row_idx, col_idx].set_ylim(0,1.35)
                ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
                ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0',])  
            ax[row_idx, col_idx].yaxis.set_ticks_position('both')
            ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=tick_labelsize, direction='inout')
            ax[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=tick_labelsize, direction='inout')  
                
            # Mean annual runoff
            ax[row_idx, col_idx].text(0.99, 0.0, plot_str, size=9, horizontalalignment='right', 
                                      verticalalignment='bottom', transform=ax[row_idx, col_idx].transAxes, 
                                      color='k', zorder=6)
                
        # Label
        if vn == 'runoff_glac_monthly':
            ylabel_str = 'Runoff (-)'
        elif vn == 'volume_glac_annual':
            ylabel_str = 'Mass (-)'
        # Y-Label
        if len(groups) == 1:
            fig.text(-0.01, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
        else:
            fig.text(0.06, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
            
        fig.delaxes(ax[1][1])
        fig.delaxes(ax[1][2])
        fig.delaxes(ax[2][1])
        fig.delaxes(ax[2][2])
        fig.delaxes(ax[3][3])
        
        # RCP Legend
        leg_size = 9
        rcp_lines = []
        for rcp in ['rcp26', 'rcp45', 'rcp60', 'rcp85']:
            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
            rcp_lines.append(line)
        rcp_labels = ['RCP ' + rcp_dict[rcp] for rcp in ['rcp26','rcp45','rcp60', 'rcp85']]
        fig.legend(rcp_lines, rcp_labels, loc=(0.79,0.125), fontsize=leg_size, labelspacing=0, handlelength=1.5, 
                   handletextpad=0.5, borderpad=0, frameon=False, ncol=1)
        
#        leg_size = 9
#        rcp_lines = []
#        for rcp in ['rcp26', 'rcp45']:
#            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
#            rcp_lines.append(line)
#        rcp_labels = ['RCP ' + rcp_dict[rcp] for rcp in ['rcp26','rcp45']]
#        fig.legend(rcp_lines, rcp_labels, loc=(0.75,0.18), fontsize=leg_size, labelspacing=0, handlelength=0.5, 
#                   handletextpad=0.5, borderpad=0, frameon=False, ncol=1)
#        rcp_lines2 = []
#        for rcp in ['rcp60', 'rcp85']:
#            line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
#            rcp_lines2.append(line)
#        rcp_labels2 = ['RCP ' + rcp_dict[rcp] for rcp in ['rcp60', 'rcp85']]
#        fig.legend(rcp_lines2, rcp_labels2, loc=(0.855,0.18), fontsize=leg_size, labelspacing=0, handlelength=0.5, 
#                   handletextpad=0.5, borderpad=0, frameon=False, ncol=1)
#        rcp_lines3 = []
#        line = Line2D([0,1],[0,1], color='grey', linewidth=3*multimodel_linewidth, alpha=0.2)
#        rcp_lines3.append(line)
#        rcp_labels3 = ['multi-model\nstandard deviation']
#        fig.legend(rcp_lines3, rcp_labels3, loc=(0.75,0.12), fontsize=leg_size, labelspacing=0, handlelength=0.5, 
#                   handletextpad=0.5, borderpad=0, frameon=False, ncol=1)
        rcp_lines4 = []
        line = Line2D([0,1],[0,1], color='grey', linewidth=0.75, linestyle='--')
        rcp_lines4.append(line)
        rcp_labels4 = ['Peak water']
        fig.legend(rcp_lines4, rcp_labels4, loc=(0.79,0.085), fontsize=leg_size, labelspacing=0, handlelength=1.5, 
                   handletextpad=0.5, borderpad=0, frameon=False, ncol=1)
            
        # Save figure
        fig.set_size_inches(7.5, 5.25)
        figure_fn = grouping + '-panels-' + vn + '_multimodel_' + str(len(rcps)) + 'rcps.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)
        
#%%
# Regional maps
if option_region_map_nodata == 1:
    figure_fp = netcdf_fp_cmip5 + '../figures/'
    grouping = 'rgi_region'
    
    east = 104
    west = 65
    south = 26.5
    north = 45
    xtick = 5
    ytick = 5
    xlabel = 'Longitude [$^\circ$]'
    ylabel = 'Latitude [$^\circ$]'
    
    labelsize = 13
    
    colorbar_dict = {'precfactor':[0,5],
                     'tempchange':[-5,5],
                     'ddfsnow':[0.0036,0.0046],
                     'dif_masschange':[-0.1,0.1]}

    # Add group and attribute of interest
    if grouping == 'rgi_region':
        group_shp = cartopy.io.shapereader.Reader(rgiO1_shp_fn)
        group_shp_attr = 'RGI_CODE'
    elif grouping == 'watershed':
        group_shp = cartopy.io.shapereader.Reader(watershed_shp_fn)
        group_shp_attr = 'watershed'
    elif grouping == 'kaab':
        group_shp = cartopy.io.shapereader.Reader(kaab_shp_fn)
        group_shp_attr = 'Name'
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(regions)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)

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
    
#    # Add contour lines
#    srtm_contour_shp = cartopy.io.shapereader.Reader(srtm_contour_fn)
#    srtm_contour_feature = cartopy.feature.ShapelyFeature(srtm_contour_shp.geometries(), cartopy.crs.PlateCarree(),
#                                                          edgecolor='black', facecolor='none', linewidth=0.15)
#    ax.add_feature(srtm_contour_feature, zorder=2)  

    # Add attribute of interest to the shapefile
    for rec in group_shp.records():
        # plot polygon outlines on top of everything with their labels
        ax.add_geometries(rec.geometry, cartopy.crs.PlateCarree(), facecolor='None', 
                          edgecolor='black', linewidth=2, zorder=3)
    for group in groups:
        print(group, title_location[group][0])
        ax.text(title_location[group][0], 
                title_location[group][1], 
                title_dict[group], horizontalalignment='center', size=12, zorder=4)
        if group == 'Karakoram':
            ax.plot([72.2, 76.2], [34.3, 35.8], color='black', linewidth=1.5)
        elif group == 'Pamir':
            ax.plot([69.2, 73], [37.3, 38.3], color='black', linewidth=1.5)     
    
    # Save figure
    fig.set_size_inches(6,4)
    fig_fn = grouping + '_only_map.png'
    fig.savefig(figure_fp + fig_fn, bbox_inches='tight', dpi=300)
    
 
#%%
if option_glaciermip_table == 1:
    startyear = 2000
    endyear = 2100
    vn = 'volume_glac_annual'
#    vn = 'area_glac_annual'
    
    rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    
    grouping = 'rgi_region'
    netcdf_fp_cmip5= '/Volumes/LaCie/HMA_PyGEM/2019_0914/spc_subset/'
    
    gcm_names = ['CanESM2', 'CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'GFDL-CM3', 'GFDL-ESM2M', 'GISS-E2-R', 'IPSL-CM5A-LR',
                 'MPI-ESM-LR', 'NorESM1-M']
    
    if vn == 'volume_glac_annual':
        output_prefix = 'Volume'
    elif vn == 'area_glac_annual':
        output_prefix = 'Area'
        
    
    output_fp = pygem_prms.output_sim_fp + 'GlacierMIP/'
    if os.path.exists(output_fp) == False:
        os.makedirs(output_fp)
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    
#%%
    # Load mass balance data
    ds_all = {}
    for rcp in rcps:
        ds_all[rcp] = {}
        for ngcm, gcm_name in enumerate(gcm_names):
#        for ngcm, gcm_name in enumerate(['CanESM2']):
        
            print(rcp, gcm_name)
            
            # Merge all data, then select group data
            for region in regions:      
                
                # Load datasets
                ds_fn = ('R' + str(region) + '--all--' + gcm_name + '_' + rcp + '_c2_ba' + 
                         str(pygem_prms.option_bias_adjustment) + '_100sets_2000_2100--subset.nc')
            
                try:
                    ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                    skip_gcm = 0
                except:
                    skip_gcm = 1
                    print('Skip', gcm_name, rcp, region)
                    
                # Bypass GCMs that are missing a rcp scenario
                if skip_gcm == 0:
                    ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                    df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs.values)
                    df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                                   str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
                    
                    # Extract time variable
                    time_values_annual = ds.coords['year_plus1'].values
                    time_values_monthly = ds.coords['time'].values
                    # Extract start/end indices for calendar year!
                    time_values_df = pd.DatetimeIndex(time_values_monthly)
                    time_values = np.array([x.year for x in time_values_df])
                    time_idx_start = np.where(time_values == startyear)[0][0]
                    time_idx_end = np.where(time_values == endyear)[0][0]
                    year_idx_start = np.where(time_values_annual == startyear)[0][0]
                    year_idx_end = np.where(time_values_annual == endyear)[0][0]
                    
                    time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]
                    var_glac_region = ds[vn].values[:,year_idx_start:year_idx_end+1,0]
                    
                    print(rcp, gcm_name, region, var_glac_region[:,-1].sum() / var_glac_region[:,0].sum() * 100)
    
                    # Merge datasets
                    if region == regions[0]:
                        var_glac_all = var_glac_region
                        df_all = df
                    else:
                        var_glac_all = np.concatenate((var_glac_all, var_glac_region), axis=0)
                        df_all = pd.concat([df_all, df], axis=0)

                    ds.close()


            if skip_gcm == 0:
                # RGIIds of only glaciers in simulations
                if ngcm == 0:
                    rgiid_df = list(df_all.RGIId.values)
                    rgiid_all = list(main_glac_rgi.RGIId.values)
                    rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
                    main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
                    main_glac_rgi.reset_index(inplace=True, drop=True)
                
                ds_all[rcp][gcm_name] = {}
                for ngroup, group in enumerate(groups):
                    # Sum volume change for group
                    group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                    varchg_group = var_glac_all[group_glac_indices,:].sum(axis=0)
    
                    ds_all[rcp][gcm_name][group] = varchg_group
                
                #%%

    # Export csv files
    output_cns = time_values_annual_subset.tolist()
    output_rns = ['Alaska','Western Canada and U.S.','Arctic Canada North','Arctic Canada South','Greenland','Iceland',
                  'Svalbard','Scandinavia','Russian Arctic','North Asia','Central Europe','Caucasus','Central Asia',
                  'South Asia West','South Asia East','Low Latitudes','Southern Andes','New Zealand',
                  'Antarctic and Subantarctic']
    group_dict = {13:'Central Asia', 14:'South Asia West', 15:'South Asia East'}
    rcp_dict = {'rcp26':'RCP26', 'rcp45':'RCP45', 'rcp60':'RCP60', 'rcp85':'RCP85'}

    for gcm_name in gcm_names: 
        for rcp in rcps:
            
            # Load datasets
            ds_fn = ('R' + str(region) + '--all--' + gcm_name + '_' + rcp + '_c2_ba' + 
                     str(pygem_prms.option_bias_adjustment) + '_100sets_2000_2100--subset.nc')
            if os.path.exists(netcdf_fp_cmip5 + ds_fn):
                print(gcm_name, rcp, 'exists')
                
                output = pd.DataFrame(np.zeros((len(output_rns), len(output_cns))) + -9999, 
                                      index=output_rns, columns=output_cns)
            
                for group in groups:   
                    # Convert volume to water equivalent
                    if vn == 'volume_glac_annual':
                        output_gcm_rcp_group = ds_all[rcp][gcm_name][group] * pygem_prms.density_ice / pygem_prms.density_water
                    elif vn == 'area_glac_annual':
                        output_gcm_rcp_group = ds_all[rcp][gcm_name][group]
                    
                    
                    output.loc[group_dict[group],:] = output_gcm_rcp_group
                    
                    # Export txt file
                    output_fn = output_prefix + '_PyGEM_' + gcm_name + '_' + rcp_dict[rcp] + '_r1i1p1.txt'
                    output.to_csv(output_fp + output_fn, sep=',', index=False)
                    
                    txt_header = 'David Rounce, drounce@alaska.edu'
                    if vn == 'volume_glac_annual':
                        txt_header += ', Volume [km3 we]'
                    elif vn == 'area_glac_annual':
                        txt_header += ', Area [km2]'
                    with open(output_fp + output_fn, 'r+') as f:
                        content = f.read()
                        f.seek(0, 0)
                        f.write(txt_header.rstrip('\r\n') + '\n' + content)
                
            else:
                print('  ', gcm_name, rcp, 'does not exist')
            
            #%%

#    # Export csv files
#    output_cns = ['year'] + gcm_names
#    
#    summary_cns = []
#    for rcp in rcps:
#        cn_mean = rcp + '-mean'
#        cn_std = rcp + '-std'
#        cn_min = rcp + '-min'
#        cn_max = rcp + '-max'
#        summary_cns.append(cn_mean)
#        summary_cns.append(cn_std)
#        summary_cns.append(cn_min)
#        summary_cns.append(cn_max)
#    output_summary = pd.DataFrame(np.zeros((len(groups),len(summary_cns))), index=groups, columns=summary_cns)
#    
#
#    for group in groups:
#        for rcp in rcps:
#            
#            print(group, rcp)
#            
#            output = pd.DataFrame(np.zeros((len(time_values_annual_subset), len(output_cns))), columns=output_cns)
#            output['year'] = time_values_annual_subset
#            
#            for gcm_name in gcm_names:
#                output[gcm_name] = ds_all[rcp][gcm_name][group]
#            
#            # Export csv file
#            if grouping == 'rgi_region':
#                grouping_prefix = 'R'
#            else:
#                grouping_prefix = ''
#            output_fn = ('GlacierMIP_' + grouping_prefix + str(group) + '_' + rcp + '_' + str(startyear) + '-' + 
#                         str(endyear) + '_volume_km3ice.csv')
#            
#            output.to_csv(output_fp + output_fn, index=False)
#            
#            vol_data = output[gcm_names].values
#            vol_remain_perc = vol_data[-1,:] / vol_data[0,:] * 100
#            
#            rcp_cns = [rcp + '-mean', rcp+'-std', rcp+'-min', rcp+'-max']
#            output_summary.loc[group, rcp_cns] = [vol_remain_perc.mean(), vol_remain_perc.std(), vol_remain_perc.min(), 
#                                                  vol_remain_perc.max()]
#            
#    # Export summary
#    output_summary_fn = ('GlacierMIP_' + grouping + '_summary_' + str(startyear) + '-' + str(endyear) + 
#                         '_volume_remaining_km3ice.csv')
#    output_summary.to_csv(output_fp + output_summary_fn)
                    
#%%      
if option_zemp_compare == 1:
    startyear = 1980
    endyear = 2016
#    vn = 'volume_glac_annual'
    vn = 'massbaltotal_glac_monthly'
    
    grouping = 'rgi_region'
    subgrouping = 'hexagon55'
#    subgrouping = 'degree'
#    degree_size = 1
    
    netcdf_fp = netcdf_fp_era
    
    zemp_fp = netcdf_fp_era + '../'
    zemp_fn = 'zemp_annual_mwe.csv'
    zemp_df = pd.read_csv(zemp_fp + zemp_fn)
    zemp_idx_start = np.where(zemp_df['year'] == startyear)[0][0]
    zemp_idx_end = np.where(zemp_df['year'] == endyear)[0][0]
    zemp_subset = zemp_df.loc[zemp_idx_start:zemp_idx_end]
    
    output_fp = netcdf_fp_era + 'figures/'
    if os.path.exists(output_fp) == False:
        os.makedirs(output_fp)
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)

    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)
    
    # Load mass balance data
    # Merge all data, then select group data
    for nregion, region in enumerate(regions):  
#    for nregion, region in enumerate([13]):    
        
        # Load datasets
        ds_fn = ('R' + str(region) + '--all--ERA-Interim_c2_ba1_100sets_1980_2017.nc')
        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
        df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs.values)
        df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
                       str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
        
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
        # Extract start/end indices for calendar year!
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        time_values_yr = np.array([x.year for x in time_values_df])
        if pygem_prms.gcm_wateryear == 1:
            time_values_yr = np.array([x.year + 1 if x.month >= 10 else x.year for x in time_values_df])
        time_idx_start = np.where(time_values_yr == startyear)[0][0]
        time_idx_end = np.where(time_values_yr == endyear)[0][0]
        time_values_monthly_subset = time_values_monthly[time_idx_start:time_idx_end + 12]
        year_idx_start = np.where(time_values_annual == startyear)[0][0]
        year_idx_end = np.where(time_values_annual == endyear)[0][0]
        time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]
        
        # Annual glacier volume [km3]
        vol_annual_glac_region = ds['volume_glac_annual'].values[:,year_idx_start:year_idx_end+2,0]
        volchg_annual_glac_region = vol_annual_glac_region[:,1:] - vol_annual_glac_region[:,:-1]
        # Annual glacier area [km2]
        area_annual_glac_region = ds['area_glac_annual'].values[:,year_idx_start:year_idx_end+1,0]
        # Annual elevation change [m]
        elevchg_annual_glac_region = volchg_annual_glac_region / area_annual_glac_region * 1000
        mb_monthly_glac_region = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
        mb_monthly_std_glac_region = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 1]
        
        # Merge datasets
        if nregion == 0:
            elevchg_annual_glac_all = elevchg_annual_glac_region
            area_annual_glac_all = area_annual_glac_region
            mb_monthly_glac_all = mb_monthly_glac_region
            mb_monthly_std_glac_all = mb_monthly_std_glac_region
            df_all = df
        else:
            elevchg_annual_glac_all = np.concatenate((elevchg_annual_glac_all, elevchg_annual_glac_region), axis=0)
            area_annual_glac_all = np.concatenate((area_annual_glac_all, area_annual_glac_region), axis=0)
            mb_monthly_glac_all = np.concatenate((mb_monthly_glac_all, mb_monthly_glac_region), axis=0)
            mb_monthly_std_glac_all = np.concatenate((mb_monthly_std_glac_all, mb_monthly_std_glac_region), axis=0)
            df_all = pd.concat([df_all, df], axis=0)
        try:
            ds.close()
        except:
            continue
        
    # RGIIds of only glaciers in simulations
    rgiid_df = list(df_all.RGIId.values)
    rgiid_all = list(main_glac_rgi.RGIId.values)
    rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
    main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
    main_glac_rgi.reset_index(inplace=True, drop=True)
    
    #%%
    # Convert monthly mass balance and standard deviation into annual values
    mb_annual_glac_all = gcmbiasadj.annual_sum_2darray(mb_monthly_glac_all)
    def annual_rsos(x):
        """ Compute annual root sum of squares for uncertainty """
        return ((x.reshape(-1,12)**2).sum(1)**0.5).reshape(x.shape[0],int(x.shape[1]/12))
    mb_annual_std_glac_all = annual_rsos(mb_monthly_std_glac_all)
    
    elevchg_annual_glac_all_fromVol = elevchg_annual_glac_all.copy()
    elevchg_annual_glac_all = mb_annual_glac_all * pygem_prms.density_water / pygem_prms.density_ice
    # Isolate the elevation change uncertainty from the area and density
    elevchg_annual_std_glac_all = (np.absolute(mb_annual_glac_all * pygem_prms.density_water / pygem_prms.density_ice) *
                                   ((mb_annual_std_glac_all / mb_annual_glac_all)**2 - 0.1**2 - 0.071**2)**0.5)
    elevchg_annual_std_glac_all = np.nan_to_num(elevchg_annual_std_glac_all, 0)
    
    ds_all = {}
    ds_all_std = {}
    for ngroup, group in enumerate(groups):
#    for ngroup, group in enumerate([groups[0]]):
#        print(group)
        
        # Group indices
        group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
        
        # Regional Mass Balance [m w.e. / yr]
        mb_mwea_annual_group = ((mb_annual_glac_all[group_glac_indices,:] * 
                                 area_annual_glac_all[group_glac_indices,:]).sum(axis=0) /
                                area_annual_glac_all[group_glac_indices,:].sum(axis=0))
        
        # Regional elevation change [m/yr]
        elevchg_annual_group = ((elevchg_annual_glac_all[group_glac_indices,:] * 
                                 area_annual_glac_all[group_glac_indices,:]).sum(axis=0) /
                                area_annual_glac_all[group_glac_indices,:].sum(axis=0))
        
        # Elevation change for each cell [m/yr]
        # Uncertainty associated with volume change based on subgroups
        #  sum standard deviations in each subgroup assuming that they are uncorrelated
        #  then use the root sum of squares using the uncertainty of each subgroup to get the uncertainty of the group
        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
        subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()

        # Elevation change uncertainty for each hexagonal grid cell 
        #  (assume glaciers in cell are perfectly correlated - sum)
        elevchg_annual_std_subgroup = np.zeros((len(subgroups_subset), elevchg_annual_group.shape[0]))
        area_annual_subgroup = np.zeros((len(subgroups_subset), elevchg_annual_group.shape[0]))
        for nsubgroup, subgroup in enumerate(subgroups_subset):
            subgroup_indices = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist()
            # subgroup uncertainty is the area-weighted sum of each glacier since assumed to be perfectly correlated
            elevchg_annual_std_subgroup[nsubgroup,:] = ((elevchg_annual_std_glac_all[subgroup_indices,:] * 
                                                         area_annual_glac_all[subgroup_indices,:]).sum(axis=0) /
                                                        area_annual_glac_all[subgroup_indices,:].sum(axis=0))
            area_annual_subgroup[nsubgroup,:] = area_annual_glac_all[subgroup_indices,:].sum(axis=0)
        
        # Elevation change uncertainty for each region 
        #  (assume individual cells within a region are uncorrelated - root sum of squares)
        elevchg_annual_std_group = ((((elevchg_annual_std_subgroup * area_annual_subgroup)**2).sum(axis=0))**0.5
                                     / area_annual_subgroup.sum(axis=0))  
        # Mass balance uncertainty
        #  (combine elevation change uncertainty with density and area uncertainty in quadrature)
        mb_mwea_annual_std_group = (np.absolute(mb_mwea_annual_group) * 
                                    ((elevchg_annual_std_group / elevchg_annual_group)**2 + 0.1**2 + 0.071**2)**0.5)
        
        ds_all[group] = mb_mwea_annual_group
        ds_all_std[group] = mb_mwea_annual_std_group
        
    #%%
    # ===== CODE THAT DOESN'T ISOLATE ELEVATION CHANGE ======
#    # Convert monthly mass balance and standard deviation into annual values
#    mb_annual_glac_all = gcmbiasadj.annual_sum_2darray(mb_monthly_glac_all)
#    def annual_rsos(x):
#        """ Compute annual root sum of squares for uncertainty """
#        return ((x.reshape(-1,12)**2).sum(1)**0.5).reshape(x.shape[0],int(x.shape[1]/12))
#    mb_annual_std_glac_all = annual_rsos(mb_monthly_std_glac_all)
#    
#    ds_all = {}
#    ds_all_std = {}
#    for ngroup, group in enumerate(groups):
#        
#        # Group indices
#        group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
#        
#        # Regional Mass Balance [m w.e. / yr]
#        mb_mwea_annual_group = ((mb_annual_glac_all[group_glac_indices,:] * 
#                                 area_annual_glac_all[group_glac_indices,:]).sum(axis=0) /
#                                area_annual_glac_all[group_glac_indices,:].sum(axis=0))
#        
#        # Elevation change for each cell [m/yr]
#        # Uncertainty associated with volume change based on subgroups
#        #  sum standard deviations in each subgroup assuming that they are uncorrelated
#        #  then use the root sum of squares using the uncertainty of each subgroup to get the uncertainty of the group
#        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
#        subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()
#
#        # Elevation change uncertainty for each hexagonal grid cell 
#        #  (assume glaciers in cell are perfectly correlated - sum)
#        mb_mwea_annual_std_subgroup = np.zeros((len(subgroups_subset), mb_mwea_annual_group.shape[0]))
#        area_annual_subgroup = np.zeros((len(subgroups_subset), mb_mwea_annual_group.shape[0]))
#        for nsubgroup, subgroup in enumerate(subgroups_subset):
#            subgroup_indices = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist()
#            # subgroup uncertainty is the area-weighted sum of each glacier since assumed to be perfectly correlated
#            mb_mwea_annual_std_subgroup[nsubgroup,:] = ((mb_annual_std_glac_all[subgroup_indices,:] * 
#                                                         area_annual_glac_all[subgroup_indices,:]).sum(axis=0) /
#                                                        area_annual_glac_all[subgroup_indices,:].sum(axis=0))
#            area_annual_subgroup[nsubgroup,:] = area_annual_glac_all[subgroup_indices,:].sum(axis=0)
#        
#        # Elevation change uncertainty for each region 
#        #  (assume individual cells within a region are uncorrelated - root sum of squares)
#        mb_mwea_annual_std_group = ((((mb_mwea_annual_std_subgroup * area_annual_subgroup)**2).sum(axis=0))**0.5
#                                     / area_annual_subgroup.sum(axis=0))  
#        
#        ds_all[group] = mb_mwea_annual_group
#        ds_all_std[group] = mb_mwea_annual_std_group

    #%%
    stats_cns = ['group', 'rmse', 'r', 'slope', 'intercept', 'p-value', 'mae', 'nse']
    group_stats = pd.DataFrame(np.zeros((len(groups), len(stats_cns))), columns=stats_cns)
    group_stats['group'] = groups
    fig, ax = plt.subplots(len(groups), 1, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0, 'hspace':0})
    for ngroup, group in enumerate(groups):
        
        zemp_group = zemp_subset[str(group)].values
        zemp_group_std = zemp_subset[str(group) + '_sig'].values
        mb_mwea_group = ds_all[group]
        mb_mwea_group_std = ds_all_std[group]
        years = zemp_subset['year'].values
        dif_group = zemp_group - mb_mwea_group
            
        # All glaciers
        ax[ngroup,0].plot(years, zemp_group, color='k', label='Zemp et al. (2019)', zorder=2)
        ax[ngroup,0].fill_between(years, zemp_group + zemp_group_std, zemp_group - zemp_group_std, 
                                  facecolor='lightgrey', label=None, zorder=1)
    
        ax[ngroup,0].plot(years, mb_mwea_group, color='b', label='Modeled', zorder=2)
        ax[ngroup,0].fill_between(years, mb_mwea_group + mb_mwea_group_std, mb_mwea_group - mb_mwea_group_std, 
                                  facecolor='dodgerblue', label=None, zorder=1)
        ax[ngroup,0].set_ylim(-1.1,0.75)
        ax[ngroup,0].set_xlim(1980,2016)
        ax[ngroup,0].xaxis.set_minor_locator(MultipleLocator(5))
        ax[ngroup,0].tick_params(axis='x', direction='inout', which='both')
        if ngroup == 0:
            ax[ngroup,0].legend(loc=(0.02,0.02), ncol=1, fontsize=10, frameon=False, handlelength=1.5, 
                                handletextpad=0.25, columnspacing=1, borderpad=0, labelspacing=0)
        if ngroup+1 < len(groups):
            ax[ngroup,0].xaxis.set_ticklabels([])
#        if ngroup + 1 == len(groups):
#            ax[ngroup,0].set_xlabel('Year', size=12)
#        else:
#            ax[ngroup,0].xaxis.set_ticklabels([])
        ax[ngroup,0].yaxis.set_ticks(np.arange(-1, 0.55, 0.5))
        
        # Statistics for comparison
        # Root-mean-square-deviation
        rmse = (np.sum((mb_mwea_group - zemp_group)**2) / zemp_group.shape[0])**0.5
        print(group, 'RMSE:', np.round(rmse,2))
        # Correlation
        slope, intercept, r_value, p_value, std_err = linregress(zemp_group, mb_mwea_group)
        print('  r_value =', np.round(r_value,2), 'slope = ', np.round(slope,2), 
              'intercept = ', np.round(intercept,2), 'p_value = ', np.round(p_value,4))
        # Mean absolute error
        mae = np.mean(np.absolute(zemp_group - mb_mwea_group))
        print('  mean absolute error:', np.round(mae,2))
        # Nash-Sutcliffe model efficiency coefficient
        nse = 1 - (np.sum((mb_mwea_group - zemp_group)**2) / np.sum((zemp_group - zemp_group.mean())**2))
        print('  NSE:', np.round(nse,2))
        # Record stats
        group_stats.loc[ngroup, ['rmse', 'r', 'slope', 'intercept', 'p-value', 'mae', 'nse']] = (
                [rmse, r_value, slope, intercept, p_value, mae, nse])
    
    # Add text
    fig.text(-0.08, 0.5, 'Mass Balance (m w.e. $\mathregular{yr^{-1}}$)', va='center', rotation='vertical', size=12)
    fig.text(0.5, 0.845, 'Central Asia', horizontalalignment='center', zorder=4, color='black', fontsize=10)
    fig.text(0.5, 0.59, 'South Asia West', horizontalalignment='center', zorder=4, color='black', fontsize=10)
    fig.text(0.5, 0.34, 'South Asia East', horizontalalignment='center', zorder=4, color='black', fontsize=10)
    fig.text(0.135, 0.84, 'A', zorder=4, color='black', fontsize=12, fontweight='bold')
    fig.text(0.135, 0.59, 'B', zorder=4, color='black', fontsize=12, fontweight='bold')
    fig.text(0.135, 0.335, 'C', zorder=4, color='black', fontsize=12, fontweight='bold')
    
    # Save figure
    fig.set_size_inches(3.25,3.75)
    fig.savefig(output_fp + 'Zemp2019_vs_ERA-Interim_' + str(startyear) + '-' + str(endyear) + '_squeezed.eps', 
                bbox_inches='tight', dpi=300)    
    # Export stats
    group_stats.to_csv(output_fp + 'Zemp2019_vs_ERA-Interim_stats.csv', index=False)
    #%%
#    
##    # ============================ OLD CODE ===========================================================================
##    # Load mass balance data
##    # Merge all data, then select group data
##    for region in regions:      
##        
##        # Load datasets
##        ds_fn = ('R' + str(region) + '--all--ERA-Interim_c2_ba1_100sets_1980_2017.nc')
##        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
##        df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs.values)
##        df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])) + '.' +
##                       str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
##        
##        # Extract time variable
##        time_values_annual = ds.coords['year_plus1'].values
##        time_values_monthly = ds.coords['time'].values
##        # Extract start/end indices for calendar year!
##        time_values_df = pd.DatetimeIndex(time_values_monthly)
##        time_values_yr = np.array([x.year for x in time_values_df])
##        if pygem_prms.gcm_wateryear == 1:
##            time_values_yr = np.array([x.year + 1 if x.month >= 10 else x.year for x in time_values_df])
##        time_idx_start = np.where(time_values_yr == startyear)[0][0]
##        time_idx_end = np.where(time_values_yr == endyear)[0][0]
##        time_values_monthly_subset = time_values_monthly[time_idx_start:time_idx_end + 12]
##        year_idx_start = np.where(time_values_annual == startyear)[0][0]
##        year_idx_end = np.where(time_values_annual == endyear)[0][0]
##        time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]
##        
##        if 'annual' in vn:
##            var_glac_region = ds[vn].values[:,year_idx_start:year_idx_end+1,0]
##        else:
##            var_glac_region_raw = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
##            var_glac_region_raw_std = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 1]
##            area_glac_region = np.repeat(ds['area_glac_annual'].values[:,year_idx_start:year_idx_end+1,0], 12, axis=1)
##            
##            # Area average
##            volchg_monthly_glac_region = var_glac_region_raw * area_glac_region
##            volchg_monthly_glac_region_std = var_glac_region_raw_std * area_glac_region
##
##        # Merge datasets
##        if region == regions[0]:
##            var_glac_all = volchg_monthly_glac_region
##            var_glac_all_std = volchg_monthly_glac_region_std
##            area_glac_all = area_glac_region
##            df_all = df
##        else:
##            var_glac_all = np.concatenate((var_glac_all, volchg_monthly_glac_region), axis=0)
##            var_glac_all_std = np.concatenate((var_glac_all_std, volchg_monthly_glac_region_std), axis=0)
##            area_glac_all = np.concatenate((area_glac_all, area_glac_region), axis=0)
##            df_all = pd.concat([df_all, df], axis=0)
##        try:
##            ds.close()
##        except:
##            continue
##        
##    # RGIIds of only glaciers in simulations
##    rgiid_df = list(df_all.RGIId.values)
##    rgiid_all = list(main_glac_rgi.RGIId.values)
##    rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
##    main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
##    main_glac_rgi.reset_index(inplace=True, drop=True)
##
##    ds_all = {}
##    ds_all_std = {}
##    for ngroup, group in enumerate(groups):
##        print(group)
##        # Sum volume change for group
##        group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
##        varchg_group = var_glac_all[group_glac_indices,:].sum(axis=0)
##        area_group = area_glac_all[group_glac_indices,:].sum(axis=0)
##        varchg_group_std_sos1 = (var_glac_all_std[group_glac_indices,:]**2).sum(axis=0)**0.5
##        
##        # Uncertainty associated with volume change based on subgroups
##        #  sum standard deviations in each subgroup assuming that they are uncorrelated
##        #  then use the root sum of squares using the uncertainty of each subgroup to get the uncertainty of the group
##        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
##        subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()
##        
##        subgroup_std = np.zeros((len(subgroups_subset), varchg_group.shape[0]))
##        for nsubgroup, subgroup in enumerate(subgroups_subset):
##            main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
##            subgroup_indices = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist()
##            # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
##            subgroup_std[nsubgroup,:] = var_glac_all_std[subgroup_indices,:].sum(axis=0)
##        varchg_group_std = (subgroup_std**2).sum(axis=0)**0.5        
##        
##        # Group's monthly mass balance [mwea]
##        mb_mwea_group = (varchg_group / area_group).reshape(-1,12).sum(1)
##        # annual uncertainty is the sum of monthly stdev since assumed to be perfectly correlated
##        mb_mwea_group_std = (varchg_group_std / area_group).reshape(-1,12).sum(1)
###        mb_mwea_group_std_rsos = ((varchg_group_std / area_group)**2).reshape(-1,12).sum(1)**0.5
##        
##        ds_all[group] = mb_mwea_group
##        ds_all_std[group] = mb_mwea_group_std
##
##    #%%
##    stats_cns = ['group', 'rmse', 'r', 'slope', 'intercept', 'p-value', 'mae', 'nse']
##    group_stats = pd.DataFrame(np.zeros((len(groups), len(stats_cns))), columns=stats_cns)
##    group_stats['group'] = groups
##    fig, ax = plt.subplots(len(groups), 1, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0, 'hspace':0})
##    for ngroup, group in enumerate(groups):
##        
##        zemp_group = zemp_subset[str(group)].values
##        zemp_group_std = zemp_subset[str(group) + '_sig'].values
##        mb_mwea_group = ds_all[group]
##        mb_mwea_group_std = ds_all_std[group]
##        years = zemp_subset['year'].values
##        dif_group = zemp_group - mb_mwea_group
##            
##        # All glaciers
##        ax[ngroup,0].plot(years, zemp_group, color='k', label='Zemp et al. (2019)', zorder=2)
##        ax[ngroup,0].fill_between(years, zemp_group + zemp_group_std, zemp_group - zemp_group_std, 
##                                  facecolor='lightgrey', label=None, zorder=1)
##    
##        ax[ngroup,0].plot(years, mb_mwea_group, color='b', label='Modeled', zorder=2)
##        ax[ngroup,0].fill_between(years, mb_mwea_group + mb_mwea_group_std, mb_mwea_group - mb_mwea_group_std, 
##                                  facecolor='dodgerblue', label=None, zorder=1)
##        ax[ngroup,0].set_ylim(-1.1,0.75)
##        ax[ngroup,0].set_xlim(1980,2016)
##        ax[ngroup,0].xaxis.set_minor_locator(MultipleLocator(5))
##        ax[ngroup,0].tick_params(axis='x', direction='inout', which='both')
##        if ngroup == 0:
##            ax[ngroup,0].legend(loc=(0.02,0.02), ncol=1, fontsize=10, frameon=False, handlelength=1.5, 
##                                handletextpad=0.25, columnspacing=1, borderpad=0, labelspacing=0)
##        if ngroup+1 < len(groups):
##            ax[ngroup,0].xaxis.set_ticklabels([])
###        if ngroup + 1 == len(groups):
###            ax[ngroup,0].set_xlabel('Year', size=12)
###        else:
###            ax[ngroup,0].xaxis.set_ticklabels([])
##        ax[ngroup,0].yaxis.set_ticks(np.arange(-1, 0.55, 0.5))
##        
##        # Statistics for comparison
##        # Root-mean-square-deviation
##        rmse = (np.sum((mb_mwea_group - zemp_group)**2) / zemp_group.shape[0])**0.5
##        print(group, 'RMSE:', np.round(rmse,2))
##        # Correlation
##        slope, intercept, r_value, p_value, std_err = linregress(zemp_group, mb_mwea_group)
##        print('  r_value =', np.round(r_value,2), 'slope = ', np.round(slope,2), 
##              'intercept = ', np.round(intercept,2), 'p_value = ', np.round(p_value,4))
##        # Mean absolute error
##        mae = np.mean(np.absolute(zemp_group - mb_mwea_group))
##        print('  mean absolute error:', np.round(mae,2))
##        # Nash-Sutcliffe model efficiency coefficient
##        nse = 1 - (np.sum((mb_mwea_group - zemp_group)**2) / np.sum((zemp_group - zemp_group.mean())**2))
##        print('  NSE:', np.round(nse,2))
##        # Record stats
##        group_stats.loc[ngroup, ['rmse', 'r', 'slope', 'intercept', 'p-value', 'mae', 'nse']] = (
##                [rmse, r_value, slope, intercept, p_value, mae, nse])
##    
##    # Add text
##    fig.text(-0.08, 0.5, 'Mass Balance (m w.e. $\mathregular{yr^{-1}}$)', va='center', rotation='vertical', size=12)
##    fig.text(0.5, 0.845, 'Central Asia', horizontalalignment='center', zorder=4, color='black', fontsize=10)
##    fig.text(0.5, 0.59, 'South Asia West', horizontalalignment='center', zorder=4, color='black', fontsize=10)
##    fig.text(0.5, 0.34, 'South Asia East', horizontalalignment='center', zorder=4, color='black', fontsize=10)
##    fig.text(0.135, 0.84, 'A', zorder=4, color='black', fontsize=12, fontweight='bold')
##    fig.text(0.135, 0.59, 'B', zorder=4, color='black', fontsize=12, fontweight='bold')
##    fig.text(0.135, 0.335, 'C', zorder=4, color='black', fontsize=12, fontweight='bold')
##    
##    # Save figure
##    fig.set_size_inches(3.25,3.75)
##    fig.savefig(output_fp + 'Zemp2019_vs_ERA-Interim_' + str(startyear) + '-' + str(endyear) + '_squeezed.eps', 
##                bbox_inches='tight', dpi=300)    
##    # Export stats
##    group_stats.to_csv(output_fp + 'Zemp2019_vs_ERA-Interim_stats.csv', index=False)
    #%%


if option_gardelle_compare == 1:
    startyear = 1980
    endyear = 2016
    
#    groups = ['Bhutan', 'Everest', 'Hindu Kush', 'Karakoram', 'Pamir', 'Spiti Lahaul', 'West Nepal', 'Yigong']
#    csv_fn = ['Bhutan_20001117_wRGIIds.csv', 'Everest_wRGIIds.csv', 'HinduKush_20001117_wRGIIds.csv', 
#              'Karakoram_wRGIIds.csv', 'Pamir_20000916_wRGIIds.csv', 'SpitiLahaul_wRGIIds.csv', 
#              'WestNepal_wRGIIds.csv', 'Yigong_19990923_wRGIIds.csv']
#    gardelle_ELAs_dict = {}
#    for n, group in enumerate(groups):
#        ds = pd.read_csv(pygem_prms.main_directory + '/../qgis_himat/Gardelle_etal2013/' + csv_fn[n])
#        A = list(ds.RGIId.unique())
#        print(group, len(A), '\n', A)
#        gardelle_ELAs_dict[group] = A
#    
#    with open(pygem_prms.main_directory + '/../qgis_himat/Gardelle_etal2013/gardelle_ELAs_dict.pkl', 'wb') as f:
#        pickle.dump(gardelle_ELAs_dict, f)

    gardelle_dict_fn = pygem_prms.main_directory + '/../qgis_himat/Gardelle_etal2013/gardelle_ELAs_dict.pkl'
    with open(gardelle_dict_fn, 'rb') as f:
        gardelle_group_RGIIds = pickle.load(f)
        
    group_data_dict = {'Yigong': [1999, 9, 4970, 320, '+', 'k', 50],
                       'Bhutan': [2000, 11, 5690, 440, 'o', 'None', 30],
                       'Everest': [2009, 10, 5840, 320, '^', 'None', 30],
                       'West Nepal': [2009, 8, 5590, 138, '*', 'None', 50],
                       'Spiti Lahaul': [2002, 8, 5390, 140, 's', 'None', 25],
                       'Hindu Kush': [2000, 9, 5050, 160, 'x', 'k', 30],
                       'Karakoram': [1998, 9, 5030, 280, 'D', 'None', 25],
                       'Pamir': [2000, 7, 4580, 250, 'v', 'None', 30]}

    grouping = 'kaab'
    
    netcdf_fp = netcdf_fp_era
    
    output_fp = netcdf_fp_era + 'figures/'
    if os.path.exists(output_fp) == False:
        os.makedirs(output_fp)
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
#%%
    # Load mass balance data
    ds_all = {}  
    # Merge all data, then select group data
    for region in regions:      
        
        # Load datasets
        ds_fn = ('R' + str(region) + '--all--ERA-Interim_c2_ba1_100sets_1980_2017.nc')
        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
        
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
        # Extract start/end indices for calendar year!
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        time_values_yr = np.array([x.year for x in time_values_df])
        if pygem_prms.gcm_wateryear == 1:
            time_values_yr = np.array([x.year + 1 if x.month >= 10 else x.year for x in time_values_df])
        time_idx_start = np.where(time_values_yr == startyear)[0][0]
        time_idx_end = np.where(time_values_yr == endyear)[0][0]
        time_values_monthly_subset = time_values_monthly[time_idx_start:time_idx_end + 12]
        year_idx_start = np.where(time_values_annual == startyear)[0][0]
        year_idx_end = np.where(time_values_annual == endyear)[0][0]
        time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]
        
        var_glac_region = ds['ELA_glac_annual'].values[:,year_idx_start:year_idx_end+1,0]
        var_glac_region_std = ds['ELA_glac_annual'].values[:,year_idx_start:year_idx_end+1,1]
#        var_glac_region = ds['snowline_glac_monthly'].values[:,time_idx_start:time_idx_end+1,0]
#        var_glac_region_std = ds['snowline_glac_monthly'].values[:,time_idx_start:time_idx_end+1,1]

        # Merge datasets
        if region == regions[0]:
            var_glac_all = var_glac_region
            var_glac_all_std = var_glac_region_std
        else:
            var_glac_all = np.concatenate((var_glac_all, var_glac_region), axis=0)
            var_glac_all_std = np.concatenate((var_glac_all_std, var_glac_region_std), axis=0)
        try:
            ds.close()
        except:
            continue

    ds_all = {}
    ds_all_std = {}
    for ngroup, group in enumerate(group_data_dict.keys()):
        # ELA for given year
        ela_year_idx = np.where(time_values_annual_subset == group_data_dict[group][0])[0][0]
        group_RGIIds = gardelle_group_RGIIds[group]
        group_glac_indices = []
        for RGIId in group_RGIIds:
            group_glac_indices.append(main_glac_rgi.loc[main_glac_rgi['RGIId'] == RGIId].index.values[0])
        ela_subset = var_glac_all[group_glac_indices, ela_year_idx]
        ela_subset_std = var_glac_all_std[group_glac_indices, ela_year_idx]
        
        ds_all[group] = [ela_subset.mean(), ela_subset.std()]
        ds_all_std[group] = [ela_subset_std.mean(), ela_subset.std()]
        
#        print(group, str(ela_subset.shape[0]), 'glaciers', 
#              np.round(ela_subset.mean(),0), '+/-', np.round(ela_subset.std(),0))
        
        #%%
    # Record all for stats
    gardelle_all = []
    era_ela_all = []
    
    # Plot
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0, 'hspace':0})
    for ngroup, group in enumerate(sorted(group_data_dict.keys())):
        
        gardelle = group_data_dict[group][2]
        gardelle_std = group_data_dict[group][3]
        era_ela = ds_all[group][0]
        era_ela_std = ds_all[group][1]
        gardelle_all.append(gardelle)
        era_ela_all.append(era_ela)
        
        group_label = group
        if group == 'Yigong':
            group_label = 'Hengduan Shan'
        
        print(group, np.round(gardelle,0), '+/-', np.round(gardelle_std,0), 'vs.', 
              np.round(era_ela,0), '+/-', np.round(era_ela_std,0))
            
        # All glaciers
        ax[0,0].scatter(gardelle, era_ela, color='k', label=group_label, marker=group_data_dict[group][4],
                        facecolor=group_data_dict[group][5], s=group_data_dict[group][6], zorder=3)
        ax[0,0].errorbar(gardelle, era_ela, xerr=gardelle_std, yerr=era_ela_std, capsize=1, linewidth=0.5, 
                         color='darkgrey', zorder=2)
    
    ax[0,0].set_xlabel('Observed ELA (m a.s.l.)', size=12)    
    ax[0,0].set_ylabel('Modeled ELA (m a.s.l.)', size=12)
    ymin = 4000
    ymax = 6500
    xmin = 4000
    xmax = 6500
    ax[0,0].set_xlim(xmin,xmax)
    ax[0,0].set_ylim(ymin,ymax)
    ax[0,0].plot([np.min([xmin,ymin]),np.max([xmax,ymax])], [np.min([xmin,ymin]),np.max([xmax,ymax])], color='k', 
                 linewidth=0.5, zorder=1)
    ax[0,0].yaxis.set_ticks(np.arange(4500, 6500, 500))
    ax[0,0].xaxis.set_ticks(np.arange(4500, 6500, 500))
    
    # Ensure proper order for legend
    handles, labels = ax[0,0].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t:t[0]))
    ax[0,0].legend(handles, labels, loc=(0.02,0.51), ncol=1, fontsize=10, frameon=False, handlelength=1, 
                   handletextpad=0.15, columnspacing=0.5, borderpad=0, labelspacing=0)
    # Add text
    fig.text(0.15, 0.85, 'D', va='center', size=12, fontweight='bold')
    # Save figure
    fig.set_size_inches(3.45,3.45)
    fig.savefig(output_fp + 'gardelle2013_compare_regional_ELA_RGIIds.eps', bbox_inches='tight', dpi=300)
    
    # Stats
    gardelle_all = np.array(gardelle_all)
    era_ela_all = np.array(era_ela_all)
    stats_cns = ['group', 'rmse', 'r', 'slope', 'intercept', 'p-value', 'mae']
    group_stats = pd.DataFrame(np.zeros((1, len(stats_cns))), columns=stats_cns)
    group_stats['group'] = 'all'
    # Statistics for comparison
    # Root-mean-square-deviation
    rmse = (np.sum((gardelle_all - era_ela_all)**2) / gardelle_all.shape[0])**0.5
    print('RMSE:', np.round(rmse,2))
    # Correlation
    slope, intercept, r_value, p_value, std_err = linregress(gardelle_all, era_ela_all)
    print('r_value =', np.round(r_value,2), 'slope = ', np.round(slope,2), 
          'intercept = ', np.round(intercept,2), 'p_value = ', np.round(p_value,6))
    # Mean absolute error
    mae = np.mean(np.absolute(gardelle_all - era_ela_all))
    print('mean absolute error:', np.round(mae,2))
    # Record stats
    group_stats.loc[0, ['rmse', 'r', 'slope', 'intercept', 'p-value', 'mae']] = (
            [rmse, r_value, slope, intercept, p_value, mae])
    group_stats.to_csv(output_fp + 'gardelle2013_stats.csv', index=False)
    #%%


if option_wgms_compare == 1:
    regions = [13, 14, 15]
    cal_datasets = ['wgms_d', 'wgms_ee']
    
    startyear=1980
    endyear=2017
    wateryear=1
    
    output_fp = netcdf_fp_era + 'figures/'
    if os.path.exists(output_fp) == False:
        os.makedirs(output_fp)
    
    dates_table  = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0, 
                                            option_wateryear=wateryear)
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Modeled Mass Balance
    ds_all = {}  
    for region in regions:      
        
        # Load datasets
        ds_fn = ('R' + str(region) + '--all--ERA-Interim_c2_ba1_100sets_1980_2017.nc')
        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
        
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
        # Extract start/end indices for calendar year!
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        time_values_yr = np.array([x.year for x in time_values_df])
        if pygem_prms.gcm_wateryear == 1:
            time_values_yr = np.array([x.year + 1 if x.month >= 10 else x.year for x in time_values_df])
        time_idx_start = np.where(time_values_yr == startyear)[0][0]
        time_idx_end = np.where(time_values_yr == endyear)[0][0]
        time_values_monthly_subset = time_values_monthly[time_idx_start:time_idx_end + 12]
        year_idx_start = np.where(time_values_annual == startyear)[0][0]
        year_idx_end = np.where(time_values_annual == endyear)[0][0]
        time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]
        
        var_glac_region_raw = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
        var_glac_region_raw_std = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 1]
        area_glac_region = np.repeat(ds['area_glac_annual'].values[:,year_idx_start:year_idx_end+1,0], 12, axis=1)
        
        # Area average
        volchg_monthly_glac_region = var_glac_region_raw
        volchg_monthly_glac_region_std = var_glac_region_raw_std

        # Merge datasets
        if region == regions[0]:
            var_glac_all = volchg_monthly_glac_region
            var_glac_all_std = volchg_monthly_glac_region_std
            area_glac_all = area_glac_region
        else:
            var_glac_all = np.concatenate((var_glac_all, volchg_monthly_glac_region), axis=0)
            var_glac_all_std = np.concatenate((var_glac_all_std, volchg_monthly_glac_region_std), axis=0)
            area_glac_all = np.concatenate((area_glac_all, area_glac_region), axis=0)
        try:
            ds.close()
        except:
            continue

    # Calibration data
    cal_data = pd.DataFrame()
    for dataset in cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset)
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
    
    # Link glacier index number from main_glac_rgi to cal_data to facilitate grabbing the data
    glacnodict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi.index.values))
    cal_data['glac_idx'] = cal_data['RGIId'].map(glacnodict)
    
    # Remove glaciers that don't have data over the entire glacier
    cal_data['elev_dif'] = cal_data['z2'] - cal_data['z1']
    main_glac_rgi['elev_range'] = main_glac_rgi['Zmax'] - main_glac_rgi['Zmin']
    # add elevation range to cal_data
    elevrange_dict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi['elev_range']))
    cal_data['elev_range'] = cal_data['RGIId'].map(elevrange_dict)
    # check difference (see if within 100 m of glacier)
    elev_margin_of_safety = 100
    cal_data['elev_check'] = cal_data['elev_dif'] - (cal_data['elev_range'] - elev_margin_of_safety)
    cal_data = cal_data[cal_data['elev_check'] > 0]
    cal_data.reset_index(drop=True, inplace=True)
    
    #%%
    cal_data['mb_mwe_era'] = np.nan
    cal_data['mb_mwea_era'] = np.nan
    cal_data['mb_mwe_era_std'] = np.nan
    for n in range(cal_data.shape[0]):
        glac_idx = cal_data.loc[n,'glac_idx']
        t1_idx = int(cal_data.loc[n,'t1_idx'])
        t2_idx = int(cal_data.loc[n,'t2_idx'])
        t1 = cal_data.loc[n,'t1']
        t2 = cal_data.loc[n,'t2']
        cal_data.loc[n,'mb_mwe_era'] = var_glac_all[glac_idx, t1_idx:t2_idx].sum()
        cal_data.loc[n,'mb_mwe_era_std'] = var_glac_all_std[glac_idx, t1_idx:t2_idx].sum() 
        cal_data.loc[n,'mb_mwe_era_std_rsos'] = ((var_glac_all_std[glac_idx, t1_idx:t2_idx]**2).sum())**0.5
        
    cal_data['mb_mwea_era'] = cal_data['mb_mwe_era'] / (cal_data['t2'] - cal_data['t1'])
    cal_data['mb_mwea_era_std'] = cal_data['mb_mwe_era_std'] / (cal_data['t2'] - cal_data['t1'])
    cal_data['mb_mwea_era_std_rsos'] = cal_data['mb_mwe_era_std_rsos'] / (cal_data['t2']-cal_data['t1'])
    cal_data['mb_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
    cal_data['year'] = (cal_data['t2'] + cal_data['t1']) / 2
    
    #%%
    # Determine whether data is seasonal or annual
    cal_data['time_difference'] = cal_data['t2'] - cal_data['t1']
    cal_data['seasonal/annual'] = 'summer'
    # Determine approximate center month of the seasonal/annual data
    cal_data['season_month'] = (cal_data['year'] - cal_data['year'].astype(int)) * 365 / 30
    cal_data.loc[cal_data['season_month'] < 4.5, 'seasonal/annual']  = 'winter'
    cal_data.loc[cal_data['season_month'] > 10.5, 'seasonal/annual']  = 'winter'
    cal_data.loc[cal_data['time_difference'] > 0.75, 'seasonal/annual']  = 'annual'
    
    # Remove data that spans less than a year
    #cal_data = cal_data[(cal_data['t2'] - cal_data['t1']) > 1]
    #cal_data.reset_index(drop=True, inplace=True)
    
    # Drop nan values
    cal_data.drop(index=np.where(np.isnan(cal_data.mb_mwe.values))[0], inplace=True)
    cal_data.reset_index(drop=True, inplace=True)
    
    #%%
    # Loop through conditions:
    condition_dict = OrderedDict()
    condition_dict['All']= cal_data.index.values
    condition_dict['All (annual)'] = (cal_data['t2'] - cal_data['t1']) >= 0.75
    condition_dict['All (seasonal)'] = (cal_data['t2'] - cal_data['t1']) < 0.75
    condition_dict['Geodetic'] = cal_data['obs_type'] == 'mb_geo'
    condition_dict['Glaciological'] = cal_data['obs_type'] == 'mb_glac'
    condition_dict['Glaciological (annual)'] = ((cal_data['seasonal/annual'] == 'annual') & 
                                                (cal_data['obs_type'] == 'mb_glac'))
    condition_dict['Glaciological (winter)'] = ((cal_data['seasonal/annual'] == 'winter') & 
                                                (cal_data['obs_type'] == 'mb_glac'))
    condition_dict['Glaciological (summer)'] = ((cal_data['seasonal/annual'] == 'summer') & 
                                                (cal_data['obs_type'] == 'mb_glac'))
#    condition_dict['Glaciological (> 1 yr)'] = ((cal_data['obs_type'] == 'mb_glac') & 
#                                                ((cal_data['t2'] - cal_data['t1']) >= 0.75))
#    condition_dict['Glaciological (< 1 yr)'] = ((cal_data['obs_type'] == 'mb_glac') & 
#                                                ((cal_data['t2'] - cal_data['t1']) < 0.75))
    
    stats_cns = ['group', 'count', 'rmse', 'r', 'slope', 'intercept', 'p-value', 'mae']
    group_stats = pd.DataFrame(np.zeros((len(condition_dict.keys()), len(stats_cns))), columns=stats_cns)
    group_stats['group'] = condition_dict.keys()
    
    for ncondition, cal_condition in enumerate(condition_dict.keys()):
#    for ncondition, cal_condition in enumerate(['Glaciological (annual)']):
        # Statistics for comparison
        cal_data_subset = cal_data.loc[condition_dict[cal_condition],:].copy()
        print('\n',cal_condition, cal_data_subset.shape[0])
        
        # Root-mean-square-deviation
        rmse = (np.sum((cal_data_subset.mb_mwea - cal_data_subset.mb_mwea_era)**2) / cal_data_subset.shape[0])**0.5
        print('  RMSE:', np.round(rmse,2))
        # Correlation
        slope, intercept, r_value, p_value, std_err = linregress(cal_data_subset.mb_mwea, cal_data_subset.mb_mwea_era)
        print('  r_value =', np.round(r_value,2), 'slope = ', np.round(slope,2), 
              'intercept = ', np.round(intercept,2), 'p_value = ', np.round(p_value,6))
        # Mean absolute error
        mae = np.mean(np.absolute(cal_data_subset.mb_mwea - cal_data_subset.mb_mwea_era))
        print('  mean absolute error:', np.round(mae,2))
        # Record stats
        group_stats.loc[ncondition, ['count', 'rmse', 'r', 'slope', 'intercept', 'p-value', 'mae']] = (
                [cal_data_subset.shape[0], rmse, r_value, slope, intercept, p_value, mae])

        
        cal_data_subset['dif_mb_mwea'] = cal_data_subset['mb_mwea'] - cal_data_subset['mb_mwea_era']
        print('  Difference stats: \n    Mean (+/-) std [mwea]:', 
          np.round(cal_data_subset['dif_mb_mwea'].mean(),2), '+/-', np.round(cal_data_subset['dif_mb_mwea'].std(),2), 
          'count:', cal_data_subset.shape[0],
          '\n    Median (+/-) std [mwea]:', 
          np.round(cal_data_subset['dif_mb_mwea'].median(),2), '+/- XXX', 
#          np.round(cal_data_subset['dif_mb_mwea'].std(),2),
          '\n    Mean standard deviation (correlated):',np.round(cal_data_subset['mb_mwea_era_std'].mean(),2),
          '\n    Mean standard deviation (uncorrelated):',np.round(cal_data_subset['mb_mwea_era_std_rsos'].mean(),2))
        
    group_stats.to_csv(output_fp + 'wgms_stats.csv', index=False)

    #%%
    
    # ===== PLOT =====
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0.3, 'hspace':0})
    
    datatypes = ['mb_geo', 'mb_glac']
    cmap = 'RdYlBu_r'
    norm = plt.Normalize(startyear, endyear)
    
    for nplot, datatype in enumerate(datatypes):
#    for nplot, datatype in enumerate(['mb_geo']):
        cal_data_plot = cal_data[cal_data['obs_type'] == datatype].copy()
        cal_data_plot.reset_index(drop=True, inplace=True)
        cal_data_plot['circ_size'] = 4
        cal_data_plot.loc[cal_data_plot['area_km2'] > 1, 'circ_size'] = 10
        cal_data_plot.loc[cal_data_plot['area_km2'] > 5, 'circ_size'] = 30
        cal_data_plot.loc[cal_data_plot['area_km2'] > 10, 'circ_size'] = 60
#        cal_data_plot.loc[cal_data_plot['area_km2'] > 20, 'circ_size'] = 60
        

        if datatype == 'mb_geo':
            # All glaciers
            a = ax[0,nplot].scatter(cal_data_plot.mb_mwea.values, cal_data_plot.mb_mwea_era.values, 
#                                    c=cal_data_plot['year'].values, cmap=cmap, norm=norm, 
                                    color='k',
                                    zorder=3, 
#                                    s=15,
                                    s=cal_data_plot.circ_size.values,
                                    marker='o')
            a.set_facecolor('none')
#            ymin = -1.25
#            ymax = 0.6
#            xmin = -1.25
#            xmax = 0.6
            ymin = -2.5
            ymax = 2.5
            xmin = -2.5
            xmax = 2.5
            ax[0,nplot].set_xlim(xmin,xmax)
            ax[0,nplot].set_ylim(ymin,ymax)
            ax[0,nplot].plot([np.min([xmin,ymin]),np.max([xmax,ymax])], [np.min([xmin,ymin]),np.max([xmax,ymax])], 
                             color='k', linewidth=0.25, zorder=1)
#            ax[0,nplot].yaxis.set_ticks(np.arange(-1, ymax+0.1, 0.5))
#            ax[0,nplot].xaxis.set_ticks(np.arange(-1, xmax+0.11, 0.5))
#            ax[0,nplot].yaxis.set_ticks(np.arange(-1, ymax+0.1, 0.5))
#            ax[0,nplot].xaxis.set_ticks(np.arange(-1, xmax+0.11, 0.5))
            
            ax[0,nplot].set_ylabel('$\mathregular{B_{mod}}$ (m w.e. $\mathregular{yr^{-1}}$)', labelpad=0, size=12)
            ax[0,nplot].set_xlabel('$\mathregular{B_{geo}}$ (m w.e. $\mathregular{yr^{-1}}$)', labelpad=0, size=12)
            # Add text
            ax[0,nplot].text(0.05, 0.95, 'E', va='center', size=12, fontweight='bold', transform=ax[0,nplot].transAxes)
            ax[0,nplot].text(0.7, 0.1, 'n=' + str(cal_data_plot.shape[0]) + '\n' + 
                             str(cal_data_plot.glacno.unique().shape[0]) + ' glaciers', va='center', ha='center', 
                             size=12, transform=ax[0,nplot].transAxes)
            slope, intercept, r_value, p_value, std_err = linregress(cal_data_plot.mb_mwea.values, 
                                                                     cal_data_plot.mb_mwea_era.values)
            print(datatype, 'r_value [mwea] =', r_value)
        
        elif datatype == 'mb_glac':
            glac_alpha = 1
            # All glaciers
#            a = ax[0,nplot].scatter(cal_data_plot.mb_mwe.values, cal_data_plot.mb_mwe_era.values, 
#                                    c=cal_data_plot['year'].values, cmap=cmap, norm=norm, 
#                                    zorder=3, s=15, marker='o')
            # Annual
            cal_data_plot_annual = cal_data_plot[cal_data_plot['seasonal/annual'] == 'annual']
            print('annual measurements:', len(cal_data_plot_annual))
            a = ax[0,nplot].scatter(cal_data_plot_annual.mb_mwe.values, cal_data_plot_annual.mb_mwe_era.values, 
                                    color='k', zorder=4, 
                                    s=cal_data_plot.circ_size.values,
#                                    s=15, 
                                    marker='o', alpha=glac_alpha)
            a.set_facecolor('none')
            cal_data_plot_summer = cal_data_plot[cal_data_plot['seasonal/annual'] == 'summer']
            print('summer measurements:', len(cal_data_plot_summer))
            b = ax[0,nplot].scatter(cal_data_plot_summer.mb_mwe.values, cal_data_plot_summer.mb_mwe_era.values, 
                                    color='red', zorder=3, 
#                                    s=15, 
                                    s=cal_data_plot.circ_size.values,
                                    marker='o', alpha=glac_alpha)
            b.set_facecolor('none')
            cal_data_plot_winter = cal_data_plot[cal_data_plot['seasonal/annual'] == 'winter']
            print('winter measurements:', len(cal_data_plot_winter))
            b = ax[0,nplot].scatter(cal_data_plot_winter.mb_mwe.values, cal_data_plot_winter.mb_mwe_era.values, 
                                    color='blue', zorder=3, 
#                                    s=15, 
                                    s=cal_data_plot.circ_size.values,
                                    marker='o', alpha=glac_alpha)
            b.set_facecolor('none')
            ymin = -2.5
            ymax = 2.5
            xmin = -2.5
            xmax = 2.5
            ax[0,nplot].set_xlim(xmin,xmax)
            ax[0,nplot].set_ylim(ymin,ymax)
            ax[0,nplot].plot([np.min([xmin,ymin]),np.max([xmax,ymax])], [np.min([xmin,ymin]),np.max([xmax,ymax])], 
                             color='k', linewidth=0.25, zorder=1)
            ax[0,nplot].yaxis.set_ticks(np.arange(-2, ymax+0.1, 1))
            ax[0,nplot].xaxis.set_ticks(np.arange(-2, xmax+0.11, 1))
            
            
            ax[0,nplot].set_ylabel('$\mathregular{B_{mod}}$ (m w.e.)', labelpad=0, size=12)
            ax[0,nplot].set_xlabel('$\mathregular{B_{glac}}$ (m w.e.)', labelpad=2, size=12)
            # Add text
            ax[0,nplot].text(0.05, 0.95, 'F', va='center', size=12, fontweight='bold', transform=ax[0,nplot].transAxes)
            ax[0,nplot].text(0.7, 0.1, 'n=' + str(cal_data_plot.shape[0]) + '\n' + 
                             str(cal_data_plot.glacno.unique().shape[0]) + ' glaciers', va='center', ha='center', 
                             size=12, transform=ax[0,nplot].transAxes)
            # Correlation coefficient
            slope, intercept, r_value, p_value, std_err = linregress(cal_data_plot.mb_mwe.values, 
                                                                     cal_data_plot.mb_mwe_era.values)
            print(datatype, 'r_value [mwe] =', r_value)
    
    # Add title
    #ax[0,nplot].set_title('Mass Balance (m w.e.)', size=12)

    # Add colorbar for years
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm._A = []
#    fig.subplots_adjust(right=0.9)
#    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.66])
#    cbar = fig.colorbar(sm, cax=cbar_ax)
#    fig.text(0.93,0.845, 'Year', size=12)
    
    
#    colorbar_dict = {'volume_norm':[0,1],
#                     'runoff_glac_monthly':[2020,2080]}
#    cmap = mpl.cm.RdYlBu
#    norm = plt.Normalize(colorbar_dict[vn][0], colorbar_dict[vn][1])
#      
#    # Add colorbar
#    cmap_alpha=1
#    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#    sm._A = []
#    cax = plt.axes([0.92, 0.38, 0.015, 0.23])
#    cbar = plt.colorbar(sm, ax=ax, cax=cax, orientation='vertical', alpha=cmap_alpha)
#    cax.xaxis.set_ticks_position('top')
#    cax.xaxis.set_tick_params(pad=0)
#    cbar.ax.tick_params(labelsize=6)
#    for n, label in enumerate(cax.xaxis.get_ticklabels()):
#        if n%2 != 0:
#            label.set_visible(False)
#    fig.text(0.965, 0.63, 'Year', ha='center', va='center', size=7)
    
    # SIZE LEGEND      
    s_sizes = [4,10,30,60]
    marker_linecolor='grey'
    marker_linewidth = 1
    circ1 = ax[0,nplot].scatter([0],[0], s=s_sizes[0], marker='o', color='grey', 
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ1.set_facecolor('none')
    circ2 = ax[0,nplot].scatter([0],[0], s=s_sizes[1], marker='o', color='grey',
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ2.set_facecolor('none')
    circ3 = ax[0,nplot].scatter([0],[0], s=s_sizes[2], marker='o', color='grey',
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ3.set_facecolor('none')
    circ4 = ax[0,nplot].scatter([0],[0], s=s_sizes[3], marker='o', color='grey',
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ4.set_facecolor('none')
    leg_fontsize = 10
    legend=fig.legend([circ1,circ2,circ3,circ4], ['0.1', '1', '5', '10'], 
              scatterpoints=1, ncol=1,
              loc='upper left',  bbox_to_anchor=(0.865,0.63),
              fontsize=leg_fontsize, labelspacing=0.3, columnspacing=0,handletextpad=0, handlelength=1,
              borderpad=0.2, framealpha=0, borderaxespad=0.2,
              )
    fig.text(0.94, 0.62, 'Area\n(km$^{2}$)', ha='center', va='center', size=leg_fontsize)
    
    # COLOR LEGEND
    circ_c1 = ax[0,nplot].scatter([0],[0], s=s_sizes[2], marker='o', color='black', linewidth=marker_linewidth)
    circ_c1.set_facecolor('none')
    circ_c2 = ax[0,nplot].scatter([0],[0], s=s_sizes[2], marker='o', color='red', linewidth=marker_linewidth)
    circ_c2.set_facecolor('none')
    circ_c3 = ax[0,nplot].scatter([0],[0], s=s_sizes[2], marker='o', color='blue', linewidth=marker_linewidth)
    circ_c3.set_facecolor('none')
    legend_c=fig.legend([circ_c1,circ_c2,circ_c3], ['annual', 'summer', 'winter'], 
              scatterpoints=1, ncol=1,
              loc='upper left',  bbox_to_anchor=(0.74,0.93),
              fontsize=leg_fontsize, labelspacing=0.3, columnspacing=0,handletextpad=0, handlelength=1,
              borderpad=0.2, framealpha=1, borderaxespad=0.2,
              )
    legend.get_frame().set_linewidth(0.5)
#    fig.text(0.97, 0.45, 'Season', ha='center', va='center', size=leg_fontsize)
    
    # Save figure
    fig.set_size_inches(6.75,3)
    fig_fn = 'wgms2017_compare.png'
    fig.savefig(output_fp + fig_fn, bbox_inches='tight', dpi=600)
        
#%%
if option_dehecq_compare == 1:
    regions = [13, 14, 15]
    
    startyear=2000
    endyear=2017
    wateryear=1
    
    dehecq_fp = pygem_prms.main_directory + '/../dehecq_velocity_anomaly/'
    dehecq_dict = {'Bhutan': 'vel_ts_bhutan_2000_2017_f0_c0.25.txt',
                   'Everest': 'vel_ts_everest_2000_2017_f0_c0.25.txt',
                   'Hindu Kush': 'vel_ts_hindu_kush_2000_2017_f0_c0.25.txt',
                   'inner_TP': 'vel_ts_inner_TP_2000_2017_f0_c0.25.txt',
                   'Karakoram': 'vel_ts_karakoram_2000_2017_f0_c0.25.txt',
                   'Kunlun': 'vel_ts_kunlun_2000_2017_f0_c0.25.txt',
                   'Pamir': 'vel_ts_pamir_2000_2017_f0_c0.25.txt',
                   'Spiti Lahaul': 'vel_ts_spiti_lahaul_2000_2017_f0_c0.25.txt',
                   'tien_shan': 'vel_ts_tien_shan_2000_2017_f0_c0.25.txt',
                   'West Nepal': 'vel_ts_west_nepal_2000_2017_f0_c0.25.txt',
                   'Yigong': 'vel_ts_nyainqentang_2000_2017_f0_c0.25.txt'}
    
    grouping = 'kaab'
    
    netcdf_fp = netcdf_fp_era
    
    output_fp = netcdf_fp_era + 'figures/'
    if os.path.exists(output_fp) == False:
        os.makedirs(output_fp)
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(regions)    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)
    
    # Load mass balance data
    ds_all = {}  
    # Merge all data, then select group data
    for region in regions:      
        
        # Load datasets
        ds_fn = ('R' + str(region) + '_ERA-Interim_c2_ba1_100sets_1980_2017.nc')
        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
        
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
        # Extract start/end indices for calendar year!
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        time_values_yr = np.array([x.year for x in time_values_df])
        if pygem_prms.gcm_wateryear == 1:
            time_values_yr = np.array([x.year + 1 if x.month >= 10 else x.year for x in time_values_df])
        time_idx_start = np.where(time_values_yr == startyear)[0][0]
        time_idx_end = np.where(time_values_yr == endyear)[0][0]
        time_values_monthly_subset = time_values_monthly[time_idx_start:time_idx_end + 12]
        year_idx_start = np.where(time_values_annual == startyear)[0][0]
        year_idx_end = np.where(time_values_annual == endyear)[0][0]
        time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]
        

        var_glac_region_raw = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
        var_glac_region_raw_std = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 1]
        area_glac_region = np.repeat(ds['area_glac_annual'].values[:,year_idx_start:year_idx_end+1,0], 12, axis=1)
        
        # Area average
        volchg_monthly_glac_region = var_glac_region_raw * area_glac_region
        volchg_monthly_glac_region_std = var_glac_region_raw_std * area_glac_region

        # Merge datasets
        if region == regions[0]:
            var_glac_all = volchg_monthly_glac_region
            var_glac_all_std = volchg_monthly_glac_region_std
            area_glac_all = area_glac_region
        else:
            var_glac_all = np.concatenate((var_glac_all, volchg_monthly_glac_region), axis=0)
            var_glac_all_std = np.concatenate((var_glac_all_std, volchg_monthly_glac_region_std), axis=0)
            area_glac_all = np.concatenate((area_glac_all, area_glac_region), axis=0)
        try:
            ds.close()
        except:
            continue
    
#    ds_all = {}
#    ds_all_std = {}
#    for ngroup, group in enumerate(groups):
        

#%%
    ds_all = {}
    ds_all_std = {}
    for ngroup, group in enumerate(dehecq_dict.keys()):
#    for ngroup, group in enumerate(['Everest']): 
        # Sum volume change for group
        group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
        varchg_group = var_glac_all[group_glac_indices,:].sum(axis=0)
        area_group = area_glac_all[group_glac_indices,:].sum(axis=0)
        
        # Uncertainty associated with volume change based on subgroups
        #  sum standard deviations in each subgroup assuming that they are uncorrelated
        #  then use the root sum of squares using the uncertainty of each subgroup to get the uncertainty of the group
        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
        subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()
        
        subgroup_std = np.zeros((len(subgroups_subset), varchg_group.shape[0]))
        for nsubgroup, subgroup in enumerate(subgroups_subset):
            main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
            subgroup_indices = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist()
            # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
            subgroup_std[nsubgroup,:] = var_glac_all_std[subgroup_indices,:].sum(axis=0)
        varchg_group_std = (subgroup_std**2).sum(axis=0)**0.5        
        
        # Group's annual mass balance [mwea]
        mb_mwea_group = (varchg_group / area_group).reshape(-1,12).sum(1)
        # annual uncertainty is the sum of monthly stdev since assumed to be perfectly correlated
        mb_mwea_group_std = (varchg_group_std / area_group).reshape(-1,12).sum(1)
#        mb_mwea_group_std_rsos = ((varchg_group_std / area_group)**2).reshape(-1,12).sum(1)**0.5

        # Normalized to get anomaly
#        mb_mwea_group_anomaly = (mb_mwea_group - mb_mwea_group.mean()) / abs(mb_mwea_group.mean())
        mb_mwea_group_anomaly = mb_mwea_group - mb_mwea_group.mean()
        mb_mwea_group_anomaly_std = mb_mwea_group_std
        print('mb anomaly:', mb_mwea_group_anomaly)
        
                
        
        ds_all[group] = mb_mwea_group_anomaly
        ds_all_std[group] = mb_mwea_group_anomaly_std
    
    #%%
    fig, ax = plt.subplots(len(dehecq_dict.keys()), 1, squeeze=False, figsize=(10,8), 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    for ngroup, group in enumerate(dehecq_dict.keys()):
        
        vel_timeseries = pd.read_csv(dehecq_fp + dehecq_dict[group])
        vel_timeseries = vel_timeseries.rename(columns={' Year':'Year', ' mean':'mean', ' Q1':'Q1', ' Q2':'Q2', 
                                                        ' Q3':'Q3', ' VSmed':'VSmed'})
        vel_anomaly = vel_timeseries['mean'].values
        vel_anomaly_low = vel_timeseries['Q1'].values
        vel_anomaly_high = vel_timeseries['Q3'].values
        

        mb_anomaly = ds_all[group]
        mb_anomaly_std = ds_all_std[group]
        years = time_values_annual_subset
            
        # All glaciers
        ax[ngroup,0].plot(years, mb_anomaly, color='k', label='Mass Balance', zorder=3)
        ax[ngroup,0].fill_between(years, mb_anomaly + mb_anomaly_std, mb_anomaly - mb_anomaly_std, 
                                  facecolor='k', alpha=0.5, label=None, zorder=2)
        ax2 = ax[ngroup,0].twinx()    
        ax2.plot(years, vel_anomaly, color='b', label='Velocity', zorder=3)
        ax2.fill_between(years, vel_anomaly_low, vel_anomaly_high, 
                         facecolor='b', alpha=0.1, label=None, zorder=1)
#        ax[ngroup,0].set_ylim(-1.1,0.75)
        ax[ngroup,0].set_xlim(2000,2017)
        if ngroup == 0:
            ax[ngroup,0].legend(loc=(0.02,0.22), ncol=1, fontsize=10, frameon=False, handlelength=1.5, 
                                handletextpad=0.25, columnspacing=1, borderpad=0, labelspacing=0)
            ax2.legend(loc=(0.02,0.02), ncol=1, fontsize=10, frameon=False, handlelength=1.5, 
                                handletextpad=0.25, columnspacing=1, borderpad=0, labelspacing=0)
        if ngroup + 1 == len(dehecq_dict.keys()):
            ax[ngroup,0].set_xlabel('Year', size=12)
        else:
            ax[ngroup,0].xaxis.set_ticklabels([])
        ax[ngroup,0].text(0.5,0.8, group, horizontalalignment='center', transform=ax[ngroup,0].transAxes)
#        ax[ngroup,0].yaxis.set_ticks(np.arange(-1, 0.55, 0.5))
    
    # Add text
    fig.text(0, 0.5, 'Mass Balance (m w.e. $\mathregular{a^{-1}}$)', va='center', rotation='vertical', size=12)
    fig.text(0.98, 0.5, 'Velocity Anomaly (m $\mathregular{a^{-1}}$)', va='center', rotation='vertical', size=12)
#    fig.text(0.5, 0.845, 'Central Asia', horizontalalignment='center', zorder=4, color='black', fontsize=10)
#    fig.text(0.5, 0.59, 'South Asia West', horizontalalignment='center', zorder=4, color='black', fontsize=10)
#    fig.text(0.5, 0.34, 'South Asia East', horizontalalignment='center', zorder=4, color='black', fontsize=10)
#    fig.text(0.135, 0.845, 'A', zorder=4, color='black', fontsize=12, fontweight='bold')
#    fig.text(0.135, 0.59, 'B', zorder=4, color='black', fontsize=12, fontweight='bold')
#    fig.text(0.135, 0.34, 'C', zorder=4, color='black', fontsize=12, fontweight='bold')
    
    # Save figure
    fig.set_size_inches(6,10)
    fig.savefig(output_fp + 'Dehecq_vs_ERA-Interim_' + str(startyear) + '-' + str(endyear) + '.png', 
                bbox_inches='tight', dpi=300)    

if option_nick_snowline == 1:
    startyear = 2015
    endyear = 2018
    vn = 'snowline_glac_monthly'
    
    netcdf_fp = netcdf_fp_era
    
    output_fp = netcdf_fp_era + 'figures/'
    if os.path.exists(output_fp) == False:
        os.makedirs(output_fp)
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(regions)
    
    # Add dictionary to select glaciers
    # Group dictionaries
    trishuli_dict_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/qgis_himat/trishuli_RGIIds.csv'
    trishuli_csv = pd.read_csv(trishuli_dict_fn)
    trishuli_dict = dict(zip(trishuli_csv.RGIId, trishuli_csv.L1_Name))
    main_glac_rgi['trishuli'] = main_glac_rgi.RGIId.map(trishuli_dict)

    # Load mass balance data
    ds_all = {}  
    # Merge all data, then select group data
    for region in regions:      
        
        # Load datasets
        ds_fn = ('R' + str(region) + '_ERA-Interim_c2_ba1_100sets_1980_2017.nc')
        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
        
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
        # Extract start/end indices for calendar year!
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        
        var_glac_region = ds[vn].values[:,:,0]
        var_glac_region_std = ds[vn].values[:,:,1]

        # Merge datasets
        if region == regions[0]:
            var_glac_all = var_glac_region
            var_glac_all_std = var_glac_region_std
        else:
            var_glac_all = np.concatenate((var_glac_all, var_glac_region), axis=0)
            var_glac_all_std = np.concatenate((var_glac_all_std, var_glac_region_std), axis=0)
        try:
            ds.close()
        except:
            continue
        
    # Trishuli indices
    glac_idx = np.where(main_glac_rgi['trishuli'].values == 'trishuli')[0]
    trishuli_data = var_glac_all[glac_idx,:]
    trishuli_data_std = var_glac_all_std[glac_idx,:]
    main_glac_rgi_trishuli = main_glac_rgi.loc[glac_idx,:]
    
    A = np.array(list(main_glac_rgi_trishuli.loc[:,'RGIId'].values))
    B = [pd.to_datetime(str(x)).strftime('%Y-%m-%d') for x in list(time_values_df.values)]
    C = pd.DataFrame(trishuli_data, index=A, columns=B)
    C_std = pd.DataFrame(trishuli_data_std, index=A, columns=B)
    C.to_csv(pygem_prms.output_filepath + 'trishuli_snowline_1979_2017.csv')
    C_std.to_csv(pygem_prms.output_filepath + 'trishuli_snowline_1979_2017_std.csv')
    

#%%
if runoff_erainterim_bywatershed == 1:
    startyear = 2001
    endyear = 2018
    
    grouping = 'watershed'
    subgrouping = 'hexagon'
    
    netcdf_fp = netcdf_fp_era
    
    regions = [13, 14, 15]
    
    
    # Merge all data, then select group data
    for region in regions:      
        # Load datasets
        ds_fn = ('R' + str(region) + '--all--ERA-Interim_c2_ba1_100sets_2000_2018.nc')
        ds = xr.open_dataset(netcdf_fp_era + ds_fn)
        df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs.values)
        glacno_region = [str(int(df.loc[x,'O1Region'])) + '.' + str(int(df.loc[x,'glacno'])).zfill(5) 
                         for x in df.index.values]
        
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
        # Extract start/end indices for calendar year!
        time_values_df = pd.DatetimeIndex(time_values_monthly)
        time_values_yr = np.array([x.year for x in time_values_df])
        if ds.time.year_type == 'water year':
            time_values_yr = np.array([x.year + 1 if x.month >= 10 else x.year for x in time_values_df])
        elif ds.time.year_type == 'custom year':
            startmonth = int(pygem_prms.startmonthday.split('-')[0])
            time_values_yr = np.array([x.year + 1 if x.month >= startmonth else x.year for x in time_values_df])
        time_idx_start = np.where(time_values_yr == startyear)[0][0]
        time_idx_end = np.where(time_values_yr == endyear)[0][0]
        time_values_monthly_subset = time_values_monthly[time_idx_start:time_idx_end + 12]
        year_idx_start = np.where(time_values_annual == startyear)[0][0]
        year_idx_end = np.where(time_values_annual == endyear)[0][0]
        time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]

        var_glac_region = ds['runoff_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
        var_glac_region_std = ds['runoff_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 1]
        
        vol_glac_region = ds['volume_glac_annual'].values[:,year_idx_start:year_idx_end+1,0]
        excess_meltwater_reg = excess_meltwater_m3(vol_glac_region)
        
        melt_glac_region_mwea = ds['melt_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
        rfrz_glac_region_mwea = ds['refreeze_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
        area_glac_region = ds['area_glac_annual'].values[:,year_idx_start:year_idx_end+1,0]
        melt_glac_region_m3 = melt_glac_region_mwea * area_glac_region.repeat(12,axis=1) * 10**6
        rfrz_glac_region_m3 = rfrz_glac_region_mwea * area_glac_region.repeat(12,axis=1) * 10**6
        
        option_include_offglac = 1
        if option_include_offglac == 1:
            var_glac_region += ds['offglac_runoff_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
            var_glac_region_std += ds['offglac_runoff_monthly'].values[:,time_idx_start:time_idx_end + 12, 1]

        # Merge datasets
        if region == regions[0]:
            glacno = glacno_region
            var_glac_all = var_glac_region
            var_glac_all_std = var_glac_region_std
            excess_meltwater_all = excess_meltwater_reg
            melt_glac_all = melt_glac_region_m3
            rfrz_glac_all = rfrz_glac_region_m3
        else:
            glacno.extend(glacno_region)
            var_glac_all = np.concatenate((var_glac_all, var_glac_region), axis=0)
            var_glac_all_std = np.concatenate((var_glac_all_std, var_glac_region_std), axis=0)
            excess_meltwater_all = np.concatenate((excess_meltwater_all, excess_meltwater_reg), axis=0)
            melt_glac_all = np.concatenate((melt_glac_all, melt_glac_region_m3), axis=0)
            rfrz_glac_all = np.concatenate((rfrz_glac_all, rfrz_glac_region_m3), axis=0)
        
        ds.close()
    
    
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(glac_no = glacno)
        
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)
    
    output_cns = ['watershed', 'runoff_gta', 'runoff_interannual_std_gta', 'runoff_interannual_std_pc', 'std_corr_gta', 
                  'std_uncorr_gta', 'std_corr_perfect_gta', 'std_corr_perfect_pc', 'melt_gta',
                  'melt_interannual_std_gta', 'excess_meltwater_gta', 'excess_meltwater_interannual_std_gta', 
                  'rfrz_gta', 'rfrz_interannual_std_gta']
    output_df = pd.DataFrame(np.zeros((len(groups),len(output_cns))), columns=output_cns)
    output_df['watershed'] = groups

    ds_all = {}
    ds_all_std = {}
    print('Mean annual runoff (+/-) 1 std [Gt/yr],', str(startyear), '-', str(endyear),'(water years) from ERA-Interim')
    for ngroup, group in enumerate(groups):
#    for ngroup, group in enumerate(['Yellow']):
        # Sum volume change for group
        group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
        
        var_group = var_glac_all[group_glac_indices,:].sum(axis=0)
        var_group_std_pc = var_glac_all_std[group_glac_indices,:].sum(axis=0)
        excess_meltwater_group = excess_meltwater_all[group_glac_indices,:].sum(axis=0)
        melt_group = melt_glac_all[group_glac_indices,:].sum(axis=0)
        rfrz_group = rfrz_glac_all[group_glac_indices,:].sum(axis=0)
        
        # Uncertainty associated with volume change based on subgroups
        #  sum standard deviations in each subgroup assuming that they are uncorrelated
        #  then use the root sum of squares using the uncertainty of each subgroup to get the uncertainty of the group
        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
        subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()
        
        subgroup_std = np.zeros((len(subgroups_subset), var_group.shape[0]))
        for nsubgroup, subgroup in enumerate(subgroups_subset):
            main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
            subgroup_indices = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist()
            # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
            subgroup_std[nsubgroup,:] = var_glac_all_std[subgroup_indices,:].sum(axis=0)
        var_group_std = (subgroup_std**2).sum(axis=0)**0.5    
        
        # Group's mean annual runoff
        group_annual_excess_melt_Gta = excess_meltwater_group.sum() / excess_meltwater_group.shape[0] * (1/1000)**3
        group_annual_excess_melt_Gta_interannual_std = excess_meltwater_group.std() * (1/1000)**3
        group_annual_melt_Gta = melt_group.sum() / (melt_group.shape[0] / 12) * (1/1000)**3
        group_annual_melt_Gta_interannual_std = melt_group.reshape(-1,12).sum(1).std() * (1/1000)**3
        group_annual_rfrz_Gta = rfrz_group.sum() / (rfrz_group.shape[0] / 12) * (1/1000)**3
        group_annual_rfrz_Gta_interannual_std = rfrz_group.reshape(-1,12).sum(1).std() * (1/1000)**3
        
        group_annual_runoff_Gta = var_group.sum() / (var_group.shape[0] / 12) * (1/1000)**3
        group_annual_runoff_Gta_interannual_std = var_group.reshape(-1,12).sum(1).std() * (1/1000)**3
        group_annual_runoff_Gta_pc = var_group_std_pc.sum() / (var_group_std_pc.shape[0] / 12) * (1/1000)**3
        # annual uncertainty is the sum of monthly stdev since assumed to be perfectly correlated
        #  take mean of that to get average uncertainty over 18 years
        group_annual_runoff_Gta_std = var_group_std.reshape(-1,12).sum(1).mean() * (1/1000)**3
        group_annual_runoff_Gta_std_rsos = ((var_group_std**2).reshape(-1,12).sum(1)**0.5).mean() * (1/1000)**3

        ds_all[group] = group_annual_runoff_Gta
        ds_all_std[group] = group_annual_runoff_Gta_std
        
        output_df.loc[ngroup,'runoff_gta'] = group_annual_runoff_Gta
        output_df.loc[ngroup,'runoff_interannual_std_gta'] = group_annual_runoff_Gta_interannual_std
        output_df.loc[ngroup,'std_corr_gta'] = group_annual_runoff_Gta_std
        output_df.loc[ngroup,'std_uncorr_gta'] = group_annual_runoff_Gta_std_rsos
        output_df.loc[ngroup,'std_corr_perfect_gta'] = group_annual_runoff_Gta_pc
        output_df.loc[ngroup,'melt_gta'] = group_annual_melt_Gta
        output_df.loc[ngroup,'melt_interannual_std_gta'] = group_annual_melt_Gta_interannual_std
        output_df.loc[ngroup,'excess_meltwater_gta'] = group_annual_excess_melt_Gta
        output_df.loc[ngroup,'excess_meltwater_interannual_std_gta'] = group_annual_excess_melt_Gta_interannual_std
        output_df.loc[ngroup,'rfrz_gta'] = group_annual_rfrz_Gta
        output_df.loc[ngroup,'rfrz_interannual_std_gta'] = group_annual_rfrz_Gta_interannual_std
        
        print(group, np.round(group_annual_runoff_Gta,3), '+/-', np.round(group_annual_runoff_Gta_std,3), '(',
              np.round(group_annual_runoff_Gta_std_rsos,3),'for uncorrelated)', 
              '\n  all perfectly correlated:', np.round(group_annual_runoff_Gta_pc,3),
              '\n  interannual std:', np.round(group_annual_runoff_Gta_interannual_std,3))
    
    output_df['runoff_interannual_std_pc'] = output_df['runoff_interannual_std_gta'] / output_df['runoff_gta'] * 100
    output_df['std_corr_perfect_pc'] = output_df['std_corr_gta'] / output_df['runoff_gta'] * 100
    
    output_df.to_csv(pygem_prms.output_sim_fp + 'watershed_eraint_' + str(time_values_annual[0]) + '-' + 
                     str(time_values_annual[-1]) + '_runoff.csv', index=False)
        
#%%

        
if option_merge_multimodel_datasets == 1:
    ds1 = xr.open_dataset(netcdf_fp_cmip5 + 'R14_multimodel_rcp45_c2_ba1_100sets_2000_2100.nc')
    ds2 = xr.open_dataset(netcdf_fp_cmip5 + 'R14_multimodel_rcp45_c2_ba1_100sets_2000_2100_good4volume.nc')
    
    #ds_vns = ['prec_glac_monthly, acc, refreeze, melt, frontal ablation, massbal
              
    ds_vns = ['runoff_glac_monthly', 'volume_glac_annual']
    
    ds3 = ds1.copy()
    
    for vn in ds_vns:
        print(vn)
        
        ds3[vn].values = ds2[vn].values
        
    ds3['area_glac_annual_allgcms'] = ds2['area_glac_annual']
    
    # Export merged dataset
    # Encoding
    # add variables to empty dataset and merge together
    encoding = {}
    noencoding_vn = ['stats', 'glac_attrs']
    if pygem_prms.output_package == 2:
        for encoding_vn in pygem_prms.output_variables_package2:
            # Encoding (specify _FillValue, offsets, etc.)
            if encoding_vn not in noencoding_vn:
                encoding[encoding_vn] = {'_FillValue': False}
    
    ds3.to_netcdf(netcdf_fp_cmip5 + 'R14_multimodel_rcp45_c2_ba1_100sets_2000_2100_modified.nc', encoding=encoding)


#%% PROPOSAL FIGURES
if option_runoff_components_proposal == 1:
    figure_fp = pygem_prms.main_directory + '/../HiMAT_2/figures/'
    
    startyear = 2015
    endyear = 2100
    
    grouping = 'watershed'
    peakwater_Nyears = 11
    
    startyear=2000
    endyear=2100

    ref_startyear = 2000
    ref_endyear = 2015
    
    plt_startyear = 2015
    plt_endyear = 2100

    multimodel_linewidth = 2
    alpha=0.2
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(regions)
    
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)
    if grouping == 'watershed':
        groups.remove('Irrawaddy')
        groups.remove('Yellow')

    # Glacier and grouped annual specific mass balance and mass change
    
#    for rcp in rcps:
    for rcp in ['rcp45']:
        print(rcp)
        ds_vn = {}
        ds_vn_std = {}
        
        ds_vns = ['volume_glac_annual', 'area_glac_annual', 
                  'prec_glac_monthly', 'acc_glac_monthly', 'refreeze_glac_monthly', 'melt_glac_monthly',
                  'offglac_prec_monthly', 'offglac_refreeze_monthly', 'offglac_melt_monthly']
        ds_vns_needarea = ['prec_glac_monthly', 'acc_glac_monthly', 'refreeze_glac_monthly', 'melt_glac_monthly',
                           'offglac_prec_monthly', 'offglac_refreeze_monthly', 'offglac_melt_monthly']
        
        for region in regions:
            
            # Load datasets
            ds_fn = 'R' + str(region) + '_multimodel_' + rcp + '_c2_ba1_100sets_2000_2100.nc'
            ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
            
            # Extract time variable
            time_values_annual = ds.coords['year_plus1'].values
            time_values_monthly = ds.coords['time'].values
            
            for vn in ds_vns:
                if region == regions[0]: 
                    ds_vn[vn] = ds[vn].values[:,:,0]
                    ds_vn_std[vn] = ds[vn].values[:,:,1]
                else:
                    ds_vn[vn] = np.concatenate((ds_vn[vn], ds[vn].values[:,:,0]), axis=0)
                    ds_vn_std[vn] = np.concatenate((ds_vn_std[vn], ds[vn].values[:,:,1]), axis=0)
            
                # Remove negative values in off glacier caused by glacier advance
                if 'offglac' in vn:
                    ds_vn[vn][ds_vn[vn] < 0] = 0
            ds.close()
            
        # Convert to annual
        ds_vn_annual = {}
        for vn in ds_vns:
            if 'monthly' in vn:
                ds_vn_annual[vn] = gcmbiasadj.annual_sum_2darray(ds_vn[vn])
            else:
                ds_vn_annual[vn] = ds_vn[vn]
                
        # Excess glacier meltwater based on volume change
        ds_vn_annual['excess_melt_annual'] = excess_meltwater_m3(ds_vn_annual['volume_glac_annual'])
        ds_vns.append('excess_melt_annual')
            
        #%%
        # Groups
        count = 0
        group_vn_annual = {}
        for ngroup, group in enumerate(groups):
            # Select subset of data
            group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
            
            group_vn_annual[group] = {}
            for vn in ds_vns:                
                if vn in ds_vns_needarea:
                    if 'offglac' in vn:                        
                        offglac_area_annual = (ds_vn_annual['area_glac_annual'][:,0][:,np.newaxis] - 
                                               ds_vn_annual['area_glac_annual'])
                        offglac_area_annual[offglac_area_annual < 0] = 0
                        group_vn_annual[group][vn] = (
                                (offglac_area_annual[group_glac_indices,:-1] * 10**6 * 
                                 ds_vn_annual[vn][group_glac_indices,:]).sum(axis=0))
                        
                    else:
                        group_vn_annual[group][vn] = (
                                (ds_vn_annual['area_glac_annual'][group_glac_indices,:-1] * 10**6 * 
                                 ds_vn_annual[vn][group_glac_indices,:]).sum(axis=0))
                else:
                    group_vn_annual[group][vn] = ds_vn_annual[vn][group_glac_indices,:].sum(axis=0)
                
            group_vn_annual[group]['runoff_glac_monthly'] = (
                    group_vn_annual[group]['melt_glac_monthly'] + group_vn_annual[group]['prec_glac_monthly'] - 
                    group_vn_annual[group]['refreeze_glac_monthly'])
            group_vn_annual[group]['offglac_runoff_monthly'] = (
                    group_vn_annual[group]['offglac_melt_monthly'] + group_vn_annual[group]['offglac_prec_monthly'] - 
                    group_vn_annual[group]['offglac_refreeze_monthly'])
            group_vn_annual[group]['total_runoff_monthly'] = (
                    group_vn_annual[group]['offglac_runoff_monthly'] + group_vn_annual[group]['runoff_glac_monthly'])
        
        #%%
        # Peakwater
        print('Peakwater by group for', rcp)
        nyears = 11
        group_peakwater = {}
        for ngroup, group in enumerate(groups):
            group_peakwater[group] = peakwater(group_vn_annual[group]['total_runoff_monthly'], 
                                               time_values_annual[:-1], nyears)
            print(group, group_peakwater[group][0], '\n  peakwater_chg[%]:', np.round(group_peakwater[group][1],0),
                  '\n  2100 chg[%]:', np.round(group_peakwater[group][2],0))
        
        if grouping == 'watershed':
            # Add Aral Sea (Amu Darya + Syr Darya) for comparison with HH2019
            group = 'Aral_Sea'
            group_peakwater['Aral_Sea'] = peakwater(group_vn_annual['Amu_Darya']['total_runoff_monthly'] + 
                                                    group_vn_annual['Syr_Darya']['total_runoff_monthly'], 
                                                    time_values_annual[:-1], nyears)
            print(group, group_peakwater[group][0], '\n  peakwater_chg[%]:', np.round(group_peakwater[group][1],0),
                  '\n  2100 chg[%]:', np.round(group_peakwater[group][2],0))
        
        #%%
        multimodel_linewidth = 1
        alpha=0.2
        
        groups_select = ['Indus', 'Brahmaputra']
            
        fig, ax = plt.subplots(1, len(groups_select), squeeze=False, sharex=False, sharey=True, 
                               gridspec_kw = {'wspace':0, 'hspace':0})
        add_group_label = 1
        
        # Cycle through groups  
        row_idx = 0
        col_idx = 0
        for ngroup, group in enumerate(groups_select):
            col_idx = ngroup
                
            # Time indices
            t1_idx_ref = np.where(time_values_annual == ref_startyear)[0][0]
            t2_idx_ref = np.where(time_values_annual == ref_endyear)[0][0] + 1

            # Multi-model statistics
            runoff_total = group_vn_annual[group]['total_runoff_monthly']
            runoff_glac_total = group_vn_annual[group]['runoff_glac_monthly']
            runoff_glac_melt = group_vn_annual[group]['melt_glac_monthly']
            runoff_glac_excess = group_vn_annual[group]['excess_melt_annual']
            runoff_glac_prec = group_vn_annual[group]['prec_glac_monthly']
            runoff_glac_refreeze = group_vn_annual[group]['refreeze_glac_monthly']
            runoff_offglac_melt = group_vn_annual[group]['offglac_melt_monthly']
            runoff_offglac_prec = group_vn_annual[group]['offglac_prec_monthly']
            runoff_offglac_refreeze = group_vn_annual[group]['offglac_refreeze_monthly']
            runoff_total_normalizer = runoff_total[t1_idx_ref:t2_idx_ref].mean()
            
            # Normalize values
            runoff_total_norm = runoff_total / runoff_total_normalizer
            runoff_glac_total_norm = runoff_glac_total / runoff_total_normalizer 
            runoff_glac_melt_norm = runoff_glac_melt / runoff_total_normalizer
            runoff_glac_excess_norm = runoff_glac_excess / runoff_total_normalizer
            runoff_glac_prec_norm = runoff_glac_prec / runoff_total_normalizer
            runoff_glac_refreeze_norm = runoff_glac_refreeze / runoff_total_normalizer
            runoff_offglac_prec_norm = runoff_offglac_prec / runoff_total_normalizer
            runoff_offglac_melt_norm = runoff_offglac_melt / runoff_total_normalizer
            runoff_offglac_refreeze_norm = runoff_offglac_refreeze / runoff_total_normalizer

            t1_idx = np.where(time_values_annual == plt_startyear)[0][0]
            t2_idx = np.where(time_values_annual == plt_endyear)[0][0] + 1
            
            # Plot
            # Total runoff (line)
            ax[row_idx, col_idx].plot(time_values_annual[t1_idx:t2_idx], runoff_total_norm[t1_idx:t2_idx], 
                                      color='k', linewidth=0.5, zorder=4)
            
            # Components
            # Glacier melt - excess on bottom (green fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    0, 
                    runoff_glac_melt_norm[t1_idx:t2_idx] - runoff_glac_excess_norm[t1_idx:t2_idx],
                    facecolor='green', alpha=0.2, label='glac melt', zorder=3)
            # Excess glacier melt on bottom (green fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx] - runoff_glac_excess_norm[t1_idx:t2_idx],
                    facecolor='darkgreen', alpha=0.4, label='glac excess', zorder=3)
            # Off-Glacier melt (blue fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx],
                    runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx],
                    facecolor='blue', alpha=0.2, label='offglac melt', zorder=3)
            # Glacier precipitation (yellow fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx],
                    (runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx] + 
                     runoff_glac_prec_norm[t1_idx:t2_idx]),
                    facecolor='yellow', alpha=0.2, label='glacier prec', zorder=3)
            # Off-glacier precipitation (red fill)
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], 
                    (runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx] + 
                     runoff_glac_prec_norm[t1_idx:t2_idx]),
                    (runoff_glac_melt_norm[t1_idx:t2_idx] + runoff_offglac_melt_norm[t1_idx:t2_idx] + 
                     runoff_glac_prec_norm[t1_idx:t2_idx] + runoff_offglac_prec_norm[t1_idx:t2_idx]),
                    facecolor='yellow', alpha=0.2, label='offglac prec', zorder=3)
            
            # Group labels
#            if add_group_label == 1:
#                ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=8, horizontalalignment='center', 
#                                          verticalalignment='top', transform=ax[row_idx, col_idx].transAxes)
            ax[row_idx, col_idx].text(0.5, 0.99, 'RCP ' + rcp_dict[rcp], size=10, horizontalalignment='center', 
                                          verticalalignment='top', transform=ax[row_idx, col_idx].transAxes)
    
            # X-label
            ax[row_idx, col_idx].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                          time_values_annual[t1_idx:t2_idx].max())
#            ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=20)
            ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(50))
            ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
#            if col_idx == 0:
#                ax[row_idx, col_idx].set_xticklabels(['2015','2050','2100'])
#            else:
            ax[row_idx, col_idx].set_xticklabels(['','2050','2100'])
                
            # Y-label
            ax[row_idx, col_idx].set_ylim(0,1.8)
            ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[row_idx, col_idx].set_yticklabels(['','0','0.5','1.0','1.5', ''])

            # Tick parameters
            ax[row_idx, col_idx].yaxis.set_ticks_position('both')
            ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=8, direction='inout', pad=0)
            ax[row_idx, col_idx].tick_params(axis='x', which='major', labelsize=8, direction='inout', pad=0)
            ax[row_idx, col_idx].tick_params(axis='both', which='minor', labelsize=2, direction='inout', pad=0)     
            
            for axis in ['top','bottom','left','right']:
                ax[row_idx, col_idx].spines[axis].set_linewidth(0.5)
#                ax[row_idx, col_idx].spines[axis].set_zorder(0)
            
            if group == groups_select[0]:
                text_size = 10
                ax[row_idx, col_idx].text(0.5, 0.05, 'glacier melt', size=text_size, horizontalalignment='center', 
                                          verticalalignment='bottom', transform=ax[row_idx, col_idx].transAxes)
                ax[row_idx, col_idx].text(0.5, 0.33, 'excess', size=text_size, horizontalalignment='center', 
                                          verticalalignment='bottom', transform=ax[row_idx, col_idx].transAxes)
            if group == 'Brahmaputra':
                ax[row_idx, col_idx].text(0.6, 0.1, 'snow', size=text_size, horizontalalignment='left', 
                                          verticalalignment='bottom', transform=ax[row_idx, col_idx].transAxes)
                ax[row_idx, col_idx].text(0.05, 0.33, 'precipitation', size=text_size, horizontalalignment='left', 
                                          verticalalignment='bottom', transform=ax[row_idx, col_idx].transAxes)

        # Line legend
#        leg_alpha = 0.2
#        leg_list = ['Total runoff', 'Glacier runoff',
#                    'Off-glacier\nprecipitation', 'Glacier\nprecipitation', 'Off-glacier\nmelt', 
#                    'Off-glacier\nrefreeze', 'Glacier melt\n(excess)', 'Glacier melt\n(equilibrium)', 
#                    'Glacier\nrefreeze']
#        line_dict = {'Total runoff':['black',1,'-',1,''], 'Glacier runoff':['black',1,'--',1,''],
#                     'Glacier melt\n(equilibrium)':['green',5,'-',leg_alpha,''], 
#                     'Glacier melt\n(excess)':['darkgreen',5,'-',0.4,''], 
#                     'Glacier\nprecipitation':['yellow',5,'-',leg_alpha,''],
#                     'Glacier\nrefreeze':['grey',5,'-',leg_alpha,'////'],
#                     'Off-glacier\nmelt':['blue',5,'-',leg_alpha,''], 
#                     'Off-glacier\nprecipitation':['red',5,'-',leg_alpha,''],
#                     'Off-glacier\nrefreeze':['grey',5,'-',leg_alpha,'....']}
#        leg_lines = []
#        leg_labels = []
#        for vn_label in leg_list:
#            if 'refreeze' in vn_label:
#                line = mpatches.Patch(facecolor=line_dict[vn_label][0], alpha=line_dict[vn_label][3], 
#                                      hatch=line_dict[vn_label][4])
#            else:
#                line = Line2D([0,1],[0,1], color=line_dict[vn_label][0], linewidth=line_dict[vn_label][1], 
#                              linestyle=line_dict[vn_label][2], alpha=line_dict[vn_label][3])
#            leg_lines.append(line)
#            leg_labels.append(vn_label)
#        fig.subplots_adjust(right=0.83)
#        fig.legend(leg_lines, leg_labels, loc=(0.83,0.38), fontsize=10, labelspacing=0.5, handlelength=1, ncol=1,
#                   handletextpad=0.5, borderpad=0, frameon=False)

        # Label
        ylabel_str = 'Runoff (-)'
        fig.text(-0.01, 0.5, ylabel_str, va='center', rotation='vertical', size=text_size)
        
        fig.set_size_inches(2.5, 1.25)
        
        figure_fn = 'runoffcomponents_mulitmodel_' + rcp +  '.png'
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
    
if option_plot_cmip5_normalizedchange_proposal == 1:
    netcdf_fp_cmip5 = '/Volumes/LaCie/PyGEM_simulations/spc_subset/'
    
#    vn = 'volume_glac_annual'
    vn = 'runoff_glac_monthly'
    
    if 'runoff' in vn:
        grouping = 'watershed'
#        grouping = 'all'
    
    # Peakwater running mean years
    nyears=11
    
    startyear = 2015
    endyear = 2100
    

    figure_fp = pygem_prms.main_directory + '/../HiMAT_2/figures/'
    runoff_fn_pkl = pygem_prms.output_sim_fp + 'figures/' + 'watershed_runoff_annual_22gcms_4rcps.pkl'
    option_plot_individual_gcms = 0
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(regions)
    # Groups
    groups, group_cn = select_groups(grouping, main_glac_rgi)

    # Load data
    if vn == 'runoff_glac_monthly' and os.path.isfile(runoff_fn_pkl):
        with open(runoff_fn_pkl, 'rb') as f:
            ds_all = pickle.load(f)
        # Load single GCM to get time values needed for plot
        ds_fn = 'R' + str(regions[0]) + '_' + gcm_names[0] + '_' + rcps[0] + '_c2_ba1_100sets_2000_2100--subset.nc'
        ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
        # Extract time variable
        time_values_annual = ds.coords['year_plus1'].values
        time_values_monthly = ds.coords['time'].values
    else:
        # Load data
        ds_all = {}
        ds_std_all = {}
        for rcp in rcps:
            ds_all[rcp] = {}
            ds_std_all[rcp] = {}
            
            for ngcm, gcm_name in enumerate(gcm_names):
                ds_all[rcp][gcm_name] = {}
                ds_std_all[rcp][gcm_name] = {}
                
                print(rcp, gcm_name)
            
                # Merge all data, then select group data
                for region in regions:        
    
                    # Load datasets
                    ds_fn = ('R' + str(region) + '_' + gcm_name + '_' + rcp + '_c2_ba1_100sets_2000_2100--subset.nc')
      
                    # Bypass GCMs that are missing a rcp scenario
                    try:
                        ds = xr.open_dataset(netcdf_fp_cmip5 + ds_fn)
                        skip_gcm = 0
                    except:
                        skip_gcm = 1
                        print('Skip', gcm_name, rcp, region)
                    
                    if skip_gcm == 0:
                        # Extract time variable
                        time_values_annual = ds.coords['year_plus1'].values
                        time_values_monthly = ds.coords['time'].values
                        # Extract data
                        vn_glac_region = ds[vn].values[:,:,0]
                        vn_glac_std_region = ds[vn].values[:,:,1]
                        
                        # Convert monthly values to annual
                        if vn == 'runoff_glac_monthly':
                            vn_offglac_region = ds['offglac_runoff_monthly'].values[:,:,0]
                            vn_offglac_std_region = ds['offglac_runoff_monthly'].values[:,:,1]                                
                            vn_glac_region += vn_offglac_region
                            vn_glac_std_region += vn_offglac_std_region                            
                            
                            vn_glac_region = gcmbiasadj.annual_sum_2darray(vn_glac_region)
                            time_values_annual = time_values_annual[:-1]                    
                            vn_glac_std_region = gcmbiasadj.annual_sum_2darray(vn_glac_std_region)
                            
                        # Merge datasets
                        if region == regions[0]:
                            vn_glac_all = vn_glac_region
                            vn_glac_std_all = vn_glac_std_region                        
                        else:
                            vn_glac_all = np.concatenate((vn_glac_all, vn_glac_region), axis=0)
                            vn_glac_std_all = np.concatenate((vn_glac_std_all, vn_glac_std_region), axis=0)
                        
                        ds.close()
                                      
                if skip_gcm == 0:
                    for ngroup, group in enumerate(groups):
                        # Select subset of data
                        group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                        vn_glac = vn_glac_all[group_glac_indices,:]
                        
                        subgroups, subgroup_cn = select_groups(subgrouping, main_glac_rgi)
    
                        # Sum volume change for group
                        group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                        vn_group = vn_glac_all[group_glac_indices,:].sum(axis=0)
    #                    area_group = area_glac_all[group_glac_indices,:].sum(axis=0)
                        
    #                    # Uncertainty associated with volume change based on subgroups
    #                    #  sum standard deviations in each subgroup assuming that they are uncorrelated
    #                    #  then use the root sum of squares using the uncertainty of each subgroup to get the 
    #                    #  uncertainty of the group
    #                    main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi[group_cn] == group]
    #                    subgroups_subset = main_glac_rgi_subset[subgroup_cn].unique()
    #
    #                    subgroup_std = np.zeros((len(subgroups_subset), vn_group.shape[0]))
    #                    for nsubgroup, subgroup in enumerate(subgroups_subset):
    #                        main_glac_rgi_subgroup = main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup]
    #                        subgroup_indices = (
    #                                main_glac_rgi.loc[main_glac_rgi[subgroup_cn] == subgroup].index.values.tolist())
    #                        # subgroup uncertainty is sum of each glacier since assumed to be perfectly correlated
    #                        subgroup_std[nsubgroup,:] = vn_glac_std_all[subgroup_indices,:].sum(axis=0)
    #                    vn_group_std = (subgroup_std**2).sum(axis=0)**0.5        
                        
                        ds_all[rcp][gcm_name][group] = vn_group
    #                    ds_std_all[rcp][gcm_name][group] = vn_group_std
        
        if vn == 'runoff_glac_monthly':
            pickle_data(runoff_fn_pkl, ds_all)
    
    #%%
    # Select multimodel data
    ds_multimodel = {}
    for rcp in rcps:
        ds_multimodel[rcp] = {}
        
        for ngroup, group in enumerate(groups):
        
            for ngcm, gcm_name in enumerate(gcm_names):
                
                print(rcp, group, gcm_name)
                
                try:
                    vn_group = ds_all[rcp][gcm_name][group]
                    skip_gcm = 0
                    
                except:
                    skip_gcm = 1
                
                if skip_gcm == 0:
                    if ngcm == 0:
                        vn_multimodel = vn_group                
                    else:
                        vn_multimodel = np.vstack((vn_multimodel, vn_group))
            
            ds_multimodel[rcp][group] = vn_multimodel
     
    # Adjust groups or their order           
    if grouping == 'watershed':
        groups.remove('Irrawaddy')
        groups.remove('Yellow')
        
    if grouping == 'himap':
        group_order = [11,17,1,2,18,4,10,7,15,3,12,20,8,0,19,5,21,14,13,16,9,6]
        groups = [x for _,x in sorted(zip(group_order,groups))]
    elif grouping == 'watershed':
        group_order = [0,10,9,3,8,4,5,6,11,1,2,7]
        groups = [x for _,x in sorted(zip(group_order,groups))]
        
    #%%
            
    multimodel_linewidth = 1
    alpha=0.2
    
    groups_select = ['Indus', 'Brahmaputra']
    rcps = ['rcp26', 'rcp45', 'rcp85']
        
    fig, ax = plt.subplots(1, len(groups_select), squeeze=False, sharex=False, sharey=True, 
                           gridspec_kw = {'wspace':0, 'hspace':0})
    add_group_label = 1
    
    rcp_colordict = {'rcp26':'b', 'rcp45':'darkgreen', 'rcp60':'m', 'rcp85':'r'}
    
    # Cycle through groups  
    row_idx = 0
    col_idx = 0
    for ngroup, group in enumerate(groups_select):
        col_idx = ngroup

        for rcp in rcps:  

            # ===== Plot =====            
            # Multi-model statistics
            vn_multimodel = ds_multimodel[rcp][group]
            vn_multimodel_mean = vn_multimodel.mean(axis=0)
            vn_multimodel_std = vn_multimodel.std(axis=0)
            vn_multimodel_stdlow = vn_multimodel_mean - vn_multimodel_std
            vn_multimodel_stdhigh = vn_multimodel_mean + vn_multimodel_std
            
            # Normalize volume by initial volume
            if vn == 'volume_glac_annual':
                vn_normalizer = vn_multimodel_mean[0]
            # Normalize runoff by mean runoff from 2000-2015
            elif vn == 'runoff_glac_monthly':
                t1_idx = np.where(time_values_annual == 2000)[0][0]
                t2_idx = np.where(time_values_annual == 2015)[0][0] + 1
                vn_normalizer = vn_multimodel.mean(axis=0)[t1_idx:t2_idx].mean()
            vn_multimodel_mean_norm = vn_multimodel_mean / vn_normalizer
            vn_multimodel_std_norm = vn_multimodel_std / vn_normalizer
            vn_multimodel_stdlow_norm = vn_multimodel_mean_norm - vn_multimodel_std_norm
            vn_multimodel_stdhigh_norm = vn_multimodel_mean_norm + vn_multimodel_std_norm
            
            t1_idx = np.where(time_values_annual == startyear)[0][0]
            t2_idx = np.where(time_values_annual == endyear)[0][0] + 1
            
            rcp_zorder_dict = {'rcp26':6, 'rcp45':8, 'rcp60':7, 'rcp85':7}
            rcp_zorder_background_dict = {'rcp26':3, 'rcp45':5, 'rcp60':4, 'rcp85':4}
            
            ax[row_idx, col_idx].plot(
                    time_values_annual[t1_idx:t2_idx], vn_multimodel_mean_norm[t1_idx:t2_idx], color=rcp_colordict[rcp], 
                    linewidth=multimodel_linewidth, label=rcp, zorder=rcp_zorder_dict[rcp])
            ax[row_idx, col_idx].plot(
                    time_values_annual[t1_idx:t2_idx], vn_multimodel_stdlow_norm[t1_idx:t2_idx], 
                    color=rcp_colordict[rcp], linewidth=0.25, linestyle='-', label=rcp, 
                    zorder=rcp_zorder_background_dict[rcp])
            ax[row_idx, col_idx].plot(
                    time_values_annual[t1_idx:t2_idx], vn_multimodel_stdhigh_norm[t1_idx:t2_idx], 
                    color=rcp_colordict[rcp], linewidth=0.25, linestyle='-', label=rcp, 
                    zorder=rcp_zorder_background_dict[rcp])
            ax[row_idx, col_idx].fill_between(
                    time_values_annual[t1_idx:t2_idx], vn_multimodel_stdlow_norm[t1_idx:t2_idx], 
                    vn_multimodel_stdhigh_norm[t1_idx:t2_idx], 
                    facecolor=rcp_colordict[rcp], alpha=0.2, label=None, 
                    zorder=rcp_zorder_background_dict[rcp])
                
            # Group labels
            ax[row_idx, col_idx].text(0.5, 0.99, title_dict[group], size=10, 
                                      horizontalalignment='center', verticalalignment='top', 
                                      transform=ax[row_idx, col_idx].transAxes)

            # X-label
            ax[row_idx, col_idx].set_xlim(time_values_annual[t1_idx:t2_idx].min(), 
                                          time_values_annual[t1_idx:t2_idx].max())
            ax[row_idx, col_idx].xaxis.set_tick_params(labelsize=12)
            ax[row_idx, col_idx].xaxis.set_major_locator(plt.MultipleLocator(50))
            ax[row_idx, col_idx].xaxis.set_minor_locator(plt.MultipleLocator(10))
            ax[row_idx, col_idx].set_xticklabels(['','',''])
                
            # Y-label
            ax[row_idx, col_idx].set_ylim(0,2.2)
            ax[row_idx, col_idx].yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax[row_idx, col_idx].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[row_idx, col_idx].set_yticklabels(['','','0.5','1.0','1.5','2.0',''])
            ax[row_idx, col_idx].yaxis.set_tick_params(labelsize=12)
#            ax[row_idx, col_idx].yaxis.set_major_locator(MaxNLocator(prune='both'))
                
            # Tick parameters
            ax[row_idx, col_idx].yaxis.set_ticks_position('both')
            ax[row_idx, col_idx].tick_params(axis='y', which='major', labelsize=8, direction='inout', pad=0)
            ax[row_idx, col_idx].tick_params(axis='y', which='minor', labelsize=8, direction='inout', pad=0) 
            ax[row_idx, col_idx].tick_params(axis='x', which='major', labelsize=8, direction='in', pad=0)
            ax[row_idx, col_idx].tick_params(axis='x', which='minor', labelsize=8, direction='in', pad=0)
            
            for axis in ['top','bottom','left','right']:
                ax[row_idx, col_idx].spines[axis].set_linewidth(0.5)
#                ax[row_idx, col_idx].spines[axis].set_zorder(0)
                       
            
            # Add value to subplot
            plot_str = ''
            if vn == 'runoff_glac_monthly' and rcp == rcps[-1]:
                group_glac_indices = main_glac_rgi.loc[main_glac_rgi[group_cn] == group].index.values.tolist()
                group_volume_Gt = ((main_glac_hyps.values[group_glac_indices,:] * 
                                    main_glac_icethickness.values[group_glac_indices,:] / 1000 * pygem_prms.density_ice / 
                                    pygem_prms.density_water).sum())
                group_runoff_Gta = ds_multimodel[rcp][group].mean(axis=0)[:15].mean() * (1/1000)**3
                plot_str = '(' + str(int(np.round(group_runoff_Gta,0))) + ' Gt $\mathregular{yr^{-1}}$)'
                plot_str_loc = 0.90

            if rcp == rcps[-1]:
                ax[row_idx, col_idx].text(0.5, plot_str_loc, plot_str, size=8, horizontalalignment='center', 
                                          verticalalignment='top', transform=ax[row_idx, col_idx].transAxes, 
                                          color='k', zorder=5)
        
    # RCP Legend
    rcp_lines = []
    for rcp in rcps:
        line = Line2D([0,1],[0,1], color=rcp_colordict[rcp], linewidth=multimodel_linewidth)
        rcp_lines.append(line)
    rcp_labels = ['RCP ' + rcp_dict[rcp] for rcp in rcps]
    if vn == 'temp_glac_annual' or vn == 'prec_glac_annual':
        legend_loc = 'upper left'
    else:
        legend_loc = 'lower left'
    ax[0,0].legend(rcp_lines, rcp_labels, loc=(0.05,0.01), fontsize=8, labelspacing=0, handlelength=1, 
                   handletextpad=0.25, borderpad=0, frameon=False, ncol=1)
    # RCP Legend
    rcp_lines = []
    line = Line2D([0,1],[0,1], color='grey', linewidth=multimodel_linewidth)
    rcp_lines.append(line)
    line = Line2D([0,1],[0,1], color='grey', linewidth=4*multimodel_linewidth, alpha=0.2)
    rcp_lines.append(line)
    rcp_labels = ['mean', 'stdev']
    ax[0,1].legend(rcp_lines, rcp_labels, loc=(0.05,0.01), fontsize=8, labelspacing=0.2, handlelength=1, 
                   handletextpad=0.6, borderpad=0, frameon=False, ncol=1)

#    # Label
#    if vn == 'runoff_glac_monthly':
#        ylabel_str = 'Runoff (-)'
#    elif vn == 'volume_glac_annual':
#        ylabel_str = 'Mass (-)'
#    # Y-Label
#    if len(groups) == 1:
#        fig.text(-0.01, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
#    else:
#        fig.text(0.03, 0.5, ylabel_str, va='center', rotation='vertical', size=12)
    
    # Save figure
    fig.set_size_inches(2.5, 1.5)
    
    figure_fn = 'runoff_multimodel_4rcps.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300, transparent=True)

#%%
if option_regional_hyps == 1:
    rgi_regionsO1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    rgi_regionsO2 = 'all'
    rgi_glac_number = 'all'
    
    # Set up plot
    ncols = 4
    nrows = int(np.ceil(len(rgi_regionsO1)/ncols))
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False, sharey=False, 
                           gridspec_kw = {'wspace':0.4, 'hspace':0.4})
    nrow, ncol = 0,0

    for nregion, region in enumerate(rgi_regionsO1):
        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], 
                                                          rgi_regionsO2='all',rgi_glac_number='all')
        main_glac_rgi.sort_values('Area', inplace=True, ascending=False)
        main_glac_rgi.reset_index(inplace=True, drop=True)
        main_glac_rgi['cum_area'] = main_glac_rgi.Area.cumsum()
        main_glac_rgi['cum_area_%'] = main_glac_rgi.Area.cumsum()
        main_glac_rgi['cum_area_%'] = main_glac_rgi.cum_area / main_glac_rgi.cum_area.values[-1] * 100

        # Plot glacier number vs. cumulative area
        ax[nrow,ncol].plot(main_glac_rgi.index.values, main_glac_rgi['cum_area_%'].values, 
                     color='k', linewidth=1, zorder=2, label='plot1')
        
        ax[nrow,ncol].set_xscale('log')
        if main_glac_rgi.shape[0] > 1000:
            ax[nrow,ncol].set_xticks([10,100,1000,10000])   
        else:
            ax[nrow,ncol].set_xticks([10,100,1000])   
        
        ax[nrow,ncol].set_ylim(0,100)
        ax[nrow,ncol].yaxis.set_major_locator(plt.MultipleLocator(20))
        ax[nrow,ncol].yaxis.set_minor_locator(plt.MultipleLocator(5))
        
        # Tick parameters
        ax[nrow,ncol].yaxis.set_ticks_position('both')
        ax[nrow,ncol].tick_params(axis='both', which='major', labelsize=12, direction='inout')
        ax[nrow,ncol].tick_params(axis='both', which='minor', labelsize=12, direction='inout')  
        ax[nrow,ncol].text(0.5,0.95, str(region) + '\n(' + str(main_glac_rgi.shape[0]) + ' glaciers)' + 
                          '\n(' + str(int(main_glac_rgi.cum_area.values[-1])) + 'km2)',
                           horizontalalignment='center', verticalalignment='top', transform=ax[nrow,ncol].transAxes)

        # Adjust row and column
        ncol += 1
        if ncol == ncols:
            nrow += 1
            ncol = 0
    
#    # Remove extra plots    
#    n_extras = len(regions)%ncols 
#    if n_extras > 0:
#        for nextra in np.arange(0,n_extras):
#            ax[nrow,ncol].axis('off')
#            ncol += 1         
        
        # Example Legend
        # Option 1: automatic based on labels
#        ax[0,0].legend(loc=(0.05, 0.05), fontsize=10, labelspacing=0.25, handlelength=1, handletextpad=0.25, borderpad=0, 
#                       frameon=False)
        # Option 2: manually define legend
        #leg_lines = []
        #labels = ['plot1', 'plot2']
        #label_colors = ['k', 'b']
        #for nlabel, label in enumerate(labels):
        #    line = Line2D([0,1],[0,1], color=label_colors[nlabel], linewidth=1)
        #    leg_lines.append(line)
        #ax[0,0].legend(leg_lines, labels, loc=(0.05,0.05), fontsize=10, labelspacing=0.25, handlelength=1, 
        #               handletextpad=0.25, borderpad=0, frameon=False)
        
    # X and Y labels
    fig.text(0.5, 0.03, 'Glacier Number (by size)', size=12, horizontalalignment='center', verticalalignment='top')
    fig.text(0.03, 0.5, 'Cumulative Area (%)', size=12, horizontalalignment='center', verticalalignment='top', 
             rotation='vertical')
    
    # Save figure
    #  figures can be saved in any format (.jpg, .png, .pdf, etc.)
    fig.set_size_inches(8, 8)
    figure_fp = os.getcwd() + '/../Output/'
    if os.path.exists(figure_fp) == False:
        os.makedirs(figure_fp)
    figure_fn = 'rgi_regions_vs_cumarea.png'
    fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)


#%%
# Plot histogram of start dates
if option_startdate == 1:
    regions = [13,14,15]
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=regions, rgi_regionsO2 = 'all', 
                                                      rgi_glac_number='all')
    main_glac_rgi['RefYear'] = [int(str(x)[0:4]) for x in main_glac_rgi.RefDate.values]
    main_glac_rgi['RefMonth'] = [int(str(x)[4:6]) for x in main_glac_rgi.RefDate.values]
    main_glac_rgi['RefDay'] = [int(str(x)[6:]) for x in main_glac_rgi.RefDate.values]
    main_glac_rgi['RefMonth_dec'] = main_glac_rgi['RefMonth'] + main_glac_rgi['RefDay'] / 30 - 1
    main_glac_rgi['RefYear_dec'] = main_glac_rgi['RefYear'] + main_glac_rgi['RefMonth_dec'] / 12
    
    print('Mean date:', main_glac_rgi.RefYear_dec.mean())
    
    refyear_bins = np.arange(1998,2015)
    data = main_glac_rgi['RefYear_dec'].values
    hist, bin_edges = np.histogram(data,refyear_bins) # make the histogram
    hist = hist/main_glac_rgi.shape[0] * 100
    fig,ax = plt.subplots()    
    # Plot the histogram heights against integers on the x axis
    ax.bar(range(len(hist)),hist,width=1, edgecolor='k') 
    # Set the ticks to the middle of the bars
    ax.set_xticks([0.5+i for i,j in enumerate(hist)])
    # Set the xticklabels to a string that tells us what the bin edges were
    ax.set_xticklabels(['{}'.format(refyear_bins[i]) for i,j in enumerate(hist)], rotation=45, ha='right')
#    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Count (%)', fontsize=12)
    # Save figure
    fig.set_size_inches(4,3)
    fig.savefig(pygem_prms.output_filepath + 'refyear_hist_HMA.png', bbox_inches='tight', dpi=300)
    
#%%
if option_excess_meltwater_diagram == 1:
    fig_fp = pygem_prms.output_sim_fp + 'figures/'
    
    glacier_mass = np.array([10, 10.5, 9, 8.5, 9, 8, 7.5, 6.5, 7.5, 7, 8, 6, 5, 4.5]) - 4
    excess_meltwater_step_x = np.array([0,1,1,4,4,10,10,11,11,12,12,12]) + 1
    excess_meltwater_step_y = np.array([0,0,1,1,2,2,4,4,5,5,5.5,5.5])
    time = np.array(np.arange(0,len(glacier_mass)))
    
    
    annual_mb = glacier_mass[1:] - glacier_mass[0:-1]    
    annual_mb_cumsum = np.zeros((len(glacier_mass)))
    annual_mb_cumsum[1:] = np.cumsum(annual_mb)
    excess_meltwater = np.array([0,1,0,0,1,0,0,0,0,0,2,1,0.5,0])
#    vol_km3 = np.reshape(vol,(-1,len(vol))) / pygem_prms.density_ice * pygem_prms.density_water / 1000**3
#    excess_meltwater = excess_meltwater_m3(vol_km3)
    
    # Create the projection
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(100, 1)
    ax1 = plt.subplot(gs[0:56,0])
    ax2 = plt.subplot(gs[60:100,0])
#    gs = mpl.gridspec.GridSpec(100, 1)
#    ax1 = plt.subplot(gs[0:60,0])
#    ax2 = plt.subplot(gs[71:100,0])

#    # First subplot (glacier mass and cumulative excess meltwater)
#    ax1.plot(time, glacier_mass, color='k', linewidth=1, zorder=2, label='Glacier mass')
##    ax1.plot(time, annual_mb_cumsum, color='k', linewidth=1, zorder=2, label='cumulative MB', linestyle='..')
#    ax1.set_xlim(0,len(glacier_mass)-1)
#    ax1.set_ylim(0,7)
#    ax1.set_ylabel('Glacier mass\n(Gt)')
#    ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
#    ax1.yaxis.set_ticks_position('both')
#    ax1.tick_params(axis='y', which='both', direction='inout')
#    
#    ax1b = ax1.twinx()
#    ax1b.plot(excess_meltwater_step_x, excess_meltwater_step_y, color='b', linewidth=1, linestyle='--', 
#             label='Excess meltwater Cumsum')
#    ax1b.set_ylim(-1,6)
#    ax1b.invert_yaxis()
#    ax1b.set_ylabel('Excess meltwater \n(cumulative, Gt)', color='b')
#    ax1b.spines['right'].set_color('b')
#    ax1b.tick_params(axis='y', colors='b')
#    ax1b.yaxis.set_minor_locator(plt.MultipleLocator(1))
#    ax1b.tick_params(axis='y', which='both', direction='inout', color='b')
    
    # First subplot (glacier mass and cumulative excess meltwater)
#    ax1.plot(time, glacier_mass, color='k', linewidth=1, zorder=2, label='Glacier mass')
    ax1.plot(time, annual_mb_cumsum, color='k', linewidth=1, zorder=2, label='cumulative MB', linestyle='-')
    ax1.set_xlim(0,len(glacier_mass)-1)
    ax1.set_ylim(-6,1)
    ax1.set_ylabel('Cumulative \nmass balance\n(Gt)')
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.yaxis.set_ticks_position('both')
    ax1.tick_params(axis='y', which='both', direction='inout')
    ax1.tick_params(labelbottom=False)
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.tick_params(axis='x', which='both', direction='inout')
    ax1.grid(which='major', axis='both', color='grey', linewidth=0.5, alpha=1)
    ax1.grid(which='minor', axis='both', color='grey', linewidth=0.25, alpha=0.5)
    
    
    ax1b = ax1.twinx()
    ax1b.plot(excess_meltwater_step_x, excess_meltwater_step_y, color='b', linewidth=2, linestyle='--', 
             label='Excess meltwater Cumsum')
    ax1b.set_ylim(-1,6)
    ax1b.invert_yaxis()
    ax1b.set_ylabel('Cumulative \nexcess meltwater \n(Gt)', labelpad=12, color='b')
    ax1b.spines['right'].set_color('b')
    ax1b.tick_params(axis='y', colors='b')
    ax1b.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1b.tick_params(axis='y', which='both', direction='inout', color='b')
    
    
    # Second subplot (annual mass balance and annual excess meltwater)
    ax2.plot(time[1:], annual_mb, color='k', linewidth=1, zorder=2, label='annual mb')
    ax2.set_xlim(0,len(glacier_mass)-1)
    ax2.set_ylim(-2.5,2.5)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax2.set_ylabel('Annual \nmass balance \n(Gt)')
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax2.tick_params(axis='x', which='both', direction='inout')
    ax2.grid(which='major', axis='both', color='grey', linewidth=0.5, alpha=1)
    ax2.grid(which='minor', axis='both', color='grey', linewidth=0.25, alpha=0.5)
    ax2.set_xlabel('Year')
    
    ax2b = ax2.twinx()
    ax2b.plot(time[1:], excess_meltwater[:-1], color='b', linewidth=1, linestyle='--', label='Excess meltwater')
    ax2b.set_ylim(-2.5,2.5)
    ax2b.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax2b.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax2b.set_ylabel('Annual \nexcess meltwater\n(Gt)', color='b')
    ax2b.spines['right'].set_color('b')
    ax2b.tick_params(axis='y', colors='b')
    ax2b.tick_params(axis='y', which='both', direction='inout', color='b')
    
    # Save figure
    fig.set_size_inches(3, 4)
    if os.path.exists(fig_fp) == False:
        os.makedirs(fig_fp)
    figure_fn = 'excess_melwater_diagram.png'
    fig.savefig(fig_fp + figure_fn, bbox_inches='tight', dpi=300)
    
    
#%% EXTRA CODE
#rgi_glac_number_fn = '../SPC_PYGEM/PyGEM/R131415_rgi_glac_number_batch_0.pkl'
#with open(rgi_glac_number_fn, 'rb') as f:
#    glac_no = pickle.load(f)
#    
##rgi_glac_number_fn = '../SPC_PYGEM/PyGEM/R131415_rgi_glac_number_batch_0_check.pkl'
##with open(rgi_glac_number_fn, 'rb') as f:
##    glac_no_check = pickle.load(f)
#
#import spc_split_glaciers
#glac_no_batches = spc_split_glaciers.split_list(glac_no, n=24, option_ordered=0)
#
#
#list_fns = ['R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--1.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--2.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--3.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--4.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--5.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--6.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--7.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--8.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--9.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--10.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--12.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--13.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--14.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--15.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--16.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--18.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--19.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--20.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--21.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--22.nc',
#            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--23.nc']
####list_fns = ['R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--8.nc', 
####            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch0--10.nc']
####list_fns = ['R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch7--14.nc', 
####            'R131415_CCSM4_rcp26_c2_ba1_100sets_2000_2100_batch7--16.nc']
##
#netcdf_fp = pygem_prms.main_directory + '/../SPC_PYGEM/'
#for i in list_fns:
#    ds = xr.open_dataset(netcdf_fp + i)
#    df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
#    print(str(int(df.loc[0,'O1Region'])) + '.' + str(int(df.loc[0,'glacno'])).zfill(5),
#          str(int(df.loc[1,'O1Region'])) + '.' + str(int(df.loc[1,'glacno'])).zfill(5),
#          str(int(df.loc[2,'O1Region'])) + '.' + str(int(df.loc[2,'glacno'])).zfill(5))
##    print(str(int(df.loc[df.shape[0]-1,'O1Region'])) + '.' + str(int(df.loc[df.shape[0]-1,'glacno'])))
    

#%%
    
# Code for individual glacier changes mass balance vs. climate
#        # Multimodel means
#        vol_glac_all_annual_multimodel_mean = vol_glac_all_annual_multimodel.mean(axis=2)
#        mb_glac_all_annual_multimodel_mean = mb_glac_all_annual_multimodel.mean(axis=2)
#        temp_glac_all_annual_multimodel_mean = temp_glac_all_annual_multimodel.mean(axis=2)
#        prectotal_glac_all_annual_multimodel_mean = prectotal_glac_all_annual_multimodel.mean(axis=2)
#        
#        # Normalize changes
#        ref_startyear = 2000
#        ref_endyear = 2015
#        
#        plt_startyear = 2015
#        plt_endyear = 2100
#        
#        ref_idx_start = np.where(time_values_annual == ref_startyear)[0][0]
#        ref_idx_end = np.where(time_values_annual == ref_endyear)[0][0] + 1
#        
#        plt_idx_start = np.where(time_values_annual == plt_startyear)[0][0]
#        plt_idx_end = np.where(time_values_annual == plt_endyear)[0][0] + 1
#        
#        #%%
#        vol_norm = vol_glac_all_annual_multimodel_mean[:,-1] / vol_glac_all_annual_multimodel_mean[:,0]
#        
#        mb_norm_value = mb_glac_all_annual_multimodel_mean[:,ref_idx_start:ref_idx_end].mean(axis=1)
#        mb_norm = mb_glac_all_annual_multimodel_mean[:,-10:].mean(axis=1) - mb_norm_value
#        
#        temp_norm_value = temp_glac_all_annual_multimodel_mean[:,ref_idx_start:ref_idx_end].mean(axis=1)
#        temp_norm = temp_glac_all_annual_multimodel_mean[:,-10:].mean(axis=1) - temp_norm_value
#        
#        prec_norm_value = prectotal_glac_all_annual_multimodel_mean[:,ref_idx_start:ref_idx_end].mean(axis=1)
#        prec_norm = prectotal_glac_all_annual_multimodel_mean[:,-10:].mean(axis=1) / prec_norm_value
#        
#        #%%
#        main_glac_rgi['vol_norm'] = vol_norm
#        glac_idx = main_glac_rgi[(main_glac_rgi.Area > 1) & (main_glac_rgi.vol_norm < 2)].index.values

#%%
if option_caldata_compare == 1:
    netcdf_fp_era = pygem_prms.output_sim_fp + 'ERA5/'
    
    regions = [1]
    cal_datasets = ['braun']
    ds_fn = 'R1--all--ERA5_c4_ba1_1sets_1995_2017.nc'
    
    startyear=1995
    endyear=2017
    wateryear=1
    
    output_fp = netcdf_fp_era + 'figures/'
    if os.path.exists(output_fp) == False:
        os.makedirs(output_fp)
    
    dates_table  = modelsetup.datesmodelrun(startyear=startyear, endyear=endyear, spinupyears=0, 
                                            option_wateryear=wateryear)
    
    # Load glaciers
    main_glac_rgi, main_glac_hyps, main_glac_icethickness = load_glacier_data(rgi_regionsO1=regions)
    
    # Modeled Mass Balance
    # Load datasets
    ds = xr.open_dataset(netcdf_fp_era + ds_fn)
    df = pd.DataFrame(ds.glacier_table.values, columns=ds.glac_attrs)
    df['RGIId'] = ['RGI60-' + str(int(df.O1Region.values[x])).zfill(2) + '.' +
                   str(int(df.glacno.values[x])).zfill(5) for x in df.index.values]
    
    # Extract time variable
    time_values_annual = ds.coords['year_plus1'].values
    time_values_monthly = ds.coords['time'].values
    # Extract start/end indices for calendar year!
    time_values_df = pd.DatetimeIndex(time_values_monthly)
    time_values_yr = np.array([x.year for x in time_values_df])
    if pygem_prms.gcm_wateryear == 1:
        time_values_yr = np.array([x.year + 1 if x.month >= 10 else x.year for x in time_values_df])
    time_idx_start = np.where(time_values_yr == startyear)[0][0]
    time_idx_end = np.where(time_values_yr == endyear)[0][0]
    time_values_monthly_subset = time_values_monthly[time_idx_start:time_idx_end + 12]
    year_idx_start = np.where(time_values_annual == startyear)[0][0]
    year_idx_end = np.where(time_values_annual == endyear)[0][0]
    time_values_annual_subset = time_values_annual[year_idx_start:year_idx_end+1]
    
    var_glac_region_raw = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 0]
    var_glac_region_raw_std = ds['massbaltotal_glac_monthly'].values[:,time_idx_start:time_idx_end + 12, 1]
    area_glac_region = np.repeat(ds['area_glac_annual'].values[:,year_idx_start:year_idx_end+1,0], 12, axis=1)
    
    # Area average
    volchg_monthly_glac_region = var_glac_region_raw
    volchg_monthly_glac_region_std = var_glac_region_raw_std

    # Merge datasets
    var_glac_all = volchg_monthly_glac_region
    var_glac_all_std = volchg_monthly_glac_region_std
    area_glac_all = area_glac_region
    df_all = df
    
    # Remove RGIIds from main_glac_rgi that are not in the model runs
    rgiid_df = list(df_all.RGIId.values)
    rgiid_all = list(main_glac_rgi.RGIId.values)
    rgi_idx = [rgiid_all.index(x) for x in rgiid_df]
    main_glac_rgi = main_glac_rgi.loc[rgi_idx,:]
    main_glac_rgi.reset_index(inplace=True, drop=True)
    
    #%%
    # Calibration data
    cal_data = pd.DataFrame()
    for dataset in cal_datasets:
        cal_subset = class_mbdata.MBData(name=dataset)
        cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi, main_glac_hyps, dates_table)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)
    
    #%%
    # Link glacier index number from main_glac_rgi to cal_data to facilitate grabbing the data
    glacnodict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi.index.values))
    cal_data['glac_idx'] = cal_data['RGIId'].map(glacnodict)
    
    # Update main_glac_rgi and simulation datasets to be consistent with cal_data
    cal_data_idx = list(cal_data.glac_idx.values)
    main_glac_rgi = main_glac_rgi.loc[cal_data_idx,:]
    main_glac_rgi.reset_index(inplace=True, drop=True)
    var_glac_all = volchg_monthly_glac_region[cal_data_idx,:]
    var_glac_all_std = volchg_monthly_glac_region_std[cal_data_idx,:]
    area_glac_all = area_glac_region[cal_data_idx,:]
    df_all = df_all.loc[cal_data_idx,:]
    cal_data.reset_index(inplace=True, drop=True)
    
    #%%
    cal_data['mb_mwe_era'] = np.nan
    cal_data['mb_mwea_era'] = np.nan
    cal_data['mb_mwe_era_std'] = np.nan
    for nglac in list(cal_data.index.values):
        glac_idx = nglac
        t1_idx = int(cal_data.loc[nglac,'t1_idx'])
        t2_idx = int(cal_data.loc[nglac,'t2_idx'])
        t1 = cal_data.loc[nglac,'t1']
        t2 = cal_data.loc[nglac,'t2']
        cal_data.loc[nglac,'mb_mwe_era'] = var_glac_all[glac_idx, t1_idx:t2_idx].sum()
#        cal_data.loc[nglac,'mb_mwe_era_std'] = var_glac_all_std[glac_idx, t1_idx:t2_idx].sum() 
#        cal_data.loc[nglac,'mb_mwe_era_std_rsos'] = ((var_glac_all_std[glac_idx, t1_idx:t2_idx]**2).sum())**0.5
    
    cal_data['time_difference'] = cal_data['t2'] - cal_data['t1']    
    cal_data['mb_mwea_era'] = cal_data['mb_mwe_era'] / (cal_data['t2'] - cal_data['t1'])
#    cal_data['mb_mwea_era_std'] = cal_data['mb_mwe_era_std'] / (cal_data['t2'] - cal_data['t1'])
#    cal_data['mb_mwea_era_std_rsos'] = cal_data['mb_mwe_era_std_rsos'] / (cal_data['t2']-cal_data['t1'])
    cal_data['mb_mwea'] = cal_data['mb_mwe'] / (cal_data['t2'] - cal_data['t1'])
    cal_data['mb_mwea_std'] = cal_data['mb_mwe_err'] / (cal_data['t2'] - cal_data['t1'])
    cal_data['mb_mwea_dif'] = cal_data['mb_mwea_era'] - cal_data['mb_mwea']
    cal_data['zscore'] = (cal_data['mb_mwea_era'] - cal_data['mb_mwea']) / cal_data['mb_mwea_std']
    
    #%%
    # Loop through conditions:
    condition_dict = OrderedDict()
    condition_dict['All']= cal_data.index.values
    condition_dict['All w data'] = cal_data['obs_type'] == 'mb_geo'
    condition_dict['All extrapolated'] = cal_data['obs_type'] == 'mb_geo_extrapolated'
    
    stats_cns = ['group', 'count', 'rmse', 'r', 'slope', 'intercept', 'p-value', 'mae']
    group_stats = pd.DataFrame(np.zeros((len(condition_dict.keys()), len(stats_cns))), columns=stats_cns)
    group_stats['group'] = condition_dict.keys()
    
    for ncondition, cal_condition in enumerate(condition_dict.keys()):
#    for ncondition, cal_condition in enumerate(['Glaciological (annual)']):
        # Statistics for comparison
        cal_data_subset = cal_data.loc[condition_dict[cal_condition],:].copy()
        print('\n',cal_condition, cal_data_subset.shape[0])
        
        # Root-mean-square-deviation
        rmse = (np.sum((cal_data_subset.mb_mwea - cal_data_subset.mb_mwea_era)**2) / cal_data_subset.shape[0])**0.5
        print('  RMSE:', np.round(rmse,2))
        # Correlation
        slope, intercept, r_value, p_value, std_err = linregress(cal_data_subset.mb_mwea, cal_data_subset.mb_mwea_era)
        print('  r_value =', np.round(r_value,2), 'slope = ', np.round(slope,2), 
              'intercept = ', np.round(intercept,2), 'p_value = ', np.round(p_value,6))
        # Mean absolute error
        mae = np.mean(np.absolute(cal_data_subset.mb_mwea - cal_data_subset.mb_mwea_era))
        print('  mean absolute error:', np.round(mae,2))
        # Record stats
        group_stats.loc[ncondition, ['count', 'rmse', 'r', 'slope', 'intercept', 'p-value', 'mae']] = (
                [cal_data_subset.shape[0], rmse, r_value, slope, intercept, p_value, mae])

        
        cal_data_subset['dif_mb_mwea'] = cal_data_subset['mb_mwea'] - cal_data_subset['mb_mwea_era']
        print('  Difference stats: \n    Mean (+/-) std [mwea]:', 
          np.round(cal_data_subset['dif_mb_mwea'].mean(),2), '+/-', np.round(cal_data_subset['dif_mb_mwea'].std(),2), 
          'count:', cal_data_subset.shape[0],
          '\n    Median (+/-) std [mwea]:', 
          np.round(cal_data_subset['dif_mb_mwea'].median(),2), '+/- XXX', 
#          np.round(cal_data_subset['dif_mb_mwea'].std(),2),
#          '\n    Mean standard deviation (correlated):',np.round(cal_data_subset['mb_mwea_era_std'].mean(),2),
#          '\n    Mean standard deviation (uncorrelated):',np.round(cal_data_subset['mb_mwea_era_std_rsos'].mean(),2)
            )
        
    group_stats.to_csv(output_fp + 'cal_compare_stats.csv', index=False)

    #%%
    # ===== PLOT =====
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10,8), gridspec_kw = {'wspace':0.3, 'hspace':0})
    
    datatypes = ['mb_geo', 'mb_geo_extrapolated']
    cmap = 'RdYlBu_r'
    norm = plt.Normalize(startyear, endyear)
    
    for nplot, datatype in enumerate(datatypes):
#    for nplot, datatype in enumerate(['mb_geo']):
        cal_data_plot = cal_data[cal_data['obs_type'] == datatype].copy()
        cal_data_plot.reset_index(drop=True, inplace=True)
        sizes = [0.01,1,5,50]
        sizes_str = [str(x) for x in sizes]
        s_sizes = [1,4,10,20]
        cal_data_plot['circ_size'] = s_sizes[0]
        cal_data_plot.loc[cal_data_plot['area_km2'] > sizes[1], 'circ_size'] = s_sizes[1]
        cal_data_plot.loc[cal_data_plot['area_km2'] > sizes[2], 'circ_size'] = s_sizes[2]
        cal_data_plot.loc[cal_data_plot['area_km2'] > sizes[3], 'circ_size'] = s_sizes[3]
        
        if datatype in ['mb_geo', 'mb_geo_extrapolated']:
            # All glaciers
            a = ax[0,nplot].scatter(cal_data_plot.mb_mwea.values, cal_data_plot.mb_mwea_era.values, 
                                    color='k', zorder=3, 
#                                    s=15,
                                    s=cal_data_plot.circ_size.values,
                                    marker='o', linewidth=0.25)
            a.set_facecolor('none')
            ymin = -5
            ymax = 3.5
            xmin = -5
            xmax = 3.5
            ax[0,nplot].set_xlim(xmin,xmax)
            ax[0,nplot].set_ylim(ymin,ymax)
            ax[0,nplot].plot([np.min([xmin,ymin]),np.max([xmax,ymax])], [np.min([xmin,ymin]),np.max([xmax,ymax])], 
                             color='k', linewidth=0.25, zorder=1)
            
            ax[0,nplot].set_ylabel('$\mathregular{B_{mod}}$ (m w.e. $\mathregular{yr^{-1}}$)', labelpad=0, size=12)
            ax[0,nplot].set_xlabel('$\mathregular{B_{geo}}$ (m w.e. $\mathregular{yr^{-1}}$)', labelpad=0, size=12)
            # Add text
            ax[0,nplot].text(0.05, 0.95, 'E', va='center', size=12, fontweight='bold', transform=ax[0,nplot].transAxes)
            ax[0,nplot].text(0.7, 0.1, 'n=' + str(cal_data_plot.shape[0]) + '\n' + 
                             str(cal_data_plot.glacno.unique().shape[0]) + ' glaciers', va='center', ha='center', 
                             size=12, transform=ax[0,nplot].transAxes)
            slope, intercept, r_value, p_value, std_err = linregress(cal_data_plot.mb_mwea.values, 
                                                                     cal_data_plot.mb_mwea_era.values)
            print(datatype, 'r_value [mwea] =', r_value)
    
    # SIZE LEGEND      
    marker_linecolor='k'
    marker_linewidth = 0.25
    circ1 = ax[0,nplot].scatter([0],[0], s=s_sizes[0], marker='o', color='grey', 
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ1.set_facecolor('none')
    circ2 = ax[0,nplot].scatter([0],[0], s=s_sizes[1], marker='o', color='grey',
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ2.set_facecolor('none')
    circ3 = ax[0,nplot].scatter([0],[0], s=s_sizes[2], marker='o', color='grey',
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ3.set_facecolor('none')
    circ4 = ax[0,nplot].scatter([0],[0], s=s_sizes[3], marker='o', color='grey',
                       edgecolor=marker_linecolor, linewidth=marker_linewidth)
    circ4.set_facecolor('none')
    leg_fontsize = 10
    legend=fig.legend([circ1,circ2,circ3,circ4], sizes_str, 
              scatterpoints=1, ncol=1,
              loc='upper left',  bbox_to_anchor=(0.865,0.63),
              fontsize=leg_fontsize, labelspacing=0.3, columnspacing=0,handletextpad=0, handlelength=1,
              borderpad=0.2, framealpha=0, borderaxespad=0.2,
              )
    fig.text(0.94, 0.62, 'Area\n(km$^{2}$)', ha='center', va='center', size=leg_fontsize)
    
    # Save figure
    fig.set_size_inches(6.75,3)
    fig_fn = 'cal_compare.png'
    fig.savefig(output_fp + fig_fn, bbox_inches='tight', dpi=600)
    
    
#thickness_fn = 'thickness_m_01_Farinotti2019_10m.csv'
#hyps_fn = 'area_km2_01_Farinotti2019_10m.csv'
#width_fn = 'width_km_01_Farinotti2019_10m.csv'
#slope_fn = 'slope_deg_01_Farinotti2019_10m.csv'
#length_fn = 'length_km_01_Farinotti2019_10m.csv'
#
#thickness = pd.read_csv(pygem_prms.hyps_filepath + thickness_fn)
#thickness_cns = list(thickness.columns)
#thickness_cns.remove('RGIId')
#thickness_values = thickness.loc[:,thickness_cns].values
#print('thickness < 0', np.where(thickness_values < 0))
#thickness_values[thickness_values < 0] = 0
#
#thickness_v2 = thickness.copy()
#thickness_v2.loc[:,thickness_cns] = thickness_values
#thickness_v2.to_csv(pygem_prms.hyps_filepath + 'updated/' + thickness_fn, index=False)
#
#hyps = pd.read_csv(pygem_prms.hyps_filepath + hyps_fn)
#hyps_values = hyps.loc[:,thickness_cns].values
#hyps_values[thickness_values == 0] = 0
#print('hyps < 0', np.where(hyps_values < 0))
#
#hyps_v2 = hyps.copy()
#hyps_v2.loc[:,thickness_cns] = hyps_values
#hyps_v2.to_csv(pygem_prms.hyps_filepath + 'updated/' + hyps_fn, index=False)
#
#
#width = pd.read_csv(pygem_prms.hyps_filepath + width_fn)
#width_values = width.loc[:,thickness_cns].values
#width_values[thickness_values == 0] = 0
#print('width < 0', np.where(width_values < 0))
#
#width_v2 = width.copy()
#width_v2.loc[:,thickness_cns] = width_values
#width_v2.to_csv(pygem_prms.hyps_filepath + 'updated/' + width_fn, index=False)
#
#slope = pd.read_csv(pygem_prms.hyps_filepath + slope_fn)
#slope_values = slope.loc[:,thickness_cns].values
#slope_values[thickness_values == 0] = 0
#print('slope < 0', np.where(slope_values < 0))
#
#slope_v2 = slope.copy()
#slope_v2.loc[:,thickness_cns] = slope_values
#slope_v2.to_csv(pygem_prms.hyps_filepath + 'updated/' + slope_fn, index=False)
#
#length = pd.read_csv(pygem_prms.hyps_filepath + length_fn)
#length_values = length.loc[:,thickness_cns].values
#length_values[thickness_values == 0] = 0
#print('length < 0', np.where(length_values < 0))
#
#length_v2 = length.copy()
#length_v2.loc[:,thickness_cns] = length_values
#length_v2.to_csv(pygem_prms.hyps_filepath + 'updated/' + length_fn, index=False)



    
    
    
    
    
    
    