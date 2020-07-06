#tusr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:13:22 2018

@author: kitreatakataglushkoff
Kitrea's hand-written copied/adjusted version of the analyze_massredistribution.py, 
which was last significantly edited Thursday July 18. 

UPDATE - Oct 9, 2018 - Kitrea double-checked code, added some comments. 
last updated Wed Nov 14 - to clean out bad data in the new large dataset.

UPDATE - March/April, 2020 (ongoing) - Zoe edited script to integrate
new parameters into the existing functions
"""
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
#from scipy.stats import median_absolute_deviation

#%% ===== FUNCTIONS =====
rgi_fp = os.getcwd() + '/../RGI/rgi60/00_rgi60_attribs/'
assert os.path.exists(rgi_fp), '01_rgi60_Alaska.csv'

def selectglaciersrgitable(glac_no=None,
                           rgi_regionsO1=None,
                           rgi_regionsO2=None,
                           rgi_glac_number=None,
                           rgi_fp=rgi_fp,
                           rgi_cols_drop=['GLIMSId','BgnDate','EndDate','Status','Connect','Linkages','Name'],
                           rgi_O1Id_colname='glacno',
                           rgi_glacno_float_colname='RGIId_float',
                           indexname='GlacNo'):
    """
    Select all glaciers to be used in the model run according to the regions and glacier numbers defined by the RGI
    glacier inventory. This function returns the rgi table associated with all of these glaciers.

    glac_no : list of strings
        list of strings of RGI glacier numbers (e.g., ['1.00001', '13.00001'])
    rgi_regionsO1 : list of integers
        list of integers of RGI order 1 regions (e.g., [1, 13])
    rgi_regionsO2 : list of integers or 'all'
        list of integers of RGI order 2 regions or simply 'all' for all the order 2 regions
    rgi_glac_number : list of strings
        list of RGI glacier numbers without the region (e.g., ['00001', '00002'])

    Output: Pandas DataFrame of the glacier statistics for each glacier in the model run
    (rows = GlacNo, columns = glacier statistics)
    """
    if glac_no is not None:
        glac_no_byregion = {}
        rgi_regionsO1 = [int(i.split('.')[0]) for i in glac_no]
        rgi_regionsO1 = list(set(rgi_regionsO1))
        for region in rgi_regionsO1:
            glac_no_byregion[region] = []
        for i in glac_no:
            region = i.split('.')[0]
            glac_no_only = i.split('.')[1]
            glac_no_byregion[int(region)].append(glac_no_only)

        for region in rgi_regionsO1:
            glac_no_byregion[region] = sorted(glac_no_byregion[region])

    # Create an empty dataframe
    rgi_regionsO1 = sorted(rgi_regionsO1)
    glacier_table = pd.DataFrame()
    for region in rgi_regionsO1:

        if glac_no is not None:
            rgi_glac_number = glac_no_byregion[region]

        for i in os.listdir(rgi_fp):
            if i.startswith(str(region).zfill(2)) and i.endswith('.csv'):
                rgi_fn = i
        print(rgi_fn)
        try:
            csv_regionO1 = pd.read_csv(rgi_fp + rgi_fn)
        except:
            csv_regionO1 = pd.read_csv(rgi_fp + rgi_fn, encoding='latin1')
        
        # Populate glacer_table with the glaciers of interest
        if rgi_regionsO2 == 'all' and rgi_glac_number == 'all':
            print("All glaciers within region(s) %s are included in this model run." % (region))
            if glacier_table.empty:
                glacier_table = csv_regionO1
            else:
                glacier_table = pd.concat([glacier_table, csv_regionO1], axis=0)
        elif rgi_regionsO2 != 'all' and rgi_glac_number == 'all':
            print("All glaciers within subregion(s) %s in region %s are included in this model run." %
                  (rgi_regionsO2, region))
            for regionO2 in rgi_regionsO2:
                if glacier_table.empty:
                    glacier_table = csv_regionO1.loc[csv_regionO1['O2Region'] == regionO2]
                else:
                    glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[csv_regionO1['O2Region'] ==
                                                                                regionO2]], axis=0))
        else:
            if len(rgi_glac_number) < 20:
                print("%s glaciers in region %s are included in this model run: %s" % (len(rgi_glac_number), region,
                                                                                       rgi_glac_number))
            else:
                print("%s glaciers in region %s are included in this model run: %s and more" %
                      (len(rgi_glac_number), region, rgi_glac_number[0:50]))

            rgiid_subset = ['RGI60-' + str(region).zfill(2) + '.' + x for x in rgi_glac_number]
            rgiid_all = list(csv_regionO1.RGIId.values)
            rgi_idx = [rgiid_all.index(x) for x in rgiid_subset]
            if glacier_table.empty:
                glacier_table = csv_regionO1.loc[rgi_idx]
            else:
                glacier_table = (pd.concat([glacier_table, csv_regionO1.loc[rgi_idx]],
                                           axis=0))

    glacier_table = glacier_table.copy()
    # reset the index so that it is in sequential order (0, 1, 2, etc.)
    glacier_table.reset_index(inplace=True)
    # change old index to 'O1Index' to be easier to recall what it is
    glacier_table.rename(columns={'index': 'O1Index'}, inplace=True)
    # Record the reference date
    glacier_table['RefDate'] = glacier_table['BgnDate']
    # if there is an end date, then roughly average the year
    enddate_idx = glacier_table.loc[(glacier_table['EndDate'] > 0), 'EndDate'].index.values
    glacier_table.loc[enddate_idx,'RefDate'] = (
            np.mean((glacier_table.loc[enddate_idx,['BgnDate', 'EndDate']].values / 10**4).astype(int),
                    axis=1).astype(int) * 10**4 + 9999)
    # drop columns of data that is not being used
    glacier_table.drop(rgi_cols_drop, axis=1, inplace=True)
    # add column with the O1 glacier numbers
    glacier_table[rgi_O1Id_colname] = (
            glacier_table['RGIId'].str.split('.').apply(pd.Series).loc[:,1].astype(int))
    glacier_table['rgino_str'] = [x.split('-')[1] for x in glacier_table.RGIId.values]
    glacier_table[rgi_glacno_float_colname] = (np.array([np.str.split(glacier_table['RGIId'][x],'-')[1]
                                                    for x in range(glacier_table.shape[0])]).astype(float))
    # set index name
    glacier_table.index.name = indexname

    print("This study is focusing on %s glaciers in region %s" % (glacier_table.shape[0], rgi_regionsO1))

    return glacier_table


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, variance**0.5


def weighted_percentile(sorted_list, weights, percentile):
    """
    Calculate weighted percentile of a sorted list
    """
    assert percentile <= 1 or percentile >=0, 'Error: Percentile outside of 0-1'
    weights_cumsum_norm_high = np.cumsum(weights) / np.sum(weights)
#     print(weights_cumsum_norm_high)
    weights_norm = weights / np.sum(weights)
    weights_cumsum_norm_low = weights_cumsum_norm_high - weights_norm
#     print(weights_cumsum_norm_low)
    
    percentile_idx_high = np.where(weights_cumsum_norm_high >= percentile)[0][0]
#     print(percentile_idx_high)
    percentile_idx_low = np.where(weights_cumsum_norm_low <= percentile)[0][-1]
#     print(percentile_idx_low)
    
    if percentile_idx_low == percentile_idx_high:
        value_percentile = sorted_list[percentile_idx_low]
    else:
        value_percentile = np.mean([sorted_list[percentile_idx_low], sorted_list[percentile_idx_high]])

    return value_percentile


def normalized_stats(norm_list):
    # Merge norm_list to make array of all glaciers with same elevation normalization space    
    max_length = len(max(norm_list,key=len)) #len of glac w most norm values
    # All data normalized: 1st column is normalized elev, the others are norm dhdt for each glacier
    norm_all = np.zeros((max_length, len(norm_list)+1))
    # First column is normalized elevation, pulled from the glac with most norm vals
    norm_all[:,0] = max(norm_list,key=len)[:,0]
    norm_all_area = norm_all.copy()
    
    norm_elev_binsize = (norm_all_area[0,0] - norm_all_area[1,0])
    
    # Loop through each glacier's normalized array (where col1 is elev_norm and col2 is norm dhdt)
    for n in range(len(norm_list)):
        norm_single = norm_list[n] # get one glacier at a time 

        # Fill in nan values for elev_norm of 0 and 1 with nearest neighbor
        norm_single[0,1] = norm_single[np.where(~np.isnan(norm_single[:,1]))[0][0], 1]
        norm_single[-1,1] = norm_single[np.where(~np.isnan(norm_single[:,1]))[0][-1], 1]
        norm_single[0,2] = norm_single[np.where(~np.isnan(norm_single[:,2]))[0][0], 2]
        norm_single[-1,2] = norm_single[np.where(~np.isnan(norm_single[:,2]))[0][-1], 2]
        # Remove nan values
        norm_single = norm_single[np.where(~np.isnan(norm_single[:,2]))] #~ is the same as !
        elev_single = norm_single[:,0]
        dhdt_single = norm_single[:,1]
        area_single = norm_single[:,2]
        area_single_cumsum = np.cumsum(area_single)
        #loop through each area value of the glacier, and add it and interpolate to add to the norm_all array.
        for r in range(0, max_length):
            
             # Find value need to interpolate to
            norm_elev_value = norm_all_area[r,0]
            norm_elev_lower = norm_elev_value - norm_elev_binsize/2
            if norm_elev_lower <= 0:
                norm_elev_lower = 0

            # ----- AREA CALCULATION -----
            if r == 0:
                area_cumsum_upper = 0
                
            if norm_elev_lower > 0:
#            if r < max_length-1:
#                print(r, norm_elev_value, norm_elev_value - norm_elev_binsize/2)                
                # Find index of value above it from dhdt_norm, which is a different size
                upper_idx = np.where(elev_single == elev_single[elev_single >= norm_elev_lower].min())[0][0]
                # Find index of value below it
#                print(len(elev_single), max_length)
#                print(elev_single, norm_elev_lower)
                lower_idx = np.where(elev_single == elev_single[elev_single < norm_elev_lower].max())[0][0]
                #get the two values, based on the indices
                upper_elev = elev_single[upper_idx]
                upper_value = area_single_cumsum[upper_idx]
                lower_elev = elev_single[lower_idx]
                lower_value = area_single_cumsum[lower_idx]
                
                #Linearly Interpolate between two values, and plug in interpolated value into norm_all
                area_cumsum_interp = (lower_value + (norm_elev_lower - lower_elev) / (upper_elev - lower_elev) * 
                                      (upper_value - lower_value))
            else:
                area_cumsum_interp = area_single_cumsum[-1]
            # Calculate area within that bin
            norm_all_area[r,n+1] = area_cumsum_interp - area_cumsum_upper
            # Update area_lower_cumsum
            area_cumsum_upper = area_cumsum_interp

            # ----- DH/DT CALCULATION -----
            if r == 0:
                #put first value dhdt value into the norm_all. n+1 because the first col is taken by the elevnorms.
                norm_all[r,n+1] = dhdt_single[0] 
            elif r == (max_length - 1):
                #put last value into the the last row for the glacier's 'stretched out'(interpolated) normalized curve.
                norm_all[r,n+1] = dhdt_single[-1] 
            else:
                # Find value need to interpolate to
                norm_elev_value = norm_all[r,0] #go through each row in the elev (col1)
                # Find index of value above it from dhdt_norm, which is a different size
                upper_idx = np.where(elev_single == elev_single[elev_single >= norm_elev_value].min())[0][0]
                # Find index of value below it
                lower_idx = np.where(elev_single == elev_single[elev_single < norm_elev_value].max())[0][0]
                #get the two values, based on the indices
                upper_elev = elev_single[upper_idx]
                upper_value = dhdt_single[upper_idx]
                lower_elev = elev_single[lower_idx]
                lower_value = dhdt_single[lower_idx]
                #Linearly Interpolate between two values, and plug in interpolated value into norm_all
                norm_all[r,n+1] = (lower_value + (norm_elev_value - lower_elev) / (upper_elev - lower_elev) * 
                                   (upper_value - lower_value))
        
     # Compute mean and standard deviation
    norm_all_stats = pd.DataFrame()
    norm_all_stats['norm_elev'] = norm_all[:,0]
    # DH/DT STATISTICS
    norm_all_stats['norm_dhdt_mean'] = np.nanmean(norm_all[:,1:], axis=1)  
    norm_all_stats['norm_dhdt_med'] = np.nanmedian(norm_all[:,1:], axis=1)  
    norm_all_stats['norm_dhdt_std'] = np.nanstd(norm_all[:,1:], axis=1)
    norm_all_stats['norm_dhdt_16perc'] = np.percentile(norm_all[:,1:], 16, axis=1)
    norm_all_stats['norm_dhdt_84perc'] = np.percentile(norm_all[:,1:], 84, axis=1)
    # AREA STATISTICS
    norm_all_stats['norm_area'] = np.nansum(norm_all_area[:,1:], axis=1)
    norm_all_stats['norm_area_perc'] = norm_all_stats['norm_area'] / norm_all_stats['norm_area'].sum() * 100
    norm_all_stats['norm_area_perc_cumsum'] = np.cumsum(norm_all_stats['norm_area_perc'])
    # area-weighted stats
    norm_all_stats['norm_dhdt_mean_areaweighted'] = np.nan
    norm_all_stats['norm_dhdt_med_areaweighted'] = np.nan
    norm_all_stats['norm_dhdt_std_areaweighted'] = np.nan
    norm_all_stats['norm_dhdt_16perc_areaweighted'] = np.nan
    norm_all_stats['norm_dhdt_84perc_areaweighted'] = np.nan
    for nrow in np.arange(0,norm_all.shape[0]):
        # Select values
        norm_values = norm_all[nrow,1:]
        area_values = norm_all_area[nrow,1:]
        # Sorted values
        area_values_sorted = [x for _,x in sorted(zip(norm_values, area_values))]
        norm_values_sorted = sorted(norm_values)
        # Statistics
        weighted_mean, weighted_std = weighted_avg_and_std(norm_values_sorted, area_values_sorted)
        weighted_med = weighted_percentile(norm_values_sorted, area_values_sorted, 0.5)
        weighted_16perc = weighted_percentile(norm_values_sorted, area_values_sorted, 0.16)
        weighted_84perc = weighted_percentile(norm_values_sorted, area_values_sorted, 0.84)
        # record stats        
        norm_all_stats.loc[nrow,'norm_dhdt_mean_areaweighted'] = weighted_mean
        norm_all_stats.loc[nrow,'norm_dhdt_std_areaweighted'] = weighted_std
        norm_all_stats.loc[nrow,'norm_dhdt_med_areaweighted'] = weighted_med
        norm_all_stats.loc[nrow,'norm_dhdt_16perc_areaweighted'] = weighted_16perc
        norm_all_stats.loc[nrow,'norm_dhdt_84perc_areaweighted'] = weighted_84perc
    
    return norm_all_stats


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
        
#%%
# TO-DO LIST:
print('\nTo-do list:\n  - code Larsen!  \n\n')

#%% ===== REGION AND GLACIER FILEPATH OPTIONS =====
# User defines regions of interest
group1 = ['01', '02', '09', '12', '13', '14', '15', '16', '17', '18']
group2 = ['03', '04']
group3 = ['05', '06', '07', '08', '10', '11']
all_regions = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']

rois = all_regions

if 'davidrounce' in os.getcwd():
    binnedcsv_fp = ('/Users/davidrounce/Documents/Dave_Rounce/DebrisGlaciers_WG/Melt_Intercomparison/output/' + 
                    'mb_bins_all/csv/')
elif 'zoescrewvala' in os.getcwd():
    binnedcsv_fp = '/Users/zoescrewvala/Documents/Alaska_REU_2019/mb_binned_data/'
else:
    assert True == False, 'add correct binnedcsv_fp'
#for roi in rois:
#    assert os.path.exists(rgi_fp), roi
# OPTION
option_plot_multipleglaciers_multiplethresholds = False
option_plot_multipleregions = True

# Columns to use for mass balance and dhdt (specify mean or median)
#dhdt_stat = 'mean'
dhdt_stat = 'med'
if dhdt_stat == 'mean':
    mb_cn = 'mb_bin_mean_mwea'
    dhdt_cn = 'dhdt_bin_mean_ma'
else:
    mb_cn = 'mb_bin_med_mwea'
    dhdt_cn = 'dhdt_bin_med_ma'
dhdt_max = 2.5
dhdt_min = -50

add_dc_classification_to_termtype = False
dc_perc_threshold = 5

# Quality control options
binsize = 50 # resample bins to remove noise
min_elevbins = 5 # minimum number of elevation bins
min_glac_area = 2 # minimum total glacier area size (km2) to consider (removes small glaciers)
perc_remove = 2.5 # percentage of glacier area to remove (1 means 1 - 99% are used); set to 0 to keep everything
min_bin_area_km2 = 0.02 # minimum binned area (km2) to remove everything else; set to 0 to keep everything
option_remove_surge_glac = True
option_remove_all_pos_dhdt = True
option_remove_dhdt_acc = True
option_remove_acc_lt_abl = True

# ===== PLOT OPTIONS =====
# Option to save figures
option_savefigs = True

fig_fp = binnedcsv_fp + '../figs/'
glacier_plots_transparency = 0.3

#%% Select Files
#    # Load file if it already exists
overwrite = False
pkl_fp =  binnedcsv_fp + '../pickle_datasets/'
if not os.path.exists(pkl_fp):
    os.makedirs(pkl_fp)
binnedcsv_all_fullfn = pkl_fp + 'binnedcsv_all.pkl'
main_glac_rgi_fullfn = pkl_fp + 'main_glac_rgi_all.pkl'

# Load pickle data if it exists
if os.path.exists(binnedcsv_all_fullfn) and not overwrite:
    # Binnedcsv data
    with open(binnedcsv_all_fullfn, 'rb') as f:
        binnedcsv_all = pickle.load(f)
    # Main_glac_rgi data
    with open(main_glac_rgi_fullfn, 'rb') as f:
        main_glac_rgi = pickle.load(f)

# Otherwise, process the data (all regions)
else:
    print('redoing pickle datasets')
    # Process all regions
    rois = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']
    # Find files for analysis; create list of all binned filenames
    binnedcsv_fullfns_all = []
    rgiids_all = []
    binnedcsv_fullfns_allrois = []
    for roi in rois:
        binnedcsv_fullfns_roi = []
        rgiids_roi = []
        if roi in ['13','14','15']:
            roi_4fp = 'HMA'
        else:
            roi_4fp = roi
        binnedcsv_fp_roi = binnedcsv_fp + roi_4fp + '/'
        for i in os.listdir(binnedcsv_fp_roi):
            if i.startswith(str(int(roi))) and i.endswith('_mb_bins.csv'):
                rgiids_roi.append(i.split('_')[0])
                binnedcsv_fullfns_roi.append(binnedcsv_fp_roi + i)
        # Sorted files        
        binnedcsv_fullfns_roi = [x for _,x in sorted(zip(rgiids_roi, binnedcsv_fullfns_roi))]
        rgiids_roi = sorted(rgiids_roi)
    
        binnedcsv_fullfns_all.extend(binnedcsv_fullfns_roi)
        binnedcsv_fullfns_allrois.append(binnedcsv_fullfns_roi)
        rgiids_all.extend(rgiids_roi)
    
    main_glac_rgi_all = selectglaciersrgitable(glac_no=rgiids_all)
    b = main_glac_rgi_all.copy()
    main_glac_rgi_all['binnedcsv_fullfn'] = binnedcsv_fullfns_all
    main_glac_rgi_all['roi'] = [x.split('-')[1].split('.')[0] for x in main_glac_rgi_all.RGIId.values]
    main_glac_rgi = main_glac_rgi_all[main_glac_rgi_all['Area'] > min_glac_area].copy()
    main_glac_rgi.reset_index(drop=True, inplace=True)
    
    # Add statistics for each glacier
    main_glac_rgi['Zmean'] = np.nan
    main_glac_rgi['PercDebris'] = np.nan
    main_glac_rgi['HypsoIndex']  = np.nan
    main_glac_rgi['AAR'] = np.nan
    main_glac_rgi['Z_maxloss_norm'] = np.nan
    main_glac_rgi['mb_abl_lt_acc'] = np.nan
    main_glac_rgi['nbins'] = np.nan
    main_glac_rgi['Size'] = np.nan
    binnedcsv_all = []
    for nglac, rgiid in enumerate(main_glac_rgi.rgino_str.values):
#    for nglac, rgiid in enumerate(main_glac_rgi.rgino_str.values[0:1]):
        if nglac%100 == 0:
            print(nglac, rgiid)
        binnedcsv_fullfn = main_glac_rgi.loc[nglac,'binnedcsv_fullfn']
        binnedcsv = pd.read_csv(binnedcsv_fullfn)
    
        # Elevation bin statistics
        bins_elev = binnedcsv['bin_center_elev_m'].values
        bins_area = binnedcsv['z1_bin_area_valid_km2'].values
        zmin = bins_elev.min()
        zmax = bins_elev.max()
        zmean, zstd = weighted_avg_and_std(bins_elev, bins_area)
        zmed =  weighted_percentile(bins_elev, bins_area, 0.5)

        # Size -- size classes from Huss et al., 2010
        if main_glac_rgi['Area'][nglac] <= 5:
            glac_size = 'Small'
        elif main_glac_rgi['Area'][nglac] > 5 and main_glac_rgi['Area'][nglac] <= 20:
            glac_size = 'Medium'
        else:
            glac_size = 'Large' 
        # Hypsometry index (McGrath et al. 2017)
        hyps_idx = (zmax - zmed) / (zmed - zmin)
        if hyps_idx > 0 and hyps_idx < 1:
            hyps_idx = -1/hyps_idx
            
        # Accumulation-area ratio (assuming median is the equilibrium line altitude)
        aar = bins_area[bins_elev >= zmed].sum() / bins_area.sum()
            
        # Relative debris-covered area
        if 'dc_bin_area_valid_km2' in binnedcsv.columns:
            dc_perc = binnedcsv['dc_bin_area_valid_km2'].sum() / binnedcsv['z1_bin_area_valid_km2'].sum() * 100
        else:     
            dc_perc = 0
        # Classify land-terminating glaciers are debris-covered if selected option (Debris = 6)
        if add_dc_classification_to_termtype:
            if main_glac_rgi.loc[nglac,'TermType'] == 0:
                if dc_perc >= dc_perc_threshold:
                    main_glac_rgi.loc[nglac,'TermType'] = 6
            
        # Normalized elevation of most negative bin (0 = Zmin, 1 = Zmax)
        maxloss_idx = np.where(binnedcsv[dhdt_cn] == binnedcsv[dhdt_cn].min())[0][0]
        z_maxloss = binnedcsv.loc[maxloss_idx,'bin_center_elev_m']
        z_maxloss_norm = (z_maxloss - zmin) / (zmax - zmin)
        
    
        # Is mass balance in ablation area more negative than the accumulation area (as we would expect)?
        binnedcsv_acc = binnedcsv[binnedcsv.bin_center_elev_m >= zmed]
        mb_acc = ((binnedcsv_acc['z1_bin_area_valid_km2'] * binnedcsv_acc[mb_cn]).sum() / 
                  binnedcsv_acc['z1_bin_area_valid_km2'].sum())
        binnedcsv_abl = binnedcsv[binnedcsv.bin_center_elev_m < zmed]
        mb_abl = ((binnedcsv_abl['z1_bin_area_valid_km2'] * binnedcsv_abl[mb_cn]).sum() / 
                  binnedcsv_abl['z1_bin_area_valid_km2'].sum())
        if mb_abl < mb_acc:
            mb_abl_lt_acc = True
        else:
            mb_abl_lt_acc = False
    
        # Add attributes
        main_glac_rgi.loc[nglac,'Zmin'] = zmin
        main_glac_rgi.loc[nglac,'Zmax'] = zmax
        main_glac_rgi.loc[nglac,'Zmed'] = zmed
        main_glac_rgi.loc[nglac,'Zmean'] = zmean
        main_glac_rgi.loc[nglac,'PercDebris'] = dc_perc
        main_glac_rgi.loc[nglac,'HypsoIndex']  = hyps_idx
        main_glac_rgi.loc[nglac,'AAR'] = aar
        main_glac_rgi.loc[nglac,'Z_maxloss_norm'] = z_maxloss_norm
        main_glac_rgi.loc[nglac,'mb_abl_lt_acc'] = mb_abl_lt_acc
        # ===== Filter out bad values ==========================================================
        # Remove bad values of dhdt
        binnedcsv.loc[binnedcsv[dhdt_cn] > dhdt_max, dhdt_cn] = np.nan
        binnedcsv.loc[binnedcsv[dhdt_cn] < dhdt_min, dhdt_cn] = np.nan
        # If dhdt is nan, remove row
        null_bins = binnedcsv.loc[pd.isnull(binnedcsv[dhdt_cn])].index.values
        binnedcsv = binnedcsv.drop(null_bins)
        binnedcsv.reset_index(inplace=True, drop=True)
        if binnedcsv.shape[0] > 0:
            #sort out glaciers based on if they have all positive dh/dt, all negative, dh/dt, or both	
            #based on evaluating, for each glacier, the max from the list of dhdt and the min from the list.
            if np.nanmin(binnedcsv[dhdt_cn].astype(float)) >= 0: 		
                glacwide_dhdt_sign = 1 #glaciers with all positive dh/dt		
            elif np.nanmax(binnedcsv[dhdt_cn].astype(float)) <= 0: 		
                glacwide_dhdt_sign = -1 #glaciers with all negative dh/dt		
            else: 		
                glacwide_dhdt_sign = 0 #glaciers with both, + and - dh/dt 		
            main_glac_rgi.loc[nglac, 'dhdt_sign'] = glacwide_dhdt_sign

        # ===== OPTION: RESAMPLE BIN SIZES =====
        elev_bins_resampled = np.arange(0 + binsize/2, binnedcsv.bin_center_elev_m.max() + binsize, binsize)
        try:
            elev_bins_resampled_idx_low = np.where(elev_bins_resampled < binnedcsv.bin_center_elev_m.min())[0][-1]
        except:
            elev_bins_resampled_idx_low = 0
        elev_bins_resampled = elev_bins_resampled[elev_bins_resampled_idx_low:]
        binnedcsv_resampled = pd.DataFrame(np.zeros((len(elev_bins_resampled), binnedcsv.shape[1])),
                                           columns=binnedcsv.columns)
        binnedcsv_resampled['bin_center_elev_m'] = elev_bins_resampled
        for nbin, elev_bin in enumerate(list(elev_bins_resampled)):
            elev_bins = binnedcsv.bin_center_elev_m.values
            elevbin_idx = np.where((elev_bins >= elev_bin - binsize/2) & (elev_bins < elev_bin + binsize/2))[0]

            if len(elevbin_idx) > 0 and binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'].sum() > 0:
                binnedcsv_resampled.loc[nbin,'z1_bin_count_valid'] = (
                        binnedcsv.loc[elevbin_idx,'z1_bin_count_valid'].sum())
                binnedcsv_resampled.loc[nbin,'z1_bin_area_valid_km2'] =  (
                        binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'].sum())
                binnedcsv_resampled.loc[nbin,'slope_bin_med'] = (
                        weighted_avg_and_std(binnedcsv.loc[elevbin_idx,'slope_bin_med'],
                                             binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'])[0])
                binnedcsv_resampled.loc[nbin,'aspect_bin_med'] = (
                        weighted_avg_and_std(binnedcsv.loc[elevbin_idx,'aspect_bin_med'], 
                                             binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'])[0])
                binnedcsv_resampled.loc[nbin,'dhdt_bin_count'] = (
                        binnedcsv.loc[elevbin_idx,'dhdt_bin_count'].sum())
                binnedcsv_resampled.loc[nbin,'dhdt_bin_area_valid_km2'] = (
                        binnedcsv.loc[elevbin_idx,'dhdt_bin_area_valid_km2'].sum())
                binnedcsv_resampled.loc[nbin,'dhdt_bin_mean_ma'] = (
                        weighted_avg_and_std(binnedcsv.loc[elevbin_idx,'dhdt_bin_mean_ma'], 
                                             binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'])[0])
                binnedcsv_resampled.loc[nbin,'dhdt_bin_med_ma'] = (
                        weighted_avg_and_std(binnedcsv.loc[elevbin_idx,'dhdt_bin_med_ma'], 
                                             binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'])[0])
                binnedcsv_resampled.loc[nbin,'mb_bin_mean_mwea'] = (
                        weighted_avg_and_std(binnedcsv.loc[elevbin_idx,'mb_bin_mean_mwea'], 
                                             binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'])[0])
                binnedcsv_resampled.loc[nbin,'mb_bin_med_mwea'] = (
                        weighted_avg_and_std(binnedcsv.loc[elevbin_idx,'mb_bin_med_mwea'], 
                                             binnedcsv.loc[elevbin_idx,'z1_bin_area_valid_km2'])[0])
                if 'dc_bin_area_valid_km2' in binnedcsv.columns:
                    binnedcsv_resampled.loc[nbin,'dc_bin_area_valid_km2'] = (
                            binnedcsv.loc[elevbin_idx,'dc_bin_area_valid_km2'].sum())
                
            binnedcsv_resampled['z1_bin_area_perc'] = (binnedcsv_resampled['z1_bin_area_valid_km2'] / 
                                                       binnedcsv_resampled['z1_bin_area_valid_km2'].sum() * 100)
            binnedcsv_resampled['z1_bin_areas_perc_cum'] = np.cumsum(binnedcsv_resampled['z1_bin_area_perc'])
            binnedcsv_resampled['z2_bin_count_valid'] = binnedcsv_resampled['z1_bin_count_valid']
            binnedcsv_resampled['z2_bin_area_valid_km2'] = binnedcsv_resampled['z1_bin_area_valid_km2']
            binnedcsv_resampled['z2_bin_area_perc'] = binnedcsv_resampled['z1_bin_area_perc']
            binnedcsv_resampled['dhdt_bin_area_perc'] = (binnedcsv_resampled['dhdt_bin_area_valid_km2'] / 
                                                         binnedcsv_resampled['dhdt_bin_area_valid_km2'].sum() * 100)
            binnedcsv_resampled['dhdt_bin_std_ma'] = np.nan
            binnedcsv_resampled['dhdt_bin_mad_ma'] = np.nan
            binnedcsv_resampled['mb_bin_std_mwea'] = np.nan
            binnedcsv_resampled['mb_bin_mad_mwea'] = np.nan
            if 'dc_bin_area_valid_km2' in binnedcsv.columns:
                binnedcsv_resampled['dc_dhdt_bin_count'] = np.nan
                binnedcsv_resampled['dc_dhdt_bin_mean_ma'] = np.nan 
                binnedcsv_resampled['dc_dhdt_bin_std_ma'] = np.nan
                binnedcsv_resampled['dc_dhdt_bin_med_ma'] = np.nan
                binnedcsv_resampled['dc_dhdt_bin_mad_ma'] = np.nan
                binnedcsv_resampled['dc_mb_bin_mean_mwea'] = np.nan 
                binnedcsv_resampled['dc_mb_bin_std_mwea'] = np.nan
                binnedcsv_resampled['dc_mb_bin_med_mwea'] = np.nan
                binnedcsv_resampled['dc_mb_bin_mad_mwea'] = np.nan
                binnedcsv_resampled['dc_bin_count_valid'] = np.nan
                binnedcsv_resampled['dc_bin_area_perc'] = (binnedcsv_resampled['dc_bin_area_valid_km2'] / 
                                                           binnedcsv_resampled['dc_bin_area_valid_km2'].sum() * 100)
                binnedcsv_resampled['dc_bin_area_perc_cum'] = np.cumsum(binnedcsv_resampled['dc_bin_area_perc'])
            binnedcsv_resampled['vm_med'] = np.nan
            binnedcsv_resampled['vm_mad'] = np.nan
            binnedcsv_resampled['H_mean'] = np.nan
            binnedcsv_resampled['H_std'] = np.nan

        # ===== Filter out the edges, where bins may be very small =====
        binnedcsv_resampled = binnedcsv_resampled[(binnedcsv_resampled['z1_bin_areas_perc_cum'] > perc_remove) & 
                                                  (binnedcsv_resampled['z1_bin_areas_perc_cum'] < 100 - perc_remove)]
        binnedcsv_resampled.reset_index(inplace=True, drop=True)
        # ===== Filter out any bins that are too small =====
        binnedcsv_resampled = binnedcsv_resampled[binnedcsv_resampled['z1_bin_area_valid_km2'] > min_bin_area_km2]
        binnedcsv_resampled.reset_index(inplace=True, drop=True)
        
        # ===== Record number of elevation bins =====
        main_glac_rgi.loc[nglac, 'nbins'] = binnedcsv_resampled.shape[0]
            
        # ===== Normalized elevation vs. ice thickness change ===============================
        if main_glac_rgi.loc[nglac, 'nbins'] > 1:
            # Normalized elevation
            #  (max elevation - bin elevation) / (max_elevation - min_elevation)
            elev_bins_resampled = binnedcsv_resampled['bin_center_elev_m'].values
            zmin_resampled = elev_bins_resampled.min()
            zmax_resampled = elev_bins_resampled.max()
            binnedcsv_resampled['elev_norm'] = (zmax_resampled - elev_bins_resampled) / (zmax_resampled - zmin_resampled)
            # Normalized ice thickness change [ma]
            #  dhdt / dhdt_max
            glac_dhdt = binnedcsv_resampled[dhdt_cn].values.astype(float)
            # Shifted normalized ice thickness change such that everything is negative
    #        binnedcsv['dhdt_norm_shifted'] = (glac_dhdt - np.nanmax(glac_dhdt))  / np.nanmin(glac_dhdt - np.nanmax(glac_dhdt))
    #        binnedcsv.loc[binnedcsv['dhdt_norm_shifted'] == -0, 'dhdt_norm_shifted'] = 0
            # Replace positive values to zero
            glac_dhdt[glac_dhdt >= 0] = 0
            if np.nanmin(glac_dhdt) != 0:
                binnedcsv_resampled['dhdt_norm_huss'] = glac_dhdt / np.nanmin(glac_dhdt)
                binnedcsv_resampled.loc[binnedcsv_resampled['dhdt_norm_huss'] == -0, 'dhdt_norm_huss'] = 0
            else:
                binnedcsv_resampled['dhdt_norm_huss'] = np.nan
                # Replace dhdt sign as this will fail if there all values are zero
                main_glac_rgi.loc[nglac, 'dhdt_sign'] = 1
            
        # Store binnedcsv data
        binnedcsv_all.append(binnedcsv_resampled)
            
    #    print(zmin, zmax, np.round(zmean), np.round(zstd), zmed, np.round(hyps_idx,2), np.round(dc_perc))
    
    # ===== Quality control ===== 
    # Remove glaciers with too few elevation bins
    main_glac_rgi = main_glac_rgi[main_glac_rgi.nbins >= min_elevbins]
    
    # Remove surging glaciers (listed as 1 possible, 2 probable, or 3 observed in main_glac_rgi)	
    if option_remove_surge_glac:
        main_glac_rgi = main_glac_rgi[(main_glac_rgi.Surging == 0) | (main_glac_rgi.Surging == 9)]
        
    # Remove glaciers with all positive dh/dt values (listed as 1 in main_glac_rgi)
    if option_remove_all_pos_dhdt: 
        main_glac_rgi = main_glac_rgi[main_glac_rgi.dhdt_sign <= 0]
    
    # Remove glaciers with max surface lowering in accumulation area
    if option_remove_dhdt_acc:
        main_glac_rgi = main_glac_rgi[main_glac_rgi.Z_maxloss_norm <= 0.5]
        
    # Remove glaciers where accumulation area has more negative mass balance than ablation area
    if option_remove_acc_lt_abl:
        main_glac_rgi = main_glac_rgi[main_glac_rgi.mb_abl_lt_acc == True]
        
    # Select subset of binnedcsv files consistent with main_glac_rgi 
    #  (do this after all removed to ensure indices are correct)
    binnedcsv_all = [binnedcsv_all[x] for x in main_glac_rgi.index.values]
    main_glac_rgi.reset_index(inplace=True, drop=True)      
        
    # Pickle datasets
    pickle_data(binnedcsv_all_fullfn, binnedcsv_all)
    pickle_data(main_glac_rgi_fullfn, main_glac_rgi)
    
#%% ===== SUBSET OF REGIONS =====
if option_plot_multipleglaciers_multiplethresholds:

    # ===== Thresholds =====
    Area_thresholds = [5, 20]
    Slope_thresholds = [15]
    HypsoIndex_thresholds = [-1.2, 1.2]
    AAR_thresholds = [0.51]
    PercDebris_thresholds = [5, 10]
    TermType_thresholds = [0.5, 1.5, 2.5]
    Form_thresholds = [0.5, 1.5]
    
    # ===== Parameter dictionary =====
    all_pars = {'TermType': TermType_thresholds, 'Area': Area_thresholds,
                'Slope': Slope_thresholds, 'HypsoIndex': HypsoIndex_thresholds,
                'AAR': AAR_thresholds, 'PercDebris': PercDebris_thresholds,
                'Form': Form_thresholds}
    
    # plot options
    stat_type = '_MEDIANS'
    option_shading = True
    
    # ===== Parameters to loop through =====
    #pars_list = ['Area', 'Slope', 'HypsoIndex', 'AAR', 'PercDebris', 'TermType']
    pars_list = ['Area']

    # ===== Divide glaciers by threshold =====
    for n in range(len(pars_list)):   
        parameter = pars_list[n]
    #    parameter = 'Area'
    #    parameter = 'Slope'
    #    parameter = 'HypsoIndex'
    #    parameter = 'AAR'
    #    parameter = 'PercDebris'
    #    parameter = 'TermType'
    #    thresholds_var = parameter + '_thresholds'
        
        # Determine subset
        if parameter == 'TermType':
            subset_idxs = []
            thresholds = list(main_glac_rgi.TermType.unique())
            thresholds.sort()
            termtype_dict = {0: 'Land', 1:'Marine', 2:'Lake', 5:'Other', 6:'Debris'}
            termtype_list = [termtype_dict[x] for x in thresholds]
            print('Term type thresholds are:', thresholds)
            for termtype_value in thresholds:
                subset_idxs.append(np.where(main_glac_rgi.TermType == termtype_value)[0])
        elif parameter == 'Form':
            subset_idxs = []
            thresholds = list(main_glac_rgi.Form.unique())
            thresholds.sort()
            form_dict = {0: 'Glacier', 1:'Ice cap', 2:'Perennial snowfield', 
                             3:'Seasonal snowfield', 9: 'Not assigned'}
            form_list = [form_dict[x] for x in thresholds]
            print('Form thresholds are:', thresholds)
            for form_value in thresholds:
                subset_idxs.append(np.where(main_glac_rgi.Form == form_value)[0])
        else:
            thresholds = all_pars[parameter]
            
            # Add maximum
            thresholds.append(main_glac_rgi[parameter].max() + 1)
            # Loop through and get subsets
            subset_idxs = []
            for n_threshold, threshold in enumerate(thresholds):
                if n_threshold == 0:
                    main_glac_rgi_subset = main_glac_rgi[main_glac_rgi[parameter] <= threshold]
                else:
                    main_glac_rgi_subset = main_glac_rgi[(main_glac_rgi[parameter] <= threshold) & 
                                                          (main_glac_rgi[parameter] > thresholds[n_threshold-1])]
                subset_idxs.append(list(main_glac_rgi_subset.index.values))
     
        # Loop through thresholds
        normlist_glac_per_threshold = []
        count_glac_per_threshold = []
        for n_threshold, threshold in enumerate(thresholds):
            # Subset indices
            subset_idx = subset_idxs[n_threshold]
            binnedcsv_subset = [binnedcsv_all[x] for x in subset_idx]
            main_glac_rgi_subset = main_glac_rgi.loc[subset_idx,:]
            main_glac_rgi_subset.reset_index(inplace=True, drop=True)
    
            normlist_glac = []
            for nglac in main_glac_rgi_subset.index.values:
                binnedcsv_glac = binnedcsv_subset[nglac]
                glac_elevnorm = binnedcsv_glac['elev_norm'].values
#                glac_dhdt_norm_huss = binnedcsv_glac['dhdt_norm_huss']
                glac_dhdt_norm_huss = binnedcsv_glac['dhdt_norm_huss']
                glac_area = binnedcsv_glac['z1_bin_area_valid_km2']
#                normlist_array = np.array([glac_elevnorm, glac_dhdt_norm_huss, glac_area]).transpose()
                normlist_array = np.array([glac_elevnorm, glac_dhdt_norm_huss, glac_area]).transpose()
#                normlist_array = np.array([glac_elevnorm, glac_dhdt_norm_huss]).transpose()
                normlist_glac.append(normlist_array)

            normlist_glac_per_threshold.append(normlist_glac)
            count_glac_per_threshold.append(len(normlist_glac))
            print('len normlist_glac:', len(normlist_glac))
        
        #%%
        # ===== PLOT =====
        option_plotarea = True
        # Plot the normalized curves
        fig_width = 5
        if option_plotarea:
            n_cols = 2
        else:
            n_cols = 1
        fig, ax = plt.subplots(len(thresholds), n_cols, squeeze=False, figsize=(fig_width,int(3*len(thresholds))), 
                               gridspec_kw = {'wspace':0.5, 'hspace':0.5})
        
        roi_str = None
        for roi_raw in rois:
            if roi_str is None:
                roi_str = str(roi_raw)
            else:
                roi_str = roi_str + '-' + str(roi_raw) 
                
        normlist_stats_all = []
        for n, threshold in enumerate(thresholds):
            if len(normlist_glac_per_threshold[n]) > 0:
                # Extract values to plot
                normlist = normlist_glac_per_threshold[n]
                
                for glac in range(len(normlist)):
                    normlist_glac = normlist[glac]
                    # Normalized elevation vs. normalized dh/dt
                    ax[n,0].plot(normlist_glac[:,0], normlist_glac[:,1], linewidth=1, alpha=glacier_plots_transparency, 
                                 label=None)
                    ax[n,0].set_ylim(max(normlist_glac[:,0]), min(normlist_glac[:,0]))
                    ax[n,0].set_xlim(0,1)
                    ax[n,0].set_ylabel('dh/dt [-]', size=12)
                    ax[n,0].set_xlabel('Elevation [-]', size=12)
                    ax[n,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
                    ax[n,0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                    ax[n,0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                    ax[n,0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))

                    
                    if parameter == 'TermType':
                        ax[n,0].set_title(('Regions_ ' + roi_str + ' -- ' + parameter + ' = ' + termtype_list[n] + 
                                          ' (' + str(len(normlist)) + ' Glaciers)'), size=12)
                    elif parameter == 'Form':
                        ax[n,0].set_title(('Regions_ ' + roi_str + ' -- ' + parameter + ' = ' + form_list[n] + 
                                          ' (' + str(len(normlist)) + ' Glaciers)'), size=12)
                    else:
                        if threshold == thresholds[0]:
                            ax[n,0].set_title((parameter + '<' + str(threshold) + ' (' + str(len(normlist)) + 
                                              ' Glaciers)'), size=12)
                        elif threshold != thresholds[-1]:
                            ax[n,0].set_title((str(thresholds[n-1]) + '<' + parameter + '<' + str(threshold) + ' (' + 
                                               str(len(normlist)) + ' Glaciers)'), size=12)
                        else:
                            ax[n,0].set_title((parameter + '>' + str(thresholds[n-1]) + ' (' + str(len(normlist)) + 
                                              ' Glaciers)'), size=12)
                        
                # Add statistics to plot
                normlist_stats = normalized_stats(normlist)
                if stat_type == '_MEDIANS':
                    ax[n,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_med_areaweighted, color='black', linewidth=2)
                if stat_type == '_MEANS':
                    ax[n,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_mean_areaweighted, color='black', linewidth=2)
                ax[n,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_16perc_areaweighted, '--', color='black', 
                             linewidth=1.5) 
                ax[n,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_84perc_areaweighted, '--', color='black', 
                             linewidth=1.5)
                
                if option_plotarea:
                    ax[n,1].plot(normlist_stats.norm_elev, normlist_stats.norm_area, color='black', 
                                 linewidth=2)
                    ax[n,1].set_xlim(0,1)
#                    ax[n,1].set_ylim(0,100)
                    ax[n,1].set_ylabel('Cumulative Area [%]', size=12)
                    ax[n,1].set_xlabel('Elevation [-]', size=12)
                    max_area = normlist_stats.norm_area.max()
                    if max_area < 100:
                        ax[n,1].yaxis.set_major_locator(plt.MultipleLocator(10))
                        ax[n,1].yaxis.set_minor_locator(plt.MultipleLocator(5))
                    else:
                        ax[n,1].yaxis.set_major_locator(plt.MultipleLocator((max_area/100).round()*10))
                        ax[n,1].yaxis.set_minor_locator(plt.MultipleLocator((max_area/100).round()*5))
                    ax[n,1].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                    ax[n,1].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
                    
                    perc_upperbnd = 100 - perc_remove
                    perc_lowerbnd = perc_remove
                    vline_upperbnd_idx = np.where(normlist_stats.norm_area_perc_cumsum < perc_upperbnd)[0][-1]
                    vline_upperbnd = normlist_stats.loc[vline_upperbnd_idx, 'norm_elev']
                    vline_lowerbnd_idx = np.where(normlist_stats.norm_area_perc_cumsum > perc_lowerbnd)[0][0]
                    vline_lowerbnd = normlist_stats.loc[vline_lowerbnd_idx, 'norm_elev']
                    ax[n,1].axvline(vline_upperbnd, linewidth=1, linestyle=':', color='grey')
                    ax[n,1].axvline(vline_lowerbnd, linewidth=1, linestyle=':', color='grey')
                    
                # Record stats to plot on separate graph
                normlist_stats_all.append(normlist_stats)
        
        # Save figure
        fig.set_size_inches(fig_width, int(len(thresholds)*3))
        threshold_str_list = [str(i) for i in thresholds]
        threshold_str_list[-1] = 'max'
        threshold_str = '-'.join(threshold_str_list)
        print(threshold_str)
        fig_fp_all = fig_fp + 'resampled_bins/'
        if not os.path.exists(fig_fp_all):
            os.makedirs(fig_fp_all)
        fig.savefig(fig_fp_all + ('rgi_' + roi_str + '-normcurves' + parameter + '_' + threshold_str + '.png'), 
                    bbox_inches='tight', dpi=300)
        plt.show()
            
        #%%
        # ===== PLOT ALL ON ONE =====
        fig_width_all = 4
        fig_height_all = 3
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(fig_width_all,fig_height_all), 
                               gridspec_kw = {'wspace':0.2, 'hspace':0.5})
        for n, normlist_stats in enumerate(normlist_stats_all):
            # Threshold label
            threshold = thresholds[n]
            num_glac = count_glac_per_threshold[n]
            if threshold == thresholds[0]:
                threshold_label = '< ' + str(threshold) + ' (' + str(num_glac) + ')'
            elif threshold != thresholds[-1]:
                threshold_label = str(thresholds[n-1]) + '-' + str(threshold) + ' (' + str(num_glac) + ')'
            else:
                threshold_label = '> ' + str(thresholds[n-1]) + ' (' + str(num_glac) + ')'
            
            
            # Add statistics to plot                           
            # ===== VARIABLES TO PLOT =====
            x_var = normlist_stats.norm_elev
            if stat_type=='_MEDIANS':
                y_var = normlist_stats.norm_dhdt_med_areaweighted
            if stat_type=='_MEANS':
                y_var = normlist_stats.norm_dhdt_mean_areaweighted
            err_low = normlist_stats.norm_dhdt_16perc_areaweighted
            err_high = normlist_stats.norm_dhdt_84perc_areaweighted
            
            # Plot median of each
            ax[0,0].plot(x_var, y_var, linewidth=2, label = threshold_label) #add label
            if option_shading:
                ax[0,0].fill_between(x_var, err_low, err_high, alpha = 0.5, linewidth=1)
            ax[0,0].set_ylim(max(normlist_glac[:,0]), min(normlist_glac[:,0]))
            ax[0,0].set_xlim(0,1)
            ax[0,0].set_ylabel('dh/dt [-]', size=12)
            ax[0,0].set_xlabel('Elevation [-]', size=12)
            ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            print(n)
            ax[0,0].set_title('Region ' + roi_str + ' -- ' + parameter,  size=12)        
            if parameter == 'TermType':
                ax[0,0].legend(termtype_list, loc='lower left', 
                  handlelength=0.5, labelspacing=1, columnspacing=1)
            elif parameter == 'Form':
                ax[0,0].legend(form_list, loc='lower left', 
                  handlelength=0.5, labelspacing=1, columnspacing=1) # + ' (' + str(num_glac) + ')'
            else:
                ax[0,0].legend(loc='lower left', handlelength=0.5, labelspacing=1, columnspacing=1)
        print(threshold_str)
        # Save figure
        fig_fp_all = fig_fp + 'resampled_bins/MEDIANS/'
        if not os.path.exists(fig_fp_all):
            os.makedirs(fig_fp_all)
        fig_fn = None
        if option_shading:
            if rois==group1:
                fig_fn = 'rgi_' + 'group1_' + 'normcurves' + parameter + '_' + threshold_str + stat_type + '.png'
            if rois==group2:
                fig_fn = 'rgi_' + 'group2_' + 'normcurves' + parameter + '_' + threshold_str + stat_type + '.png'
            if rois==group3:
                fig_fn = 'rgi_' + 'group3_' + 'normcurves' + parameter + '_' + threshold_str + stat_type + '.png'
        else:
            if rois==group1:
                fig_fn = ('rgi_' + 'group1_' + 'normcurves' + parameter + '_' + threshold_str + '_noshading' + 
                          stat_type + '.png')
            if rois==group2:
                fig_fn = ('rgi_' + 'group2_' + 'normcurves' + parameter + '_' + threshold_str + '_noshading' + 
                          stat_type + '.png')
            if rois==group3:
                fig_fn = ('rgi_' + 'group3_' + 'normcurves' + parameter + '_' + threshold_str + '_noshading' + 
                          stat_type + '.png')
        if fig_fn is None:
            fig_fn = 'rgi_' + roi_str + 'normcurves' + parameter + '_' + threshold_str + stat_type + '.png'
                
        fig.savefig(fig_fp_all + fig_fn ,  bbox_inches='tight', dpi=300)
        plt.show()
    
#%% ===== COMPARE REGIONS =====
if option_plot_multipleregions:

    # ===== plot specification options =====
    option_shading = False

    # ===== Ranges =====
    Area_range = [5, 20]
    Slope_range = [15, 20]
    TermType_range = [-0.5, 0.5]
    Form_range = [0.5, 1.5]
    
    # ==== Ranges dictionary =====
    ranges_dict = {'Area': Area_range, 'Slope': Slope_range, 
                   'TermType': TermType_range, 'Form': Form_range}

   
    # ===== Select parameter and subset ======
    parameter = 'Area'
    subset = 'Small'
    # ===== Parameter dictionaries =====
    if parameter == 'TermType':
        param_dict = {0: 'Land', 1:'Marine', 2:'Lake', 5:'Other', 6:'Debris'}
    elif parameter == 'Form':
        param_dict = {0: 'Glacier', 1:'Ice cap', 2:'Perennial snowfield', 
                         3:'Seasonal snowfield', 9: 'Not assigned'}

# ======== loop through and compare regions ==========
# ======== select if/for statements based on how you want to compare regions
#              Option 1. group compare (more than 2)
#              Option 2. all pairs in a group (all possible pairings, 1-2, 2-3, 1-3, etc)
#              Option 3. all rois to one region compare (in pairs, 1-2, 1-3, 1-4, etc.)

    if 1 == 1:                                          # option 1 or 3
#    for roi1 in rois:                                    # option 2
#        roi1 = '12'                                      # option 3
        if 1 == 1:                                       # option 1
#        for roi2 in rois:                                # option 2 or 3
            if 1 == 1:                                   # option 1
#            if roi1 != roi2 and int(roi1) < int(roi2):   # option 2
#            if roi1 != roi2: # option 3
                # select regions
                regs_to_comp = rois                      # option 1
#                regs_to_comp = [roi1, roi2]              # option 2 or 3
                # option for error shading 
            #    option_shading = True
    
                # ===== Plot setup =====
                fig_width_all = 4
                fig_height_all = 3
                fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(fig_width_all,fig_height_all), 
                                           gridspec_kw = {'wspace':0.2, 'hspace':0.5})
                ax[0,0].set_ylabel('Normalized Ice Thinning [-]', size=12)
                ax[0,0].set_xlabel('Normalized Elevation [-]', size=12)
                ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
                ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
                
                for roi in regs_to_comp:
                    # Loop through and get subset
                    subset_idxs = []
                    for n in range(len(main_glac_rgi)):
                        if main_glac_rgi['roi'][n] == roi:
                            if parameter == 'Area':
                                if (main_glac_rgi[parameter][n] <= 5):
                                    glac_size = 'Small'
                                elif (main_glac_rgi['Area'][n] > 5 and main_glac_rgi['Area'][n] <= 20):
                                    glac_size = 'Medium'
                                else:
                                    glac_size = 'Large'
                                if glac_size == subset:
                                    subset_idxs.append(n)
                            else:
                                if (main_glac_rgi[parameter][n] > ranges_dict[parameter][0] and 
                                    main_glac_rgi[parameter][n] < ranges_dict[parameter][1]):
                                    subset_idxs.append(n)
                    if subset_idxs != []:
                        binnedcsv_subset = [binnedcsv_all[x] for x in subset_idxs]
                        main_glac_rgi_subset = main_glac_rgi.loc[subset_idxs,:]
                        main_glac_rgi_subset.reset_index(inplace=True, drop=True)
                        normlist_glac = []
                        for nglac in main_glac_rgi_subset.index.values:
                            binnedcsv_glac = binnedcsv_subset[nglac]
                            glac_elevnorm = binnedcsv_glac['elev_norm'].values
                            glac_dhdt_norm_huss = binnedcsv_glac['dhdt_norm_huss']
                            glac_area = binnedcsv_glac['z1_bin_area_valid_km2']
                            normlist_array = np.array([glac_elevnorm, glac_dhdt_norm_huss, glac_area]).transpose()
                            normlist_glac.append(normlist_array)
                        # ===== PLOT =====
                        # Plot the normalized curves
                       
                        # Add statistics to plot                           
                        normlist_stats_all = []
                        normlist_stats = normalized_stats(normlist_glac)
                        
                        stat_type = 'median'
                        x_var = normlist_stats.norm_elev
                        y_var = normlist_stats.norm_dhdt_med
                        err_low = normlist_stats.norm_dhdt_16perc
                        err_high = normlist_stats.norm_dhdt_84perc
            #            error = normlist_stats.norm_dhdt_mad
                        error = normlist_stats.norm_dhdt_std
                        
                        # Plot median of each
                        ax[0,0].plot(x_var, y_var, linewidth=2, label = roi) #add label
                        if option_shading:
                            ax[0,0].fill_between(x_var, err_low, err_high, alpha = 0.5, linewidth=1)
                    
                ax[0,0].set_ylim(0,1)
                ax[0,0].set_xlim(0,1)
                plt.gca().invert_yaxis()
                ax[0,0].set_title(parameter + ' -- ' + subset,  size=12)        
            
                legend = ax[0,0].legend(loc='lower left', ncol=3, fontsize='large', handlelength=0.5) 
                # Save figure
                if option_savefigs:
                    fig_fp_all = fig_fp + 'resampled_bins/region_comparisons/'
                    if not os.path.exists(fig_fp_all):
                        os.makedirs(fig_fp_all)
                    if option_shading==False:
                        fig_fn = ('compareregions' + ''.join(regs_to_comp) + parameter + '_' + subset + 
                                  '_noshading_MEDIANS.png')
                    else:
                        if stat_type == 'median':
                            fig_fn = ('compareregions' + '_' + ''.join(regs_to_comp) + '_' + parameter + '_' + subset + 
                                      '_MEDIANS.png')
                        else:
                            fig_fn = ('compareregions' + '_' + ''.join(regs_to_comp) + parameter + '_' + subset + 
                                      '_MEANS.png')
                fig.savefig(fig_fp_all + fig_fn ,  bbox_inches='tight', dpi=300)
                plt.show()