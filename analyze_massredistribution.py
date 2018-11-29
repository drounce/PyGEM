#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:13:22 2018

@author: kitreatakataglushkoff
Kitrea's hand-written copied/adjusted version of the analyze_massredistribution.py, 
which was last significantly edited Thursday July 18. 

UPDATE - Oct 9, 2018 - Kitrea double-checked code, added some comments. 
last updated Wed Nov 14 - to clean out bad data in the new large dataset. 
"""
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import copy

import pygem_input as input
import pygemfxns_modelsetup as modelsetup

# Tips, comments, and old portions of code no longer used have been moved to bottom of file

#%% ===== REGION AND GLACIER FILEPATH OPTIONS =====
# User defines regions of interest
rgi_regionO1 = [13, 14, 15]
#rgi_regionO1 = [15]
search_binnedcsv_fn = (input.main_directory + '/../DEMs/Shean_2018_1109/aster_2000-2018_20181109_bins/*_mb_bins.csv') 

#%% ===== PLOT OPTIONS =====
# Option to save figures
option_savefigs = 1
fig_fp = input.main_directory + '/../Output/figures/massredistribution/'

# Plot histogram options
option_plot_histogram = 0
histogram_parameters = ['Area', 'Zmed', 'Slope', 'PercDebris']
#histogram_parameters = ['Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect', 'Lmax', 'PercDebris']

# Plot dhdt of each glacier options
option_plot_eachglacier = 0

# Plot glaciers above and below a given parameter threshold (*MAIN FUNCTION TO RUN)
option_plot_multipleglaciers_single_thresholds = 0
# run for specific parameter or all parameters
option_run_specific_pars = 0

# Plot glaciers above and below a given set of multiple thresholds
option_plot_multipleglaciers_multiplethresholds = 0

# Plot glacier characteristics to see if parameters are related
option_plot_compareparameters = 1

#option_plot_multipleglaciers_binned_parameter = 0 #glaciers within a characteristic's defined range
#option_plot_multipleglaciers_indiv_subdivisions = 0 #glaciers binned into 6 categories. (NOT USED)
#option_plots_threshold = 0 #scatter plots relating glacier stats

# Columns to use for mass balance and dhdt (specify mean or median)
mb_cn = 'mb_bin_med_mwea'
dhdt_cn = 'dhdt_bin_med_ma'
dhdt_max = 2.5
dhdt_min = -4
# Threshold for tossing glaciers with too much missing data
perc_area_valid_threshold = 90
# Switch to use merged data or not (0 = don't use, 1 = use merged data)
option_use_mergedata = 0
# Remove glacier options  (surging, all positive dhdt, etc.)
option_remove_surge_glac = 1
option_remove_all_pos_dhdt = 1
option_remove_dhdt_acc = 1
acc_dhdt_threshold = 0.5

# Legend option (switch to show legend on multi-glacier figures or not)
option_show_legend = 0
# Transparency value (between 0 & 1: 0 = no plot, 1 = full opaque)
glacier_plots_transparency = 0.3



#user-defined stored variables for ideal thresholds, for each region and parameter
Area_15_thresholds = list(range(5,40, 5))		
Area_13_thresholds = list(range(5, 120, 5))		
Area_13_thresholds.extend([150, 200, 250, 300, 350]) #if histogram has 2 separate ranges use .extend		
Slope_15_thresholds = list(range(10,26,2)) 		
Slope_13_thresholds = list(range(5, 40, 2)) 
PercDebris_13_thresholds = list(range(0,65,5))
PercDebris_15_thresholds = list(range(0, 65, 5))
Zmin_13_thresholds = list(range(2600,5800, 200))
Zmin_15_thresholds = list(range(3500, 6500, 500))
Zmed_13_thresholds = list(range(3800, 6600, 200))
Zmed_15_thresholds = list(range(4750, 7000, 500))
Aspect_13_thresholds = list(range(0, 450, 90))
Aspect_15_thresholds =  list(range(0, 450, 90))
Zmax_15_thresholds = list(range(6000, 7600, 200))
Zmax_13_thresholds = list(range(4000, 7600, 200))
Lmax_15_thresholds = list(range(4000, 14000, 2000))
Lmax_13_thresholds = list(range(4400, 40000, 2000))
Lmax_13_thresholds.extend([56000, 58000, 6000])
dhdt_13_thresholds = [1]

Area_14_thresholds = list(range(5, 120, 5,))
Area_14_thresholds.extend([150, 200, 250, 300, 350])
Zmin_14_thresholds = list(range(2600, 5800, 200))
Zmax_14_thresholds = list(range(5000, 7600, 200))
Zmed_14_thresholds = list(range(3800,6400, 200))
Slope_14_thresholds = list(range(10, 42, 2))
Aspect_14_thresholds = list(range(0,450,90))
Lmax_14_thresholds = list(range(4000, 45000,2000))
PercDebris_14_thresholds = list(range(0, 65,5))

#For plotting one parameter at a time
#User defines parameter for multi-glacier and histogram runs		
#set the threshold equal to one of the above, defined thresholds, depending on the current 		
#keep in mind for threshold, that the subplots are examining >= and < the threshold		

#If you have not yet evaluated the histograms to define the threshold ranges, 
#then you must define the following variable

#For plotting multiple parameters in one run
#Create dictionary.  key = parameter found in main_glac_rgi, value = thresholds 
all_13_pars = {'Area': Area_13_thresholds, 'Zmin': Zmin_13_thresholds , 
               'Zmax':Zmax_13_thresholds, 'Zmed': Zmed_13_thresholds,
               'Slope': Slope_13_thresholds, 'Aspect': Aspect_13_thresholds,
               'Lmax': Lmax_13_thresholds, 'PercDebris': PercDebris_13_thresholds}

all_14_pars = {'Area': Area_14_thresholds, 'Zmin': Zmin_14_thresholds , 
               'Zmax':Zmax_14_thresholds, 'Zmed': Zmed_14_thresholds,
               'Slope': Slope_14_thresholds, 'Aspect': Aspect_14_thresholds,
               'Lmax': Lmax_14_thresholds, 'PercDebris': PercDebris_14_thresholds}

all_15_pars = {'Area': Area_15_thresholds , 'Zmin': Zmin_15_thresholds , 
               'Zmax':Zmax_15_thresholds, 'Zmed': Zmed_15_thresholds,
               'Slope': Slope_15_thresholds, 'Aspect': Aspect_15_thresholds,
               'Lmax': Lmax_15_thresholds, 'PercDebris': PercDebris_15_thresholds}

#If only plotting one parameter in the run, define the parameter of interest
pars_dict = {'PercDebris': PercDebris_13_thresholds}

if option_run_specific_pars == 1:   
    region_pars = pars_dict
else: 
    if rgi_regionO1[0] == 13: 
        region_pars = all_13_pars
    elif rgi_regionO1[0] == 14:
        region_pars = all_14_pars
    elif rgi_regionO1[0] == 15: 
        region_pars = all_15_pars
    else: 
        print("Please Check Region Specification")


#Binned CSV column name conversion dictionary
#  change column names so they are easier to work with (remove spaces, etc.)
sheancoldict = {'# bin_center_elev_m': 'bin_center_elev_m',
                ' z1_bin_count_valid': 'z1_bin_count_valid',
                ' z1_bin_area_valid_km2': 'z1_bin_area_valid_km2',
                ' z1_bin_area_perc': 'z1_bin_area_perc',
                ' z2_bin_count_valid': 'z2_bin_count_valid',
                ' z2_bin_area_valid_km2': 'z2_bin_area_valid_km2',
                ' z2_bin_area_perc': 'z2_bin_area_perc',
                ' dhdt_bin_count' : 'dhdt_bin_count',
                ' dhdt_bin_area_valid_km2' : 'dhdt_bin_area_valid_km2',
                ' dhdt_bin_area_perc' : 'dhdt_bin_area_perc',
                ' dhdt_bin_med_ma': 'dhdt_bin_med_ma',
                ' dhdt_bin_mad_ma': 'dhdt_bin_mad_ma',
                ' dhdt_bin_mean_ma': 'dhdt_bin_mean_ma',
                ' dhdt_bin_std_ma': 'dhdt_bin_std_ma',
                ' mb_bin_med_mwea': 'mb_bin_med_mwea',
                ' mb_bin_mad_mwea': 'mb_bin_mad_mwea',
                ' mb_bin_mean_mwea': 'mb_bin_mean_mwea',
                ' mb_bin_std_mwea': 'mb_bin_std_mwea',
                ' debris_thick_med_m': 'debris_thick_med_m',
                ' debris_thick_mad_m': 'debris_thick_mad_m',
                ' perc_debris': 'perc_debris',
                ' perc_pond': 'perc_pond',
                ' perc_clean': 'perc_clean',
                ' dhdt_debris_med' : 'dhdt_debris_med',
                ' dhdt_pond_med' : 'dhdt_pond_med',
                ' dhdt_clean_med' : 'dhdt_clean_med',
                ' vm_med' : 'vm_med',
                ' vm_mad' : 'vm_mad',
                ' H_mean' : 'H_mean',
                ' H_std' : 'H_std'}

#%% Select Files
# Find files for analysis; create list of all binnedcsv filenames (fn)
binnedcsv_files_all = glob.glob(search_binnedcsv_fn)

# Fill in dataframe of glacier names and RGI IDs, of ALL glaciers with binnedcsv, regardless of the region
df_glacnames_all = pd.DataFrame() #empty df
df_glacnames_all['reg_glacno'] = [x.split('/')[-1].split('_')[0] for x in binnedcsv_files_all]
df_glacnames_all['RGIId'] = 'RGI60-' + df_glacnames_all.reg_glacno
#   turn region column values from object to float to int, to store just reg
df_glacnames_all['region'] = df_glacnames_all.reg_glacno.astype(float).astype(int)
#   split glacno into list of reg and id, and store just the id part as an object
df_glacnames_all['glacno_str'] = (df_glacnames_all.reg_glacno.str.split('.').apply(lambda x: x[1]))
#   store the same value as glacno_str, but as an int
df_glacnames_all['glacno'] = df_glacnames_all.glacno_str.astype(int)

# Define df_glacnames containing ONLY the data for desired region(s) 
df_glacnames = df_glacnames_all[df_glacnames_all.region.isin(rgi_regionO1) == True]
# make list of all binnedcsv file pathway names
binnedcsv_files = [binnedcsv_files_all[x] for x in df_glacnames.index.values]
# Sort glaciers by region and glacier number
binnedcsv_files = sorted(binnedcsv_files)
df_glacnames = df_glacnames.sort_values('reg_glacno')
df_glacnames.reset_index(drop=True, inplace=True)

# Create dataframe with RGI attributes for each glacier
main_glac_rgi = pd.DataFrame()

for n, region in enumerate(rgi_regionO1):
    print('Region', region)
    df_glacnames_reg = df_glacnames[df_glacnames.region == region] #temp df for one reg at a time
    rgi_glac_number = df_glacnames_reg['glacno_str'].tolist()
    #If statement to avoid errors associated with regions that have no glaciers
    if len(rgi_glac_number) > 0: 
        #pullselect data from fxn outputs of pygemfxnsmodelsetup, and from 
        #pathways and vars defined in pygem input file 
        main_glac_rgi_reg= modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2='all', 
                                                             rgi_glac_number=rgi_glac_number)
        # concatenate regions
        main_glac_rgi = main_glac_rgi.append(main_glac_rgi_reg, ignore_index=True)

#%%MAIN DATASET
# ds is the main dataset for this analysis and is a list of lists (order of glaciers can be found in df_glacnames)
#  Data for each glacier is held in a sublist
ds = []
norm_list = []
for n in range(len(binnedcsv_files)):
    # Note: RuntimeWarning: invalid error encountered in greater than is due 
    #  to nan and zero values being included in array. This error can be ignored.
    # Process binned geodetic data
    binnedcsv = pd.read_csv(binnedcsv_files[n])
    # Rename columns so they are easier to read
    binnedcsv = binnedcsv.rename(columns=sheancoldict)
    
    # ===== Filter out bad values ==========================================================
    # Replace strings of nan with nan and make all columns floats or ints
    binnedcsv = binnedcsv.replace([' nan'], [np.nan])
    for col in binnedcsv.columns.values:
        binnedcsv[col] = pd.to_numeric(binnedcsv[col])
    # Remove bad values of dhdt
    binnedcsv.loc[binnedcsv[dhdt_cn] > dhdt_max, dhdt_cn] = np.nan
    binnedcsv.loc[binnedcsv[dhdt_cn] < dhdt_min, dhdt_cn] = np.nan
    # If dhdt is nan, remove row
    null_bins = binnedcsv.loc[pd.isnull(binnedcsv[dhdt_cn])].index.values
    binnedcsv = binnedcsv.drop(null_bins)
    # Add percent area valid to main_glac_rgi
    main_glac_rgi.loc[n, 'perc_areavalid'] = binnedcsv['z1_bin_area_perc'].sum()
    # Debris thickness
    binnedcsv['debris_thick_med_m'] = binnedcsv['debris_thick_med_m'].astype(float)
    binnedcsv.loc[pd.isnull(binnedcsv['debris_thick_med_m']), 'debris_thick_med_m'] = 0
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] < 0, 'debris_thick_med_m'] = 0
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] > 5, 'debris_thick_med_m'] = 0
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] == -0, 'debris_thick_med_m'] = 0       
    #Percent Debris
    binnedcsv.loc[binnedcsv['perc_debris'] > 100, 'perc_debris'] = 0
    binnedcsv.loc[binnedcsv['perc_debris'] <= 0, 'perc_debris'] = 0
    # Supraglacial ponds
    binnedcsv.loc[binnedcsv['perc_pond'] > 100, 'perc_pond'] = 0
    binnedcsv.loc[binnedcsv['perc_pond'] <= 0, 'perc_pond'] = 0
    # Clean ice
    binnedcsv.loc[binnedcsv['perc_clean'] > 100, 'perc_clean'] = 0
    binnedcsv.loc[binnedcsv['perc_clean'] <= 0, 'perc_clean'] = 0
    
    # Find glacier-wide debris perc for each glacier, and add to main_glac_rgi
    glacwide_debris = ((binnedcsv['z1_bin_area_valid_km2']*binnedcsv['perc_debris']).sum() 
                       / binnedcsv['z1_bin_area_valid_km2'].sum())
    main_glac_rgi.loc[n, 'PercDebris'] = glacwide_debris
    #sort out glaciers based on if they have all positive dh/dt, all negative, dh/dt, or both	
    #based on evaluating, for each glacier, the max from the list of dhdt and the min from the list. 	
    if np.nanmin(binnedcsv[dhdt_cn].astype(float)) >= 0: 		
        glacwide_dhdt_sign = 1 #glaciers with all positive dh/dt		
    elif np.nanmax(binnedcsv[dhdt_cn].astype(float)) <= 0: 		
        glacwide_dhdt_sign = -1 #glaciers with all negative dh/dt		
    else: 		
        glacwide_dhdt_sign = 0 #glaciers with both, + and - dh/dt 		
    main_glac_rgi.loc[n, 'dhdt_sign'] = glacwide_dhdt_sign
    
    # ===== Normalized elevation vs. ice thickness change ===============================
    # Normalized elevation
    #  (max elevation - bin elevation) / (max_elevation - min_elevation)
    glac_elev = binnedcsv.bin_center_elev_m.values
    binnedcsv['elev_norm'] = (glac_elev[-1] - glac_elev) / (glac_elev[-1] - glac_elev[0])
    # Normalized ice thickness change [ma]
    #  dhdt / dhdt_max
    glac_dhdt = binnedcsv[dhdt_cn].values.astype(float)
    # Shifted normalized ice thickness change such that everything is negative
    binnedcsv['dhdt_norm_shifted'] = (glac_dhdt - np.nanmax(glac_dhdt))  / np.nanmin(glac_dhdt - np.nanmax(glac_dhdt))
    binnedcsv.loc[binnedcsv['dhdt_norm_shifted'] == -0, 'dhdt_norm_shifted'] = 0
    # Replace positive values to zero
    glac_dhdt[glac_dhdt >= 0] = 0
    binnedcsv['dhdt_norm_huss'] = glac_dhdt / np.nanmin(glac_dhdt)
    binnedcsv.loc[binnedcsv['dhdt_norm_huss'] == -0, 'dhdt_norm_huss'] = 0
    
    # ===== ADD DATA TO MAIN DATASET =====================================================
    # ds is the main datset, n is index of each glacier
    # Keep only glaciers with enough good data based on percentage area valid
    if main_glac_rgi.loc[n, 'perc_areavalid'] > perc_area_valid_threshold:
        ds.append([n, df_glacnames.loc[n, 'RGIId'], binnedcsv, main_glac_rgi.loc[n]])
    #    ds.append([n, df_glacnames.loc[n, 'RGIId'], binnedcsv, main_glac_rgi.loc[n], main_glac_hyps.loc[n], 
    #               main_glac_icethickness.loc[n], ds_merged_bins])
    
#%% Remove Unwanted Glaciers
# NOTE: TO USE MAIN_GLAC_RGI ATTRIBUTES, NEED TO ACCESS THEM VIA THE DATASET
# remove them from both ds and norm_list
remove_idx = []

# Indices to remove Surging glaciers (listed as 1 possible, 2 probable, or 3 observed in main_glac_rgi)	
if option_remove_surge_glac == 1: 
    # Remove indices
    remove_idx_surge = [i for i in range(len(ds)) if ((ds[i][3].Surging != 9) and (ds[i][3].Surging != 0))]
    # Add unique values to list
    for i in remove_idx_surge:
        if i not in remove_idx:
            remove_idx.append(i)

# Indices to remove glaciers with all positive dh/dt values (listed as 1 in main_glac_rgi)
if option_remove_all_pos_dhdt == 1: 
    #add index of glaciers with all pos values to the Int64 Index list
    remove_idx_allposdhdt = [i for i in range(len(ds)) if ds[i][3].dhdt_sign == 1]
    for i in remove_idx_allposdhdt:
        if i not in remove_idx:
            remove_idx.append(i)

# Indices to remove glaciers who have max surface lowering in accumulation area
if option_remove_dhdt_acc == 1:
    remove_idx_acc = []
    for glac in range(len(ds)):
        glac_elevnorm = ds[glac][2]['elev_norm'].values
        glac_dhdt_norm = ds[glac][2]['dhdt_norm_huss'].values
        acc_idx = np.where(glac_elevnorm < 0.5)[0]
        if (glac_dhdt_norm[acc_idx] > acc_dhdt_threshold).any():
            remove_idx_acc.append(glac)
    for i in remove_idx_acc:
        if i not in remove_idx:
            remove_idx.append(i)
    
# ===== Remove glaciers =====
all_glac_idx = range(len(ds))
ds = [ds[i] for i in all_glac_idx if i not in remove_idx]

# ===== Normalized elevation versus ice thickness change list ======
# List of np.array where first column is elev_norm and second column is dhdt_norm
# Each item is a glacier
norm_list = [np.array([ds[i][2]['elev_norm'].values, ds[i][2]['dhdt_norm_huss'].values]).transpose() 
             for i in range(len(ds))]


#%% MEAN AND STANDARD DEVIATIONS OF CURVES (black lines to add onto plots)
def normalized_stats(norm_list): 
    # Merge norm_list to make array of all glaciers with same elevation normalization space
    max_length = len(max(norm_list,key=len)) #len of glac w most norm values
    norm_all = np.zeros((max_length, len(norm_list)+1)) #array: each col a glac, each row a norm dhdt val to be interpolated 
    # First column is normalized elevation, pulled from the glac with most norm vals
    norm_all[:,0] = max(norm_list,key=len)[:,0]  
    
    # Loop through each glacier's normalized array (where col1 is elev_norm and col2 is norm dhdt)
    for n in range(len(norm_list)):
#        print(main_glac_rgi.loc[n,'RGIId']) #NOT SURE IF THIS WILL SHOW THE CORRECT CORRESPONDING GLACIER
        norm_single = norm_list[n] # get one glacier at a time 
        
#        #Skip over glaciers that contain only NaN values for normalized dhdt 
#        #(I added this so that it could run, but I want to be sure it doesn't have weird implications.)
#        if np.isnan(norm_single[:,1][0]) == True and np.isnan(norm_single[:,1][-1]) == True:
##            print('The current glacier likely only contains NaNs, and is being skipped')
#            continue
#        #also skip over glaciers that contain almost all 0 values .
        
        # Fill in nan values for elev_norm of 0 and 1 with nearest neighbor
        norm_single[0,1] = norm_single[np.where(~np.isnan(norm_single[:,1]))[0][0], 1]
        norm_single[-1,1] = norm_single[np.where(~np.isnan(norm_single[:,1]))[0][-1], 1]
        # Remove nan values. 
        norm_single = norm_single[np.where(~np.isnan(norm_single[:,1]))] #~ is the same as !
        elev_single = norm_single[:,0] #set name for first col of a given glac
        dhdt_single = norm_single[:,1] #set name for the second col of a given glac
        #loop through each dhdt value of the glacier, and add it and interpolate to add to the 
        #norm_all array. 
        for r in range(0, max_length):
            if r == 0:
                norm_all[r,n+1] = dhdt_single[0] #put the first value dhdt value into the norm_all. n+1 because the first col is taken by the elevnorms.
            elif r == (max_length - 1):
                norm_all[r,n+1] = dhdt_single[-1] #put the last value into the the last row for the glacier's 'stretched out'(interpolated) normalized curve.
            else:
                # Find value need to interpolate to
                norm_elev_value = norm_all[r,0] #go through each row in the elev (col1)
                # Find index of value above it from dhdt_norm, which is a different size
                upper_idx = np.where(elev_single == elev_single[elev_single >= norm_elev_value].min())[0][0]
                # Find index of value below it
                lower_idx = np.where(elev_single == elev_single[elev_single < norm_elev_value].max())[0][0]
                #get the two values, based on the indices. 
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
    norm_all_stats['norm_dhdt_mean'] = np.nanmean(norm_all[:,1:], axis=1)  
    norm_all_stats['norm_dhdt_std'] = np.nanstd(norm_all[:,1:], axis=1)
    norm_all_stats['norm_dhdt_68high'] = norm_all_stats['norm_dhdt_mean'] + norm_all_stats['norm_dhdt_std']
    norm_all_stats['norm_dhdt_68low'] = norm_all_stats['norm_dhdt_mean'] - norm_all_stats['norm_dhdt_std']
    norm_all_stats.loc[norm_all_stats['norm_dhdt_68high'] > 1, 'norm_dhdt_68high'] = 1
    norm_all_stats.loc[norm_all_stats['norm_dhdt_68low'] < 0, 'norm_dhdt_68low'] = 0
    return norm_all_stats

norm_stats = normalized_stats(norm_list)

#%% Plots comparing glacier parameters to see if any are related
if option_plot_compareparameters == 1:
    parameter1 = 'PercDebris'
    parameter2 = 'Slope'
    A = np.array([ds[x][3][parameter1] for x in range(len(ds))])
    B = np.array([ds[x][3][parameter2] for x in range(len(ds))])
    
    param_label_dict = {'Area': 'Area [km2]',
                        'PercDebris': 'Debris cover[%]',
                        'Slope':'Slope [deg]'}
    # ===== PLOT =====
    fig_width = 4
    fig_height = 3
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(fig_width,fig_height), 
                           gridspec_kw = {'wspace':0.2, 'hspace':0.5})
    ax[0,0].scatter(A,B, color='k', s=1)
    ax[0,0].set_xlabel(param_label_dict[parameter1], size=14)
    ax[0,0].set_ylabel(param_label_dict[parameter2], size=14)
    # Save figure
    fig.savefig(fig_fp + ('scatter_' + parameter1 + '_' + parameter2 + '.png'), bbox_inches='tight', dpi=300)
    

#%% Plots for a histogram of parameter (distribution of values)		
def plot_var_histogram():		     		
   #plot histogram, where x-axis is the testing_var values, and y-axis is how many glaciers have that given x-axis value	
   for parameter in histogram_parameters:
       parameter_values = np.array([ds[i][3][parameter] for i in range(len(ds))])
       plt.hist(parameter_values, 50)		
       plt.xlabel(parameter)	
       plt.ylabel('Number of glaciers')		
       plt.title(parameter + ' Distribution' ' Region' + str(rgi_regionO1))		
       plt.minorticks_on()		
        		
       if option_savefigs == 1:		
           hist_fp = fig_fp + 'histograms/'
           if not os.path.exists(hist_fp):
               os.makedirs(hist_fp)
           plt.savefig(hist_fp + parameter + '_histogram_reg_' + str(rgi_regionO1), bbox_inches='tight')		
       plt.show()		
       		
       parameter_lower_bound = int(parameter_values.min())		
       parameter_upper_bound = np.ceil(parameter_values.max())		
       print('Range of '+ parameter+ ': (' + str(parameter_lower_bound) + ', ' + str(parameter_upper_bound) + ')')

if option_plot_histogram == 1:
    plot_var_histogram()
    

#%% Plots for a single glacier    
def plot_eachglacier(ds, option_merged_dataset=0):
    # Set position of dataset to plot in list based on using merged or unmerged elev bin data
    #  [2 = 10m, 6 = merged]
    if option_merged_dataset == 0:
        ds_position = 2
    elif option_merged_dataset == 1:
        ds_position = 6
    
    individual_fp = fig_fp + 'individual_plots/'
    if not os.path.exists(individual_fp):
        os.makedirs(individual_fp)
    
    # Loop through glaciers and plot
    for glac in range(len(ds)):
        #pull values from binnedcsv into vars
        glac_elevbins = ds[glac][ds_position]['bin_center_elev_m']
        glac_area_t1 = ds[glac][ds_position]['z1_bin_area_valid_km2']
        glac_area_t2 = ds[glac][ds_position]['z2_bin_area_valid_km2']
        glac_mb_mwea = ds[glac][ds_position][mb_cn]
        glac_debristhick_cm = ds[glac][ds_position]['debris_thick_med_m'] * 100
        glac_debrisperc = ds[glac][ds_position]['perc_debris']
        glac_pondperc = ds[glac][ds_position]['perc_pond']
        glac_elevnorm = ds[glac][ds_position]['elev_norm']
        glac_dhdt_med = ds[glac][ds_position]['dhdt_bin_med_ma']
        glac_dhdt_norm_huss = ds[glac][ds_position]['dhdt_norm_huss']
        glac_dhdt_norm_shifted = ds[glac][ds_position]['dhdt_norm_shifted']
        glac_elevs = ds[glac][ds_position]['bin_center_elev_m']
        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
        glac_name = ds[glac][1].split('-')[1]
        
         # dhdt (raw) vs. elevation (raw)
        plt.figure(figsize = (20, 12))
        plt.plot(glac_elevs, glac_dhdt_med, label=glac_name)
        plt.gca().invert_xaxis()
        plt.xlabel('Elevation (m)')
        plt.ylabel('Ice thickness Change [m/a]')
        plt.title('Raw dh/dt\n')
        plt.minorticks_on() 
            
        # Plot Elevation bins vs. Area, Mass balance, and Debris thickness/pond coverage/ debris coverage
        plt.figure(figsize=(10,6))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.suptitle(ds[glac][1], y=0.94)
        # Elevation vs. Area
        plt.subplot(1,3,1)
        plt.plot(glac_area_t1, glac_elevbins, label='t1')
        plt.plot(glac_area_t2, glac_elevbins, label='t2')
        plt.ylabel('Elevation [masl, WGS84]')
        plt.xlabel('Glacier area [km2]')
        plt.minorticks_on()
        plt.legend()
        # Elevation vs. Mass Balance
        plt.subplot(1,3,2)
        plt.plot(glac_mb_mwea, glac_elevbins, 'k-', label=str(round(glacwide_mb_mwea, 2)) + ' mwea')
        #  k refers to the color (k=black, b=blue, g=green, etc.)
        #  - refers to using a line (-- is a dashed line, o is circle points, etc.)
        plt.ylabel('Elevation [masl, WGS84]')
        plt.xlabel('Mass balance [mwea]')
        plt.xlim(-3, 3)
        plt.xticks(np.arange(-3, 3 + 1, 1))
        plt.axvline(x=0, color='k')
        plt.fill_betweenx(glac_elevbins, glac_mb_mwea, 0, where=glac_mb_mwea<0, color='r', alpha=0.5)
        plt.fill_betweenx(glac_elevbins, glac_mb_mwea, 0, where=glac_mb_mwea>0, color='b', alpha=0.5)
        plt.legend(loc=1)
        plt.minorticks_on()
        plt.gca().axes.get_yaxis().set_visible(False)
        # Elevation vs. Debris Area, Pond Area, Thickness
        plt.subplot(1,3,3)
        plt.plot(glac_debrisperc, glac_elevbins, label='Debris area')
        plt.plot(glac_pondperc, glac_elevbins, label='Pond area')
        plt.plot(glac_debristhick_cm, glac_elevbins, 'k-', label='Thickness')
        plt.ylabel('Elevation [masl, WGS84]')
        plt.xlabel('Debris thickness [cm], Area [%]')
        plt.minorticks_on()
        plt.legend()
        plt.gca().axes.get_yaxis().set_visible(False)
        if option_savefigs == 1:
            plt.savefig(individual_fp + '/mb_fig' + ds[glac][1] + '_mb_aed.png', bbox_inches='tight')
        plt.show()
        
        # Elevation change vs. Elevation
        plt.figure(figsize=(10,3))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        # Normalized curves using range of dh/dt
        plt.subplot(1,3,1)
        plt.plot(glac_elevs, glac_dhdt_med, label=ds[glac][1])
        plt.gca().invert_xaxis()
        plt.xlabel('Elevation [m]')
        plt.ylabel('dh/dt [m/a]')
        plt.title('dhdt vs elevation')
        plt.minorticks_on()
        plt.legend()
        # Normalized curves using dhdt max (according to Huss)
        plt.subplot(1,3,2)
        plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label=ds[glac][1])
        plt.xlabel('Normalized elev range')
        plt.ylabel('Normalized dh/dt [ma]')
        plt.title('huss normalization')
        if glac_dhdt_med.min() < 0:
            plt.gca().invert_yaxis()
        plt.minorticks_on()
        plt.legend()
        # Normalized curves shifting all values to be negative
        plt.subplot(1,3,3)
        plt.plot(glac_elevnorm, glac_dhdt_norm_shifted, label=ds[glac][1])
        plt.ylim(1,0)
        plt.xlabel('Normalized elev range')
        plt.title('shifted normalization')
        plt.minorticks_on()
        plt.legend()
        if option_savefigs == 1:
            plt.savefig(individual_fp + 'Single_Plots' + ds[glac][1] + '_normcurves.png', bbox_inches='tight')
        plt.show()

if option_plot_eachglacier == 1:
    plot_eachglacier(ds, option_merged_dataset=option_use_mergedata)
    
#%% Plot multiple glaciers on the same plot  
def plot_multipleglaciers_single_threshold(ds, option_merged_dataset=0, parameter='Area', threshold_n=0):
    # Set position of dataset to plot in list based on using merged or unmerged data
    #  [2 = 10m, 6 = merged]
    if option_merged_dataset == 0:
        ds_position = 2 #refer to binnedcsv
    elif option_merged_dataset == 1:
        ds_position = 6 #refers to the ds of merged elev bin data
        
    #plot empty figure
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    #set counters to keep track of total number of glac > and < threshold
    count_lt = 0
    count_gt = 0
    norm_list_gt = []
    norm_list_lt = []
    
    # Parameter values
#    parameter_values = np.array([ds[i][3][parameter] for i in range(len(ds))])
    
    #loop through each glacier, in order of ascending parameter, accessing binnedcsv values
    for glac in range(len(ds)):
        glac_rgi = ds[glac][3]
#        glac_elevbins = ds[glac][ds_position]['bin_center_elev_m']
#        glac_area_t1 = ds[glac][ds_position]['z1_bin_area_valid_km2']
#        glac_area_t2 = ds[glac][ds_position]['z2_bin_area_valid_km2']
#        glac_area_t1_perc = ds[glac][ds_position]['z1_bin_area_perc']
#        glac_bin_count_t1 = ds[glac][ds_position]['z1_bin_count_valid']
#        glac_mb_mwea = ds[glac][ds_position][mb_cn]
#        glac_debristhick_cm = ds[glac][ds_position]['debris_thick_med_m'] * 100
#        glac_debrisperc = ds[glac][ds_position]['perc_debris']
#        glac_pondperc = ds[glac][ds_position]['perc_pond']
        glac_elevnorm = ds[glac][ds_position]['elev_norm']
        glac_dhdt_norm_huss = ds[glac][ds_position]['dhdt_norm_huss']
#        glac_dhdt_norm_shifted = ds[glac][ds_position]['dhdt_norm_shifted']
        glac_dhdt_med = ds[glac][ds_position]['dhdt_bin_med_ma']
#        glac_dhdt_mean = ds[glac][ds_position]['dhdt_bin_mean_ma']
#        glac_dhdt_std = ds[glac][ds_position]['dhdt_bin_std_ma']
        glac_elevs = ds[glac][ds_position]['bin_center_elev_m']
#        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
        glac_name = ds[glac][1].split('-')[1]
        
        # Subset parameters based on column name and threshold
        if glac_rgi[parameter] < threshold_n:
            count_lt += 1
            # Make list of array containing elev_norm and dhdt_norm_huss 
            norm_list_lt.append(np.array([glac_elevnorm.values, 
                                          glac_dhdt_norm_huss.values]).transpose())
            
            # dhdt (raw) vs. elevation (raw)
            plt.subplot(2,2,1)
            plt.plot(glac_elevs, glac_dhdt_med, label=glac_name)
            if count_lt == 1:
                plt.gca().invert_xaxis()
            plt.xlabel('Elevation (m)')
            plt.ylabel('dh/dt [m/a]')
            plt.title('Raw dh/dt\n' + parameter + '<' + str(threshold_n))
            plt.minorticks_on()    

            # Huss Norm dhdt vs. Norm Elev
            plt.subplot(2,2,2)
            plt.rcParams.update({'font.size': 12})
            plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label=glac_name, alpha=glacier_plots_transparency)
            plt.xlabel('Normalized Elevation Range')
            plt.ylabel('Normalized dh/dt')
            if count_lt == 1:
                plt.gca().invert_yaxis()
            plt.title('Huss Normalization (' + str(count_lt) + ' Glaciers)\n' + parameter + '<' + str(threshold_n))
            plt.minorticks_on()
            
            if option_show_legend == 1:
                plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
                
        # Subset parameters based on column name and threshold
        elif glac_rgi[parameter] >= threshold_n:
            count_gt += 1
            # Make list of array containing elev_norm and dhdt_norm_huss 
            norm_list_gt.append(np.array([glac_elevnorm.values,
                                          glac_dhdt_norm_huss.values]).transpose())
            
            # dhdt vs. elevation
            plt.subplot(2,2,3)
            plt.plot(glac_elevs, glac_dhdt_med, label=glac_name)
            if count_gt == 1:
                plt.gca().invert_xaxis()
            plt.xlabel('Elevation (m)')
            plt.ylabel('dh/dt [m/a]')
            plt.title('Raw dh/dt\n' + parameter + '>' + str(threshold_n))
            plt.minorticks_on()    
            
            # Normalized curves using dhdt max (according to Huss)
            plt.subplot(2,2,4)
            plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label=glac_name, alpha=glacier_plots_transparency)
            plt.xlabel('Normalized Elevation Range')
            plt.ylabel('Normalized dh/dt')
            if count_gt == 1:
                plt.gca().invert_yaxis()
            plt.title('Huss Normalization (' + str(count_gt) +' Glaciers)\n' + parameter + '>' + str(threshold_n))
            plt.minorticks_on()
      
            #display legend, if defined as such in "Input Data" section
            if option_show_legend == 1:
                plt.legend(bbox_to_anchor=(1.2, 1), loc=3, borderaxespad=0.)
    
    print(count_gt, 'Glaciers above threshold for', parameter)
    print(count_lt, 'Glaciers below threshold for', parameter)

    # Put mean and plus/minus 1 standard deviation on normalized plots
    norm_lt_stats = pd.DataFrame()
    norm_gt_stats = pd.DataFrame()

    if count_lt > 1:
        norm_lt_stats = normalized_stats(norm_list_lt)
        # Less than threshold plots
        plt.subplot(2,2,2)
        plt.plot(norm_lt_stats.norm_elev, norm_lt_stats.norm_dhdt_mean, color='black', linewidth=2)
        plt.plot(norm_lt_stats.norm_elev, norm_lt_stats.norm_dhdt_68high, '--', color='black', linewidth=1.5)
        plt.plot(norm_lt_stats.norm_elev, norm_lt_stats.norm_dhdt_68low, '--', color='black', linewidth=1.5)
    if count_gt > 1:
        norm_gt_stats = normalized_stats(norm_list_gt)
        # Greater than threshold plots
        plt.subplot(2,2,4)
        plt.plot(norm_gt_stats.norm_elev, norm_gt_stats.norm_dhdt_mean, color='black', linewidth=2)
        plt.plot(norm_gt_stats.norm_elev, norm_gt_stats.norm_dhdt_68high, '--', color='black', linewidth=1.5)
        plt.plot(norm_gt_stats.norm_elev, norm_gt_stats.norm_dhdt_68low, '--', color='black', linewidth=1.5)
    
    # Add title to subplot
    plot_fn = 'R' + str(rgi_regionO1) +'_' +  parameter + '_' + str(threshold_n)
    plt.suptitle(plot_fn)
    # Save and show figure
    if option_savefigs == 1:
        multiglacier_fp = fig_fp + 'multiple_glaciers/'
        if not os.path.exists(multiglacier_fp):
            os.makedirs(multiglacier_fp)
        plt.savefig(multiglacier_fp + plot_fn + '_dhdt_elev_curves.png', bbox_inches='tight')
    plt.show()
    return norm_gt_stats, norm_lt_stats

if option_plot_multipleglaciers_single_thresholds == 1:
#    norm_gt_stats, norm_lt_stats = plot_multipleglaciers_single_threshold(
#            ds, option_merged_dataset=0, parameter='Area', threshold_n=5)
    #loop through each parameter, and its respective threshold list
    for parameter, thresholds in (region_pars.items()):
        #loop through each threshold within the list of thresholds of a prmtr
        for threshold_n in thresholds:
            #call the fxn
            norm_gt_stats, norm_lt_stats = plot_multipleglaciers_single_threshold(
                    ds, parameter=parameter, threshold_n=threshold_n)


#%% Plot multiple glaciers on the same plot  
def plot_multipleglaciers_multiplethresholds(ds, parameter='Area', thresholds_raw=[0]):
    """
    Plot all glaciers for multiple thresholds
    
    Parameters
    ----------
    ds : list of lists
        main dataset containing elevation, dh/dt, glacier rgi table and other data for each glacier
    parameter : str
        parameter name (needs to match parameter name in glacier rgi table)
    thresholds_raw : list of integers
        threshold values; they are considered "raw" because the function automatically includes a greater than of the 
        last threshold, so [5, 10] will look at 3 thresholds: "< 5", "5 - 10", and "> 10"
    
    Returns
    -------
    Two plots of the normalized elevation dh/dt curves.
        1. Normalized elevation vs normalized dh/dt with mean and standard deviation included with each threshold having
           a separate subplot
        2. Mean normalized elevation vs. normalized dh/dt for each threshold on a single plot
    """
    # Set position of dataset to plot in list based on using merged or unmerged data
    ds_position = 2 #refer to binnedcsv
    
    # Sort list according to parameter
    ds_sorted = copy.deepcopy(ds)
    for i in ds_sorted:
        i.append(i[3][parameter])
    ds_sorted.sort(key=lambda x: x[4])
    
    # Add maximum threshold to threshold such that don't need a greater than statement
    max_list = max(ds_sorted, key=lambda x: x[4])
    max_threshold = max_list[4] + 1
    thresholds=copy.deepcopy(thresholds_raw)
    thresholds.append(max_threshold)
    
    # Count number of glaciers per threshold and record list of values for each threshold's plot
    count_glac_per_threshold = []
    normlist_glac_per_threshold = []
    for n, threshold in enumerate(thresholds):
        count_glac = 0
        normlist_glac = []
        
        for glac in range(len(ds_sorted)):
            glac_rgi = ds_sorted[glac][3]
            glac_elevnorm = ds_sorted[glac][ds_position]['elev_norm']
            glac_dhdt_norm_huss = ds_sorted[glac][ds_position]['dhdt_norm_huss']
#            glac_dhdt_med = ds_sorted[glac][ds_position]['dhdt_bin_med_ma']
#            glac_elevs = ds_sorted[glac][ds_position]['bin_center_elev_m']
#            glac_name = ds_sorted[glac][1].split('-')[1]
        
            if n == 0:
                if glac_rgi[parameter] < threshold:
                    count_glac += 1
                    # Make list of array containing elev_norm and dhdt_norm_huss 
                    normlist_glac.append(np.array([glac_elevnorm.values, 
                                                    glac_dhdt_norm_huss.values]).transpose())
            else:
                if thresholds[n-1] < glac_rgi[parameter] < threshold:
                    count_glac += 1
                    # Make list of array containing elev_norm and dhdt_norm_huss
                    normlist_glac.append(np.array([glac_elevnorm.values, 
                                                    glac_dhdt_norm_huss.values]).transpose())
        # Record glaciers per threshold
        count_glac_per_threshold.append(count_glac)
        normlist_glac_per_threshold.append(normlist_glac)
    
    # ===== PLOT =====
    # Plot the normalized curves
    fig_width = 5
    fig, ax = plt.subplots(len(thresholds), 1, squeeze=False, figsize=(fig_width,int(3*len(thresholds))), 
                           gridspec_kw = {'wspace':0.2, 'hspace':0.5})
            
    normlist_stats_all = []
    for n, threshold in enumerate(thresholds):

        # Extract values to plot
        normlist = normlist_glac_per_threshold[n]
        
        for glac in range(len(normlist)):
            normlist_glac = normlist[glac]
        
            # Normalized elevation vs. normalized dh/dt
            ax[n,0].plot(normlist_glac[:,0], normlist_glac[:,1], linewidth=1, alpha=glacier_plots_transparency, 
                         label=None)
            ax[n,0].set_ylim(max(normlist_glac[:,0]), min(normlist_glac[:,0]))
            ax[n,0].set_xlim(0,1)
            ax[n,0].set_ylabel('Normalized Elevation [-]', size=12)
            ax[n,0].set_xlabel('Normalized dh/dt [-]', size=12)
            ax[n,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax[n,0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax[n,0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax[n,0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
            
            if threshold == thresholds[0]:
                ax[n,0].set_title((parameter + '<' + str(threshold) + ' (' + str(len(normlist)) + ' Glaciers)'), 
                                  size=12)
            elif threshold != thresholds[-1]:
                ax[n,0].set_title((str(thresholds[n-1]) + '<' + parameter + '<' + str(threshold) + ' (' + 
                                   str(len(normlist)) + ' Glaciers)'), size=12)
            else:
                ax[n,0].set_title((parameter + '>' + str(thresholds[n-1]) + ' (' + str(len(normlist)) + ' Glaciers)'), 
                                  size=12)
                
        # Add statistics to plot        
        normlist_stats = normalized_stats(normlist)
        ax[n,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_mean, color='black', linewidth=2)
        ax[n,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_68high, '--', color='black', linewidth=1.5) 
        ax[n,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_68low, '--', color='black', linewidth=1.5)
        # Record stats to plot on separate graph
        normlist_stats_all.append(normlist_stats)
    
    # Save figure
    fig.set_size_inches(fig_width, int(len(thresholds)*3))
    threshold_str_list = [str(i) for i in thresholds]
    threshold_str_list[-1] = 'max'
    threshold_str = '-'.join(threshold_str_list)
    fig.savefig(fig_fp + ('normcurves' + parameter + '_' + threshold_str + '.png'), bbox_inches='tight', dpi=300)
                
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
        
        # Plot mean of each
        ax[0,0].plot(normlist_stats.norm_elev, normlist_stats.norm_dhdt_mean, linewidth=2, label=threshold_label)
        ax[0,0].set_ylim(max(normlist_glac[:,0]), min(normlist_glac[:,0]))
        ax[0,0].set_xlim(0,1)
        ax[0,0].set_ylabel('Normalized Elevation [-]', size=12)
        ax[0,0].set_xlabel('Normalized dh/dt [-]', size=12)
        ax[0,0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax[0,0].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax[0,0].xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax[0,0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax[0,0].legend(loc='lower left')
    
    # Save figure
    fig.savefig(fig_fp + ('normcurves' + parameter + '_' + threshold_str + '_MEANS.png'), bbox_inches='tight', dpi=300)


if option_plot_multipleglaciers_multiplethresholds == 1:
    plot_multipleglaciers_multiplethresholds(ds, parameter='Area', thresholds_raw=[5,10])
        
    

#%% 
##%%PLOT Multiple Glaciers, with threshold bins for the parameter
#def plot_multipleglaciers_binned_parameter(glacier_list, option_merged_dataset, parameter='Area', threshold_n=0):
#    # Set position of dataset to plot in list based on using merged or unmerged data
#    #  [2 = 10m, 6 = merged]
#    if option_merged_dataset == 0:
#        ds_position = 2 #refer to binnedcsv
#    elif option_merged_dataset == 1:
#        ds_position = 6 #refers to the ds of merged elev bin data
#        
#    #plot empty figure
#    plt.figure(figsize=(10,6))
#    plt.subplots_adjust(wspace=0.4, hspace=0.6)
#    
#    #set counters to keep track of total number of glac > and < threshold
#    count_current = 0
#    norm_list_thresholdbin = []
#
#    #sort main_glac_rgi by parameter, in ascending order
#    main_glac_rgi.sort_values(by=[parameter], inplace = True)
#    prmtr_sorted_glacier_list = list(main_glac_rgi.index.values) #list of desired order to loop through ds
#    #loop through each glacier, in order of ascending parameter, accessing binnedcsv values
#    for glac in prmtr_sorted_glacier_list:
#        glac_rgi = ds[glac][3] #accessing the glacier's main_glac_rgi row
#        glac_elevbins = ds[glac][ds_position]['bin_center_elev_m']
#        glac_area_t1 = ds[glac][ds_position]['z1_bin_area_valid_km2']
#        glac_area_t2 = ds[glac][ds_position]['z2_bin_area_valid_km2']
#        glac_area_t1_perc = ds[glac][ds_position]['z1_bin_area_perc']
#        glac_bin_count_t1 = ds[glac][ds_position]['z1_bin_count_valid']
#        glac_mb_mwea = ds[glac][ds_position][mb_cn]
#        glac_debristhick_cm = ds[glac][ds_position]['debris_thick_med_m'] * 100
#        glac_debrisperc = ds[glac][ds_position]['perc_debris']
#        glac_pondperc = ds[glac][ds_position]['perc_pond']
#        glac_elevnorm = ds[glac][ds_position]['elev_norm']
#        glac_dhdt_norm_huss = ds[glac][ds_position]['dhdt_norm_huss']
##        glac_dhdt_norm_range = ds[glac][ds_position]['dhdt_norm_range']
#        glac_dhdt_norm_shifted = ds[glac][ds_position]['dhdt_norm_shifted']
#        glac_dhdt_med = ds[glac][ds_position]['dhdt_bin_med_ma']
#        glac_dhdt_mean = ds[glac][ds_position]['dhdt_bin_mean_ma']
#        glac_dhdt_std = ds[glac][ds_position]['dhdt_bin_std_ma']
#        glac_elevs = ds[glac][ds_position]['bin_center_elev_m']
#        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
#        t1 = 2000
#        t2 = 2015
#        glac_name = ds[glac][1].split('-')[1]
##        
#        
#        # Subset parameters based on column name and threshold
#        if glac_rgi[parameter] >= threshold_n and glac_rgi[parameter] < thresholds[next_threshold]: 
#            print(glac_name, ' is between ', threshold_n, ' and ', thresholds[next_threshold])
#            count_current += 1
#            # Make list of array containing elev_norm and dhdt_norm_huss 
#            norm_list_thresholdbin.append(np.array([glac_elevnorm.values, glac_dhdt_norm_huss.values]).transpose())
#            
#            # dhdt (raw) vs. elevation (raw)
#            plt.subplot(2,2,1)
#            plt.plot(glac_elevs, glac_dhdt_med, label=glac_name)
#            if count_current == 1:
#                plt.gca().invert_xaxis()
#            plt.xlabel('Elevation (m)')
#            plt.ylabel('dh/dt [m/a]')
#            plt.title('Raw dh/dt')
#            plt.minorticks_on()    
#
#            # Huss Norm dhdt vs. Norm Elev
#            plt.subplot(2,2,2)
#            plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label=glac_name, alpha=0.3)
#            plt.xlabel('Normalized Elevation Range')
#            plt.ylabel('Normalized dh/dt')
#            if count_current == 1:
#                plt.gca().invert_yaxis()
#            plt.title('Huss Normalization')
#            plt.minorticks_on()
#            
#            if option_show_legend == 1:
#                plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
#                
#    print(count_current, 'glaciers total' )
#
#    # Put mean and plus/minus 1 standard deviation on normalized plots
#    norm_thresholdbin_stats = pd.DataFrame()
#    if count_current > 1:
#        norm_thresholdbin_stats = normalized_stats(norm_list_thresholdbin)
#        # Less than threshold plots
#        plt.subplot(2,2,2)
#        plt.plot(norm_thresholdbin_stats.norm_elev, norm_thresholdbin_stats.dhdt_mean, color='black', linewidth=2)
#        plt.plot(norm_thresholdbin_stats.norm_elev, norm_thresholdbin_stats.dhdt_68high, '--', color='black', linewidth=1.5)
#        plt.plot(norm_thresholdbin_stats.norm_elev, norm_thresholdbin_stats.dhdt_68low, '--', color='black', linewidth=1.5)
#  
#    # Add title to subplot
#    plot_fn = ('reg_' + str(rgi_regionO1) +'_' +  parameter +'_(' + str(threshold_n) +
#                            ',' + str(thresholds[next_threshold]) + ')_' + str(bin_size))
#    plt.suptitle(plot_fn + '\n '+  str(count_current) + ' glaciers')
#    # Save and show figure
#    if option_savefigs == 1:
#        plt.savefig(input.output_filepath + 'figures/Multi_Glac_Plots_Binned_Parameter/' + plot_fn + '_dhdt_elev_curves.png', bbox_inches='tight')
#    plt.show()
#    return norm_thresholdbin_stats
#
##%%PLOT Multiple Glaciers, divided into 6 bins, each with equal num of glaciers
## NOTE - This section is not used. Sorting in this way does NOT appear to be 
## as effective of a way to bin or analyze the data for our purposes. 
#def plot_multipleglaciers_equal_subdivisions(glacier_list, option_merged_dataset, parameter='Area', threshold_n=0):
#    # Set position of dataset to plot in list based on using merged or unmerged data
#    #  [2 = 10m, 6 = merged]
#    if option_merged_dataset == 0:
#        ds_position = 2 #refer to binnedcsv
#    elif option_merged_dataset == 1:
#        ds_position = 6 #refers to the ds of merged elev bin data
#        
#    #plot empty figure
#    plt.figure(figsize=(10,6))
#    plt.subplots_adjust(wspace=0.4, hspace=0.6)
#    
#    #set counters to keep track of total number of glac > and < threshold
#    count_current = 0
#    norm_list_thresholdbin = []
#    
#    #sort main_glac_rgi by parameter, in ascending order
#    main_glac_rgi.sort_values(by=[parameter], inplace = True)
#    prmtr_sorted_glacier_list = list(main_glac_rgi.index.values) #list of desired order to loop through ds
#    #loop through each glacier, in order of ascending parameter, accessing binnedcsv values
#    sample_size = len(main_glac_rgi//6)
#    for glac in prmtr_sorted_glacier_list:
#        glac_rgi = ds[glac][3]
#        glac_elevbins = ds[glac][ds_position]['bin_center_elev_m']
#        glac_area_t1 = ds[glac][ds_position]['z1_bin_area_valid_km2']
#        glac_area_t2 = ds[glac][ds_position]['z2_bin_area_valid_km2']
#        glac_area_t1_perc = ds[glac][ds_position]['z1_bin_area_perc']
#        glac_bin_count_t1 = ds[glac][ds_position]['z1_bin_count_valid']
#        glac_mb_mwea = ds[glac][ds_position][mb_cn]
#        glac_debristhick_cm = ds[glac][ds_position]['debris_thick_med_m'] * 100
#        glac_debrisperc = ds[glac][ds_position]['perc_debris']
#        glac_pondperc = ds[glac][ds_position]['perc_pond']
#        glac_elevnorm = ds[glac][ds_position]['elev_norm']
#        glac_dhdt_norm_huss = ds[glac][ds_position]['dhdt_norm_huss']
##        glac_dhdt_norm_range = ds[glac][ds_position]['dhdt_norm_range']
#        glac_dhdt_norm_shifted = ds[glac][ds_position]['dhdt_norm_shifted']
#        glac_dhdt_med = ds[glac][ds_position]['dhdt_bin_med_ma']
#        glac_dhdt_mean = ds[glac][ds_position]['dhdt_bin_mean_ma']
#        glac_dhdt_std = ds[glac][ds_position]['dhdt_bin_std_ma']
#        glac_elevs = ds[glac][ds_position]['bin_center_elev_m']
#        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
#        t1 = 2000
#        t2 = 2015
#        glac_name = ds[glac][1].split('-')[1]
#        glac_rgi[parameter].sort()
#        
#        # Subset parameters based on column name and threshold
#        if glac_rgi[parameter] >= threshold_n and glac_rgi[parameter] < thresholds[next_threshold]: 
#            print(glac_name, ' is between ', threshold_n, ' and ', thresholds[next_threshold])
#            count_current += 1
#            # Make list of array containing elev_norm and dhdt_norm_huss 
#            norm_list_thresholdbin.append(np.array([glac_elevnorm.values, glac_dhdt_norm_huss.values]).transpose())
#            
#            # dhdt (raw) vs. elevation (raw)
#            plt.subplot(2,2,1)
#            plt.plot(glac_elevs, glac_dhdt_med, label=glac_name)
#            if count_current == 1:
#                plt.gca().invert_xaxis()
#            plt.xlabel('Elevation (m)')
#            plt.ylabel('dh/dt [m/a]')
#            plt.title('Raw dh/dt')
#            plt.minorticks_on()    
#
#            # Huss Norm dhdt vs. Norm Elev
#            plt.subplot(2,2,2)
#            plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label=glac_name, alpha=0.3)
#            plt.xlabel('Normalized Elevation Range')
#            plt.ylabel('Normalized dh/dt')
#            if count_current == 1:
#                plt.gca().invert_yaxis()
#            plt.title('Huss Normalization')
#            plt.minorticks_on()
#            
#            if option_show_legend == 1:
#                plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
#                
#    # Put mean and plus/minus 1 standard deviation on normalized plots
#    norm_thresholdbin_stats = pd.DataFrame()
#    if count_current > 1:
#        norm_thresholdbin_stats = normalized_stats(norm_list_thresholdbin)
#        # Less than threshold plots
#        plt.subplot(2,2,2)
#        plt.plot(norm_thresholdbin_stats.norm_elev, norm_thresholdbin_stats.dhdt_mean, color='black', linewidth=2)
#        plt.plot(norm_thresholdbin_stats.norm_elev, norm_thresholdbin_stats.dhdt_68high, '--', color='black', linewidth=1.5)
#        plt.plot(norm_thresholdbin_stats.norm_elev, norm_thresholdbin_stats.dhdt_68low, '--', color='black', linewidth=1.5)
#  
#    # Add title to subplot
#    plot_fn = ('reg_' + str(rgi_regionO1) +'_' +  parameter +'_(' + str(threshold_n) +
#                            ',' + str(thresholds[next_threshold]) + ')_' + str(bin_size))
#    plt.suptitle(plot_fn + '\n '+  str(count_current) + ' glaciers')
#    # Save and show figure
#    if option_savefigs == 1:
#        plt.savefig(input.output_filepath + 'figures/Multi_Glac_Plots_Binned_Parameter/' + plot_fn + '_dhdt_elev_curves.png', bbox_inches='tight')
#    plt.show()
#    return norm_thresholdbin_stats


#%% Call Functions (based on user-defined options from section 1)
# Index of glaciers to loop through
#glacier_list = list(range(0,len(norm_list)))


#if option_plot_multipleglaciers_binned_parameter == 1:
#    #loop through each threshold, and run the multipleglacier plot fxn
#    for parameter, thresholds in (region_pars.items()):
#        next_threshold = 1
#        for threshold_n in thresholds:
#            norm_thresholdbin_stats = (plot_multipleglaciers_binned_parameter(
#                    glacier_list, option_merged_dataset=1,parameter=parameter,
#                    threshold_n=threshold_n))
#            if next_threshold < (len(thresholds) - 1):
#                next_threshold += 1
#            
##if option_plot_multipleglaciers_equal_subdivisions == 1:
#            
##%% PLOTS USED TO DETERMINE THRESHOLDS FOR DISCARDING POOR DATA
#    #these plots inform the section '%Main Dataset', which removes poor data
## Normalized Elevation vs. Normalized Ice Thickness Change
#glacier_list = list(range(0,len(norm_list)))
#plt.figure(figsize=(10,6))
#plt.subplots_adjust(wspace=0.4, hspace=0.6)
#
#if option_merged_dataset == 1:
#    list_pos = 6
#else:
#    list_pos = 2
#    
#if option_plots_threshold == 1:
#    for glac in glacier_list:  
#        glac_elevbins = ds[glac][list_pos]['bin_center_elev_m']
#        glac_area_t1 = ds[glac][list_pos]['z1_bin_area_valid_km2']
#        glac_area_t2 = ds[glac][list_pos]['z2_bin_area_valid_km2']
#        glac_area_t1_perc = ds[glac][list_pos]['z1_bin_area_perc']
#        glac_bin_count_t1 = ds[glac][list_pos]['z1_bin_count_valid']
#        glac_mb_mwea = ds[glac][list_pos][mb_cn]
#        glac_debristhick_cm = ds[glac][list_pos]['debris_thick_med_m'] * 100
#        glac_debrisperc = ds[glac][list_pos]['perc_debris']
#        glac_pondperc = ds[glac][list_pos]['perc_pond']
#        glac_elevnorm = ds[glac][list_pos]['elev_norm']
#        glac_dhdt_norm_huss = ds[glac][list_pos]['dhdt_norm_huss']
##        glac_dhdt_norm_range = ds[glac][list_pos]['dhdt_norm_range']
#        glac_dhdt_norm_shifted = ds[glac][list_pos]['dhdt_norm_shifted']
#        glac_dhdt_med = ds[glac][list_pos]['dhdt_bin_med_ma']
#        glac_dhdt_mean = ds[glac][list_pos]['dhdt_bin_mean_ma']
#        glac_dhdt_std = ds[glac][list_pos]['dhdt_bin_std_ma']
#        glac_elevs = ds[glac][list_pos]['bin_center_elev_m']
#        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
#        t1 = 2000
#        t2 = 2015
#        glac_name = ds[glac][1].split('-')[1]
#    
#        # ====== Relationship between valid area, mean dhdt and std dhdt =====    
#        plt.subplot(2,3,1)
#        plt.plot(glac_area_t1, glac_dhdt_mean, 'o', mfc='none')
#        plt.xlim(0, 3)
#        plt.xlabel('valid area t1 [km2]')
#        plt.ylabel('mean dhdt [ma]')
#        
#        plt.subplot(2,3,2)
#        plt.plot(glac_area_t1, glac_dhdt_std, 'o', mfc='none')
#        plt.xlim(0,3)
#        plt.xlabel('valid area t1 [km2]')
#        plt.ylabel('std dhdt [ma]')
#        
#        plt.subplot(2,3,3)
#        plt.plot(glac_area_t1, glac_area_t2, 'o', mfc='none')
#        plt.xlabel('valid area t1 [km2]')
#        plt.ylabel('valid area t2 [km2]')
#        
#        plt.subplot(2,3,4)
#        plt.plot(glac_area_t1, glac_dhdt_med, 'o', mfc='none')
#        plt.xlabel('valid area t1 [km2]')
#        plt.ylabel('median dhdt [ma]')
#        
#        plt.subplot(2,3,5)
#        plt.plot(glac_area_t1_perc, glac_dhdt_std, 'o', mfc='none')
#        plt.xlim(0,3)
#        plt.xlabel('perc total area')
#        plt.ylabel('std dhdt [ma]')
#        
#        plt.subplot(2,3,6)
#        plt.plot(glac_dhdt_mean, glac_dhdt_std, 'o', mfc='none')
#        plt.xlim()
#        plt.xlabel('mean dhdt [ma]')
#        plt.ylabel('std dhdt [ma]')
#        
#        if option_show_legend == 1:
#            plt.legend()		
#        plot_fn = 'discard_eval_'  + glac_name
#		
#        if option_savefigs == 1:		
#            plt.savefig(input.output_filepath + 'figures/discard_threshold_plots/' + plot_fn + '.png', bbox_inches='tight')			        
#    
#    plt.show()






# =============================================== OLD CODE/TEXT =======================================================
#%% TIPS
#  - columns in a dataframe can be accessed using df['column_name'] or 
#       df.column_name
#  - .iloc uses column 'positions' to index into a dataframe 
#       (ex. ds_all.iloc[0,0] = 13.00175)
#    while .loc uses column 'names' to index into a dataframe 
#       (ex. ds_all.loc[0, 'reg_glacno'] = 13.00175)
#  - When indexing into lists it is best to use list comprehension.  
#           List comprehension is essentially an efficient
#    for loop for lists and is best read backwards.  For example:
#      A = [binnedcsv_files_all[x] for x in ds.index.values]
#    means that for every value (x) in ds.index.values, select
#        binnedcsv_files_all[x] and add it to the list.
#  - lists also have the ability to store many objects of different forms.  
#       For example it can store individual values,strings, numpy arrays, 
#       pandas series/dataframes, etc.  Therefore, depending on what you are 
#       trying to achieve, you  may want to use a combination of different 
#       indexing methods (ex. list comprehension to access a pandas dataframe,
#    followed by pandas indexing to access an element within the dataframe).
#  - Accessing list of lists: first index is going to access which sublist 
#       you're looking at, while second index is 
#    going to access that element of the list.  For example,
#      ds[0] access the first glacier and ds[0][0] accesses the first element 
#       of the first glacier. in this manner, ds[1][0] accesses the first 
#       element of the second glacier(sublist), ds[1][1] accesses the second  
#    element of the second glacier (sublist), etc. 

#%% Runnings this File - Tips from Kitrea		
#     Before running, make sure that the directory paths are accurate for the user's 
#    computer folders (search_binnedcsv_fn = ....) and (df_glacnames_al[reg_glacno'] = ...)
#	 For each region or combination of regions, this is the suggested flow of how		
#	 to logically run through the code: 		
#	 1. User defines input region		
#	 2. run option_plot_histogram = 1 and option_plot_eachglacier = 1 
#           (keep all run options turned off to 0)		
#	 3. based on the histograms, user defines logical range and step size for
#           setting thresholds of each possible parameter 		
#	       define these as variables, within "input data" section. 
#           Remember, that the list excludes last value so put one extra step		
#	 4. Now, set option_plot_multipleglaciers_single_threshold = 1. 
#           Plug in which parameter you'd like to examine, 
#           by redefining 'parameter = ...' in the Input Data section. Or, run all
#           parameters in one run with option_run_specific_pars = 0. 
#           If you would like to save your figures, make sure to define an output path.
#	 5. Run! 	
#	 6. Now all your png files are stored, and ready to access. 
#    Note- to change the size of the plots (in order to optimize for presentations, etc)
#           change font size and plot size within the code itself.    

#%% OLD MERGED BINS CODE - removed because David Shean now merges the bins to 50 m
    # ===== MERGED BINS ==================================================================
    # NOTE: Bins are now 50 m (11/15/2018)
#    #Merge the bins (continuing within for-loop)
#    bin_start = (binnedcsv['bin_center_elev_m'].min() 
#                 - binnedcsv['bin_center_elev_m'].min() / bin_size % 1 * bin_size)
#    bin_end = (int(binnedcsv['bin_center_elev_m'].max() / bin_size)
#               * bin_size + bin_size / 2)
#    #  do plus/minus bin_size from the center to get the values to merge, 
#    #  determining the middle value of each elevation bin
#    merged_bin_center_elev_m = (np.arange(bin_start + bin_size / 2, bin_end 
#                                          + bin_size / 2, bin_size))
#    
#    merged_cols = binnedcsv.columns.values #new df with same columns as binnedcsv
#    #df filled with nans, with as many rows as merged bins, and as many columns as 
#    # in merged_cols (aka all the binnedcsv columns)
#    ds_merged_bins = (pd.DataFrame(np.full([merged_bin_center_elev_m.shape[0],
#                                            len(merged_cols)], np.nan),
#        columns=merged_cols))
#    ds_merged_bins['bin_center_elev_m'] = merged_bin_center_elev_m 
#    
#    #loop through each merged elevation bin (note, this is all still within
#    #the greater for-loop through binnedcsv files)
#    for nbin in range(merged_bin_center_elev_m.shape[0]): 
#        bin_center_elev = merged_bin_center_elev_m[nbin]
#        #find idx in binnedcsv with elev that will fit into the current 
#        #merge-bin-size interval. (binnedcsv.bin_center_elev_m refers to the
#        # specific glacier's list of all 10m interval bins. bin_center_elev 
#        # refers to currently looped large merge-bin-size middle. so first 
#        bin_idx = binnedcsv.loc[((binnedcsv.bin_center_elev_m > bin_center_elev - bin_size/2) & 
#                                 (binnedcsv.bin_center_elev_m <= bin_center_elev + bin_size/2))].index.values        
#        #for each column, store values at the given indexes that fit into merged bin 
#        bin_counts = binnedcsv.loc[bin_idx, 'z1_bin_count_valid'].values #how many pixels present in that elevation bin
#        bin_dhdt_med = binnedcsv.loc[bin_idx, 'dhdt_bin_med_ma'].astype(float).values
#        bin_dhdt_mean = binnedcsv.loc[bin_idx, 'dhdt_bin_mean_ma'].astype(float).values
#        bin_mb_med_mwea = binnedcsv.loc[bin_idx, 'mb_bin_med_mwea'].astype(float).values
#        bin_mb_mean_mwea = binnedcsv.loc[bin_idx, 'mb_bin_mean_mwea'].astype(float).values
#        bin_debris_med_m = binnedcsv.loc[bin_idx, 'debris_thick_med_m'].values
#        bin_debris_perc = binnedcsv.loc[bin_idx, 'perc_debris'].values
#        bin_pond_perc = binnedcsv.loc[bin_idx, 'perc_pond'].values
#        bin_clean_perc = binnedcsv.loc[bin_idx, 'perc_clean'].astype(float).values
#        # for z1 and z2 raw measures, sum all of the values within each merged 
#        # bin together, add to df
#        ds_merged_bins.loc[nbin, 'z1_bin_count_valid'] = np.nansum(bin_counts)
#        ds_merged_bins.loc[nbin, 'z1_bin_area_valid_km2'] = (
#                np.nansum(binnedcsv.loc[bin_idx, 'z1_bin_area_valid_km2'].values))
#        ds_merged_bins.loc[nbin, 'z1_bin_area_perc'] = (
#                np.nansum(binnedcsv.loc[bin_idx, 'z1_bin_area_perc'].values))
#        ds_merged_bins.loc[nbin, 'z2_bin_count_valid'] = (
#                np.nansum(binnedcsv.loc[bin_idx, 'z2_bin_count_valid'].values))
#        ds_merged_bins.loc[nbin, 'z2_bin_area_valid_km2'] = (
#                np.nansum(binnedcsv.loc[bin_idx, 'z2_bin_area_valid_km2'].values))
#        ds_merged_bins.loc[nbin, 'z2_bin_area_perc'] = (
#                np.nansum(binnedcsv.loc[bin_idx, 'z2_bin_area_perc'].values))
#        #as long as there are valid bins of data, find bin_count-weighted
#        #  average for all other measures, and store into df. This df contains all the
#        #merged bin data. 
#        if np.nansum(bin_counts) > 0:
#            ds_merged_bins.loc[nbin, 'dhdt_bin_med_ma'] = (
#                    np.nansum(bin_counts * bin_dhdt_med) / np.nansum(bin_counts))
#            ds_merged_bins.loc[nbin, 'dhdt_bin_mean_ma'] = (
#                    np.nansum(bin_counts * bin_dhdt_mean) / np.nansum(bin_counts))
#            ds_merged_bins.loc[nbin, 'mb_bin_med_mwea'] = (
#                    np.nansum(bin_counts * bin_mb_med_mwea) / np.nansum(bin_counts))
#            ds_merged_bins.loc[nbin, 'mb_bin_mean_mwea'] = (
#                    np.nansum(bin_counts * bin_mb_mean_mwea) / np.nansum(bin_counts))
#            ds_merged_bins.loc[nbin, 'debris_thick_med_m'] = (
#                    np.nansum(bin_counts * bin_debris_med_m) / np.nansum(bin_counts))
#            ds_merged_bins.loc[nbin, 'perc_debris'] = (
#                    np.nansum(bin_counts * bin_debris_perc) / np.nansum(bin_counts))
#            ds_merged_bins.loc[nbin, 'perc_pond'] = (
#                    np.nansum(bin_counts * bin_pond_perc) / np.nansum(bin_counts))
#            ds_merged_bins.loc[nbin, 'perc_clean'] = (
#                    np.nansum(bin_counts * bin_clean_perc) / np.nansum(bin_counts))
#            
#    # Normalized elevation (MERGED)
#    #  (max elevation - bin elevation) / (max_elevation - min_elevation)
#    ds_merged_bins['elev_norm'] = ((merged_bin_center_elev_m[-1] 
#                                    - merged_bin_center_elev_m) 
#                                    / (merged_bin_center_elev_m[-1] 
#                                    - merged_bin_center_elev_m[0]))
#    # Normalized ice thickness change [ma] (MERGED)
#    #  dhdt / dhdt_max (note the max is found by using np.nanmin of a (-) val)
#    merged_glac_dhdt = ds_merged_bins[dhdt_cn].values
#    # Remove positive elevations and replace with zero (MERGED)
#    merged_glac_dhdt[merged_glac_dhdt >= 0] = 0
#    ds_merged_bins['dhdt_norm_huss'] = merged_glac_dhdt / np.nanmin(merged_glac_dhdt)  
##    ds_merged_bins['dhdt_norm_range'] = merged_glac_dhdt / (np.nanmin(merged_glac_dhdt) - np.nanmax(merged_glac_dhdt))
#    # Shifted normalized ice thickness change such that everything is negative (MERGED)
#    ds_merged_bins['dhdt_norm_shifted'] = ((merged_glac_dhdt
#                                            - np.nanmax(merged_glac_dhdt)) 
#                                            / np.nanmin(merged_glac_dhdt - 
#                                                        np.nanmax(merged_glac_dhdt)))
#            
##    if all(np.isnan(ds_merged_bins['dhdt_norm_huss'].values)) == False:
#    #note: this is now done later in this section
    
#%% Kitrea's try for removing bad values
#remove glaciers where the max dhdt is not in the accumulation zone.
#if the max surface lowering (where norm > 0.75) is within the accumulation zone 
#(where norm_elev <0.5) then discard that glacier. 
#for glac in range(len(ds)): 
#    glacname = ds[glac][1]
#    print('evaluating', glacname, glac)
#    high_dhdt_norm =  (ds[glac][6]['dhdt_norm_huss'] >  0.75)
#    high_dhdt_norm_idx = np.where((ds[glac][6]['dhdt_norm_huss'] >  0.75))
#    accumulation_elev_norm = (ds[glac][6]['elev_norm'] < 0.5)
#    accumulation_elev_norm_idx = np.where(ds[1][6]['elev_norm'] < 0.5)
#    for idx in high_dhdt_norm_idx: 
#        if idx in accumulation_elev_norm_idx[0]: 
#            print('high dhdt found in accumulation zone of ', glacname)
#            removable_idx = removable_idx.union(main_glac_rgi[main_glac_rgi['RGIId'] == glacname].index)
#        else: 
#            print('high accumulation found, but not in accumulation zone of ', glacname)

##do the actual removal, based on the index you just created^
#if (option_remove_outliers == 1 or option_remove_all_pos_dhdt == 1 or 
#    option_remove_surge_glac == 1): 
#    #Drop all indices corresponding to unwanted glaciers, and reset index. 
#    main_glac_rgi.drop(removable_idx, inplace = True)
#    main_glac_rgi.reset_index(drop = True, inplace = True)
#    #change Int64Index to a list of ints, in reverse order
#    reversed_list_removable_idx = sorted(list(removable_idx), reverse = True)
#    #looping through indices, delete them from ds and norm_list 
#    # note, no reset needed bcs index is automatically reset after removal for lists)
#    for removing_idx_int in reversed_list_removable_idx:
#        glac_name = ds[removing_idx_int][1] #pull glacname from ds
#        print('Removing glacier ', glac_name)
#        del ds[removing_idx_int]
#        del norm_list[removing_idx_int]
#    print(len(reversed_list_removable_idx), ' total glaciers removed from ' +
#              'main_glac_rgi, ds, and norm_list')
