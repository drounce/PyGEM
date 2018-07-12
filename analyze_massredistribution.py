"""
Mass redistribution analysis
"""
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import pygem_input as input
import pygemfxns_modelsetup as modelsetup

#%% TIPS
#  - columns in a dataframe can be accessed using df['column_name'] or df.column_name
#  - .iloc uses column 'positions' to index into a dataframe (ex. ds_all.iloc[0,0] = 13.00175)
#    while .loc uses column 'names' to index into a dataframe (ex. ds_all.loc[0, 'reg_glacno'] = 13.00175)
#  - When indexing into lists it is best to use list comprehension.  List comprehension is essentially an efficient
#    for loop for lists and is best read backwards.  For example:
#      A = [binnedcsv_files_all[x] for x in ds.index.values]
#    means that for every value (x) in ds.index.values, select binnedcsv_files_all[x] and add it to the list.
#  - lists also have the ability to store many objects of different forms.  For example it can store individual values,
#    strings, numpy arrays, pandas series/dataframes, etc.  Therefore, depending on what you are trying to achieve, you 
#    may want to use a combination of different indexing methods (ex. list comprehension to access a pandas dataframe,
#    followed by pandas indexing to access an element within the dataframe).
#  - Accessing list of lists: first index is going to access which sublist you're looking at, while second index is 
#    going to access that element of the list.  For example,
#      ds[0] access the first glacier and ds[0][0] accesses the first element of the first glacier
#    in this manner, ds[1][0] accesses the first element of the second glacier(sublist), ds[1][1] accesses the second  
#    element of the second glacier (sublist), etc. 

#%% Input data
#rgi_regionO1 = [13, 14, 15]
rgi_regionO1 = [15]
search_binnedcsv_fn = input.main_directory + '\\..\\DEMs\\mb_bins_sample_20180323\\*_mb_bins.csv'
#search_binnedcsv_fn = ('/Users/kitreatakataglushkoff/Documents/All_Documents/SUMMER_2018/Glaciology/HiMAT/DEMs/' + 
#                       'mb_bins_sample_20180323/*_mb_bins.csv')
#set default parameter based on all glaciers of defined region 
prmtr = 'region_' + str(rgi_regionO1) + '_all_glac' 

# Option to make individual glacier plots
option_plot_eachglacier = 0
option_plot_multipleglaciers = 1
option_plots_threshold = 0
# Option to save
option_savefigs = 1

# Merged dataset options
option_merged_dataset = 1
bin_size = 100

# Column name for analysis (note: pixel size is 8 m)
mb_cn = 'mb_bin_med_mwea'
dhdt_cn = 'dhdt_bin_med_ma'

# binned csv column name convsersion dictionary
#  change column names so they are easier to work with (remove spaces, etc.)
sheancoldict = {'# bin_center_elev_m': 'bin_center_elev_m',
                ' z1_bin_count_valid': 'z1_bin_count_valid',
                ' z1_bin_area_valid_km2': 'z1_bin_area_valid_km2',
                ' z1_bin_area_perc': 'z1_bin_area_perc',
                ' z2_bin_count_valid': 'z2_bin_count_valid',
                ' z2_bin_area_valid_km2': 'z2_bin_area_valid_km2',
                ' z2_bin_area_perc': 'z2_bin_area_perc',
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
                ' perc_clean': 'perc_clean'}

#%% Select files
# Find files for analysis
binnedcsv_files_all = glob.glob(search_binnedcsv_fn)
#rgi_files = glob.glob(search_rgiv6_fn)

# RGIId's of available glaciers
#  note: 
df_glacnames_all = pd.DataFrame()
df_glacnames_all['reg_glacno'] = [x.split('\\')[-1].split('_')[0] for x in binnedcsv_files_all]
#df_glacnames_all['reg_glacno'] = [x.split('/')[-1].split('_')[0] for x in binnedcsv_files_all]
df_glacnames_all['RGIId'] = 'RGI60-' + df_glacnames_all['reg_glacno'] 
df_glacnames_all['region'] = df_glacnames_all.reg_glacno.astype(float).astype(int)
df_glacnames_all['glacno_str'] = df_glacnames_all.reg_glacno.str.split('.').apply(lambda x: x[1])
df_glacnames_all['glacno'] = df_glacnames_all.reg_glacno.str.split('.').apply(lambda x: x[1]).astype(int)
# Drop glaciers that are not in correct region
df_glacnames = df_glacnames_all[df_glacnames_all.region.isin(rgi_regionO1) == True]
binnedcsv_files = [binnedcsv_files_all[x] for x in df_glacnames.index.values] 
df_glacnames.reset_index(drop=True, inplace=True)
# create a dictionary between index and glacno
glacidx_dict = dict(zip(df_glacnames['reg_glacno'], df_glacnames.index.values))

main_glac_rgi = pd.DataFrame()
main_glac_hyps = pd.DataFrame()
main_glac_icethickness = pd.DataFrame()
for n, region in enumerate(rgi_regionO1):
    print('Region', region)
    df_glacnames_reg = df_glacnames[df_glacnames.region == region]
    rgi_glac_number = df_glacnames_reg['glacno_str'].tolist()
    # This if statement avoids errors associated with regions that have no glaciers
    if len(rgi_glac_number) > 0: 
        main_glac_rgi_reg = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2 = 'all', 
                                                              rgi_glac_number=rgi_glac_number)
        # Glacier hypsometry [km**2], total area
        main_glac_hyps_reg = modelsetup.import_Husstable(main_glac_rgi_reg, [region], input.hyps_filepath, 
                                                         input.hyps_filedict, input.hyps_colsdrop)
        # Ice thickness [m], average
        main_glac_icethickness_reg = modelsetup.import_Husstable(main_glac_rgi_reg, [region], input.thickness_filepath, 
                                                                 input.thickness_filedict, input.thickness_colsdrop)
        main_glac_hyps_reg[main_glac_icethickness_reg == 0] = 0

        # concatenate regions
        main_glac_rgi = main_glac_rgi.append(main_glac_rgi_reg, ignore_index=True)
        main_glac_hyps = main_glac_hyps.append(main_glac_hyps_reg, ignore_index=True)
        main_glac_icethickness = main_glac_icethickness.append(main_glac_icethickness_reg, ignore_index=True)

#%% MAIN DATASET
# add an empty column to main_glac_rgi to be filled with glac-wide debris perc
#main_glac_rgi['glacwide_debris'] = ''
# ds is the main dataset for this analysis and is a list of lists (order of glaciers can be found in df_glacnames)
#  Data for each glacier is held in a sublist
ds = [[] for x in binnedcsv_files]
for n in range(len(binnedcsv_files)):
    # Process binned geodetic data
    binnedcsv = pd.read_csv(binnedcsv_files[n])
    # Rename columns so they are easier to read
    binnedcsv = binnedcsv.rename(columns=sheancoldict)
    # Remove poor values
    # Surface lowering (use dhdt mean and standard deviation)
    binnedcsv.loc[binnedcsv['dhdt_bin_mean_ma'].astype(float) > 2, binnedcsv.columns.tolist()[1:]] = np.nan
    binnedcsv.loc[binnedcsv['dhdt_bin_mean_ma'].astype(float) < -3, binnedcsv.columns.tolist()[1:]] = np.nan
    binnedcsv.loc[binnedcsv['dhdt_bin_std_ma'].astype(float) > 3, binnedcsv.columns.tolist()[1:]] = np.nan
    # Debris thickness
    binnedcsv['debris_thick_med_m'] = binnedcsv['debris_thick_med_m'].astype(float)
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] < 0, 'debris_thick_med_m'] = 0
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] > 5, 'debris_thick_med_m'] = 0
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] == -0, 'debris_thick_med_m'] = 0
    binnedcsv['perc_debris'] = binnedcsv['perc_debris'].astype(float)
    binnedcsv.loc[binnedcsv['perc_debris'] > 100, 'perc_debris'] = 0
    binnedcsv.loc[binnedcsv['perc_debris'] <= 0, 'perc_debris'] = 0
    # Supraglacial ponds
    binnedcsv['perc_pond'] = binnedcsv['perc_pond'].astype(float)
    binnedcsv.loc[binnedcsv['perc_pond'] > 100, 'perc_pond'] = 0
    binnedcsv.loc[binnedcsv['perc_pond'] <= 0, 'perc_pond'] = 0
    # Clean ice
    binnedcsv['perc_clean'] = binnedcsv['perc_clean'].astype(float)
    binnedcsv.loc[binnedcsv['perc_clean'] > 100, 'perc_clean'] = 0
    binnedcsv.loc[binnedcsv['perc_clean'] <= 0, 'perc_clean'] = 0
    
    # Find glacier-wide debris perc for each glacier, and add to main_glac_rgi
    glacwide_debris = float((binnedcsv['z1_bin_area_valid_km2']*binnedcsv['perc_debris']).sum() 
                            / binnedcsv['z1_bin_area_valid_km2'].sum())
    main_glac_rgi.loc[n, 'PercDebris'] = glacwide_debris
    
    # Normalized elevation
    #  (max elevation - bin elevation) / (max_elevation - min_elevation)
    glac_elev = binnedcsv.bin_center_elev_m.values
    binnedcsv['elev_norm'] = (glac_elev[-1] - glac_elev) / (glac_elev[-1] - glac_elev[0])
    # Normalized ice thickness change [ma]
    #  dhdt / dhdt_max
    glac_dhdt = binnedcsv[dhdt_cn].values
    binnedcsv['dhdt_norm_huss'] = glac_dhdt / np.nanmin(glac_dhdt)
    binnedcsv['dhdt_norm_range'] = glac_dhdt / (np.nanmin(glac_dhdt) - np.nanmax(glac_dhdt))
    # Shifted normalized ice thickness change such that everything is negative
    binnedcsv['dhdt_norm_shifted'] = (glac_dhdt - np.nanmax(glac_dhdt)) / np.nanmin(glac_dhdt - np.nanmax(glac_dhdt)) 

    # Merge the bins
    bin_start = binnedcsv['bin_center_elev_m'].min() - binnedcsv['bin_center_elev_m'].min() / bin_size % 1 * bin_size
    bin_end = int(binnedcsv['bin_center_elev_m'].max() / bin_size) * bin_size + bin_size / 2
    merged_bin_center_elev_m = np.arange(bin_start + bin_size / 2, bin_end + bin_size / 2, bin_size)
    #  do plus/minus bin_size from the center to get the values to merge
    
    merged_cols = binnedcsv.columns.values
    ds_merged_bins = pd.DataFrame(np.full([merged_bin_center_elev_m.shape[0], len(merged_cols)], np.nan), 
                                  columns=merged_cols)
    ds_merged_bins['bin_center_elev_m'] = merged_bin_center_elev_m
    for nbin in range(merged_bin_center_elev_m.shape[0]):
        bin_center_elev = merged_bin_center_elev_m[nbin]
        bin_idx = binnedcsv.loc[((binnedcsv.bin_center_elev_m > bin_center_elev - bin_size/2) & 
                                 (binnedcsv.bin_center_elev_m <= bin_center_elev + bin_size/2))].index.values
        bin_counts = binnedcsv.loc[bin_idx, 'z1_bin_count_valid'].values
        bin_dhdt_med = binnedcsv.loc[bin_idx, 'dhdt_bin_med_ma'].values
        bin_dhdt_mean = binnedcsv.loc[bin_idx, 'dhdt_bin_mean_ma'].astype(float).values
        bin_mb_med_mwea = binnedcsv.loc[bin_idx, 'mb_bin_med_mwea'].values
        bin_mb_mean_mwea = binnedcsv.loc[bin_idx, 'mb_bin_mean_mwea'].astype(float).values
        bin_debris_med_m = binnedcsv.loc[bin_idx, 'debris_thick_med_m'].values
        bin_debris_perc = binnedcsv.loc[bin_idx, 'perc_debris'].values
        bin_pond_perc = binnedcsv.loc[bin_idx, 'perc_pond'].values
        bin_clean_perc = binnedcsv.loc[bin_idx, 'perc_clean'].astype(float).values
        ds_merged_bins.loc[nbin, 'z1_bin_count_valid'] = np.nansum(bin_counts)
        ds_merged_bins.loc[nbin, 'z1_bin_area_valid_km2'] = (
                np.nansum(binnedcsv.loc[bin_idx, 'z1_bin_area_valid_km2'].values))
        ds_merged_bins.loc[nbin, 'z1_bin_area_perc'] = np.nansum(binnedcsv.loc[bin_idx, 'z1_bin_area_perc'].values)
        ds_merged_bins.loc[nbin, 'z2_bin_count_valid'] = np.nansum(binnedcsv.loc[bin_idx, 'z2_bin_count_valid'].values)
        ds_merged_bins.loc[nbin, 'z2_bin_area_valid_km2'] = (
                np.nansum(binnedcsv.loc[bin_idx, 'z2_bin_area_valid_km2'].values))
        ds_merged_bins.loc[nbin, 'z2_bin_area_perc'] = np.nansum(binnedcsv.loc[bin_idx, 'z2_bin_area_perc'].values)
        if np.nansum(bin_counts) > 0:
            ds_merged_bins.loc[nbin, 'dhdt_bin_med_ma'] = np.nansum(bin_counts * bin_dhdt_med) / np.nansum(bin_counts)
            ds_merged_bins.loc[nbin, 'dhdt_bin_mean_ma'] = np.nansum(bin_counts * bin_dhdt_mean) / np.nansum(bin_counts) 
            ds_merged_bins.loc[nbin, 'mb_bin_med_mwea'] = (
                    np.nansum(bin_counts * bin_mb_med_mwea) / np.nansum(bin_counts))
            ds_merged_bins.loc[nbin, 'mb_bin_mean_mwea'] = (
                    np.nansum(bin_counts * bin_mb_mean_mwea) / np.nansum(bin_counts))
            ds_merged_bins.loc[nbin, 'debris_thick_med_m'] = (
                    np.nansum(bin_counts * bin_debris_med_m) / np.nansum(bin_counts))
            ds_merged_bins.loc[nbin, 'perc_debris'] = np.nansum(bin_counts * bin_debris_perc) / np.nansum(bin_counts)
            ds_merged_bins.loc[nbin, 'perc_pond'] = np.nansum(bin_counts * bin_pond_perc) / np.nansum(bin_counts)
            ds_merged_bins.loc[nbin, 'perc_clean'] = np.nansum(bin_counts * bin_clean_perc) / np.nansum(bin_counts)
        
    # Normalized elevation
    #  (max elevation - bin elevation) / (max_elevation - min_elevation)
    merged_bin_center_elev_m
    ds_merged_bins['elev_norm'] = ((merged_bin_center_elev_m[-1] - merged_bin_center_elev_m) / 
                                   (merged_bin_center_elev_m[-1] - merged_bin_center_elev_m[0]))
    # Normalized ice thickness change [ma]
    #  dhdt / dhdt_max
    merged_glac_dhdt = ds_merged_bins[dhdt_cn].values
    ds_merged_bins['dhdt_norm_huss'] = merged_glac_dhdt / np.nanmin(merged_glac_dhdt)
    ds_merged_bins['dhdt_norm_range'] = merged_glac_dhdt / (np.nanmin(merged_glac_dhdt) - np.nanmax(merged_glac_dhdt))
    # Shifted normalized ice thickness change such that everything is negative
    ds_merged_bins['dhdt_norm_shifted'] = ((merged_glac_dhdt - np.nanmax(merged_glac_dhdt)) / 
                                           np.nanmin(merged_glac_dhdt - np.nanmax(merged_glac_dhdt)))

    ds[n] = [n, df_glacnames.loc[n, 'RGIId'], binnedcsv, main_glac_rgi.loc[n], main_glac_hyps.loc[n], 
             main_glac_icethickness.loc[n], ds_merged_bins]
    
#%% Select glaciers based on a certain parameter
##define the range to examine
#bin_high = 1000
#bin_low = 0
#
##note: possible main_glac_rgi_  testing_var str: CenLon, CenLat, Area, Zmin, Zmax, 
##                        Zmed, Slope, Aspect, Lmax, Form, TermType, PercDebris
#testing_var = 'Slope'
#
##(*kitrea currently working on this area, and has commented out things that 
## would interfere with David's elevation bin-size changes )
#
##Lower 20% of variable range 
#
#
##Upper 20% of variable range
##upper_bin_high = np.ceil((main_glac_rgi[testing_var].max()) #the upper int of the max
##lower_bin_high = 
#
#print('Range of '+ testing_var+ ': (' + str(main_glac_rgi[testing_var].min()) + ', ' + 
#                                    str(main_glac_rgi[testing_var].max()) + ')')
#
##redefine the dataframe to only include glaciers with desired parameter
#subset_indices = main_glac_rgi[main_glac_rgi[testing_var].between(bin_low,bin_high)].index.values
#ds = [ds[x] for x in subset_indices]
#prmtr = 'Reg_' + str(rgi_regionO1)+ '_' + str(bin_low) + '<' + testing_var + '<' + str(bin_high)
#print('These glaciers are based on the parameter, ', prmtr)


#%% Plots for a single glacier    
def plot_eachglacier(glacier_list, option_merged_dataset=0):
    # Set position of dataset to plot in list based on using merged or unmerged data
    #  [2 = 10m, 6 = merged]
    if option_merged_dataset == 0:
        ds_position = 2
    elif option_merged_dataset == 1:
        ds_position = 6
        
    # Loop through glacier list and plot
    for glac in glacier_list:
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
#        glac_dhdt_norm_range = ds[glac][ds_position]['dhdt_norm_range']
        glac_dhdt_norm_shifted = ds[glac][ds_position]['dhdt_norm_shifted']
        glac_elevs = ds[glac][ds_position]['bin_center_elev_m']
        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum() 
        #check if this is same way Shean did it. Also, does it make more sense for ths to be in the above, 
        # adjustment section?
        t1 = 2000
        t2 = 2015
        
        # Plot Elevation bins vs. Area, Mass balance, and Debris thickness/pond coverage/ debris coverage
        plt.figure(figsize=(10,6))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.suptitle(ds[glac][1], y=0.94)
        # Elevation vs. Area
        plt.subplot(1,3,1)
        plt.plot(glac_area_t1, glac_elevbins, label=t1)
        plt.plot(glac_area_t2, glac_elevbins, label=t2)
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
            plt.savefig(input.output_filepath + 'figures/' + ds[glac][1] + '_mb_aed.png', bbox_inches='tight')
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
            plt.savefig(input.output_filepath + 'figures/' + ds[glac][1] + '_normcurves.png', bbox_inches='tight')
        plt.show()
    
#%% Plots for a single glacier    
def plot_multipleglaciers(glacier_list, option_merged_dataset, parameter='Area', threshold=0):
    # Set position of dataset to plot in list based on using merged or unmerged data
    #  [2 = 10m, 6 = merged]
    if option_merged_dataset == 0:
        ds_position = 2
    elif option_merged_dataset == 1:
        ds_position = 6
        
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    count = 0
    for glac in glacier_list:  
        glac_rgi = ds[glac][3]
        glac_elevbins = ds[glac][ds_position]['bin_center_elev_m']
        glac_area_t1 = ds[glac][ds_position]['z1_bin_area_valid_km2']
        glac_area_t2 = ds[glac][ds_position]['z2_bin_area_valid_km2']
        glac_area_t1_perc = ds[glac][ds_position]['z1_bin_area_perc']
        glac_bin_count_t1 = ds[glac][ds_position]['z1_bin_count_valid']
        #  THIS IS THE PERCENT OF THE TOTAL AREA
        glac_mb_mwea = ds[glac][ds_position][mb_cn]
        glac_debristhick_cm = ds[glac][ds_position]['debris_thick_med_m'] * 100
        glac_debrisperc = ds[glac][ds_position]['perc_debris']
        glac_pondperc = ds[glac][ds_position]['perc_pond']
        glac_elevnorm = ds[glac][ds_position]['elev_norm']
        glac_dhdt_norm_huss = ds[glac][ds_position]['dhdt_norm_huss']
        glac_dhdt_norm_range = ds[glac][ds_position]['dhdt_norm_range']
        glac_dhdt_norm_shifted = ds[glac][ds_position]['dhdt_norm_shifted']
        glac_dhdt_med = ds[glac][ds_position]['dhdt_bin_med_ma']
        glac_dhdt_mean = ds[glac][ds_position]['dhdt_bin_mean_ma']
        glac_dhdt_std = ds[glac][ds_position]['dhdt_bin_std_ma']
        glac_elevs = ds[glac][ds_position]['bin_center_elev_m']
        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
        t1 = 2000
        t2 = 2015
        glac_name = ds[glac][1].split('-')[1]
        
        # Subset parameters based on column name and threshold
        if glac_rgi[parameter] >= threshold:
            count = count + 1
            # dhdt vs. elevation
            plt.subplot(2,3,1)
            plt.plot(glac_elevs, glac_dhdt_med, label=glac_name)
            if count == 1:
                plt.gca().invert_xaxis()
            plt.xlabel('Elevation range')
            plt.ylabel('dh/dt [m/a]')
            plt.minorticks_on()    
            # No Normalization curves using range of dh/dt
            plt.subplot(2,3,2)
            plt.plot(glac_elevnorm, glac_dhdt_med, label=glac_name)
            plt.ylim(glac_dhdt_med.min(), glac_dhdt_med.max())
            plt.xlabel('Normalized elev range')
            plt.ylabel('dh/dt [m/a]')
            plt.minorticks_on()    
            # Normalized curves using dhdt max (according to Huss)
            plt.subplot(2,3,4)
            plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label=glac_name)
            plt.xlabel('Normalized elev range')
            plt.ylabel('Normalized dh/dt')
            plt.ylim(1,float(glac_dhdt_norm_huss.min())) #changed from int to float
            plt.title('Huss Normalization')
            plt.minorticks_on()
            # Normalized curves using range of dh/dt
            plt.subplot(2,3,5)
            plt.plot(glac_elevnorm, glac_dhdt_norm_range, label=glac_name)
            plt.ylim(1, -1)
            plt.xlabel('Normalized elev range')
            plt.title('Range Normalization')
            plt.minorticks_on()
            # Normalized curves truncating all positive dh/dt to zero
            plt.subplot(2,3,6)
            plt.plot(glac_elevnorm, glac_dhdt_norm_shifted, label=glac_name)
            plt.ylim(1,0)
            plt.xlabel('Normalized elev range')
            plt.title('Shifted Normalization')
            plt.minorticks_on()
        #    plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
    # Add title to subplot
    plot_fn = 'normcurves_' + parameter + '_' + str(threshold) + '_gt' 
    plt.suptitle(plot_fn + ' (' + str(count) + ' glaciers)')
    # Save and show figure
    if option_savefigs == 1:
        plt.savefig(input.output_filepath + 'figures/normalized_curves/' + plot_fn + '.png', bbox_inches='tight')
    plt.show()
    
    # PLOT FIGURE OF THE OPPOSITE THRESHOLD AS WELL
    plt.figure(figsize=(10,6))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    count = 0
    for glac in glacier_list:  
        glac_rgi = ds[glac][3]
        glac_elevbins = ds[glac][ds_position]['bin_center_elev_m']
        glac_area_t1 = ds[glac][ds_position]['z1_bin_area_valid_km2']
        glac_area_t2 = ds[glac][ds_position]['z2_bin_area_valid_km2']
        glac_area_t1_perc = ds[glac][ds_position]['z1_bin_area_perc']
        glac_bin_count_t1 = ds[glac][ds_position]['z1_bin_count_valid']
        #  THIS IS THE PERCENT OF THE TOTAL AREA
        glac_mb_mwea = ds[glac][ds_position][mb_cn]
        glac_debristhick_cm = ds[glac][ds_position]['debris_thick_med_m'] * 100
        glac_debrisperc = ds[glac][ds_position]['perc_debris']
        glac_pondperc = ds[glac][ds_position]['perc_pond']
        glac_elevnorm = ds[glac][ds_position]['elev_norm']
        glac_dhdt_norm_huss = ds[glac][ds_position]['dhdt_norm_huss']
        glac_dhdt_norm_range = ds[glac][ds_position]['dhdt_norm_range']
        glac_dhdt_norm_shifted = ds[glac][ds_position]['dhdt_norm_shifted']
        glac_dhdt_med = ds[glac][ds_position]['dhdt_bin_med_ma']
        glac_dhdt_mean = ds[glac][ds_position]['dhdt_bin_mean_ma']
        glac_dhdt_std = ds[glac][ds_position]['dhdt_bin_std_ma']
        glac_elevs = ds[glac][ds_position]['bin_center_elev_m']
        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
        t1 = 2000
        t2 = 2015
        glac_name = ds[glac][1].split('-')[1]
        
        # Subset parameters based on column name and threshold
        if glac_rgi[parameter] < threshold:
            count = count + 1
            # dhdt vs. elevation
            plt.subplot(2,3,1)
            plt.plot(glac_elevs, glac_dhdt_med, label=glac_name)
            if count == 1:
                plt.gca().invert_xaxis()
            plt.xlabel('Elevation range')
            plt.ylabel('dh/dt [m/a]')
            plt.minorticks_on()    
            # No Normalization curves using range of dh/dt
            plt.subplot(2,3,2)
            plt.plot(glac_elevnorm, glac_dhdt_med, label=glac_name)
            plt.ylim(glac_dhdt_med.min(), glac_dhdt_med.max())
            plt.xlabel('Normalized elev range')
            plt.ylabel('dh/dt [m/a]')
            plt.minorticks_on()    
            # Normalized curves using dhdt max (according to Huss)
            plt.subplot(2,3,4)
            plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label=glac_name)
            plt.xlabel('Normalized elev range')
            plt.ylabel('Normalized dh/dt')
            plt.ylim(1,float(glac_dhdt_norm_huss.min())) #changed from int to float
            plt.title('Huss Normalization')
            plt.minorticks_on()
            # Normalized curves using range of dh/dt
            plt.subplot(2,3,5)
            plt.plot(glac_elevnorm, glac_dhdt_norm_range, label=glac_name)
            plt.ylim(1, -1)
            plt.xlabel('Normalized elev range')
            plt.title('Range Normalization')
            plt.minorticks_on()
            # Normalized curves truncating all positive dh/dt to zero
            plt.subplot(2,3,6)
            plt.plot(glac_elevnorm, glac_dhdt_norm_shifted, label=glac_name)
            plt.ylim(1,0)
            plt.xlabel('Normalized elev range')
            plt.title('Shifted Normalization')
            plt.minorticks_on()
        #    plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
        
    # Add title to subplot
    plot_fn = 'normcurves_' + parameter + '_' + str(threshold) + '_lt' 
    plt.suptitle(plot_fn + ' (' + str(count) + ' glaciers)')
    # Save and show figure
    if option_savefigs == 1:
        plt.savefig(input.output_filepath + 'figures/normalized_curves/' + plot_fn + '.png', bbox_inches='tight')
    plt.show()
  
#%% PLOT OPTIONS
# Index of glaciers to loop through
glacier_list = list(range(0,len(ds)))
parameter = 'Area'
thresholds = [10, 15, 20, 25, 30]

if option_plot_eachglacier == 1:
    plot_eachglacier(glacier_list, option_merged_dataset=1)
    
if option_plot_multipleglaciers == 1:
    for n in thresholds:
        plot_multipleglaciers(glacier_list, option_merged_dataset=1, parameter=parameter, threshold=n)
    

#%% PLOTS USED TO DETERMINE THRESHOLDS FOR DISCARDING POOR DATA
# Normalized Elevation vs. Normalized Ice Thickness Change
glacier_list = list(range(0,len(ds)))
plt.figure(figsize=(10,6))
plt.subplots_adjust(wspace=0.4, hspace=0.6)

if option_merged_dataset == 1:
    list_pos = 6
else:
    list_pos = 2
    
if option_plots_threshold == 1:
    for glac in glacier_list:  
        glac_elevbins = ds[glac][list_pos]['bin_center_elev_m']
        glac_area_t1 = ds[glac][list_pos]['z1_bin_area_valid_km2']
        glac_area_t2 = ds[glac][list_pos]['z2_bin_area_valid_km2']
        glac_area_t1_perc = ds[glac][list_pos]['z1_bin_area_perc']
        glac_bin_count_t1 = ds[glac][list_pos]['z1_bin_count_valid']
        #  THIS IS THE PERCENT OF THE TOTAL AREA
        glac_mb_mwea = ds[glac][list_pos][mb_cn]
        glac_debristhick_cm = ds[glac][list_pos]['debris_thick_med_m'] * 100
        glac_debrisperc = ds[glac][list_pos]['perc_debris']
        glac_pondperc = ds[glac][list_pos]['perc_pond']
        glac_elevnorm = ds[glac][list_pos]['elev_norm']
        glac_dhdt_norm_huss = ds[glac][list_pos]['dhdt_norm_huss']
        glac_dhdt_norm_range = ds[glac][list_pos]['dhdt_norm_range']
        glac_dhdt_norm_shifted = ds[glac][list_pos]['dhdt_norm_shifted']
        glac_dhdt_med = ds[glac][list_pos]['dhdt_bin_med_ma']
        glac_dhdt_mean = ds[glac][list_pos]['dhdt_bin_mean_ma']
        glac_dhdt_std = ds[glac][list_pos]['dhdt_bin_std_ma']
        glac_elevs = ds[glac][list_pos]['bin_center_elev_m']
        glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
        t1 = 2000
        t2 = 2015
        glac_name = ds[glac][1].split('-')[1]
    
        # ====== Relationship between valid area, mean dhdt and std dhdt =====    
        plt.subplot(2,3,1)
        plt.plot(glac_area_t1, glac_dhdt_mean, 'o', mfc='none')
        plt.xlim(0, 3)
        plt.xlabel('valid area t1 [km2]')
        plt.ylabel('mean dhdt [ma]')
        
        plt.subplot(2,3,2)
        plt.plot(glac_area_t1, glac_dhdt_std, 'o', mfc='none')
        plt.xlim(0,3)
        plt.xlabel('valid area t1 [km2]')
        plt.ylabel('std dhdt [ma]')
        
        plt.subplot(2,3,3)
        plt.plot(glac_area_t1, glac_area_t2, 'o', mfc='none')
        plt.xlabel('valid area t1 [km2]')
        plt.ylabel('valid area t2 [km2]')
        
        plt.subplot(2,3,4)
        plt.plot(glac_area_t1, glac_dhdt_med, 'o', mfc='none')
        plt.xlabel('valid area t1 [km2]')
        plt.ylabel('median dhdt [ma]')
        
        plt.subplot(2,3,5)
        plt.plot(glac_area_t1_perc, glac_dhdt_std, 'o', mfc='none')
        plt.xlim(0,3)
        plt.xlabel('perc total area')
        plt.ylabel('std dhdt [ma]')
        
        plt.subplot(2,3,6)
        plt.plot(glac_dhdt_mean, glac_dhdt_std, 'o', mfc='none')
        plt.xlim()
        plt.xlabel('mean dhdt [ma]')
        plt.ylabel('std dhdt [ma]')
    
    plt.show()