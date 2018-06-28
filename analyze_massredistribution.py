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
#    while .iloc uses column 'names' to index into a dataframe (ex. ds_all.loc[0, 'reg_glacno'] = 13.00175)
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
#search_rgiv6_fn = input.main_directory + '\\..\\RGI\\rgi60\\00_rgi60_attribs\\' + '*'

# Option to save
option_savefigs = 1

# Column name for analysis
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

elev_bins = main_glac_hyps.columns.values.astype(int)

#%% MAIN DATASET
# ds is the main dataset for this analysis and is a list of lists (order of glaciers can be found in df_glacnames)
#  Data for each glacier is held in a sublist
#   0 
ds = [[] for x in binnedcsv_files]
for n in range(len(binnedcsv_files)):
    # Process binned geodetic data
    binnedcsv = pd.read_csv(binnedcsv_files[n])
    # Rename columns so they are easier to read
    binnedcsv = binnedcsv.rename(columns=sheancoldict)
    # Remove poor values (ex. debris thickness)
    binnedcsv['debris_thick_med_m'] = binnedcsv['debris_thick_med_m'].astype(float)
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] < 0, 'debris_thick_med_m'] = 0
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] > 5, 'debris_thick_med_m'] = 0
    binnedcsv.loc[binnedcsv['debris_thick_med_m'] == -0, 'debris_thick_med_m'] = 0
    binnedcsv['perc_debris'] = binnedcsv['perc_debris'].astype(float)
    binnedcsv.loc[binnedcsv['perc_debris'] > 100, 'perc_debris'] = 0
    binnedcsv['perc_pond'] = binnedcsv['perc_pond'].astype(float)
    binnedcsv.loc[binnedcsv['perc_pond'] > 100, 'perc_pond'] = 0
    # Normalized elevation
    #  (max elevation - bin elevation) / (max_elevation - min_elevation)
    glac_elev = binnedcsv.bin_center_elev_m.values
    binnedcsv['elev_norm'] = (glac_elev[-1] - glac_elev) / (glac_elev[-1] - glac_elev[0])
    # Normalized ice thickness change [ma]
    #  dhdt / dhdt_max
    glac_dhdt = binnedcsv[dhdt_cn].values
    binnedcsv['dhdt_norm_huss'] = glac_dhdt / glac_dhdt.min()
    binnedcsv['dhdt_norm_range'] = glac_dhdt / (glac_dhdt.min() - glac_dhdt.max())
    glac_dhdt_adj = glac_dhdt.copy()
    glac_dhdt_adj[glac_dhdt_adj > 0] = 0
    binnedcsv['dhdt_norm_adj'] = glac_dhdt_adj / glac_dhdt_adj.min()

    ds[n] = [n, df_glacnames.loc[n, 'RGIId'], binnedcsv, main_glac_rgi.loc[n], main_glac_hyps.loc[n], 
             main_glac_icethickness.loc[n]]

#%% Select glaciers based on a certain parameter
subset_indices = main_glac_rgi[main_glac_rgi['Area'] > 30].index.values
ds = [ds[x] for x in subset_indices]
    
#%% Plots for a single glacier
# Enter the index of the glacier or loop through them all
#glacier_list = [0]
glacier_list = list(range(0,len(ds)))

for glac in glacier_list:
    glac_elevbins = ds[glac][2]['bin_center_elev_m']
    glac_area_t1 = ds[glac][2]['z1_bin_area_valid_km2']
    glac_area_t2 = ds[glac][2]['z2_bin_area_valid_km2']
    glac_mb_mwea = ds[glac][2][mb_cn]
    glac_debristhick_cm = ds[glac][2]['debris_thick_med_m'] * 100
    glac_debrisperc = ds[glac][2]['perc_debris']
    glac_pondperc = ds[glac][2]['perc_pond']
    glac_elevnorm = ds[glac][2]['elev_norm']
    glac_dhdt_norm_huss = ds[glac][2]['dhdt_norm_huss']
    glac_dhdt_norm_range = ds[glac][2]['dhdt_norm_range']
    glac_dhdt_norm_adj = ds[glac][2]['dhdt_norm_adj']
    glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
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
    # Elevation vs. 
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
    
    # Normalized Elevation vs. Normalized Ice Thickness Change
    plt.figure(figsize=(10,3))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # Normalized curves using dhdt max (according to Huss)
    plt.subplot(1,3,1)
    plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label='huss')
    plt.xlabel('Normalized elev range')
    plt.ylabel('Normalized dh/dt [ma]')
    plt.ylim(1,int(glac_dhdt_norm_huss.min()))
    plt.minorticks_on()
    plt.legend()
    # Normalized curves using range of dh/dt
    plt.subplot(1,3,2)
    plt.plot(glac_elevnorm, glac_dhdt_norm_range, label='range')
    plt.ylim(1, -1)
    plt.xlabel('Normalized elev range')
    plt.minorticks_on()
    plt.legend()
    # Normalized curves truncating all positive dh/dt to zero
    plt.subplot(1,3,3)
    plt.plot(glac_elevnorm, glac_dhdt_norm_adj, label='adjusted')
    plt.ylim(1,-0.1)
    plt.xlabel('Normalized elev range')
    plt.minorticks_on()
    plt.legend()
    if option_savefigs == 1:
        plt.savefig(input.output_filepath + 'figures/' + ds[glac][1] + '_normcurves.png', bbox_inches='tight')
    plt.show()
    
#%% Plot multiple glaciers on the same plot
# Normalized Elevation vs. Normalized Ice Thickness Change
plt.figure(figsize=(10,3))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
for glac in glacier_list:  
    glac_elevbins = ds[glac][2]['bin_center_elev_m']
    glac_area_t1 = ds[glac][2]['z1_bin_area_valid_km2']
    glac_area_t2 = ds[glac][2]['z2_bin_area_valid_km2']
    glac_mb_mwea = ds[glac][2][mb_cn]
    glac_debristhick_cm = ds[glac][2]['debris_thick_med_m'] * 100
    glac_debrisperc = ds[glac][2]['perc_debris']
    glac_pondperc = ds[glac][2]['perc_pond']
    glac_elevnorm = ds[glac][2]['elev_norm']
    glac_dhdt_norm_huss = ds[glac][2]['dhdt_norm_huss']
    glac_dhdt_norm_range = ds[glac][2]['dhdt_norm_range']
    glac_dhdt_norm_adj = ds[glac][2]['dhdt_norm_adj']
    glacwide_mb_mwea = (glac_area_t1 * glac_mb_mwea).sum() / glac_area_t1.sum()
    t1 = 2000
    t2 = 2015
    # Normalized curves using dhdt max (according to Huss)
    plt.subplot(1,3,1)
    plt.plot(glac_elevnorm, glac_dhdt_norm_huss, label='huss')
    plt.xlabel('Normalized elev range')
    plt.ylabel('Normalized dh/dt [ma]')
    plt.ylim(1,int(glac_dhdt_norm_huss.min()))
    plt.minorticks_on()
    plt.legend()
    # Normalized curves using range of dh/dt
    plt.subplot(1,3,2)
    plt.plot(glac_elevnorm, glac_dhdt_norm_range, label='range')
    plt.ylim(1, -1)
    plt.xlabel('Normalized elev range')
    plt.minorticks_on()
    plt.legend()
    # Normalized curves truncating all positive dh/dt to zero
    plt.subplot(1,3,3)
    plt.plot(glac_elevnorm, glac_dhdt_norm_adj, label='adjusted')
    plt.ylim(1,-0.1)
    plt.xlabel('Normalized elev range')
    plt.minorticks_on()
    plt.legend()
plt.show()
    
    