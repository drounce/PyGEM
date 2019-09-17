"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

# Built-in libraries
import os
import argparse
# External libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import pygemfxns_modelsetup as modelsetup

print('\ndhdt analysis performed separately using shean_mb_parallel.py\n')

# ===== INPUT DATA =====
hyps_fn = ('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/IceThickness_Farinotti/output/' +
           'area_km2_01_Farinotti2019_10m.csv')

dems_output_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/DEMs/Berthier/output/'
mb_summary_fn_list = ['AK_Pen_mb_20190912_2256.csv', 'AR_C_mb_20190913_0735.csv', 'AR_E_mb_20190913_0735.csv',
                      'AR_W_mb_20190913_0835.csv', 'Chugach_mb_20190913_0744.csv', 'Coast_mb_20190912_2308.csv',
                      'Kenai_mb_20190912_2301.csv', 'StElias_mb_20190913_0836.csv']
mb_summary_fn = 'AK_all_20190913.csv'
mb_mwea_all_fn = 'AK_all_20190913_wextrapolations.csv'

binned_fp = dems_output_fp + 'csv/'
fig_fp = dems_output_fp + 'figures/all/'

if os.path.exists(fig_fp) == False:
    os.makedirs(fig_fp)

valid_perc_threshold = 90
min_area_km2 = 5
mb_cn = 'mb_bin_med_mwea'
mb_max = 2.5
mb_min = -5
option_normelev = 'huss'        # Terminus = 1, Top = 0
#option_normelev = 'larsen'     # Terminus = 0, Top = 1

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

def norm_stats(norm_list, option_normelev=option_normelev, option_norm_limits=False):
    """
    Statistics associated with normalized elevation data

    Parameters
    ----------
    norm_list : list of np.array
        each item is a np.array (col 1: normalized elevation, col 2: mb, dhdt, normalized mb, or normalized dhdt)
    option_norm_limits : boolean
        option to place limits on the normalized dh/dt of 0 and 1

    Returns
    -------
    norm_all_stats : pd.DataFrame
        statistics associated with the normalized values
    """
    # Merge norm_list to make array of all glaciers with same elevation normalization space
#    max_length = len(max(norm_list,key=len)) #len of glac w most norm values
#    norm_all = np.zeros((max_length, len(norm_list)+1)) #array: each col a glac, each row a norm dhdt val to be interpolated
#    # First column is normalized elevation, pulled from the glac with most norm vals
#    norm_all[:,0] = max(norm_list,key=len)[:,0]

    # Interpolate to common normalized elevation for all glaciers
    norm_elev = np.arange(0,1.01,0.01)
    norm_all = np.zeros((len(norm_elev), len(norm_list)+1)) #array: each col a glac, each row norm dhdt val interpolated
    norm_all[:,0] = norm_elev

    # Loop through each glacier's normalized array (where col1 is elev_norm and col2 is mb or dhdt)
    for n, norm_single in enumerate(norm_list):

        if option_normelev == 'huss':
            norm_single = norm_single[::-1]

        # Fill in nan values for elev_norm of 0 and 1 with nearest neighbor
        nonan_idx = np.where(~np.isnan(norm_single[:,1]))[0]
        norm_single[0,1] = norm_single[nonan_idx[0], 1]
        norm_single[-1,1] = norm_single[nonan_idx[-1], 1]
        # Remove nan values.
        norm_single = norm_single[nonan_idx]
        elev_single = norm_single[:,0] #set name for first col of a given glac
        dhdt_single = norm_single[:,1] #set name for the second col of a given glac
        #loop through each dhdt value of the glacier, and add it and interpolate to add to the norm_all array.
        for r in range(0, norm_all.shape[0]):
            if r == 0:
                # put first value dhdt value into the norm_all. n+1 because the first col is taken by the elevnorms
                norm_all[r,n+1] = dhdt_single[0]
            elif r == (norm_all.shape[0] - 1):
                #put last value into the the last row for the glacier's 'stretched out'(interpolated) normalized curve
                norm_all[r,n+1] = dhdt_single[-1]
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
    norm_all_stats['norm_dhdt_med'] = np.nanmedian(norm_all[:,1:], axis=1)
    norm_all_stats['norm_dhdt_nmad'] = (1.483 *
                  np.median(np.absolute((norm_all[:,1:] - norm_all_stats['norm_dhdt_med'][:,np.newaxis])), axis=1))
    norm_all_stats['norm_dhdt_mean'] = np.nanmean(norm_all[:,1:], axis=1)
    norm_all_stats['norm_dhdt_std'] = np.nanstd(norm_all[:,1:], axis=1)
    norm_all_stats['norm_dhdt_68high'] = norm_all_stats['norm_dhdt_mean'] + norm_all_stats['norm_dhdt_std']
    norm_all_stats['norm_dhdt_68low'] = norm_all_stats['norm_dhdt_mean'] - norm_all_stats['norm_dhdt_std']
    if option_norm_limits:
        norm_all_stats.loc[norm_all_stats['norm_dhdt_68high'] > 1, 'norm_dhdt_68high'] = 1
        norm_all_stats.loc[norm_all_stats['norm_dhdt_68low'] < 0, 'norm_dhdt_68low'] = 0
    return norm_all_stats


# ===== START PROCESSING =====
# Load mass balance summary data
if os.path.exists(dems_output_fp + mb_summary_fn):
    mb_summary = pd.read_csv(dems_output_fp + mb_summary_fn)
else:
    # Merge files
    for n_fn, fn in enumerate(mb_summary_fn_list):
        mb_summary_subset = pd.read_csv(dems_output_fp + fn)
        mb_summary_subset['region'] = fn.split('_mb')[0]
        if n_fn == 0:
            mb_summary = mb_summary_subset
        else:
            mb_summary = mb_summary.append(mb_summary_subset)
    # Sort and add glacier number
    mb_summary = mb_summary.sort_values('RGIId')
    mb_summary.reset_index(inplace=True, drop=True)
    mb_summary['glacno'] = [str(int(x)).zfill(2) + '.' + str(int(np.round(x%1*10**5))).zfill(5)
                            for x in mb_summary['RGIId']]
    # Export dataset
    mb_summary.to_csv(dems_output_fp + mb_summary_fn, index=False)

# ===== PROCESS DATA =====
print('Glaciers total:', mb_summary.shape[0])
if ~(type(mb_summary.loc[0,'glacno']) == str):
    mb_summary['glacno'] = [str(int(x)).zfill(2) + '.' + str(int(np.round(x%1*10**5))).zfill(5)
                            for x in mb_summary['RGIId']]
mb_summary = mb_summary.loc[mb_summary['valid_area_perc'] >= valid_perc_threshold]
mb_summary.reset_index(inplace=True, drop=True)
mb_summary = mb_summary.loc[mb_summary['area_m2'] / 1e6 >= min_area_km2]
mb_summary.reset_index(inplace=True, drop=True)
print('Glaciers total after % threshold:', mb_summary.shape[0])

glacno_list = list(mb_summary.glacno.values)
main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)

# ===== BINNED DATA =====
binned_list = []
for glacno in glacno_list:
    csv_str = str(int(glacno.split('.')[0])) + '.' + glacno.split('.')[1]

    for i in os.listdir(binned_fp):
        if i.startswith(csv_str) and i.endswith('_mb_bins.csv'):
            binned_ds = pd.read_csv(binned_fp + i, na_values=' nan')

            # Rename columns so they are easier to read
            binned_ds = binned_ds.rename(columns=sheancoldict)
            # Remove bad values of dhdt
            binned_ds.loc[binned_ds[mb_cn] > mb_max, mb_cn] = np.nan
            binned_ds.loc[binned_ds[mb_cn] < mb_min, mb_cn] = np.nan
            # If dhdt is nan, remove row
            null_bins = binned_ds.loc[pd.isnull(binned_ds[mb_cn])].index.values
            binned_ds = binned_ds.drop(null_bins)

            # ===== BINNED DATA NORMALIZATIONS =====
            elev_cn = binned_ds.columns[0]
            glac_elev = binned_ds[elev_cn].values
            glac_mb = binned_ds[mb_cn].values.astype(float)
            # Larsen normalization (terminus = 0, top = 1)
            if option_normelev == 'larsen':
                binned_ds['elev_norm'] = (glac_elev - glac_elev[0]) / (glac_elev[-1] - glac_elev[0])
            # Huss normalization (terminus = 1, top = 0)
            elif option_normelev == 'huss':
                binned_ds['elev_norm'] = (glac_elev[-1] - glac_elev) / (glac_elev[-1] - glac_elev[0])

            # Normalized ice thickness change [ma]
            #  dhdt / dhdt_max
            # Shifted normalized ice thickness change such that everything is negative
            binned_ds['mb_norm_shifted'] = (glac_mb - np.nanmax(glac_mb))  / np.nanmin(glac_mb - np.nanmax(glac_mb))
            binned_ds.loc[binned_ds['mb_norm_shifted'] == -0, 'mb_norm_shifted'] = 0
            # Replace positive values to zero
            glac_mb[glac_mb >= 0] = 0
            binned_ds['mb_norm_huss'] = glac_mb / np.nanmin(glac_mb)
            binned_ds.loc[binned_ds['mb_norm_huss'] == -0, 'mb_norm_huss'] = 0

            # Append to list
            binned_list.append(binned_ds)


#%% ===== ELEVATION VS MASS BALANCE PLOTS======
# List of np.array where first column is elev_norm and second column is mass balance
elev_mb_list = [np.array([i[elev_cn].values, i[mb_cn].values]).transpose() for i in binned_list]
normelev_mb_list = [np.array([i['elev_norm'].values, i[mb_cn].values]).transpose() for i in binned_list]
normelev_mb_stats = norm_stats(normelev_mb_list)

# Estimate a curve
def curve_func(x, a, b, c, d):
    return (x + a)**d + b * (x + a) + c
p0 = [1,1,1,1]
coeffs, matcov = curve_fit(curve_func, normelev_mb_stats['norm_elev'].values,
                           normelev_mb_stats['norm_dhdt_med'].values, p0)
curve_x = np.arange(0,1.01,0.01)
curve_y = curve_func(curve_x, coeffs[0], coeffs[1], coeffs[2], coeffs[3])

# Plot
figwidth=6.5
figheight=8
fig, ax = plt.subplots(2, 1, squeeze=False, sharex=False, sharey=False,
                       figsize=(figwidth,figheight), gridspec_kw = {'wspace':0.4, 'hspace':0.25})
max_elev = 0
for n, i in enumerate(elev_mb_list):
    glac_elev = i[:,0]
    glac_mb = i[:,1]
    glac_elev_norm = normelev_mb_list[n][:,0]

    if glac_elev.max() > max_elev:
        max_elev = glac_elev.max()
        max_elev = np.ceil(max_elev/500)*500

    # Elevation vs MB
    ax[0,0].plot(glac_elev, glac_mb, linewidth=0.5, alpha=0.5)

    # Norm Elevation vs MB
    #  note: zorder overrides alpha, only alpha if same zorder
    ax[1,0].plot(glac_elev_norm, glac_mb, linewidth=0.5, alpha=0.2, zorder=1)
    ax[1,0].plot(normelev_mb_stats['norm_elev'], normelev_mb_stats['norm_dhdt_med'],
                 color='k', linewidth=1, alpha=1, zorder=2)

    ax[1,0].fill_between(normelev_mb_stats['norm_elev'], normelev_mb_stats['norm_dhdt_med'],
                         normelev_mb_stats['norm_dhdt_med'] + normelev_mb_stats['norm_dhdt_nmad'],
                         color='dimgray', alpha=0.5, zorder=1)
    ax[1,0].fill_between(normelev_mb_stats['norm_elev'], normelev_mb_stats['norm_dhdt_med'],
                         normelev_mb_stats['norm_dhdt_med'] - normelev_mb_stats['norm_dhdt_nmad'],
                         color='dimgray', alpha=0.5, zorder=1)
    ax[1,0].plot(curve_x, curve_y,
                 color='k', linewidth=1, alpha=1, linestyle='--', zorder=2)

# niceties - Elevation vs MB
ax[0,0].set_xlabel('Elevation (m a.s.l.)', fontsize=12)
ax[0,0].xaxis.set_major_locator(MultipleLocator(500))
ax[0,0].xaxis.set_minor_locator(MultipleLocator(100))
ax[0,0].set_xlim(0, max_elev)
ax[0,0].set_ylabel('Mass Balance (m w.e. $\mathregular{yr{-1}}$)', fontsize=12, labelpad=10)
ax[0,0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax[0,0].yaxis.set_major_locator(MultipleLocator(0.5))
ax[0,0].yaxis.set_minor_locator(MultipleLocator(0.1))

# niceties - Norm Elevation vs MB
ax[1,0].set_xlabel('Normalized Elevation (-)', fontsize=12)
ax[1,0].xaxis.set_major_locator(MultipleLocator(0.25))
ax[1,0].xaxis.set_minor_locator(MultipleLocator(0.05))
ax[1,0].set_xlim(0,1)
ax[1,0].set_ylabel('Mass Balance (m w.e. $\mathregular{yr{-1}}$)', fontsize=12, labelpad=10)
#ax[1,0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax[1,0].yaxis.set_major_locator(MultipleLocator(0.5))
ax[1,0].yaxis.set_minor_locator(MultipleLocator(0.1))
ax[1,0].set_ylim(-3,1.5)

# Save figure
fig.set_size_inches(figwidth,figheight)
figure_fn = 'elev_mb_all_gt' + str(valid_perc_threshold) + 'pct_gt' + str(min_area_km2) + 'km2.png'
fig.savefig(fig_fp + figure_fn, bbox_inches='tight', dpi=300)

#%% ===== REGIONAL ELEVATION VS MASS BALANCE PLOTS======
subregions = 'O2Regions'
if subregions == 'Berthier':
    regions = sorted(set(mb_summary.region.values))
    subregion_dict = {}
    for region in regions:
        subregion_dict[region] = region
elif subregions == 'O2Regions':
    regions = sorted(set(main_glac_rgi.O2Region.values))
    subregion_dict = {2:'Alaska Range', 3:'Alaska Pena', 4:'W Chugach Mtns', 5:'St Elias Mtns', 6:'N Coast Ranges',
                      9999:'All Alaska'}
reg_normelev_mb_dict = {}
regions.append(9999)

ncols = 2
nrows = int(np.ceil(len(regions)/ncols))

figwidth, figheight = 6.5, 8
fig, ax = plt.subplots(nrows, ncols, squeeze=False, sharex=False, sharey=False,
                       figsize=(figwidth,figheight), gridspec_kw = {'wspace':0.25, 'hspace':0.4})
ncol = 0
nrow = 0
for region in regions:
    if subregions == 'Berthier':
        reg_idx = list(np.where(mb_summary['region'] == region)[0])
    elif subregions =='O2Regions':
        reg_idx = list(np.where(main_glac_rgi['O2Region'] == region)[0])
    if region == 9999:
        reg_idx = main_glac_rgi.index.values

    reg_binned_list = [binned_list[i] for i in reg_idx]

    reg_elev_mb_list = [np.array([i[elev_cn].values, i[mb_cn].values]).transpose() for i in reg_binned_list]
    reg_normelev_mb_list = [np.array([i['elev_norm'].values, i[mb_cn].values]).transpose() for i in reg_binned_list]
    reg_normelev_mb_stats = norm_stats(reg_normelev_mb_list)
    reg_normelev_mb_dict[region] = dict(zip((reg_normelev_mb_stats['norm_elev'].values * 100).astype(int),
                                            reg_normelev_mb_stats['norm_dhdt_med'].values))

    # Estimate a curve
    x = reg_normelev_mb_stats['norm_elev'].values
    y = reg_normelev_mb_stats['norm_dhdt_med'].values
    def curve_func(x, a, b, c, d):
        return (x + a)**d + b * (x + a) + c
    p0 = [1,1,1,1]
    coeffs, matcov = curve_fit(curve_func, x, y, p0)
    curve_x = np.arange(0,1.01,0.01)
    curve_y = curve_func(curve_x, coeffs[0], coeffs[1], coeffs[2], coeffs[3])

    for n, i in enumerate(reg_elev_mb_list):
        glac_elev = i[:,0]
        glac_mb = i[:,1]
        glac_elev_norm = reg_normelev_mb_list[n][:,0]

        # Norm Elevation vs MB
        #  note: zorder overrides alpha, only alpha if same zorder
        ax[nrow,ncol].plot(glac_elev_norm, glac_mb, linewidth=0.5, alpha=0.2, zorder=1)
        ax[nrow,ncol].plot(reg_normelev_mb_stats['norm_elev'], reg_normelev_mb_stats['norm_dhdt_med'],
                           color='k', linewidth=1, alpha=1, zorder=2)
        ax[nrow,ncol].fill_between(reg_normelev_mb_stats['norm_elev'], reg_normelev_mb_stats['norm_dhdt_med'],
                                   reg_normelev_mb_stats['norm_dhdt_med'] + reg_normelev_mb_stats['norm_dhdt_nmad'],
                                   color='dimgray', alpha=0.5, zorder=1)
        ax[nrow,ncol].fill_between(reg_normelev_mb_stats['norm_elev'], reg_normelev_mb_stats['norm_dhdt_med'],
                                   reg_normelev_mb_stats['norm_dhdt_med'] - reg_normelev_mb_stats['norm_dhdt_nmad'],
                                   color='dimgray', alpha=0.5, zorder=1)
        ax[nrow,ncol].plot(curve_x, curve_y,
                           color='k', linewidth=1, alpha=1, linestyle='--', zorder=2)

    # niceties - Norm Elevation vs MB
    ax[nrow,ncol].xaxis.set_major_locator(MultipleLocator(0.25))
    ax[nrow,ncol].xaxis.set_minor_locator(MultipleLocator(0.05))
    ax[nrow,ncol].set_xlim(0,1)
    ax[nrow,ncol].yaxis.set_major_locator(MultipleLocator(1))
    ax[nrow,ncol].yaxis.set_minor_locator(MultipleLocator(0.25))
    ax[nrow,ncol].set_ylim(-3,1.5)
     # Title
    ax[nrow,ncol].text(0.5, 1.01, subregion_dict[region], size=10, horizontalalignment='center',
                       verticalalignment='bottom', transform=ax[nrow,ncol].transAxes)

    # Adjust row and column
    ncol += 1
    if ncol == ncols:
        nrow += 1
        ncol = 0

# Y-label
fig.text(0.04, 0.5, 'Mass Balance (m w.e. $\mathregular{yr^{-1}}$)', va='center', ha='center',
         rotation='vertical', size=12)
fig.text(0.5, 0.08, 'Normalized Elevation', va='center', ha='center', size=12)

# Save figure
fig.set_size_inches(figwidth,figheight)
figure_fn = 'elev_mb_regional_gt' + str(valid_perc_threshold) + 'pct_gt' + str(min_area_km2) + 'km2.png'
fig.savefig(fig_fp + figure_fn, bbox_inches='tight', dpi=300)

#%% ===== EXTRAPOLATE MASS BALANCE CURVES TO EVERY GLACIER =====
main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=[1], rgi_regionsO2='all', rgi_glac_number='all')
# Load hypsometry data
glac_hyps_all_df = pd.read_csv(hyps_fn)
glac_hyps_all = glac_hyps_all_df.iloc[:,1:].values

glac_elevnorm_all = np.zeros(glac_hyps_all.shape)
glac_hyps_all_mask = glac_hyps_all.copy()
glac_hyps_all_mask[glac_hyps_all_mask > 0] = 1
elev_bins = np.array(glac_hyps_all_df.columns[1:].astype(int))
elev_bin_dict = dict(zip(np.arange(0,len(elev_bins)), elev_bins))
glac_min = np.array([elev_bin_dict[i] for i in list(glac_hyps_all_mask.argmax(axis=1))])
glac_max = np.array([elev_bin_dict[i]
                    for i in list(glac_hyps_all.shape[1] - 1 - glac_hyps_all_mask[:,::-1].argmax(axis=1))])

# Normalized elevation (consistent with hypsometry)
glac_elev_all = glac_hyps_all_mask * elev_bins[np.newaxis,:]
if option_normelev == 'larsen':
    glac_elevnorm_all = ((glac_elev_all - glac_min[:,np.newaxis]) * glac_hyps_all_mask /
                         (glac_max - glac_min)[:,np.newaxis])
elif option_normelev == 'huss':
    glac_elevnorm_all = ((glac_max[:,np.newaxis] - glac_elev_all) * glac_hyps_all_mask /
                         (glac_max - glac_min)[:,np.newaxis])
glac_elevnorm_all[glac_elevnorm_all==0] = 0
# Fill in glaciers with single bin
for singlebin_row in list(np.where(glac_max - glac_min == 0)[0]):
    singlebin_col = glac_hyps_all_mask.argmax(axis=1)[np.where(glac_max - glac_min == 0)[0]]
    glac_elevnorm_all[singlebin_row,singlebin_col] = 0.5

# Glacier regions used for indexing
glac_region = (np.zeros(glac_hyps_all.shape[0]) + 9999).astype(int)
for region in regions:
    region_idx = np.where(main_glac_rgi_all.O2Region == region)[0]
    glac_region[region_idx] = region

# Use regional norm elevation vs mb curves to extrapolate to all glaciers
glac_mb_all = np.zeros(glac_hyps_all.shape)

for region in regions:
    region_idx = np.where(glac_region == region)[0]

    # Partition for each region
    glac_region_mask = np.zeros(glac_hyps_all.shape)
    glac_region_mask[region_idx,:] = 1

    # Convert normalized elevation to integers to work with dictionary and mask for specific region
    glac_elevnorm_reg = (glac_elevnorm_all * 100).astype(int)

    # Apply region mb vs norm elevation dictionary
    glac_mb_reg = np.zeros(glac_elevnorm_reg.shape)
    for k, v in reg_normelev_mb_dict[region].items():
        glac_mb_reg[glac_elevnorm_reg == k] = v
    # mask values for areas without glacier hypsometry
    glac_mb_reg_masked = glac_mb_reg * glac_hyps_all_mask * glac_region_mask

    glac_mb_all[glac_mb_reg_masked != 0] = glac_mb_reg_masked[glac_mb_reg_masked != 0]

# Glacier-wide mass balance
glac_wide_mb = (glac_mb_all * glac_hyps_all).sum(axis=1) / glac_hyps_all.sum(axis=1)
main_glac_rgi_all['mb_mwea'] = glac_wide_mb

# Uncertainty based on extrapolation method
#  alternative: use the upper and lower bounds of regional curves for uncertainty
rgi_all_idx = []
for rgiid in main_glac_rgi.RGIId.values:
    rgi_all_idx.append(np.where(main_glac_rgi_all.RGIId.values == rgiid)[0][0])

mb_summary['mb_mwea_extrapolated'] = main_glac_rgi_all.loc[rgi_all_idx,'mb_mwea'].values
mb_summary['dif_mb'] = mb_summary.mb_mwea - mb_summary.mb_mwea_extrapolated
print('Potential bias (mean/median):', np.round(mb_summary.dif_mb.mean(),2), np.round(mb_summary.dif_mb.median(),2))

mb_summary['abs_dif_mb'] = np.absolute(mb_summary.mb_mwea - mb_summary.mb_mwea_extrapolated)
print('Uncertainty (mean/median):', np.round(mb_summary.abs_dif_mb.mean(),2),np.round(mb_summary.abs_dif_mb.median(),2))
mb_summary['mb_mwea_sigma'] = mb_summary.abs_dif_mb.mean()

# ===== EXPORT DATA AND EXTRAPOLATED VALUES =====
reg_t1_dict = {2: 1953., 3: 1950., 4: 1952., 5: 1968., 6: 1966, 9999: 1957.8}
reg_t2_dict = {2: 2004.75, 3: 2007.75, 4: 2007.75, 5: 2006.75, 6: 2007.75, 9999: 2006.75}

glac_wide_mb_export = pd.DataFrame(np.zeros((main_glac_rgi_all.shape[0],8)),
                                   columns=['RGIId', 'region_id', 'region', 'area_km2', 'mb_mwea', 'mb_mwea_sigma',
                                            't1', 't2'])
glac_wide_mb_export['RGIId'] = main_glac_rgi_all.RGIId
glac_wide_mb_export['region_id'] = glac_region
glac_wide_mb_export['region'] = glac_wide_mb_export.region_id.map(subregion_dict)
glac_wide_mb_export['area_km2'] = main_glac_rgi_all.Area
glac_wide_mb_export['mb_mwea'] = main_glac_rgi_all.mb_mwea
glac_wide_mb_export['mb_mwea_sigma'] = mb_summary.abs_dif_mb.mean()
glac_wide_mb_export['t1'] = main_glac_rgi_all.O2Region.map(reg_t1_dict).values
glac_wide_mb_export['t2'] = main_glac_rgi_all.O2Region.map(reg_t2_dict).values
# overwrite extrapolated values with observations
glac_wide_mb_export.loc[rgi_all_idx,'mb_mwea'] = mb_summary.mb_mwea.values
glac_wide_mb_export.loc[rgi_all_idx,'t1'] = mb_summary.t1.values
glac_wide_mb_export.loc[rgi_all_idx,'t2'] = mb_summary.t2.values
print('\nUncertainty from extrapolated values used for all due to issues with static analysis\n')
glac_wide_mb_export.to_csv(dems_output_fp + mb_mwea_all_fn)
