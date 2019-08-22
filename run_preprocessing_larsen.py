"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

# Built-in libraries
import os
import gdal
import argparse
# External libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from scipy import interpolate
from scipy import ndimage
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup

 

#%% TO-DO LIST:
# - clean up create lapse rate input data (put it all in input.py)

#%%
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    option_farinotti2019_input : int
        Switch for processing lapse rates (default = 0 (no))
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="select pre-processing options")
    # add arguments
    parser.add_argument('-option_farinotti2019_input', action='store', type=int, default=0,
                        help='option to produce Farinotti 2019 input products (1=yes, 0=no)')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off)')    
    return parser


def import_raster(raster_fn):
    """Open raster and obtain the values in its first band as an array
    Output: array of raster values
    """
    # open raster dataset
    raster_ds = gdal.Open(raster_fn)
    # extract band information and get values
    raster_band = raster_ds.GetRasterBand(1)
    raster_values = raster_band.ReadAsArray()
    # extra cell size
    gt = raster_ds.GetGeoTransform()
    pixel_x, pixel_y = gt[1], -gt[5]
    return raster_values, pixel_x, pixel_y


def filldem(dem, threshold=500, windowsize=5, burn_windowsize=3, glac_mask=None, option_onlyglaciers=1):
    """ Fill DEM based on a given threshold below median values 
    
    Parameters
    ----------
    dem : np.array
        raw DEM to be filled
    threshold : np.float
        threshold compared to median value of surrounding pixels to check if pixel is good or not
    windowsize : int
        size of focal window to compute median statistics for fill values
    burn_windowsize : int
        size of focal window use to burn in pixels surrounding nan to check if they have issues as well
    glac_mask : np.array
        glacier mask (same size as dem)
    option_onlyglaciers : int
        switch to only fill glacier values and only fill those values with good glacier pixels; otherwise, fill will be
        done using median value from both glacier and non-glacier pixels
    """
    if glac_mask is None or option_onlyglaciers == 0:
        glac_mask = np.ones(dem.shape)
    
    dem_init = dem.copy()
    # Burn in nan values to surrounding pixels
    if burn_windowsize > windowsize:
        burn_windowsize = windowsize
    dem_nanplus = ndimage.filters.generic_filter(dem, np.min, size=burn_windowsize)

    # Mask of nan values to check against threshold
    dem_nanplus_mask = np.ones(dem.shape)
    dem_nanplus_mask[np.where(np.isnan(dem_nanplus))] = 0
    # Threshold based on median from surrounding pixels to identify bad pixels
    dem_median = ndimage.filters.generic_filter(dem, np.nanmedian, size=windowsize)
    # Threshold to remove pixels (burn-in nan values for pixels surrounding existing nans with poor values)
    dem_threshold = dem_median - threshold
    dem_threshold[np.isnan(dem_threshold)] = -threshold
    
    # Burn in values that don't pass threshold
    dem[np.isnan(dem)] = -9999
    dem_test = dem - dem_threshold
    dem[dem_test < 0] = np.nan
    dem[glac_mask == 0] = np.nan

    # Fill remaining values with median of good values
    dem_median = ndimage.filters.generic_filter(dem, np.nanmedian, size=windowsize)
    dem_filled = dem.copy()
    dem_filled[np.isnan(dem)] = dem_median[np.isnan(dem)]
    dem_filled[glac_mask == 0] = dem_init[glac_mask == 0]
    
    # Replace nan values with -9999 for filled DEM
    dem_filled[np.isnan(dem_filled)] = -9999
        
    return dem_filled


def extract_hyps(main_glac_rgi, binsize):
    #%%
    """ Extract hypsometry and other features if desired based on main_glac_rgi and bin size 
    
    Limitations
    -----------
    Currently skips any pixels that have bad DEM values, defined as a pixel < minimum glacier elevation
    """
    rgi_regionsO1 = [main_glac_rgi.loc[0,'O1Region']]
    
    # Filepath 
    dem_fp = (input.main_directory + '/../IceThickness_Farinotti/surface_DEMs_RGI60/surface_DEMs_RGI60-' + 
              "{:02d}".format(rgi_regionsO1[0]) + '/')
    thickness_fp = (input.main_directory + '/../IceThickness_Farinotti/composite_thickness_RGI60-all_regions/' + 
                    'RGI60-' + "{:02d}".format(rgi_regionsO1[0]) + '/')
    
    elev_bins_all = np.arange(binsize / 2, main_glac_rgi.Zmax.max() + binsize / 2, binsize).astype(int)
    df_cns = ['RGIId']
    for elev_bin in elev_bins_all:
        df_cns.append(elev_bin)
    main_glac_hyps = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_thickness = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_width = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_length = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_slope = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_hyps['RGIId'] = main_glac_rgi['RGIId']
    main_glac_thickness['RGIId'] = main_glac_rgi['RGIId']
    main_glac_width['RGIId'] = main_glac_rgi['RGIId']
    main_glac_length['RGIId'] = main_glac_rgi['RGIId']
    main_glac_slope['RGIId'] = main_glac_rgi['RGIId']
    
    # Loop through glaciers to derive various attributes
    rgiid_list = list(main_glac_rgi['RGIId'].values)
    for n_rgiid, rgiid in enumerate(rgiid_list):
        
        # Load filenames
        thickness_fn = rgiid + '_thickness.tif'
        dem_fn = 'surface_DEM_' + rgiid + '.tif'
        
        # Import tifs
        thickness, thickness_pixel_x, thickness_pixel_y = import_raster(thickness_fp + thickness_fn)
        dem_raw, dem_pixel_x, dem_pixel_y = import_raster(dem_fp + dem_fn)
        
#        if n_rgiid % 500 == 0:
#            glacier_area_total = len(np.where(thickness > 0)[0]) * dem_pixel_x * dem_pixel_y / 10**6
#            print('glacier area [km2]:', np.round(glacier_area_total,2), 
#                  'vs RGI [km2]:', np.round(main_glac_rgi.loc[n_rgiid,'Area'],2))
            
        # Glacier mask      
        glac_mask = np.zeros(thickness.shape)
        glac_mask[thickness > 0] = 1  
        
        # Test for large discrepancies between RGI60, DEM, or data gaps
        glac_pix_total = np.sum(glac_mask)
        
        # DEM
        dem_masked = dem_raw.copy()
        dem_masked[dem_masked < 0] = np.nan
        dem_masked[(glac_mask == 0)] = np.nan
        dem_masked = np.ma.masked_invalid(dem_masked)
        
        glac_pix_ltZmin = len(np.where(dem_masked < main_glac_rgi.loc[n_rgiid,'Zmin'])[0]) / glac_pix_total * 100
        glac_pix_gtZmax = len(np.where(dem_masked > main_glac_rgi.loc[n_rgiid,'Zmax'])[0]) / glac_pix_total * 100
        
        if np.max([glac_pix_ltZmin, glac_pix_gtZmax]) > 10:
            print('\n',rgiid, 'poor agreement with RGI60:\n    pixels < Zmin [%]:', np.round(glac_pix_ltZmin,1), 
                  '\n    pixels > Zmin [%]:', np.round(glac_pix_gtZmax,1))
            skip_processing = 1
        else:
            skip_processing = 0
        
        if skip_processing == 0:
            # Remove bad pixels: negative values and glacier pixels below minimum elevation
            dem_raw[dem_raw < 0] = -9999
            dem_raw[(glac_mask == 1) & (dem_raw < main_glac_rgi.loc[n_rgiid,'Zmin'])] = -9999
            
            
            # Fill bad pixels: option_onlyglaciers controls if filling done using only glaciers or surrounding terrain
            nan_glacpixels = np.where((dem_raw < 0) & (glac_mask ==1))
            if len(nan_glacpixels[0]) > 0:
                nan_glacpixels_init = np.where((dem_raw < 0) & (glac_mask ==1))
                windowsize = 5
                while len(nan_glacpixels[0]) > 0 and windowsize < 26:
                    
                    nanpixels_prefill = len(nan_glacpixels[0])
                    
                    dem_filled = filldem(dem_raw, glac_mask=glac_mask, windowsize=windowsize, option_onlyglaciers=1)          
                    nan_glacpixels = np.where((dem_filled < 0) & (glac_mask ==1))
                    
                    print(n_rgiid, rgiid, 'WindowSize:', windowsize, '# NaN pixels:', nanpixels_prefill, 
                          'Post-fill:', len(nan_glacpixels[0]))
                    
                    windowsize += 4
                
#                for n in list(np.arange(0,len(nan_glacpixels_init[0]))):
#                    if dem_filled[nan_glacpixels_init[0][n], nan_glacpixels_init[1][n]] < 0:
#                        print(nan_glacpixels_init[0][n], nan_glacpixels_init[1][n], dem_filled[nan_glacpixels_init[0][n]])
            else:
                dem_filled = dem_raw
            
            # DEM into bins
            dem = np.zeros(thickness.shape)
            dem_rounded = np.zeros(thickness.shape)
            dem[glac_mask == 1] = dem_filled[glac_mask == 1]
            dem_rounded[glac_mask == 1] = binsize * (dem[glac_mask == 1] / binsize).astype(int) + binsize / 2
            dem_rounded = dem_rounded.astype(int)
            
            # Unique bins exluding zero
            elev_bins = list(np.unique(dem_rounded))
            elev_bins.remove(0)
    
            for elev_bin in elev_bins:
    #        for elev_bin in elev_bins[0:10]:
                
                if debug:
                    print('\nElevation bin:', elev_bin)
                    
                bin_mask = np.where(dem_rounded == elev_bin)
                
                # Area [km2] - bin total
                bin_hyps = len(bin_mask[0]) * dem_pixel_x * dem_pixel_y / 10**6
                
                # Thickness [m] - bin mean
                bin_thickness = thickness[bin_mask[0], bin_mask[1]].mean()
    
                # Slope [deg] - bin mean
                grad_x, grad_y = np.gradient(dem_filled, dem_pixel_x, dem_pixel_y)
                slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
                slope_deg = np.rad2deg(slope)
                bin_slope = np.mean(slope_deg[bin_mask])
                
                # Length [km] - based on the mean slope and bin elevation
                bin_length = binsize / np.tan(np.deg2rad(bin_slope)) / 1000
                
                # Width [km] - based on length (inherently slope) and bin area
                bin_width = bin_hyps / bin_length
                
                # Record properties
    #            print(n_rgiid, elev_bin, bin_hyps)
                main_glac_hyps.loc[n_rgiid, elev_bin] = bin_hyps
                main_glac_thickness.loc[n_rgiid, elev_bin] = bin_thickness
                main_glac_width.loc[n_rgiid, elev_bin] = bin_width
                main_glac_length.loc[n_rgiid, elev_bin] = bin_length
                main_glac_slope.loc[n_rgiid, elev_bin] = bin_slope
            
            if main_glac_hyps.shape[1] > len(df_cns):
                print(n_rgiid, rgiid, main_glac_hyps.shape[1])
    
    #%%
    return main_glac_hyps, main_glac_thickness, main_glac_width, main_glac_width, main_glac_slope


parser = getparser()
args = parser.parse_args()

if args.debug == 1:
    debug = True
else:
    debug = False

#%%
larsen_summary = pd.read_csv(input.larsen_fp + input.larsen_fn)
larsen_summary = larsen_summary.sort_values('RGIId')
larsen_summary.reset_index(drop=True, inplace=True)
glacno = sorted([x.split('-')[1].split('.')[1] for x in larsen_summary.RGIId.values])

# Add directory names to Larsen dataset
glac_names = list(larsen_summary.name.values)
glac_names_nospace = [x.replace(' ','') for x in glac_names]
glac_names_nospace[glac_names_nospace.index('TlikakilaN.Fork')] = 'TlikakilaNorthFork'
glac_names_nospace[glac_names_nospace.index('TlikakilaFork')] = 'TlikakilaGlacierFork'
glac_names_nospace[glac_names_nospace.index('Melbern')] = 'GrandPacificMelbern'
larsen_summary['name_nospace'] = glac_names_nospace
larsen_summary['startyear_str'] = [str(x)[:4] for x in larsen_summary.date0.values]
larsen_summary['endyear_str'] = [str(x)[:4] for x in larsen_summary.date1.values]

# Replace Mendenhall with '2000F', '2012F'
# Lemon Creek with '1993F', '2012F'
# Taku '1993F', '2012F'
# Nizina is not there
# Yanert is not available for the given time periods - others are


#%%

rgi_regionsO1 = [1]
main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2='all', 
                                                  rgi_glac_number=glacno)

binsize = 30 # elevation bin must be an integer greater than 1

# Glacier hypsometry [km**2], total area
# NEED TO FIX BROKEN FUNCTION FOR PROCESSING FARINOTTI ET AL. (2019) DATA
#main_glac_hyps, main_glac_thickness, main_glac_width, main_glac_width, main_glac_slope = (
#        extract_hyps(main_glac_rgi, binsize))
main_glac_hyps_10m = modelsetup.import_Husstable(main_glac_rgi, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
binsize_rgi = int(main_glac_hyps_10m.columns[1]) - int(main_glac_hyps_10m.columns[0])

#%%
add_cols = 3 - (main_glac_hyps_10m.shape[1] % 3)
for ncol in np.arange(0,add_cols):
    colname = str(int(main_glac_hyps_10m.columns[-1]) + binsize_rgi)
    main_glac_hyps_10m[colname] = 0

hyps_10m = main_glac_hyps_10m.values
hyps_30m = hyps_10m.reshape(-1,3).sum(1).reshape(hyps_10m.shape[0], int(hyps_10m.shape[1]/3))
hyps_30m_cns = list(main_glac_hyps_10m.columns.values[1::3].astype(int))
main_glac_hyps_30m = pd.DataFrame(hyps_30m, columns=hyps_30m_cns)

#%%

data_header = ['E', 'DZ', 'DZ25', 'DZ75', 'AAD', 'MassChange', 'MassBal', 'NumData']
larsen_summary['mb_mwea_v2'] = np.nan
larsen_summary['mb_gta_v2'] = np.nan
larsen_summary['area_rgi'] = np.nan

for nglac, glac_name in enumerate(list(larsen_summary.name.values)):
    
    print(nglac)
    #%%
    larsen_glac_fp = input.main_directory + '/../DEMs/larsen/data/'
    larsen_glac_fn = (larsen_summary.loc[nglac,'name_nospace'] + '.' + larsen_summary.loc[nglac,'startyear_str'] + '.' +
                      larsen_summary.loc[nglac,'endyear_str'] + '.output.txt')
    
#    larsen_glac_fn = 'Taku.2007.2014.output.txt'
    
    if os.path.isfile(larsen_glac_fp + larsen_glac_fn):
        data = np.genfromtxt(larsen_glac_fp + larsen_glac_fn, skip_header=3)
        df = pd.DataFrame(data, columns=data_header)
        # Shift bins by 15 so elevations based on center of bin and not bottom of bin
        df['E'] = df.E + 15
        
#        if larsen_summary.loc[nglac,'term_type'] == 'Tidewater':
#            print(nglac, larsen_summary.loc[nglac,'term_type'], df.loc[0,'E'])
        
        # Check if all bins accounted for
        rgi_hyps_raw = np.array(main_glac_hyps_30m.loc[nglac,:].values)
        rgi_hyps_idx = np.where(rgi_hyps_raw > 0)[0]
        
        rgi_bin_min = hyps_30m_cns[rgi_hyps_idx[0]]
        rgi_bin_max = hyps_30m_cns[rgi_hyps_idx[-1]]
        

        # ===== EXTEND TERMINUS (if needed) =====
        larsen_bin_min = df.loc[0,'E']
        if rgi_bin_min > larsen_bin_min:
            rgi_bin_min = larsen_bin_min
            print(glac_name, 'Larsen terminus is lower')
            
        elif rgi_bin_min < larsen_bin_min:
            n_bins2add = int((larsen_bin_min - rgi_bin_min) / binsize)
            df_2append = pd.DataFrame(np.full((n_bins2add,len(df.columns)),np.nan), columns=df.columns)
            df_2append['E'] = np.arange(rgi_bin_min, larsen_bin_min, binsize)
            df_2append['DZ'] = df.loc[0,'DZ']
            df = df_2append.append(df)
            df.reset_index(inplace=True, drop=True)
            
        # ===== EXPAND ACCUMULATION AREA (if needed) =====
        larsen_bin_max = df.loc[df.shape[0]-1,'E']
        if rgi_bin_max < larsen_bin_max:
            rgi_bin_max = larsen_bin_max 
            print(glac_name, 'Larsen terminus is higher')
            
        elif rgi_bin_max > larsen_bin_max:
            # Append more bins
            n_bins2add = int((rgi_bin_max - larsen_bin_max) / binsize)
            df_2append = pd.DataFrame(np.full((n_bins2add,len(df.columns)),np.nan), columns=df.columns)
            df_2append['E'] = np.arange(larsen_bin_max + 15, larsen_bin_max + n_bins2add * binsize, binsize)
            df = df.append(df_2append)
            df.reset_index(inplace=True, drop=True)
            
            # Set accumulation at top bin to zero
            df.loc[df.shape[0]-1,'DZ'] = 0
            # Linearly interpolate other values
            df['DZ'] = df.DZ.interpolate(method='linear')
            
        df['E_norm'] = (df.E - df.E.min()) / (df.E.max() - df.E.min())
        
        df['hyps_km2'] =  main_glac_hyps_30m.loc[nglac, rgi_bin_min:rgi_bin_max].values
        
#        #%%
#        A = main_glac_hyps_30m.loc[nglac, rgi_bin_min:rgi_bin_max].values
#        B = hyps_30m_cns[rgi_hyps_idx[0]:rgi_hyps_idx[-1]+1]
#        #%%
        
        df['mb_mwea'] = df.DZ * 850/1000
        df['mb_gta'] = df.mb_mwea / 1000 * df.hyps_km2
        
        glac_mb_gta = df.mb_gta.sum()
        glac_mb_mwea = glac_mb_gta / df.hyps_km2.sum() * 1000
        print('Mass loss [Gt yr-1]:', np.round(glac_mb_gta,3))
        print('Mass loss [mwe yr-1]:', np.round(glac_mb_mwea,2))
        
        larsen_summary.loc[nglac, 'mb_mwea_v2'] = glac_mb_mwea
        larsen_summary.loc[nglac, 'mb_gta_v2'] = glac_mb_gta
        larsen_summary.loc[nglac, 'area_rgi'] = df.hyps_km2.sum()
    
        # Elevation Change vs. Normalized Elevation
        fig, ax = plt.subplots(1, 1, squeeze=False, sharex=False, sharey=False, 
                               gridspec_kw = {'wspace':0.4, 'hspace':0.15})
        ax[0,0].plot(df.E_norm.values, df.DZ.values, color='k', linewidth=1, zorder=2, label='Baird')
        
        ax[0,0].set_xlabel('Normalized Elevation', size=12) 
        ax[0,0].set_ylabel('Elevation Change (m yr-1)', size=12)    
        
        ax[0,0].set_xlim(0,1)       
        
        # Save figure
        #  figures can be saved in any format (.jpg, .png, .pdf, etc.)
        fig.set_size_inches(6, 4)
        figure_fp = larsen_glac_fp + '/figures/'
        if os.path.exists(figure_fp) == False:
            os.makedirs(figure_fp)
        figure_fn = larsen_glac_fn.replace('output.txt','_elevchg.png')
        fig.savefig(figure_fp + figure_fn, bbox_inches='tight', dpi=300)
        fig.clf()
        
    else:
        print(glac_name, 'filename not correct or not available')

                
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
