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


parser = getparser()
args = parser.parse_args()

if args.debug == 1:
    debug = True
else:
    debug = False

#%%
if args.option_farinotti2019_input == 1:
    print("\nProcess the ice thickness and surface elevation data from Farinotti (2019) to produce area," +
          "ice thickness, width, and length for each elevation bin\n")
    
    print('\n\n\nDELETE SCRIPT AND REPLACE WITH THE ONE IN LARSEN DATASET PROCESSING\n\n\n')
    
    rgi_regionsO1 = [1]
    binsize = 10 # elevation bin must be an integer greater than 1
    output_fp = input.main_directory + '/../IceThickness_Farinotti/'
    
    for region in rgi_regionsO1:
        # Filepath 
        dem_fp = (input.main_directory + '/../IceThickness_Farinotti/surface_DEMs_RGI60/surface_DEMs_RGI60-' + 
                  "{:02d}".format(region) + '/')
        thickness_fp = (input.main_directory + '/../IceThickness_Farinotti/composite_thickness_RGI60-all_regions/' + 
                        'RGI60-' + "{:02d}".format(region) + '/')
        
        # Glaciers
        main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=[region], rgi_regionsO2='all',
                                                          rgi_glac_number='all')
        
        elev_bins_all = np.arange(binsize / 2, main_glac_rgi.Zmax.max() + binsize / 2, binsize).astype(int)
        df_cns = ['RGIId']
        for elev_bin in elev_bins_all:
            df_cns.append(elev_bin)
        main_glac_hyps = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
        main_glac_thickness = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
        main_glac_width = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
        main_glac_length = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
        main_glac_slope = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
        
        # Loop through glaciers to derive various attributes
        rgiid_list = list(main_glac_rgi['RGIId'].values)
        for n_rgiid, rgiid in enumerate(rgiid_list):
            if n_rgiid % 500 == 0:
                print(rgiid)
            
            # Load filenames
            thickness_fn = rgiid + '_thickness.tif'
            dem_fn = 'surface_DEM_' + rgiid + '.tif'
            
            # Import tifs
            thickness, thickness_pixel_x, thickness_pixel_y = import_raster(thickness_fp + thickness_fn)
            dem_raw, dem_pixel_x, dem_pixel_y = import_raster(dem_fp + dem_fn)
            
            if n_rgiid % 500 == 0:
                glacier_area_total = len(np.where(thickness > 0)[0]) * dem_pixel_x * dem_pixel_y / 10**6
                print('glacier area [km2]:', np.round(glacier_area_total,2), 
                      'vs RGI [km2]:', np.round(main_glac_rgi.loc[n_rgiid,'Area'],2))
                
            # Loop through glacier indices
            glac_idx = np.where(thickness > 0)
            
            # Glacier mask            
            glac_mask = np.zeros(thickness.shape)
            glac_mask[glac_idx[0],glac_idx[1]] = 1
            
            # DEM into bins
            dem = np.zeros(thickness.shape)
            dem_rounded = np.zeros(thickness.shape)
            dem[glac_idx[0],glac_idx[1]] = dem_raw[glac_idx[0],glac_idx[1]]
            dem_rounded[glac_idx[0],glac_idx[1]] = (binsize * (dem[glac_idx[0],glac_idx[1]] / binsize).astype(int) 
                                                    + binsize / 2)
            dem_rounded = dem_rounded.astype(int)
            
            # Unique bins exluding zero
            elev_bins = list(np.unique(dem_rounded))
            elev_bins.remove(0)

            for elev_bin in elev_bins:
                
                if debug:
                    print('\nElevation bin:', elev_bin)
                    
                bin_mask = np.where(dem_rounded == elev_bin)
                
                # Area [km2] - bin total
                bin_hyps = len(bin_mask[0]) * dem_pixel_x * dem_pixel_y / 10**6
                
                # Thickness [m] - bin mean
                bin_thickness = thickness[bin_mask[0], bin_mask[1]].mean()

                # Slope [deg] - bin mean
                grad_x, grad_y = np.gradient(dem_raw, dem_pixel_x, dem_pixel_y)
                slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
                slope_deg = np.rad2deg(slope)
                bin_slope = np.mean(slope_deg[bin_mask])
                
                # Length [km] - based on the mean slope and bin elevation
                bin_length = binsize / np.tan(np.deg2rad(bin_slope)) / 1000
                
                # Width [km] - based on length (inherently slope) and bin area
                bin_width = bin_hyps / bin_length
                
                # Record properties
                main_glac_hyps.loc[n_rgiid, elev_bin] = bin_hyps
                main_glac_thickness.loc[n_rgiid, elev_bin] = bin_thickness
                main_glac_width.loc[n_rgiid, elev_bin] = bin_width
                main_glac_length.loc[n_rgiid, elev_bin] = bin_length
                main_glac_slope.loc[n_rgiid, elev_bin] = bin_slope
                
                if debug:
                    print('Area [km2]:', bin_hyps, '(Pixels:', len(bin_mask[0]), ')',
                          '\nThickness [m]:', np.round(bin_thickness,1),
                          '\nLength [km]:', np.round(bin_length,3), 'Width [km]:', np.round(bin_width,3), 
                          '\nSlope [deg]:', np.round(bin_slope,1))
        
        # Export results
        main_glac_hyps.to_csv(output_fp + 'area_km2_' + "{:02d}".format(region) + '_Farinotti2019_' + 
                              str(binsize) + 'm.csv', index=False)
        main_glac_thickness.to_csv(output_fp + 'thickness_m_' + "{:02d}".format(region) + '_Farinotti2019_' + 
                                   str(binsize) + 'm.csv', index=False)
        main_glac_width.to_csv(output_fp + 'width_km_' + "{:02d}".format(region) + '_Farinotti2019_' + 
                               str(binsize) + 'm.csv', index=False)
        main_glac_length.to_csv(output_fp + 'length_km_' + "{:02d}".format(region) + '_Farinotti2019_' + 
                                str(binsize) + 'm.csv', index=False)
        main_glac_slope.to_csv(output_fp + 'slope_deg_' + "{:02d}".format(region) + '_Farinotti2019_' + 
                               str(binsize) + 'm.csv', index=False)
                
                
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
