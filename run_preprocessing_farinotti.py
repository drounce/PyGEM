"""
pygemfxns_preprocessing.py is a list of the model functions that are used to preprocess the data into the proper format.

"""

# Built-in libraries
import os
import argparse
# External libraries
from osgeo import gdal
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely

from pygeotools.lib import iolib, warplib, geolib, timelib, malib

import pygemfxns_modelsetup as modelsetup


#Function to generate a 3-panel plot for input arrays
def plot3panel(dem_list, clim=None, titles=None, cmap='inferno', label=None, overlay=None, fn=None):
    fig, axa = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,5))
    alpha = 1.0
    for n, ax in enumerate(axa):
        #Gray background
        ax.set_facecolor('0.5')
        #Force aspect ratio to match images
        ax.set(aspect='equal')
        #Turn off axes labels/ticks
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles is not None:
            ax.set_title(titles[n])
        #Plot background shaded relief map
        if overlay is not None:
            alpha = 0.7
            axa[n].imshow(overlay[n], cmap='gray', clim=(1,255))
    #Plot each array
    im_list = [axa[i].imshow(dem_list[i], clim=clim, cmap=cmap, alpha=alpha) for i in range(len(dem_list))]
    fig.tight_layout()
    fig.colorbar(im_list[0], ax=axa.ravel().tolist(), label=label, extend='both', shrink=0.5)
    if fn is not None:
        fig.savefig(fn, bbox_inches='tight', pad_inches=0, dpi=150)

#Input DEM filenames
dem_ref_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/DEMs/Alaska_albers_V3_mac/Alaska_albers_V3.tif'
thickness_fp_prefix = ('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/IceThickness_Farinotti/' +
                       'composite_thickness_RGI60-all_regions/')
dem_farinotti_fp = ('/Users/davidrounce/Documents/Dave_Rounce/HiMAT/IceThickness_Farinotti/surface_DEMs_RGI60/' +
                    'surface_DEMs_RGI60-01/')
output_fp = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/IceThickness_Farinotti/output/'
fig_fp = output_fp + 'figures/'
if os.path.exists(output_fp) == False:
    os.makedirs(output_fp)
if os.path.exists(fig_fp) == False:
    os.makedirs(fig_fp)

rgi_regionsO1 = [1]                 # RGI Order 1 regions
binsize = 10                        # elevation bin (must be an integer greater than 1)
dem_poorquality_threshold = 200     # threshold used to identify problems with Farinotti DEM
option_plot_DEMsraw = True          # Option to plot the raw DEMs
option_plot_DEMs = False             # Option to plot the masked DEMs
debug = False

# ===== LOAD GLACIERS ======
glacno_wpoor_DEM = []
for region in rgi_regionsO1:

    thickness_fp = thickness_fp_prefix + 'RGI60-' + str(region).zfill(2) + '/'

    glacno_list = []
    for i in os.listdir(thickness_fp):
        if i.endswith('_thickness.tif'):
            glacno_list.append(i.split('-')[1].split('_')[0])
    glacno_list = sorted(glacno_list)

    # print('\n\nDELETE ME - SWITCH TO COMPLETE LIST\n\n')
    # glacno_list = ['01.03622']
    # glacno_list = glacno_list[10000:10010]

    # Load RGI glacier data
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glacno_list)
    # setup empty datasets
    elev_bins_all = np.arange(binsize / 2, main_glac_rgi.Zmax.max() + binsize / 2, binsize).astype(int)
    df_cns = ['RGIId']
    for elev_bin in elev_bins_all:
        df_cns.append(elev_bin)
    main_glac_hyps = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_thickness = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_width = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_length = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_slope = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(df_cns))), columns=df_cns)
    main_glac_hyps['RGIId'] = main_glac_rgi.RGIId.values
    main_glac_thickness['RGIId'] = main_glac_rgi.RGIId.values
    main_glac_width['RGIId'] = main_glac_rgi.RGIId.values
    main_glac_length['RGIId'] = main_glac_rgi.RGIId.values
    main_glac_slope['RGIId'] = main_glac_rgi.RGIId.values

    # ===== PROCESS EACH GLACIER ======
    for nglac, glacno in enumerate(glacno_list):
        # print(nglac, glacno)
        thickness_fn = thickness_fp + 'RGI60-' + glacno + '_thickness.tif'
        dem_farinotti_fn = dem_farinotti_fp + 'surface_DEM_RGI60-' + glacno + '.tif'

        # Reproject, resample, warp rasters to common extent, grid size, etc.
        #  note: use thickness for the reference to avoid unrealistic extrapolations, e.g., negative thicknesses
        #        also using equal area increases areas significantly compared to RGI
        raster_fn_list = [dem_ref_fn, dem_farinotti_fn, thickness_fn]
        ds_list = warplib.memwarp_multi_fn(raster_fn_list, extent='intersection', res='min', t_srs=thickness_fn)
        # print('\n\nSWITCH BACK TO THICKNESS_FN AFTER OTHERS CORRECTED!\n\n')
        # ds_list = warplib.memwarp_multi_fn(raster_fn_list, extent='intersection', res='min', t_srs=dem_ref_fn)

        # masked arrays using ice thickness estimates
        dem_ref_raw, dem_far_raw, thickness = [iolib.ds_getma(i) for i in ds_list]
        dem_ref = dem_ref_raw.copy()
        dem_ref.mask = thickness.mask
        dem_far = dem_far_raw.copy()
        dem_far.mask = thickness.mask

        # DEM selection for binning computations
        # if exceeds threshold, then use the reference
        if (abs(main_glac_rgi.loc[nglac,'Zmin'] - dem_far.min()) > dem_poorquality_threshold or
            abs(main_glac_rgi.loc[nglac,'Zmax'] - dem_far.max()) > dem_poorquality_threshold):
            print('  Check Glacier ' + glacno + ': use Christian DEM instead of Farinotti')
            print('\n     RGI Zmin/Zmax:', main_glac_rgi.loc[nglac,'Zmin'], '/', main_glac_rgi.loc[nglac,'Zmax'])
            print('     Farinotti Zmin/Zmax:', np.round(dem_far.min(),0), '/', np.round(dem_far.max(),0))
            print('     Christian Zmin/Zmax:', np.round(dem_ref.min(),0), '/', np.round(dem_ref.max(),0), '\n')
            glacno_wpoor_DEM.append(glacno)
            dem = dem_ref
            dem_raw = dem_ref_raw

            # ===== PLOT DEMS TO CHECK =====
            if option_plot_DEMsraw:
                dem_list_raw = [dem_ref_raw, dem_far_raw, thickness]
                titles = ['DEM-Christian-raw', 'DEM-Farinotti-raw', 'Thickness']
                clim = malib.calcperc(dem_list_raw[0], (2,98))
                plot3panel(dem_list_raw, clim, titles, 'inferno', 'Elevation (m WGS84)', fn=fig_fp + glacno +
                           '_dem_raw.png')

            if option_plot_DEMs:
                dem_list = [dem_ref, dem_far, thickness]
                titles = ['DEM-Christian', 'DEM-Farinotti', 'Thickness']
                clim = malib.calcperc(dem_list[0], (2,98))
                plot3panel(dem_list, clim, titles, 'inferno', 'Elevation (m WGS84)', fn=fig_fp + glacno + '_dem.png')
        # otherwise, use Farinotti
        else:
            dem = dem_far
            dem_raw = dem_far_raw

        #Extract x and y pixel resolution (m) from geotransform
        gt = ds_list[0].GetGeoTransform()
        px_res = (gt[1], -gt[5])
        #Calculate pixel area in m^2
        px_area = px_res[0]*px_res[1]

        if debug:
            print('\nx_res [m]:', np.round(px_res[0],1), 'y_res[m]:', np.round(px_res[1],1),'\n')

        # ===== USE SHAPEFILE OR SINGLE POLYGON TO CLIP =====
        # shp_fn = '/Users/davidrounce/Documents/Dave_Rounce/HiMAT/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
        # #Create binary mask from polygon shapefile to match our warped raster datasets
        # shp_mask = geolib.shp2array(shp_fn, ds_list[0])
        # #Now apply the mask to each array
        # dem_list_shpclip = [np.ma.array(dem, mask=shp_mask) for dem in dem_list]
        # plot3panel(dem_list_shpclip, clim, titles, 'inferno', 'Elevation (m WGS84)', fn=output_fp + 'dem_shpclp.png')
        # rgi_alaska = gpd.read_file(shp_fn)
        # print(rgi_alaska.head())
        # rgi_alaska.plot();
        # print(rgi_alaska.crs)
        # # print('\nGeometry_type:\n',rgi_alaska[0:5].geom_type)
        # # print('\nArea (NOTE THESE ARE IN DEGREES!):\n',rgi_alaska[0:5].geometry.area)
        # # print('\nBounds:\n',rgi_alaska[0:5].geometry.bounds)
        # rgi_alaska.plot(column='O2Region', categorical=True, legend=True, figsize=(14,6))
        # rgiid = 'RGI60-' + glacno
        # rgi_single = rgi_alaska[rgi_alaska['RGIId'] == rgiid]
        # # export to
        # rgi_single_fn = 'rgi_single.shp'
        # rgi_single.to_file(rgi_single_fn)
        # #Create binary mask from polygon shapefile to match our warped raster datasets
        # rgi_single_mask = geolib.shp2array(rgi_single_fn, ds_list[0])
        # #Now apply the mask to each array
        # dem_list_shpclip = [np.ma.array(dem, mask=rgi_single_mask) for dem in dem_list]
        # plot3panel(dem_list_shpclip, clim, titles, 'inferno', 'Elevation (m WGS84)', fn=output_fp + 'dem_single.png')
        # =============================================================================================================

        if debug:
            glacier_area_total = thickness.count() * px_res[0] * px_res[1] / 10**6
            print(glacno, 'glacier area [km2]:', np.round(glacier_area_total,2),
                  'vs RGI [km2]:', np.round(main_glac_rgi.loc[nglac,'Area'],2))

        # Remove negative elevation values
        dem[dem < 0] = 0
        dem.mask = thickness.mask

        elev_bin_min = binsize * (dem.min() / binsize).astype(int)
        elev_bin_max = binsize * (dem.max() / binsize).astype(int) + binsize

        print(nglac, glacno, elev_bin_min, elev_bin_max)

        # if elev_bin_min < 0:
        #     print(nglac, glacno, elev_bin_min, elev_bin_max)
        #     debug_fp = input.output_sim_fp + 'debug/'
        #     # Create filepath if it does not exist
        #     if os.path.exists(debug_fp) == False:
        #         os.makedirs(debug_fp)
        #     debug_df = pd.DataFrame(np.zeros((1,1)), columns=['count'])
        #     debug_df.iloc[0,0] = 1
        #     debug_fn_loaded = str(glacno) + '_nglac' + str(nglac) + '_minlt0_.csv'
        #     debug_df.to_csv(debug_fp + debug_fn_loaded)

        elev_bin_edges = np.arange(elev_bin_min, elev_bin_max+binsize, binsize)
        elev_bins = (elev_bin_edges[0:-1] + binsize/2).astype(int)

        # Hypsometry [km2]
        #  must used .compressed() in histogram to exclude masked values
        hist, elev_bin_edges = np.histogram(dem.reshape(-1).compressed(), bins=elev_bin_edges)
        bin_hyps = hist * px_res[0] * px_res[1] / 10**6
        if debug:
            print('Zmin/Zmax:', np.round(dem.min(),0), '/', np.round(dem.max(),0), '\n')
            print('elev_bin_edges:', elev_bin_edges)
            print('hist:', hist)
            print('total area:', hist.sum() * px_res[0] * px_res[1] / 10**6)

        # Mean thickness [m]
        hist_thickness, elev_bin_edges = np.histogram(dem.reshape(-1).compressed(), bins=elev_bin_edges,
                                                      weights=thickness.reshape(-1).compressed())
        bin_thickness = hist_thickness / hist

        # Mean Slope [deg]
        # --> MAY WANT TO RESAMPLE TO SMOOTH DEM PRIOR TO ESTIMATING SLOPE
        grad_x, grad_y = np.gradient(dem_raw, px_res[0], px_res[1])
        slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
        slope_deg = np.rad2deg(slope)
        slope_deg.mask = dem.mask
        hist_slope, elev_bin_edges = np.histogram(dem.reshape(-1).compressed(), bins=elev_bin_edges,
                                                  weights=slope_deg.reshape(-1).compressed())
        bin_slope = hist_slope / hist

        # Length [km] - based on the mean slope and bin elevation
        bin_length = binsize / np.tan(np.deg2rad(bin_slope)) / 1000

        # Width [km] - based on length (inherently slope) and bin area
        bin_width = bin_hyps / bin_length

        # Record properties
        # Check if need to expand columns
        missing_cns = sorted(list(set(elev_bins) - set(df_cns)))
        if len(missing_cns) > 0:
            for missing_cn in missing_cns:
                main_glac_hyps[missing_cn] = 0
                main_glac_thickness[missing_cn] = 0
                main_glac_width[missing_cn] = 0
                main_glac_length[missing_cn] = 0
                main_glac_slope[missing_cn] = 0
        # Record data
        main_glac_hyps.loc[nglac, elev_bins] = bin_hyps
        main_glac_thickness.loc[nglac, elev_bins] = bin_thickness
        main_glac_width.loc[nglac, elev_bins] = bin_width
        main_glac_length.loc[nglac, elev_bins] = bin_length
        main_glac_slope.loc[nglac, elev_bins] = bin_slope

    # Remove NaN values
    main_glac_hyps = main_glac_hyps.fillna(0)
    main_glac_thickness = main_glac_thickness.fillna(0)
    main_glac_width = main_glac_width.fillna(0)
    main_glac_length = main_glac_length.fillna(0)
    main_glac_slope = main_glac_slope.fillna(0)
    # Remove negative values
    main_glac_hyps[main_glac_hyps < 0] = 0
    main_glac_thickness[main_glac_thickness < 0] = 0
    main_glac_width[main_glac_width < 0] = 0
    main_glac_length[main_glac_length < 0] = 0
    main_glac_slope[main_glac_slope < 0] = 0
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

##%%
#import pandas as pd
#import pygem_input as input
#area = pd.read_csv(input.hyps_filepath + 'area_km2_01_Farinotti2019_10m_old.csv')
#thickness = pd.read_csv(input.hyps_filepath + 'thickness_m_01_Farinotti2019_10m_old.csv')
#length = pd.read_csv(input.hyps_filepath + 'length_km_01_Farinotti2019_10m_old.csv')
#slope = pd.read_csv(input.hyps_filepath + 'slope_deg_01_Farinotti2019_10m_old.csv')
#width = pd.read_csv(input.hyps_filepath + 'width_km_01_Farinotti2019_10m_old.csv')
