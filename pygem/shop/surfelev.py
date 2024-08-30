"""
set of functions to handle binned NASA Operation IceBridge surface elevation data
bstober 20240830
"""

import os, glob, json, pickle, datetime, warnings
import datetime
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib.pyplot as plt
import pygem_input as pygem_prms


def get_rgi7id(rgi6id='', debug=False):
    """
    return RGI version 7 glacier id for a given RGI version 6 id
    """
    rgi6id = rgi6id.split('.')[0].zfill(2) + '.' + rgi6id.split('.')[1]
    # get appropriate RGI7 Id from PyGEM RGI6 Id
    rgi7_6_df = pd.read_csv(pygem_prms.rgi_fp + '/RGI2000-v7.0-G-01_alaska-rgi6_links.csv')
    rgi7_6_df['rgi7_id'] = rgi7_6_df['rgi7_id'].str.split('RGI2000-v7.0-G-').str[1]
    rgi7_6_df['rgi6_id'] = rgi7_6_df['rgi6_id'].str.split('RGI60-').str[1]
    rgi7id = rgi7_6_df.loc[lambda rgi7_6_df: rgi7_6_df['rgi6_id'] == rgi6id,'rgi7_id'].tolist()[0]
    if debug:
        print(f'RGI6:{rgi6id} -> RGI7:{rgi7id}')
    return rgi7id


def date_check(dt_obj):
    """
    if survey date in given month <daysinmonth/2 assign it to beginning of month, else assign to beginning of next month (for consistency with monthly PyGEM timesteps)
    """
    dim = pd.Series(dt_obj).dt.daysinmonth.iloc[0]
    if dt_obj.day < dim // 2:
        dt_obj_ = datetime.datetime(year=dt_obj.year, month=dt_obj.month, day=1)
    else:
        dt_obj_ = datetime.datetime(year=dt_obj.year, month=dt_obj.month+1, day=1)
    return dt_obj_


def load_oib(rgi7id):
    """
    load Operation IceBridge data
    """
    oib_fpath = glob.glob(pygem_prms.oib_fp  + f'/diffstats5_*{rgi7id}*.json')
    if len(oib_fpath)==0:
        return
    else:
        oib_fpath = oib_fpath[0]
    # load diffstats file
    with open(oib_fpath, 'rb') as f:
        oib_dict = json.load(f)
    return oib_dict


def oib_filter_on_pixel_count(arr, pctl = 15):
    """
    filter oib diffs by perntile pixel count
    """
    arr=arr.astype(float)
    arr[arr==0] = np.nan
    mask = arr < np.nanpercentile(arr,pctl)
    arr[mask] = np.nan
    return arr


def oib_terminus_mask(survey_date, cop30_diffs, debug=False):
    """
    create mask of missing terminus ice using last oib survey
    """
    try:
        # find peak we'll bake in the assumption that terminus thickness has decreased over time - we'll thus look for a trough if yr>=2013 (cop30 date)
        if survey_date.year<2013:
            arr = cop30_diffs
        else:
            arr = -1*cop30_diffs
        pk = signal.find_peaks(arr, distance=200)[0][0]
        if debug:
            plt.figure()
            plt.plot(cop30_diffs)
            plt.axvline(pk,c='r')
            plt.show()

        return(np.arange(0,pk+1,1))

    except Exception as err:
        if debug:
            print(f'_filter_terminus_missing_ice error: {err}')
    return []


def get_oib_diffs(oib_dict, aggregate=None, debug=False):
    """
    loop through OIB dataset, get double differences
    diffs_stacked: np.ndarray (#bins, #surveys)
    """
    seasons = list(set(oib_dict.keys()).intersection(['march','may','august']))
    cop30_diffs_list = [] # instantiate list to hold median binned survey differences from cop30
    oib_dates = [] # instantiate list to hold survey dates
    for ssn in seasons:
        for yr in list(oib_dict[ssn].keys()):
            # get survey date
            doy_int = int(np.ceil(oib_dict[ssn][yr]['mean_doy']))
            dt_obj = datetime.datetime.strptime(f'{int(yr)}-{doy_int}', '%Y-%j')
            oib_dates.append(date_check(dt_obj))
            # get survey data and filter by pixel count
            diffs = np.asarray(oib_dict[ssn][yr]['bin_vals']['bin_median_diffs_vec'])
            diffs = oib_filter_on_pixel_count(diffs, 15)
            cop30_diffs_list.append(diffs)
    # sort by survey dates
    inds = np.argsort(oib_dates).tolist()
    oib_dates = [oib_dates[i] for i in inds]
    cop30_diffs_list = [cop30_diffs_list[i] for i in inds]
    # filter missing ice at terminus based on last survey
    terminus_mask = oib_terminus_mask(oib_dates[-1], cop30_diffs_list[-1], debug=False)    
    if debug:
        print(f'OIB survey dates:\n{", ".join([str(dt.year)+"-"+str(dt.month)+"-"+str(dt.day) for dt in oib_dates])}')
    # do double differencing
    diffs_stacked = np.column_stack(cop30_diffs_list)
    # apply terminus mask across all surveys
    diffs_stacked[terminus_mask,:] = np.nan
    # get bin centers
    bin_centers = (np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']) + 
                np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'])) / 2
    bin_area = oib_dict['aad_dict']['hist_bin_areas_m2']

    if aggregate:
        # aggregate both model and obs to 100 m bins
        nbins = int(np.ceil((bin_centers[-1] - bin_centers[0]) // aggregate))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            y = np.column_stack([stats.binned_statistic(x=bin_centers, values=x, statistic=np.nanmean, bins=nbins)[0] for x in diffs_stacked.T])
            bin_edges = stats.binned_statistic(x=bin_centers, values=diffs_stacked[:,0], statistic=np.nanmean, bins=nbins)[1]
            bin_area  = stats.binned_statistic(x=bin_centers, values=bin_area, statistic=np.nanmean, bins=bin_edges)[0]
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            diffs_stacked = y

    return bin_centers, bin_area, diffs_stacked, pd.Series(oib_dates)







































# # Module logger
# log = logging.getLogger(__name__)

# @entity_task(log, writes=['consensus_mass'])
# def consensus_gridded(gdir, h_consensus_fp=pygem_prms.h_consensus_fp, add_mass=True, add_to_gridded=True):
#     """Bin consensus ice thickness and add total glacier mass to the given glacier directory
    
#     Updates the 'inversion_flowlines' save file and creates new consensus_mass.pkl
    
#     Parameters
#     ----------
#     gdir : :py:class:`oggm.GlacierDirectory`
#         where to write the data
#     """
#     # If binned mb data exists, then write to glacier directory
#     h_fn = h_consensus_fp + 'RGI60-' + gdir.rgi_region + '/' + gdir.rgi_id + '_thickness.tif'
#     assert os.path.exists(h_fn), 'Error: h_consensus_fullfn for ' + gdir.rgi_id + ' does not exist.'
        
#     # open consensus ice thickness estimate
#     h_dr = rasterio.open(h_fn, 'r', driver='GTiff')
#     h = h_dr.read(1).astype(rasterio.float32)
    
#     # Glacier mass [kg]
#     glacier_mass_raw = (h * h_dr.res[0] * h_dr.res[1]).sum() * pygem_prms.density_ice
# #    print(glacier_mass_raw)

#     if add_mass:
#         # Pickle data
#         consensus_fn = gdir.get_filepath('consensus_mass')
#         with open(consensus_fn, 'wb') as f:
#             pickle.dump(glacier_mass_raw, f)
        
    
#     if add_to_gridded:
#         rasterio_to_gdir(gdir, h_fn, 'consensus_h', resampling='bilinear')
#         output_fn = gdir.get_filepath('consensus_h')
#         # append the debris data to the gridded dataset
#         with rasterio.open(output_fn) as src:
#             grids_file = gdir.get_filepath('gridded_data')
#             with ncDataset(grids_file, 'a') as nc:
#                 # Mask values
#                 glacier_mask = nc['glacier_mask'][:] 
#                 data = src.read(1) * glacier_mask
#                 # Pixel area
#                 pixel_m2 = abs(gdir.grid.dx * gdir.grid.dy)
#                 # Glacier mass [kg] reprojoected (may lose or gain mass depending on resampling algorithm)
#                 glacier_mass_reprojected = (data * pixel_m2).sum() * pygem_prms.density_ice
#                 # Scale data to ensure conservation of mass during reprojection
#                 data_scaled = data * glacier_mass_raw / glacier_mass_reprojected
# #                glacier_mass = (data_scaled * pixel_m2).sum() * pygem_prms.density_ice
# #                print(glacier_mass)
                
#                 # Write data
#                 vn = 'consensus_h'
#                 if vn in nc.variables:
#                     v = nc.variables[vn]
#                 else:
#                     v = nc.createVariable(vn, 'f8', ('y', 'x', ), zlib=True)
#                 v.units = 'm'
#                 v.long_name = 'Consensus ice thicknness'
#                 v[:] = data_scaled
    

# @entity_task(log, writes=['inversion_flowlines'])
# def consensus_binned(gdir):
#     """Bin consensus ice thickness ice estimates.
    
#     Updates the 'inversion_flowlines' save file.
    
#     Parameters
#     ----------
#     gdir : :py:class:`oggm.GlacierDirectory`
#         where to write the data
#     """
#     flowlines = gdir.read_pickle('inversion_flowlines')
#     fl = flowlines[0]
    
#     assert len(flowlines) == 1, 'Error: binning debris data set up only for single flowlines at present'
    
#     # Add binned debris thickness and enhancement factors to flowlines
#     ds = xr.open_dataset(gdir.get_filepath('gridded_data'))
#     glacier_mask = ds['glacier_mask'].values
#     topo = ds['topo_smoothed'].values
#     h = ds['consensus_h'].values

#     # Only bin on-glacier values
#     idx_glac = np.where(glacier_mask == 1)
#     topo_onglac = topo[idx_glac]
#     h_onglac = h[idx_glac]

#     # Bin edges        
#     nbins = len(fl.dis_on_line)
#     z_center = (fl.surface_h[0:-1] + fl.surface_h[1:]) / 2
#     z_bin_edges = np.concatenate((np.array([topo[idx_glac].max() + 1]), 
#                                   z_center, 
#                                   np.array([topo[idx_glac].min() - 1])))
#     # Loop over bins and calculate the mean debris thickness and enhancement factor for each bin
#     h_binned = np.zeros(nbins) 
#     for nbin in np.arange(0,len(z_bin_edges)-1):
#         bin_max = z_bin_edges[nbin]
#         bin_min = z_bin_edges[nbin+1]
#         bin_idx = np.where((topo_onglac < bin_max) & (topo_onglac >= bin_min))
#         try:
#             h_binned[nbin] = h_onglac[bin_idx].mean()
#         except:
#             h_binned[nbin] = 0
            
#     fl.consensus_h = h_binned
    
#     # Overwrite pickle
#     gdir.write_pickle(flowlines, 'inversion_flowlines')
        