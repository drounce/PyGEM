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
    rgi7_6_df = pd.read_csv(pygem_prms.oib_fp + '../rgi7id_to_rgi6id.csv')
    rgi7_6_df['rgi7id'] = rgi7_6_df['rgi7id'].str.split('RGI2000-v7.0-G-').str[1]
    rgi7_6_df['rgi6id'] = rgi7_6_df['rgi6id'].str.split('RGI60-').str[1]
    rgi7id = rgi7_6_df.loc[lambda rgi7_6_df: rgi7_6_df['rgi6id'] == rgi6id,'rgi7id'].tolist()
    if len(rgi7id)==1:
        if debug:
            print(f'RGI6:{rgi6id} -> RGI7:{rgi7id[0]}')
        return rgi7id[0]
    elif len(rgi7id)==0:
        raise IndexError(f'No matching RGI7Id for {rgi6id}')
    elif len(rgi7id)>1:
        raise IndexError(f'More than one matching RGI7Id for {rgi6id}')


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
    sigmas_list = [] # also list to hold iqr
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
            sigmas_list.append(np.asarray(oib_dict[ssn][yr]['bin_vals']['bin_interquartile_range_diffs_vec']))  # take binned interquartile range diffs as sigma_obs
    # sort by survey dates
    inds = np.argsort(oib_dates).tolist()
    oib_dates = [oib_dates[i] for i in inds]
    cop30_diffs_list = [cop30_diffs_list[i] for i in inds]
    sigmas_list = [sigmas_list[i] for i in inds]
    # filter missing ice at terminus based on last survey
    terminus_mask = oib_terminus_mask(oib_dates[-1], cop30_diffs_list[-1], debug=False)
    if debug:
        print(f'OIB survey dates:\n{", ".join([str(dt.year)+"-"+str(dt.month)+"-"+str(dt.day) for dt in oib_dates])}')
    # stack diffs
    diffs_stacked = np.column_stack(cop30_diffs_list)
    sigmas_stacked = np.column_stack(sigmas_list)
    # apply terminus mask across all surveys
    diffs_stacked[terminus_mask,:] = np.nan
    sigmas_stacked[terminus_mask,:] = np.nan    
    # get bin centers
    bin_centers = (np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']) + 
                np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'])) / 2
    bin_area = oib_dict['aad_dict']['hist_bin_areas_m2']

    bin_edges = oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']
    bin_edges.append(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'][-1])
    bin_edges = np.asarray(bin_edges)

    if aggregate:
        # aggregate both model and obs to 100 m bins
        nbins = int(np.ceil((bin_centers[-1] - bin_centers[0]) // aggregate))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            y = np.column_stack([stats.binned_statistic(x=bin_centers, values=x, statistic=np.nanmean, bins=nbins)[0] for x in diffs_stacked.T])
            bin_edges = stats.binned_statistic(x=bin_centers, values=diffs_stacked[:,0], statistic=np.nanmean, bins=nbins)[1]
            s = np.column_stack([stats.binned_statistic(x=bin_centers, values=x, statistic=np.nanmean, bins=bin_edges)[0] for x in sigmas_stacked.T])
            bin_area  = stats.binned_statistic(x=bin_centers, values=bin_area, statistic=np.nanmean, bins=bin_edges)[0]
            diffs_stacked = y
            sigmas_stacked = s

    return bin_edges, bin_area, diffs_stacked, sigmas_stacked, pd.Series(oib_dates)
