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

def _get_rgi7id(rgi6id='', debug=False):
    """
    return RGI version 7 glacier id for a given RGI version 6 id

    """
    rgi6id = rgi6id.split('.')[0].zfill(2) + '.' + rgi6id.split('.')[1]
    # get appropriate RGI7 Id from PyGEM RGI6 Id
    rgi7_6_df = pd.read_csv(pygem_prms.oib_fp + '../oibak_rgi6_rgi7_ids.csv')
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
    

def _get_rgi6id(rgi7id='', debug=False):
    """
    return RGI version 7 glacier id for a given RGI version 6 id

    """
    rgi7id = rgi7id.split('-')[0].zfill(2) + '-' + rgi7id.split('-')[1]
    # get appropriate RGI7 Id from PyGEM RGI6 Id
    rgi7_6_df = pd.read_csv(pygem_prms.oib_fp + '../oibak_rgi6_rgi7_ids.csv')
    rgi7_6_df['rgi7id'] = rgi7_6_df['rgi7id'].str.split('RGI2000-v7.0-G-').str[1]
    rgi7_6_df['rgi6id'] = rgi7_6_df['rgi6id'].str.split('RGI60-').str[1]
    rgi6id = rgi7_6_df.loc[lambda rgi7_6_df: rgi7_6_df['rgi7id'] == rgi7id,'rgi6id'].tolist()
    if len(rgi6id)==1:
        if debug:
            print(f'RGI7:{rgi7id} -> RGI6:{rgi6id[0]}')
        return rgi6id[0]
    elif len(rgi6id)==0:
        raise IndexError(f'No matching RGI6Id for {rgi7id}')
    elif len(rgi6id)>1:
        raise IndexError(f'More than one matching RGI6Id for {rgi7id}')


def _date_check(dt_obj):
    """
    if survey date in given month <daysinmonth/2 assign it to beginning of month, else assign to beginning of next month (for consistency with monthly PyGEM timesteps)
    """
    dim = pd.Series(dt_obj).dt.daysinmonth.iloc[0]
    if dt_obj.day < dim // 2:
        dt_obj_ = datetime.datetime(year=dt_obj.year, month=dt_obj.month, day=1)
    else:
        dt_obj_ = datetime.datetime(year=dt_obj.year, month=dt_obj.month+1, day=1)
    return dt_obj_


def _load(rgi7id):
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


def _filter_on_pixel_count(arr, pctl = 15):
    """
    filter oib diffs by perntile pixel count
    """
    arr=arr.astype(float)
    arr[arr==0] = np.nan
    mask = arr < np.nanpercentile(arr,pctl)
    return mask


def _terminus_mask(survey_dates, lowest_bin, cop30_diffs, debug=False):
    """
    create mask of missing terminus ice using last oib survey
    """
    inds = range(len(survey_dates))[::-1]

    try:
        for i in inds:
            tmp = cop30_diffs[lowest_bin:lowest_bin+50,i]
            if np.isnan(tmp).all():
                continue
            else:
                # find peak we'll bake in the assumption that terminus thickness has decreased over time - we'll thus look for a trough if yr>=2013 (cop30 date)
                if survey_dates[i].year>2013:
                    idx = np.nanargmin(tmp) + lowest_bin
                else:
                    tmp = -1*tmp
                    idx = np.nanargmax(tmp) + lowest_bin
            if debug:
                plt.figure()
                cmap=plt.cm.rainbow(np.linspace(0, 1, len(inds)))
                for i in inds[::-1]:
                    plt.plot(cop30_diffs[:,i],label=f'{survey_dates[i].year}:{survey_dates[i].month}:{survey_dates[i].day}',c=cmap[i])
                plt.axvline(idx,c='k',ls=':')
                plt.legend(loc='upper right')
                plt.show()

            return(np.arange(0,idx+1,1))

    except Exception as err:
        if debug:
            print(f'_filter_terminus_missing_ice error: {err}')
        return []


def _get_diffs(oib_dict, filter_count_pctl=10, debug=False):
    """
    loop through OIB dataset, get double differences
    diffs_stacked: np.ndarray (#bins, #surveys)
    """
    seasons = list(set(oib_dict.keys()).intersection(['march','may','august']))
    cop30_diffs_list = [] # instantiate list to hold median binned survey differences from cop30
    sigmas_list = [] # also list to hold sigma_obs
    oib_dates = [] # instantiate list to hold survey dates
    for ssn in seasons:
        for yr in list(oib_dict[ssn].keys()):
            # get survey date
            doy_int = int(np.ceil(oib_dict[ssn][yr]['mean_doy']))
            dt_obj = datetime.datetime.strptime(f'{int(yr)}-{doy_int}', '%Y-%j')
            oib_dates.append(_date_check(dt_obj))
            # get survey data and filter by pixel count
            diffs = np.asarray(oib_dict[ssn][yr]['bin_vals']['bin_median_diffs_vec'])
            counts = np.asarray(oib_dict[ssn][yr]['bin_vals']['bin_count_vec'])
            mask = _filter_on_pixel_count(counts, filter_count_pctl)
            diffs[mask] = np.nan
            cop30_diffs_list.append(diffs)
            sigmas = 2*np.asarray(oib_dict[ssn][yr]['bin_vals']['bin_interquartile_range_diffs_vec']) # take 2x binned interquartile range diffs as sigma_obs
            sigmas[mask] = np.nan
            sigmas_list.append(sigmas)
    # sort by survey dates
    inds = np.argsort(oib_dates).tolist()
    oib_dates = [oib_dates[i] for i in inds]
    cop30_diffs_list = [cop30_diffs_list[i] for i in inds]
    sigmas_list = [sigmas_list[i] for i in inds]
    if debug:
        print(f'OIB survey dates:\n{", ".join([str(dt.year)+"-"+str(dt.month)+"-"+str(dt.day) for dt in oib_dates])}')
    # stack diffs
    diffs_stacked = np.column_stack(cop30_diffs_list)
    sigmas_stacked = np.column_stack(sigmas_list)
    # get bin centers
    bin_centers = (np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']) + 
                np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'])) / 2
    bin_area = oib_dict['aad_dict']['hist_bin_areas_m2']

    bin_edges = oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']
    bin_edges.append(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'][-1])
    bin_edges = np.asarray(bin_edges)

    return bin_centers, bin_area, diffs_stacked, sigmas_stacked, np.asarray(oib_dates)


def _rebin(bin_centers, bin_area, bin_diffs, bin_sigmas, agg=100):
    if agg:
        # aggregate both model and obs to 100 m bins
        nbins = int(np.ceil((bin_centers[-1] - bin_centers[0]) // agg))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            y = np.column_stack([stats.binned_statistic(x=bin_centers, values=x, statistic=np.nanmean, bins=nbins)[0] for x in bin_diffs.T])
            bin_edges = stats.binned_statistic(x=bin_centers, values=bin_diffs[:,0], statistic=np.nanmean, bins=nbins)[1]
            s = np.column_stack([stats.binned_statistic(x=bin_centers, values=x, statistic=np.nanmean, bins=bin_edges)[0] for x in bin_sigmas.T])
            bin_area  = stats.binned_statistic(x=bin_centers, values=bin_area, statistic=np.nanmean, bins=bin_edges)[0]
            bin_diffs = y
            bin_sigmas = s

        return bin_edges, bin_area, bin_diffs, bin_sigmas