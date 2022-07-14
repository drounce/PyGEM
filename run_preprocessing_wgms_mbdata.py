"""
Process the WGMS data to connect with RGIIds and evaluate potential precipitation biases

"""

# Built-in libraries
import argparse
import os
# External libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import median_abs_deviation
# Local libraries
import class_climate
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup


#%% ----- ARGUMENT PARSER -----
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    subset_winter : int
        option to process wgms winter data (1=yes, 0=no)
    estimate_kp : int
        option to estimate precipitation factors from winter data (1=yes, 0=no)
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="select pre-processing options")
    # add arguments
    parser.add_argument('-subset_winter', action='store', type=int, default=0,
                        help='option to process wgms winter data (1=yes, 0=no)')
    parser.add_argument('-estimate_kp', action='store', type=int, default=0,
                        help='option to estimate precipitation factors from winter data (1=yes, 0=no)')
    parser.add_argument('-mb_data_fill_wreg_hugonnet', action='store', type=int, default=0,
                        help='option to fill mass balance with regional stats (1=yes, 0=no)')
    parser.add_argument('-mb_data_removeFA', action='store', type=int, default=0,
                        help='option to fill mass balance with regional stats (1=yes, 0=no)')
    return parser

parser = getparser()
args = parser.parse_args()


#%% ----- INPUT DATA FOR EACH OPTION s-----
if args.subset_winter == 1 or args.estimate_kp == 1:
    # ===== WGMS DATA =====
    wgms_fp = pygem_prms.main_directory +  '/../WGMS/DOI-WGMS-FoG-2020-08/'
    wgms_eee_fn = 'WGMS-FoG-2020-08-EEE-MASS-BALANCE-POINT.csv'
    wgms_ee_fn = 'WGMS-FoG-2020-08-EE-MASS-BALANCE.csv'
    wgms_e_fn = 'WGMS-FoG-2020-08-E-MASS-BALANCE-OVERVIEW.csv'
    wgms_id_fn = 'WGMS-FoG-2020-08-AA-GLACIER_ID_LUT.csv'
    
    wgms_output_fp = pygem_prms.output_filepath + 'wgms/'
    wgms_ee_winter_fn = 'WGMS-FoG-2019-12-EE-MASS-BALANCE-winter_processed.csv'
    wgms_ee_winter_fn_subset = wgms_ee_winter_fn.replace('.csv', '-subset.csv')
    wgms_ee_winter_fn_kp = wgms_ee_winter_fn.replace('.csv', '-subset-kp.csv')
    wgms_reg_kp_stats_fn = 'WGMS-FoG-2019-12-reg_kp_summary.csv'
    subset_time_value = 20000000
    
if args.mb_data_fill_wreg_hugonnet == 1:
    # ===== HUGONNET GEODETIC DATA =====
    hugonnet_fp = pygem_prms.main_directory + '/../DEMs/Hugonnet2020/'
    hugonnet_fn = 'df_pergla_global_20yr.csv'

if args.mb_data_removeFA == 1:
    # ===== WILL CALVING DATA =====
    # Calving data
    will_fp = pygem_prms.main_directory + '/../calving_data/'
    will_fn = 'Northern_hemisphere_calving_flux_Kochtitzky_et_al_for_David_Rounce_with_melt_v13.csv'
    
    will_supplement_fn = 'Table_S2_Northern_hemisphere_frontal_ablation_Kochtitzky_et_al_v12.csv'
    
    # Calving glaciers with multiple RGIIds combined together in Will's analysis
    fa_multiple_glac_fp = pygem_prms.main_directory + '/../calving_data/final_layers/'
    fa_multiple_glac_fn = 'Multiple_glaciers_with_one_front_RGI_codes_speadsheet.csv'
    
    debug=True

#%% ----- PROCESS WINTER DATA -----
if args.subset_winter == 1:
    # Load data 
    wgms_e_df = pd.read_csv(wgms_fp + wgms_e_fn, encoding='unicode_escape')
    wgms_ee_df_raw = pd.read_csv(wgms_fp + wgms_ee_fn, encoding='unicode_escape')
    wgms_eee_df_raw = pd.read_csv(wgms_fp + wgms_eee_fn, encoding='unicode_escape')
    wgms_id_df = pd.read_csv(wgms_fp + wgms_id_fn, encoding='unicode_escape')
    
    # Map dictionary
    wgms_id_dict = dict(zip(wgms_id_df.WGMS_ID, wgms_id_df.RGI_ID))
    wgms_ee_df_raw['rgiid_raw'] = wgms_ee_df_raw.WGMS_ID.map(wgms_id_dict)
    wgms_ee_df_raw = wgms_ee_df_raw.dropna(subset=['rgiid_raw'])
    wgms_eee_df_raw['rgiid_raw'] = wgms_eee_df_raw.WGMS_ID.map(wgms_id_dict)
    wgms_eee_df_raw = wgms_eee_df_raw.dropna(subset=['rgiid_raw'])
    
    # Link RGIv5.0 with RGIv6.0
    rgi60_fp = pygem_prms.main_directory +  '/../RGI/rgi60/00_rgi60_attribs/'
    rgi50_fp = pygem_prms.main_directory +  '/../RGI/00_rgi50_attribs/'
    
    # Process each region
    regions_str = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']
    rgi60_df = None
    rgi50_df = None
    for reg_str in regions_str:
        # RGI60 data
        for i in os.listdir(rgi60_fp):
            if i.startswith(reg_str) and i.endswith('.csv'):
                rgi60_df_reg = pd.read_csv(rgi60_fp + i, encoding='unicode_escape')
        # append datasets
        if rgi60_df is None:
            rgi60_df = rgi60_df_reg
        else:
            rgi60_df = pd.concat([rgi60_df, rgi60_df_reg], axis=0)
    
        # RGI50 data
        for i in os.listdir(rgi50_fp):
            if i.startswith(reg_str) and i.endswith('.csv'):
                rgi50_df_reg = pd.read_csv(rgi50_fp + i, encoding='unicode_escape')
        # append datasets
        if rgi50_df is None:
            rgi50_df = rgi50_df_reg
        else:
            rgi50_df = pd.concat([rgi50_df, rgi50_df_reg], axis=0)
        
    # Merge based on GLIMSID
    glims_rgi50_dict = dict(zip(rgi50_df.GLIMSId, rgi50_df.RGIId))
    rgi60_df['RGIId_50'] = rgi60_df.GLIMSId.map(glims_rgi50_dict)
    rgi60_df_4dict = rgi60_df.dropna(subset=['RGIId_50'])
    rgi50_rgi60_dict = dict(zip(rgi60_df_4dict.RGIId_50, rgi60_df_4dict.RGIId))
    rgi60_self_dict = dict(zip(rgi60_df.RGIId, rgi60_df.RGIId))
    rgi50_rgi60_dict.update(rgi60_self_dict)
    
    # Add RGIId for version 6 to WGMS
    wgms_ee_df_raw['rgiid'] = wgms_ee_df_raw.rgiid_raw.map(rgi50_rgi60_dict)
    wgms_eee_df_raw['rgiid'] = wgms_eee_df_raw.rgiid_raw.map(rgi50_rgi60_dict)
    
    # Drop points without data
    wgms_ee_df = wgms_ee_df_raw.dropna(subset=['rgiid'])
    wgms_eee_df = wgms_eee_df_raw.dropna(subset=['rgiid'])
    
    # Winter balances only
    wgms_ee_df_winter = wgms_ee_df.dropna(subset=['WINTER_BALANCE'])
    wgms_ee_df_winter = wgms_ee_df_winter.sort_values('rgiid')
    wgms_ee_df_winter.reset_index(inplace=True, drop=True)
    
    # Add the winter time period using the E-MASS-BALANCE-OVERVIEW file
    wgms_e_cns2add = []
    for cn in wgms_e_df.columns:
        if cn not in wgms_ee_df_winter.columns:
            wgms_e_cns2add.append(cn)
            wgms_ee_df_winter[cn] = np.nan
            
    for nrow in np.arange(wgms_ee_df_winter.shape[0]):
        if nrow%500 == 0:
            print(nrow, 'of', wgms_ee_df_winter.shape[0])
        name = wgms_ee_df_winter.loc[nrow,'NAME']
        wgmsid = wgms_ee_df_winter.loc[nrow,'WGMS_ID']
        year = wgms_ee_df_winter.loc[nrow,'YEAR']
        
        try:
            e_idx = np.where((wgms_e_df['NAME'] == name) & 
                             (wgms_e_df['WGMS_ID'] == wgmsid) & 
                             (wgms_e_df['Year'] == year))[0][0]
        except:
            e_idx = None
        
        if e_idx is not None:
            wgms_ee_df_winter.loc[nrow,wgms_e_cns2add] = wgms_e_df.loc[e_idx,wgms_e_cns2add]
    
    # Export data
    if not os.path.exists(wgms_output_fp):
        os.makedirs(wgms_output_fp)
    wgms_ee_df_winter.to_csv(wgms_output_fp + wgms_ee_winter_fn, index=False)
                
    # Export subset of data
    wgms_ee_df_winter_subset = wgms_ee_df_winter.loc[wgms_ee_df_winter['BEGIN_PERIOD'] > subset_time_value]
    wgms_ee_df_winter_subset = wgms_ee_df_winter_subset.dropna(subset=['END_WINTER'])
    wgms_ee_df_winter_subset.to_csv(wgms_output_fp + wgms_ee_winter_fn_subset, index=False)


#%% ----- WINTER PRECIPITATION COMPARISON -----
if args.estimate_kp == 1:
    # Load data
    assert os.path.exists(wgms_output_fp + wgms_ee_winter_fn_subset), 'wgms_ee_winter_fn_subset does not exist!'
    wgms_df = pd.read_csv(wgms_output_fp + wgms_ee_winter_fn_subset, encoding='unicode_escape')
    
    # Process dates
    wgms_df.loc[:,'BEGIN_PERIOD'] = wgms_df.loc[:,'BEGIN_PERIOD'].values.astype(np.int).astype(str)
    wgms_df['BEGIN_YEAR'] = [int(x[0:4]) for x in wgms_df.loc[:,'BEGIN_PERIOD']]
    wgms_df['BEGIN_MONTH'] = [int(x[4:6]) for x in list(wgms_df.loc[:,'BEGIN_PERIOD'])]
    wgms_df['BEGIN_DAY'] = [int(x[6:]) for x in list(wgms_df.loc[:,'BEGIN_PERIOD'])]
    wgms_df['BEGIN_YEARMONTH'] = [x[0:6] for x in list(wgms_df.loc[:,'BEGIN_PERIOD'])]
    wgms_df.loc[:,'END_WINTER'] = wgms_df.loc[:,'END_WINTER'].values.astype(np.int).astype(str)
    wgms_df['END_YEAR'] = [int(x[0:4]) for x in wgms_df.loc[:,'END_WINTER']]
    wgms_df['END_MONTH'] = [int(x[4:6]) for x in list(wgms_df.loc[:,'END_WINTER'])]
    wgms_df['END_DAY'] = [int(x[6:]) for x in list(wgms_df.loc[:,'END_WINTER'])]
    wgms_df['END_YEARMONTH'] = [x[0:6] for x in list(wgms_df.loc[:,'END_WINTER'])]
    
    # ===== PROCESS UNIQUE GLACIERS =====
    rgiids_unique = list(wgms_df['rgiid'].unique())
    glac_no = [x.split('-')[1] for x in rgiids_unique]
    
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    
    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear, spinupyears=0,
            option_wateryear=pygem_prms.gcm_wateryear)
    dates_table_yearmo = [str(dates_table.loc[x,'year']) + str(dates_table.loc[x,'month']).zfill(2) 
                          for x in range(dates_table.shape[0])]
    
    # ===== LOAD CLIMATE DATA =====
    # Climate class
    gcm = class_climate.GCM(name=pygem_prms.ref_gcm_name)
    
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi,
                                                                  dates_table)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi,
                                                                  dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1]
    if pygem_prms.use_constant_lapserate:
        gcm_lr = np.zeros(gcm_temp.shape) + pygem_prms.lapserate
    else:
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    
    # ===== PROCESS THE OBSERVATIONS ======
    prec_cn = pygem_prms.ref_gcm_name + '_prec'
    wgms_df[prec_cn] = np.nan
    wgms_df['kp'] = np.nan
    wgms_df['ndays'] = np.nan
    for glac in range(main_glac_rgi.shape[0]):
        print(glac, main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])
        rgiid = glacier_rgi_table.RGIId
        
        wgms_df_single = (wgms_df.loc[wgms_df['rgiid'] == rgiid]).copy()
        glac_idx = wgms_df_single.index.values
        wgms_df_single.reset_index(inplace=True, drop=True)
        
        wgms_df_single[prec_cn] = np.nan
        for nobs in range(wgms_df_single.shape[0]):
            
            # Only process good data
            # - dates are provided and real
            # - spans more than one month
            # - positive winter balance (since we don't account for melt)
            if ((wgms_df_single.loc[nobs,'BEGIN_MONTH'] >= 1 and wgms_df_single.loc[nobs,'BEGIN_MONTH'] <= 12) and
                (wgms_df_single.loc[nobs,'BEGIN_DAY'] >= 1 and wgms_df_single.loc[nobs,'BEGIN_DAY'] <= 31) and
                (wgms_df_single.loc[nobs,'END_MONTH'] >= 1 and wgms_df_single.loc[nobs,'END_MONTH'] <= 12) and
                (wgms_df_single.loc[nobs,'END_DAY'] >= 1 and wgms_df_single.loc[nobs,'END_DAY'] <= 31) and
                (wgms_df_single.loc[nobs,'BEGIN_PERIOD'] < wgms_df_single.loc[nobs,'END_WINTER']) and
                (wgms_df_single.loc[nobs,'BEGIN_YEARMONTH'] != wgms_df_single.loc[nobs,'END_YEARMONTH']) and
                (wgms_df_single.loc[nobs,'WINTER_BALANCE'] > 0)
                ):
                # Begin index
                idx_begin = dates_table_yearmo.index(wgms_df_single.loc[nobs,'BEGIN_YEARMONTH'])
                idx_end = dates_table_yearmo.index(wgms_df_single.loc[nobs,'END_YEARMONTH'])
                
                # Fraction of the months to remove
                remove_prec_begin = (gcm_prec[glac,idx_begin] * 
                                     wgms_df_single.loc[nobs,'BEGIN_DAY'] / dates_table.loc[idx_begin,'daysinmonth'])
                remove_prec_end = (gcm_prec[glac,idx_end] * 
                                   (1 - wgms_df_single.loc[nobs,'END_DAY'] / dates_table.loc[idx_end,'daysinmonth']))
                
                # Winter Precipitation
                gcm_prec_winter = gcm_prec[glac,idx_begin:idx_end+1].sum() - remove_prec_begin - remove_prec_end
                wgms_df_single.loc[nobs,prec_cn] = gcm_prec_winter
                
                # Number of days
                ndays = (dates_table.loc[idx_begin:idx_end,'daysinmonth'].sum() - wgms_df_single.loc[nobs,'BEGIN_DAY']
                         - (dates_table.loc[idx_end,'daysinmonth'] - wgms_df_single.loc[nobs,'END_DAY']))
                wgms_df_single.loc[nobs,'ndays'] = ndays
            
        # Estimate precipitation factors
        # - assumes no melt and all snow (hence a convservative/underestimated estimate)
        wgms_df_single['kp'] = wgms_df_single['WINTER_BALANCE'] / 1000 / wgms_df_single[prec_cn]
    
        # Record precipitation, precipitation factors, and number of days in main dataframe
        wgms_df.loc[glac_idx,prec_cn] = wgms_df_single[prec_cn].values
        wgms_df.loc[glac_idx,'kp'] = wgms_df_single['kp'].values
        wgms_df.loc[glac_idx,'ndays'] = wgms_df_single['ndays'].values
        
    # Drop nan values
    wgms_df_wkp = wgms_df.dropna(subset=['kp']).copy()
    wgms_df_wkp.reset_index(inplace=True, drop=True)
    if not os.path.exists(wgms_output_fp):
        os.makedirs(wgms_output_fp)
    wgms_df_wkp.to_csv(wgms_output_fp + wgms_ee_winter_fn_kp, index=False)

    # Calculate stats for all and each region
    wgms_df_wkp['reg'] = [x.split('-')[1].split('.')[0] for x in wgms_df_wkp['rgiid'].values]
    reg_unique = list(wgms_df_wkp['reg'].unique())
    
    # Output dataframe
    reg_kp_cns = ['region', 'count_obs', 'count_glaciers', 'kp_mean', 'kp_std', 'kp_med', 'kp_nmad', 'kp_min', 'kp_max']
    reg_kp_df = pd.DataFrame(np.zeros((len(reg_unique)+1,len(reg_kp_cns))), columns=reg_kp_cns)
    
    # Only those with at least 1 month of data
    wgms_df_wkp = wgms_df_wkp.loc[wgms_df_wkp['ndays'] >= 30]
    
    # All stats
    reg_kp_df.loc[0,'region'] = 'all'
    reg_kp_df.loc[0,'count_obs'] = wgms_df_wkp.shape[0]
    reg_kp_df.loc[0,'count_glaciers'] = len(wgms_df_wkp['rgiid'].unique())
    reg_kp_df.loc[0,'kp_mean'] = np.mean(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_std'] = np.std(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_med'] = np.median(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_nmad'] = median_abs_deviation(wgms_df_wkp.kp.values, scale='normal')
    reg_kp_df.loc[0,'kp_min'] = np.min(wgms_df_wkp.kp.values)
    reg_kp_df.loc[0,'kp_max'] = np.max(wgms_df_wkp.kp.values)
    
    # Regional stats
    for nreg, reg in enumerate(reg_unique):
        wgms_df_wkp_reg = wgms_df_wkp.loc[wgms_df_wkp['reg'] == reg]
        
        reg_kp_df.loc[nreg+1,'region'] = reg
        reg_kp_df.loc[nreg+1,'count_obs'] = wgms_df_wkp_reg.shape[0]
        reg_kp_df.loc[nreg+1,'count_glaciers'] = len(wgms_df_wkp_reg['rgiid'].unique())
        reg_kp_df.loc[nreg+1,'kp_mean'] = np.mean(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_std'] = np.std(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_med'] = np.median(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_nmad'] = median_abs_deviation(wgms_df_wkp_reg.kp.values, scale='normal')
        reg_kp_df.loc[nreg+1,'kp_min'] = np.min(wgms_df_wkp_reg.kp.values)
        reg_kp_df.loc[nreg+1,'kp_max'] = np.max(wgms_df_wkp_reg.kp.values)
        
        
        print('region', reg)
        print('  count:', wgms_df_wkp_reg.shape[0])
        print('  glaciers:', len(wgms_df_wkp_reg['rgiid'].unique()))
        print('  mean:', np.mean(wgms_df_wkp_reg.kp.values))
        print('  std :', np.std(wgms_df_wkp_reg.kp.values))
        print('  med :', np.median(wgms_df_wkp_reg.kp.values))
        print('  nmad:', median_abs_deviation(wgms_df_wkp_reg.kp.values, scale='normal'))
        print('  min :', np.min(wgms_df_wkp_reg.kp.values))
        print('  max :', np.max(wgms_df_wkp_reg.kp.values))
        
    reg_kp_df.to_csv(wgms_output_fp + wgms_reg_kp_stats_fn, index=False)
    

#%% ----- FILL MASS BALANCE DATASET WITH REGIONAL STATISTICS -----
if args.mb_data_fill_wreg_hugonnet == 1:
    print('Filling in missing data with regional estimates...')
    
    #%%
#    hugonnet_rgi_glacno_cn = 'rgiid'
#    hugonnet_mb_cn = 'dmdtda'
#    hugonnet_mb_err_cn = 'err_dmdtda'
#    hugonnet_time1_cn = 't1'
#    hugonnet_time2_cn = 't2'
#    hugonnet_area_cn = 'area_km2'
    
    df_fp = hugonnet_fp
    df_fn = hugonnet_fn
    
    # Load mass balance measurements and identify unique rgi regions 
    df = pd.read_csv(df_fp + df_fn)
    df = df.rename(columns={"rgiid": "RGIId"})
    
    # Load glaciers
    rgiids = [x for x in df.RGIId.values if x.startswith('RGI60-')]
    glac_no = [x.split('-')[1] for x in rgiids]
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)
    main_glac_rgi['O1Region'] = [int(x) for x in main_glac_rgi['O1Region']]
    
#%%
    # Regions with data
    dict_rgi_regionsO1 = dict(zip(main_glac_rgi.RGIId, main_glac_rgi.O1Region))
    df['O1Region'] = df.RGIId.map(dict_rgi_regionsO1)
    rgi_regionsO1 = sorted(df['O1Region'].unique().tolist())
    rgi_regionsO1 = [int(x) for x in rgi_regionsO1 if np.isnan(x) == False]

    # Add mass balance and uncertainty to main_glac_rgi
    dict_rgi_mb = dict(zip(df.RGIId, df.dmdtda))
    dict_rgi_mb_sigma = dict(zip(df.RGIId, df.err_dmdtda))
    dict_rgi_area = dict(zip(df.RGIId, df.area))
    main_glac_rgi['mb_mwea'] = main_glac_rgi.RGIId.map(dict_rgi_mb)
    main_glac_rgi['mb_mwea_sigma'] = main_glac_rgi.RGIId.map(dict_rgi_mb_sigma)
    main_glac_rgi['area_hugonnet'] = main_glac_rgi.RGIId.map(dict_rgi_area)
    
    def weighted_avg_and_std(values, weights):
        """
        Return the weighted average and standard deviation.
    
        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values-average)**2, weights=weights)
        return average, variance**0.5
    
    #%%
    all_sigma_mean = main_glac_rgi['mb_mwea_sigma'].mean()
    all_sigma_std = main_glac_rgi['mb_mwea_sigma'].std()
    all_sigma_threshold = all_sigma_mean + 3 * all_sigma_std
    
    print('all sigma threshold:', np.round(all_sigma_threshold,2))
    
    main_glac_rgi_filled = main_glac_rgi.copy()
    df_filled = df.copy()
    for reg in rgi_regionsO1:
        main_glac_rgi_subset = main_glac_rgi.loc[main_glac_rgi.O1Region == reg, :]
        
        # Too high of sigma causes large issues for model
        #  sigma theoretically should be independent of region
        reg_sigma_mean = main_glac_rgi_subset['mb_mwea_sigma'].mean()
        reg_sigma_std = main_glac_rgi_subset['mb_mwea_sigma'].std()
        reg_sigma_threshold = reg_sigma_mean + 3 * reg_sigma_std
        # Don't penalize regions that are well-measured, so use all threshold as minimum
        if reg_sigma_threshold < all_sigma_threshold:
            reg_sigma_threshold = all_sigma_threshold
        
        rm_idx = main_glac_rgi_subset.loc[main_glac_rgi_subset.mb_mwea_sigma > reg_sigma_threshold,:].index.values
        main_glac_rgi_filled.loc[rm_idx,'mb_mwea'] = np.nan
        main_glac_rgi_filled.loc[rm_idx,'mb_mwea_sigma'] = np.nan
        
        rgi_subset_good = main_glac_rgi_subset.loc[main_glac_rgi_subset['mb_mwea_sigma'] <= reg_sigma_threshold,:]
        
        reg_mb_mean, reg_mb_std = weighted_avg_and_std(rgi_subset_good.mb_mwea, rgi_subset_good.area_hugonnet)
        
        print(reg, np.round(reg_sigma_threshold,2), 'exclude:', len(rm_idx),
              '  mb mean/std:', np.round(reg_mb_mean,2), np.round(reg_mb_std,2))
        
        # Replace nan values
        nan_idx = main_glac_rgi_filled.loc[np.isnan(main_glac_rgi_filled.mb_mwea) & 
                                           (main_glac_rgi_filled.O1Region == reg), :].index.values
                                           
        main_glac_rgi_filled.loc[nan_idx,'mb_mwea'] = reg_mb_mean
        main_glac_rgi_filled.loc[nan_idx,'mb_mwea_sigma'] = reg_mb_std
        
    # Map back onto original dataset
    dict_rgi_mb_filled_mean = dict(zip(main_glac_rgi_filled.RGIId, main_glac_rgi_filled.mb_mwea))
    dict_rgi_mb_filled_sigma = dict(zip(main_glac_rgi_filled.RGIId, main_glac_rgi_filled.mb_mwea_sigma))
    df_filled['mb_mwea'] = df.RGIId.map(dict_rgi_mb_filled_mean)
    df_filled['mb_mwea_err'] = df.RGIId.map(dict_rgi_mb_filled_sigma)        
    
    # Export dataset
    df_filled.to_csv(df_fp + df_fn.replace('.csv','-filled.csv'), index=False)
    
    #%%
    # ----- REPLACE REGION 12 -----
    #  - GLIMSId and RGIId were connected using join by location with greatest overlapping area in QGIS
    df_filled = pd.read_csv(df_fp + df_fn.replace('.csv','-filled.csv'))
    shp_df = pd.read_csv('/Users/drounce/Documents/HiMAT/DEMs/Hugonnet2020/12_rgi60_wromain_mb.csv')
    mb_df = df_filled.copy()

    glac_dict = dict(zip(shp_df['glac_id'], shp_df['RGIId']))
    glac_dict_df = pd.DataFrame.from_dict(glac_dict, orient='index')
    #glac_dict_df.to_csv('/Users/drounce/Documents/HiMAT/DEMs/Hugonnet2020/12_GLIMSId_RGIId_dict.csv')
    mb_df['rgiid-rgi60'] = mb_df['RGIId'].map(glac_dict)
    mb_df = mb_df.dropna(axis=0, subset=['rgiid-rgi60', 'dmdtda'])
    mb_df.reset_index(inplace=True, drop=True)
    
    # Load glaciers
    glac_no_12 = [x.split('-')[1] for x in shp_df['RGIId'].values]
    main_glac_rgi_12 = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=['12'], rgi_regionsO2='all', rgi_glac_number='all')
    main_glac_rgi_12['O1Region'] = [int(x) for x in main_glac_rgi_12['O1Region']]

    # Add mass balance and uncertainty to main_glac_rgi
    mb_df['RGIId'] = mb_df['rgiid-rgi60']
    dict_rgi_mb_12 = dict(zip(mb_df.RGIId, mb_df.dmdtda))
    dict_rgi_mb_sigma_12 = dict(zip(mb_df.RGIId, mb_df.err_dmdtda))
    dict_rgi_area_12 = dict(zip(mb_df.RGIId, mb_df.area))
    main_glac_rgi_12['mb_mwea'] = main_glac_rgi_12.RGIId.map(dict_rgi_mb_12)
    main_glac_rgi_12['mb_mwea_sigma'] = main_glac_rgi_12.RGIId.map(dict_rgi_mb_sigma_12)
    main_glac_rgi_12['area_hugonnet'] = main_glac_rgi_12.RGIId.map(dict_rgi_area_12)
    
    
    print('all sigma threshold:', np.round(all_sigma_threshold,2))
    
    
    main_glac_rgi_filled_12 = main_glac_rgi_12.copy()
    idx_12 = [x for x in df.index.values if df.loc[x,'RGIId'].startswith('G')]
    df_filled_12 = df.loc[idx_12,:].copy()

    for reg in [12]:
        main_glac_rgi_subset = main_glac_rgi_12.loc[main_glac_rgi_12.O1Region == reg, :]

        # Too high of sigma causes large issues for model
        #  sigma theoretically should be independent of region
        reg_sigma_mean = main_glac_rgi_subset['mb_mwea_sigma'].mean(skipna=True)
        reg_sigma_std = main_glac_rgi_subset['mb_mwea_sigma'].std(skipna=True)
        reg_sigma_threshold = reg_sigma_mean + 3 * reg_sigma_std
        # Don't penalize regions that are well-measured, so use all threshold as minimum
        if reg_sigma_threshold < all_sigma_threshold:
            reg_sigma_threshold = all_sigma_threshold
        
        rm_idx = main_glac_rgi_subset.loc[main_glac_rgi_subset.mb_mwea_sigma > reg_sigma_threshold,:].index.values
        main_glac_rgi_filled.loc[rm_idx,'mb_mwea'] = np.nan
        main_glac_rgi_filled.loc[rm_idx,'mb_mwea_sigma'] = np.nan
        
        rgi_subset_good = main_glac_rgi_subset.loc[main_glac_rgi_subset['mb_mwea_sigma'] <= reg_sigma_threshold,:]
        
        reg_mb_mean, reg_mb_std = weighted_avg_and_std(rgi_subset_good.mb_mwea, rgi_subset_good.area_hugonnet)
        
        print(reg, np.round(reg_sigma_threshold,2), 'exclude:', len(rm_idx),
              '  mb mean/std:', np.round(reg_mb_mean,2), np.round(reg_mb_std,2))
        
        # Replace nan values
        nan_idx = main_glac_rgi_filled_12.loc[np.isnan(main_glac_rgi_filled_12.mb_mwea) & 
                                              (main_glac_rgi_filled_12.O1Region == reg), :].index.values
                                           
        main_glac_rgi_filled_12.loc[nan_idx,'mb_mwea'] = reg_mb_mean
        main_glac_rgi_filled_12.loc[nan_idx,'mb_mwea_sigma'] = reg_mb_std
        
    # Map back onto original dataset
    df_filled_12['rgiid-rgi60'] = df_filled_12['RGIId'].map(glac_dict)
    df_filled_12['RGIId'] = df_filled_12['rgiid-rgi60']
    df_filled_12_nonan = df_filled_12.dropna(axis=0, subset=['RGIId', 'dmdtda'])
    
    # Create Region 12 dataframe
    df_filled_12_all = pd.DataFrame(np.zeros((main_glac_rgi_filled_12.shape[0], df_filled.shape[1])), columns=df_filled.columns)
    df_filled_12_all['RGIId'] = main_glac_rgi_filled_12['RGIId']
    df_filled_12_all['area'] = main_glac_rgi_filled_12['Area']
    df_filled_12_all['lat'] = main_glac_rgi_filled_12['CenLat']
    df_filled_12_all['lon'] = main_glac_rgi_filled_12['CenLon']
    df_filled_12_all['period'] = '2000-01-01_2020-01-01'
    df_filled_12_all['reg'] = 12
    df_filled_12_all['t1'] = '2000-01-01'
    df_filled_12_all['t2'] = '2020-01-01'
    df_filled_12_all['O1Region'] = 12
    dict_c1 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.valid_obs))
    df_filled_12_all['valid_obs'] = df_filled_12_all['RGIId'].map(dict_c1)
    dict_c2 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.perc_area_meas))
    df_filled_12_all['perc_area_meas'] = df_filled_12_all['RGIId'].map(dict_c2)
    dict_c3 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.err_cont))
    df_filled_12_all['err_cont'] = df_filled_12_all['RGIId'].map(dict_c3)
    dict_c4 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.perc_err_cont))
    df_filled_12_all['perc_err_cont'] = df_filled_12_all['RGIId'].map(dict_c4)
    dict_c5 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.dmdtda))
    df_filled_12_all['dmdtda'] = df_filled_12_all['RGIId'].map(dict_c5)
    dict_c6 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.err_dmdtda))
    df_filled_12_all['err_dmdtda'] = df_filled_12_all['RGIId'].map(dict_c6)
    dict_c7 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.dhdt))
    df_filled_12_all['dhdt'] = df_filled_12_all['RGIId'].map(dict_c7)
    dict_c8 = dict(zip(df_filled_12_nonan.RGIId, df_filled_12_nonan.err_dhdt))
    df_filled_12_all['err_dhdt'] = df_filled_12_all['RGIId'].map(dict_c8)
    dict_rgi_mb_filled_mean_12 = dict(zip(main_glac_rgi_filled_12.RGIId, main_glac_rgi_filled_12.mb_mwea))
    dict_rgi_mb_filled_sigma_12 = dict(zip(main_glac_rgi_filled_12.RGIId, main_glac_rgi_filled_12.mb_mwea_sigma))
    df_filled_12_all['mb_mwea'] = df_filled_12_all.RGIId.map(dict_rgi_mb_filled_mean_12)
    df_filled_12_all['mb_mwea_err'] = df_filled_12_all.RGIId.map(dict_rgi_mb_filled_sigma_12)
    
    # Append Region 12 dataframe
    df_filled_all = df_filled.append(df_filled_12_all)
    df_filled_all.reset_index(inplace=True, drop=True)

    # Remove Region 12 GLIMS
    idx_rgi = list(np.where(df_filled_all.RGIId.str.startswith('RGI60').values)[0])
    df_filled_all = df_filled_all.loc[idx_rgi,:].copy()

    df_filled_all = df_filled_all.sort_values('RGIId')
    df_filled_all.reset_index(inplace=True, drop=True)

    # Export dataset
    df_filled_all.to_csv(df_fp + df_fn.replace('.csv','-filled.csv'), index=False)


#%% ===== REMOVE FRONTAL ABLATION FROM MB DATASETS =====
if args.mb_data_removeFA == 1:
    mb_data_df = pd.read_csv(pygem_prms.hugonnet_fp + pygem_prms.hugonnet_fn)
    mb_data_df['mb_mwea_romain'] = mb_data_df['mb_mwea'].copy() 
    mb_data_df['mb_mwea_err_romain'] = mb_data_df['mb_mwea_err'].copy() 
    fa_multiple_glac_df = pd.read_csv(fa_multiple_glac_fp + fa_multiple_glac_fn)

    # Load supplemental data that has additional information on quality
    fa_data_df = pd.read_csv(will_fp + will_fn)
    fa_data_supp_df = pd.read_csv(will_fp + will_supplement_fn, skiprows=24, header=1)
    fa_data_supp_df = fa_data_supp_df.drop(axis=0, index=0)
    fa_data_supp_df.reset_index(inplace=True, drop=True)
    supp_rgiids_list = list(fa_data_supp_df.RGI_Id)
    
    fa_gta_cn = 'Frontal_ablation_2000_to_2020_gt_per_yr_mean'
    fa_gta_err_cn = 'Frontal_ablation_2000_to_2020_gt_per_yr_mean_err'
    fa_gta_term_cn = 'terminus_gt_change_per_year_total_without_melt'
    
    def mwea_to_gta(mwea, area_m2):
        return mwea * pygem_prms.density_water * area_m2 / 1e12
    def gta_to_mwea(gta, area_m2):
        """ area in m2 """
        return gta * 1e12 / pygem_prms.density_water / area_m2

    fa_data_df['fa_mwea'] = np.nan
    fa_data_df['fa_mwea_err'] = np.nan
    fa_data_df['Romain_mwea_raw'] = np.nan
    fa_data_df['Romain_mwea_raw_err'] = np.nan
    fa_data_df['Romain_area_km2'] = np.nan
    fa_data_df['Romain_gta_raw'] = np.nan
    fa_data_df['Romain_gta_raw_err'] = np.nan
    fa_data_df['Romain_gta_mbtot'] = np.nan
    fa_data_df['Romain_gta_mbtot_err'] = np.nan
    fa_data_df['Romain_gta_mbclim'] = np.nan
    fa_data_df['Romain_gta_mbclim_err'] = np.nan
    fa_data_df['Romain_mwea_mbtot'] = np.nan
    fa_data_df['Romain_mwea_mbtot_err'] = np.nan
    fa_data_df['Romain_mwea_mbclim'] = np.nan
    fa_data_df['Romain_mwea_mbclim_err'] = np.nan
    fa_data_df['thick_measured_yn'] = np.nan
    mb_rgiids_list = list(mb_data_df.RGIId.values)
    
    # RGIIds
    rgiids_wmultiples = np.unique(fa_multiple_glac_df.RGIid_for_frontal_ablation)
    rgiids_fa_data = list(fa_data_df.RGIId)
    rgiids_mb_data = list(mb_data_df.RGIId)
    
    #%%
    for nglac, rgiid in enumerate(rgiids_fa_data):
        
        for batman in [0]:
#        if rgiid in ['RGI60-01.10689']:
            
            if debug:
                print('\n' + rgiid)
            
            # Aggregate data from multiple glaciers if needed, since Will's processing included multiple glaciers sometimes
            if rgiid in rgiids_wmultiples:
                #    for rgiid in rgiids_wmultiples[1:2]:
                
                rgiids_multiple_list = list(
                        fa_multiple_glac_df.loc[fa_multiple_glac_df['RGIid_for_frontal_ablation']==rgiid,'RGIId'].values)
            
                # Combine mass balance from both glaciers, remove calving, and set both to be average
                fa_idx = rgiids_fa_data.index(rgiid)
                fa_gta = fa_data_df.loc[fa_idx,fa_gta_cn]
                fa_gta_err = fa_data_df.loc[fa_idx,fa_gta_err_cn]
                mb_gta_list = []
                mb_gta_err_list = []
                area_m2_list = []
                for rgiid_single in rgiids_multiple_list:
                    mb_idx = rgiids_mb_data.index(rgiid_single)
                    
    #                print(rgiid_single, mb_data_df.loc[mb_idx,'mb_mwea_romain'], mb_data_df.loc[mb_idx,'area'])
                    
                    mb_gta_single = mwea_to_gta(mb_data_df.loc[mb_idx,'mb_mwea_romain'], 
                                                mb_data_df.loc[mb_idx,'area'] * 1e6)
                    mb_gta_err_single = mwea_to_gta(mb_data_df.loc[mb_idx,'mb_mwea_err_romain'],
                                                    mb_data_df.loc[mb_idx,'area'] * 1e6)                    
                    mb_gta_list.append(mb_gta_single)
                    mb_gta_err_list.append(mb_gta_err_single)
                    area_m2_list.append(mb_data_df.loc[mb_idx,'area'] * 1e6)
    
                mb_gta = np.array(mb_gta_list).sum()
                mb_gta_err = (np.array(mb_gta_err_list)**2).sum()**0.5
                area_m2 = np.array(area_m2_list).sum()
                
                mb_mwea = gta_to_mwea(mb_gta, area_m2)
                mb_mwea_err = gta_to_mwea(mb_gta_err, area_m2)
                
                fa_mwea = gta_to_mwea(fa_gta, area_m2)
                fa_mwea_err = gta_to_mwea(fa_gta_err, area_m2)
                
            # Otherwise load individual glacier data
            else:
                mb_idx = mb_rgiids_list.index(rgiid)
            
                # Mass balance
                mb_mwea = mb_data_df.loc[mb_idx,pygem_prms.hugonnet_mb_cn]
                mb_mwea_err = mb_data_df.loc[mb_idx,pygem_prms.hugonnet_mb_err_cn]
                area_m2 = mb_data_df.loc[mb_idx,'area'] * 1e6
                fa_data_df.loc[nglac,'Romain_mwea_raw'] = mb_data_df.loc[mb_idx,pygem_prms.hugonnet_mb_cn]
                fa_data_df.loc[nglac,'Romain_mwea_raw_err'] = mb_data_df.loc[mb_idx,pygem_prms.hugonnet_mb_err_cn]
            
                # Frontal Ablation (gta)
                fa_gta = fa_data_df.loc[nglac,fa_gta_cn]
                fa_gta_err = fa_data_df.loc[nglac,fa_gta_err_cn]
                # convert to mwea
                fa_mwea = gta_to_mwea(fa_gta, area_m2)
                fa_mwea_err = gta_to_mwea(fa_gta_err, area_m2)
                fa_data_df.loc[nglac,'fa_mwea'] = fa_mwea
                fa_data_df.loc[nglac,'fa_mwea_err'] = fa_mwea_err
            
            # Convert mass balance to Gta
            mb_gta_raw = mwea_to_gta(mb_mwea, area_m2)
            mb_gta_raw_err = mwea_to_gta(mb_mwea_err, area_m2)
            fa_data_df.loc[nglac,'Romain_gta_raw'] = mb_gta_raw
            fa_data_df.loc[nglac,'Romain_gta_raw_err'] = mb_gta_raw_err
            
            # Total mass balance corrected for frontal ablation of retreat below sea level
            # assume 50-90% below sea level (70% is mean)
            fa_gta_term_bsl = fa_data_df.loc[nglac, fa_gta_term_cn] * 0.7
            mb_gta_mbtot = mb_gta_raw + fa_gta_term_bsl
            fa_data_df.loc[nglac,'Romain_gta_mbtot'] = mb_gta_mbtot
            # assume 95% confidence in this so z-score = 1.96, which gives stdev of 0.10
            fa_gta_term_bsl_err = fa_gta_err * 0.1
            # sum of squares to aggregate uncertainties
            mb_gta_mbtot_err = (mb_gta_raw_err**2 + fa_gta_term_bsl_err**2)**0.5
            fa_data_df.loc[nglac,'Romain_gta_mbtot_err'] = mb_gta_mbtot_err
            
            if debug:
                print('  mb_tot (gta):', mb_gta_mbtot, mb_gta_mbtot_err)
            
            # Climatic mass balance corrected for frontal ablation
            #  - equals total mass balance minus frontal ablation
            #  note: adding it here because frontal ablation loss is positive in Will's format
            mb_gta_mbclim = mb_gta_mbtot + fa_gta
            fa_data_df.loc[nglac,'Romain_gta_mbclim'] = mb_gta_mbclim
            # sum of squares to aggregate error
            mb_gta_mbclim_err = (mb_gta_mbtot_err**2 + fa_gta_err**2)**0.5
            fa_data_df.loc[nglac,'Romain_gta_mbclim_err'] = mb_gta_mbclim_err
                
            # Convert to mwea 
            fa_data_df.loc[nglac,'Romain_mwea_mbtot'] = gta_to_mwea(mb_gta_mbtot, area_m2) 
            fa_data_df.loc[nglac,'Romain_mwea_mbtot_err'] = gta_to_mwea(mb_gta_mbtot_err, area_m2) 
            fa_data_df.loc[nglac,'Romain_mwea_mbclim'] = gta_to_mwea(mb_gta_mbclim, area_m2)
            fa_data_df.loc[nglac,'Romain_mwea_mbclim_err'] = gta_to_mwea(mb_gta_mbclim_err, area_m2)
            
            if debug:
                print('  mb_tot (mwea):', np.round(gta_to_mwea(mb_gta_mbtot, area_m2),2), np.round(gta_to_mwea(mb_gta_mbtot_err, area_m2),2))
                print('  mb_clim (mwea):', np.round(gta_to_mwea(mb_gta_mbclim, area_m2),2), np.round(gta_to_mwea(mb_gta_mbclim_err, area_m2),2))
                
            # Record area
            mb_idx = mb_rgiids_list.index(rgiid)
            fa_data_df.loc[nglac,'area_km2'] = mb_data_df.loc[mb_idx,'area']
                
            # Update if thickness was measured
            try:
                fa_supp_idx = supp_rgiids_list.index(rgiid)
                fa_data_df.loc[nglac,'thick_measured_yn'] = fa_data_supp_df.loc[fa_supp_idx,'do_we_have_an_observation_in_middle_fifth']
            except:
                fa_data_df.loc[nglac,'thick_measured_yn'] = np.nan
                
            # ----- UPDATE ROMAIN'S DATA -----
            if rgiid in rgiids_wmultiples:
                for rgiid_single in rgiids_multiple_list:
                    mb_idx = mb_rgiids_list.index(rgiid_single)
                    mb_data_df.loc[mb_idx,'mb_mwea'] = gta_to_mwea(mb_gta_mbclim, area_m2)
                    mb_data_df.loc[mb_idx,'mb_mwea_err'] = gta_to_mwea(mb_gta_mbclim_err, area_m2)
            else:
                mb_data_df.loc[mb_idx,'mb_mwea'] = gta_to_mwea(mb_gta_mbclim, area_m2)
                mb_data_df.loc[mb_idx,'mb_mwea_err'] = gta_to_mwea(mb_gta_mbclim_err, area_m2)
                
    # Export Romain's data
    mb_data_df.to_csv(pygem_prms.hugonnet_fp + pygem_prms.hugonnet_fn.replace('.csv','-FAcorrected.csv'), index=False)
    
    # Export frontal ablation data for Will
    fa_data_df.to_csv(will_fp + will_fn.replace('.csv','-wromainMB.csv'), index=False)
    #%%
    # Plot the data for each region
    fa_data_df['O1Region'] = [int(x.split('-')[1].split('.')[0]) for x in fa_data_df.RGIId.values]
    regions = [int(x.split('-')[1].split('.')[0]) for x in fa_data_df.RGIId.values]
    regions_unique = sorted(list(np.unique(fa_data_df.O1Region.values)))
    
    rgi_reg_dict = {'all':'Global',
                1:'Alaska',
                2:'W Canada/USA',
                3:'Arctic Canada (North)',
                4:'Arctic Canada (South)',
                5:'Greenland',
                6:'Iceland',
                7:'Svalbard',
                8:'Scandinavia',
                9:'Russian Arctic',
                10:'North Asia',
                11:'Central Europe',
                12:'Caucasus/Middle East',
                13:'Central Asia',
                14:'South Asia (West)',
                15:'South Asia (East)',
                16:'Low Latitudes',
                17:'Southern Andes',
                18:'New Zealand',
                19:'Antarctica/Subantarctic'
                }
    
    for reg in regions_unique:
#    for reg in [1]:
        fa_data_df_subset = fa_data_df.loc[fa_data_df['O1Region']==reg]
#        fa_data_df_subset = fa_data_df_subset.loc[fa_data_df_subset['Romain_mwea_mbclim']<0,:]
#        fa_data_df_subset = fa_data_df_subset.dropna(subset=['thick_measured_yn'])
        
        # ----- FIGURES -----
        fig, ax = plt.subplots(2, 4, squeeze=False, sharex=False, sharey=False, 
                               gridspec_kw = {'wspace':0.5, 'hspace':0.25})
        
        # Frontal ablation (gta)
#        ax[0,0].scatter(fa_data_df_subset['area_km2'], fa_data_df_subset[fa_gta_cn], 
#                        marker='o', edgecolors='k', facecolors='None', linewidth=0.5, s=3)
        ax[0,0].errorbar(fa_data_df_subset['area_km2'], fa_data_df_subset[fa_gta_cn], 
                         yerr=fa_data_df_subset[fa_gta_err_cn], fmt='o',
                         marker='o', mec='k', mew=0.5, mfc='none', markersize=3, c='k', lw=0.25)
        ax[0,0].set_ylabel('FA (gta)')
        ax[0,0].set_xlabel('Area (km2)')
#        ax[0,0].set_xlim(left=0)
        ax[0,0].set_xscale('log')
        ax[0,0].set_ylim(bottom=0)
        
        ax[1,0].hist(fa_data_df_subset[fa_gta_cn].values, bins=20, color='grey')
        ax[1,0].set_xlabel('FA (Gta)')
        ax[1,0].set_ylabel('Count')
        
        # Frontal Ablation (mwea)
#        ax[0,1].scatter(fa_data_df_subset['area_km2'], fa_data_df_subset['fa_mwea'], 
#                        marker='o', edgecolors='k', facecolors='None', linewidth=0.5, s=3)
        ax[0,1].errorbar(fa_data_df_subset['area_km2'], fa_data_df_subset['fa_mwea'], 
                         yerr=fa_data_df_subset['fa_mwea_err'], fmt='o',
                         marker='o', mec='k', mew=0.5, mfc='none', markersize=3, c='k', lw=0.25)
        ax[0,1].set_ylabel('FA (mwea)')
        ax[0,1].set_xlabel('Area (km2)')
#        ax[0,1].set_xlim(left=0)
        ax[0,1].set_xscale('log')
        ax[0,1].set_ylim(bottom=0)
        
        ax[1,1].hist(fa_data_df_subset['fa_mwea'].values, bins=20, color='grey')
        ax[1,1].set_xlabel('FA (mwea)')
        ax[1,1].set_ylabel('Count')
        
        # Climatic mass balance (mwea)
#        ax[0,2].scatter(fa_data_df_subset['area_km2'], fa_data_df_subset['Romain_mwea_mbclim'], 
#                        marker='o', edgecolors='k', facecolors='None', linewidth=0.5, s=3)
        ax[0,2].errorbar(fa_data_df_subset['area_km2'], fa_data_df_subset['Romain_mwea_mbclim'], 
                         yerr=fa_data_df_subset['Romain_mwea_mbclim_err'], fmt='o',
                         marker='o', mec='k', mew=0.5, mfc='none', markersize=3, c='k', lw=0.25)
        ax[0,2].set_ylabel('B_clim (mwea)')
        ax[0,2].set_xlabel('Area (km2)')
#        ax[0,2].set_xlim(left=0)
        ax[0,2].set_xscale('log')
        ax[0,2].axhline(0, color='k', lw=0.5)
        
        ax[1,2].hist(fa_data_df_subset['Romain_mwea_mbclim'].values, bins=20, color='grey')
        ax[1,2].set_xlabel('B_clim (mwea)')
        ax[1,2].set_ylabel('Count')
        
        
        # Climatic mass balance from Romain vs. corrected (mwea)
        rgiids_fa_subset = list(fa_data_df_subset.RGIId)
        mb_idx_list = [rgiids_mb_data.index(x) for x in rgiids_fa_subset]
        mb_data_df_subset = mb_data_df.loc[mb_idx_list,:]
        ax[0,3].scatter(mb_data_df_subset['mb_mwea_romain'], mb_data_df_subset['mb_mwea'], 
                        marker='o', edgecolors='k', facecolors='None', linewidth=0.5, s=3)
        ax[0,3].set_xlabel('Uncorrected B_clim (mwea)')
        ax[0,3].set_ylabel('Corrected B_clim (mwea)')
        ax[0,3].set_xlim(np.min([mb_data_df_subset['mb_mwea_romain'].min(),mb_data_df_subset['mb_mwea'].min()]),
                         np.max([mb_data_df_subset['mb_mwea_romain'].max(),mb_data_df_subset['mb_mwea'].max()]))
        ax[0,3].set_ylim(np.min([mb_data_df_subset['mb_mwea_romain'].min(),mb_data_df_subset['mb_mwea'].min()]),
                         np.max([mb_data_df_subset['mb_mwea_romain'].max(),mb_data_df_subset['mb_mwea'].max()]))
        ax[0,3].plot([-10,10],[-10,10], color='k', lw=0.5)
        
        ax[1,3].hist(mb_data_df_subset['mb_mwea_romain'].values, bins=20, color='grey')
        ax[1,3].set_xlabel('Uncor B_clim (mwea)')
        ax[1,3].set_ylabel('Count')
        
        fig.suptitle(rgi_reg_dict[reg])

        # Save figure
        fig_fn = (str(reg) + '-' + rgi_reg_dict[reg] + '_fa_diagnostics.png')
        fig.set_size_inches(10,6)
        plt.show()
        fig_fp = will_fp + 'figs/'
        if not os.path.exists(fig_fp):
            os.makedirs(fig_fp)
        fig.savefig(fig_fp + fig_fn, bbox_inches='tight', dpi=300)
        
        # Export mb_data_df_subset
        mb_data_df_subset.to_csv(will_fp + str(reg).zfill(2) + '_mbdata_FA_corrected.csv', index=False)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    