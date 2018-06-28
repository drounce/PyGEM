"""
class of mass balance data and functions associated with manipulating the dataset to be in the proper format
"""

import pandas as pd
import numpy as np
import calendar
import datetime

import pygem_input as input
import pygemfxns_modelsetup as modelsetup

class MBData():
    # GCM is the global climate model for a glacier evolution model run
    # When calling ERA-Interim, should be able to use defaults
    # When calling 
    def __init__(self, 
                 name='shean',
                 rgi_regionO1=input.rgi_regionsO1[0]
                 ):
        
        # Source of climate data
        self.name = name
        # Set parameters for ERA-Interim and CMIP5 netcdf files
        if self.name == 'shean': 
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.shean_fp
            self.ds_fn = input.shean_fn
            self.rgi_glacno_cn = input.shean_rgi_glacno_cn
            self.mb_mwea_cn = input.shean_mb_cn
            self.mb_mwea_err_cn = input.shean_mb_err_cn
            self.t1_cn = input.shean_time1_cn
            self.t2_cn = input.shean_time2_cn
            self.area_cn = input.shean_area_cn
            self.mb_vol_cn = input.shean_vol_cn
            self.mb_vol_err_cn = input.shean_vol_err_cn
            
        elif self.name == 'brun':
            self.data_fp = input.brun_fp,
            
        elif self.name == 'mauer':
            self.data_fp = input.mauer_fp
            
        elif self.name == 'wgms_ee':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.wgms_fp
            self.ds_fn = input.wgms_ee_fn_preprocessed
            self.rgi_glacno_cn = input.wgms_ee_rgi_glacno_cn
            self.mb_mwe_cn = input.wgms_ee_mb_cn
            self.mb_mwe_err_cn = input.wgms_ee_mb_err_cn
            self.t1_cn = input.wgms_ee_t1_cn
            self.period_cn = input.wgms_ee_period_cn
            self.z1_cn = input.wgms_ee_z1_cn
            self.z2_cn = input.wgms_ee_z2_cn
            self.obs_type_cn = input.wgms_obs_type_cn
            
        elif self.name == 'wgms_d':
            self.rgi_regionO1 = rgi_regionO1
            self.ds_fp = input.wgms_fp
            self.ds_fn = input.wgms_d_fn_preprocessed
            self.rgi_glacno_cn = input.wgms_ee_rgi_glacno_cn
            self.mb_cn = input.wgms_d_thickness_chg_cn
            
            
            wgms_d_fn_preprocessed = 'wgms_d_rgiv6_preprocessed.csv' 
            wgms_d_rgi_glacno_cn = 'glacno'
            wgms_d_mb_cn = 'BALANCE'
            wgms_d_mb_err_cn = 'BALANCE_UNC'
            wgms_d_t1_cn = 't1'
            wgms_d_t2_cn = 't2'
            wgms_d_z1_cn = 'z1'
            wgms_d_z2_cn = 'z2'
            wgms_d_obs_type_cn = 'obs_type'

            
    def masschange_total(self, main_glac_rgi, main_glac_hyps, dates_table):
        """
        Calculate the total mass change for various datasets and output it in a format that is useful for calibration
        """       
        # Column names of output
        ds_output_cols = ['RGIId', 'glacno', 'obs_type', 'mb_mwe', 'mb_mwe_err', 'sla_m',  'z1_idx', 'z2_idx', 'z1', 
                          'z2', 't1_idx', 't2_idx', 't1', 't2', 't1_datetime', 't2_datetime', 'area_km2', 'WGMS_ID']
        # Reset rgi index so it is consistent with hyps and other data
        main_glac_rgi.reset_index(drop=True, inplace=True)
        # Dictionary linking O1Index to Index
        indexdict = dict(zip(main_glac_rgi['O1Index'], main_glac_rgi.index.values))
        
            
        # Dataset specific calculations
        if self.name == 'shean':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn)
            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            ds_reg.reset_index(drop=True, inplace=True)
            # Glacier number and index for comparison
            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
            ds_reg['O1Index'] = (ds_reg['glacno'] - 1).astype(int)
            ds_reg['RGIId'] = ('RGI60-' + str(input.rgi_regionsO1[0]) + '.' + 
                               (ds_reg['glacno'] / 10**5).astype(str).str.split('.').str[1])
            # Select glaciers with mass balance data
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi[input.rgi_O1Id_colname]) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = (
                    (main_glac_hyps.iloc[ds['O1Index'].map(indexdict)].values != 0).argmax(axis=1).astype(int))
            ds['z2_idx'] = (
                    (main_glac_hyps.iloc[ds['O1Index'].map(indexdict)].values.cumsum(1)).argmax(axis=1).astype(int))
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            ds['area_km2'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2'] = (
                        main_glac_hyps.iloc[indexdict[ds.loc[x,'O1Index']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
            # Time indices
            ds['t1'] = ds[self.t1_cn]
            ds['t2'] = ds[self.t2_cn]
            ds['t1_year'] = ds['t1'].astype(int)
            ds['t1_month'] = (ds['t1'] % ds['t1_year'] * 12 + 1).astype(int)
            ds['t1_day'] = ((ds['t1'] % ds['t1_year'] * 12 + 1) % 1 * 29).astype(int)
            ds['t2_year'] = ds['t2'].astype(int)
            ds['t2_month'] = int(9)
            ds['t2_day'] = int(1)
            ds['t2'] = ds['t2_year'] + ds['t2_month'] / 12
            ds['t1_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
            ds['t2_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            year_decimal_min = dates_table.loc[0,'year'] + dates_table.loc[0,'month'] / 12
            year_decimal_max = (dates_table.loc[dates_table.shape[0]-1,'year'] + 
                                (dates_table.loc[dates_table.shape[0]-1,'month'] + 1) / 12)
            ds = ds[ds['t1'] > year_decimal_min]
            ds = ds[ds['t2'] < year_decimal_max]
            ds.reset_index(drop=True, inplace=True)            
            # Determine time indices (exclude spinup years, since massbal fxn discards spinup years)
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            # Specific mass balance [mwea]
            ds['mb_mwe'] = ds[self.mb_mwea_cn] * (ds['t2'] - ds['t1'])
            ds['mb_mwe_err'] = ds[self.mb_mwea_err_cn] * (ds['t2'] - ds['t1']) 
#            # Total mass change [Gt]
#            ds['mb_gt'] = ds[self.mb_vol_cn] * (ds['t2'] - ds['t1']) * (1/1000)**3 * input.density_water / 1000
#            ds['mb_gt_err'] = ds[self.mb_vol_err_cn] * (ds['t2'] - ds['t1']) * (1/1000)**3 * input.density_water / 1000
            # Observation type
            ds['obs_type'] = 'mb_geo'
            
        elif self.name == 'brun':
            print('code brun')
            
        elif self.name == 'mauer':
            print('code mauer')
            
        elif self.name == 'wgms_ee':
            # Load all data
            ds_all = pd.read_csv(self.ds_fp + self.ds_fn, encoding='latin1')
            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
            # Select data for specific region
            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
            # Glacier number and index for comparison
            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
            ds_reg['O1Index'] = (ds_reg['glacno'] - 1).astype(int)
            # Fill in glaciers without reference data
            ds_reg.loc[ds_reg.BEGIN_PERIOD.isnull(), 'BEGIN_PERIOD'] = (
                    (ds_reg.loc[ds_reg.BEGIN_PERIOD.isnull(), 'YEAR'] - 1) * 10**4 + 9999)
            ds_reg.loc[ds_reg.END_PERIOD.isnull(), 'END_PERIOD'] = (
                    ds_reg.loc[ds_reg.END_PERIOD.isnull(), 'YEAR'] * 10**4 + 9999)
            # Select glaciers from those being modeled using main_glac_rgi
            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi[input.rgi_O1Id_colname]) == True)[0],:]
                  ).copy()
            ds.reset_index(drop=True, inplace=True)
            # Elevation indices
            elev_bins = main_glac_hyps.columns.values.astype(int)
            elev_bin_interval = elev_bins[1] - elev_bins[0]
            ds['z1_idx'] = np.nan
            ds['z2_idx'] = np.nan
            ds.loc[ds[self.z1_cn] == 9999, 'z1_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z1_cn] == 9999, 'O1Index'].map(indexdict)].values != 0)
                     .argmax(axis=1))
            ds.loc[ds[self.z2_cn] == 9999, 'z2_idx'] = (
                    (main_glac_hyps.iloc[ds.loc[ds[self.z2_cn] == 9999, 'O1Index'].map(indexdict)].values.cumsum(1))
                     .argmax(axis=1))
            ds.loc[ds[self.z1_cn] != 9999, 'z1_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z1_cn] != 9999, self.z1_cn].shape[0],1)) - 
                      ds.loc[ds[self.z1_cn] != 9999, self.z1_cn][:,np.newaxis]) > 0).argmax(axis=1))
            ds.loc[ds[self.z2_cn] != 9999, 'z2_idx'] = (
                    ((np.tile(elev_bins, (ds.loc[ds[self.z2_cn] != 9999, self.z2_cn].shape[0],1)) - 
                      ds.loc[ds[self.z2_cn] != 9999, self.z2_cn][:,np.newaxis]) > 0).argmax(axis=1) - 1)
            ds['z1_idx'] = ds['z1_idx'].values.astype(int)
            ds['z2_idx'] = ds['z2_idx'].values.astype(int)
            # Lower and upper bin elevations [masl]
            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
            # Area [km2]
            ds['area_km2'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'area_km2'] = (
                        main_glac_hyps.iloc[indexdict[ds.loc[x,'O1Index']], 
                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
            ds = ds[ds['area_km2'] > 0]
            ds.reset_index(drop=True, inplace=True)
            # Time indices
            #  winter and summer balances typically have the same data for 'BEGIN_PERIOD' and 'END_PERIOD' as the annual
            #  measurements, so need to set these dates manually
            ds['t1_year'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t1_month'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t1_day'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)
            ds['t2_year'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
            ds['t2_month'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
            ds['t2_day'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)
            # if annual measurement and month/day unknown for start or end period, then replace with water year
            # Add latitude 
            latdict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi['CenLat']))
            ds['CenLat'] = ds['RGIId'].map(latdict)
            ds['lat_category'] = np.nan
            ds.loc[ds['CenLat'] >= input.lat_threshold, 'lat_category'] = 'northernmost'
            ds.loc[(ds['CenLat'] < input.lat_threshold) & (ds['CenLat'] > 0), 'lat_category'] = 'north'
            ds.loc[(ds['CenLat'] <= 0) & (ds['CenLat'] > -1*input.lat_threshold), 'lat_category'] = 'south'
            ds.loc[ds['CenLat'] <= -1*input.lat_threshold, 'lat_category'] = 'southernmost'
            ds['months_wintersummer'] = ds['lat_category'].map(input.monthdict)
            ds['winter_begin'] = ds['months_wintersummer'].apply(lambda x: x[0])
            ds['winter_end'] = ds['months_wintersummer'].apply(lambda x: x[1])
            ds['summer_begin'] = ds['months_wintersummer'].apply(lambda x: x[2])
            ds['summer_end'] = ds['months_wintersummer'].apply(lambda x: x[3])
            # annual start
            ds.loc[ds['t1_month'] == 99, 't1_month'] = ds.loc[ds['t1_month'] == 99, 'winter_begin']
            ds.loc[ds['t1_day'] == 99, 't1_day'] = 1
            ds.loc[ds['t2_month'] == 99, 't2_month'] = ds.loc[ds['t2_month'] == 99, 'winter_begin'] - 1
            for x in range(ds.shape[0]):
                if ds.loc[x, 't2_day'] == 99:
                    try:
                        ds.loc[x, 't2_day'] = (
                                dates_table.loc[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                (ds.loc[x, 't2_month'] == dates_table['month']), 'daysinmonth']
                                                .values[0])
                    except:
                        ds.loc[x, 't2_day'] = 28
            # If period is summer/winter, adjust dates accordingly
            for x in range(ds.shape[0]):
                if (((ds.loc[x, 'lat_category'] == 'north') or (ds.loc[x, 'lat_category'] == 'northern')) and 
                    (ds.loc[x, 'period'] == 'summer')):
                    ds.loc[x, 't1_year'] = ds.loc[x, 't1_year'] + 1
                    ds.loc[x, 't1_month'] = ds.loc[x, 'summer_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'summer_end']
                elif (((ds.loc[x, 'lat_category'] == 'south') or (ds.loc[x, 'lat_category'] == 'southernmost')) and 
                    (ds.loc[x, 'period'] == 'summer')):
                    ds.loc[x, 't1_month'] = ds.loc[x, 'summer_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'summer_end']
                elif (((ds.loc[x, 'lat_category'] == 'north') or (ds.loc[x, 'lat_category'] == 'northern')) and 
                    (ds.loc[x, 'period'] == 'winter')):
                    ds.loc[x, 't1_month'] = ds.loc[x, 'winter_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'winter_end']
                elif (((ds.loc[x, 'lat_category'] == 'south') or (ds.loc[x, 'lat_category'] == 'southernmost')) and 
                    (ds.loc[x, 'period'] == 'summer')):
                    ds.loc[x, 't1_year'] = ds.loc[x, 't1_year'] + 1
                    ds.loc[x, 't1_month'] = ds.loc[x, 'winter_begin']
                    ds.loc[x, 't2_month'] = ds.loc[x, 'winter_end']
                ds.loc[x, 't1_day'] = 1
                ds.loc[x, 't2_day'] = calendar.monthrange(ds.loc[x, 't2_year'], ds.loc[x, 't2_month'])[1]
            # Drop measurements outside of calibration period
            ds['t1_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
            ds['t2_datetime'] = pd.to_datetime(
                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
            ds['t1_doy'] = ds.t1_datetime.dt.strftime("%j").astype(float)
            ds['t2_doy'] = ds.t2_datetime.dt.strftime("%j").astype(float)
            ds['t1_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t2_daysinyear'] = (
                    (pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':12, 'day':31})) - 
                     pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
            ds['t1'] = ds.t1_year + ds.t1_doy / ds.t1_daysinyear
            ds['t2'] = ds.t2_year + ds.t2_doy / ds.t2_daysinyear
            # Min and max times in decimal years for comparison
            min_datetime = datetime.datetime(dates_table.loc[0,'year'], dates_table.loc[0,'month'], 
                                             calendar.monthrange(dates_table.loc[0,'year'], dates_table.loc[0,'month'])
                                             [0])
            min_julianday = pd.Series(min_datetime).dt.strftime("%j").astype(float).iloc[0]
            if dates_table.loc[0, 'year'] % 4 == 0:
                min_daysinyear = 366
            else:
                min_daysinyear = 365     
            year_decimal_min = dates_table.loc[0,'year'] + min_julianday / min_daysinyear
            max_datetime = datetime.datetime(dates_table.loc[0,'year'], dates_table.loc[0,'month'], 
                                             calendar.monthrange(dates_table.loc[0,'year'], dates_table.loc[0,'month'])
                                             [1])
            max_julianday = pd.Series(max_datetime).dt.strftime("%j").astype(float).iloc[0]
            if dates_table.loc[dates_table.shape[0] - 1, 'year'] % 4 == 0:
                max_daysinyear = 366
            else:
                max_daysinyear = 365     
            year_decimal_max = dates_table.loc[dates_table.shape[0]-1,'year'] + max_julianday / max_daysinyear
            ds = ds[ds['t1'] > year_decimal_min]
            ds = ds[ds['t2'] < year_decimal_max]
            ds.reset_index(drop=True, inplace=True)
            # Annual, summer, and winter time indices
            #  exclude spinup years, since massbal fxn discards spinup years
            ds['t1_idx'] = np.nan
            ds['t2_idx'] = np.nan
            for x in range(ds.shape[0]):
                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
            # Specific mass balance [mwe]
            ds['mb_mwe'] = ds[self.mb_mwe_cn] / 1000
            ds['mb_mwe_err'] = ds[self.mb_mwe_err_cn] / 1000
#            # Total mass change [Gt]
#            ds['mb_gt'] = ds[self.mb_mwe_cn] / 1000 * ds['area_km2'] * 1000**2 * input.density_water / 1000 / 10**9
#            ds['mb_gt_err'] = (ds[self.mb_mwe_err_cn] / 1000 * ds['area_km2'] * 1000**2 * input.density_water / 1000 
#                               / 10**9)
            # Observation type
            ds['obs_type'] = 'mb_glac'
            
            
            
            
#        elif self.name == 'wgms_d':
#            # Load all data
#            ds_all = pd.read_csv(self.ds_fp + self.ds_fn, encoding='latin1')
#            ds_all['RegO1'] = ds_all[self.rgi_glacno_cn].values.astype(int)
#            # Select data for specific region
#            ds_reg = ds_all[ds_all['RegO1']==self.rgi_regionO1].copy()
#            # Glacier number and index for comparison
#            ds_reg['glacno'] = ((ds_reg[self.rgi_glacno_cn] % 1) * 10**5).round(0).astype(int)
#            ds_reg['O1Index'] = (ds_reg['glacno'] - 1).astype(int)
#            # Fill in glaciers without reference data
#            ds_reg.loc[ds_reg.BEGIN_PERIOD.isnull(), 'BEGIN_PERIOD'] = (
#                    (ds_reg.loc[ds_reg.BEGIN_PERIOD.isnull(), 'YEAR'] - 1) * 10**4 + 9999)
#            ds_reg.loc[ds_reg.END_PERIOD.isnull(), 'END_PERIOD'] = (
#                    ds_reg.loc[ds_reg.END_PERIOD.isnull(), 'YEAR'] * 10**4 + 9999)
#            # Select glaciers from those being modeled using main_glac_rgi
#            ds = (ds_reg.iloc[np.where(ds_reg['glacno'].isin(main_glac_rgi[input.rgi_O1Id_colname]) == True)[0],:]
#                  ).copy()
#            ds.reset_index(drop=True, inplace=True)
#            # Elevation indices
#            elev_bins = main_glac_hyps.columns.values.astype(int)
#            elev_bin_interval = elev_bins[1] - elev_bins[0]
#            ds['z1_idx'] = np.nan
#            ds['z2_idx'] = np.nan
#            ds.loc[ds[self.z1_cn] == 9999, 'z1_idx'] = (
#                    (main_glac_hyps.iloc[ds.loc[ds[self.z1_cn] == 9999, 'O1Index'].map(indexdict)].values != 0)
#                     .argmax(axis=1))
#            ds.loc[ds[self.z2_cn] == 9999, 'z2_idx'] = (
#                    (main_glac_hyps.iloc[ds.loc[ds[self.z2_cn] == 9999, 'O1Index'].map(indexdict)].values.cumsum(1))
#                     .argmax(axis=1))
#            ds.loc[ds[self.z1_cn] != 9999, 'z1_idx'] = (
#                    ((np.tile(elev_bins, (ds.loc[ds[self.z1_cn] != 9999, self.z1_cn].shape[0],1)) - 
#                      ds.loc[ds[self.z1_cn] != 9999, self.z1_cn][:,np.newaxis]) > 0).argmax(axis=1))
#            ds.loc[ds[self.z2_cn] != 9999, 'z2_idx'] = (
#                    ((np.tile(elev_bins, (ds.loc[ds[self.z2_cn] != 9999, self.z2_cn].shape[0],1)) - 
#                      ds.loc[ds[self.z2_cn] != 9999, self.z2_cn][:,np.newaxis]) > 0).argmax(axis=1) - 1)
#            ds['z1_idx'] = ds['z1_idx'].values.astype(int)
#            ds['z2_idx'] = ds['z2_idx'].values.astype(int)
#            # Lower and upper bin elevations [masl]
#            ds['z1'] = elev_bins[ds['z1_idx'].values] - elev_bin_interval/2
#            ds['z2'] = elev_bins[ds['z2_idx'].values] + elev_bin_interval/2
#            # Area [km2]
#            ds['area_km2'] = np.nan
#            for x in range(ds.shape[0]):
#                ds.loc[x,'area_km2'] = (
#                        main_glac_hyps.iloc[indexdict[ds.loc[x,'O1Index']], 
#                                            ds.loc[x,'z1_idx']:ds.loc[x,'z2_idx']+1].sum())
#            ds = ds[ds['area_km2'] > 0]
#            ds.reset_index(drop=True, inplace=True)
#            # Time indices
#            #  winter and summer balances typically have the same data for 'BEGIN_PERIOD' and 'END_PERIOD' as the annual
#            #  measurements, so need to set these dates manually
#            ds['t1_year'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
#            ds['t1_month'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
#            ds['t1_day'] = ds['BEGIN_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)
#            ds['t2_year'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[:4].astype(int)
#            ds['t2_month'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[4:6].astype(int)
#            ds['t2_day'] = ds['END_PERIOD'].astype(str).str.split('.').str[0].str[6:].astype(int)
#            # if annual measurement and month/day unknown for start or end period, then replace with water year
#            # Add latitude 
#            latdict = dict(zip(main_glac_rgi['RGIId'], main_glac_rgi['CenLat']))
#            ds['CenLat'] = ds['RGIId'].map(latdict)
#            ds['lat_category'] = np.nan
#            ds.loc[ds['CenLat'] >= input.lat_threshold, 'lat_category'] = 'northernmost'
#            ds.loc[(ds['CenLat'] < input.lat_threshold) & (ds['CenLat'] > 0), 'lat_category'] = 'north'
#            ds.loc[(ds['CenLat'] <= 0) & (ds['CenLat'] > -1*input.lat_threshold), 'lat_category'] = 'south'
#            ds.loc[ds['CenLat'] <= -1*input.lat_threshold, 'lat_category'] = 'southernmost'
#            ds['months_wintersummer'] = ds['lat_category'].map(input.monthdict)
#            ds['winter_begin'] = ds['months_wintersummer'].apply(lambda x: x[0])
#            ds['winter_end'] = ds['months_wintersummer'].apply(lambda x: x[1])
#            ds['summer_begin'] = ds['months_wintersummer'].apply(lambda x: x[2])
#            ds['summer_end'] = ds['months_wintersummer'].apply(lambda x: x[3])
#            # annual start
#            ds.loc[ds['t1_month'] == 99, 't1_month'] = ds.loc[ds['t1_month'] == 99, 'winter_begin']
#            ds.loc[ds['t1_day'] == 99, 't1_day'] = 1
#            ds.loc[ds['t2_month'] == 99, 't2_month'] = ds.loc[ds['t2_month'] == 99, 'winter_begin'] - 1
#            for x in range(ds.shape[0]):
#                if ds.loc[x, 't2_day'] == 99:
#                    try:
#                        ds.loc[x, 't2_day'] = (
#                                dates_table.loc[(ds.loc[x, 't2_year'] == dates_table['year']) & 
#                                                (ds.loc[x, 't2_month'] == dates_table['month']), 'daysinmonth']
#                                                .values[0])
#                    except:
#                        ds.loc[x, 't2_day'] = 28
#            # If period is summer/winter, adjust dates accordingly
#            for x in range(ds.shape[0]):
#                if (((ds.loc[x, 'lat_category'] == 'north') or (ds.loc[x, 'lat_category'] == 'northern')) and 
#                    (ds.loc[x, 'period'] == 'summer')):
#                    ds.loc[x, 't1_year'] = ds.loc[x, 't1_year'] + 1
#                    ds.loc[x, 't1_month'] = ds.loc[x, 'summer_begin']
#                    ds.loc[x, 't2_month'] = ds.loc[x, 'summer_end']
#                elif (((ds.loc[x, 'lat_category'] == 'south') or (ds.loc[x, 'lat_category'] == 'southernmost')) and 
#                    (ds.loc[x, 'period'] == 'summer')):
#                    ds.loc[x, 't1_month'] = ds.loc[x, 'summer_begin']
#                    ds.loc[x, 't2_month'] = ds.loc[x, 'summer_end']
#                elif (((ds.loc[x, 'lat_category'] == 'north') or (ds.loc[x, 'lat_category'] == 'northern')) and 
#                    (ds.loc[x, 'period'] == 'winter')):
#                    ds.loc[x, 't1_month'] = ds.loc[x, 'winter_begin']
#                    ds.loc[x, 't2_month'] = ds.loc[x, 'winter_end']
#                elif (((ds.loc[x, 'lat_category'] == 'south') or (ds.loc[x, 'lat_category'] == 'southernmost')) and 
#                    (ds.loc[x, 'period'] == 'summer')):
#                    ds.loc[x, 't1_year'] = ds.loc[x, 't1_year'] + 1
#                    ds.loc[x, 't1_month'] = ds.loc[x, 'winter_begin']
#                    ds.loc[x, 't2_month'] = ds.loc[x, 'winter_end']
#            # Drop measurements outside of calibration period
#            ds['t1_datetime'] = pd.to_datetime(
#                    pd.DataFrame({'year':ds.t1_year.values, 'month':ds.t1_month.values, 'day':ds.t1_day.values}))
#            ds['t2_datetime'] = pd.to_datetime(
#                    pd.DataFrame({'year':ds.t2_year.values, 'month':ds.t2_month.values, 'day':ds.t2_day.values}))
#            ds['t1_doy'] = ds.t1_datetime.dt.strftime("%j").astype(float)
#            ds['t2_doy'] = ds.t2_datetime.dt.strftime("%j").astype(float)
#            ds['t1_daysinyear'] = (
#                    (pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':12, 'day':31})) - 
#                     pd.to_datetime(pd.DataFrame({'year':ds.t1_year.values, 'month':1, 'day':1}))).dt.days + 1)
#            ds['t2_daysinyear'] = (
#                    (pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':12, 'day':31})) - 
#                     pd.to_datetime(pd.DataFrame({'year':ds.t2_year.values, 'month':1, 'day':1}))).dt.days + 1)
#            ds['t1'] = ds.t1_year + ds.t1_doy / ds.t1_daysinyear
#            ds['t2'] = ds.t2_year + ds.t2_doy / ds.t2_daysinyear
#            year_decimal_min = dates_table.loc[0,'year'] + dates_table.loc[0,'month'] / 12
#            year_decimal_max = (dates_table.loc[dates_table.shape[0]-1,'year'] + 
#                                (dates_table.loc[dates_table.shape[0]-1,'month'] + 1) / 12)
#            ds = ds[ds['t1'] > year_decimal_min]
#            ds = ds[ds['t2'] < year_decimal_max]
#            ds.reset_index(drop=True, inplace=True)
#            # Annual, summer, and winter time indices
#            #  exclude spinup years, since massbal fxn discards spinup years
#            ds['t1_idx'] = np.nan
#            ds['t2_idx'] = np.nan
#            for x in range(ds.shape[0]):
#                ds.loc[x,'t1_idx'] = (dates_table[(ds.loc[x, 't1_year'] == dates_table['year']) & 
#                                                  (ds.loc[x, 't1_month'] == dates_table['month'])].index.values[0])
#                ds.loc[x,'t2_idx'] = (dates_table[(ds.loc[x, 't2_year'] == dates_table['year']) & 
#                                                  (ds.loc[x, 't2_month'] == dates_table['month'])].index.values[0])
#            # Specific mass balance [mwe]
#            ds['mb_mwe'] = ds[self.mb_mwe_cn] / 1000
#            ds['mb_mwe_err'] = ds[self.mb_mwe_err_cn] / 1000
##            # Total mass change [Gt]
##            ds['mb_gt'] = ds[self.mb_mwe_cn] / 1000 * ds['area_km2'] * 1000**2 * input.density_water / 1000 / 10**9
##            ds['mb_gt_err'] = (ds[self.mb_mwe_err_cn] / 1000 * ds['area_km2'] * 1000**2 * input.density_water / 1000 
##                               / 10**9)
#            # Observation type
#            ds['obs_type'] = 'mb_glac'
            
        # Select output
        ds_output = ds.loc[:, ds_output_cols].sort_values(['glacno', 't1_idx'])
        ds_output.reset_index(drop=True, inplace=True)
        return ds_output


#%% Testing
if __name__ == '__main__':
    # Glacier selection
    rgi_regionsO1 = [7]
    rgi_glac_number = 'all'
#    rgi_glac_number = ['03473', '03733']
    #rgi_glac_number = ['00038', '00046', '00049', '00068', '00118', '00119', '00164', '00204', '00211', '03473', '03733']
#    rgi_glac_number = ['00038', '00046', '00049', '00068', '00118', '00119', '03507', '03473', '03591', '03733', '03734']
#    rgi_glac_number = ['00038', '00046', '00049', '00068', '00118', '00119', '03507', '03473', '03591', '03733']
#    rgi_glac_number = ['03591']
    
    # Required input
    gcm_startyear = 2000
    gcm_endyear = 2015
    gcm_spinupyears = 5
    option_calibration = 1
    
    # Select glaciers
    main_glac_rgi = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2 = 'all', 
                                                      rgi_glac_number=rgi_glac_number)
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, rgi_regionsO1, input.hyps_filepath, 
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Determine dates_table_idx that coincides with data
    dates_table, start_date, end_date = modelsetup.datesmodelrun(gcm_startyear, gcm_endyear, spinupyears=0)
    
    elev_bins = main_glac_hyps.columns.values.astype(int)
    elev_bin_interval = elev_bins[1] - elev_bins[0]
    
    # Testing    
#    mb1 = MBData(name='shean')
#    mb1 = MBData(name='wgms_ee', rgi_regionO1=rgi_regionsO1[0])
#    ds, ds_output = mb1.masschange_total(main_glac_rgi, main_glac_hyps, dates_table)
    
#    cal_datasets = ['shean', 'wgms_ee']
#    cal_datasets = ['shean']
    cal_datasets = ['wgms_ee']
#    cal_datasets = ['wgms_d']
    
    cal_data = pd.DataFrame()
    for dataset in cal_datasets:
        cal_subset = MBData(name=dataset, rgi_regionO1=rgi_regionsO1[0])
        cal_subset_data = cal_subset.masschange_total(main_glac_rgi, main_glac_hyps, dates_table)
        cal_data = cal_data.append(cal_subset_data, ignore_index=True)
    cal_data = cal_data.sort_values(['glacno', 't1_idx'])
    cal_data.reset_index(drop=True, inplace=True)

#%%
