"""Run a model simulation."""
# Default climate data is ERA-Interim; specify CMIP5 by specifying a filename to the argument:
#    (Command line) python run_simulation_list_multiprocess.py -gcm_list_fn=C:\...\gcm_rcpXX_filenames.txt
#      - Default is running ERA-Interim in parallel with five processors.
#    (Spyder) %run run_simulation_list_multiprocess.py C:\...\gcm_rcpXX_filenames.txt -option_parallels=0
#      - Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.
# Spyder cannot run parallels, so always set -option_parallels=0 when testing in Spyder.

# Built-in libraries
import os
import argparse
import multiprocessing
import time
import inspect
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
import pickle
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import pygemfxns_output as output
import class_climate

#%% ===== SCRIPT SPECIFIC INPUT DATA =====
# Required input
# Time period
gcm_startyear = 2000
gcm_endyear = 2100
gcm_spinupyears = 0

# Bias adjustment option (options defined in run_gcmbiasadj script; 0 means no correction)
#option_bias_adjustment = 2

#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    gcm_list_fn (optional) : str
        text file that contains the climate data to be used in the model simulation
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run simulations from gcm list in parallel")
    # add arguments
    parser.add_argument('-gcm_list_fn', action='store', type=str, default=input.ref_gcm_name,
                        help='text file full of commands to run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-biasadj_fn', action='store', type=str, default=None,
                        help='Filename of bias adjustments for cases where automatic loading does not work')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    return parser


def calc_stats(vn, ds, stats_cns=input.sim_stat_cns):
    """
    Calculate stats for a given variable
    
    Parameters
    ----------
    vn : str
        variable name
    ds : xarray dataset
        dataset of output with all ensemble simulations
    
    Returns
    -------
    stats : np.array
        Statistics related to a given variable
    """
    data = ds[vn].values[0,:,:]
    if 'mean' in stats_cns:
        stats = data.mean(axis=1)[:,np.newaxis]
    if 'std' in stats_cns:
        stats = np.append(stats, data.std(axis=1)[:,np.newaxis], axis=1)
    if '2.5%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 2.5, axis=1)[:,np.newaxis], axis=1)
    if '25%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 25, axis=1)[:,np.newaxis], axis=1)
    if 'median' in stats_cns:
        stats = np.append(stats, np.median(data, axis=1)[:,np.newaxis], axis=1)
    if '75%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 75, axis=1)[:,np.newaxis], axis=1)
    if '97.5%' in stats_cns:
        stats = np.append(stats, np.percentile(data, 97.5, axis=1)[:,np.newaxis], axis=1)
    return stats

def netcdf_add_metadata(ds, glacier_rgi_table):
    """
    Add metadata to netcdf file.
    
    Parameters
    ----------
    ds : xarray dataset
        xarray dataset for an individual glacier
    glacier_rgi_table : pd.Series
        glacier rgi table, which stores information about the glacier
    
    Returns
    -------
    ds : xarray dataset
        modified xarray dataset containing long_name, units, comments, etc.
    encoding : dict
        dictionary containing encodings used for export, e.g., turning filling off
    """
    # Global attributes
    # Glacier properties
    ds_existing_attrs = list(ds.attrs.keys())
    for attr_vn in input.output_glacier_attr_vns:
        # Check if attribute exists
        if (attr_vn not in ds_existing_attrs) and (attr_vn in list(glacier_rgi_table.index.values)):
            ds.attrs[attr_vn] = glacier_rgi_table[attr_vn]
    # Calendar year
    if input.option_wateryear == 1:
        ds.attrs['year_type'] = 'water year'
    elif input.option_wateryear == 2:
        ds.attrs['year_type'] = 'calendar year'
    elif input.option_wateryear == 3:
        ds.attrs['year_type'] = 'custom year (user defined start/end months)'
    # Add attributes for given package
    if input.output_package == 2:
        ds.time.attrs['long_name'] = 'date'
        ds.stats.attrs['long_name'] = 'variable statistics'
        ds.stats.attrs['comment'] = '% refers to percentiles'
        ds.year.attrs['long_name'] = 'years'
        ds.year.attrs['comment'] = 'years referring to the start of each year'
        ds.year_plus1.attrs['long_name'] = 'years plus one additional year'
        ds.year_plus1.attrs['comment'] = (
                'additional year allows one to record glacier dimension changes at end of model run')
        if input.option_wateryear == 1:
            ds.year_plus1.attrs['unit'] = 'water year'
            ds.year.attrs['unit'] = 'water year'
        elif input.option_wateryear == 2:
            ds.year_plus1.attrs['unit'] = 'calendar year'
            ds.year.attrs['unit'] = 'calendar year'
        else:
            ds.year_plus1.attrs['unit'] = 'custom year'
            ds.year.attrs['unit'] = 'custom year'          
        ds.temp_glac_monthly.attrs['long_name'] = 'glacier-wide mean air temperature'
        ds.temp_glac_monthly.attrs['units'] = 'degC'
        ds.temp_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.temp_glac_monthly.attrs['comment'] = (
                'each elevation bin is weighted equally to compute the mean temperature, and bins where the glacier' 
                ' no longer exists due to retreat have been removed')
        ds.prec_glac_monthly.attrs['long_name'] = 'glacier-wide precipitation (liquid)'
        ds.prec_glac_monthly.attrs['units'] = 'm'
        ds.prec_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.prec_glac_monthly.attrs['comment'] = 'only the liquid precipitation, solid precipitation excluded'
        ds.acc_glac_monthly.attrs['long_name'] = 'glacier-wide accumulation'
        ds.acc_glac_monthly.attrs['units'] = 'm w.e.'
        ds.acc_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.acc_glac_monthly.attrs['comment'] = 'only the solid precipitation'
        ds.refreeze_glac_monthly.attrs['long_name'] = 'glacier-wide refreeze'
        ds.refreeze_glac_monthly.attrs['units'] = 'm w.e.'
        ds.refreeze_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.melt_glac_monthly.attrs['long_name'] = 'glacier-wide melt'
        ds.melt_glac_monthly.attrs['units'] = 'm w.e.'
        ds.melt_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.frontalablation_glac_monthly.attrs['long_name'] = 'glacier-wide frontal ablation'
        ds.frontalablation_glac_monthly.attrs['units'] = 'm w.e.'
        ds.frontalablation_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.frontalablation_glac_monthly.attrs['comment'] = (
                'mass losses from calving, subaerial frontal melting, sublimation above the waterline and '
                + 'subaqueous frontal melting below the waterline')
        ds.massbaltotal_glac_monthly.attrs['long_name'] = 'glacier-wide total mass balance'
        ds.massbaltotal_glac_monthly.attrs['units'] = 'm w.e.'
        ds.massbaltotal_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.massbaltotal_glac_monthly.attrs['comment'] = (
                'total mass balance is the sum of the climatic mass balance and frontal ablation')
        ds.runoff_glac_monthly.attrs['long_name'] = 'glacier runoff'
        ds.runoff_glac_monthly.attrs['units'] = 'm**3'
        ds.runoff_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.runoff_glac_monthly.attrs['comment'] = 'runoff from the glacier terminus, which moves over time'
        ds.snowline_glac_monthly.attrs['long_name'] = 'transient snowline'
        ds.snowline_glac_monthly.attrs['units'] = "m a.s.l."
        ds.snowline_glac_monthly.attrs['temporal_resolution'] = 'monthly'
        ds.snowline_glac_monthly.attrs['comment'] = 'transient snowline is altitude separating snow from ice/firn'
        ds.area_glac_annual.attrs['long_name'] = 'glacier area'
        ds.area_glac_annual.attrs['units'] = 'km**2'
        ds.area_glac_annual.attrs['temporal_resolution'] = 'annual'
        ds.area_glac_annual.attrs['comment'] = 'area used for the duration of the defined start/end of year'
        ds.volume_glac_annual.attrs['long_name'] = 'glacier volume'
        ds.volume_glac_annual.attrs['units'] = "km**3 ice"
        ds.volume_glac_annual.attrs['temporal_resolution'] = 'annual'
        ds.volume_glac_annual.attrs['comment'] = 'volume based on area and ice thickness used for that year'
        ds.ELA_glac_annual.attrs['long_name'] = 'annual equilibrium line altitude'
        ds.ELA_glac_annual.attrs['units'] = 'm a.s.l.'
        ds.ELA_glac_annual.attrs['temporal_resolution'] = 'annual'
        ds.ELA_glac_annual.attrs['comment'] = (
                'equilibrium line altitude is the elevation where the climatic mass balance is zero')
        # Encoding (specify _FillValue, offsets, etc.)
        encoding = {'time': {'_FillValue': False},
                    'year': {'_FillValue': False},
                    'year_plus1': {'_FillValue': False},
                    'temp_glac_monthly': {'_FillValue': False},
                    'prec_glac_monthly': {'_FillValue': False},
                    'acc_glac_monthly': {'_FillValue': False},
                    'refreeze_glac_monthly': {'_FillValue': False},
                    'melt_glac_monthly': {'_FillValue': False},
                    'frontalablation_glac_monthly': {'_FillValue': False},
                    'massbaltotal_glac_monthly': {'_FillValue': False},
                    'runoff_glac_monthly': {'_FillValue': False},
                    'snowline_glac_monthly': {'_FillValue': False},
                    'area_glac_annual': {'_FillValue': False},
                    'volume_glac_annual': {'_FillValue': False},
                    'ELA_glac_annual': {'_FillValue': False}
                    }
    return ds, encoding


def main(list_packed_vars):
    """
    Model simulation
    
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
        
    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    # Unpack variables
    count = list_packed_vars[0]
    chunk = list_packed_vars[1]
    main_glac_rgi_all = list_packed_vars[2]
    chunk_size = list_packed_vars[3]
    gcm_name = list_packed_vars[4]

    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
        
    if gcm_name != input.ref_gcm_name:
        rcp_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]

    # ===== LOAD GLACIER DATA =====
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :].copy()
    # Glacier hypsometry [km**2], total area
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
    # Ice thickness [m], average
    main_glac_icethickness = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.thickness_filepath,
                                                         input.thickness_filedict, input.thickness_colsdrop)
    main_glac_hyps[main_glac_icethickness == 0] = 0
    # Width [km], average
    main_glac_width = modelsetup.import_Husstable(main_glac_rgi, input.rgi_regionsO1, input.width_filepath,
                                                  input.width_filedict, input.width_colsdrop)
    elev_bins = main_glac_hyps.columns.values.astype(int)
    # Volume [km**3] and mean elevation [m a.s.l.]
    main_glac_rgi['Volume'], main_glac_rgi['Zmean'] = modelsetup.hypsometrystats(main_glac_hyps, main_glac_icethickness)
    
    # Select dates including future projections
    dates_table = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, spinupyears=gcm_spinupyears)
    # Synthetic simulation dates
    if input.option_synthetic_sim == 1:
        dates_table_synthetic, synthetic_start, synthetic_end = modelsetup.datesmodelrun(
                startyear=input.synthetic_startyear, endyear=input.synthetic_endyear, spinupyears=0)
        
    # ===== LOAD CLIMATE DATA =====
    if gcm_name == 'ERA-Interim' or gcm_name == 'COAWST':
        gcm = class_climate.GCM(name=gcm_name)
        # Check that end year is reasonable
        if (gcm_endyear > int(time.strftime("%Y"))) and (input.option_synthetic_sim == 0):
            print('\n\nEND YEAR BEYOND AVAILABLE DATA FOR ERA-INTERIM. CHANGE END YEAR.\n\n')
    else:
        gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp_scenario)
    
    if input.option_synthetic_sim == 0:        
        # Air temperature [degC]
        gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, 
                                                                     dates_table)
        # Precipitation [m]
        gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, 
                                                                     dates_table)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)          
        # Lapse rate
        if gcm_name == 'ERA-Interim':
            gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
#        else:
#            # Mean monthly lapse rate
#            ref_lr_monthly_avg_all = np.genfromtxt(gcm.lr_fp + gcm.lr_fn, delimiter=',')
#            ref_lr_monthly_avg = ref_lr_monthly_avg_all[main_glac_rgi['O1Index'].values]
#            gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))    
        # COAWST data has two domains, so need to merge the two domains
        if gcm_name == 'COAWST':
            gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn,
                                                                             main_glac_rgi, dates_table)
            gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, 
                                                                             main_glac_rgi, dates_table)
            gcm_elev_d01 = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn_d01, gcm.elev_vn, main_glac_rgi)
            # Check if glacier outside of high-res (d02) domain
            for glac in range(main_glac_rgi.shape[0]):
                glac_lat = main_glac_rgi.loc[glac,input.rgi_lat_colname]
                glac_lon = main_glac_rgi.loc[glac,input.rgi_lon_colname]
                if (~(input.coawst_d02_lat_min <= glac_lat <= input.coawst_d02_lat_max) or 
                    ~(input.coawst_d02_lon_min <= glac_lon <= input.coawst_d02_lon_max)):
                    gcm_prec[glac,:] = gcm_prec_d01[glac,:]
                    gcm_temp[glac,:] = gcm_temp_d01[glac,:]
                    gcm_elev[glac] = gcm_elev_d01[glac]
  
    elif input.option_synthetic_sim == 1:
        # Air temperature [degC]
        gcm_temp_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, 
                                                                          dates_table_synthetic)
        # Precipitation [m]
        gcm_prec_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, 
                                                                          dates_table_synthetic)
        # Elevation [m asl]
        gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)  
        # Lapse rate
        gcm_lr_tile, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, 
                                                                        dates_table_synthetic)
        # Future simulation based on synthetic (replicated) data; add spinup years; dataset restarts after spinupyears 
        datelength = dates_table.shape[0] - gcm_spinupyears * 12
        n_tiles = int(np.ceil(datelength / dates_table_synthetic.shape[0]))
        gcm_temp = np.append(gcm_temp_tile[:,:gcm_spinupyears*12], np.tile(gcm_temp_tile,(1,n_tiles))[:,:datelength], 
                             axis=1)
        gcm_prec = np.append(gcm_prec_tile[:,:gcm_spinupyears*12], np.tile(gcm_prec_tile,(1,n_tiles))[:,:datelength], 
                             axis=1)
        gcm_lr = np.append(gcm_lr_tile[:,:gcm_spinupyears*12], np.tile(gcm_lr_tile,(1,n_tiles))[:,:datelength], axis=1)
        # Temperature and precipitation sensitivity adjustments
        gcm_temp = gcm_temp + input.synthetic_temp_adjust
        gcm_prec = gcm_prec * input.synthetic_prec_factor
        

#%%
    # ===== BIAS CORRECTIONS =====
    # Bias adjustments from given filename in argument parser
    if args.biasadj_fn is not None:
        print('load bias adjustment from filename')
        main_glac_biasadj_all = pd.read_csv(input.biasadj_fp + args.biasadj_fn, index_col=0)
        # Bias adjustment option
        option_bias_adjustment = int(args.biasadj_fn.split('biasadj_opt')[1][0])
    else:
        option_bias_adjustment = input.option_bias_adjustment
        if option_bias_adjustment != 0:
            if gcm_name == 'COAWST':
                biasadj_fn = ('R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_biasadj_opt' + 
                              str(input.option_bias_adjustment) + '_' + str(input.startyear) + '_' + str(input.endyear) 
                              + '_wy' + str(input.option_wateryear) + '.csv')
            else:
                biasadj_fn = ('R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + '_biasadj_opt' +
                              str(input.option_bias_adjustment) + '_' + str(input.startyear) + '_' + str(input.endyear) 
                              + '_wy' + str(input.option_wateryear) + '.csv')
            main_glac_biasadj_all = pd.read_csv(input.biasadj_fp + biasadj_fn, index_col=0)
        
    # Option 0 - no bias adjustment
    if option_bias_adjustment == 0:
        gcm_temp_adj = gcm_temp
        gcm_prec_adj = gcm_prec
        gcm_elev_adj = gcm_elev
#    # Option 1
#    elif option_bias_adjustment == 1:
#        gcm_temp_adj = gcm_temp + main_glac_modelparams['temp_adj'].values[:,np.newaxis]
#        gcm_prec_adj = gcm_prec * main_glac_modelparams['prec_adj'].values[:,np.newaxis]
#        gcm_elev_adj = gcm_elev
    # Option 2
    elif option_bias_adjustment == 2:
        main_glac_biasadj = main_glac_biasadj_all.loc[main_glac_rgi['O1Index'].values,:]
        tempvar_cols = ['tempvar_' + str(n) for n in range(1,13)]
        tempavg_cols = ['tempavg_' + str(n) for n in range(1,13)]
        tempadj_cols = ['tempadj_' + str(n) for n in range(1,13)]
        precadj_cols = ['precadj_' + str(n) for n in range(1,13)]
        lr_cols = ['lravg_' + str(n) for n in range(1,13)]
        bias_adj_prec = main_glac_biasadj[precadj_cols].values
        variability_monthly_std = main_glac_biasadj[tempvar_cols].values
        gcm_temp_monthly_avg = main_glac_biasadj[tempavg_cols].values
        gcm_temp_monthly_adj = main_glac_biasadj[tempadj_cols].values
        # Monthly temperature bias adjusted according to monthly average
        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Bias adjusted temperature accounting for monthly mean and variability
        gcm_temp_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
        # Bias adjusted precipitation
        gcm_prec_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
        # Updated elevation, since adjusted according to reference elevation
        gcm_elev_adj = main_glac_biasadj['new_gcmelev'].values
        # Mean monthly lapse rate
        ref_lr_monthly_avg = main_glac_biasadj[lr_cols].values
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))    
#    # Option 3
#    elif option_bias_adjustment == 3:
#        tempadj_cols = ['tempadj_' + str(n) for n in range(1,13)]
#        precadj_cols = ['precadj_' + str(n) for n in range(1,13)]
#        bias_adj_prec = main_glac_modelparams[precadj_cols].values
#        bias_adj_temp = main_glac_modelparams[tempadj_cols].values
#        # Bias adjusted temperature
#        gcm_temp_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
#        # Bias adjusted precipitation
#        gcm_prec_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
#        # Updated elevation, since adjusted according to reference elevation
#        gcm_elev_adj = main_glac_modelparams['new_gcmelev'].values
#%%
    # ===== RUN MASS BALANCE =====
    for glac in range(main_glac_rgi.shape[0]):
        if glac%200 == 0:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_gcm_elev = gcm_elev_adj[glac]
        glacier_gcm_prec = gcm_prec_adj[glac,:]
        glacier_gcm_temp = gcm_temp_adj[glac,:]
        glacier_gcm_lrgcm = gcm_lr[glac,:]
        glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
        glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
        icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
        width_t0 = main_glac_width.iloc[glac,:].values.astype(float)

        # get glacier number
        glacier_RGIId = main_glac_rgi.iloc[glac]['RGIId'][6:]
        
        if debug:
            print(glacier_RGIId)
        
        # Create netcdf file    
        if input.output_package != 0:
            output_sim_fp = input.output_sim_fp + gcm_name + '/reg' + str(input.rgi_regionsO1[0]) + '/'
            # Create filepath if it does not exist
            if os.path.exists(output_sim_fp) == False:
                os.makedirs(output_sim_fp)
            # Netcdf filename
            if (gcm_name == 'ERA-Interim') or (gcm_name == 'COAWST'):
                netcdf_fn = (gcm_name + '_c' + str(input.option_calibration) 
                             + '_ba' + str(option_bias_adjustment) + '_' +  str(input.sim_iters) + 'sets' + '_' + 
                             str(gcm_startyear) + '_' + str(gcm_endyear) + '--' + glacier_RGIId + '.nc')
            else:
                netcdf_fn = (gcm_name + '_' + rcp_scenario + '_c' + str(input.option_calibration) + 
                             '_ba' + str(option_bias_adjustment) + '_' +  str(input.sim_iters) + 'sets' + '_' + 
                             str(gcm_startyear) + '_' + str(gcm_endyear) + '--' + glacier_RGIId + '.nc')
        
            if debug:
                print(netcdf_fn)
        
            main_glac_rgi_float = main_glac_rgi.copy()
            main_glac_rgi_float.drop(labels=['RGIId'], axis=1, inplace=True)
            output.netcdfcreate(netcdf_fn, main_glac_rgi_float.iloc[[glac],:], main_glac_hyps.iloc[[glac],:], 
                                dates_table, output_filepath=output_sim_fp, nsims=input.sim_iters)
            
        if debug:
            print(glacier_RGIId)   
            
        if input.option_import_modelparams == 1:
            if input.option_calibration == 1:
                print('DELETE ME - OPTION CALIBRATION 1/2 SHOULD BE IMPORTED THE SAME WAY!')
            ds_mp = xr.open_dataset(input.modelparams_fp_dict[input.rgi_regionsO1[0]] + glacier_RGIId + '.nc')
            cn_subset = input.modelparams_colnames
            modelparameters_all = (pd.DataFrame(ds_mp['mp_value'].sel(chain=0).values, 
                                                columns=ds_mp.mp.values)[cn_subset])
        else:
            modelparameters_all = (
                    pd.DataFrame(np.asarray([input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, 
                                             input.ddfice, input.tempsnow, input.tempchange]).reshape(1,-1), 
                                             columns=input.modelparams_colnames))
        
        # Set the number of iterations and determine every kth iteration to use for the ensemble
        if (input.option_calibration == 1) or (modelparameters_all.shape[0] == 1):
            sim_iters = 1
        elif input.option_calibration == 2:
            sim_iters = input.sim_iters
            # Select every kth iteration
            mp_spacing = int((modelparameters_all.shape[0] - input.sim_burn) / sim_iters)
            mp_idx_start = np.arange(input.sim_burn, input.sim_burn + mp_spacing)
            np.random.shuffle(mp_idx_start)
            mp_idx_start = mp_idx_start[0]
            mp_idx_all = np.arange(mp_idx_start, modelparameters_all.shape[0], mp_spacing)
            
        # Loop through model parameters
        for n_iter in range(sim_iters):

            if sim_iters == 1:
                modelparameters = modelparameters_all.mean()  
            else:
                mp_idx = mp_idx_all[n_iter]
                modelparameters = modelparameters_all.iloc[mp_idx,:]
            
            # run mass balance calculation
            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
                massbalance.runmassbalance(modelparameters[0:8], glacier_rgi_table, glacier_area_t0, icethickness_t0,
                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                           option_areaconstant=0))
            # Annual glacier-wide mass balance [m w.e.]
            glac_wide_massbaltotal_annual = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
            # Average annual glacier-wide mass balance [m w.e.a.]
            mb_mwea = glac_wide_massbaltotal_annual.mean()
            #  units: m w.e. based on initial area
            # Volume change [%]
#            if icethickness_t0.max() > 0:
#                glac_vol_change_perc = ((glac_wide_volume_annual[-1] - glac_wide_volume_annual[0]) /
#                                        glac_wide_volume_annual[0] * 100)
            
            if debug:
                print('mb_model [mwea]:', mb_mwea.round(6))

            # write to netcdf file
            if input.output_package != 0:
                output.netcdfwrite(netcdf_fn, 0, modelparameters, glacier_rgi_table, elev_bins, glac_bin_temp,
                                   glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
                                   glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual,
                                   glac_bin_area_annual, glac_bin_icethickness_annual, glac_bin_width_annual,
                                   glac_bin_surfacetype_annual, output_filepath=output_sim_fp, sim=n_iter)
                
        # Convert netcdf of ensembles to a netcdf containing stats of the ensembles
        output_sim_stats = output_sim_fp + 'stats/'
        if os.path.exists(output_sim_stats) == False:
                os.makedirs(output_sim_stats)
        # Open netcdf
        ds = xr.open_dataset(output_sim_fp + netcdf_fn)
        if input.output_package == 2:
            # Stats filename
            netcdf_stats_fn = netcdf_fn.split('--')[0] + '_stats--' + netcdf_fn.split('--')[1]
            # List of variables
            ds_vns = []
            for vn in ds.variables:
                ds_vns.append(vn)
            for vn in ds_vns[0:8]:
                ds_vns.remove(vn)
            count_vn = 0
            for vn in ds_vns:
                count_vn += 1
                stats = calc_stats(vn, ds)
                # Determine time coordinate of the variable
                for t_name in input.time_names:
                    if t_name in ds[vn].coords:
                        time_coord = t_name
                # Create dataset for variable
                output_ds = xr.Dataset({vn: ((time_coord, 'stats'), stats)},
                                       coords={time_coord: ds[vn][time_coord].values,
                                               'stats': input.sim_stat_cns})
                # Merge datasets of stats into one output
                if count_vn == 1:
                    output_ds_all = output_ds
                else:
                    output_ds_all = xr.merge((output_ds_all, output_ds))
            # Add global and variable attributes
            output_ds_all, encoding = netcdf_add_metadata(output_ds_all, glacier_rgi_table)
            # Export new file
            output_ds_all.to_netcdf(output_sim_stats + netcdf_stats_fn, encoding=encoding)
            # Remove existing file
            os.remove(output_sim_fp + netcdf_fn)
            
        # Mean and standard deviation of glacier-wide mass balance
        mb_mwea_all = (ds.massbaltotal_glac_monthly.values[0,:,:]).sum(axis=0) / (dates_table.shape[0] / 12)
        print('mb_model [mwea] mean:', round(mb_mwea_all.mean(),3), 'std:', round(mb_mwea_all.std(),3))       


    #%% Export variables as global to view in variable explorer
    if (args.option_parallels == 0) or (main_glac_rgi_all.shape[0] < 2 * args.num_simultaneous_processes):
        global main_vars
        main_vars = inspect.currentframe().f_locals

    print('\nProcessing time of', gcm_name, 'for', count,':',time.time()-time_start, 's')

#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    
    if args.debug == 1:
        debug = True
    else:
        debug = False
    
    # Reference GCM name
    gcm_name = args.gcm_list_fn
    print('Climate data is:', gcm_name)

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    else:
        rgi_glac_number = input.rgi_glac_number    

    # Select all glaciers in a region
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=rgi_glac_number)
    # Processing needed for netcdf files
    main_glac_rgi_all_float = main_glac_rgi_all.copy()
    main_glac_rgi_all_float.drop(labels=['RGIId'], axis=1, inplace=True)
    main_glac_hyps = modelsetup.import_Husstable(main_glac_rgi_all, input.rgi_regionsO1, input.hyps_filepath,
                                                 input.hyps_filedict, input.hyps_colsdrop)
    dates_table = modelsetup.datesmodelrun(startyear=gcm_startyear, endyear=gcm_endyear, spinupyears=gcm_spinupyears)
    
    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]
        
    # Read GCM names from command file
    if args.gcm_list_fn == input.ref_gcm_name:
        gcm_list = [input.ref_gcm_name]
    else:
        with open(args.gcm_list_fn, 'r') as gcm_fn:
            gcm_list = gcm_fn.read().splitlines()
            rcp_scenario = os.path.basename(args.gcm_list_fn).split('_')[1]
            print('Found %d gcms to process'%(len(gcm_list)))

    # Loop through all GCMs
    for gcm_name in gcm_list:
        print('Processing:', gcm_name)
        # Pack variables for multiprocessing
        list_packed_vars = []
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n = n + 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])

        # Parallel processing
        if args.option_parallels != 0:
            print('Processing in parallel with ' + str(num_cores) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])

    print('Total processing time:', time.time()-time_start, 's')

#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
    # Place local variables in variable explorer
    if args.option_parallels == 0:
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
#        rcp_scenario = main_vars['rcp_scenario']
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        if input.option_synthetic_sim == 1:
            dates_table_synthetic = main_vars['dates_table_synthetic']
            gcm_temp_tile = main_vars['gcm_temp_tile']
            gcm_prec_tile = main_vars['gcm_prec_tile']
            gcm_lr_tile = main_vars['gcm_lr_tile']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_temp_adj = main_vars['gcm_temp_adj']
        gcm_prec_adj = main_vars['gcm_prec_adj']
        gcm_elev_adj = main_vars['gcm_elev_adj']
        main_glac_biasadj = main_vars['main_glac_biasadj']
        gcm_temp_lrglac = main_vars['gcm_lr']
        modelparameters = main_vars['modelparameters']
        glac_wide_massbaltotal = main_vars['glac_wide_massbaltotal']
        glac_wide_area_annual = main_vars['glac_wide_area_annual']
        glac_wide_volume_annual = main_vars['glac_wide_volume_annual']
        glacier_rgi_table = main_vars['glacier_rgi_table']
        glacier_gcm_temp = main_vars['glacier_gcm_temp']
        glacier_gcm_prec = main_vars['glacier_gcm_prec']
        glacier_gcm_elev = main_vars['glacier_gcm_elev']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm'][gcm_spinupyears*12:]
        glacier_area_t0 = main_vars['glacier_area_t0']
        icethickness_t0 = main_vars['icethickness_t0']
        width_t0 = main_vars['width_t0']
        glac_bin_frontalablation = main_vars['glac_bin_frontalablation']
        glac_bin_area_annual = main_vars['glac_bin_area_annual']
        glac_bin_massbalclim_annual = main_vars['glac_bin_massbalclim_annual']
        glac_bin_melt = main_vars['glac_bin_melt']
        glac_bin_acc = main_vars['glac_bin_acc']
        glac_bin_refreeze = main_vars['glac_bin_refreeze']
        glac_bin_temp = main_vars['glac_bin_temp']
        glac_bin_prec = main_vars['glac_bin_prec']
        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']
        modelparameters_all = main_vars['modelparameters_all']
        sim_iters = main_vars['sim_iters']
        mp_idx = main_vars['mp_idx']
        mp_idx_all = main_vars['mp_idx_all']
        netcdf_fn = main_vars['netcdf_fn']
        ds = main_vars['ds']
        output_ds_all = main_vars['output_ds_all']