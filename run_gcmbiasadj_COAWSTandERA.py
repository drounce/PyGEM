"""Run bias adjustments a given climate dataset"""

# Built-in libraries
import os
import argparse
import multiprocessing
import inspect
import time
from time import strftime
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
from scipy.optimize import minimize
import pickle
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import pygemfxns_massbalance as massbalance
import class_climate

#%% SCRIPT-SPECIFIC INPUT
ref_gcm_name = 'COAWST'
gcm_name = 'ERA-Interim'


#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    ref_gcm_name (optional) : str
        reference gcm name
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    progress_bar : int
        Switch for turning the progress bar on or off (default = 0 (off))
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run gcm bias corrections from gcm list in parallel")
    # add arguments
    parser.add_argument('-gcm_file', action='store', type=str, default='gcm_rcpXX_filenames.txt',
                        help='text file full of commands to run')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=2,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-debug', action='store', type=int, default=0,
                        help='Boolean for debugging to turn it on or off (default 0 is off')
    return parser


def main(list_packed_vars):
    """
    Climate data bias adjustment
    
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
        
    Returns
    -------
    csv files of bias adjustment output
        The bias adjustment parameters are output instead of the actual temperature and precipitation to reduce file
        sizes.  Additionally, using the bias adjustment will cause the GCM climate data to use the reference elevation
        since the adjustments were made from the GCM climate data to be consistent with the reference dataset.
    """
    # Unpack variables
    count = list_packed_vars[0]
    chunk = list_packed_vars[1]
    main_glac_rgi_all = list_packed_vars[2]
    chunk_size = list_packed_vars[3]
#    gcm_name = list_packed_vars[4]

    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()
    rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]

    # ===== LOAD OTHER GLACIER DATA =====
    main_glac_rgi = main_glac_rgi_all.iloc[chunk:chunk + chunk_size, :]
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

    # Select dates including future projections
    
    # If reference climate data starts or ends before or after the GCM data, then adjust reference climate data such
    # that the reference and GCM span the same period of time.
    if input.startyear >= input.gcm_startyear:
        ref_startyear = input.startyear
    else:
        ref_startyear = input.gcm_startyear
    if input.endyear <= input.gcm_endyear:
        ref_endyear = input.endyear
    else:
        ref_endyear = input.gcm_endyear
    dates_table_ref = modelsetup.datesmodelrun(startyear=ref_startyear, endyear=ref_endyear, 
                                               spinupyears=input.spinupyears)
    dates_table = modelsetup.datesmodelrun(startyear=input.gcm_startyear, endyear=input.gcm_endyear,
                                           spinupyears=input.spinupyears)

    # ===== LOAD CLIMATE DATA =====
    # GCM climate data
    gcm = class_climate.GCM(name='ERA-Interim')
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    
    # Reference climate data
    ref_gcm = class_climate.GCM(name='COAWST')
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn, main_glac_rgi, 
                                                                     dates_table_ref)
    ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn, main_glac_rgi, 
                                                                     dates_table_ref)
    ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)
    # Use lapse rates from ERA-Interim
    ref_lr = gcm_lr.copy()
    ref_lr_monthly_avg = (ref_lr.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                          .reshape(12,-1).transpose())
    
    # GCM subset to agree with reference time period to calculate bias corrections
    gcm_subset_idx_start = np.where(dates_table.date.values == dates_table_ref.date.values[0])[0][0]
    gcm_subset_idx_end = np.where(dates_table.date.values == dates_table_ref.date.values[-1])[0][0]
    gcm_temp_subset = gcm_temp[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    gcm_prec_subset = gcm_prec[:,gcm_subset_idx_start:gcm_subset_idx_end+1]
    gcm_lr_subset = gcm_lr[:,0:gcm_subset_idx_start:gcm_subset_idx_end+1]

    #%% ===== BIAS CORRECTIONS =====
    # OPTION 2: Adjust temp and prec according to Huss and Hock (2015) accounts for means and interannual variability
    if input.option_bias_adjustment == 2:
        # Bias adjustment parameters
        main_glac_bias_adj_colnames = ['RGIId', 'ref', 'GCM', 'rcp_scenario', 'new_gcmelev']
        main_glac_bias_adj = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_bias_adj_colnames))),
                                          columns=main_glac_bias_adj_colnames)
        main_glac_bias_adj['RGIId'] = main_glac_rgi['RGIId'].values
        main_glac_bias_adj['ref'] = ref_gcm_name
        main_glac_bias_adj['GCM'] = gcm_name
        main_glac_bias_adj['rcp_scenario'] = rcp_scenario
        main_glac_bias_adj['new_gcmelev'] = ref_elev

        tempvar_cols = []
        tempavg_cols = []
        tempadj_cols = []
        precadj_cols = []
        lravg_cols = []
        # Monthly temperature variability
        for n in range(1,13):
            tempvar_colname = 'tempvar_' + str(n)
            main_glac_bias_adj[tempvar_colname] = np.nan
            tempvar_cols.append(tempvar_colname)
        # Monthly mean temperature
        for n in range(1,13):
            tempavg_colname = 'tempavg_' + str(n)
            main_glac_bias_adj[tempavg_colname] = np.nan
            tempavg_cols.append(tempavg_colname)
        # Monthly temperature adjustment
        for n in range(1,13):
            tempadj_colname = 'tempadj_' + str(n)
            main_glac_bias_adj[tempadj_colname] = np.nan
            tempadj_cols.append(tempadj_colname)
        # Monthly precipitation adjustment
        for n in range(1,13):
            precadj_colname = 'precadj_' + str(n)
            main_glac_bias_adj[precadj_colname] = np.nan
            precadj_cols.append(precadj_colname)
        # Monthly mean lapse rate
        for n in range(1,13):
            lravg_cn = 'lravg_' + str(n)
            main_glac_bias_adj[lravg_cn] = np.nan
            lravg_cols.append(lravg_cn)

        # Remove spinup years, so adjustment performed over calibration period
        ref_temp_nospinup = ref_temp[:,input.spinupyears*12:]
        gcm_temp_nospinup = gcm_temp_subset[:,input.spinupyears*12:]
        ref_prec_nospinup = ref_prec[:,input.spinupyears*12:]
        gcm_prec_nospinup = gcm_prec_subset[:,input.spinupyears*12:]
        
        # TEMPERATURE BIAS CORRECTIONS
        # Mean monthly temperature
        ref_temp_monthly_avg = (ref_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_temp_monthly_avg = (gcm_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        
        ref_temp_nospinup_adj = ref_temp_nospinup + ref_lr * (gcm_elev - ref_elev)

#        # Monthly bias adjustment
#        gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
#        # Monthly temperature bias adjusted according to monthly average
#        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
#        # Mean monthly temperature bias adjusted according to monthly average
#        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
#        # Calculate monthly standard deviation of temperature
#        ref_temp_monthly_std = (ref_temp_nospinup.reshape(-1,12).transpose()
#                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
#        gcm_temp_monthly_std = (gcm_temp_nospinup.reshape(-1,12).transpose()
#                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
#        variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
#        # Bias adjusted temperature accounting for monthly mean and variability
#        gcm_temp_bias_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
#        
#        # PRECIPITATION BIAS CORRECTIONS
#        # Calculate monthly mean precipitation
#        ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
#                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
#        gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
#                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
#        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
#        # Bias adjusted precipitation accounting for differences in monthly mean
#        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
#
#        # Record adjustment parameters
#        main_glac_bias_adj[precadj_cols] = bias_adj_prec
#        main_glac_bias_adj[tempvar_cols] = variability_monthly_std
#        main_glac_bias_adj[tempavg_cols] = gcm_temp_monthly_avg
#        main_glac_bias_adj[tempadj_cols] = gcm_temp_monthly_adj
#        main_glac_bias_adj[lravg_cols] = ref_lr_monthly_avg
#
#
#    #%% EXPORT THE ADJUSTMENT VARIABLES (greatly reduces space)
#    # Set up directory to store climate data
#    if os.path.exists(input.output_filepath + 'temp/') == False:
#        os.makedirs(input.output_filepath + 'temp/')
#    # Temperature and precipitation parameters
#    if gcm_name == 'COAWST' or gcm_name == 'ERA-Interim':
#        output_biasadjparams_fn = ('R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_biasadj_opt' + 
#                                   str(input.option_bias_adjustment) + '_' + str(ref_startyear) + '_' + str(ref_endyear) 
#                                   + '--' + str(count) + '.csv')
#    else:
#        output_biasadjparams_fn = ('R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + 
#                                   '_biasadj_opt' + str(input.option_bias_adjustment) + '_' + str(ref_startyear) + '_' 
#                                   + str(ref_endyear) + '--' + str(count) + '.csv')
#    main_glac_bias_adj.to_csv(input.output_filepath + 'temp/' + output_biasadjparams_fn)

    #%% Export variables as global to view in variable explorer
    if args.option_parallels == 0:
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
    print('Reference climate data is:', ref_gcm_name)
    
    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    else:
        rgi_glac_number = input.rgi_glac_number   

    # Select glaciers and define chunks
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2 = 'all',
                                                          rgi_glac_number=input.rgi_glac_number)
    # Define chunk size for parallel processing
    if args.option_parallels != 0:
        num_cores = int(np.min([main_glac_rgi_all.shape[0], args.num_simultaneous_processes]))
        chunk_size = int(np.ceil(main_glac_rgi_all.shape[0] / num_cores))
    else:
        # if not running in parallel, chunk size is all glaciers
        chunk_size = main_glac_rgi_all.shape[0]

    # Read GCM names from command file
#    with open(args.gcm_file, 'r') as gcm_fn:
#        gcm_list = gcm_fn.read().splitlines()
#        rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
#        print('Found %d gcm(s) to process'%(len(gcm_list)))
    gcm_list = [gcm_name]

    # Loop through all GCMs
    for gcm_name in gcm_list:
        # Pack variables for multiprocessing
        list_packed_vars = []
        n = 0
        for chunk in range(0, main_glac_rgi_all.shape[0], chunk_size):
            n += 1
            list_packed_vars.append([n, chunk, main_glac_rgi_all, chunk_size, gcm_name])

        # Parallel processing
        if args.option_parallels != 0:
            print('Processing', gcm_name, 'in parallel')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # No parallel processing
        else:
            print('Processing', gcm_name, 'without parallel')
            # Loop through the chunks and export bias adjustments
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])

#        # Combine bias adjustment parameters into single file
#        output_list = []
#        check_str = 'R' + str(input.rgi_regionsO1[0])
#        # Sorted list of files to merge
#        output_list = []
#        for i in os.listdir(input.output_filepath + 'temp/'):
#            if i.startswith(check_str):
#                output_list.append(i)
#        output_list = sorted(output_list)
#        # Merge files
#        list_count = 0
#        for i in output_list:
#            list_count += 1
#            # Append results
#            if list_count == 1:
#                output_all = pd.read_csv(input.output_filepath + 'temp/' + i, index_col=0)
#            else:
#                output_2join = pd.read_csv(input.output_filepath + 'temp/' + i, index_col=0)
#                output_all = output_all.append(output_2join, ignore_index=True)
#            # Remove file after its been merged
#            os.remove(input.output_filepath + 'temp/' + i)
#        # Export joined files
#        output_all.to_csv(input.biasadj_fp + i.split('--')[0] + '.csv')

    print('Total processing time:', time.time()-time_start, 's')

#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
    # Place local variables in variable explorer
    if args.option_parallels == 0:
        main_vars_list = list(main_vars.keys())
        main_glac_rgi = main_vars['main_glac_rgi']
        main_glac_hyps = main_vars['main_glac_hyps']
        main_glac_icethickness = main_vars['main_glac_icethickness']
        main_glac_width = main_vars['main_glac_width']
        elev_bins = main_vars['elev_bins']
        dates_table = main_vars['dates_table']
        dates_table_ref = main_vars['dates_table_ref']
        
        ref_temp = main_vars['ref_temp']
        ref_prec = main_vars['ref_prec']
        ref_elev = main_vars['ref_elev']
        ref_lr = main_vars['ref_lr']
        ref_lr_monthly_avg = main_vars['ref_lr_monthly_avg']
        gcm_temp = main_vars['gcm_temp']
        gcm_prec = main_vars['gcm_prec']
        gcm_elev = main_vars['gcm_elev']
        gcm_lr = main_vars['gcm_lr']
        gcm_temp_subset = main_vars['gcm_temp_subset']
        gcm_prec_subset = main_vars['gcm_prec_subset']
        gcm_lr_subset = main_vars['gcm_lr_subset']
        main_glac_bias_adj = main_vars['main_glac_bias_adj']
        ref_temp_monthly_avg = main_vars['ref_temp_monthly_avg']
        gcm_temp_monthly_avg = main_vars['gcm_temp_monthly_avg']
        
#        gcm_temp_bias_adj = main_vars['gcm_temp_bias_adj']
        

        

#        # Adjust temperature and precipitation to 'Zmed' so variables can properly be compared
#        glacier_elev_zmed = glacier_rgi_table.loc['Zmed']
#        glacier_ref_temp_zmed = ((glacier_ref_temp + glacier_ref_lrgcm * (glacier_elev_zmed - glacier_ref_elev)
#                                  )[input.spinupyears*12:])
#        glacier_ref_prec_zmed = (glacier_ref_prec * modelparameters['precfactor'])[input.spinupyears*12:]
#        #  recall 'precfactor' is used to adjust for precipitation differences between gcm elev and zmed
#        if input.option_bias_adjustment == 1:
#            glacier_gcm_temp_zmed = ((glacier_gcm_temp_adj + glacier_gcm_lrgcm * (glacier_elev_zmed - glacier_gcm_elev)
#                                      )[input.spinupyears*12:])
#            glacier_gcm_prec_zmed = (glacier_gcm_prec_adj * modelparameters['precfactor'])[input.spinupyears*12:]
#        elif (input.option_bias_adjustment == 2) or (input.option_bias_adjustment == 3):
#            glacier_gcm_temp_zmed = ((glacier_gcm_temp_adj + glacier_gcm_lrgcm * (glacier_elev_zmed - glacier_ref_elev)
#                                      )[input.spinupyears*12:])
#            glacier_gcm_prec_zmed = (glacier_gcm_prec_adj * modelparameters['precfactor'])[input.spinupyears*12:]

    #    # Plot reference vs. GCM temperature and precipitation
    #    # Monthly trends
    #    months = dates_table['date'][input.spinupyears*12:]
    #    years = np.unique(dates_table['wateryear'].values)[input.spinupyears:]
    #    # Temperature
    #    plt.plot(months, glacier_ref_temp_zmed, label='ref_temp')
    #    plt.plot(months, glacier_gcm_temp_zmed, label='gcm_temp')
    #    plt.ylabel('Monthly temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(months, glacier_ref_prec_zmed, label='ref_prec')
    #    plt.plot(months, glacier_gcm_prec_zmed, label='gcm_prec')
    #    plt.ylabel('Monthly precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #
    #    # Annual trends
    #    glacier_ref_temp_zmed_annual = glacier_ref_temp_zmed.reshape(-1,12).mean(axis=1)
    #    glacier_gcm_temp_zmed_annual = glacier_gcm_temp_zmed.reshape(-1,12).mean(axis=1)
    #    glacier_ref_prec_zmed_annual = glacier_ref_prec_zmed.reshape(-1,12).sum(axis=1)
    #    glacier_gcm_prec_zmed_annual = glacier_gcm_prec_zmed.reshape(-1,12).sum(axis=1)
    #    # Temperature
    #    plt.plot(years, glacier_ref_temp_zmed_annual, label='ref_temp')
    #    plt.plot(years, glacier_gcm_temp_zmed_annual, label='gcm_temp')
    #    plt.ylabel('Mean annual temperature [degC]')
    #    plt.legend()
    #    plt.show()
    #    # Precipitation
    #    plt.plot(years, glacier_ref_prec_zmed_annual, label='ref_prec')
    #    plt.plot(years, glacier_gcm_prec_zmed_annual, label='gcm_prec')
    #    plt.ylabel('Total annual precipitation [m]')
    #    plt.legend()
    #    plt.show()
    #    # Mass balance - bar plot
    #    bar_width = 0.35
    #    plt.bar(years, glac_wide_massbaltotal_annual_ref, bar_width, label='ref_MB')
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual_gcm, bar_width, label='gcm_MB')
    #    plt.ylabel('Glacier-wide mass balance [mwea]')
    #    plt.legend()
    #    plt.show()
    #    # Cumulative mass balance - bar plot
    #    glac_wide_massbaltotal_annual_ref_cumsum = np.cumsum(glac_wide_massbaltotal_annual_ref)
    #    glac_wide_massbaltotal_annual_gcm_cumsum = np.cumsum(glac_wide_massbaltotal_annual_gcm)
    #    bar_width = 0.35
    #    plt.bar(years, glac_wide_massbaltotal_annual_ref_cumsum, bar_width, label='ref_MB')
    #    plt.bar(years+bar_width, glac_wide_massbaltotal_annual_gcm_cumsum, bar_width, label='gcm_MB')
    #    plt.ylabel('Cumulative glacier-wide mass balance [mwe]')
    #    plt.legend()
    #    plt.show()
    #    # Histogram of differences
    #    mb_dif = main_glac_bias_adj['ref_mb_mwea'] - main_glac_bias_adj['gcm_mb_mwea']
    #    plt.hist(mb_dif)
    #    plt.show()
