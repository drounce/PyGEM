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

#%% ===== SCRIPT SPECIFIC INPUT DATA =====
option_run_mb = 0 # only for options 2 and 3


#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    gcm_file : str
        full filepath to text file that has list of gcm names to be processed
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
    parser.add_argument('gcm_file', action='store', type=str, default='gcm_rcpXX_filenames.txt',
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
    gcm_name = list_packed_vars[4]

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
    # Reference climate data
    ref_gcm = class_climate.GCM(name=input.ref_gcm_name)
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    ref_temp, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.temp_fn, ref_gcm.temp_vn, main_glac_rgi, 
                                                                     dates_table_ref)
    ref_prec, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.prec_fn, ref_gcm.prec_vn, main_glac_rgi, 
                                                                     dates_table_ref)
    ref_elev = ref_gcm.importGCMfxnearestneighbor_xarray(ref_gcm.elev_fn, ref_gcm.elev_vn, main_glac_rgi)
    ref_lr, ref_dates = ref_gcm.importGCMvarnearestneighbor_xarray(ref_gcm.lr_fn, ref_gcm.lr_vn, main_glac_rgi, 
                                                                   dates_table_ref)
    ref_lr_monthly_avg = (ref_lr.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
                          .reshape(12,-1).transpose())
    
    # GCM climate data
    if gcm_name == 'ERA-Interim' or gcm_name == 'COAWST':
        gcm = class_climate.GCM(name=gcm_name)
    else:
        gcm = class_climate.GCM(name=gcm_name, rcp_scenario=rcp_scenario)
    # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    if gcm_name == 'ERA-Interim':
        gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    else:
        gcm_lr = np.tile(ref_lr_monthly_avg, int(gcm_temp.shape[1]/12))
    # COAWST data has two domains, so need to merge the two domains
    if gcm_name == 'COAWST':
        gcm_temp_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn_d01, gcm.temp_vn, main_glac_rgi, 
                                                                         dates_table)
        gcm_prec_d01, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn_d01, gcm.prec_vn, main_glac_rgi, 
                                                                         dates_table)
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
        main_glac_biasadj_colnames = ['RGIId', 'ref', 'GCM', 'rcp_scenario', 'new_gcmelev']
        main_glac_biasadj = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_biasadj_colnames))),
                                          columns=main_glac_biasadj_colnames)
        main_glac_biasadj['RGIId'] = main_glac_rgi['RGIId'].values
        main_glac_biasadj['ref'] = input.ref_gcm_name
        main_glac_biasadj['GCM'] = gcm_name
        main_glac_biasadj['rcp_scenario'] = rcp_scenario
        main_glac_biasadj['new_gcmelev'] = ref_elev

        tempvar_cols = []
        tempavg_cols = []
        tempadj_cols = []
        precadj_cols = []
        lravg_cols = []
        # Monthly temperature variability
        for n in range(1,13):
            tempvar_colname = 'tempvar_' + str(n)
            main_glac_biasadj[tempvar_colname] = np.nan
            tempvar_cols.append(tempvar_colname)
        # Monthly mean temperature
        for n in range(1,13):
            tempavg_colname = 'tempavg_' + str(n)
            main_glac_biasadj[tempavg_colname] = np.nan
            tempavg_cols.append(tempavg_colname)
        # Monthly temperature adjustment
        for n in range(1,13):
            tempadj_colname = 'tempadj_' + str(n)
            main_glac_biasadj[tempadj_colname] = np.nan
            tempadj_cols.append(tempadj_colname)
        # Monthly precipitation adjustment
        for n in range(1,13):
            precadj_colname = 'precadj_' + str(n)
            main_glac_biasadj[precadj_colname] = np.nan
            precadj_cols.append(precadj_colname)
        # Monthly mean lapse rate
        for n in range(1,13):
            lravg_cn = 'lravg_' + str(n)
            main_glac_biasadj[lravg_cn] = np.nan
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
        # Monthly bias adjustment
        gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
        # Monthly temperature bias adjusted according to monthly average
        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Mean monthly temperature bias adjusted according to monthly average
        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
        # Calculate monthly standard deviation of temperature
        ref_temp_monthly_std = (ref_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
        gcm_temp_monthly_std = (gcm_temp_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).std(1).reshape(12,-1).transpose())
        variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
        # Bias adjusted temperature accounting for monthly mean and variability
        gcm_temp_bias_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
        
        # PRECIPITATION BIAS CORRECTIONS
        # Calculate monthly mean precipitation
        ref_prec_monthly_avg = (ref_prec_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        gcm_prec_monthly_avg = (gcm_prec_nospinup.reshape(-1,12).transpose()
                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
        # Bias adjusted precipitation accounting for differences in monthly mean
        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))

        # Record adjustment parameters
        main_glac_biasadj[precadj_cols] = bias_adj_prec
        main_glac_biasadj[tempvar_cols] = variability_monthly_std
        main_glac_biasadj[tempavg_cols] = gcm_temp_monthly_avg
        main_glac_biasadj[tempadj_cols] = gcm_temp_monthly_adj
        main_glac_biasadj[lravg_cols] = ref_lr_monthly_avg
       
#    #%%
#    # OPTION 1: Adjust temp and prec such that ref and GCM mass balances over calibration period are equal
#    elif input.option_bias_adjustment == 1:
#        # Model parameters
#        if input.option_bias_adjustment == 1:
#            main_glac_modelparams_all = pd.read_csv(filepath_modelparams + filename_modelparams, index_col=0)
#            main_glac_modelparams = main_glac_modelparams_all.loc[main_glac_rgi['O1Index'].values, :] 
#        # Bias adjustment parameters
#        main_glac_biasadj_colnames = ['RGIId', 'ref', 'GCM', 'rcp_scenario', 'temp_adj', 'prec_adj', 'ref_mb_mwea',
#                                       'ref_vol_change_perc', 'gcm_mb_mwea', 'gcm_vol_change_perc', 'lrgcm', 'lrglac',
#                                       'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
#        main_glac_biasadj = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_biasadj_colnames))),
#                                          columns=main_glac_biasadj_colnames)
#        main_glac_biasadj['RGIId'] = main_glac_rgi['RGIId'].values
#        main_glac_biasadj['ref'] = input.ref_gcm_name
#        main_glac_biasadj['GCM'] = gcm_name
#        main_glac_biasadj['rcp_scenario'] = rcp_scenario
#        main_glac_biasadj[modelparams_colnames] = main_glac_modelparams[modelparams_colnames].values
#
#        # BIAS ADJUSTMENT CALCULATIONS
#        for glac in range(main_glac_rgi.shape[0]):
#            if glac%200 == 0:
#                print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
#            glacier_rgi_table = main_glac_rgi.iloc[glac, :]
#            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
#            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
#            modelparameters = main_glac_modelparams.loc[main_glac_modelparams.index.values[glac],modelparams_colnames]
#            glac_idx_t0 = glacier_area_t0.nonzero()[0]
#
#            if icethickness_t0.max() > 0:
#                surfacetype, firnline_idx = massbalance.surfacetypebinsinitial(glacier_area_t0, glacier_rgi_table,
#                                                                               elev_bins)
#                surfacetype_ddf_dict = massbalance.surfacetypeDDFdict(modelparameters, option_DDF_firn=0)
#                #  option_DDF_firn=0 uses DDF_snow in accumulation area because not account for snow vs. firn here
#                surfacetype_ddf = np.zeros(glacier_area_t0.shape)
#                for surfacetype_idx in surfacetype_ddf_dict:
#                    surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
#                # Reference data
#                glacier_ref_temp = ref_temp[glac,:]
#                glacier_ref_prec = ref_prec[glac,:]
#                glacier_ref_elev = ref_elev[glac]
#                glacier_ref_lrgcm = ref_lr[glac,:]
#                glacier_ref_lrglac = ref_lr[glac,:]
#                # GCM data
#                glacier_gcm_temp = gcm_temp_subset[glac,:]
#                glacier_gcm_prec = gcm_prec_subset[glac,:]
#                glacier_gcm_elev = gcm_elev[glac]
#                glacier_gcm_lrgcm = gcm_lr_subset[glac,:]
#                glacier_gcm_lrglac = gcm_lr_subset[glac,:]
#
#                # AIR TEMPERATURE: Downscale the gcm temperature [deg C] to each bin
#                if input.option_temp2bins == 1:
#                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
#                    glac_bin_temp_ref = (
#                            glacier_ref_temp + glacier_ref_lrgcm *
#                            (glacier_rgi_table.loc[input.option_elev_ref_downscale] - glacier_ref_elev) +
#                            glacier_ref_lrglac * (elev_bins -
#                            glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis]
#                            + modelparameters['tempchange'])
#                    glac_bin_temp_gcm = (
#                            glacier_gcm_temp + glacier_gcm_lrgcm *
#                            (glacier_rgi_table.loc[input.option_elev_ref_downscale] - glacier_gcm_elev) +
#                            glacier_gcm_lrglac * (elev_bins -
#                            glacier_rgi_table.loc[input.option_elev_ref_downscale])[:,np.newaxis]
#                            + modelparameters['tempchange'])
#                # remove off-glacier values
#                glac_bin_temp_ref[glacier_area_t0==0,:] = 0
#                glac_bin_temp_gcm[glacier_area_t0==0,:] = 0
#                # TEMPERATURE BIAS CORRECTIONS
#                # Energy available for melt [degC day]
#                # daysinmonth = dates_table_ref['daysinmonth'].values  
#                melt_energy_available_ref = glac_bin_temp_ref * daysinmonth
#                melt_energy_available_ref[melt_energy_available_ref < 0] = 0
#                # Melt [mwe for each month]
#                melt_ref = melt_energy_available_ref * surfacetype_ddf[:,np.newaxis]
#                # Melt volume total [mwe * km2]
#                melt_vol_ref = (melt_ref * glacier_area_t0[:,np.newaxis]).sum()
#                # Optimize bias adjustment such that PDD are equal
#                def objective(bias_adj_glac):
#                    glac_bin_temp_gcm_adj = glac_bin_temp_gcm + bias_adj_glac
#                    melt_energy_available_gcm = glac_bin_temp_gcm_adj * daysinmonth
#                    melt_energy_available_gcm[melt_energy_available_gcm < 0] = 0
#                    melt_gcm = melt_energy_available_gcm * surfacetype_ddf[:,np.newaxis]
#                    melt_vol_gcm = (melt_gcm * glacier_area_t0[:,np.newaxis]).sum()
#                    return abs(melt_vol_ref - melt_vol_gcm)
#                # - initial guess
#                bias_adj_init = 0
#                # - run optimization
#                bias_adj_temp_opt = minimize(objective, bias_adj_init, method='SLSQP', tol=1e-5)
#                bias_adj_temp_init = bias_adj_temp_opt.x
#                glac_bin_temp_gcm_adj = glac_bin_temp_gcm + bias_adj_temp_init
#                # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
#                glac_bin_acc_ref = np.zeros(glac_bin_temp_ref.shape)
#                glac_bin_acc_gcm = np.zeros(glac_bin_temp_ref.shape)
#                glac_bin_prec_ref = np.zeros(glac_bin_temp_ref.shape)
#                glac_bin_prec_gcm = np.zeros(glac_bin_temp_ref.shape)
#                if input.option_prec2bins == 1:
#                    # Precipitation using precipitation factor and precipitation gradient
#                    #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
#                    glac_bin_precsnow_ref = (glacier_ref_prec * modelparameters['precfactor'] *
#                                             (1 + modelparameters['precgrad'] * (elev_bins -
#                                             glacier_rgi_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
#                    glac_bin_precsnow_gcm = (glacier_gcm_prec * modelparameters['precfactor'] *
#                                             (1 + modelparameters['precgrad'] * (elev_bins -
#                                             glacier_rgi_table.loc[input.option_elev_ref_downscale]))[:,np.newaxis])
#                # Option to adjust prec of uppermost 25% of glacier for wind erosion and reduced moisture content
#                if input.option_preclimit == 1:
#                    # If elevation range > 1000 m, apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
#                    if elev_bins[glac_idx_t0[-1]] - elev_bins[glac_idx_t0[0]] > 1000:
#                        # Indices of upper 25%
#                        glac_idx_upper25 = glac_idx_t0[(glac_idx_t0 - glac_idx_t0[0] + 1) / glac_idx_t0.shape[0] * 100 > 75]
#                        # Exponential decay according to elevation difference from the 75% elevation
#                        #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
#                        glac_bin_precsnow_ref[glac_idx_upper25,:] = (
#                                glac_bin_precsnow_ref[glac_idx_upper25[0],:] * np.exp(-1*(elev_bins[glac_idx_upper25] -
#                                elev_bins[glac_idx_upper25[0]]) / (elev_bins[glac_idx_upper25[-1]] -
#                                elev_bins[glac_idx_upper25[0]]))[:,np.newaxis])
#                        glac_bin_precsnow_gcm[glac_idx_upper25,:] = (
#                                glac_bin_precsnow_gcm[glac_idx_upper25[0],:] * np.exp(-1*(elev_bins[glac_idx_upper25] -
#                                elev_bins[glac_idx_upper25[0]]) / (elev_bins[glac_idx_upper25[-1]] -
#                                elev_bins[glac_idx_upper25[0]]))[:,np.newaxis])
#                        # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
#                        for month in range(glac_bin_precsnow_ref.shape[1]):
#                            glac_bin_precsnow_ref[glac_idx_upper25[(glac_bin_precsnow_ref[glac_idx_upper25,month] < 0.875 *
#                            glac_bin_precsnow_ref[glac_idx_t0,month].max()) &
#                            (glac_bin_precsnow_ref[glac_idx_upper25,month] != 0)], month] = (
#                                                                0.875 * glac_bin_precsnow_ref[glac_idx_t0,month].max())
#                            glac_bin_precsnow_gcm[glac_idx_upper25[(glac_bin_precsnow_gcm[glac_idx_upper25,month] < 0.875 *
#                            glac_bin_precsnow_gcm[glac_idx_t0,month].max()) &
#                            (glac_bin_precsnow_gcm[glac_idx_upper25,month] != 0)], month] = (
#                                                                0.875 * glac_bin_precsnow_gcm[glac_idx_t0,month].max())
#                # Separate total precipitation into liquid (glac_bin_prec) and solid (glac_bin_acc)
#                if input.option_accumulation == 1:
#                    # if temperature above threshold, then rain
#                    glac_bin_prec_ref[glac_bin_temp_ref > modelparameters['tempsnow']] = (
#                        glac_bin_precsnow_ref[glac_bin_temp_ref > modelparameters['tempsnow']])
#                    glac_bin_prec_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow']] = (
#                        glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow']])
#                    # if temperature below threshold, then snow
#                    glac_bin_acc_ref[glac_bin_temp_ref <= modelparameters['tempsnow']] = (
#                        glac_bin_precsnow_ref[glac_bin_temp_ref <= modelparameters['tempsnow']])
#                    glac_bin_acc_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow']] = (
#                        glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow']])
#                elif input.option_accumulation == 2:
#                    # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
#                    glac_bin_prec_ref = (
#                            (1/2 + (glac_bin_temp_ref - modelparameters['tempsnow']) / 2) * glac_bin_precsnow_ref)
#                    glac_bin_prec_gcm = (
#                            (1/2 + (glac_bin_temp_gcm_adj - modelparameters['tempsnow']) / 2) * glac_bin_precsnow_gcm)
#                    glac_bin_acc_ref = glac_bin_precsnow_ref - glac_bin_prec_ref
#                    glac_bin_acc_gcm = glac_bin_precsnow_gcm - glac_bin_prec_gcm
#                    # If temperature above maximum threshold, then all rain
#                    glac_bin_prec_ref[glac_bin_temp_ref > modelparameters['tempsnow'] + 1] = (
#                        glac_bin_precsnow_ref[glac_bin_temp_ref > modelparameters['tempsnow'] + 1])
#                    glac_bin_prec_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow'] + 1] = (
#                        glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow'] + 1])
#                    glac_bin_acc_ref[glac_bin_temp_ref > modelparameters['tempsnow'] + 1] = 0
#                    glac_bin_acc_gcm[glac_bin_temp_gcm_adj > modelparameters['tempsnow'] + 1] = 0
#                    # If temperature below minimum threshold, then all snow
#                    glac_bin_acc_ref[glac_bin_temp_ref <= modelparameters['tempsnow'] - 1] = (
#                            glac_bin_precsnow_ref[glac_bin_temp_ref <= modelparameters['tempsnow'] - 1])
#                    glac_bin_acc_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow'] - 1] = (
#                            glac_bin_precsnow_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow'] - 1])
#                    glac_bin_prec_ref[glac_bin_temp_ref <= modelparameters['tempsnow'] - 1] = 0
#                    glac_bin_prec_gcm[glac_bin_temp_gcm_adj <= modelparameters['tempsnow'] - 1] = 0
#                # remove off-glacier values
#                glac_bin_acc_ref[glacier_area_t0==0,:] = 0
#                glac_bin_acc_gcm[glacier_area_t0==0,:] = 0
#                glac_bin_prec_ref[glacier_area_t0==0,:] = 0
#                glac_bin_prec_gcm[glacier_area_t0==0,:] = 0
#                # account for hypsometry
#                glac_bin_acc_ref_warea = glac_bin_acc_ref * glacier_area_t0[:,np.newaxis]
#                glac_bin_acc_gcm_warea = glac_bin_acc_gcm * glacier_area_t0[:,np.newaxis]
#                # precipitation bias adjustment
#                bias_adj_prec_init = glac_bin_acc_ref_warea.sum() / glac_bin_acc_gcm_warea.sum()
#
#                # BIAS ADJUSTMENT PARAMETER OPTIMIZATION such that mass balance between two datasets are equal
#                bias_adj_params = np.zeros((2))
#                bias_adj_params[0] = bias_adj_temp_init
#                bias_adj_params[1] = bias_adj_prec_init
#
#                def objective_2(bias_adj_params):
#                    # Reference data
#                    # Mass balance
#                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
#                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
#                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
#                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
#                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
#                                                   width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec,
#                                                   glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac,
#                                                   dates_table_ref, option_areaconstant=1))
#                    # Annual glacier-wide mass balance [m w.e.]
#                    glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#                    # Average annual glacier-wide mass balance [m w.e.a.]
#                    mb_mwea_ref = glac_wide_massbaltotal_annual_ref.mean()
#
#                    # GCM data
#                    # Bias corrections
#                    glacier_gcm_temp_adj = glacier_gcm_temp + bias_adj_params[0]
#                    glacier_gcm_prec_adj = glacier_gcm_prec * bias_adj_params[1]
#
#                    # Mass balance
#                    (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
#                     glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
#                     glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
#                     glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
#                     glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                        massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
#                                                   width_t0, elev_bins, glacier_gcm_temp_adj, glacier_gcm_prec_adj,
#                                                   glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac,
#                                                   dates_table_ref, option_areaconstant=1))
#                    # Annual glacier-wide mass balance [m w.e.]
#                    glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#                    # Average annual glacier-wide mass balance [m w.e.a.]
#                    mb_mwea_gcm = glac_wide_massbaltotal_annual_gcm.mean()
#                    return abs(mb_mwea_ref - mb_mwea_gcm)
#                # CONSTRAINTS
#                #  everything goes on one side of the equation compared to zero
#                def constraint_temp_prec(bias_adj_params):
#                    return -1 * (bias_adj_params[0] * (bias_adj_params[1] - 1))
#                    #  To avoid increases/decreases in temp compensating for increases/decreases in prec, respectively,
#                    #  ensure that if temp increases, then prec decreases, and vice versa.  This works because
#                    #  (prec_adj - 1) is positive or negative for increases or decrease, respectively, so multiplying
#                    #  this by temp_adj gives a positive or negative value.  We want it to always be negative, but since
#                    #  inequality constraint is for >= 0, we multiply it by -1.
#                # Define constraint type for each function
#                con_temp_prec = {'type':'ineq', 'fun':constraint_temp_prec}
#                #  inequalities are non-negative, i.e., >= 0
#                # Select constraints used to optimize precfactor
#                cons = [con_temp_prec]
#                # INITIAL GUESS
#                bias_adj_params_init = bias_adj_params
#                # Run the optimization
#                bias_adj_params_opt_raw = minimize(objective_2, bias_adj_params_init, method='SLSQP', constraints=cons,
#                                                   tol=1e-3)
#                # Record the optimized parameters
#                bias_adj_params_opt = bias_adj_params_opt_raw.x
#                main_glac_biasadj.loc[glac, ['temp_adj', 'prec_adj']] = bias_adj_params_opt
#
#                # Compute mass balances to have output data
#                # Reference data
#                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
#                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
#                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
#                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
#                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
#                                               width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec,
#                                               glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac,
#                                               dates_table_ref, option_areaconstant=1))
#                # Annual glacier-wide mass balance [m w.e.]
#                glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#                # Average annual glacier-wide mass balance [m w.e.a.]
#                mb_mwea_ref = glac_wide_massbaltotal_annual_ref.mean()
#                #  units: m w.e. based on initial area
#                # Volume change [%]
#                if icethickness_t0.max() > 0:
#                    glac_vol_change_perc_ref = (mb_mwea_ref / 1000 * glac_wide_area_annual[0] *
#                                                glac_wide_massbaltotal_annual_ref.shape[0] / glac_wide_volume_annual[0]
#                                                * 100)
#                # Record reference results
#                main_glac_biasadj.loc[glac, ['ref_mb_mwea', 'ref_vol_change_perc']] = (
#                        [mb_mwea_ref, glac_vol_change_perc_ref])
#
#                # Climate data
#                # Bias corrections
#                glacier_gcm_temp_adj = glacier_gcm_temp + bias_adj_params_opt[0]
#                glacier_gcm_prec_adj = glacier_gcm_prec * bias_adj_params_opt[1]
#                (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
#                 glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
#                 glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
#                 glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
#                 glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                    massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
#                                               width_t0, elev_bins, glacier_gcm_temp_adj, glacier_gcm_prec_adj,
#                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac,
#                                               dates_table_ref, option_areaconstant=1))
#                # Annual glacier-wide mass balance [m w.e.]
#                glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#                # Average annual glacier-wide mass balance [m w.e.a.]
#                mb_mwea_gcm = glac_wide_massbaltotal_annual_gcm.mean()
#                #  units: m w.e. based on initial area
#                # Volume change [%]
#                if icethickness_t0.max() > 0:
#                    glac_vol_change_perc_gcm = (mb_mwea_gcm / 1000 * glac_wide_area_annual[0] *
#                                                glac_wide_massbaltotal_annual_gcm.shape[0] / glac_wide_volume_annual[0]
#                                                * 100)
#                # Record GCM results
#                main_glac_biasadj.loc[glac, ['gcm_mb_mwea', 'gcm_vol_change_perc']] = (
#                        [mb_mwea_gcm, glac_vol_change_perc_gcm])
#    
#    #%%
#    # OPTION 3: Adjust temp and prec such mean monthly temp and mean annual precipitation are equal
#    elif input.option_bias_adjustment == 3:
#        # Bias adjustment parameters
#        main_glac_biasadj_colnames = ['RGIId', 'ref', 'GCM', 'rcp_scenario', 'ref_mb_mwea', 'ref_vol_change_perc',
#                                       'gcm_mb_mwea', 'gcm_vol_change_perc', 'new_gcmelev', 'lrgcm', 'lrglac',
#                                       'precfactor', 'precgrad', 'ddfsnow', 'ddfice', 'tempsnow', 'tempchange']
#        main_glac_biasadj = pd.DataFrame(np.zeros((main_glac_rgi.shape[0],len(main_glac_biasadj_colnames))),
#                                          columns=main_glac_biasadj_colnames)
#        main_glac_biasadj['RGIId'] = main_glac_rgi['RGIId'].values
#        main_glac_biasadj['ref'] = input.ref_gcm_name
#        main_glac_biasadj['GCM'] = gcm_name
#        main_glac_biasadj['rcp_scenario'] = rcp_scenario
#        main_glac_biasadj['new_gcmelev'] = ref_elev
#        main_glac_biasadj[modelparams_colnames] = main_glac_modelparams[modelparams_colnames].values
#
#        tempadj_cols = []
#        precadj_cols = []
#        # Monthly temperature adjustment
#        for n in range(1,13):
#            tempadj_colname = 'tempadj_' + str(n)
#            main_glac_biasadj[tempadj_colname] = np.nan
#            tempadj_cols.append(tempadj_colname)
#        # Monthly precipitation adjustment
#        for n in range(1,13):
#            precadj_colname = 'precadj_' + str(n)
#            main_glac_biasadj[precadj_colname] = np.nan
#            precadj_cols.append(precadj_colname)
#
#        # Remove spinup years, so adjustment performed over calibration period
#        ref_temp_nospinup = ref_temp[:,input.spinupyears*12:]
#        gcm_temp_nospinup = gcm_temp_subset[:,input.spinupyears*12:]
#        ref_prec_nospinup = ref_prec[:,input.spinupyears*12:]
#        gcm_prec_nospinup = gcm_prec_subset[:,input.spinupyears*12:]
#        # TEMPERATURE BIAS CORRECTIONS
#        # Mean monthly temperature
#        ref_temp_monthly_avg = (ref_temp_nospinup.reshape(-1,12).transpose()
#                                .reshape(-1,int(ref_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
#        gcm_temp_monthly_avg = (gcm_temp_nospinup.reshape(-1,12).transpose()
#                                .reshape(-1,int(gcm_temp_nospinup.shape[1]/12)).mean(1).reshape(12,-1).transpose())
#        # Monthly bias adjustment
#        bias_adj_temp = ref_temp_monthly_avg - gcm_temp_monthly_avg
#        # Bias adjusted temperature accounting for monthly mean
#        gcm_temp_bias_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
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
#        main_glac_biasadj[precadj_cols] = bias_adj_prec
#        main_glac_biasadj[tempadj_cols] = bias_adj_temp
#
#    # MASS BALANCE: compute for model comparisons
#    if ((input.option_bias_adjustment == 2) or (input.option_bias_adjustment == 3)) and (option_run_mb == 1):
#        # Compute mass balances to have output data
#        for glac in range(main_glac_rgi.shape[0]):
#            if glac%500 == 0:
#                print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])
#            glacier_rgi_table = main_glac_rgi.iloc[glac, :]
#            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)
#            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
#            modelparameters = main_glac_modelparams.loc[main_glac_modelparams.index.values[glac],modelparams_colnames]
#            glac_idx_t0 = glacier_area_t0.nonzero()[0]
#
#            # Reference data
#            glacier_ref_temp = ref_temp[glac,:]
#            glacier_ref_prec = ref_prec[glac,:]
#            glacier_ref_elev = ref_elev[glac]
#            glacier_ref_lrgcm = ref_lr[glac,:]
#            glacier_ref_lrglac = ref_lr[glac,:]
#            # GCM data
#            glacier_gcm_temp_adj = gcm_temp_bias_adj[glac,:]
#            glacier_gcm_prec_adj = gcm_prec_bias_adj[glac,:]
#            glacier_gcm_elev = ref_elev[glac]
#            #  using the REFERENCE elev here because the adjusted temp is corrected for the mean ref temp already
#            glacier_gcm_lrgcm = gcm_lr_subset[glac,:]
#            glacier_gcm_lrglac = gcm_lr_subset[glac,:]
#
#            # Mass balance
#            # Reference data
#            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
#             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
#             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
#             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
#             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
#                                           width_t0, elev_bins, glacier_ref_temp, glacier_ref_prec,
#                                           glacier_ref_elev, glacier_ref_lrgcm, glacier_ref_lrglac,
#                                           dates_table_ref, option_areaconstant=1))
#            # Annual glacier-wide mass balance [m w.e.]
#            glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#            # Average annual glacier-wide mass balance [m w.e.a.]
#            mb_mwea_ref = glac_wide_massbaltotal_annual_ref.mean()
#            #  units: m w.e. based on initial area
#            # Volume change [%]
#            if icethickness_t0.max() > 0:
#                glac_vol_change_perc_ref = (mb_mwea_ref / 1000 * glac_wide_area_annual[0] *
#                                            glac_wide_massbaltotal_annual_ref.shape[0] / glac_wide_volume_annual[0]
#                                            * 100)
#            main_glac_biasadj.loc[glac,'ref_mb_mwea'] = mb_mwea_ref
#            main_glac_biasadj.loc[glac,'ref_vol_change_perc'] = glac_vol_change_perc_ref
#
#            # GCM data
#            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt,
#             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual,
#             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual,
#             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack,
#             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0,
#                                           width_t0, elev_bins, glacier_gcm_temp_adj, glacier_gcm_prec_adj,
#                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac,
#                                           dates_table_ref, option_areaconstant=1))
#            # Annual glacier-wide mass balance [m w.e.]
#            glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#            # Average annual glacier-wide mass balance [m w.e.a.]
#            mb_mwea_gcm = glac_wide_massbaltotal_annual_gcm.mean()
#            #  units: m w.e. based on initial area
#            # Volume change [%]
#            if icethickness_t0.max() > 0:
#                glac_vol_change_perc_gcm = (mb_mwea_gcm / 1000 * glac_wide_area_annual[0] *
#                                            glac_wide_massbaltotal_annual_ref.shape[0] / glac_wide_volume_annual[0]
#                                            * 100)
#
#            main_glac_biasadj.loc[glac,'gcm_mb_mwea'] = mb_mwea_gcm
#            main_glac_biasadj.loc[glac,'gcm_vol_change_perc'] = glac_vol_change_perc_gcm


    #%% EXPORT THE ADJUSTMENT VARIABLES (greatly reduces space)
    # Set up directory to store climate data
    if os.path.exists(input.output_filepath + 'temp/') == False:
        os.makedirs(input.output_filepath + 'temp/')
    # Temperature and precipitation parameters
    if gcm_name == 'COAWST':
        output_biasadjparams_fn = ('R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_biasadj_opt' + 
                                   str(input.option_bias_adjustment) + '_' + str(ref_startyear) + '_' + str(ref_endyear) 
                                   + '--' + str(count) + '.csv')
    else:
        output_biasadjparams_fn = ('R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario + 
                                   '_biasadj_opt' + str(input.option_bias_adjustment) + '_' + str(ref_startyear) + '_' 
                                   + str(ref_endyear) + '--' + str(count) + '.csv')
    main_glac_biasadj.to_csv(input.output_filepath + 'temp/' + output_biasadjparams_fn)

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
    print('Reference climate data is:', input.ref_gcm_name)
    
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
    with open(args.gcm_file, 'r') as gcm_fn:
        gcm_list = gcm_fn.read().splitlines()
        rcp_scenario = os.path.basename(args.gcm_file).split('_')[1]
        print('Found %d gcm(s) to process'%(len(gcm_list)))

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

        # Combine bias adjustment parameters into single file
        output_list = []
        if gcm_name == 'COAWST':
            check_str = 'R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name
        else:
            check_str = 'R' + str(input.rgi_regionsO1[0]) + '_' + gcm_name + '_' + rcp_scenario
        # Sorted list of files to merge
        output_list = []
        for i in os.listdir(input.output_filepath + 'temp/'):
            if i.startswith(check_str):
                output_list.append(i)
        output_list = sorted(output_list)
        # Merge files
        list_count = 0
        for i in output_list:
            list_count += 1
            # Append results
            if list_count == 1:
                output_all = pd.read_csv(input.output_filepath + 'temp/' + i, index_col=0)
            else:
                output_2join = pd.read_csv(input.output_filepath + 'temp/' + i, index_col=0)
                output_all = output_all.append(output_2join, ignore_index=True)
            # Remove file after its been merged
            os.remove(input.output_filepath + 'temp/' + i)
        # Sort the gcm bias adjustment dataframe
        output_all_cns = list(output_all.columns)
        output_all['RGIId_float'] = (np.array([np.str.split(output_all['RGIId'][x],'-')[1] 
                                     for x in range(output_all.shape[0])]).astype(float))
        output_all['glacno'] = ((output_all['RGIId_float'] % 1) * 10**5).round(0).astype(int)
        output_all = output_all.sort_values(['glacno'])
        output_all = output_all[output_all_cns]
        output_all.reset_index(drop=True, inplace=True)
        # Set up directory to store bias adjustment data
        if not os.path.exists(input.biasadj_fp):
            os.makedirs(input.biasadj_fp)
        # Export joined files
        output_all.to_csv(input.biasadj_fp + i.split('--')[0] + '.csv')

    print('Total processing time:', time.time()-time_start, 's')

#%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
    # Place local variables in variable explorer
    if args.option_parallels == 0:
        main_vars_list = list(main_vars.keys())
        gcm_name = main_vars['gcm_name']
        rcp_scenario = main_vars['rcp_scenario']
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
        main_glac_biasadj = main_vars['main_glac_biasadj']
        gcm_temp_bias_adj = main_vars['gcm_temp_bias_adj']
        
#        glacier_rgi_table = main_vars['glacier_rgi_table']
#        glacier_ref_temp = main_vars['glacier_ref_temp']
#        glacier_ref_prec = main_vars['glacier_ref_prec']
#        glacier_ref_elev = main_vars['glacier_ref_elev']
#        glacier_ref_lrgcm = main_vars['glacier_ref_lrgcm']
#        glacier_gcm_temp_adj = main_vars['glacier_gcm_temp_adj']
#        glacier_gcm_prec_adj = main_vars['glacier_gcm_prec_adj']
#        glacier_gcm_elev = main_vars['glacier_gcm_elev']
#        glacier_gcm_lrgcm = main_vars['glacier_gcm_lrgcm']
#        modelparameters = main_vars['modelparameters']
#        glac_wide_massbaltotal_annual_gcm = main_vars['glac_wide_massbaltotal_annual_gcm']
#        glac_wide_massbaltotal_annual_ref = main_vars['glac_wide_massbaltotal_annual_ref']
#        main_glac_biasadj = main_vars['main_glac_biasadj']
#        glacier_area_t0 = main_vars['glacier_area_t0']
#        icethickness_t0 = main_vars['icethickness_t0']
#        width_t0 = main_vars['width_t0']
        

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
    #    mb_dif = main_glac_biasadj['ref_mb_mwea'] - main_glac_biasadj['gcm_mb_mwea']
    #    plt.hist(mb_dif)
    #    plt.show()
