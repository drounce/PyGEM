""" RETRIEVE PRIOR PARAMETERS """

# Built-in libraries
import argparse
import multiprocessing
import os
import glob
# External libraries
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize
import xarray as xr
# Local libraries
import class_climate
import class_mbdata
import pygem_input as input
import pygemfxns_massbalance as massbalance
import pygemfxns_modelsetup as modelsetup


def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    option_parallels (optional) : int
        switch to use parallels or not
    spc_region (optional) : str
        RGI region number for supercomputer 
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers that used to run batches on the supercomputer
    pickle_glacno : int
        switch to pickle the glacier numbers
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-spc_region', action='store', type=int, default=None,
                        help='rgi region number for supercomputer')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')    
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    parser.add_argument('-pickle_glacno', action='store', type=int, default=0,
                        help='Switch to pickle glacier numbers or not (1 - pickle, 0 - do not)')
    return parser

def load_glacierdata_byglacno(glac_no, option_loadhyps_climate=1):
    """ Load glacier data, climate data, and calibration data for list of glaciers 
    
    Parameters
    ----------
    glac_no : list
        list of glacier numbers (ex. ['13.0001', 15.00001'])
    
    Returns
    -------
    main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, gcm_temp, gcm_prec, gcm_elev, gcm_lr, 
    cal_data, dates_table
    """
    glac_no_byregion = {}
    regions = [int(i.split('.')[0]) for i in glac_no]
    regions = list(set(regions))
    for region in regions:
        glac_no_byregion[region] = []
    for i in glac_no:
        region = i.split('.')[0]
        glac_no_only = i.split('.')[1]
        glac_no_byregion[int(region)].append(glac_no_only)
        
    for region in regions:
        glac_no_byregion[region] = sorted(glac_no_byregion[region])
        
    # Load data for glaciers
    dates_table_nospinup = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, spinupyears=0)
    dates_table = modelsetup.datesmodelrun(startyear=input.startyear, endyear=input.endyear, 
                                           spinupyears=input.spinupyears)
    
    count = 0
    for region in regions:
        count += 1
        # ====== GLACIER data =====
        main_glac_rgi_region = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=[region], rgi_regionsO2 = 'all', rgi_glac_number=glac_no_byregion[region])
        # Glacier hypsometry
        main_glac_hyps_region = modelsetup.import_Husstable(
                main_glac_rgi_region, [region], input.hyps_filepath,input.hyps_filedict, input.hyps_colsdrop)
        # ===== CALIBRATION DATA =====
        cal_data_region = pd.DataFrame()
        for dataset in input.cal_datasets:
            cal_subset = class_mbdata.MBData(name=dataset, rgi_regionO1=region)
            cal_subset_data = cal_subset.retrieve_mb(main_glac_rgi_region, main_glac_hyps_region, dates_table_nospinup)
            cal_data_region = cal_data_region.append(cal_subset_data, ignore_index=True)
        cal_data_region = cal_data_region.sort_values(['glacno', 't1_idx'])
        cal_data_region.reset_index(drop=True, inplace=True)
        
        # ===== OTHER DATA =====
        if option_loadhyps_climate == 1:
            # Ice thickness [m], average
            main_glac_icethickness_region = modelsetup.import_Husstable(
                    main_glac_rgi_region, [region], input.thickness_filepath, input.thickness_filedict, 
                    input.thickness_colsdrop)
            main_glac_hyps_region[main_glac_icethickness_region == 0] = 0
            # Width [km], average
            main_glac_width_region = modelsetup.import_Husstable(
                    main_glac_rgi_region, [region], input.width_filepath, input.width_filedict, input.width_colsdrop)
            # ===== CLIMATE DATA =====
            gcm = class_climate.GCM(name=input.ref_gcm_name)
            # Air temperature [degC], Precipitation [m], Elevation [masl], Lapse rate [K m-1]
            gcm_temp_region, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
                    gcm.temp_fn, gcm.temp_vn, main_glac_rgi_region, dates_table_nospinup)
            gcm_prec_region, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
                    gcm.prec_fn, gcm.prec_vn, main_glac_rgi_region, dates_table_nospinup)
            gcm_elev_region = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi_region)
            # Lapse rate [K m-1]
            gcm_lr_region, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(
                    gcm.lr_fn, gcm.lr_vn, main_glac_rgi_region, dates_table_nospinup)
        
        # ===== APPEND DATASETS =====
        if count == 1:
            main_glac_rgi = main_glac_rgi_region
            cal_data = cal_data_region
        
            if option_loadhyps_climate == 1:
                main_glac_hyps = main_glac_hyps_region
                main_glac_icethickness = main_glac_icethickness_region
                main_glac_width = main_glac_width_region
                gcm_temp = gcm_temp_region
                gcm_prec = gcm_prec_region
                gcm_elev = gcm_elev_region
                gcm_lr = gcm_lr_region
                
        else:
            main_glac_rgi = main_glac_rgi.append(main_glac_rgi_region)
            cal_data = cal_data.append(cal_data_region)
            
            if option_loadhyps_climate == 1:
                # If more columns in region, then need to expand existing dataset
                if main_glac_hyps_region.shape[1] > main_glac_hyps.shape[1]:
                    all_col = list(main_glac_hyps.columns.values)
                    reg_col = list(main_glac_hyps_region.columns.values)
                    new_cols = [item for item in reg_col if item not in all_col]
                    for new_col in new_cols:
                        main_glac_hyps[new_col] = 0
                        main_glac_icethickness[new_col] = 0
                        main_glac_width[new_col] = 0
                elif main_glac_hyps_region.shape[1] < main_glac_hyps.shape[1]:
                    all_col = list(main_glac_hyps.columns.values)
                    reg_col = list(main_glac_hyps_region.columns.values)
                    new_cols = [item for item in all_col if item not in reg_col]
                    for new_col in new_cols:
                        main_glac_hyps_region[new_col] = 0
                        main_glac_icethickness_region[new_col] = 0
                        main_glac_width_region[new_col] = 0
                main_glac_hyps = main_glac_hyps.append(main_glac_hyps_region)
                main_glac_icethickness = main_glac_icethickness.append(main_glac_icethickness_region)
                main_glac_width = main_glac_width.append(main_glac_width_region)
            
                gcm_temp = np.vstack([gcm_temp, gcm_temp_region])
                gcm_prec = np.vstack([gcm_prec, gcm_prec_region])
                gcm_elev = np.concatenate([gcm_elev, gcm_elev_region])
                gcm_lr = np.vstack([gcm_lr, gcm_lr_region])
            
    # reset index
    main_glac_rgi.reset_index(inplace=True, drop=True)
    cal_data.reset_index(inplace=True, drop=True)
    
    if option_loadhyps_climate == 1:
        main_glac_hyps.reset_index(inplace=True, drop=True)
        main_glac_icethickness.reset_index(inplace=True, drop=True)
        main_glac_width.reset_index(inplace=True, drop=True)
    
    if option_loadhyps_climate == 0:
        return main_glac_rgi, cal_data
    else:
        return (main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, 
                gcm_temp, gcm_prec, gcm_elev, gcm_lr, 
                cal_data, dates_table)
        
def retrieve_prior_parameters(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, width_t0, elev_bins, 
                              glacier_gcm_temp, glacier_gcm_prec, glacier_gcm_elev, glacier_gcm_lrgcm, 
                              glacier_gcm_lrglac, dates_table, t1_idx, t2_idx, t1, t2, observed_massbal, mb_obs_min,
                              mb_obs_max):    
    def mb_mwea_calc(modelparameters, option_areaconstant=1):
        """
        Run the mass balance and calculate the mass balance [mwea]
        """
        # Mass balance calculations
        (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
         glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
         glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
         glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
         glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, 
         offglac_wide_refreeze, offglac_wide_melt, offglac_wide_snowpack, offglac_wide_runoff) = (
            massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                       width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                       glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                       option_areaconstant=option_areaconstant))
        # Compute glacier volume change for every time step and use this to compute mass balance
        glac_wide_area = glac_wide_area_annual[:-1].repeat(12)
        # Mass change [km3 mwe]
        #  mb [mwea] * (1 km / 1000 m) * area [km2]
        glac_wide_masschange = glac_wide_massbaltotal / 1000 * glac_wide_area
        # Mean annual mass balance [mwea]
        mb_mwea = glac_wide_masschange[t1_idx:t2_idx+1].sum() / glac_wide_area[0] * 1000 / (t2 - t1)
        return mb_mwea
    
    def find_tempchange_opt(tempchange_4opt, precfactor_4opt):
        """
        Find optimal temperature based on observed mass balance
        """
        # Use a subset of model parameters to reduce number of constraints required
        modelparameters[2] = precfactor_4opt
        modelparameters[7] = tempchange_4opt[0]
        # Mean annual mass balance [mwea]
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        return abs(mb_mwea - observed_massbal)
    
    def find_precfactor_opt(precfactor_4opt, tempchange_4opt):
        """
        Find optimal temperature based on observed mass balance
        """
        # Use a subset of model parameters to reduce number of constraints required
        modelparameters[2] = precfactor_4opt[0]
        modelparameters[7] = tempchange_4opt
        # Mean annual mass balance [mwea]
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        return abs(mb_mwea - observed_massbal)
    
    # ----- TEMPBIAS: max accumulation -----
    # Lower temperature bound based on max positive mass balance adjusted to avoid edge effects
    # Temperature at the lowest bin
    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
    lowest_bin = np.where(glacier_area_t0 > 0)[0][0]
    tempchange_max_acc = (-1 * (glacier_gcm_temp + glacier_gcm_lrgcm * 
                                (elev_bins[lowest_bin] - glacier_gcm_elev)).max())
    tempchange_boundlow = tempchange_max_acc
    # Compute max accumulation [mwea]
    modelparameters[2] = 1
    modelparameters[7] = -100
    mb_max_acc = mb_mwea_calc(modelparameters, option_areaconstant=1)
    
    
    # ----- TEMPBIAS: UPPER BOUND -----
    # MAXIMUM LOSS - AREA EVOLVING
    mb_max_loss = (-1 * (glacier_area_t0 * icethickness_t0).sum() / glacier_area_t0.sum() * 
                   input.density_ice / input.density_water / (t2 - t1))
    # Looping forward and backward to ensure optimization does not get stuck
    modelparameters[2] = 1
    modelparameters[7] = tempchange_boundlow
    mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=0)
    # use absolute value because with area evolving the maximum value is a limit
    while mb_mwea_1 - mb_max_loss > 0:
        modelparameters[7] = modelparameters[7] + 1
        mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=0)
    # Looping backward for tempchange at max loss 
    while abs(mb_mwea_1 - mb_max_loss) < 0.005:
        modelparameters[7] = modelparameters[7] - input.tempchange_step
        mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=0)
    tempchange_max_loss = modelparameters[7] + input.tempchange_step

    
    # MB_OBS_MIN - AREA CONSTANT
    # Check if tempchange below min observation; if not, adjust upper TC bound to include mb_obs_min
    mb_tc_boundhigh = mb_mwea_calc(modelparameters, option_areaconstant=1)
    if mb_tc_boundhigh < mb_obs_min:
        tempchange_boundhigh = tempchange_max_loss
    else:
        modelparameters[2] = 1
        modelparameters[7] = tempchange_boundlow
        mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
        # Loop forward
        while mb_mwea_1 > mb_obs_min:
            modelparameters[7] = modelparameters[7] + 1
            mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
        # Loop back
        while mb_mwea_1 < mb_obs_min:
            modelparameters[7] = modelparameters[7] - input.tempchange_step
            mb_mwea_1 = mb_mwea_calc(modelparameters, option_areaconstant=1)
        tempchange_boundhigh = modelparameters[7] + input.tempchange_step
        
#    print('mb_max_loss:', np.round(mb_max_loss,2), 
#          'TC_max_loss_AreaEvolve:', np.round(tempchange_max_loss,2),
#          '\nmb_AreaConstant:', np.round(mb_tc_boundhigh,2), 
#          'TC_boundhigh:', np.round(tempchange_boundhigh,2), 
#          '\nmb_obs_min:', np.round(mb_obs_min,2))
    
    # ----- TEMPBIAS: LOWER BOUND -----
    # AVOID EDGE EFFECTS (ONLY RELEVANT AT TC LOWER BOUND)
    def mb_norm_calc(mb):
        """ Normalized mass balance based on max accumulation and max loss """
        return (mb - mb_max_loss) / (mb_max_acc - mb_max_loss)
    def tc_norm_calc(tc):
        """ Normalized temperature change based on max accumulation and max loss """
        return (tc - tempchange_max_acc) / (tempchange_max_loss - tempchange_max_acc)
    
    modelparameters[2] = 1
    if input.tempchange_edge_method == 'mb':
        modelparameters[7] = tempchange_max_acc
        mb_mwea = mb_max_acc
        while mb_mwea > mb_max_acc - input.tempchange_edge_mb:
            modelparameters[7] = modelparameters[7] + input.tempchange_step
            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        tempchange_boundlow = modelparameters[7]
    elif input.tempchange_edge_method == 'mb_norm':
        modelparameters[7] = tempchange_max_acc
        mb_norm = mb_norm_calc(mb_max_acc)
        while mb_norm > input.tempchange_edge_mbnorm:
            modelparameters[7] = modelparameters[7] + input.tempchange_step
            mb_norm = mb_norm_calc(mb_mwea_calc(modelparameters, option_areaconstant=1))
        tempchange_boundlow = modelparameters[7]
    elif input.tempchange_edge_method == 'mb_norm_slope':
        tempchange_boundlow = tempchange_max_acc
        mb_slope = 0
        while mb_slope > input.tempchange_edge_mbnormslope:
            tempchange_boundlow += input.tempchange_step
            modelparameters[7] = tempchange_boundlow + 0.5
            tc_norm_2 = tc_norm_calc(modelparameters[7])
            mb_norm_2 = mb_norm_calc(mb_mwea_calc(modelparameters, option_areaconstant=1))
            modelparameters[7] = tempchange_boundlow - 0.5
            tc_norm_1 = tc_norm_calc(modelparameters[7])
            mb_norm_1 = mb_norm_calc(mb_mwea_calc(modelparameters, option_areaconstant=1))
            mb_slope = (mb_norm_2 - mb_norm_1) / (tc_norm_2 - tc_norm_1)
            
#    mb_tc_boundlow = mb_mwea_calc(modelparameters, option_areaconstant=1)
#    print('\nmb_max_acc:', np.round(mb_max_acc,2), 'TC_max_acc:', np.round(tempchange_max_acc,2),
#          '\nmb_TC_boundlow_PF1:', np.round(mb_tc_boundlow,2), 
#          'TC_boundlow:', np.round(tempchange_boundlow,2),
#          '\nmb_obs_max:', np.round(mb_obs_max,2)
#          )
        
    # ----- OTHER PARAMETERS -----
    # Assign TC_sigma
    tempchange_sigma = input.tempchange_sigma
    if (tempchange_boundhigh - tempchange_boundlow) / 6 < tempchange_sigma:
        tempchange_sigma = (tempchange_boundhigh - tempchange_boundlow) / 6
    
    tempchange_init = 0
    if tempchange_boundlow > 0:
        tempchange_init = tempchange_boundlow 
    elif tempchange_boundhigh < 0:
        tempchange_init = tempchange_boundhigh
        
    # OPTIMAL PRECIPITATION FACTOR (TC = 0 or TC_boundlow)
    # Find optimized tempchange in agreement with observed mass balance
    tempchange_4opt = tempchange_init
#    print('tempchange_4opt:', tempchange_init)
    precfactor_opt_init = [1]
    precfactor_opt_bnds = (0, 10)
    precfactor_opt_all = minimize(find_precfactor_opt, precfactor_opt_init, args=(tempchange_4opt), 
                                  bounds=[precfactor_opt_bnds], method='L-BFGS-B')
    precfactor_opt = precfactor_opt_all.x[0]
    
    try:
        precfactor_opt1 = precfactor_opt[0]
    except:
        precfactor_opt1 = precfactor_opt
#    print('\nprecfactor_opt:', precfactor_opt)
    
    # Adjust precfactor so it's not < 0.5 or greater than 5
    precfactor_opt_low = 0.5
    precfactor_opt_high = 5
    if precfactor_opt < precfactor_opt_low:
        precfactor_opt = precfactor_opt_low
        tempchange_opt_init = [(tempchange_boundlow + tempchange_boundhigh) / 2]
        tempchange_opt_bnds = (tempchange_boundlow, tempchange_boundhigh)
        tempchange_opt_all = minimize(find_tempchange_opt, tempchange_opt_init, args=(precfactor_opt), 
                                      bounds=[tempchange_opt_bnds], method='L-BFGS-B')
        tempchange_opt = tempchange_opt_all.x[0]
    elif precfactor_opt > precfactor_opt_high:
        precfactor_opt = precfactor_opt_high
        tempchange_opt_init = [(tempchange_boundlow + tempchange_boundhigh) / 2]
        tempchange_opt_bnds = (tempchange_boundlow, tempchange_boundhigh)
        tempchange_opt_all = minimize(find_tempchange_opt, tempchange_opt_init, args=(precfactor_opt), 
                                      bounds=[tempchange_opt_bnds], method='L-BFGS-B')
        tempchange_opt = tempchange_opt_all.x[0]
    else:
        tempchange_opt = tempchange_4opt
        
    try:
        tempchange_opt1 = tempchange_opt[0]
    except:
        tempchange_opt1 = tempchange_opt
        
#    print('\nprecfactor_opt (adjusted):', precfactor_opt)
#    print('tempchange_opt:', tempchange_opt)

    # TEMPCHANGE_SIGMA: derived from mb_obs_min and mb_obs_max
    # MB_obs_min
    #  note: tempchange_boundhigh does not require a limit because glacier is not evolving
    tempchange_adj = input.tempchange_step
    modelparameters[2] = precfactor_opt
    modelparameters[7] = tempchange_opt + tempchange_adj
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    while mb_mwea > mb_obs_min:    
        tempchange_adj += input.tempchange_step
        modelparameters[7] = tempchange_opt + tempchange_adj
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        
#    print('tempchange_adj_4min:', tempchange_adj)
        
    # Expand upper bound if necessary
    if modelparameters[7] > tempchange_boundhigh:
        tempchange_boundhigh = modelparameters[7]
    # MB_obs_max
    modelparameters[2] = precfactor_opt
    modelparameters[7] = tempchange_opt - tempchange_adj
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    while mb_mwea < mb_obs_max and modelparameters[7] > tempchange_boundlow:
        tempchange_adj += input.tempchange_step
        modelparameters[7] = tempchange_opt - tempchange_adj
        if modelparameters[7] < tempchange_boundlow:
            modelparameters[7] = tempchange_boundlow
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        
#    print('tempchange_adj_4max:', tempchange_adj)

    tempchange_sigma = tempchange_adj / 3
    
    # PRECIPITATION FACTOR: LOWER BOUND
    # Check PF_boundlow = 0
    modelparameters[2] = 0
    modelparameters[7] = tempchange_opt + tempchange_sigma
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    if mb_mwea > mb_obs_min:
        # Adjust TC_opt if PF=0 and TC = TC_opt + TC_sigma doesn't reach minimum observation
        precfactor_boundlow = 0
        while mb_mwea > mb_obs_min:
            tempchange_opt += input.tempchange_step
            modelparameters[7] = tempchange_opt + tempchange_sigma
            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    else:
        # Determine lower bound
        precfactor_boundlow = precfactor_opt
        modelparameters[2] = precfactor_boundlow
        modelparameters[7] = tempchange_opt + tempchange_sigma
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
        while mb_mwea > mb_obs_min and precfactor_boundlow > 0:
            precfactor_boundlow -= input.precfactor_step
            if precfactor_boundlow < 0:
                precfactor_boundlow = 0
            modelparameters[2] = precfactor_boundlow
            mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
            
    # PRECIPITATION FACTOR: UPPER BOUND
    precfactor_boundhigh = precfactor_opt
    modelparameters[2] = precfactor_boundhigh
    modelparameters[7] = tempchange_opt - tempchange_sigma
    mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    while mb_mwea < mb_obs_max:
        precfactor_boundhigh += input.precfactor_step
        modelparameters[2] = precfactor_boundhigh
        mb_mwea = mb_mwea_calc(modelparameters, option_areaconstant=1)
    
    # TEMPERATURE BIAS: RE-CENTER
    precfactor_mu = (precfactor_boundlow + precfactor_boundhigh) / 2
    if abs(precfactor_opt - precfactor_mu) > 0.01:
        tempchange_opt_init = [tempchange_opt]
        tempchange_opt_bnds = (tempchange_boundlow, tempchange_boundhigh)
        tempchange_opt_all = minimize(find_tempchange_opt, tempchange_opt_init, args=(precfactor_mu), 
                                      bounds=[tempchange_opt_bnds], method='L-BFGS-B')
        tempchange_opt = tempchange_opt_all.x[0]
    
    
    precfactor_start = precfactor_mu
    if tempchange_boundlow < tempchange_opt + input.tempchange_mu_adj < tempchange_boundhigh:
        tempchange_mu = tempchange_opt + input.tempchange_mu_adj
    else:
        tempchange_mu = tempchange_opt
    tempchange_mu = tempchange_opt + input.tempchange_mu_adj
    # Remove tempchange from bounds
    if tempchange_mu >= tempchange_boundhigh - tempchange_sigma:
        tempchange_mu = tempchange_boundhigh - tempchange_sigma
    elif tempchange_mu <= tempchange_boundlow + tempchange_sigma:
        tempchange_mu = tempchange_boundlow + tempchange_sigma
    tempchange_start = tempchange_mu

    return (precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start, tempchange_boundlow, 
            tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start, tempchange_max_loss, 
            tempchange_max_acc, mb_max_loss, mb_max_acc, precfactor_opt1, tempchange_opt1)
        
        
def main(list_packed_vars):
    # Unpack variables
    count = list_packed_vars[0]
    main_glac_rgi = list_packed_vars[1]
    main_glac_hyps = list_packed_vars[2]
    main_glac_icethickness = list_packed_vars[3]
    main_glac_width = list_packed_vars[4]
    gcm_temp = list_packed_vars[5]
    gcm_prec = list_packed_vars[6]
    gcm_elev = list_packed_vars[7]
    gcm_lr = list_packed_vars[8]
    cal_data = list_packed_vars[9]
    dates_table = list_packed_vars[10]
    
    # Elevation bins
    elev_bins = main_glac_hyps.columns.values.astype(int) 
    
    prior_cns = ['glacier_str', 'pf_bndlow', 'pf_bndhigh', 'pf_mu', 'tc_bndlow', 'tc_bndhigh', 'tc_mu', 'tc_std', 
                 'ddfsnow_bndlow', 'ddfsnow_bndhigh', 'ddfsnow_mu', 'ddfsnow_std', 'mb_max_loss', 'mb_max_acc', 
                 'tc_maxloss', 'tc_max_acc','pf_opt_init', 'tc_opt_init']
    priors_df = pd.DataFrame(np.zeros((main_glac_rgi.shape[0], len(prior_cns))), columns=prior_cns)
    glacier_str_all = [i.split('-')[1] for i in main_glac_rgi['RGIId'].values]
    priors_df['glacier_str'] = glacier_str_all
    priors_df['ddfsnow_mu'] = input.ddfsnow_mu
    priors_df['ddfsnow_std'] = input.ddfsnow_sigma
    priors_df['ddfsnow_bndlow'] = input.ddfsnow_boundlow
    priors_df['ddfsnow_bndhigh'] = input.ddfsnow_boundhigh
    
    for n, glac_str_wRGI in enumerate(main_glac_rgi['RGIId'].values):
#    for n, glac_str_wRGI in enumerate([main_glac_rgi['RGIId'].values[0]]):
        # Glacier string
        glacier_str = glac_str_wRGI.split('-')[1]
        print(count, n, glacier_str)
            
        # Glacier number
        glacno = int(glacier_str.split('.')[1])
        # RGI information
        glac_idx = main_glac_rgi.index.values[n]
        glacier_rgi_table = main_glac_rgi.loc[glac_idx, :]
        # Calibration data
        glacier_cal_data = (cal_data.loc[glac_idx,:]).copy()        
        # Select observed mass balance, error, and time data
        t1 = glacier_cal_data['t1']
        t2 = glacier_cal_data['t2']
        t1_idx = int(glacier_cal_data['t1_idx'])
        t2_idx = int(glacier_cal_data['t2_idx'])
        observed_massbal = glacier_cal_data['mb_mwe'] / (t2 - t1)
        observed_error = glacier_cal_data['mb_mwe_err'] / (t2 - t1)
        mb_obs_max = observed_massbal + 3 * observed_error
        mb_obs_min = observed_massbal - 3 * observed_error

        # MCMC Analysis
        ds = xr.open_dataset(netcdf_fp + glacier_str + '.nc') 
        
        netcdf_fp_new = netcdf_fp + 'wpriors/'
        if os.path.exists(netcdf_fp_new) == False:
            os.makedirs(netcdf_fp_new)
        
        if os.path.isfile(netcdf_fp_new + glacier_str + '.nc'):
            ds_priors = pd.Series(ds['priors'].values, index=ds['dim_0'])
        else:
        
            # Select subsets of data
            glacier_gcm_elev = gcm_elev[n]
            glacier_gcm_temp = gcm_temp[n,:]
            glacier_gcm_lrgcm = gcm_lr[n,:]
            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
            glacier_gcm_prec = gcm_prec[n,:]
            glacier_area_t0 = main_glac_hyps.iloc[n,:].values.astype(float)
            icethickness_t0 = main_glac_icethickness.iloc[n,:].values.astype(float)
            width_t0 = main_glac_width.iloc[n,:].values.astype(float)
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            # Set model parameters
            modelparameters = [input.lrgcm, input.lrglac, input.precfactor, input.precgrad, input.ddfsnow, input.ddfice,
                               input.tempsnow, input.tempchange]
            
            # RETRIEVE PARAMETERS FOR PRIOR DISTRIBUTIONS
            (precfactor_boundlow, precfactor_boundhigh, precfactor_mu, precfactor_start, tempchange_boundlow, 
             tempchange_boundhigh, tempchange_mu, tempchange_sigma, tempchange_start, tempchange_max_loss, 
             tempchange_max_acc, mb_max_loss, mb_max_acc, precfactor_opt_init, tempchange_opt_init) = (
                     retrieve_prior_parameters(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
                                               width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
                                               glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, 
                                               t1_idx, t2_idx, t1, t2, observed_massbal, mb_obs_min, mb_obs_max))
    
            # Record parameters        
            priors_df.loc[n, 'pf_bndlow'] = precfactor_boundlow
            priors_df.loc[n, 'pf_bndhigh'] = precfactor_boundhigh
            priors_df.loc[n, 'pf_mu'] = precfactor_mu
            priors_df.loc[n, 'tc_bndlow'] = tempchange_boundlow
            priors_df.loc[n, 'tc_bndhigh'] = tempchange_boundhigh
            priors_df.loc[n, 'tc_mu'] = tempchange_mu
            priors_df.loc[n, 'tc_std'] = tempchange_sigma
            priors_df.loc[n, 'mb_max_loss'] = mb_max_loss
            priors_df.loc[n, 'mb_max_acc'] = mb_max_acc
            priors_df.loc[n, 'tc_max_loss'] = tempchange_max_loss
            priors_df.loc[n, 'tc_max_acc'] = tempchange_max_acc
            priors_df.loc[n, 'pf_opt_init'] = precfactor_opt_init
            priors_df.loc[n, 'tc_opt_init'] = tempchange_opt_init
        
            ds_prior_cns = prior_cns[1:]
            ds_priors = priors_df.loc[n, ds_prior_cns]
            ds['priors'] = ds_priors
            
            ds.to_netcdf(netcdf_fp_new + glacier_str + '.nc')
        ds.close()
        
        print(count, n, glacier_str, '\n', ds_priors)
        
def pickle_data(fn, data):
    """Pickle data
    
    Parameters
    ----------
    fn : str
        filename including filepath
    data : list, etc.
        data to be pickled
    
    Returns
    -------
    .pkl file
        saves .pkl file of the data
    """
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

#%% PARALLEL PROCESSING
if __name__ == '__main__':
#    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()        
    
    # RGI region
    if args.spc_region is not None:
        rgi_regionsO1 = [int(args.spc_region)]
    else:
        rgi_regionsO1 = input.rgi_regionsO1
        
    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            rgi_glac_number = pickle.load(f)
    elif args.rgi_glac_number is not None:
        rgi_glac_number = [args.rgi_glac_number]
    else:
        rgi_glac_number = input.rgi_glac_number    
    glac_no = rgi_glac_number
        
    netcdf_fp = input.modelparams_fp_dict[rgi_regionsO1[0]]
    
    
    if args.pickle_glacno == 1:
        # Pickle glaciers (should be done first)
        filelist = []
        for region in rgi_regionsO1:
            filelist.extend(glob.glob(netcdf_fp + str(region) + '*.nc'))
        
        glac_no = []
        reg_no = []
        for netcdf in filelist:
            glac_str = netcdf.split('/')[-1].split('.nc')[0]
            glac_no.append(glac_str)
            reg_no.append(glac_str.split('.')[0])
        glac_no = sorted(glac_no)
        
        pickle_fn = netcdf_fp + 'R' + str(rgi_regionsO1[0]) + '_' + str(len(glac_no)) + 'glac.pkl'
        pickle_data(pickle_fn, glac_no)
    else:
        
#        glac_no = [glac_no[0]]
        
        (main_glac_rgi, main_glac_hyps, main_glac_icethickness, main_glac_width, 
         gcm_temp, gcm_prec, gcm_elev, gcm_lr, cal_data, dates_table) = load_glacierdata_byglacno(glac_no)
        
        
        # Define chunk size for parallel processing
        if args.option_parallels != 0:
            num_cores = int(np.min([main_glac_rgi.shape[0], args.num_simultaneous_processes]))
            chunk_size = int(np.ceil(main_glac_rgi.shape[0] / num_cores))
        else:
            # if not running in parallel, chunk size is all glaciers
            chunk_size = main_glac_rgi.shape[0]
            
        # Pack variables for multiprocessing
        list_packed_vars = []
        n = 0
        for chunk in range(0, main_glac_rgi.shape[0], chunk_size):
            n += 1
            main_glac_rgi_chunk = main_glac_rgi.loc[chunk:chunk+chunk_size-1].copy()
            main_glac_hyps_chunk = main_glac_hyps.loc[chunk:chunk+chunk_size-1].copy()
            main_glac_icethickness_chunk = main_glac_icethickness.loc[chunk:chunk+chunk_size-1].copy()
            main_glac_width_chunk = main_glac_width.loc[chunk:chunk+chunk_size-1].copy()
            gcm_temp_chunk = gcm_temp[chunk:chunk+chunk_size]
            gcm_prec_chunk = gcm_prec[chunk:chunk+chunk_size]
            gcm_elev_chunk = gcm_elev[chunk:chunk+chunk_size]
            gcm_lr_chunk = gcm_lr[chunk:chunk+chunk_size]
            cal_data_chunk = cal_data.loc[chunk:chunk+chunk_size-1]
            
            list_packed_vars.append([n,
                                     main_glac_rgi_chunk, 
                                     main_glac_hyps_chunk, 
                                     main_glac_icethickness_chunk, 
                                     main_glac_width_chunk,
                                     gcm_temp_chunk,
                                     gcm_prec_chunk,
                                     gcm_elev_chunk,
                                     gcm_lr_chunk,
                                     cal_data_chunk,
                                     dates_table
                                     ])
    
        # Parallel processing
        if args.option_parallels != 0:
            print('Processing in parallel with ' + str(num_cores) + ' cores...')
            with multiprocessing.Pool(args.num_simultaneous_processes) as p:
                p.map(main,list_packed_vars)
        # If not in parallel, then only should be one loop
        else:
            for n in range(len(list_packed_vars)):
                main(list_packed_vars[n])