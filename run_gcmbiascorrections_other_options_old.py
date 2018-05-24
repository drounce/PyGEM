# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:06:30 2018

@author: David
"""

#    elif option_bias_adjustment == 2:
#        # Huss and Hock (2015)
#        # TEMPERATURE BIAS CORRECTIONS
#        # Calculate monthly mean temperature
#        ref_temp_monthly_avg = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        gcm_temp_monthly_avg = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        gcm_temp_monthly_adj = ref_temp_monthly_avg - gcm_temp_monthly_avg
#        # Monthly temperature bias adjusted according to monthly average
#        t_mt = gcm_temp + np.tile(gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
#        # Mean monthly temperature bias adjusted according to monthly average
#        t_m25avg = np.tile(gcm_temp_monthly_avg + gcm_temp_monthly_adj, int(gcm_temp.shape[1]/12))
#        # Calculate monthly standard deviation of temperature
#        ref_temp_monthly_std = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).std(1)
#                                .reshape(12,-1).transpose())
#        gcm_temp_monthly_std = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).std(1)
#                                .reshape(12,-1).transpose())
#        variability_monthly_std = ref_temp_monthly_std / gcm_temp_monthly_std
#        # Bias adjusted temperature accounting for monthly mean and variability
#        gcm_temp_bias_adj = t_m25avg + (t_mt - t_m25avg) * np.tile(variability_monthly_std, int(gcm_temp.shape[1]/12))
#        # PRECIPITATION BIAS CORRECTIONS
#        # Calculate monthly mean precipitation
#        ref_prec_monthly_avg = (ref_prec.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        gcm_prec_monthly_avg = (gcm_prec_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
#        # Bias adjusted precipitation accounting for differences in monthly mean
#        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
#        
#        # MASS BALANCES FOR DATA COMPARISON
#        main_glac_wide_volume_loss_perc = np.zeros(main_glac_rgi.shape[0])
#        for glac in range(main_glac_rgi.shape[0]):
##        for glac in [0]:
#            # Glacier data
#            modelparameters = main_glac_modelparams[glac,:]
#            glacier_rgi_table = main_glac_rgi.loc[glac, :]
#            glacier_gcm_elev = ref_elev[glac]
#            glacier_gcm_prec = ref_prec[glac,:]
#            glacier_gcm_temp = ref_temp[glac,:]
#            glacier_gcm_lrgcm = ref_lr[glac,:]
#            glacier_gcm_lrglac = glacier_gcm_lrgcm.copy()
#            glacier_area_t0 = main_glac_hyps.iloc[glac,:].values.astype(float)   
#            # Inclusion of ice thickness and width, i.e., loading values may be only required for Huss mass redistribution!
#            icethickness_t0 = main_glac_icethickness.iloc[glac,:].values.astype(float)
#            width_t0 = main_glac_width.iloc[glac,:].values.astype(float)
#            
#            # Mass balance for reference data
#            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
#             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
#             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
#                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
#                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
#                                           dates_table_subset, option_calibration=1))
#            # Total volume loss
#            glac_wide_massbaltotal_annual_ref = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#            glac_wide_volume_loss_total_ref = (
#                    np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_ref.shape] * 
#                              glac_wide_massbaltotal_annual_ref / 1000)[-1])
#            
#            # Mass balance for GCM data
#            glacier_gcm_temp = gcm_temp_bias_adj[glac,0:ref_temp.shape[1]]
#            glacier_gcm_prec = gcm_prec_bias_adj[glac,0:ref_temp.shape[1]]
#            (glac_bin_temp, glac_bin_prec, glac_bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
#             glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
#             glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
#             glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, 
#             glac_wide_area_annual, glac_wide_volume_annual, glac_wide_ELA_annual) = (
#                massbalance.runmassbalance(modelparameters, glacier_rgi_table, glacier_area_t0, icethickness_t0, 
#                                           width_t0, elev_bins, glacier_gcm_temp, glacier_gcm_prec, 
#                                           glacier_gcm_elev, glacier_gcm_lrgcm, glacier_gcm_lrglac, 
#                                           dates_table_subset, option_calibration=1))
#            # Total volume loss
#            glac_wide_massbaltotal_annual_gcm = np.sum(glac_wide_massbaltotal.reshape(-1,12), axis=1)
#            glac_wide_volume_loss_total_gcm = (
#                    np.cumsum(glac_wide_area_annual[glac_wide_massbaltotal_annual_gcm.shape] * 
#                              glac_wide_massbaltotal_annual_gcm / 1000)[-1])
#            
##        # PRINTING BIAS ADJUSTMENT OPTION 2
##        # Temperature parameters
##        output_tempvar = (gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_tempvar_' + 
##                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
##        output_tempavg = (gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_tempavg_' + 
##                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
##        output_tempadj = (gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_tempadj_' + 
##                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
##        np.savetxt(output_filepath + output_tempvar, variability_monthly_std, delimiter=",") 
##        np.savetxt(output_filepath + output_tempavg, gcm_temp_monthly_avg, delimiter=",") 
##        np.savetxt(output_filepath + output_tempadj, gcm_temp_monthly_adj, delimiter=",")
##        # Precipitation parameters
##        output_precadj = (gcm_name + '_' + rcp_scenario + '_biasadjparams_hh2015_mon_precadj_' + 
##                          str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) + '.csv')
##        np.savetxt(output_filepath + output_precadj, bias_adj_prec, delimiter=",")  
##        # Reference elevation (same for all GCMs - only need to export once; needed because bias correcting to the 
##        #  reference, which has a specific elevation)
###        np.savetxt(output_filepath)
##        # Lapse rate - monthly average (same for all GCMs - only need to export once)
##        output_filename_lr = ('biasadj_mon_lravg_' + str(gcm_startyear - gcm_spinupyears) + '_' + str(gcm_endyear) +
##                              '.csv')
##        if os.path.exists(output_filepath + output_filename_lr) == False:
##            np.savetxt(output_filepath + output_filename_lr, ref_lr_monthly_avg, delimiter=",")
        

# OLD TEMP BIAS CORRECTIONS
##    elif option_bias_adjustment == 3:
##        # Reference - GCM difference
##        bias_adj_temp= (ref_temp - gcm_temp_subset).mean(axis=1)
##        # Bias adjusted temperature accounting for mean of entire time period
###        gcm_temp_bias_adj = gcm_temp + bias_adj_temp[:,np.newaxis]
##    elif option_bias_adjustment == 4:
##        # Calculate monthly mean temperature
##        ref_temp_monthly_avg = (ref_temp.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
##                                .reshape(12,-1).transpose())
##        gcm_temp_monthly_avg = (gcm_temp_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
##                                .reshape(12,-1).transpose())
##        bias_adj_temp = ref_temp_monthly_avg - gcm_temp_monthly_avg
##        # Bias adjusted temperature accounting for monthly mean
###        gcm_temp_bias_adj = gcm_temp + np.tile(bias_adj_temp, int(gcm_temp.shape[1]/12))
#    if option_bias_adjustment == 1:
#        # Remove negative values for positive degree day calculation
#        ref_temp_pos = ref_temp.copy()
#        ref_temp_pos[ref_temp_pos < 0] = 0
#        # Select days per month
#        daysinmonth = dates_table['daysinmonth'].values[0:ref_temp.shape[1]]
#        # Cumulative positive degree days [degC*day] for reference period
#        ref_PDD = (ref_temp_pos * daysinmonth).sum(1)
#        # Optimize bias adjustment such that PDD are equal
#        bias_adj_temp = np.zeros(ref_temp.shape[0])
#        for glac in range(ref_temp.shape[0]):
#            ref_PDD_glac = ref_PDD[glac]
#            gcm_temp_glac = gcm_temp_subset[glac,:]
#            def objective(bias_adj_glac):
#                gcm_temp_glac_adj = gcm_temp_glac + bias_adj_glac
#                gcm_temp_glac_adj[gcm_temp_glac_adj < 0] = 0
#                gcm_PDD_glac = (gcm_temp_glac_adj * daysinmonth).sum()
#                return abs(ref_PDD_glac - gcm_PDD_glac)
#            # - initial guess
#            bias_adj_init = 0      
#            # - run optimization
#            bias_adj_temp_opt = minimize(objective, bias_adj_init, method='SLSQP', tol=1e-5)
#            bias_adj_temp[glac] = bias_adj_temp_opt.x
##        gcm_temp_bias_adj = gcm_temp + bias_adj_temp[:,np.newaxis]
# OLD PREC BIAS CORRECTIONS
#if option_bias_adjustment == 1:
#        # Temperature consistent with precipitation elevation
#        #  T = T_gcm + lr_gcm * (z_ref - z_gcm) + tempchange + bias_adjustment
#        ref_temp4prec = ((ref_temp_raw + ref_lr*(glac_elev4prec - ref_elev)[:,np.newaxis]) + (modelparameters[:,7] + 
#                         bias_adj_temp)[:,np.newaxis])
#        gcm_temp4prec = ((gcm_temp_raw + gcm_lr*(glac_elev4prec - gcm_elev)[:,np.newaxis]) + (modelparameters[:,7] + 
#                         bias_adj_temp)[:,np.newaxis])[:,0:ref_temp.shape[1]]
#        # Snow accumulation should be consistent for reference and gcm datasets
#        if input.option_accumulation == 1:
#            # Single snow temperature threshold
#            ref_snow = np.zeros(ref_temp.shape)
#            gcm_snow = np.zeros(ref_temp.shape)
#            for glac in range(main_glac_rgi.shape[0]):
#                ref_snow[glac, ref_temp4prec[glac,:] < modelparameters[glac,6]] = (
#                        ref_prec[glac, ref_temp4prec[glac,:] < modelparameters[glac,6]])
#                gcm_snow[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6]] = (
#                        gcm_prec_subset[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6]])
#        elif input.option_accumulation == 2:
#            # Linear snow threshold +/- 1 degree
#            # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
#            ref_snow = (1/2 + (ref_temp4prec - modelparameters[:,6][:,np.newaxis]) / 2) * ref_prec
#            gcm_snow = (1/2 + (gcm_temp4prec - modelparameters[:,6][:,np.newaxis]) / 2) * gcm_prec_subset
#            # If temperature above or below the max or min, then all rain or snow, respectively. 
#            for glac in range(main_glac_rgi.shape[0]):
#                ref_snow[glac, ref_temp4prec[glac,:] > modelparameters[glac,6] + 1] = 0 
#                ref_snow[glac, ref_temp4prec[glac,:] < modelparameters[glac,6] - 1] = (
#                        ref_prec[glac, ref_temp4prec[glac,:] < modelparameters[glac,6] - 1])
#                gcm_snow[glac, gcm_temp4prec[glac,:] > modelparameters[glac,6] + 1] = 0
#                gcm_snow[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6] - 1] = (
#                        gcm_prec_subset[glac, gcm_temp4prec[glac,:] < modelparameters[glac,6] - 1])
#        # precipitation bias adjustment
#        bias_adj_prec = ref_snow.sum(1) / gcm_snow.sum(1)
##        gcm_prec_bias_adj = gcm_prec * bias_adj_prec[:,np.newaxis]
#    else:
#        # Calculate monthly mean precipitation
#        ref_prec_monthly_avg = (ref_prec.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        gcm_prec_monthly_avg = (gcm_prec_subset.reshape(-1,12).transpose().reshape(-1,int(ref_temp.shape[1]/12)).mean(1)
#                                .reshape(12,-1).transpose())
#        bias_adj_prec = ref_prec_monthly_avg / gcm_prec_monthly_avg
#        # Bias adjusted precipitation accounting for differences in monthly mean
##        gcm_prec_bias_adj = gcm_prec * np.tile(bias_adj_prec, int(gcm_temp.shape[1]/12))
