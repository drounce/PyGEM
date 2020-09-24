"""
fxns_massbalance.py is a list of functions that are used to compute the mass
associated with each glacier for PyGEM.
"""
import numpy as np
#import pandas as pd
import pygem.pygem_input as pygem_prms

#========= FUNCTIONS (alphabetical order) ===================================
def runmassbalance(modelparameters, glacier_rgi_table, glacier_area_initial, icethickness_initial, width_initial, 
                   heights, glacier_gcm_temp, glacier_gcm_tempstd, glacier_gcm_prec, glacier_gcm_elev, 
                   glacier_gcm_lrgcm, glacier_gcm_lrglac, dates_table, option_areaconstant=0, 
                   constantarea_years=pygem_prms.constantarea_years, frontalablation_k=None,
                   glacier_debrismf=None, debug=False, debug_refreeze=False, hindcast=0):
    """
    Runs the mass balance and mass redistribution allowing the glacier to evolve.
    Parameters
    ----------
    modelparameters : pd.Series
        Model parameters (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
        Order of model parameters should not be changed as the run mass balance script relies on this order.
    glacier_rgi_table : pd.Series
        Table of glacier's RGI information
    glacier_area_t0 : np.ndarray
        Initial glacier area [km2] for each elevation bin
    icethickness_t0 : np.ndarray
        Initial ice thickness [m] for each elevation bin
    width_t0 : np.ndarray
        Initial glacier width [km] for each elevation bin
    heights : np.ndarray
        height of elevation bins [masl]
    glacier_gcm_temp : np.ndarray
        GCM temperature [degC] at each time step based on nearest neighbor to the glacier
    glacier_gcm_prec : np.ndarray
        GCM precipitation (solid and liquid) [m] at each time step based on nearest neighbor to the glacier
    glacier_gcm_elev : float
        GCM elevation [masl] at each time step based on nearest neighbor to the glacier
    glacier_gcm_lrgcm : np.ndarray
        GCM lapse rate [K m-1] from the GCM to the glacier for each time step based on nearest neighbor to the glacier
    glacier_gcm_lrglac : np.ndarray
        GCM lapse rate [K m-1] over the glacier for each time step based on nearest neighbor to the glacier
    dates_table : pd.DataFrame
        Table of dates, year, month, daysinmonth, wateryear, and season for each timestep
    option_areaconstant : int
        switch to keep glacier area constant or not (default 0 allows glacier area to change annually)
    debug : Boolean
        option to turn on print statements for development or debugging of code (default False)
    Returns
    -------
    bin_temp : np.ndarray
        Temperature [degC] for each elevation bin and timestep
    bin_prec : np.ndarray
        Precipitation (only liquid) [m] for each elevation bin and timestep
    bin_acc : np.ndarray
        Accumulation (solid precipitation) [mwe] for each elevation bin and timestep
    glac_bin_refreeze : np.ndarray
        Refreeze [mwe] for each elevation bin and timestep
    glac_bin_snowpack : np.ndarray
        Snowpack [mwe] for each elevation bin and timestep
    glac_bin_melt : np.ndarray
        Melt [mwe] for each elevation bin and timestep
    glac_bin_frontalablation : np.ndarray
        Frontal ablation [mwe] for each elevation bin and timestep
    glac_bin_massbalclim : np.ndarray
        Climatic mass balance [mwe] for each elevation bin and timestep
    glac_bin_massbalclim_annual : np.ndarray
        Climatic mass balance [mwe] for each elevation bin and year
    glac_bin_area_annual : np.ndarray
        Glacier area [km2] for each elevation bin and year
    glac_bin_icethickness_annual : np.ndarray
        Ice thickness [m] for each elevation bin and year
    glac_bin_width_annual : np.ndarray
        Glacier width [km] for each elevation bin and year
    glac_bin_surfacetype_annual : np.ndarray
        Surface type [see dictionary] for each elevation bin and year
    glac_wide_massbaltotal : np.ndarray
        Glacier-wide total mass balance (climatic mass balance - frontal ablation) [mwe] for each timestep
    glac_wide_runoff : np.ndarray
        Glacier-wide runoff [m3] for each timestep
    glac_wide_snowline : np.ndarray
        Snowline altitude [masl] for each timestep
    glac_wide_snowpack : np.ndarray
        Glacier-wide snowpack [km3 we] for each timestep
    glac_wide_area_annual : np.ndarray
        Glacier-wide area [km2] for each timestep
    glac_wide_volume_annual : np.ndarray
        Glacier-wide volume [km3 ice] for each timestep
    glac_wide_ELA_annual : np.ndarray
        Equilibrium line altitude [masl] for each year
        
    offglac_wide_runoff : np.ndarray
        Off-glacier runoff [m3] for each timestep
    offglac_bin_acc : np.ndarray
        Off-glacier accumulation (solid precipitation) [mwe] for each elevation bin and timestep
    offglac_bin_refreeze : np.ndarray
        Off-glacier refreeze [mwe] for each elevation bin and timestep
    offglac_bin_snowpack : np.ndarray
        Off-glacier snowpack [mwe] for each elevation bin and timestep
    offglac_bin_melt : np.ndarray
        Off-glacier melt [mwe] for each elevation bin and timestep
    """       
    if debug:
        print('\n\nDEBUGGING MASS BALANCE FUNCTION\n\n')
        
    # Select annual divisor and columns
    if pygem_prms.timestep == 'monthly':
        annual_divisor = 12
    annual_columns = np.arange(0, int(dates_table.shape[0] / 12))
    # Variables to export
    nbins = heights.shape[0]
    nmonths = glacier_gcm_temp.shape[0]
    nyears = annual_columns.shape[0]
    bin_temp = np.zeros((nbins,nmonths))
    bin_prec = np.zeros((nbins,nmonths))
    bin_acc = np.zeros((nbins,nmonths))
    bin_refreezepotential = np.zeros((nbins,nmonths))
    bin_refreeze = np.zeros((nbins,nmonths))
    bin_meltsnow = np.zeros((nbins,nmonths))
    bin_meltglac = np.zeros((nbins,nmonths))
    bin_melt = np.zeros((nbins,nmonths))
    bin_snowpack = np.zeros((nbins,nmonths))
    glac_bin_refreeze = np.zeros((nbins,nmonths))
    glac_bin_melt = np.zeros((nbins,nmonths))
    glac_bin_frontalablation = np.zeros((nbins,nmonths))
    glac_bin_snowpack = np.zeros((nbins,nmonths))
    glac_bin_massbalclim = np.zeros((nbins,nmonths))
    glac_bin_massbalclim_annual = np.zeros((nbins,nyears))
    glac_bin_surfacetype_annual = np.zeros((nbins,nyears))
    glac_bin_icethickness_annual = np.zeros((nbins, nyears + 1))
    glac_bin_area_annual = np.zeros((nbins, nyears + 1))
    glac_bin_width_annual = np.zeros((nbins, nyears + 1))

    offglac_bin_prec = np.zeros((nbins,nmonths))
    offglac_bin_melt = np.zeros((nbins,nmonths))
    offglac_bin_refreeze = np.zeros((nbins,nmonths))
    offglac_bin_snowpack = np.zeros((nbins,nmonths))
    offglac_bin_area_annual = np.zeros((nbins, nyears + 1))
    
    # Local variables
    bin_precsnow = np.zeros((nbins,nmonths))
    refreeze_potential = np.zeros(nbins)
    snowpack_remaining = np.zeros(nbins)
    dayspermonth = dates_table['daysinmonth'].values
    surfacetype_ddf = np.zeros(nbins)
    glac_idx_initial = glacier_area_initial.nonzero()[0]
    glacier_area_t0 = glacier_area_initial.copy()
    icethickness_t0 = icethickness_initial.copy()
    width_t0 = width_initial.copy()
    if pygem_prms.option_refreezing == 1:
        # Refreezing layers density, volumetric heat capacity, and thermal conductivity
        rf_dens_expb = (pygem_prms.rf_dens_bot / pygem_prms.rf_dens_top)**(1/(pygem_prms.rf_layers-1))
        rf_layers_dens = np.array([pygem_prms.rf_dens_top * rf_dens_expb**x for x in np.arange(0,pygem_prms.rf_layers)])
        rf_layers_ch = (1 - rf_layers_dens/1000) * pygem_prms.ch_air + rf_layers_dens/1000 * pygem_prms.ch_ice
        rf_layers_k = (1 - rf_layers_dens/1000) * pygem_prms.k_air + rf_layers_dens/1000 * pygem_prms.k_ice
        te_rf = np.zeros((pygem_prms.rf_layers,nbins))   # temp of each layer for each elev bin from present time step
        tl_rf = np.zeros((pygem_prms.rf_layers,nbins))   # temp of each layer for each elev bin from previous time step
        refr = np.zeros(nbins)                      # refreeze in each bin
        rf_cold = np.zeros(nbins)                   # refrezee cold content or "potential" refreeze
    
    # Sea level for marine-terminating glaciers
    sea_level = 0
    rgi_region = int(glacier_rgi_table.RGIId.split('-')[1].split('.')[0])
    if frontalablation_k == None:
         frontalablation_k0 = pygem_prms.frontalablation_k0dict[rgi_region]
    
    #  glac_idx_initial is used with advancing glaciers to ensure no bands are added in a discontinuous section
    # Run mass balance only on pixels that have an ice thickness (some glaciers do not have an ice thickness)
    #  otherwise causes indexing errors causing the model to fail
    if icethickness_t0.max() > 0:    
        if pygem_prms.option_adjusttemp_surfelev == 1:
            # ice thickness initial is used to adjust temps to changes in surface elevation
            icethickness_initial = icethickness_t0.copy()
            icethickness_initial[0:icethickness_initial.nonzero()[0][0]] = (
                    icethickness_initial[icethickness_initial.nonzero()[0][0]])
            #  bins that advance need to have an initial ice thickness; otherwise, the temp adjustment will be based on
            #  ice thickness - 0, which is wrong  Since advancing bins take the thickness of the previous bin, set the
            #  initial ice thickness of all bins below the terminus to the ice thickness at the terminus.
        # Compute the initial surface type [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
        surfacetype, firnline_idx = surfacetypebinsinitial(glacier_area_t0, glacier_rgi_table, heights)
        # Create surface type DDF dictionary (manipulate this function for calibration or for each glacier)
        surfacetype_ddf_dict = surfacetypeDDFdict(modelparameters)
        
    # ANNUAL LOOP (daily or monthly timestep contained within loop)
    for year in range(0, nyears): 
        # Check ice still exists:
        if icethickness_t0.max() > 0:    
        
            if debug:
                print(year, 'max ice thickness [m]:', icethickness_t0.max())

            # Glacier indices
            glac_idx_t0 = glacier_area_t0.nonzero()[0]
            
            # Off-glacier area and indices
            if option_areaconstant == 0:
                offglac_bin_area_annual[:,year] = glacier_area_initial - glacier_area_t0
                offglac_idx = np.where(offglac_bin_area_annual[:,year] > 0)[0]
            
            
            # Functions currently set up for monthly timestep
            #  only compute mass balance while glacier exists
            if (pygem_prms.timestep == 'monthly') and (glac_idx_t0.shape[0] != 0):      
                
                # AIR TEMPERATURE: Downscale the gcm temperature [deg C] to each bin
                if pygem_prms.option_temp2bins == 1:
                    # Downscale using gcm and glacier lapse rates
                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
                    bin_temp[:,12*year:12*(year+1)] = (glacier_gcm_temp[12*year:12*(year+1)] + 
                         glacier_gcm_lrgcm[12*year:12*(year+1)] * 
                         (glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale] - glacier_gcm_elev) + 
                         glacier_gcm_lrglac[12*year:12*(year+1)] * (heights - 
                         glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale])[:,np.newaxis] + modelparameters[7])
                # Option to adjust air temperature based on changes in surface elevation
                #  note: OGGM automatically updates the bin elevation, so this step is not needed
                if pygem_prms.option_adjusttemp_surfelev == 1 and pygem_prms.hyps_data in ['Huss', 'Farinotti']:
                    # T_air = T_air + lr_glac * (icethickness_present - icethickness_initial)
                    bin_temp[:,12*year:12*(year+1)] = (bin_temp[:,12*year:12*(year+1)] + 
                                                       glacier_gcm_lrglac[12*year:12*(year+1)] * 
                                                       (icethickness_t0 - icethickness_initial)[:,np.newaxis])
                
                # PRECIPITATION/ACCUMULATION: Downscale the precipitation (liquid and solid) to each bin
                if pygem_prms.option_prec2bins == 1:
                    # Precipitation using precipitation factor and precipitation gradient
                    #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
                    bin_precsnow[:,12*year:12*(year+1)] = (glacier_gcm_prec[12*year:12*(year+1)] * 
                            modelparameters[2] * (1 + modelparameters[3] * (heights - 
                            glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale]))[:,np.newaxis])
                # Option to adjust prec of uppermost 25% of glacier for wind erosion and reduced moisture content
                if pygem_prms.option_preclimit == 1:
                    # If elevation range > 1000 m, apply corrections to uppermost 25% of glacier (Huss and Hock, 2015)
                    if np.abs(heights[glac_idx_t0[-1]] - heights[glac_idx_t0[0]]) > 1000:
                        # Indices of upper 25%
                        glac_idx_upper25 = (
                                glac_idx_t0[(heights[glac_idx_t0] - heights[glac_idx_t0].min()) / 
                                            (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()) * 100 >= 75])                        
                        # Exponential decay according to elevation difference from the 75% elevation
                        #  prec_upper25 = prec * exp(-(elev_i - elev_75%)/(elev_max- - elev_75%))
                        # height at 75% of the elevation
                        height_75 = heights[glac_idx_upper25].min()
                        glac_idx_75 = np.where(heights == height_75)[0][0]
                        # exponential decay
                        bin_precsnow[glac_idx_upper25,12*year:12*(year+1)] = (
                                bin_precsnow[glac_idx_75,12*year:12*(year+1)] * 
                                np.exp(-1*(heights[glac_idx_upper25] - height_75) / 
                                       (heights[glac_idx_upper25].max() - heights[glac_idx_upper25].min()))
                                [:,np.newaxis])
                        
                        # Precipitation cannot be less than 87.5% of the maximum accumulation elsewhere on the glacier
                        for month in range(0,12):
                            bin_precsnow[glac_idx_upper25[(bin_precsnow[glac_idx_upper25,month] < 0.875 * 
                                bin_precsnow[glac_idx_t0,month].max()) & 
                                (bin_precsnow[glac_idx_upper25,month] != 0)], month] = (
                                                                0.875 * bin_precsnow[glac_idx_t0,month].max())
                # Separate total precipitation into liquid (bin_prec) and solid (bin_acc)
                if pygem_prms.option_accumulation == 1:
                    # if temperature above threshold, then rain
                    bin_prec[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6]] = (
                        bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6]])
                    # if temperature below threshold, then snow
                    bin_acc[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6]] = (
                        bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6]])
                elif pygem_prms.option_accumulation == 2:
                    # If temperature between min/max, then mix of snow/rain using linear relationship between min/max
                    bin_prec[:,12*year:12*(year+1)] = ((1/2 + (bin_temp[:,12*year:12*(year+1)] - 
                            modelparameters[6]) / 2) * bin_precsnow[:,12*year:12*(year+1)])
                    bin_acc[:,12*year:12*(year+1)] = (bin_precsnow[:,12*year:12*(year+1)] - 
                           bin_prec[:,12*year:12*(year+1)])
                    # If temperature above maximum threshold, then all rain
                    bin_prec[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1] = (
                            bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > 
                                         modelparameters[6] + 1])
                    bin_acc[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] > modelparameters[6] + 1] = 0
                    # If temperature below minimum threshold, then all snow
                    (bin_acc[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 1]
                        )= (bin_precsnow[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= 
                                         modelparameters[6] - 1])
                    bin_prec[:,12*year:12*(year+1)][bin_temp[:,12*year:12*(year+1)] <= modelparameters[6] - 1] = 0
                
                
                # ENTER MONTHLY LOOP (monthly loop required as )
                for month in range(0,12):
                    # Step is the position as a function of year and month, which improves readability
                    step = 12*year + month
                    
                    # ACCUMULATION, MELT, REFREEZE, AND CLIMATIC MASS BALANCE
                    # Snowpack [m w.e.] = snow remaining + new snow
                    bin_snowpack[:,step] = snowpack_remaining + bin_acc[:,step]
                    
                    # MELT [m w.e.]
                    # energy available for melt [degC day]   
                    if pygem_prms.option_ablation == 1:
                        # option 1: energy based on monthly temperature
                        melt_energy_available = bin_temp[:,step]*dayspermonth[step]
                        melt_energy_available[melt_energy_available < 0] = 0
                    elif pygem_prms.option_ablation == 2:
                        # option 2: monthly temperature superimposed with daily temperature variability 
                        # daily temperature variation in each bin for the monthly timestep
                        bin_tempstd_daily = np.repeat(
                                np.random.normal(loc=0, scale=glacier_gcm_tempstd[step], size=dayspermonth[step])
                                .reshape(1,dayspermonth[step]), nbins, axis=0)
                        # daily temperature in each bin for the monthly timestep
                        bin_temp_daily = bin_temp[:,step][:,np.newaxis] + bin_tempstd_daily
                        # remove negative values
                        bin_temp_daily[bin_temp_daily < 0] = 0
                        # Energy available for melt [degC day] = sum of daily energy available
                        melt_energy_available = bin_temp_daily.sum(axis=1)
                    # SNOW MELT [m w.e.]
                    bin_meltsnow[:,step] = surfacetype_ddf_dict[2] * melt_energy_available
                    # snow melt cannot exceed the snow depth
                    bin_meltsnow[bin_meltsnow[:,step] > bin_snowpack[:,step], step] = (
                            bin_snowpack[bin_meltsnow[:,step] > bin_snowpack[:,step], step])
                    # GLACIER MELT (ice and firn) [m w.e.]
                    # energy remaining after snow melt [degC day]
                    melt_energy_available = melt_energy_available - bin_meltsnow[:,step] / surfacetype_ddf_dict[2]
                    # remove low values of energy available caused by rounding errors in the step above
                    melt_energy_available[abs(melt_energy_available) < pygem_prms.tolerance] = 0
                    # DDF based on surface type [m w.e. degC-1 day-1]
                    for surfacetype_idx in surfacetype_ddf_dict: 
                        surfacetype_ddf[surfacetype == surfacetype_idx] = surfacetype_ddf_dict[surfacetype_idx]
                    if pygem_prms.include_debris and glacier_debrismf is not None:
                        surfacetype_ddf = surfacetype_ddf * glacier_debrismf
                        
                    bin_meltglac[glac_idx_t0,step] = surfacetype_ddf[glac_idx_t0] * melt_energy_available[glac_idx_t0]
                    # TOTAL MELT (snow + glacier)
                    #  off-glacier need to include melt of refreeze because there are no glacier dynamics,
                    #  but on-glacier do not need to account for this (simply assume refreeze has same surface type)
                    bin_melt[:,step] = bin_meltglac[:,step] + bin_meltsnow[:,step]  
                    
                    # REFREEZING
                    if pygem_prms.option_refreezing == 1:
                        # Refreeze based on heat conduction approach (Huss and Hock 2015)    
                        # refreeze time step (s)
                        rf_dt = 3600 * 24 * dates_table.loc[step,'daysinmonth'] / pygem_prms.rf_dsc
                        
                        if pygem_prms.option_rf_limit_meltsnow == 1:
                            bin_meltlimit = bin_meltsnow.copy()
                        else:
                            bin_meltlimit = bin_melt.copy()
                            
                        # Loop through each elevation bin of glacier
                        for nbin, gidx in enumerate(glac_idx_t0):
                                    
                            # COMPUTE HEAT CONDUCTION - BUILD COLD RESERVOIR                            
                            # If no melt, then build up cold reservoir (compute heat conduction)
                            if bin_melt[gidx,step] < pygem_prms.rf_meltcrit:
                                
                                if debug_refreeze and nbin == 0 and step < 12:
                                    print('\nMonth ' + str(dates_table.loc[step,'month']), 'Computing heat conduction')
                                
                                # Set refreeze equal to 0
                                refr[gidx] = 0
                                # Loop through multiple iterations to converge on a solution
                                #  -> this will loop through 0, 1, 2
                                for h in np.arange(0, pygem_prms.rf_dsc):                                  
                                    # Compute heat conduction in layers (loop through rows)
                                    #  go from 1 to rf_layers-1 to avoid indexing errors with "j-1" and "j+1"
                                    #  "j+1" is set to zero, which is fine for temperate glaciers but inaccurate for cold/polythermal glaciers
                                    for j in np.arange(1, pygem_prms.rf_layers-1):
                                        # Assume temperature of first layer equals air temperature
                                        #  assumption probably wrong, but might still work at annual average
                                        # Since next line uses tl_rf for all calculations, set tl_rf[0] to present mean 
                                        #  monthly air temperature to ensure the present calculations are done with the 
                                        #  present time step's air temperature
                                        tl_rf[0, gidx] = bin_temp[gidx,step]
                                        # Temperature for each layer
                                        te_rf[j,gidx] = (tl_rf[j,gidx] + 
                                             rf_dt * rf_layers_k[j] / rf_layers_ch[j] / pygem_prms.rf_dz**2 * 0.5 * 
                                             ((tl_rf[j-1,gidx] - tl_rf[j,gidx]) - (tl_rf[j,gidx] - tl_rf[j+1,gidx]))) 
                                        # Update previous time step
                                        tl_rf[:,gidx] = te_rf[:,gidx]
                                   
                                if debug_refreeze and nbin == 0 and step < 12:
                                    print('tl_rf:', ["{:.2f}".format(x) for x in tl_rf[:,gidx]])
                                            
                            # COMPUTE REFREEZING - TAP INTO "COLD RESERVOIR" or potential refreezing  
                            else:
                                
                                if debug_refreeze and nbin == 0 and step < 12:
                                    print('\nMonth ' + str(dates_table.loc[step,'month']), 'Computing refreeze')
                                
                                # Refreezing over firn surface
                                if (surfacetype[gidx] == 2) or (surfacetype[gidx] == 3):
                                    nlayers = pygem_prms.rf_layers-1
                                # Refreezing over ice surface
                                else:
                                    # Approximate number of layers of snow on top of ice
                                    smax = np.round((bin_snowpack[gidx,step] / (rf_layers_dens[0] / 1000) + pygem_prms.pp) / pygem_prms.rf_dz, 0)
                                    # if there is very little snow on the ground (SWE > 0.06 m for pp=0.3), then still set smax (layers) to 1
                                    if bin_snowpack[gidx,step] > 0 and smax == 0:
                                        smax=1
                                    # if no snow on the ground, then set to rf_cold to NoData value
                                    if smax == 0:
                                        rf_cold[gidx] = 0
                                    # if smax greater than the number of layers, set to max number of layers minus 1
                                    if smax > pygem_prms.rf_layers - 1:
                                        smax = pygem_prms.rf_layers - 1
                                    nlayers = int(smax)
                        
                                # Compute potential refreeze, "cold reservoir", from temperature in each layer 
                                # only calculate potential refreezing first time it starts melting each year
                                if rf_cold[gidx] == 0 and tl_rf[:,gidx].min() < 0:
                                    
                                    if debug_refreeze and nbin == 0 and step < 12:
                                        print('calculating potential refreeze from ' + str(nlayers) + ' layers')
                                    
                                    for j in np.arange(0,nlayers):
                                        j += 1
                                        # units: (degC) * (J K-1 m-3) * (m) * (kg J-1) * (m3 kg-1)
                                        rf_cold_layer = tl_rf[j,gidx] * rf_layers_ch[j] * pygem_prms.rf_dz / pygem_prms.Lh_rf / pygem_prms.density_water
                                        rf_cold[gidx] -= rf_cold_layer
                                        
                                        if debug_refreeze and nbin == 0 and step < 12:
                                            print('j:', j, 'tl_rf @ j:', np.round(tl_rf[j,gidx],2), 
                                                           'ch @ j:', np.round(rf_layers_ch[j],2), 
                                                           'rf_cold_layer @ j:', np.round(rf_cold_layer,2),
                                                           'rf_cold @ j:', np.round(rf_cold[gidx],2))
                                    
                                    if debug_refreeze and nbin == 0 and step < 12:
                                        print('rf_cold:', np.round(rf_cold[gidx],2))
                                        
                                # Compute refreezing
                                # If melt and liquid prec < potential refreeze, then refreeze all melt and liquid prec
                                if (bin_meltlimit[gidx,step] + bin_prec[gidx,step]) < rf_cold[gidx]:
                                    refr[gidx] = bin_meltlimit[gidx,step] + bin_prec[gidx,step]
                                # otherwise, refreeze equals the potential refreeze
                                elif rf_cold[gidx] > 0:
                                    refr[gidx] = rf_cold[gidx]
                                else:
                                    refr[gidx] = 0

                                # Track the remaining potential refreeze                                    
                                rf_cold[gidx] -= (bin_meltlimit[gidx,step] + bin_prec[gidx,step])
                                # if potential refreeze consumed, then set to 0 and set temperature to 0 (temperate firn)
                                if rf_cold[gidx] < 0:
                                    rf_cold[gidx] = 0
                                    tl_rf[:,gidx] = 0
                                    
                            # Record refreeze
                            bin_refreeze[gidx,step] = refr[gidx]
                            
                            if debug_refreeze and step < 12 and nbin==0:
                                print('Month ' + str(dates_table.loc[step,'month']),
                                      'Rf_cold remaining:', np.round(rf_cold[gidx],2),
                                      'Snow depth:', np.round(bin_snowpack[glac_idx_t0[nbin],step],2),
                                      'Snow melt:', np.round(bin_meltsnow[glac_idx_t0[nbin],step],2),
                                      'Rain:', np.round(bin_prec[glac_idx_t0[nbin],step],2),
                                      'Rfrz:', np.round(bin_refreeze[gidx,step],2)
                                      )             
                                
                    elif pygem_prms.option_refreezing == 2:
                        # Refreeze based on annual air temperature (Woodward etal. 1997)
                        #  R(m) = (-0.69 * Tair + 0.0096) * 1 m / 100 cm
                        # calculate annually and place potential refreeze in user defined month
                        if dates_table.loc[step,'month'] == pygem_prms.rf_month:                        
                            bin_temp_annual = annualweightedmean_array(bin_temp[:,12*year:12*(year+1)], 
                                                                       dates_table.iloc[12*year:12*(year+1),:])
                            bin_refreezepotential_annual = (-0.69 * bin_temp_annual + 0.0096) * 1/100
                            # Remove negative refreezing values
                            bin_refreezepotential_annual[bin_refreezepotential_annual < 0] = 0
                            bin_refreezepotential[:,step] = bin_refreezepotential_annual
                            # Reset refreeze potential every year
                            if bin_refreezepotential[:,step].max() > 0:
                                refreeze_potential = bin_refreezepotential[:,step]

                        
                        if debug_refreeze and step < 12:
                            print('Month' + str(dates_table.loc[step,'month']),
                                  'Refreeze potential:', np.round(refreeze_potential[glac_idx_t0[0]],3),
                                  'Snow depth:', np.round(bin_snowpack[glac_idx_t0[0],step],2),
                                  'Snow melt:', np.round(bin_meltsnow[glac_idx_t0[0],step],2),
                                  'Rain:', np.round(bin_prec[glac_idx_t0[0],step],2))
                            
                            
                        # Refreeze [m w.e.]
                        #  refreeze cannot exceed rain and melt (snow & glacier melt)
                        bin_refreeze[:,step] = bin_meltsnow[:,step] + bin_prec[:,step]
                        # refreeze cannot exceed snow depth
                        bin_refreeze[bin_refreeze[:,step] > bin_snowpack[:,step], step] = (
                                bin_snowpack[bin_refreeze[:,step] > bin_snowpack[:,step], step])
                        # refreeze cannot exceed refreeze potential
                        bin_refreeze[bin_refreeze[:,step] > refreeze_potential, step] = (
                                refreeze_potential[bin_refreeze[:,step] > refreeze_potential])
                        bin_refreeze[abs(bin_refreeze[:,step]) < pygem_prms.tolerance, step] = 0
                        # update refreeze potential
                        refreeze_potential = refreeze_potential - bin_refreeze[:,step]
                        refreeze_potential[abs(refreeze_potential) < pygem_prms.tolerance] = 0
                    
                    if step < 12 and debug_refreeze:
                        print('refreeze bin ' + str(int(glac_idx_t0[0]*10)) + ':', 
                                np.round(bin_refreeze[glac_idx_t0[0],step],3))
                    
                    # SNOWPACK REMAINING [m w.e.]
                    snowpack_remaining = bin_snowpack[:,step] - bin_meltsnow[:,step]
                    snowpack_remaining[abs(snowpack_remaining) < pygem_prms.tolerance] = 0
                    
                    # Record values                  
                    glac_bin_melt[glac_idx_t0,step] = bin_melt[glac_idx_t0,step]
                    glac_bin_refreeze[glac_idx_t0,step] = bin_refreeze[glac_idx_t0,step]
                    glac_bin_snowpack[glac_idx_t0,step] = bin_snowpack[glac_idx_t0,step]
                    # CLIMATIC MASS BALANCE [m w.e.]
                    glac_bin_massbalclim[glac_idx_t0,step] = (
                            bin_acc[glac_idx_t0,step] + glac_bin_refreeze[glac_idx_t0,step] - 
                            glac_bin_melt[glac_idx_t0,step])
                    
                    # OFF-GLACIER ACCUMULATION, MELT, REFREEZE, AND SNOWPAC
                    if option_areaconstant == 0:
                        # precipitation, refreeze, and snowpack are the same both on- and off-glacier
                        offglac_bin_prec[offglac_idx,step] = bin_prec[offglac_idx,step]
                        offglac_bin_refreeze[offglac_idx,step] = bin_refreeze[offglac_idx,step]
                        offglac_bin_snowpack[offglac_idx,step] = bin_snowpack[offglac_idx,step]
                        # Off-glacier melt includes both snow melt and melting of refreezing
                        #  (this is not an issue on-glacier because energy remaining melts underlying snow/ice)
                        # melt of refreezing (assumed to be snow)
                        offglac_meltrefreeze = surfacetype_ddf_dict[2] * melt_energy_available
                        # melt of refreezing cannot exceed refreezing
                        offglac_meltrefreeze[offglac_meltrefreeze > bin_refreeze[:,step]] = (
                                bin_refreeze[:,step][offglac_meltrefreeze > bin_refreeze[:,step]])
                        # off-glacier melt = snow melt + refreezing melt
                        offglac_bin_melt[offglac_idx,step] = (bin_meltsnow[offglac_idx,step] + 
                                                              offglac_meltrefreeze[offglac_idx])
                
                # ===== RETURN TO ANNUAL LOOP =====
                # Mass loss cannot exceed glacier volume
                #  mb [mwea] = -1 * sum{area [km2] * ice thickness [m]} / total area [km2] * density_ice / density_water
                mb_max_loss = (-1 * (glacier_area_t0 * icethickness_t0 * pygem_prms.density_ice / pygem_prms.density_water).sum() 
                               / glacier_area_t0.sum())
                # Check annual climatic mass balance
                mb_mwea = ((glacier_area_t0 * glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
                            glacier_area_t0.sum()) 
                
                
                    
                # If mass loss exceeds glacier mass, reduce melt to ensure the glacier completely melts without excess
                if mb_mwea < mb_max_loss:  
                    
                    if debug:
                         print('mb_mwea (before):', np.round(mb_mwea,3), 'mb_max_loss:', np.round(mb_max_loss,3))
                         
                    mb_dif = mb_max_loss - mb_mwea
                   
                    glac_wide_melt = ((glac_bin_melt[:,12*year:12*(year+1)] * glacier_area_t0[:,np.newaxis]).sum() / 
                                       glacier_area_t0.sum())
                    # adjust using tolerance to avoid any rounding errors that would leave a little glacier volume left
                    glac_bin_melt[:,12*year:12*(year+1)] = (glac_bin_melt[:,12*year:12*(year+1)] * 
                                                            (1 + pygem_prms.tolerance - mb_dif / glac_wide_melt))
                    glac_bin_massbalclim[:,12*year:12*(year+1)] = (
                            bin_acc[:,12*year:12*(year+1)] + glac_bin_refreeze[:,12*year:12*(year+1)] - 
                            glac_bin_melt[:,12*year:12*(year+1)])    
                    # Check annual climatic mass balance
                    mb_mwea = ((glacier_area_t0 * glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)).sum() / 
                                glacier_area_t0.sum()) 
                    
                    if debug:
                        print('mb_check after adjustment (should equal mass loss):', np.round(mb_mwea,3))
                    
                
                # FRONTAL ABLATION
                # Glacier bed altitude [masl]
                glac_idx_minelev = np.where(heights == heights[glac_idx_t0].min())[0][0]
                glacier_bedelev = (heights[glac_idx_minelev] - icethickness_initial[glac_idx_minelev])
                
                if debug and glacier_rgi_table['TermType'] != 0:
                    print('\nyear:', year, '\n sea level:', sea_level, 'bed elev:', np.round(glacier_bedelev, 2))

                # If glacier bed below sea level, compute frontal ablation
                if glacier_bedelev < sea_level and glacier_rgi_table['TermType'] != 0:
                    # Volume [m3] and bed elevation [masl] of each bin
                    glac_bin_volume = glacier_area_t0 * 10**6 * icethickness_t0
                    glac_bin_bedelev = np.zeros((glacier_area_t0.shape))
                    glac_bin_bedelev[glac_idx_t0] = heights[glac_idx_t0] - icethickness_initial[glac_idx_t0]
                    
                    # Option 1: Use Huss and Hock (2015) frontal ablation parameterizations
                    #  Frontal ablation using width of lowest bin can severely overestimate the actual width of the
                    #  calving front. Therefore, use estimated calving width from satellite imagery as appropriate.
                    if pygem_prms.option_frontalablation_k == 1 and frontalablation_k == None:
                        # Calculate frontal ablation parameter based on slope of lowest 100 m of glacier
                        glac_idx_slope = np.where((heights <= heights[glac_idx_t0].min() + 100) & 
                                                  (heights >= heights[glac_idx_t0].min()))[0]
                        elev_change = np.abs(heights[glac_idx_slope[0]] - heights[glac_idx_slope[-1]])
                        # length of lowest 100 m of glacier
                        length_lowest100m = (glacier_area_t0[glac_idx_slope] / 
                                             width_t0[glac_idx_slope] * 1000).sum()
                        # slope of lowest 100 m of glacier
                        slope_lowest100m = np.rad2deg(np.arctan(elev_change/length_lowest100m))
                        # Frontal ablation parameter
                        frontalablation_k = frontalablation_k0 * slope_lowest100m

                    # Calculate frontal ablation
                    # Bed elevation with respect to sea level
                    #  negative when bed is below sea level (Oerlemans and Nick, 2005)
                    waterdepth = sea_level - glacier_bedelev
                    # Glacier length [m]
                    length = (glacier_area_t0[width_t0 > 0] / width_t0[width_t0 > 0]).sum() * 1000
                    # Height of calving front [m]
                    height_calving = np.max([pygem_prms.af*length**0.5, 
                                             pygem_prms.density_water / pygem_prms.density_ice * waterdepth])
                    # Width of calving front [m]
                    if pygem_prms.hyps_data in ['oggm']:
                        width_calving = width_t0[np.where(heights == heights[glac_idx_t0].min())[0][0]] * 1000
                    elif pygem_prms.hyps_data in ['Huss', 'Farinotti']:
                        if glacier_rgi_table.RGIId in pygem_prms.width_calving_dict:
                            width_calving = np.float64(pygem_prms.width_calving_dict[glacier_rgi_table.RGIId])
                        else:
                            width_calving = width_t0[glac_idx_t0[0]] * 1000                    
                    # Volume loss [m3] due to frontal ablation
                    frontalablation_volumeloss = (
                            np.max([0, (frontalablation_k * waterdepth * height_calving)]) * width_calving)
                    # Maximum volume loss is volume of bins with their bed elevation below sea level
                    glac_idx_fa = np.where((glac_bin_bedelev < sea_level) & (glacier_area_t0 > 0))[0]
                    frontalablation_volumeloss_max = glac_bin_volume[glac_idx_fa].sum()
                    if frontalablation_volumeloss > frontalablation_volumeloss_max:
                        frontalablation_volumeloss = frontalablation_volumeloss_max
                        
                    

                    if debug:
                        print('frontalablation_k:', frontalablation_k)
                        print('width calving:', width_calving)
                        print('frontalablation_volumeloss [m3]:', frontalablation_volumeloss)
                        print('frontalablation_massloss [Gt]:', frontalablation_volumeloss * pygem_prms.density_water / 
                              pygem_prms.density_ice / 10**9)
                        print('frontalalabion_volumeloss_max [Gt]:', frontalablation_volumeloss_max * pygem_prms.density_water / 
                              pygem_prms.density_ice / 10**9)
#                        print('glac_idx_fa:', glac_idx_fa)
#                        print('glac_bin_volume:', glac_bin_volume[0])
#                        print('glac_idx_fa[bin_count]:', glac_idx_fa[0])
#                        print('glac_bin_volume[glac_idx_fa[bin_count]]:', glac_bin_volume[glac_idx_fa[0]])
#                        print('glacier_area_t0[glac_idx_fa[bin_count]]:', glacier_area_t0[glac_idx_fa[0]])
#                        print('glac_bin_frontalablation:', glac_bin_frontalablation[glac_idx_fa[0], step])
                    
                    # Frontal ablation [mwe] in each bin
                    bin_count = 0
                    while (frontalablation_volumeloss > pygem_prms.tolerance) and (bin_count < len(glac_idx_fa)):
                        # Sort heights to ensure it's universal (works with OGGM and Huss)
                        heights_calving_sorted = np.argsort(heights[glac_idx_fa])
                        calving_bin_idx = heights_calving_sorted[bin_count]
                        # Check if entire bin removed or not
                        if frontalablation_volumeloss >= glac_bin_volume[glac_idx_fa[calving_bin_idx]]:
                            glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] = (
                                    glac_bin_volume[glac_idx_fa[calving_bin_idx]] / 
                                    (glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6) 
                                    * pygem_prms.density_ice / pygem_prms.density_water)   
                        else:
                            glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] = (
                                    frontalablation_volumeloss / (glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6)
                                    * pygem_prms.density_ice / pygem_prms.density_water)
                        frontalablation_volumeloss += (
                                -1 * glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] * pygem_prms.density_water 
                                / pygem_prms.density_ice * glacier_area_t0[glac_idx_fa[calving_bin_idx]] * 10**6)                        
                                                
                        if debug:
                            print('glacier idx:', glac_idx_fa[calving_bin_idx], 
                                  'volume loss:', (glac_bin_frontalablation[glac_idx_fa[calving_bin_idx], step] * 
                                  glacier_area_t0[glac_idx_fa[calving_bin_idx]] * pygem_prms.density_water / 
                                  pygem_prms.density_ice * 10**6).round(0))
                            print('remaining volume loss:', frontalablation_volumeloss, 'tolerance:', pygem_prms.tolerance)
                        
                        bin_count += 1         
                            
                    if debug:
                        print('frontalablation_volumeloss remaining [m3]:', frontalablation_volumeloss)
                        print('ice thickness:', icethickness_t0[glac_idx_fa[0]].round(0), 
                              'waterdepth:', waterdepth.round(0), 
                              'height calving front:', height_calving.round(0), 
                              'width [m]:', (width_calving).round(0))   
                        

                # SURFACE TYPE
                # Annual surface type [-]
                # Annual climatic mass balance [m w.e.], which is used to determine the surface type
                glac_bin_massbalclim_annual[:,year] = glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1)
                
                if debug:
                    print('glacier data:', heights[glac_idx_t0[0:5]], icethickness_t0[glac_idx_t0[0:5]])
#                    print('glac_bin_massbalclim:', glac_bin_massbalclim[:,12*year:12*(year+1)].sum(1))
                    print('glacier-wide climatic mass balance:', 
                          (glac_bin_massbalclim_annual[:,year] * glacier_area_t0).sum() / glacier_area_t0.sum())
#                    print('Climatic mass balance:', glac_bin_massbalclim_annual[:,year].sum())
                
#                # Compute the surface type for each bin
#                print('surface type2:', surfacetype[glac_idx_t0])
#                surfacetype, firnline_idx = surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year)
#                print('surface type3:', surfacetype[glac_idx_t0])
#                print('WHATS GOING ON WITH THIS??\n')
###                print(elev_bins)

                # MASS REDISTRIBUTION
                # Mass redistribution ignored for calibration and spinup years (glacier properties constant)
                if (option_areaconstant == 1) or (year < pygem_prms.ref_spinupyears) or (year < constantarea_years):
                    glacier_area_t1 = glacier_area_t0
                    icethickness_t1 = icethickness_t0
                    width_t1 = width_t0                    
                else:
                    
                    if debug:
                        print('area is changing')
                        
                    # First, remove volume lost to frontal ablation
                    #  changes to _t0 not _t1, since t1 will be done in the mass redistribution
                    if glac_bin_frontalablation[:,step].max() > 0:
                        # Frontal ablation loss [mwe]
                        #  fa_change tracks whether entire bin is lost or not
                        fa_change = abs(glac_bin_frontalablation[:, step] * pygem_prms.density_water / pygem_prms.density_ice
                                        - icethickness_t0)
                        fa_change[fa_change <= pygem_prms.tolerance] = 0
                        
                        if debug:
                            bins_wfa = np.where(glac_bin_frontalablation[:,step] > 0)[0]
                            print('glacier area t0:', glacier_area_t0[bins_wfa].round(3))
                            print('ice thickness t0:', icethickness_t0[bins_wfa].round(1))
                            print('frontalablation [m ice]:', (glac_bin_frontalablation[bins_wfa, step] * 
                                  pygem_prms.density_water / pygem_prms.density_ice).round(1))
                            print('frontal ablation [mice] vs icethickness:', fa_change[bins_wfa].round(1))
                        
                        # Check if entire bin is removed
                        glacier_area_t0[np.where(fa_change == 0)[0]] = 0
                        icethickness_t0[np.where(fa_change == 0)[0]] = 0
                        width_t0[np.where(fa_change == 0)[0]] = 0
                        # Otherwise, reduce glacier area such that glacier retreats and ice thickness remains the same
                        #  A_1 = (V_0 - V_loss) / h_1,  units: A_1 = (m ice * km2) / (m ice) = km2
                        glacier_area_t0[np.where(fa_change != 0)[0]] = (
                                (glacier_area_t0[np.where(fa_change != 0)[0]] * 
                                 icethickness_t0[np.where(fa_change != 0)[0]] - 
                                 glacier_area_t0[np.where(fa_change != 0)[0]] * 
                                 glac_bin_frontalablation[np.where(fa_change != 0)[0], step] * pygem_prms.density_water 
                                 / pygem_prms.density_ice) / icethickness_t0[np.where(fa_change != 0)[0]])
                        
                        if debug:
                            print('glacier area t1:', glacier_area_t0[bins_wfa].round(3))
                            print('ice thickness t1:', icethickness_t0[bins_wfa].round(1))
                    
                    # Redistribute mass if glacier was not fully removed by frontal ablation
                    if glacier_area_t0.max() > 0:
                        # Mass redistribution according to Huss empirical curves
                        glacier_area_t1, icethickness_t1, width_t1 = (
                                massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, 
                                                       glac_bin_massbalclim_annual, year, glac_idx_initial, 
                                                       glacier_area_initial, heights,
                                                       debug=False, 
                                                       hindcast=hindcast))
                        # update surface type for bins that have retreated
                        surfacetype[glacier_area_t1 == 0] = 0
                        
#                        print('\n-----')
#                        print(len(np.where(glacier_area_t0 > 0)[0]))
#                        print(len(np.where(glacier_area_t1 > 0)[0]))
#                        print(surfacetype)
##                        print(glacier_area_t1)
#                        print('-----\n')
#                        print('delete me')
#                        print(elev_bins)
#                        if year == 0:
#                            print('\n\nHERE IS WHERE I LEFT OFF CHECKING!\n\n')
                        
                        # update surface type for bins that have advanced 
                        surfacetype[(surfacetype == 0) & (glacier_area_t1 != 0)] = (
                                surfacetype[glacier_area_t0.nonzero()[0][0]])
                    else:
                        glacier_area_t1 = np.zeros(glacier_area_t0.shape)
                        icethickness_t1 = np.zeros(glacier_area_t0.shape)
                        width_t1 = np.zeros(glacier_area_t0.shape)
                        surfacetype = np.zeros(glacier_area_t0.shape)
                        
                # Record glacier properties (area [km**2], thickness [m], width [km])
                # if first year, record initial glacier properties (area [km**2], ice thickness [m ice], width [km])
                if year == 0:
                    glac_bin_area_annual[:,year] = glacier_area_initial
                    glac_bin_icethickness_annual[:,year] = icethickness_initial
                    glac_bin_width_annual[:,year] = width_initial
                # record the next year's properties as well
                # 'year + 1' used so the glacier properties are consistent with mass balance computations
                glac_bin_icethickness_annual[:,year + 1] = icethickness_t1
                glac_bin_area_annual[:,year + 1] = glacier_area_t1
                glac_bin_width_annual[:,year + 1] = width_t1
                # Update glacier properties for the mass balance computations
                icethickness_t0 = icethickness_t1.copy()
                glacier_area_t0 = glacier_area_t1.copy()
                width_t0 = width_t1.copy()   
                glac_bin_surfacetype_annual[:,year] = surfacetype
                #%%         
    
    # Remove the spinup years of the variables that are being exported
    if pygem_prms.timestep == 'monthly':
        colstart = pygem_prms.ref_spinupyears * annual_divisor
        colend = glacier_gcm_temp.shape[0] + 1
    bin_temp = bin_temp[:,colstart:colend]
    bin_prec = bin_prec[:,colstart:colend]
    bin_acc = bin_acc[:,colstart:colend]
    bin_melt = bin_melt[:,colstart:colend]
    bin_refreeze = bin_refreeze[:,colstart:colend]
    bin_snowpack = bin_snowpack[:,colstart:colend]
    glac_bin_refreeze = glac_bin_refreeze[:,colstart:colend]
    glac_bin_snowpack = glac_bin_snowpack[:,colstart:colend]
    glac_bin_melt = glac_bin_melt[:,colstart:colend]
    glac_bin_frontalablation = glac_bin_frontalablation[:,colstart:colend]
    glac_bin_massbalclim = glac_bin_massbalclim[:,colstart:colend]
    glac_bin_massbalclim_annual = glac_bin_massbalclim_annual[:,pygem_prms.ref_spinupyears:nyears+1]
    glac_bin_area_annual = glac_bin_area_annual[:,pygem_prms.ref_spinupyears:nyears+1]
    glac_bin_icethickness_annual = glac_bin_icethickness_annual[:,pygem_prms.ref_spinupyears:nyears+1]
    glac_bin_width_annual = glac_bin_width_annual[:,pygem_prms.ref_spinupyears:nyears+1]
    glac_bin_surfacetype_annual = glac_bin_surfacetype_annual[:,pygem_prms.ref_spinupyears:nyears+1]
    
    # Additional output:
    glac_bin_area = glac_bin_area_annual[:,0:glac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
    glac_wide_area = glac_bin_area.sum(axis=0)    
    glac_wide_prec = calc_glacwide(bin_prec, glac_bin_area, glac_wide_area)
    glac_wide_acc = calc_glacwide(bin_acc, glac_bin_area, glac_wide_area)
    glac_wide_refreeze = calc_glacwide(glac_bin_refreeze, glac_bin_area, glac_wide_area)
    glac_wide_melt = calc_glacwide(glac_bin_melt, glac_bin_area, glac_wide_area)
    glac_wide_frontalablation = calc_glacwide(glac_bin_frontalablation, glac_bin_area, glac_wide_area) 
    glac_wide_massbalclim = glac_wide_acc + glac_wide_refreeze - glac_wide_melt
    glac_wide_massbaltotal = glac_wide_massbalclim - glac_wide_frontalablation
    glac_wide_runoff = calc_runoff(glac_wide_prec, glac_wide_melt, glac_wide_refreeze, glac_wide_area)
    glac_wide_snowpack = calc_glacwide(glac_bin_snowpack, glac_bin_area, glac_wide_area)
    glac_wide_snowline = (glac_bin_snowpack > 0).argmax(axis=0)
    glac_wide_snowline[glac_wide_snowline > 0] = (heights[glac_wide_snowline[glac_wide_snowline > 0]] - 
                                                  pygem_prms.binsize/2)
    glac_wide_area_annual = glac_bin_area_annual.sum(axis=0)
    glac_wide_volume_annual = (glac_bin_area_annual * glac_bin_icethickness_annual / 1000).sum(axis=0)
    glac_wide_ELA_annual = (glac_bin_massbalclim_annual > 0).argmax(axis=0)
    glac_wide_ELA_annual[glac_wide_ELA_annual > 0] = (heights[glac_wide_ELA_annual[glac_wide_ELA_annual > 0]] - 
                                                      pygem_prms.binsize/2)  

    # OFF-GLACIER Output
    if option_areaconstant == 0:
        offglac_bin_area_annual = offglac_bin_area_annual[:,pygem_prms.ref_spinupyears:nyears+1]   
        offglac_bin_area = offglac_bin_area_annual[:,0:offglac_bin_area_annual.shape[1]-1].repeat(12,axis=1)
        offglac_wide_area = offglac_bin_area.sum(axis=0)
        offglac_wide_prec = calc_glacwide(offglac_bin_prec, offglac_bin_area, offglac_wide_area)
    #    offglac_wide_acc = calc_glacwide(bin_acc, offglac_bin_area, offglac_wide_area)
        offglac_wide_refreeze = calc_glacwide(offglac_bin_refreeze, offglac_bin_area, offglac_wide_area)
        offglac_wide_melt = calc_glacwide(offglac_bin_melt, offglac_bin_area, offglac_wide_area)
        offglac_wide_snowpack = calc_glacwide(offglac_bin_snowpack, offglac_bin_area, offglac_wide_area)
        offglac_wide_runoff = calc_runoff(offglac_wide_prec, offglac_wide_melt, offglac_wide_refreeze, 
                                          offglac_wide_area)        
    else:
        offglac_wide_prec = np.zeros(glac_wide_area.shape)
        offglac_wide_refreeze = np.zeros(glac_wide_area.shape)
        offglac_wide_melt = np.zeros(glac_wide_area.shape)
        offglac_wide_snowpack = np.zeros(glac_wide_area.shape)
        offglac_wide_runoff = np.zeros(glac_wide_area.shape)
    
    # Return the desired output
    return (bin_temp, bin_prec, bin_acc, glac_bin_refreeze, glac_bin_snowpack, glac_bin_melt, 
            glac_bin_frontalablation, glac_bin_massbalclim, glac_bin_massbalclim_annual, glac_bin_area_annual, 
            glac_bin_icethickness_annual, glac_bin_width_annual, glac_bin_surfacetype_annual, 
            glac_wide_massbaltotal, glac_wide_runoff, glac_wide_snowline, glac_wide_snowpack, glac_wide_area_annual, 
            glac_wide_volume_annual, glac_wide_ELA_annual, offglac_wide_prec, offglac_wide_refreeze, offglac_wide_melt,
            offglac_wide_snowpack, offglac_wide_runoff)  


#%% ===================================================================================================================
def annualweightedmean_array(var, dates_table):
    """
    Calculate annual mean of variable according to the timestep.
    
    Monthly timestep will group every 12 months, so starting month is important.
    
    Parameters
    ----------
    var : np.ndarray
        Variable with monthly or daily timestep
    dates_table : pd.DataFrame
        Table of dates, year, month, daysinmonth, wateryear, and season for each timestep
    Returns
    -------
    var_annual : np.ndarray
        Annual weighted mean of variable
    """        
    if pygem_prms.timestep == 'monthly':
        dayspermonth = dates_table['daysinmonth'].values.reshape(-1,12)
        #  creates matrix (rows-years, columns-months) of the number of days per month
        daysperyear = dayspermonth.sum(axis=1)
        #  creates an array of the days per year (includes leap years)
        weights = (dayspermonth / daysperyear[:,np.newaxis]).reshape(-1)
        #  computes weights for each element, then reshapes it from matrix (rows-years, columns-months) to an array, 
        #  where each column (each monthly timestep) is the weight given to that specific month
        var_annual = (var*weights[np.newaxis,:]).reshape(-1,12).sum(axis=1).reshape(-1,daysperyear.shape[0])
        #  computes matrix (rows - bins, columns - year) of weighted average for each year
        #  explanation: var*weights[np.newaxis,:] multiplies each element by its corresponding weight; .reshape(-1,12) 
        #    reshapes the matrix to only have 12 columns (1 year), so the size is (rows*cols/12, 12); .sum(axis=1) 
        #    takes the sum of each year; .reshape(-1,daysperyear.shape[0]) reshapes the matrix back to the proper 
        #    structure (rows - bins, columns - year)
        # If averaging a single year, then reshape so it returns a 1d array
        if var_annual.shape[1] == 1:
            var_annual = var_annual.reshape(var_annual.shape[0])
    elif pygem_prms.timestep == 'daily':
        print('\nError: need to code the groupbyyearsum and groupbyyearmean for daily timestep.'
              'Exiting the model run.\n')
        exit()
    return var_annual

def calc_glacwide(bin_var, area_bin, area_wide):
    """Calculate glacier wide sum of a variable"""
    var_wide = np.zeros(bin_var.shape[1])
    var_wide_mkm2 = (bin_var * area_bin).sum(axis=0)
    var_wide[var_wide_mkm2 > 0] = var_wide_mkm2[var_wide_mkm2 > 0] / area_wide[var_wide_mkm2 > 0]    
    return var_wide


def calc_runoff(prec_wide, melt_wide, refreeze_wide, area_wide):
    """
    Calculate runoff from precipitation, melt, and refreeze [units: m3]
      units: (m + m w.e. - m w.e.) * km**2 * (1000 m / 1 km)**2 = m**3
    """
    return (prec_wide + melt_wide - refreeze_wide) * area_wide * 1000**2
   

def massredistributionHuss(glacier_area_t0, icethickness_t0, width_t0, glac_bin_massbalclim_annual, year, 
                           glac_idx_initial, glacier_area_initial, heights, debug=False, hindcast=0):
    """
    Mass redistribution according to empirical equations from Huss and Hock (2015) accounting for retreat/advance.
    glac_idx_initial is required to ensure that the glacier does not advance to area where glacier did not exist before
    (e.g., retreat and advance over a vertical cliff)
    
    Parameters
    ----------
    glacier_area_t0 : np.ndarray
        Glacier area [km2] from previous year for each elevation bin
    icethickness_t0 : np.ndarray
        Ice thickness [m] from previous year for each elevation bin
    width_t0 : np.ndarray
        Glacier width [km] from previous year for each elevation bin
    glac_bin_massbalclim_annual : np.ndarray
        Climatic mass balance [m w.e.] for each elevation bin and year
    year : int
        Count of the year of model run (first year is 0)
    glac_idx_initial : np.ndarray
        Initial glacier indices
    glacier_area_initial : np.ndarray
        Initial glacier array used to determine average terminus area in event that glacier is only one bin
    debug : Boolean
        option to turn on print statements for development or debugging of code (default False)
    Returns
    -------
    glacier_area_t1 : np.ndarray
        Updated glacier area [km2] for each elevation bin
    icethickness_t1 : np.ndarray
        Updated ice thickness [m] for each elevation bin
    width_t1 : np.ndarray
        Updated glacier width [km] for each elevation bin
    """        
    # Reset the annual glacier area and ice thickness
    glacier_area_t1 = np.zeros(glacier_area_t0.shape)
    icethickness_t1 = np.zeros(glacier_area_t0.shape)
    width_t1 = np.zeros(glacier_area_t0.shape)
    # Annual glacier-wide volume change [km**3]
    glacier_volumechange = ((glac_bin_massbalclim_annual[:, year] / 1000 * pygem_prms.density_water / 
                             pygem_prms.density_ice * glacier_area_t0).sum())
    # For hindcast simulations, volume change is the opposite
    if hindcast == 1:
        glacier_volumechange = -1 * glacier_volumechange
        
    if debug:
        print('\nDebugging Mass Redistribution Huss function\n')
        print('glacier volume change:', glacier_volumechange)
    #  units: [m w.e.] * (1 km / 1000 m) * (1000 kg / (1 m water * m**2) * (1 m ice * m**2 / 900 kg) * [km**2] 
    #         = km**3 ice          
    # If volume loss is less than the glacier volume, then redistribute mass loss/gains across the glacier;
    #  otherwise, the glacier disappears (area and thickness were already set to zero above)
    if -1 * glacier_volumechange < (icethickness_t0 / 1000 * glacier_area_t0).sum():
        # Determine where glacier exists
        
        # Check for negative glacier areas
        #  shouldn't need these three lines, but Anna mentioned she was finding negative areas somehow? 2019/01/30
        glacier_area_t0[glacier_area_t0 < 0] = 0
        icethickness_t0[glacier_area_t0 < 0] = 0
        width_t0[glacier_area_t0 < 0] = 0
        
        glac_idx_t0 = glacier_area_t0.nonzero()[0]
        # Compute ice thickness [m ice], glacier area [km**2] and ice thickness change [m ice] after 
        #  redistribution of gains/losses
        if pygem_prms.option_massredistribution == 1:
            # Option 1: apply mass redistribution using Huss' empirical geometry change equations
            icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                    massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0,
                                                glacier_volumechange, glac_bin_massbalclim_annual[:, year],
                                                heights, debug=False))
            if debug:
                print(icethickness_t0.max(), icethickness_t1.max(), glacier_area_t0.max(), glacier_area_t1.max())

        # Glacier retreat
        #  if glacier retreats (ice thickness < 0), then ice thickness is set to zero, and some volume change will need 
        #   to be redistributed across the rest of the glacier
        while glacier_volumechange_remaining < 0:
            glacier_area_t0_retreated = glacier_area_t1.copy()
            icethickness_t0_retreated = icethickness_t1.copy()
            width_t0_retreated = width_t1.copy()
            glacier_volumechange_remaining_retreated = glacier_volumechange_remaining.copy()
            glac_idx_t0_retreated = glacier_area_t0_retreated.nonzero()[0]            
            # Set climatic mass balance for the case when there are less than 3 bins  
            #  distribute the remaining glacier volume change over the entire glacier (remaining bins)
            massbal_clim_retreat = np.zeros(glacier_area_t0_retreated.shape)
            massbal_clim_retreat[glac_idx_t0_retreated] = (glacier_volumechange_remaining / 
                                                           glacier_area_t0_retreated.sum() * 1000)
            # Mass redistribution 
            if pygem_prms.option_massredistribution == 1:
                # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                        massredistributioncurveHuss(icethickness_t0_retreated, glacier_area_t0_retreated, 
                                                    width_t0_retreated, glac_idx_t0_retreated, 
                                                    glacier_volumechange_remaining_retreated, massbal_clim_retreat,
                                                    heights)) 

        # Glacier advances 
        #  based on ice thickness change exceeding threshold
        #  Overview:
        #    1. If last bin is not full, i.e., area >= area of average terminus bin, then fill it up
        #    2. Add new bin and fill it up
        #    3. If additional volume after adding new bin, then redistribute mass gain across all bins again,
        #       i.e., increase the ice thickness and width
        #    4. Repeat adding a new bin and redistributing the mass until no addiitonal volume is left
        while (icethickness_change > pygem_prms.icethickness_advancethreshold).any() == True: 
            if debug:
                print('advancing glacier')
            # Record glacier area and ice thickness before advance corrections applied
            glacier_area_t1_raw = glacier_area_t1.copy()
            icethickness_t1_raw = icethickness_t1.copy()
            width_t1_raw = width_t1.copy()
            # Index bins that are advancing
            icethickness_change[icethickness_change <= pygem_prms.icethickness_advancethreshold] = 0
            glac_idx_advance = icethickness_change.nonzero()[0]
            # Update ice thickness based on maximum advance threshold [m ice]
            icethickness_t1[glac_idx_advance] = (icethickness_t1[glac_idx_advance] - 
                           (icethickness_change[glac_idx_advance] - pygem_prms.icethickness_advancethreshold))
            # Update glacier area based on reduced ice thicknesses [km**2]
            if pygem_prms.option_glaciershape == 1:
                # Glacier area for parabola [km**2] (A_1 = A_0 * (H_1 / H_0)**0.5)
                glacier_area_t1[glac_idx_advance] = (glacier_area_t1_raw[glac_idx_advance] * 
                               (icethickness_t1[glac_idx_advance] / icethickness_t1_raw[glac_idx_advance])**0.5)
                # Glacier width for parabola [km] (w_1 = w_0 * A_1 / A_0)
                width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                              / glacier_area_t1_raw[glac_idx_advance])
            elif pygem_prms.option_glaciershape == 2:
                # Glacier area constant for rectangle [km**2] (A_1 = A_0)
                glacier_area_t1[glac_idx_advance] = glacier_area_t1_raw[glac_idx_advance]
                # Glacier with constant for rectangle [km] (w_1 = w_0)
                width_t1[glac_idx_advance] = width_t1_raw[glac_idx_advance]
            elif pygem_prms.option_glaciershape == 3:
                # Glacier area for triangle [km**2] (A_1 = A_0 * H_1 / H_0)
                glacier_area_t1[glac_idx_t0] = (glacier_area_t1_raw[glac_idx_t0] * 
                               icethickness_t1[glac_idx_t0] / icethickness_t1_raw[glac_idx_t0])
                # Glacier width for triangle [km] (w_1 = w_0 * A_1 / A_0)
                width_t1[glac_idx_advance] = (width_t1_raw[glac_idx_advance] * glacier_area_t1[glac_idx_advance] 
                                              / glacier_area_t1_raw[glac_idx_advance])
            # Advance volume [km**3]
            advance_volume = ((glacier_area_t1_raw[glac_idx_advance] * 
                              icethickness_t1_raw[glac_idx_advance] / 1000).sum() - 
                              (glacier_area_t1[glac_idx_advance] * icethickness_t1[glac_idx_advance] / 
                               1000).sum())

            # Advance characteristics
            # Indices that define the glacier terminus
            glac_idx_terminus = (
                    glac_idx_t0[(heights[glac_idx_t0] - heights[glac_idx_t0].min()) / 
                                (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()) * 100 
                                < pygem_prms.terminus_percentage])
            # For glaciers with so few bands that the terminus is not identified (ex. <= 4 bands for 20% threshold),
            #  then use the information from all the bands
            if glac_idx_terminus.shape[0] <= 1:
                glac_idx_terminus = glac_idx_t0.copy()
            
            if debug:
                print('glacier index terminus:',glac_idx_terminus)
                
            # Average area of glacier terminus [km**2]
            #  exclude the bin at the terminus, since this bin may need to be filled first
            try:
                minelev_idx = np.where(heights == heights[glac_idx_terminus].min())[0][0]
                glac_idx_terminus_removemin = list(glac_idx_terminus)
                glac_idx_terminus_removemin.remove(minelev_idx)
                terminus_area_avg = np.mean(glacier_area_t0[glac_idx_terminus_removemin])
            except:  
                glac_idx_terminus_initial = (
                    glac_idx_initial[(heights[glac_idx_initial] - heights[glac_idx_initial].min()) / 
                                (heights[glac_idx_initial].max() - heights[glac_idx_initial].min()) * 100 
                                < pygem_prms.terminus_percentage])
                if glac_idx_terminus_initial.shape[0] <= 1:
                    glac_idx_terminus_initial = glac_idx_initial.copy()
                    
                minelev_idx = np.where(heights == heights[glac_idx_terminus_initial].min())[0][0]
                glac_idx_terminus_removemin = list(glac_idx_terminus_initial)
                glac_idx_terminus_removemin.remove(minelev_idx)
                terminus_area_avg = np.mean(glacier_area_t0[glac_idx_terminus_removemin])

            # Check if the last bin's area is below the terminus' average and fill it up if it is
            if (glacier_area_t1[minelev_idx] < terminus_area_avg) and (icethickness_t0[minelev_idx] <
               icethickness_t0[glac_idx_terminus].mean()):
                # Volume required to fill the bin at the terminus
                advance_volume_fillbin = (icethickness_t1[minelev_idx] / 1000 * 
                                          (terminus_area_avg - glacier_area_t1[minelev_idx]))
                # If the advance volume is less than that required to fill the bin, then fill the bin as much as
                #  possible by adding area (thickness remains the same - glacier front is only thing advancing)
                if advance_volume < advance_volume_fillbin:
                    # add advance volume to the bin (area increases, thickness and width constant)
                    glacier_area_t1[minelev_idx] = (glacier_area_t1[minelev_idx] + 
                                                    advance_volume / (icethickness_t1[minelev_idx] / 1000))
                    # set advance volume equal to zero
                    advance_volume = 0
                else:
                    # fill the bin (area increases, thickness and width constant)
                    glacier_area_t1[minelev_idx] = (glacier_area_t1[minelev_idx] + 
                                                    advance_volume_fillbin / (icethickness_t1[minelev_idx] / 1000))
                    advance_volume = advance_volume - advance_volume_fillbin                    

            # With remaining advance volume, add a bin or redistribute over existing bins if no bins left
            if advance_volume > 0:
                # Indices for additional bins below the terminus
                below_glac_idx = np.where(heights < heights[glacier_area_t1 > 0].min())[0]
                # if no more bins below, then distribute volume over the glacier without further adjustments
                if len(below_glac_idx) == 0:
                    glacier_area_t1 = glacier_area_t1_raw
                    icethickness_t1 = icethickness_t1_raw
                    width_t1 = width_t1_raw
                    advance_volume = 0
                    
                # otherwise, add a bin with thickness and width equal to the previous bin and fill it up
                else:
                    # Sort heights below terminus to ensure it's universal (works with OGGM and Huss)
                    elev_bin2add_sorted = heights[below_glac_idx][np.argsort(heights[below_glac_idx])[::-1]]
                    bin2add_count = 0
                    glac_idx_bin2add = np.where(heights == elev_bin2add_sorted[bin2add_count])[0][0] 
                    # Check if bin2add is in a discontinuous section of the initial glacier
                    while (heights[glac_idx_bin2add] > heights[glac_idx_initial].min() and 
                           glac_idx_bin2add in list(glac_idx_initial)):
                        # Advance should not occur in a discontinuous section of the glacier (e.g., vertical drop),
                        #  so change the bin2add to the next bin down valley
                        bin2add_count += 1
                        glac_idx_bin2add = np.where(heights == elev_bin2add_sorted[bin2add_count])[0][0]
                        
                    # ice thickness of new bin equals ice thickness of bin at the terminus
                    icethickness_t1[glac_idx_bin2add] = icethickness_t1[minelev_idx]
                    width_t1[glac_idx_bin2add] = width_t1[minelev_idx]
                    # volume required to fill the bin at the terminus
                    advance_volume_fillbin = icethickness_t1[glac_idx_bin2add] / 1000 * terminus_area_avg 
                    # If the advance volume is unable to fill entire bin, then fill it as much as possible
                    if advance_volume < advance_volume_fillbin:
                        # add advance volume to the bin (area increases, thickness and width constant)
                        glacier_area_t1[glac_idx_bin2add] = (advance_volume / (icethickness_t1[glac_idx_bin2add]
                                                             / 1000))
                        advance_volume = 0
                    else:
                        # fill the bin (area increases, thickness and width constant)
                        glacier_area_t1[glac_idx_bin2add] = terminus_area_avg
                        advance_volume = advance_volume - advance_volume_fillbin
                        
            # update the glacier indices
            glac_idx_t0 = glacier_area_t1.nonzero()[0]
            massbal_clim_advance = np.zeros(glacier_area_t1.shape)
            # Record glacier area and ice thickness before advance corrections applied
            glacier_area_t1_raw = glacier_area_t1.copy()
            icethickness_t1_raw = icethickness_t1.copy()
            width_t1_raw = width_t1.copy()
            # If a full bin has been added and volume still remains, then redistribute mass across the
            #  glacier, thereby enabling the bins to get thicker once again prior to adding a new bin.
            #  This is important for glaciers that have very thin ice at the terminus as this prevents the 
            #  glacier from having a thin layer of ice advance tremendously far down valley without thickening.
            if advance_volume > 0:
                if pygem_prms.option_massredistribution == 1:
                    # Option 1: apply mass redistribution using Huss' empirical geometry change equations
                    icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining = (
                            massredistributioncurveHuss(icethickness_t1, glacier_area_t1, width_t1, glac_idx_t0, 
                                                        advance_volume, massbal_clim_advance, heights))
            # update ice thickness change
            icethickness_change = icethickness_t1 - icethickness_t1_raw

    return glacier_area_t1, icethickness_t1, width_t1


def massredistributioncurveHuss(icethickness_t0, glacier_area_t0, width_t0, glac_idx_t0, glacier_volumechange, 
                                massbalclim_annual, heights, debug=False):
    """
    Apply the mass redistribution curves from Huss and Hock (2015).
    This is paired with massredistributionHuss, which takes into consideration retreat and advance.
    
    To-do list
    ----------
    - volume-length scaling
    - volume-area scaling
    - pair with OGGM flow model
    
    Parameters
    ----------
    icethickness_t0 : np.ndarray
        Ice thickness [m] from previous year for each elevation bin
    glacier_area_t0 : np.ndarray
        Glacier area [km2] from previous year for each elevation bin
    width_t0 : np.ndarray
        Glacier width [km] from previous year for each elevation bin
    massbalclim_annual : np.ndarray
        Annual climatic mass balance [m w.e.] for each elevation bin for a single year
    glac_idx_t0 : np.ndarray
        glacier indices for present timestep
    glacier_volumechange : float
        glacier-wide volume change [km3] based on the annual climatic mass balance
    Returns
    -------
    glacier_area_t1 : np.ndarray
        Updated glacier area [km2] for each elevation bin
    icethickness_t1 : np.ndarray
        Updated ice thickness [m] for each elevation bin
    width_t1 : np.ndarray
        Updated glacier width [km] for each elevation bin
    icethickness_change : np.ndarray
        Ice thickness change [m] for each elevation bin
    glacier_volumechange_remaining : float
        Glacier volume change remaining, which could occur if there is less ice in a bin than melt, i.e., retreat
    """           
    if debug:
        print('\nDebugging mass redistribution curve Huss\n')
    # Apply Huss redistribution if there are at least 3 elevation bands; otherwise, use the mass balance
    # reset variables
    icethickness_t1 = np.zeros(glacier_area_t0.shape)
    glacier_area_t1 = np.zeros(glacier_area_t0.shape)
    width_t1 = np.zeros(glacier_area_t0.shape) 
    if glac_idx_t0.shape[0] > 3:
        # Select the factors for the normalized ice thickness change curve based on glacier area
        if glacier_area_t0.sum() > 20:
            [gamma, a, b, c] = [6, -0.02, 0.12, 0]
        elif glacier_area_t0.sum() > 5:
            [gamma, a, b, c] = [4, -0.05, 0.19, 0.01]
        else:
            [gamma, a, b, c] = [2, -0.30, 0.60, 0.09]
        # reset variables
        elevrange_norm = np.zeros(glacier_area_t0.shape)
        icethicknesschange_norm = np.zeros(glacier_area_t0.shape)
        # Normalized elevation range [-]
        #  (max elevation - bin elevation) / (max_elevation - min_elevation)
        elevrange_norm[glacier_area_t0 > 0] = ((heights[glac_idx_t0].max() - heights[glac_idx_t0]) / 
                                               (heights[glac_idx_t0].max() - heights[glac_idx_t0].min()))
        
        #  using indices as opposed to elevations automatically skips bins on the glacier that have no area
        #  such that the normalization is done only on bins where the glacier lies
        # Normalized ice thickness change [-]
        icethicknesschange_norm[glacier_area_t0 > 0] = ((elevrange_norm[glacier_area_t0 > 0] + a)**gamma + 
                                                        b*(elevrange_norm[glacier_area_t0 > 0] + a) + c)
        #  delta_h = (h_n + a)**gamma + b*(h_n + a) + c
        #  indexing is faster here
        # limit the icethicknesschange_norm to between 0 - 1 (ends of fxns not exactly 0 and 1)
        icethicknesschange_norm[icethicknesschange_norm > 1] = 1
        icethicknesschange_norm[icethicknesschange_norm < 0] = 0
        # Huss' ice thickness scaling factor, fs_huss [m ice]         
        #  units: km**3 / (km**2 * [-]) * (1000 m / 1 km) = m ice
        fs_huss = glacier_volumechange / (glacier_area_t0 * icethicknesschange_norm).sum() * 1000
        if debug:
            print('fs_huss:', fs_huss)
        # Volume change [km**3 ice]
        bin_volumechange = icethicknesschange_norm * fs_huss / 1000 * glacier_area_t0

    # Otherwise, compute volume change in each bin based on the climatic mass balance
    else:
        bin_volumechange = massbalclim_annual / 1000 * glacier_area_t0        
    if pygem_prms.option_glaciershape == 1:
        # Ice thickness at end of timestep for parabola [m ice]
        #  run in two steps to avoid errors with negative numbers and fractional exponents
        #  H_1 = (H_0**1.5 + delta_Vol * H_0**0.5 / A_0)**(2/3)
        icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**1.5 + 
                       (icethickness_t0[glac_idx_t0] / 1000)**0.5 * bin_volumechange[glac_idx_t0] / 
                       glacier_area_t0[glac_idx_t0])
        icethickness_t1[icethickness_t1 < 0] = 0
        icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(2/3) * 1000        
        # Glacier area for parabola [km**2]
        #  A_1 = A_0 * (H_1 / H_0)**0.5
        glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * (icethickness_t1[glac_idx_t0] / 
                                        icethickness_t0[glac_idx_t0])**0.5)
        # Glacier width for parabola [km]
        #  w_1 = w_0 * (A_1 / A_0)
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
    elif pygem_prms.option_glaciershape == 2:
        # Ice thickness at end of timestep for rectangle [m ice]
        #  H_1 = H_0 + delta_Vol / A_0
        icethickness_t1[glac_idx_t0] = (((icethickness_t0[glac_idx_t0] / 1000) + 
                                         bin_volumechange[glac_idx_t0] / glacier_area_t0[glac_idx_t0]) * 1000)
        # Glacier area constant for rectangle [km**2]
        #  A_1 = A_0
        glacier_area_t1[glac_idx_t0] = glacier_area_t0[glac_idx_t0]
        # Glacier width constant for rectangle [km]
        #  w_1 = w_0
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0]
    elif pygem_prms.option_glaciershape == 3:
        # Ice thickness at end of timestep for triangle [m ice]
        #  run in two steps to avoid errors with negative numbers and fractional exponents
        icethickness_t1[glac_idx_t0] = ((icethickness_t0[glac_idx_t0] / 1000)**2 + 
                       bin_volumechange[glac_idx_t0] * (icethickness_t0[glac_idx_t0] / 1000) / 
                       glacier_area_t0[glac_idx_t0])                                   
        icethickness_t1[icethickness_t1 < 0] = 0
        icethickness_t1[glac_idx_t0] = icethickness_t1[glac_idx_t0]**(1/2) * 1000
        # Glacier area for triangle [km**2]
        #  A_1 = A_0 * H_1 / H_0
        glacier_area_t1[glac_idx_t0] = (glacier_area_t0[glac_idx_t0] * icethickness_t1[glac_idx_t0] / 
                                        icethickness_t0[glac_idx_t0])
        # Glacier width for triangle [km]
        #  w_1 = w_0 * (A_1 / A_0)
        width_t1[glac_idx_t0] = width_t0[glac_idx_t0] * glacier_area_t1[glac_idx_t0] / glacier_area_t0[glac_idx_t0]
    # Ice thickness change [m ice]
    icethickness_change = icethickness_t1 - icethickness_t0
    # Compute the remaining volume change
    bin_volumechange_remaining = bin_volumechange - ((glacier_area_t1 * icethickness_t1 - glacier_area_t0 * 
                                                      icethickness_t0) / 1000)
    # remove values below tolerance to avoid rounding errors
    bin_volumechange_remaining[abs(bin_volumechange_remaining) < pygem_prms.tolerance] = 0
    # Glacier volume change remaining - if less than zero, then needed for retreat
    glacier_volumechange_remaining = bin_volumechange_remaining.sum()    
    return icethickness_t1, glacier_area_t1, width_t1, icethickness_change, glacier_volumechange_remaining
    

def surfacetypebinsannual(surfacetype, glac_bin_massbalclim_annual, year_index):
    """
    Update surface type according to climatic mass balance over the last five years.  
    
    If 5-year climatic balance is positive, then snow/firn.  If negative, then ice/debris.
    Convention: 0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris
    
    Function Options:
      > 1 (default) - update surface type according to Huss and Hock (2015)
      > 2 - Radic and Hock (2011)
    Huss and Hock (2015): Initially, above median glacier elevation is firn and below is ice. Surface type updated for
      each elevation band and month depending on the specific mass balance.  If the cumulative balance since the start 
      of the mass balance year is positive, then snow is assigned. If the cumulative mass balance is negative (i.e., 
      all snow of current mass balance year has melted), then bare ice or firn is exposed. Surface type is assumed to 
      be firn if the elevation band's average annual balance over the preceding 5 years (B_t-5_avg) is positive. If
      B_t-5_avg is negative, surface type is ice.
          > climatic mass balance calculated at each bin and used with the mass balance over the last 5 years to 
            determine whether the surface is firn or ice.  Snow is separate based on each month.
    Radic and Hock (2011): "DDF_snow is used above the ELA regardless of snow cover.  Below the ELA, use DDF_ice is 
      used only when snow cover is 0.  ELA is calculated from the observed annual mass balance profiles averaged over 
      the observational period and is kept constant in time for the calibration period.  For the future projections, 
      ELA is set to the mean glacier height and is time dependent since glacier volume, area, and length are time 
      dependent (volume-area-length scaling).
      Bliss et al. (2014) uses the same as Valentina's model
    
    Parameters
    ----------
    surfacetype : np.ndarray
        Surface type for each elevation bin
    glac_bin_massbalclim_annual : np.ndarray
        Annual climatic mass balance for each year and each elevation bin
    year_index : int
        Count of the year of model run (first year is 0)
    Returns
    -------
    surfacetype : np.ndarray
        Updated surface type for each elevation bin
    firnline_idx : int
        Firn line index
    """        
    # Next year's surface type is based on the bin's average annual climatic mass balance over the last 5 years.  If 
    #  less than 5 years, then use the average of the existing years.
    if year_index < 5:
        # Calculate average annual climatic mass balance since run began
        massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,0:year_index+1].mean(1)
    else:
        massbal_clim_mwe_runningavg = glac_bin_massbalclim_annual[:,year_index-4:year_index+1].mean(1)
    # If the average annual specific climatic mass balance is negative, then the surface type is ice (or debris)
    surfacetype[(surfacetype !=0 ) & (massbal_clim_mwe_runningavg <= 0)] = 1
    # If the average annual specific climatic mass balance is positive, then the surface type is snow (or firn)
    surfacetype[(surfacetype != 0) & (massbal_clim_mwe_runningavg > 0)] = 2
    # Compute the firnline index
    try:
        # firn in bins >= firnline_idx
        firnline_idx = np.where(surfacetype==2)[0][0]
    except:
        # avoid errors if there is no firn, i.e., the entire glacier is melting
        firnline_idx = np.where(surfacetype!=0)[0][-1]
    # Apply surface type model options
    # If firn surface type option is included, then snow is changed to firn
    if pygem_prms.include_firn:
        surfacetype[surfacetype == 2] = 3
    return surfacetype, firnline_idx


def surfacetypebinsinitial(glacier_area, glacier_table, elev_bins):
    """
    Define initial surface type according to median elevation such that the melt can be calculated over snow or ice.
    
    Convention: (0 = off-glacier, 1 = ice, 2 = snow, 3 = firn, 4 = debris).
    Function options: 1 =
    
    Function options specified in pygem_pygem_prms.py:
    - option_surfacetype_initial
        > 1 (default) - use median elevation to classify snow/firn above the median and ice below
        > 2 - use mean elevation instead
    
    To-do list
    ----------
    Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions
    
    Parameters
    ----------
    glacier_area : np.ndarray
        Glacier area [km2] from previous year for each elevation bin
    glacier_table : pd.Series
        Table of glacier's RGI information
    elev_bins : np.ndarray
        Elevation bins [masl]
    Returns
    -------
    surfacetype : np.ndarray
        Updated surface type for each elevation bin
    firnline_idx : int
        Firn line index
    """        
    surfacetype = np.zeros(glacier_area.shape)
    # Option 1 - initial surface type based on the median elevation
    if pygem_prms.option_surfacetype_initial == 1:
        surfacetype[(elev_bins < glacier_table.loc['Zmed']) & (glacier_area > 0)] = 1
        surfacetype[(elev_bins >= glacier_table.loc['Zmed']) & (glacier_area > 0)] = 2
    # Option 2 - initial surface type based on the mean elevation
    elif pygem_prms.option_surfacetype_initial ==2:
        surfacetype[(elev_bins < glacier_table['Zmean']) & (glacier_area > 0)] = 1
        surfacetype[(elev_bins >= glacier_table['Zmean']) & (glacier_area > 0)] = 2
    else:
        assert 0==1, "This option for 'option_surfacetype' does not exist. Please choose an option that exists."
    # Compute firnline index
    try:
        # firn in bins >= firnline_idx
        firnline_idx = np.where(surfacetype==2)[0][0]
    except:
        # avoid errors if there is no firn, i.e., the entire glacier is melting
        firnline_idx = np.where(surfacetype!=0)[0][-1]
    # If firn is included, then specify initial firn conditions
    if pygem_prms.include_firn:
        surfacetype[surfacetype == 2] = 3
        #  everything initially considered snow is considered firn, i.e., the model initially assumes there is no snow 
        #  on the surface anywhere.
    return surfacetype, firnline_idx


def surfacetypeDDFdict(modelparameters, include_firn=pygem_prms.include_firn,
                       option_ddf_firn=pygem_prms.option_ddf_firn):
    """
    Create a dictionary of surface type and its respective degree-day factor.
    
    Convention: [0=off-glacier, 1=ice, 2=snow, 3=firn, 4=debris]
    
    modelparameters[lr_gcm, lr_glac, prec_factor, prec_grad, DDF_snow, DDF_ice, T_snow]
    
    To-do list
    ----------
    Add option_surfacetype_initial to specify an AAR ratio and apply this to estimate initial conditions
    
    Parameters
    ----------
    modelparameters : pd.Series
        Model parameters (lrgcm, lrglac, precfactor, precgrad, ddfsnow, ddfice, tempsnow, tempchange)
        Order of model parameters should not be changed as the run mass balance script relies on this order
    include_firn : Boolean
        Option to include or exclude firn (specified in pygem_pygem_prms.py)
    option_ddf_firn : int
        Option for the degree day factor of firn to be the average of snow and ice or a different value
    Returns
    -------
    surfacetype_ddf_dict : dictionary
        Dictionary relating the surface types with their respective degree day factors
    """        
    surfacetype_ddf_dict = {
            0: modelparameters[4],
            1: modelparameters[5],
            2: modelparameters[4]}
    if include_firn == 1:
        if option_ddf_firn == 0:
            surfacetype_ddf_dict[3] = modelparameters[4]
        elif option_ddf_firn == 1:
            surfacetype_ddf_dict[3] = np.mean([modelparameters[4],modelparameters[5]])
    return surfacetype_ddf_dict