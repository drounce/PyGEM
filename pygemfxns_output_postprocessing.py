# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 09:04:46 2017

@author: David Rounce

pygemfxns_output.py is a list of functions that are used for post-processing the model results.  Specifically, these 
functions are meant to interact with the main model output to extract things like runoff, ELA, SLA, etc., which will
be specified by the user.  This allows the main script to run as quickly as possible and record only the minimum amount
of model results.
"""
import numpy as np
import pandas as pd 
import netCDF4 as nc
from time import strftime
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pickle

import pygem_input as input
import pygemfxns_modelsetup as modelsetup
import cartopy


#%%===== PLOT FUNCTIONS =============================================================================================
def plot_latlonvar(lons, lats, variable, rangelow, rangehigh, title, xlabel, ylabel, colormap, east, west, south, north, 
                   xtick, ytick):
    """
    Plot a variable according to its latitude and longitude
    """
    # Create the projection
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    # Add country borders for reference
    ax.add_feature(cartopy.feature.BORDERS)
    # Set the extent
    ax.set_extent([east, west, south, north], cartopy.crs.PlateCarree())
    # Label title, x, and y axes
    plt.title(title)
    ax.set_xticks(np.arange(east,west+1,xtick), cartopy.crs.PlateCarree())
    ax.set_yticks(np.arange(south,north+1,ytick), cartopy.crs.PlateCarree())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the data 
    plt.scatter(lons, lats, c=variable, cmap=colormap)
    #  plotting x, y, size [s=__], color bar [c=__]
    plt.clim(rangelow,rangehigh)
    #  set the range of the color bar
    plt.colorbar(fraction=0.02, pad=0.04)
    #  fraction resizes the colorbar, pad is the space between the plot and colorbar
    plt.show()
    

def plot_caloutput(data):
    """
    Plot maps and histograms of the calibration parameters to visualize results
    """
    # Set extent
    east = int(round(data['CenLon'].min())) - 1
    west = int(round(data['CenLon'].max())) + 1
    south = int(round(data['CenLat'].min())) - 1
    north = int(round(data['CenLat'].max())) + 1
    xtick = 1
    ytick = 1
    # Select relevant data
    lats = data['CenLat'][:]
    lons = data['CenLon'][:]
    precfactor = data['precfactor'][:]
    tempchange = data['tempchange'][:]
    ddfsnow = data['ddfsnow'][:]
    calround = data['calround'][:]
    massbal = data['MB_geodetic_mwea']
    # Plot regional maps
    plot_latlonvar(lons, lats, massbal, 'Geodetic mass balance [mwea]', 'longitude [deg]', 'latitude [deg]', east, west, 
               south, north, xtick, ytick)
    plot_latlonvar(lons, lats, precfactor, 'precipitation factor', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    plot_latlonvar(lons, lats, tempchange, 'Temperature bias [degC]', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    plot_latlonvar(lons, lats, ddfsnow, 'DDF_snow [m w.e. d-1 degC-1]', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    plot_latlonvar(lons, lats, calround, 'Calibration round', 'longitude [deg]', 'latitude [deg]', east, west, 
                   south, north, xtick, ytick)
    # Plot histograms
    data.hist(column='MB_difference_mwea', bins=50)
    plt.title('Mass Balance Difference [mwea]')
    data.hist(column='precfactor', bins=50)
    plt.title('Precipitation factor [-]')
    data.hist(column='tempchange', bins=50)
    plt.title('Temperature bias [degC]')
    data.hist(column='ddfsnow', bins=50)
    plt.title('DDFsnow [mwe d-1 degC-1]')
    plt.xticks(rotation=60)
    data.hist(column='calround', bins = [0.5, 1.5, 2.5, 3.5])
    plt.title('Calibration round')
    plt.xticks([1, 2, 3])

#%%===== PLOTTING GRID SEARCH FOR A GLACIER ======
#data = nc.Dataset(input.main_directory + '/../Output/calibration_gridsearchcoarse_R15_20180324.nc', 'r+')
## Extract glacier information
#main_glac_rgi = pd.DataFrame(data['glacierinfo'][:], columns=data['glacierinfoheader'][:])
## Import calibration data for comparison
#main_glac_calmassbal = pd.read_csv(input.main_directory + '/../Output/R15_shean_geodeticmb_sorted.csv')
## Set glacier number
## Ngozumpa Glacier
#glac = 595
## Khumbu Glacier
##glac = 667
## Rongbuk Glacier
##glac = 1582
## East Rongbuk Glacier
##glac = 1607
#
## Select model parameters
#grid_modelparameters = data['grid_modelparameters'][:]
#precfactor = grid_modelparameters[:,2]
#tempchange = grid_modelparameters[:,7]
#ddfsnow = grid_modelparameters[:,4]
#precgrad = grid_modelparameters[:,3]
#ddfsnow_unique = np.unique(ddfsnow)
#precgrad_unique = np.unique(precgrad)
## Calculate mass balance and difference between modeled and measured
#massbaltotal_mwea = data['massbaltotal_glac_monthly'][glac,:,:].sum(axis=1) / (main_glac_calmassbal.loc[glac,'year2'] - main_glac_calmassbal.loc[glac,'year1'] + 1)
#massbaltotal_mwea_cal = main_glac_calmassbal.loc[glac,'mb_mwea']
#difference = np.zeros((grid_modelparameters.shape[0],1))
#difference[:,0] = massbaltotal_mwea - massbaltotal_mwea_cal
#
#data_hist = np.concatenate((grid_modelparameters, difference), axis=1)
#data_hist = pd.DataFrame(data_hist, columns=['lrglac','lrgcm','precfactor','precgrad','ddfsnow','ddfice','tempsnow','tempchange','difference'])
## Plot histograms
##data_hist.hist(column='difference', bins=20)
##plt.title('Mass Balance Difference [mwea]')
#
## Plot map of calibration parameters
## setup the plot
#fig, ax = plt.subplots(1,1, figsize=(5,5))  
#markers = ['v','o','^']
#labels = ['0.0001', '0.0003', '0.0005']
## define the colormap
#cmap = plt.cm.jet_r
## extract all colors from the .jet map
#cmaplist = [cmap(i) for i in range(cmap.N)]
## create the new map
#cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
## define the bins and normalize
#stepmin = -5
#stepmax = 6
#stepsize = 2
#bounds = np.arange(stepmin, stepmax, stepsize)
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#for n in range(precgrad_unique.shape[0]):
#    precfactor_subset = precfactor[precgrad == precgrad_unique[n]]
#    if n == 0:
#        precfactor_subset = precfactor_subset - 0.05
#    elif n == 2:
#        precfactor_subset = precfactor_subset + 0.05
#    tempchange_subset = tempchange[precgrad == precgrad_unique[n]]
#    ddfsnow_subset = ddfsnow[precgrad == precgrad_unique[n]]
#    difference_subset = difference[precgrad == precgrad_unique[n]]
#    # Set size of markers based on DDFsnow
#    ddfsnow_norm = ddfsnow_subset.copy()
#    ddfsnow_norm[ddfsnow_subset == ddfsnow_unique[0]] = 10
#    ddfsnow_norm[ddfsnow_subset == ddfsnow_unique[1]] = 30
#    ddfsnow_norm[ddfsnow_subset == ddfsnow_unique[2]] = 50
#    ddfsnow_norm[ddfsnow_subset == ddfsnow_unique[3]] = 70
#    ddfsnow_norm[ddfsnow_subset == ddfsnow_unique[4]] = 90
#    # make the scatter
#    scat = ax.scatter(precfactor_subset, difference_subset, s=ddfsnow_norm, marker=markers[n], c=tempchange_subset, cmap=cmap, norm=norm, label=labels[n])
## create the colorbar
#cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
##cb = plt.colorbar()
#tick_loc = bounds + stepsize/2
#cb.set_ticks(tick_loc)
#cb.set_ticklabels((bounds + stepsize/2).astype(int))
#cb.set_label('Tempchange [degC]')
##ax.set_title('TITLE')
#plt.xlabel('precfactor')
#plt.xlim((0.65, 1.85))
#plt.xticks([0.75, 1, 1.25, 1.5, 1.75])
#plt.ylabel('Difference [m w.e.a.]')
#plt.ylim((-2,2))
##plt.legend(loc=2)
#plt.show()
#fig.savefig(input.main_directory + '/../output/' + main_glac_rgi.loc[glac,'RGIID'] + '_gridsearch.png')


#%% ===== PLOTTING: Future simulations =====
#netcdf_output = nc.Dataset(input.main_directory + '/../Output/PyGEM_R15O2_MPI-ESM-LR_rcp26_2000_2100_20180504.nc.')
## Select relevant data
#glacier_data = pd.DataFrame(netcdf_output['glacierparameter'][:])
#glacier_data.columns = netcdf_output['glacierparameters'][:]
#lats = glacier_data['lat'].values.astype(float)
#lons = glacier_data['lon'].values.astype(float)
#massbal_total = netcdf_output['massbaltotal_glac_monthly'][:]
#massbal_total_mwea = massbal_total.sum(axis=1)/(massbal_total.shape[1]/12)
#volume_glac_annual = netcdf_output['volume_glac_annual'][:]
#volume_reg_annual = volume_glac_annual.sum(axis=0)
#volume_reg_annualnorm = volume_reg_annual / volume_reg_annual[0]
#runoff_glac_monthly = netcdf_output['runoff_glac_monthly'][:]
#runoff_reg_monthly = runoff_glac_monthly.mean(axis=0)
#acc_glac_monthly = netcdf_output['acc_glac_monthly'][:]
#acc_reg_monthly = acc_glac_monthly.mean(axis=0)
#acc_reg_annual = np.sum(acc_reg_monthly.reshape(-1,12), axis=1)
#refreeze_glac_monthly = netcdf_output['refreeze_glac_monthly'][:]
#refreeze_reg_monthly = refreeze_glac_monthly.mean(axis=0)
#refreeze_reg_annual = np.sum(refreeze_reg_monthly.reshape(-1,12), axis=1)
#melt_glac_monthly = netcdf_output['melt_glac_monthly'][:]
#melt_reg_monthly = melt_glac_monthly.mean(axis=0)
#melt_reg_annual = np.sum(melt_reg_monthly.reshape(-1,12), axis=1)
#massbaltotal_glac_monthly = netcdf_output['massbaltotal_glac_monthly'][:]
#massbaltotal_reg_monthly = massbaltotal_glac_monthly.mean(axis=0)
#massbaltotal_reg_annual = np.sum(massbaltotal_reg_monthly.reshape(-1,12), axis=1)
#
#years = np.arange(2000, 2100 + 1)
#years_plus1 = np.arange(2000, 2100 + 2)
#month = np.arange(2000, 2100 + 1, 1/12)
#plt.plot(years_plus1,volume_reg_annualnorm, label='Region 15 O2')
##plt.plot(years,volume_reg_annualnorm14, label='Region 14')
#plt.ylabel('Volume normalized [-]', size=15)
#plt.legend()
#plt.show()
##plt.plot(month,runoff_reg_monthly, label='Region 15 O2')
##plt.ylabel('Runoff [m3 / month]', size=15)
##plt.legend()
##plt.show()
##plt.plot(month, massbaltotal_reg_monthly, label='massbal_total')
##plt.plot(month, acc_reg_monthly, label='accumulation')
##plt.plot(month, refreeze_reg_monthly, label='refreeze')
##plt.plot(month, -1*melt_reg_monthly, label='melt')
##plt.ylabel('monthly regional mean [m.w.e.] / month')
##plt.legend()
##plt.show()
#plt.plot(years, massbaltotal_reg_annual, label='massbal_total')
#plt.plot(years, acc_reg_annual, label='accumulation')
#plt.plot(years, refreeze_reg_annual, label='refreeze')
#plt.plot(years, -1*melt_reg_annual, label='melt')
#plt.ylabel('Region 15 O2 annual mean [m.w.e.]', size=15)
#plt.legend()
#plt.show()
#
## Set extent
#east = int(round(lons.min())) - 1
#west = int(round(lons.max())) + 1
#south = int(round(lats.min())) - 1
#north = int(round(lats.max())) + 1
#xtick = 1
#ytick = 1
## Plot regional maps
#plot_latlonvar(lons, lats, massbal_total_mwea, -1.5, 0.5, 'Modeled mass balance [mwea]', 'longitude [deg]', 
#               'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick)
## Plot only positive mass balance
#lons_pos = lons[massbal_total_mwea > 0]
#lats_pos = lats[massbal_total_mwea > 0]
#massbal_total_mwea_pos = massbal_total_mwea[massbal_total_mwea > 0]
#plot_latlonvar(lons_pos, lats_pos, massbal_total_mwea_pos, 0, 0.5, 'Modeled mass balance [mwea]', 'longitude [deg]', 
#               'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick)

#%%===== PLOTTING ===========================================================================================
#netcdf_output15 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion15_ERAInterim_calSheanMB_nearest_20180306.nc', 'r+')
#netcdf_output15 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion15_ERAInterim_calSheanMB_nearest_20180403.nc', 'r+')
#netcdf_output14 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion14_ERAInterim_calSheanMB_nearest_20180313.nc', 'r+')
#netcdf_output14 = nc.Dataset(input.main_directory + '/../Output/PyGEM_output_rgiregion14_ERAInterim_calSheanMB_transferAvg_20180313.nc', 'r+')

## Select relevant data
#glacier_data15 = pd.DataFrame(netcdf_output15['glacierparameter'][:])
#glacier_data15.columns = netcdf_output15['glacierparameters'][:]
#lats15 = glacier_data15['lat'].values.astype(float)
#lons15 = glacier_data15['lon'].values.astype(float)
#massbal_total15 = netcdf_output15['massbaltotal_glac_monthly'][:]
#massbal_total_mwea15 = massbal_total15.sum(axis=1)/(massbal_total15.shape[1]/12)
#volume_glac_annual15 = netcdf_output15['volume_glac_annual'][:]
#volume_reg_annual15 = volume_glac_annual15.sum(axis=0)
#volume_reg_annualnorm15 = volume_reg_annual15 / volume_reg_annual15[0]
#runoff_glac_monthly15 = netcdf_output15['runoff_glac_monthly'][:]
#runoff_reg_monthly15 = runoff_glac_monthly15.mean(axis=0)
#acc_glac_monthly15 = netcdf_output15['acc_glac_monthly'][:]
#acc_reg_monthly15 = acc_glac_monthly15.mean(axis=0)
#acc_reg_annual15 = np.sum(acc_reg_monthly15.reshape(-1,12), axis=1)
#refreeze_glac_monthly15 = netcdf_output15['refreeze_glac_monthly'][:]
#refreeze_reg_monthly15 = refreeze_glac_monthly15.mean(axis=0)
#refreeze_reg_annual15 = np.sum(refreeze_reg_monthly15.reshape(-1,12), axis=1)
#melt_glac_monthly15 = netcdf_output15['melt_glac_monthly'][:]
#melt_reg_monthly15 = melt_glac_monthly15.mean(axis=0)
#melt_reg_annual15 = np.sum(melt_reg_monthly15.reshape(-1,12), axis=1)
#massbaltotal_glac_monthly15 = netcdf_output15['massbaltotal_glac_monthly'][:]
#massbaltotal_reg_monthly15 = massbaltotal_glac_monthly15.mean(axis=0)
#massbaltotal_reg_annual15 = np.sum(massbaltotal_reg_monthly15.reshape(-1,12), axis=1)
##glacier_data14 = pd.DataFrame(netcdf_output14['glacierparameter'][:])
##glacier_data14.columns = netcdf_output14['glacierparameters'][:]
##lats14 = glacier_data14['lat'].values.astype(float)
##lons14 = glacier_data14['lon'].values.astype(float)
##massbal_total14 = netcdf_output14['massbaltotal_glac_monthly'][:]
##massbal_total_mwea14 = massbal_total14.sum(axis=1)/(massbal_total14.shape[1]/12)
##volume_glac_annual14 = netcdf_output14['volume_glac_annual'][:]
##volume_reg_annual14 = volume_glac_annual14.sum(axis=0)
##volume_reg_annualnorm14 = volume_reg_annual14 / volume_reg_annual14[0]
##runoff_glac_monthly14 = netcdf_output14['runoff_glac_monthly'][:]
##runoff_reg_monthly14 = runoff_glac_monthly14.mean(axis=0)
##acc_glac_monthly14 = netcdf_output14['acc_glac_monthly'][:]
##acc_reg_monthly14 = acc_glac_monthly14.mean(axis=0)
##acc_reg_annual14 = np.sum(acc_reg_monthly14.reshape(-1,12), axis=1)
##refreeze_glac_monthly14 = netcdf_output14['refreeze_glac_monthly'][:]
##refreeze_reg_monthly14 = refreeze_glac_monthly14.mean(axis=0)
##refreeze_reg_annual14 = np.sum(refreeze_reg_monthly14.reshape(-1,12), axis=1)
##melt_glac_monthly14 = netcdf_output14['melt_glac_monthly'][:]
##melt_reg_monthly14 = melt_glac_monthly14.mean(axis=0)
##melt_reg_annual14 = np.sum(melt_reg_monthly14.reshape(-1,12), axis=1)
##massbaltotal_glac_monthly14 = netcdf_output14['massbaltotal_glac_monthly'][:]
##massbaltotal_reg_monthly14 = massbaltotal_glac_monthly14.mean(axis=0)
##massbaltotal_reg_annual14 = np.sum(massbaltotal_reg_monthly14.reshape(-1,12), axis=1)
#years = np.arange(2000, 2016 + 1)
#month = np.arange(2000, 2016, 1/12)
#plt.plot(years,volume_reg_annualnorm15, label='Region 15')
##plt.plot(years,volume_reg_annualnorm14, label='Region 14')
#plt.ylabel('Volume normalized [-]', size=15)
#plt.legend()
#plt.show()
#plt.plot(month,runoff_reg_monthly15, label='Region 15')
#plt.ylabel('Runoff [m3 / month]', size=15)
#plt.legend()
#plt.show()
#plt.plot(month, massbaltotal_reg_monthly15, label='massbal_total')
#plt.plot(month, acc_reg_monthly15, label='accumulation')
#plt.plot(month, refreeze_reg_monthly15, label='refreeze')
#plt.plot(month, -1*melt_reg_monthly15, label='melt')
#plt.ylabel('monthly regional mean [m.w.e.] / month')
#plt.legend()
#plt.show()
#plt.plot(years[0:16], massbaltotal_reg_annual15, label='massbal_total')
#plt.plot(years[0:16], acc_reg_annual15, label='accumulation')
#plt.plot(years[0:16], refreeze_reg_annual15, label='refreeze')
#plt.plot(years[0:16], -1*melt_reg_annual15, label='melt')
#plt.ylabel('Region 15 annual mean [m.w.e.]', size=15)
#plt.legend()
#plt.show()
#
##lons = np.concatenate((lons14, lons15), axis=0)
##lats = np.concatenate((lats14, lats15), axis=0)
##massbal_total_mwea = np.concatenate((massbal_total_mwea14, massbal_total_mwea15), axis=0)
#lons = lons15
#lats = lats15
#massbal_total_mwea = massbal_total_mwea15
#
## Set extent
#east = int(round(lons.min())) - 1
#west = int(round(lons.max())) + 1
#south = int(round(lats.min())) - 1
#north = int(round(lats.max())) + 1
#xtick = 1
#ytick = 1
## Plot regional maps
#plot_latlonvar(lons, lats, massbal_total_mwea, -1.5, 0.5, 'Modeled mass balance [mwea]', 'longitude [deg]', 
#               'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick)

#%%### ====== PLOTTING FOR CALIBRATION FUNCTION ======================================================================
### Plot histograms and regional variations
##data13 = pd.read_csv(input.main_directory + '/../Output/calibration_R13_20180318_Opt01solutionspaceexpanding.csv')
##data13 = data13.dropna()
##data14 = pd.read_csv(input.main_directory + '/../Output/calibration_R14_20180313_Opt01solutionspaceexpanding.csv')
##data14 = data14.dropna()
##data15 = pd.read_csv(input.main_directory + '/../Output/calibration_R15_20180306_Opt01solutionspaceexpanding.csv')
#data15 = pd.read_csv(input.main_directory + '/../Output/calibration_R15_20180403_Opt02solutionspaceexpanding.csv')
#data15 = data15.dropna()
#data = data15
#
## Concatenate the data
##frames = [data13, data14, data15]
##data = pd.concat(frames)
#
### Fill in values with average 
### Subset all values that have data
##data_subset = data.dropna()
##data_subset_params = data_subset[['lrgcm','lrglac','precfactor','precgrad','ddfsnow','ddfice','tempsnow','tempchange']]
##data_subset_paramsavg = data_subset_params.mean()
##paramsfilled = data[['lrgcm','lrglac','precfactor','precgrad','ddfsnow','ddfice','tempsnow','tempchange']]
##paramsfilled = paramsfilled.fillna(data_subset_paramsavg)    
#
## Set extent
#east = int(round(data['CenLon'].min())) - 1
#west = int(round(data['CenLon'].max())) + 1
#south = int(round(data['CenLat'].min())) - 1
#north = int(round(data['CenLat'].max())) + 1
#xtick = 1
#ytick = 1
## Select relevant data
#lats = data['CenLat'][:]
#lons = data['CenLon'][:]
#precfactor = data['precfactor'][:]
#precgrad = data['precgrad'][:]
#tempchange = data['tempchange'][:]
#ddfsnow = data['ddfsnow'][:]
#calround = data['calround'][:]
#massbal = data['MB_geodetic_mwea']
#massbal_difference = data['MB_difference_mwea']
## Plot regional maps
#plot_latlonvar(lons, lats, massbal, -1.5, 0.5, 'Geodetic mass balance [mwea]', 'longitude [deg]', 'latitude [deg]', 
#               'RdBu', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, massbal_difference, 0, 0.15, 'abs(Mass balance difference) [mwea]', 'longitude [deg]', 
#               'latitude [deg]', 'jet_r', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, precfactor, 0.85, 1.3, 'Precipitation factor [-]', 'longitude [deg]', 'latitude [deg]', 
#               'jet_r', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, precgrad, 0.0001, 0.0002, 'Precipitation gradient [% m-1]', 'longitude [deg]', 'latitude [deg]', 
#               'jet_r', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, tempchange, -2, 2, 'Temperature bias [degC]', 'longitude [deg]', 'latitude [deg]', 
#               'jet', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, ddfsnow, 0.003, 0.005, 'DDF_snow [m w.e. d-1 degC-1]', 'longitude [deg]', 'latitude [deg]', 
#               'jet', east, west, south, north, xtick, ytick)
#plot_latlonvar(lons, lats, calround, 1, 3, 'Calibration round', 'longitude [deg]', 'latitude [deg]', 
#               'jet_r', east, west, south, north, xtick, ytick)
## Plot histograms
#data.hist(column='MB_difference_mwea', bins=50)
#plt.title('Mass Balance Difference [mwea]')
#data.hist(column='precfactor', bins=50)
#plt.title('Precipitation factor [-]')
#data.hist(column='precgrad', bins=50)
#plt.title('Precipitation gradient [% m-1]')
#data.hist(column='tempchange', bins=50)
#plt.title('Temperature bias [degC]')
#data.hist(column='ddfsnow', bins=50)
#plt.title('DDFsnow [mwe d-1 degC-1]')
#plt.xticks(rotation=60)
#data.hist(column='calround', bins = [0.5, 1.5, 2.5, 3.5])
#plt.title('Calibration round')
#plt.xticks([1, 2, 3])
#    
## run plot function
#output.plot_caloutput(data)