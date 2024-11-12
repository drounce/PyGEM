""" derive binned monthly ice thickness and mass from PyGEM simulation """

# Built-in libraries
import argparse
import collections
import copy
import inspect
import multiprocessing
import os
import glob
import sys
import time
# External libraries
import numpy as np
import xarray as xr
# pygem imports
import pygem.setup.config as config
# read config
pygem_prms = config.read_config()

# ----- FUNCTIONS -----
def getparser():
    """
    Use argparse to add arguments from the command line
    """
    parser = argparse.ArgumentParser(description="process monthly ice thickness for PyGEM simulation")
    # add arguments
    parser.add_argument('-simpath', action='store', type=str, nargs='+',
                        help='path to PyGEM binned simulation (can take multiple)')
    parser.add_argument('-binned_simdir', action='store', type=str, default=None,
                        help='directory with binned simulations for which to process monthly thickness')
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')

    return parser


def get_binned_monthly(bin_massbalclim_monthly, bin_massbalclim_annual, bin_mass_annual, bin_thick_annual):
    """
    funciton to calculate the monthly binned ice thickness and mass
    from annual climatic mass balance and annual ice thickness products

    to determine monthlyt thickness and mass, we must account for flux divergence
    this is not so straight-forward, as PyGEM accounts for ice dynamics at the 
    end of each model year and not on a monthly timestep.
    here, monthly thickness and mass is determined assuming 
    the flux divergence is constant throughout the year.
    
    annual flux divergence is first estimated by combining the annual binned change in ice 
    thickness and the annual binned mass balance. then, assume flux divergence is constant 
    throughout the year (divide annual by 12 to get monthly flux divergence).

    monthly binned flux divergence can then be combined with 
    monthly binned climatic mass balance to get monthly binned change in ice thickness

    
    Parameters
    ----------
    bin_massbalclim_monthly : float
        ndarray containing the climatic mass balance for each model month computed by PyGEM
        shape : [#glac, #elevbins, #months]
    bin_massbalclim_annual : float
        ndarray containing the climatic mass balance for each model year computed by PyGEM
        shape : [#glac, #elevbins, #years]
    bin_mass_annual : float
        ndarray containing the average (or median) binned ice mass computed by PyGEM
        shape : [#glac, #elevbins, #years]
    bin_thick_annual : float
        ndarray containing the average (or median) binned ice thickness at computed by PyGEM
        shape : [#glac, #elevbins, #years]

    Returns
    -------
    bin_thick_monthly: float
        ndarray containing the binned monthly ice thickness
        shape : [#glac, #elevbins, #years]

    bin_mass_monthly: float
        ndarray containing the binned monthly ice mass
        shape : [#glac, #elevbins, #years]
    """

    # get change in thickness from previous year for each elevation bin
    delta_thick_annual = np.diff(bin_thick_annual, axis=-1)

    # get annual binned flux divergence as annual binned climatic mass balance (-) annual binned ice thickness
    # account for density contrast (convert climatic mass balance in m w.e. to m ice)
    flux_div_annual = (
            (bin_massbalclim_annual[:,:,1:] * 
            pygem_prms['constants']['density_ice'] / 
            pygem_prms['constants']['density_water']) - 
            delta_thick_annual)

    ### to get monthly thickness and mass we need monthly flux divergence ###
    # we'll assume the flux divergence is constant througohut the year (is this a good assumption?)
    # ie. take annual values and divide by 12 - use numpy repeat to repeat values across 12 months
    flux_div_monthly = np.repeat(flux_div_annual / 12, 12, axis=-1)

    # get monthly binned change in thickness assuming constant flux divergence throughout the year
    # account for density contrast (convert monthly climatic mass balance in m w.e. to m ice)
    bin_thickchange_monthly = (
            (bin_massbalclim_monthly *
            pygem_prms['constants']['density_ice'] /
            pygem_prms['constants']['density_water']) -
            flux_div_monthly)
    
    # get binned monthly thickness = running thickness change + initial thickness
    running_delta_thick_monthly = np.cumsum(bin_thickchange_monthly, axis=-1)
    bin_thick_monthly =  running_delta_thick_monthly + bin_thick_annual[:,:,0][:,:,np.newaxis] 

    ### get monthly mass ###
    # note, this requires knowledge of binned glacier area
    # we do not have monthly binned area (as glacier dynamics are performed on an annual timestep in PyGEM),
    # so we'll resort to using the annual binned glacier mass and thickness in order to get to binned glacier area
    ########################
    # first convert bin_mass_annual to bin_voluma_annual
    bin_volume_annual = bin_mass_annual / pygem_prms['constants']['density_ice']
    # now get area: use numpy divide where denominator is greater than 0 to avoid divide error
    # note, indexing of [:,:,1:] so that annual area array has same shape as flux_div_annual
    bin_area_annual = np.divide(
            bin_volume_annual[:,:,1:], 
            bin_thick_annual[:,:,1:], 
            out=np.full(bin_thick_annual[:,:,1:].shape, np.nan), 
            where=bin_thick_annual[:,:,1:]>0)

    # tile to get monthly area, assuming area is constant thoughout the year
    bin_area_monthly = np.tile(bin_area_annual, 12)

    # combine monthly thickess and area to get mass
    bin_mass_monthly = bin_thick_monthly * bin_area_monthly * pygem_prms['constants']['density_ice']

    return bin_thick_monthly, bin_mass_monthly


def update_xrdataset(input_ds, bin_thick_monthly, bin_mass_monthly):
    """
    update xarray dataset to add new fields

    Parameters
    ----------
    xrdataset : xarray Dataset
        existing xarray dataset
    newdata : ndarray 
        new data array
    description: str
        describing new data field

    output_ds : xarray Dataset
        empty xarray dataset that contains variables and attributes to be filled in by simulation runs
    encoding : dictionary
        encoding used with exporting xarray dataset to netcdf
    """
    # coordinates
    glac_values = input_ds.glac.values
    time_values = input_ds.time.values
    bin_values = input_ds.bin.values

    output_coords_dict = collections.OrderedDict()
    output_coords_dict['bin_thick_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))
    output_coords_dict['bin_mass_monthly'] = (
            collections.OrderedDict([('glac', glac_values), ('bin',bin_values), ('time', time_values)]))

    # Attributes dictionary
    output_attrs_dict = {}
    output_attrs_dict['bin_thick_monthly'] = {
            'long_name': 'binned monthly ice thickness',
            'units': 'm',
            'temporal_resolution': 'monthly',
            'comment': 'monthly ice thickness binned by surface elevation'}
    output_attrs_dict['bin_mass_monthly'] = {
            'long_name': 'binned monthly ice mass',
            'units': 'kg',
            'temporal_resolution': 'monthly',
            'comment': 'monthly ice mass binned by surface elevation'}


    # Add variables to empty dataset and merge together
    count_vn = 0
    encoding = {}
    for vn in output_coords_dict.keys():
        empty_holder = np.zeros([len(output_coords_dict[vn][i]) for i in list(output_coords_dict[vn].keys())])
        output_ds = xr.Dataset({vn: (list(output_coords_dict[vn].keys()), empty_holder)},
                               coords=output_coords_dict[vn])
        count_vn += 1
        # Merge datasets of stats into one output
        if count_vn == 1:
            output_ds_all = output_ds
        else:
            output_ds_all = xr.merge((output_ds_all, output_ds))
    # Add attributes
    for vn in output_ds_all.variables:
        try:
            output_ds_all[vn].attrs = output_attrs_dict[vn]
        except:
            pass
        # Encoding (specify _FillValue, offsets, etc.)
        encoding[vn] = {'_FillValue': None,
                        'zlib':True,
                        'complevel':9
                        }    

    output_ds_all['bin_thick_monthly'].values = (
            bin_thick_monthly
            )
    output_ds_all['bin_mass_monthly'].values = (
            bin_mass_monthly
            )

    return output_ds_all, encoding


def run(simpath):
    """
    create binned monthly mass change data product
    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels
    Returns
    -------
    binned_ds : netcdf Dataset
        updated binned netcdf containing binned monthly ice thickness and mass
    """

    if os.path.isfile(simpath):
        # open dataset
        binned_ds = xr.open_dataset(simpath)

        # calculate monthly change in mass
        bin_thick_monthly, bin_mass_monthly = get_binned_monthly(
                                                    binned_ds.bin_massbalclim_monthly.values, 
                                                    binned_ds.bin_massbalclim_annual.values, 
                                                    binned_ds.bin_mass_annual.values,
                                                    binned_ds.bin_thick_annual.values
                                                    )

        # update dataset to add monthly mass change
        output_ds_binned, encoding_binned = update_xrdataset(binned_ds, bin_thick_monthly, bin_mass_monthly)

        # close input ds before write
        binned_ds.close()

        # append to existing binned netcdf
        output_ds_binned.to_netcdf(simpath, mode='a', encoding=encoding_binned, engine='netcdf4')
        print(output_ds_binned)
        # close datasets
        output_ds_binned.close()

    return


def main():
    time_start = time.time()
    args = getparser().parse_args()

    simpath = None
    if args.binned_simdir:
        # get list of sims
        simpath = glob.glob(args.binned_simdir+'*.nc')
    else:
        if args.simpath:
            simpath = args.simpath
    print(simpath,os.path.isfile(simpath[0]))
    if simpath:
        # number of cores for parallel processing
        if args.ncores > 1:
            ncores = int(np.min([len(simpath), args.ncores]))
        else:
            ncores = 1

        # Parallel processing
        print('Processing with ' + str(args.ncores) + ' cores...')
        with multiprocessing.Pool(args.ncores) as p:
            p.map(run,simpath)

    print('Total processing time:', time.time()-time_start, 's')
    
if __name__ == "__main__":
    main()