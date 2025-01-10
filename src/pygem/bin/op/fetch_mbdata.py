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
# oggm
from oggm import utils
# pygem imports
import pygem.setup.config as config
# check for config
config.ensure_config()
# read the config
pygem_prms = config.read_config()
import pygem.pygem_modelsetup as modelsetup


# hugonnet filepath
hugonnet_fp = f"{pygem_prms['root']}/DEMs/Hugonnet2020/"


def run(fn='', debug=False, overwrite=False):
    """
    pull geodetic mass balance data from OGGM
    The original 'raw' were acquired and combined from https://doi.org/10.6096/13 (time series/dh_<RGI_REGION01>_rgi60_pergla_rates)
    The combined global data have been modified in three ways (code):
    1. the glaciers in RGI region 12 (Caucasus) had to be manually linked to the product by Hugonnet because of large errors in the RGI outlines. The resulting product used by OGGM in region 12 has large uncertainties.
    2. outliers have been filtered as following: all glaciers with an error estimate larger than 3 at the RGI region level are filtered out
    3. all missing data (including outliers) are attributed with the regional average.

    See https://docs.oggm.org/en/latest/reference-mass-balance-data.html and https://nbviewer.org/urls/cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/convert.ipynb for further information.

    dmdtda represents the average climatic mass balance (in units meters water-equivalent per year) over a given period \\frac{\\partial mass}{\\partial time \\partial area}
    """
    mbdf = utils.get_geodetic_mb_dataframe()
    if debug:
        print('MB data loaded from OGGM:')
        print(mbdf.head())

    # pull only 2000-2020 period
    mbdf_subset = mbdf[mbdf.period=='2010-01-01_2020-01-01']

    # reset the index and rename the resulting column to 'rgiid'
    mbdf_subset = mbdf_subset.reset_index().rename(columns={'index': 'rgiid'})

    # sort by the 'rgiid' column
    mbdf_subset = mbdf_subset.sort_values(by='rgiid')

    if len(fn.split('.')) == 1:
        fn+='.csv'
    if os.path.isfile(hugonnet_fp+fn) and not overwrite:
        raise FileExistsError(f'The filled global geodetic mass balance file already exists, pass `-o` to overwrite: {hugonnet_fp+fn}')
    
    mbdf_subset.to_csv(hugonnet_fp+fn, index=False)
    if debug:
        print(f'Filled global geodetic mass balance data saved to: {hugonnet_fp+fn}')


def main():
    parser = argparse.ArgumentParser(description="grab filled Hugonnet et al. 2021 geodetic mass balance data from OGGM and converts to a format PyGEM utilizes")
    # add arguments
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-fname', type=str, required=True,
                        help='Output file name')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Flag to overwrite existing geodetic mass balance data')
    parser.add_argument('-v', '--debug', action='store_true',
                        help='Flag for debugging')
    args = parser.parse_args()

    run(args.fname, args.debug, args.overwrite)


if __name__ == "__main__":
    main()