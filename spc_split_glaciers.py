"""Split glaciers into lists to run on separate nodes on the supercomputer"""

# Built-in libraries
import argparse
import os
# External libraries
import numpy as np
import pickle
# Local libraries
import pygem_input as input
import pygemfxns_modelsetup as modelsetup


def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    n_batches (optional) : int
        number of nodes being used on the supercomputer
    spc_region (optional) : str
        RGI region number for supercomputer
    ignore_regionname (optional) : int
        switch to ignore region name or not (1 ignore it, 0 use region)
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-n_batches', action='store', type=int, default=1,
                        help='number of nodes to split the glaciers amongst')
    parser.add_argument('-spc_region', action='store', type=int, default=None,
                        help='rgi region number for supercomputer')
    parser.add_argument('-ignore_regionname', action='store', type=int, default=0,
                        help='switch to include the region name or not in the batch filenames')
    parser.add_argument('-add_cal', action='store', type=int, default=0,
                        help='switch to add "cal" to batch filenames')
    parser.add_argument('-load_glacno')
    parser.add_argument('-glacno_fn', action='store', type=str, default=None,
                        help='load specific glacier numbers from file name')
    return parser


def split_list(lst, n=1):
    """
    Split list into batches for the supercomputer.
    
    Parameters
    ----------
    lst : list
        List that you want to split into separate batches
    n : int
        Number of batches to split glaciers into.
    
    Returns
    -------
    lst_batches : list
        list of n lists that have sequential values in each list
    """
    # If batches is more than list, then there will be one glacier in each batch
    if n > len(lst):
        n = len(lst)
    n_perlist_low = int(len(lst)/n)
    n_perlist_high = int(np.ceil(len(lst)/n))
    lst_copy = lst.copy()
    count = 0
    lst_batches = []
    for x in np.arange(n):
        count += 1
        if count <= len(lst) % n:
            lst_subset = lst_copy[0:n_perlist_high]
            lst_batches.append(lst_subset)
            [lst_copy.remove(i) for i in lst_subset]
        else:
            lst_subset = lst_copy[0:n_perlist_low]
            lst_batches.append(lst_subset)
            [lst_copy.remove(i) for i in lst_subset]
    return lst_batches    
 

parser = getparser()
args = parser.parse_args()

# RGI region
if args.spc_region is not None:
    rgi_regionsO1 = [int(args.spc_region)]
else:
    rgi_regionsO1 = input.rgi_regionsO1
    
# Check if the batch files need to be updated
batch_list = []
count_glac = 0
batch_str = 'rgi_glac_number_batch_'
for i in os.listdir():

    # Check string
    if args.ignore_regionname == 0:
        check_str = 'R' + str(rgi_regionsO1[0]) + '_' + batch_str
    elif args.ignore_regionname == 1:
        check_str = batch_str
        
    if args.add_cal == 1:
        check_str = 'Cal_' + check_str
    
    # List batch fns and count total number of glaciers
    if i.startswith(check_str) and i.endswith('.pkl'):
        with open(i, 'rb') as f:
            rgi_glac_number = pickle.load(f)
            batch_list.append(i)
        
        count_glac += len(rgi_glac_number)
    

# Load glacier numbers
if args.glacno_fn is not None:
    with open(input.modelparams_fp_dict[rgi_regionsO1[0]] + args.glacno_fn, 'rb') as f:
        rgi_glac_number = pickle.load(f)
elif input.rgi_glac_number == 'all':
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=rgi_regionsO1, rgi_regionsO2='all',
                                                          rgi_glac_number='all')
    # Create list of glacier numbers as strings with 5 digits
    glacno = main_glac_rgi_all.glacno.values
    rgi_glac_number = [str(x).zfill(5) for x in glacno]    
else:
    rgi_glac_number = input.rgi_glac_number
    

# Check if need to update old files or not
#  (different number of glaciers or batches)
if count_glac != len(rgi_glac_number) or args.n_batches != len(batch_list):
    # Delete old files
    for i in batch_list:
        os.remove(i)
    
    # Split list of glacier numbers
    rgi_glac_number_batches = split_list(rgi_glac_number, n=args.n_batches)

    # Export new lists
    for n in range(len(rgi_glac_number_batches)):
    #    print('Batch', n, ':\n', rgi_glac_number_batches[n], '\n')
        if args.ignore_regionname == 0:
            batch_fn = 'R' + str(rgi_regionsO1[0]) + '_' + batch_str + str(n) + '.pkl'
        elif args.ignore_regionname == 1:
            batch_fn = batch_str + str(n) + '.pkl'
        
        if args.add_cal == 1:
            batch_fn = 'Cal_' + batch_fn
            
    #    print('Batch', n, ':\n', batch_fn, '\n')
        with open(batch_fn, 'wb') as f:
            pickle.dump(rgi_glac_number_batches[n], f)
            
