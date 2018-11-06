"""Split glaciers into lists to run on separate nodes on the supercomputer"""

# Built-in libraries
import argparse
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
    n_batches : int
        number of nodes being used on the supercomputer
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-n_batches', action='store', type=int, default=1,
                        help='number of nodes to split the glaciers amongst')
    return parser


def split_list(lst, n=1):
    """
    Split list of glaciers into batches for the supercomputer.
    
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
        if count <= n_perlist_low:
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
if input.rgi_glac_number == 'all':
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(rgi_regionsO1=input.rgi_regionsO1, rgi_regionsO2='all',
                                                          rgi_glac_number='all')
    # Create list of glacier numbers as strings with 5 digits
    glacno = main_glac_rgi_all.glacno.values
    rgi_glac_number = [str(x).zfill(5) for x in glacno]    
else:
    rgi_glac_number = input.rgi_glac_number
    
rgi_glac_number_batches = split_list(rgi_glac_number, n=args.n_batches)

# Export lists
for n in range(len(rgi_glac_number_batches)):
#    print('Batch', n, ':\n', rgi_glac_number_batches[n], '\n')
    batch_fn = 'rgi_glac_number_batch_' + str(n) + '.pkl'
    print('Batch', n, ':\n', batch_fn, '\n')
    with open(batch_fn, 'wb') as f:
        pickle.dump(rgi_glac_number_batches[n], f)
