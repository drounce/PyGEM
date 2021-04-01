"""Split glaciers into lists to run on separate nodes on the supercomputer"""

# Built-in libraries
import argparse
import os
# External libraries
import numpy as np
import pickle
# Local libraries
import pygem.pygem_input as pygem_prms
import pygem.pygem_modelsetup as modelsetup


def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    n_batches (optional) : int
        number of nodes being used on the supercomputer
    ignore_regionname (optional) : int
        switch to ignore region name or not (1 ignore it, 0 use region)
    add_cal : int
        switch to add "Cal" to the batch filenames such that calibration and simulation can be run at same time
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by 
         regional variations)
        
    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-n_batches', action='store', type=int, default=1,
                        help='number of nodes to split the glaciers amongst')
    parser.add_argument('-ignore_regionname', action='store', type=int, default=0,
                        help='switch to include the region name or not in the batch filenames')
    parser.add_argument('-add_cal', action='store', type=int, default=0,
                        help='switch to add "cal" to batch filenames')
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-startno', action='store', type=int, default=None,
                        help='starting number of rgi glaciers')
    parser.add_argument('-endno', action='store', type=int, default=None,
                        help='starting number of rgi glaciers')
    return parser


def glac_num_fromrange(int_low, int_high):
    """
    Generate list of glaciers for all numbers between two integers.

    Parameters
    ----------
    int_low : int64
        low value of range
    int_high : int64
        high value of range

    Returns
    -------
    y : list
        list of rgi glacier numbers
    """
    x = (np.arange(int_low, int_high+1)).tolist()
    y = [str(i).zfill(5) for i in x]
    return y


def split_list(lst, n=1, option_ordered=1):
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
    if option_ordered == 1:
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
    
    else:
        if n > len(lst):
            n = len(lst)
    
        lst_batches = [[] for x in np.arange(n)]
        nbatch = 0
        for count, x in enumerate(lst):
            if count%n == 0:
                nbatch = 0
    
            lst_batches[nbatch].append(x)
            
            nbatch += 1
            
    return lst_batches    
 

if __name__ == '__main__':
    parser = getparser()
    args = parser.parse_args()   
        
    # Count glaciers in existing batch
    batch_list = []
    count_glac = 0
    batch_str = 'rgi_glac_number_'
    # region string
    regions_str = 'R'
    for region in pygem_prms.rgi_regionsO1:
        regions_str += str(region)
    # check files
    for i in os.listdir():
            
        if args.ignore_regionname == 0:
            check_str = regions_str + '_' + batch_str
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
    
    # Select all glaciers
    if not args.startno is None and not args.endno is None:
        rgi_glac_number = glac_num_fromrange(int(args.startno), int(args.endno))
        glac_no = None
    else:
        rgi_glac_number = pygem_prms.rgi_glac_number
        glac_no = pygem_prms.glac_no
    
    main_glac_rgi_all = modelsetup.selectglaciersrgitable(
            rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2 =pygem_prms.rgi_regionsO2, 
            rgi_glac_number=rgi_glac_number, glac_no=glac_no)
    glacno_str = [x.split('-')[1] for x in main_glac_rgi_all.RGIId.values]
    

    # Split list of glacier numbers
    rgi_glac_number_batches = split_list(glacno_str, n=args.n_batches, option_ordered=args.option_ordered)

    # Export new lists
    for n in range(len(rgi_glac_number_batches)):
    #    print('Batch', n, ':\n', rgi_glac_number_batches[n], '\n')
        if args.ignore_regionname == 0:
            batch_fn = regions_str + '_' + batch_str
        elif args.ignore_regionname == 1:
            batch_fn = batch_str
            
        # Add start and end numbers
        if not args.startno is None and not args.endno is None:
            batch_fn += str(args.startno) + '-' + str(args.endno) + 'glac_'
            
        # add batch number and .pkl
        batch_fn += 'batch_' + str(n) + '.pkl'
        
        if args.add_cal == 1:
            batch_fn = 'Cal_' + batch_fn
            
        print('Batch', n, ':\n', batch_fn, '\n')
        with open(batch_fn, 'wb') as f:
            pickle.dump(rgi_glac_number_batches[n], f)