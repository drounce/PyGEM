"""Split glaciers into lists to run on separate nodes on the supercomputer"""

# Built-in libraries
import argparse
# External libraries
import pickle
# Local libraries
import pygem_input as input


def getparser():
    """
    Use argparse to add arguments from the command line
    
    Parameters
    ----------
    nnodes : int
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
    list of lists that have n values in each list
    """
    return [lst[i::n] for i in range(n)]
    

parser = getparser()
args = parser.parse_args()

rgi_glac_number_batches = split_list(input.rgi_glac_number, n=args.n_batches)

# Export lists
for n in range(len(rgi_glac_number_batches)):
    print('Batch', n, ':\n', rgi_glac_number_batches[n], '\n')
    batch_fn = 'rgi_glac_number_batch_' + str(n) + '.pkl'
    with open(batch_fn, 'wb') as f:
        pickle.dump(rgi_glac_number_batches[n], f)