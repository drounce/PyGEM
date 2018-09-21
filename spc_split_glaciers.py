"""List of model inputs required to run PyGEM"""

# External libraries
import numpy as np
import pickle
# Local libraries
import pygem_input as input


def split_list(lst, n_perlist=24):
    """
    Split list of glaciers into batches for the supercomputer.
    
    Parameters
    ----------
    lst : list
        List that you want to split into separate batches
    n_perlist : int
        Number of values you want in each batch.
    
    Returns
    -------
    list of lists that have n values in each list
    """
    n_batches = int(np.ceil(len(lst) / n_perlist))
    return [lst[i::n_batches] for i in range(n_batches)]
    

rgi_glac_number_batches = split_list(input.rgi_glac_number)

# Export lists
for n in range(len(rgi_glac_number_batches)):
    print('Batch', n, ':\n', rgi_glac_number_batches[n], '\n')
    batch_fn = 'rgi_glac_number_batch_' + str(n) + '.pkl'
    with open(batch_fn, 'wb') as f:
        pickle.dump(rgi_glac_number_batches[n], f)