import pyDOE as pe
import numpy as np
import copy
import pandas as pd

# This boolean is useful for debugging. If true, a number
# of print statements are activated through the running
# of the model
debug = False

"""
This is a small package that contains functions
to sample from the random probability spaces
of the parameters and mass balance values outputed
by the MCMC algorithm. It contains functions that
perform latin hypercube sampling. This function was
originally used, but gives a larger margin of error
in mass balance values than desired.

The package also contains a function that performs a
stratified random sampling for one variable, which is
currently used by the model. It samples parameter
sets from the space of mass balances they result in,
which produces an ensemble that has the same mean and
standard deviation as the observations.

"""

def lh_sample(tempchange, precfactor, ddfsnow,
              samples=10, criterion=None):
    """
    Performes a latin_hypercube sampling of
    the given random probability distributions.

    Takes the desired random probability distributions
    (each in the form of arrays of numbers) and
    and performs a latin hypercube sampling of these
    distributions using pyDOE and the specificed
    pyDOE algorithm. Returns a pandas DataFrame with
    each row represnting an LH sampling of the parameters
    (column names tempchange, precfactor and ddfsnow)

    Note: given traces must have the same length

    Parameters
    ----------
    tempchange : numpy.array
        The trace (or markov chain) of tempchange
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    ddfsnow : numpy.array
        The trace (or markov chain) of ddfsnow
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    precfactor : numpy.array
        The trace (or markov chain) of precfactor
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    samples : int
        Number of samples to be returned. (default: 10)
    criterion: str
        a string that tells the pyDOE lhs function
        how to sample the points (default: None,
        which simply randomizes the points within
        the intervals):
            “center” or “c”: center the points within
            the sampling intervals
            “maximin” or “m”: maximize the minimum
            distance between points, but place the
            point in a randomized location within its
            interval
            “centermaximin” or “cm”: same as “maximin”,
            but centered within the intervals
            “correlation” or “corr”: minimize the
            maximum correlation coefficient


    Returns
    -------
    pandas.dataframe
        DataFrame with each row representing a sampling.
        The column each represent a parameter of interest
        (tempchange, precfactor, ddfsnow)

    """

    # copy the given traces
    tempchange_copy = copy.deepcopy(tempchange)
    precfactor_copy = copy.deepcopy(precfactor)
    ddfsnow_copy = copy.deepcopy(ddfsnow)

    if debug:
        print('copy of tempchange:', tempchange_copy)
        print('copy of precfactor:', precfactor_copy)
        print('copy of ddfsnow:', ddfsnow_copy)

    traces = [tempchange_copy, precfactor_copy,
              ddfsnow_copy]

    # sort the traces
    for trace in traces:
        trace.sort()


    if debug:
        print('sorted copy of tempchange:', tempchange_copy)
        print('sorted copy of precfactor:', precfactor_copy)
        print('sorted copy of ddfsnow:', ddfsnow_copy)

    lhd = pe.lhs(n=3, samples=samples, criterion=criterion)


    if debug:
        print('copy of lhs samples:', lhd)

    lhd = np.ndarray.transpose(lhd)

    if debug:
        print('copy of lhs samples transposed:', lhd)

    for i in range(len(traces)):
        lhd[i] *= len(traces[i])


    lhd = lhd.astype(int)

    if debug:
        print('copy of lhs samples ints:', lhd)

    names = ['tempchange', 'precfactor', 'ddfsnow']
    samples = pd.DataFrame()

    for i in range(len(traces)):
        array = []
        for j in range(len(lhd[i])):
            array.append(traces[i][lhd[i][j]])
        samples[names[i]] = array

    if debug:
        print(samples, type(samples))

    return samples

# in its current form this function is broken,
# but it simply needs access to the mass
# balance script to calculate mass balance
# given certain parameters.
# TODO: fix this
def find_mass_balace(samples, replace=True):
    """
    Calculated mass balances for given set of
    parameters

    Takes a pandas dataframe of sample sets and
    adds a column for the mass balances (average
    annual change in S.W.E. over the
    time period of David Shean's observations)
    Takes a dataframe of parameter sets generated
    by a latin hypercube sampling of random
    probability distributions) and returns a new
    dataframe (can replace the old one if desired)
    with the mass balances included.

    Parameters
    ----------
    samples : pandas.dataframe
        A dataframe where each row represents sets of
        parameters for the model.Columns are 'tempchange',
        'precfactor' and 'ddfsnow'
    replace : boolean
        True if dataframe should be unchanged and a new
        dataframe should be returned, if False the given
        dataframe will be modified and a copy will not be
        created. default True

    Returns
    -------
    pandas.dataframe
         A dataframe where each row represents sets of
        parameters for the model and the mass balance
        that results from using these parameters.
        Columns are 'tempchange','precfactor', 'ddfsnow'
        and 'massbalance'

    """

    # make copy if replace is True
    if replace:
        samples = copy.deepcopy(samples)

    # calculate massbalances row by row
    massbalances = []
    for index, row in samples.iterrows():
        if debug:
            print(row['precfactor'], row['tempchange'], row['ddfsnow'])

        # caluclate mass balance for each set of params
        massbalance = v2.get_mass_balance(precfactor=row['precfactor'],
                                          tempchange=row['tempchange'],
                                          ddfsnow=row['ddfsnow'])

        if debug:
            print('massbalance', massbalance)

        # add to dataframe
        massbalances.append(massbalance)

    # add to dataframe
    samples['massbal'] = massbalances

    return samples

def stratified_sample(tempchange, ddfsnow, precfactor, massbal,
            samples=10, criterion=None):
    """
    Performes a stratified sampling of
    the given random probability distribution.

    Takes the output of the MCMC sampling in the
    form of the traces of each of the three variables
    we look at (tempchange, ddfsnow, precfactor) and
    the trace of the massbalance, and performs a
    stratified sampling of these traces using only the
    distribution of mass balances. The space of mass
    balance values is divided into chunks of equal
    probability (based on the numbr of samples) and
    from each chunk a value is chosen based on the
    chosen criterion (default: random). This way,
    samples which have already been outputed by the
    MCMC algorithm are chosen from in a manner that
    reflects their probability distribution more accurately
    than a purely random sampling. One can think of this
    sampling method as the equivalent of a latin hypercube
    sampling in one dimension.

    This function uses
    pyDOE and the specificed pyDOE algorithm. Returns
    a pandas dataframe with each row representing one
    sampling, ie a single set of parameters
    and their respective mass balance.

    Note: given traces must have the same length

    Parameters
    ----------
    tempchange : numpy.array
        The trace (or markov chain) of tempchange
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    ddfsnow : numpy.array
        The trace (or markov chain) of ddfsnow
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    precfactor : numpy.array
        The trace (or markov chain) of precfactor
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    massbal : numpy.array
        The trace (or markov chain) of mass balance
        outputed by the MCMC sampling. Each trace
        is represented by an array of discrete
        values or points
    samples : int
        Number of samples to be returned. (default: 10)
    criterion: str
        a string that tells the pyDOE lhs function
        how to sample the points (default: None,
        which simply randomizes the points within
        the intervals):
            “center” or “c”: center the points within
            the sampling intervals
            “maximin” or “m”: maximize the minimum
            distance between points, but place the
            point in a randomized location within its
            interval
            “centermaximin” or “cm”: same as “maximin”,
            but centered within the intervals
            “correlation” or “corr”: minimize the
            maximum correlation coefficient


    Returns
    -------
    pandas.DataFrame
        Dataframe with each row representing a sampling, i.e.
        each row is a set of parameters and their respective
        mass balance. Includes a column for each parameter and
        for massbal

    """

    # make dataframe out of given traces
    df = pd.DataFrame({'tempchange': tempchange,
                       'precfactor': precfactor,
                       'ddfsnow': ddfsnow, 'massbal': massbal})

    # sort dataframe based on values of massbal and add
    # extra indices based on sorted order
    sort_df = df.sort_values('massbal')
    sort_df['sorted_index'] = np.arange(len(sort_df))

    if debug:
        print('sorted_df\n', sort_df)

    # use pyDOE, get lhs sampling for 1 distribution
    lhd = pe.lhs(n=1, samples=samples, criterion=criterion)

    # convert lhs to indices for dataframe
    lhd = (lhd * len(df)).astype(int)
    lhd = lhd.ravel()

    if debug:
        print('lhd\n', lhd)

    # take sampling with lhs indices, re-sort according to
    # original trace indices
    subset = sort_df.iloc[lhd]
    subset = subset.sort_index()

    if debug:
        print('subset\n', subset)

    return subset
