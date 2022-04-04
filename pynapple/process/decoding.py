# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 23:34:48
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-04 17:39:59

"""
"""

import numpy as np
from .. import core as nap


def decode_1d(tuning_curves, group, ep, bin_size, time_units = 's', feature=None):
    """
    Performs Bayesian decoding over a one dimensional feature.
    See: 
    Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J. 
    (1998). Interpreting neuronal population activity by 
    reconstruction: unified framework with application to 
    hippocampal place cells. Journal of neurophysiology, 79(2), 
    1017-1044.
    
    Parameters
    ----------
    tuning_curves : pandas.DataFrame
        Each column is the tuning curve of one neuron relative to the feature. 
        Index should be the center of the bin.
    group : TsGroup or dict of Ts/Tsd object.
        A group of neurons with the same index as tuning curves column names.
    ep : IntervalSet
        The epoch on which decoding is computed
    bin_size : float
        Bin size. Default is second. Use the parameter time_units to change it.
    time_units : str, optional
        Time unit of the bin size ('s' [default], 'ms', 'us').
    feature : Tsd, optional
        The 1d feature used to compute the tuning curves. Used to correct for occupancy.
        If feature is not passed, the occupancy is uniform.
    
    Returns
    -------
    Tsd
        The decoded feature
    TsdFrame
        The probability distribution of the decoded feature for each time bin
    
    Raises
    ------
    RuntimeError
        If group is not a dict of Ts/Tsd or TsGroup.
        If different size of neurons for tuning_curves and group.
        If indexes don't match between tuning_curves and group.    
    """
    if isinstance(group, dict):
        newgroup = nap.TsGroup(group, time_support = ep)
    elif isinstance(group, nap.TsGroup):
        newgroup = group.restrict(ep)
    else:
        raise RuntimeError("Unknown format for group")

    if tuning_curves.shape[1] != len(newgroup):
        raise RuntimeError("Different shapes for tuning_curves and group")

    if not np.all(tuning_curves.columns.values == np.array(newgroup.keys())):
        raise RuntimeError("Difference indexes for tuning curves and group keys")   

    # Bin spikes
    count = newgroup.count(bin_size, ep, time_units)

    # Occupancy
    if feature is None:
        occupancy = np.ones(tuning_curves.shape[0])
    else:
        diff = np.diff(tuning_curves.index.values)
        bins = tuning_curves.index.values[:-1] - diff/2
        bins = np.hstack((bins, [bins[-1]+diff[-1],bins[-1]+2*diff[-1]])) # assuming the size of the last 2 bins is equal
        occupancy,_ = np.histogram(feature, bins)

    # Transforming to pure numpy array
    tc = tuning_curves.values
    ct = count.values

    bin_size_s = nap.TimeUnits.format_timestamps(np.array([bin_size]), time_units)[0]

    p1 = np.exp(-bin_size_s*tc.sum(1))    
    p2 = occupancy/occupancy.sum()

    ct2 = np.tile(ct[:,np.newaxis,:], (1,tc.shape[0],1))

    p3 = np.prod(tc**ct2, -1)

    p = p1 * p2 * p3
    p = p / p.sum(1)[:,np.newaxis]

    idxmax = np.argmax(p, 1)

    p = nap.TsdFrame(
        t=count.index.values,
        d=p,
        time_support=ep,
        columns=tuning_curves.index.values)

    decoded = nap.Tsd(
        t=count.index.values,
        d=tuning_curves.index.values[idxmax],
        time_support=ep)

    return decoded, p

def decode_2d(tuning_curves, group, ep, bin_size, xy, time_units='s', features=None):
    """
    Performs Bayesian decoding over a two dimensional feature.
    See: 
    Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J. 
    (1998). Interpreting neuronal population activity by 
    reconstruction: unified framework with application to 
    hippocampal place cells. Journal of neurophysiology, 79(2), 
    1017-1044.
    
    Parameters
    ----------
    tuning_curves : dict
        Dictionnay of 2d tuning curves (one for each neuron).
    group : TsGroup or dict of Ts/Tsd object.
        A group of neurons with the same keys as tuning_curves dictionnary.
    ep : IntervalSet
        The epoch on which decoding is computed
    bin_size : float
        Bin size. Default is second. Use the parameter time_units to change it.
    xy : tuple
        A tuple of bin positions for the tuning curves i.e. xy=(x,y)
    time_units : str, optional
        Time unit of the bin size ('s' [default], 'ms', 'us').        
    features : TsdFrame
        The 2 columns features used to compute the tuning curves. Used to correct for occupancy.
        If feature is not passed, the occupancy is uniform.
    
    Returns
    -------
    Tsd
        The decoded feature in 2d
    numpy.ndarray
        The probability distribution of the decoded trajectory for each time bin
    
    Raises
    ------
    RuntimeError
        If group is not a dict of Ts/Tsd or TsGroup.
        If different size of neurons for tuning_curves and group.
        If indexes don't match between tuning_curves and group.  
    
    """
    
    if type(group) is dict:
        newgroup = nap.TsGroup(group, time_support = ep)
        numcells = len(newgroup)
    elif type(group) is nap.TsGroup:
        newgroup = group.restrict(ep)
        numcells = len(newgroup)
    else:
        raise RuntimeError("Unknown format for group")

    if len(tuning_curves) != numcells:
        raise RuntimeError("Different shapes for tuning_curves and group")

    if not np.all(np.array(list(tuning_curves.keys())) == np.array(newgroup.keys())):
        raise RuntimeError("Difference indexes for tuning curves and group keys")   

    # Bin spikes
    # if type(newgroup) is not nap.TsdFrame:
    count = newgroup.count(bin_size, ep, time_units)
    # else:
    #     #Spikes already "binned" with continuous TsdFrame input
    #     count = newgroup

    indexes = list(tuning_curves.keys())

    # Occupancy
    if features is None:
        occupancy = np.ones_like(tuning_curves[indexes[0]]).flatten()
    else:
        binsxy = []
        for i in range(len(xy)):
            diff = np.diff(xy[i])
            bins = xy[i][:-1] - diff/2
            bins = np.hstack((bins, [bins[-1]+diff[-1],bins[-1]+2*diff[-1]])) # assuming the size of the last 2 bins is equal
            binsxy.append(bins)
    
        occupancy, _, _ = np.histogram2d(
            features.iloc[:,0],
            features.iloc[:,1],
            [binsxy[0], binsxy[1]])
        occupancy = occupancy.flatten()

    # Transforming to pure numpy array
    tc = np.array([tuning_curves[i] for i in tuning_curves.keys()])
    tc = tc.reshape(tc.shape[0], np.prod(tc.shape[1:]))
    tc = tc.T

    ct = count.values

    bin_size_s = nap.TimeUnits.format_timestamps(np.array([bin_size]), time_units)[0]

    p1 = np.exp(-bin_size_s*np.nansum(tc, 1))
    p2 = occupancy/occupancy.sum()

    ct2 = np.tile(ct[:,np.newaxis,:], (1,tc.shape[0],1))

    p3 = np.nanprod(tc**ct2, -1)

    p = p1 * p2 * p3
    p = p / p.sum(1)[:,np.newaxis]

    idxmax = np.argmax(p, 1)

    p = p.reshape(p.shape[0], len(xy[0]), len(xy[1]))

    idxmax2d = np.unravel_index(idxmax, (len(xy[0]), len(xy[1])))

    if features is not None:
        cols = features.columns
    else:
        cols = np.arange(2)

    decoded = nap.TsdFrame(
        t = count.index.values,
        d = np.vstack((xy[0][idxmax2d[0]], xy[1][idxmax2d[1]])).T,
        time_support = ep,
        columns=cols
        )

    return decoded, p