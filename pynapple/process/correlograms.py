# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 11:39:55
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-01 16:03:42


import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
from scipy import signal
from itertools import combinations
from .. import core as nap


#########################################################
# CORRELATION
#########################################################
@jit(nopython=True)
def cross_correlogram(t1, t2, binsize, windowsize):
    """
    Performs the discrete cross-correlogram of two time series. 
    The units should be in s for all arguments.
    Return the firing rate of the series t2 relative to the timings of t1.
    See compute_crosscorrelogram, compute_autocorrelogram and compute_eventcorrelogram
    for wrappers of this function.
    
    Parameters
    ----------
    t1 : numpy.ndarray
        The timestamps of the reference time series (in seconds)
    t2 : numpy.ndarray
        The timestamps of the target time series (in seconds)
    binsize : float
        The bin size (in seconds)
    windowsize : float
        The window size (in seconds)
    
    Returns
    -------
    numpy.ndarray
        The cross-correlogram
    numpy.ndarray
        Center of the bins (in s)
    
    """
    # nbins = ((windowsize//binsize)*2)    

    nt1 = len(t1)
    nt2 = len(t2)

    nbins = int((windowsize*2)//binsize)
    if np.floor(nbins/2)*2 == nbins:
        nbins = nbins+1

    w = ((nbins/2) * binsize)
    C = np.zeros(nbins)
    i2 = 0

    for i1 in range(nt1):
        lbound = t1[i1] - w
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2+1
        while i2 > 0 and t2[i2-1] > lbound:
            i2 = i2-1

        rbound = lbound
        l = i2
        for j in range(nbins):
            k = 0
            rbound = rbound+binsize
            while l < nt2 and t2[l] < rbound:
                l = l+1
                k = k+1

            C[j] += k

    C = C/(nt1 * binsize)

    m = -w + binsize/2
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m+j*binsize

    return C, B

def compute_autocorrelogram(group, binsize, windowsize, ep=None, norm=True, time_units='s'):
    """
    Computes the autocorrelogram of a group of Ts/Tsd objects.
    The group can be passed directly as a TsGroup object.
    
    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects to auto-correlate
    binsize : float
        The bin size. Default is second. 
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which auto-corrs are computed. 
        If None, the epoch is the time support of the group.
    norm : bool, optional
         If True, autocorrelograms are normalized to baseline (i.e. divided by the average rate)
         If False, autoorrelograms are returned as the rate (Hz) of the time series (relative to itself)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    
    Returns
    -------
    pandas.DataFrame
        _
    
    Raises
    ------
    RuntimeError
        group must be TsGroup
    """
    if type(group) is nap.TsGroup:
        if isinstance(ep, nap.IntervalSet):            
            newgroup = group.restrict(ep)
        else:
            newgroup = group
    else:
        raise RuntimeError("Unknown format for group")

    autocorrs = {}

    binsize = nap.TimeUnits.format_timestamps(binsize, time_units)[0]
    windowsize = nap.TimeUnits.format_timestamps(windowsize, time_units)[0]

    for n in newgroup.keys():
        spk_time = newgroup[n].as_units('s').index.values
        auc, times = cross_correlogram(spk_time, spk_time, binsize, windowsize)
        # times = nap.TimeUnits.return_timestamps(times, 's')
        autocorrs[n] = pd.Series(index=times, data=auc, dtype='float')
        
    autocorrs = pd.DataFrame.from_dict(autocorrs)

    if norm:    
        autocorrs = autocorrs / newgroup.get_info('freq')

    autocorrs.loc[0] = 0.0

    return autocorrs.astype('float')

def compute_crosscorrelogram(group, binsize, windowsize, ep=None, norm=True, time_units='s',reverse=False):
    """
    Computes all the pairwise cross-correlograms from a group of Ts/Tsd objects.
    The group can be passed directly as a TsGroup object.
    The reference Ts/Tsd and target are chosen based on the builtin itertools.combinations function.
    For example if indexes are [0,1,2], the function computes cross-correlograms 
    for the pairs (0,1), (0, 2), and (1, 2). The left index gives the reference time series.
    To reverse the order, set reverse=True.
    
    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects to cross-correlate
    binsize : float
        The bin size. Default is second. 
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which cross-corrs are computed. 
        If None, the epoch is the time support of the group.
    norm : bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-orrelograms are returned as the rate (Hz) of the target time series ((relative to the reference time series)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    reverse : bool, optional
        To reverse the pair order
    
    Returns
    -------
    pandas.DataFrame
        _
    
    Raises
    ------
    RuntimeError
        group must be TsGroup
    
    """
    if type(group) is nap.TsGroup:
        if isinstance(ep, nap.IntervalSet):            
            newgroup = group.restrict(ep)
        else:
            newgroup = group
    else:
        raise RuntimeError("Unknown format for group")

    neurons = list(newgroup.keys())
    pairs = list(combinations(neurons, 2))
    if reverse: pairs = list(map(lambda n: (n[1],n[0]), pairs))
    crosscorrs = {}

    binsize = nap.TimeUnits.format_timestamps(binsize, time_units)[0]
    windowsize = nap.TimeUnits.format_timestamps(windowsize, time_units)[0]

    for i, j in pairs:
        spk1 = newgroup[i].as_units('s').index.values
        spk2 = newgroup[j].as_units('s').index.values            
        auc, times = cross_correlogram(spk1, spk2, binsize, windowsize)        
        crosscorrs[(i,j)] = pd.Series(index = times,data = auc, dtype='float')
        
    crosscorrs = pd.DataFrame.from_dict(crosscorrs)

    if norm:    
        freq = newgroup.get_info('freq')
        freq2 = pd.Series(index=pairs,data=list(map(lambda n:freq.loc[n[1]], pairs)))
        crosscorrs = crosscorrs / freq2

    return crosscorrs.astype('float')

def compute_eventcorrelogram(group, event, binsize, windowsize, ep=None, norm=True, time_units='s'):
    """
    Computes the correlograms of a group of Ts/Tsd objects with another single Ts/Tsd object
    The time of reference is the event times.
    
    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects to correlate with the event
    event : Ts/Tsd
        The event to correlate the each of the time series in the group with.
    binsize : float
        The bin size. Default is second. 
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which cross-corrs are computed. 
        If None, the epoch is the time support of the event.
    norm : bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-orrelograms are returned as the rate (Hz) of the target time series (relative to the event time series)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    Returns
    -------
    pandas.DataFrame
        _
    
    Raises
    ------
    RuntimeError
        group must be TsGroup
    
    """
    if ep is None:
        ep = event.time_support
        tsd1 = event.as_units('s').index.values
    else:
        tsd1 = event.restrict(ep).as_units('s').index.values

    if type(group) is nap.TsGroup:
        newgroup = group.restrict(ep)
    else:
        raise RuntimeError("Unknown format for group")

    crosscorrs = {}

    binsize = nap.TimeUnits.format_timestamps(binsize, time_units)[0]
    windowsize = nap.TimeUnits.format_timestamps(windowsize, time_units)[0]

    for n in newgroup.keys():        
        spk_time = newgroup[n].as_units('s').index.values
        auc, times = cross_correlogram(tsd1, spk_time, binsize, windowsize)        
        crosscorrs[n] = pd.Series(index = times,data = auc, dtype='float')
        
    crosscorrs = pd.DataFrame.from_dict(crosscorrs)

    if norm:    
        crosscorrs = crosscorrs / newgroup.get_info('freq')

    return crosscorrs.astype('float')
