# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 11:39:55
# @Last Modified by:   gviejo
# @Last Modified time: 2022-01-03 17:44:36


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
    The units should be in ms for all arguments.
    Return the firing rate of the series t2 relative to the timings of t1.
    
    
    Parameters
    ----------
    t1 : numpy.ndarray
        The timestamps of the reference time series (in ms)
    t2 : numpy.ndarray
        The timestamps of the target time series (in ms)
    binsize : float
        The bin size (in ms)
    windowsize : float
        The window size (in ms)
    
    Returns
    -------
    numpy.ndarray
        The cross-correlogram
    numpy.ndarray
        Center of the bins (in ms)
    
    """
    # nbins = ((windowsize//binsize)*2)    

    nt1 = len(t1)
    nt2 = len(t2)

    nbins = (windowsize*2)//binsize
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

    C = C/(nt1 * binsize/1000)

    m = -w + binsize/2
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m+j*binsize

    return C, B

def compute_autocorrelogram(group, ep, binsize, windowsize, norm=True):
    """
    Computes the autocorrelogram of a group of Ts/Tsd objects.
    The group can be passed directly as a TsGroup object.
    
    Parameters
    ----------
    group: TsGroup or dict of Ts/Tsd objects
        The input Ts/Tsd objects 
    ep: IntervalSet
        The epoch on which auto-corrs are computed
    binsize: float
        The bin size (in ms)
    windowsize: float
        The window size (in ms)
    norm: bool, optional
         If True, autocorrelograms are normalized to baseline (i.e. divided by the average rate)
         If False, autoorrelograms are returned as the rate (Hz) of the time series (relative to itself)

    Returns
    -------
    pandas.DataFrame
        _
    
    Raises
    ------
    RuntimeError
        group must be TsGroup or dictionnary of Ts/Tsd objects
    """
    if type(group) is dict:
        newgroup = nap.TsGroup(group, time_support = ep)
    elif type(group) is nap.TsGroup:
        newgroup = group.restrict(ep)
    else:
        raise RuntimeError("Unknown format for group")

    autocorrs = {}

    for n in newgroup.keys():        
        spk_time = newgroup[n].as_units('ms').index.values        
        auc, times = cross_correlogram(spk_time, spk_time, binsize, windowsize)
        autocorrs[n] = pd.Series(index = times,data = auc)
        
    autocorrs = pd.DataFrame.from_dict(autocorrs)

    if norm:    
        autocorrs = autocorrs / newgroup.get_info('freq')

    autocorrs.loc[0] = 0.0

    return autocorrs

def compute_crosscorrelogram(group, ep, binsize, windowsize, norm=True,reverse=False):
    """
    Computes all the pairwise cross-correlograms from a group of Ts/Tsd objects.
    The group can be passed directly as a TsGroup object.
    The reference Ts/Tsd and target are chosen based on the builtin itertools.combinations function.
    For example if indexes are [0,1,2], the function computes cross-correlograms 
    for the pairs (0,1), (0, 2), and (1, 2). The left index gives the reference time series.
    To reverse the order, set reverse=True.
    
    Parameters
    ----------
    group: TsGroup or dict of Ts/Tsd objects
        The Ts/Tsd objects to cross-correlate
    ep: IntervalSet
        The epoch on which cross-correlograms are computed
    binsize: float
        The bin size in ms
    windowsize: float
        The window size in ms
    norm: bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-orrelograms are returned as the rate (Hz) of the target time series ((relative to the reference time series)
    reverse : bool, optional
        To reverse the pair order
    
    Returns
    -------
    pandas.DataFrame
        _
    
    Raises
    ------
    RuntimeError
        group must be TsGroup or dictionnary of Ts/Tsd objects
    
    """
    if type(group) is dict:
        newgroup = nap.TsGroup(group, time_support = ep)
    elif type(group) is nap.TsGroup:
        newgroup = group.restrict(ep)
    else:
        raise RuntimeError("Unknown format for group")

    neurons = list(newgroup.keys())
    pairs = list(combinations(neurons, 2))
    if reverse: pairs = list(map(lambda n: (n[1],n[0]), pairs))
    crosscorrs = {}

    for i, j in pairs:
        spk1 = newgroup[i].as_units('ms').index.values
        spk2 = newgroup[j].as_units('ms').index.values            
        auc, times = cross_correlogram(spk1, spk2, binsize, windowsize)
        crosscorrs[(i,j)] = pd.Series(index = times,data = auc)
        
    crosscorrs = pd.DataFrame.from_dict(crosscorrs)

    if norm:    
        freq = newgroup.get_info('freq')
        freq2 = pd.Series(index=pairs,data=list(map(lambda n:freq.loc[n[1]], pairs)))
        crosscorrs = crosscorrs / freq2

    return crosscorrs

def compute_eventcorrelogram(group, event, ep, binsize, windowsize, norm=True):
    """
    Computes the correlograms of a group of Ts/Tsd objects with another single Ts/Tsd object
    The group can be passed directly as a TsGroup object.
    
    Parameters
    ----------
    group : TsGroup or dict of Ts/Tsd objects
        The Ts/Tsd objects to correlate with the event
    event : Ts/Tsd
        The event to correlate the each of the time series in the group with.
    ep : IntervalSet
        The epoch on which cross-correlograms are computed
    binsize : float
        The bin size in ms
    windowsize : float
        The window size in ms
    norm : bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-orrelograms are returned as the rate (Hz) of the target time series (relative to the event time series)
    
    Returns
    -------
    pandas.DataFrame
        _
    
    Raises
    ------
    RuntimeError
        group must be TsGroup or dictionnary of Ts/Tsd objects
    
    """
    if type(group) is dict:
        newgroup = nap.TsGroup(group, time_support = ep)
    elif type(group) is nap.TsGroup:
        newgroup = group.restrict(ep)
    else:
        raise RuntimeError("Unknown format for group")

    crosscorrs = {}

    tsd1 = event.restrict(ep).as_units('ms').index.values

    for n in newgroup.keys():        
        spk_time = newgroup[n].as_units('ms').index.values        
        auc, times = cross_correlogram(tsd1, spk_time, binsize, windowsize)
        crosscorrs[n] = pd.Series(index = times,data = auc)
        
    crosscorrs = pd.DataFrame.from_dict(crosscorrs)

    if norm:    
        crosscorrs = crosscorrs / newgroup.get_info('freq')

    return crosscorrs
