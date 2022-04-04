# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 23:33:42
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-04 15:44:43


import warnings
import numpy as np
import pandas as pd
from .. import core as nap


def compute_1d_tuning_curves(group, feature, nb_bins, ep=None, minmax=None):
    """
    Computes 1-dimensional tuning curves relative to a 1d feature.
    
    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd for which the tuning curves will be computed 
    feature : Tsd
        The 1-dimensional target feature (e.g. head-direction)
    nb_bins : int
        Number of bins in the tuning curve
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target feature
   
    Returns
    -------
    pandas.DataFrame
        DataFrame to hold the tuning curves
   
    Raises
    ------
    RuntimeError
        If group is not a TsGroup object.
   
    """
    if not isinstance(group, nap.TsGroup):
        raise RuntimeError("Unknown format for group")

    if minmax is None:
        bins = np.linspace(np.min(feature), np.max(feature), nb_bins+1)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins+1)
    idx = bins[0:-1]+np.diff(bins)/2

    tuning_curves = pd.DataFrame(index=idx, columns=list(group.keys()))    

    if isinstance(ep, nap.IntervalSet):
        group_value = group.value_from(feature, ep)
        occupancy, _ = np.histogram(feature.restrict(ep).values, bins)
    else:
        group_value = group.value_from(feature)
        occupancy, _ = np.histogram(feature.values, bins)

    for k in group_value:
        count, _ = np.histogram(group_value[k].values, bins) 
        count = count/occupancy
        count[np.isnan(count)] = 0.0
        tuning_curves[k] = count
        tuning_curves[k] = count*feature.rate

    return tuning_curves

def compute_2d_tuning_curves(group, feature, nb_bins, ep=None, minmax=None):
    """
    Computes 2-dimensional tuning curves relative to a 2d feature
    
    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd for which the tuning curves will be computed 
    feature : TsdFrame
        The 2d feature (i.e. 2 columns features).
    nb_bins : int
        Number of bins in the tuning curves
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves given as:
        (minx, maxx, miny, maxy)
        If None, the boundaries are inferred from the target variable
    
    Returns
    -------
    tuple
        A tuple containing: \n
        tc (dict): Dictionnary of the tuning curves with dimensions (nb_bins, nb_bins).\n
        xy (list): List of bins center in the two dimensions
    
    Raises
    ------
    RuntimeError
        If group is not a TsGroup object or if feature is not 2 columns only.
    
    """
    if feature.shape[1] != 2:
        raise RuntimeError("feature should have 2 columns only.")

    if type(group) is not nap.TsGroup:
        raise RuntimeError("Unknown format for group")

    if isinstance(ep, nap.IntervalSet):
        feature = feature.restrict(ep)
    else:
        ep = feature.time_support

    cols = list(feature.columns)    
    groups_value = {}
    binsxy = {}
    
    for i, c in enumerate(cols):
        groups_value[c] = group.value_from(feature[c], ep)
        if minmax is None:
            bins = np.linspace(np.min(feature[c]), np.max(feature[c]), nb_bins+1)
        else:
            bins = np.linspace(minmax[i+i%2], minmax[i+1+i%2], nb_bins+1)
        binsxy[c] = bins

    occupancy, _, _ = np.histogram2d(
        feature[cols[0]].values, 
        feature[cols[1]].values, 
        [binsxy[cols[0]], binsxy[cols[1]]])

    tc = {}
    for n in group.keys():
        count,_,_ = np.histogram2d(
            groups_value[cols[0]][n].values,
            groups_value[cols[1]][n].values,
            [binsxy[cols[0]], binsxy[cols[1]]]
            )
        count = count / occupancy
        # count[np.isnan(count)] = 0.0
        tc[n] = count * feature.rate

    xy = [binsxy[c][0:-1] + np.diff(binsxy[c])/2 for c in binsxy.keys()]
    
    return tc, xy

def compute_1d_mutual_info(tc, feature, ep=None, minmax=None, bitssec=False):
    """
    Mutual information as defined in 
    
    Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993). 
    An information-theoretic approach to deciphering the hippocampal code. 
    In Advances in neural information processing systems (pp. 1030-1037).
    
    Parameters
    ----------
    tc : pandas.DataFrame or numpy.ndarray
        Tuning curves in columns
    feature : Tsd
        The feature that was used to compute the tuning curves
    ep : IntervalSet, optional
        The epoch over which the tuning curves were computed
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target feature
    bitssec : bool, optional
        By default, the function return bits per spikes.
        Set to true for bits per seconds
    
    Returns
    -------
    pandas.DataFrame
        Spatial Information (default is bits/spikes)
    """
    if type(tc) is pd.DataFrame:
        columns = tc.columns.values
        fx = np.atleast_2d(tc.values)
    elif type(tc) is np.ndarray:
        columns = np.arange(tc.shape[1])
        fx = np.atleast_2d(tc)

    nb_bins = tc.shape[0]+1
    if minmax is None:
        bins = np.linspace(np.min(feature), np.max(feature), nb_bins)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins)
    idx = bins[0:-1]+np.diff(bins)/2

    if isinstance(ep, nap.IntervalSet):
        occupancy, _ = np.histogram(feature.restrict(ep).values, bins)
    else:
        occupancy, _ = np.histogram(feature.values, bins)
    occupancy = occupancy / occupancy.sum()
    occupancy = occupancy[:,np.newaxis]

    fr = np.sum(fx * occupancy, 0)
    fxfr = fx/fr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logfx = np.log2(fxfr)        
    logfx[np.isinf(logfx)] = 0.0
    SI = np.sum(occupancy * fx * logfx, 0)

    if bitssec:
        SI = pd.DataFrame(index = columns, columns = ['SI'], data = SI)    
        return SI
    else:
        SI = SI / fr
        SI = pd.DataFrame(index = columns, columns = ['SI'], data = SI)
        return SI

def compute_2d_mutual_info(tc, features, ep=None, minmax=None, bitssec=False):
    """
    Mutual information as defined in 
        
    Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993). 
    An information-theoretic approach to deciphering the hippocampal code. 
    In Advances in neural information processing systems (pp. 1030-1037).

    Parameters
    ----------
    tc : dict or numpy.ndarray
        If array, first dimension should be the neuron
    features : TsdFrame
        The 2 columns features that were used to compute the tuning curves
    ep : IntervalSet, optional
        The epoch over which the tuning curves were computed
        If None, the epoch is the time support of the feature.
    minmax: tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target features
    bitssec: bool, optional
        By default, the function return bits per spikes.
        Set to true for bits per seconds

    Returns
    -------
    pandas.DataFrame
        Spatial Information (default is bits/spikes)
    """
    # A bit tedious here
    if type(tc) is dict:
        fx = np.array([tc[i] for i in tc.keys()])
        idx = list(tc.keys())
    elif type(tc) is np.ndarray:
        fx = tc
        idx = np.arange(len(tc))

    nb_bins = (fx.shape[1]+1,fx.shape[2]+1)

    cols = features.columns

    bins = []
    for i, c in enumerate(cols):
        if minmax is None:
            bins.append(np.linspace(np.min(features[c]), np.max(features[c]), nb_bins[i]))
        else:
            bins.append(np.linspace(minmax[i+i%2], minmax[i+1+i%2], nb_bins[i]))
            
    if isinstance(ep, nap.IntervalSet):
        features = features.restrict(ep)

    occupancy, _, _ = np.histogram2d(features[cols[0]].values, features[cols[1]].values, [bins[0], bins[1]])
    occupancy = occupancy / occupancy.sum()

    fr = np.nansum(fx * occupancy, (1,2))
    fr = fr[:,np.newaxis,np.newaxis]
    fxfr = fx/fr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logfx = np.log2(fxfr)        
    logfx[np.isinf(logfx)] = 0.0
    SI = np.nansum(occupancy * fx * logfx, (1,2))

    if bitssec:
        SI = pd.DataFrame(index = idx, columns = ['SI'], data = SI)    
        return SI
    else:
        SI = SI / fr[:,0,0]
        SI = pd.DataFrame(index = idx, columns = ['SI'], data = SI)
        return SI

def compute_1d_tuning_curves_continous(tsdframe, feature, nb_bins, ep=None, minmax=None):
    """
    Computes 1-dimensional tuning curves relative to a feature with continous data.
    
    Parameters
    ----------
    tsdframe : Tsd or TsdFrame
        Input data (e.g. continus calcium data 
        where each column is the calcium activity of one neuron)
    feature : Tsd
        The feature (one column)
    nb_bins : int
        Number of bins in the tuning curves
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target feature
    
    Returns
    -------
    pandas.DataFrame
        DataFrame to hold the tuning curves    
    
    Raises
    ------
    RuntimeError
        If tsdframe is not a Tsd or a TsdFrame object.
    
    """
    if not isinstance(tsdframe, (nap.Tsd, nap.TsdFrame)):
        raise RuntimeError("Unknown format for tsdframe.")

    if isinstance(ep, nap.IntervalSet):
        feature = feature.restrict(ep)
        tsdframe = tsdframe.restrict(ep)
    else:
        tsdframe = tsdframe.restrict(feature.time_support)

    if minmax is None:
        bins = np.linspace(np.min(feature), np.max(feature), nb_bins+1)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins+1)

    align_times = tsdframe.value_from(feature)
    idx = np.digitize(align_times.values, bins)-1
    tmp = pd.DataFrame(tsdframe).groupby(idx).mean()
    tmp = tmp.reindex(np.arange(0, len(bins)-1))
    tmp.index = pd.Index(bins[0:-1]+np.diff(bins)/2)

    tmp = tmp.fillna(0)

    return tmp

def compute_2d_tuning_curves_continuous(tsdframe, features, nb_bins, ep=None, minmax=None):
    """
    Computes 2-dimensional tuning curves relative to a 2d feature with continous data.
    
    Parameters
    ----------
    tsdframe : Tsd or TsdFrame
        Input data (e.g. continus calcium data 
        where each column is the calcium activity of one neuron)
    features : TsdFrame
        The 2d feature (two columns).
    nb_bins : int
        Number of bins in the tuning curves
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves given as:
        (minx, maxx, miny, maxy)
        If None, the boundaries are inferred from the target variable
    
    Returns
    -------
    tuple
        A tuple containing: \n
        tc (dict): Dictionnary of the tuning curves with dimensions (nb_bins, nb_bins).\n
        xy (list): List of bins center in the two dimensions
    
    Raises
    ------
    RuntimeError
        If tsdframe is not a Tsd/TsdFrame or if features is not 2 columns
    
    """
    if not isinstance(tsdframe, (nap.Tsd, nap.TsdFrame)):
        raise RuntimeError("Unknown format for tsdframe.")

    if isinstance(ep, nap.IntervalSet):
        features = features.restrict(ep)
        tsdframe = tsdframe.restrict(ep)
    else:
        tsdframe = tsdframe.restrict(features.time_support)

    if features.shape[1] != 2:
        raise RuntimeError("features input is not 2 columns.")

    cols = list(features.columns)

    binsxy = {}
    idxs = {}

    for i, c in enumerate(cols):
        if minmax is None:
            bins = np.linspace(np.min(features[c]), np.max(features[c]), nb_bins+1)
        else:
            bins = np.linspace(minmax[i+i%2], minmax[i+1+i%2], nb_bins+1)

        align_times = tsdframe.value_from(features[c], ep)
        idxs[c] = np.digitize(align_times.values, bins)-1
        binsxy[c] = bins

    idxs = pd.DataFrame(idxs)

    tc_np = np.zeros((tsdframe.shape[1], nb_bins, nb_bins))*np.nan

    for k, tmp in idxs.groupby(cols):
        if (0<=k[0]<nb_bins) and (0<=k[1]<nb_bins):
            tc_np[:,k[0],k[1]] = tsdframe.iloc[tmp.index].mean(0).values
    
    tc_np[np.isnan(tc_np)] = 0.0

    xy = [binsxy[c][0:-1] + np.diff(binsxy[c])/2 for c in binsxy.keys()]
    
    tc = {c:tc_np[i] for i, c in enumerate(tsdframe.columns)}

    return tc, xy