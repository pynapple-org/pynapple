# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 23:33:42
# @Last Modified by:   gviejo
# @Last Modified time: 2022-01-06 15:27:55

import numpy as np
import pandas as pd
from .. import core as nap


def compute_1d_tuning_curves(group, feature, ep, nb_bins, minmax=None):
    """
    Computes 1-dimensional tuning curves relative to a 1d feature.
    
    Parameters
    ----------
    group: TsGroup or dict of Ts/Tsd objects
        The group of Ts/Tsd for which the tuning curves will be computed 
    feature: Tsd
        The 1-dimensional target feature (e.g. head-direction)
    ep: IntervalSet
        The epoch on which tuning curves are computed
    nb_bins: int
        Number of bins in the tuning curve
    minmax: tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target feature
    
    Returns
    -------
    pandas.DataFrame
        DataFrame to hold the tuning curves
    """
    if type(group) is dict:
        group = nap.TsGroup(group, time_support = ep)

    group_value = group.value_from(feature, ep)

    if minmax is None:
        bins = np.linspace(np.min(feature), np.max(feature), nb_bins)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins)
    idx = bins[0:-1]+np.diff(bins)/2

    tuning_curves = pd.DataFrame(index = idx, columns = list(group.keys()))    

    occupancy, _     = np.histogram(feature.values, bins)

    for k in group_value:
        count, bin_edges = np.histogram(group_value[k].values, bins) 
        count = count/occupancy
        count[np.isnan(count)] = 0.0
        tuning_curves[k] = count
        tuning_curves[k] = count*feature.rate

    return tuning_curves

def compute_2d_tuning_curves(group, feature, ep, nb_bins, minmax=None):
    """
    Computes 2-dimensional tuning curves relative to a 2d feature
    
    Parameters
    ----------
    group: TsGroup or dict of Ts/Tsd objects
        The group input
    feature: 2d TsdFrame
        The 2d feature.
    ep: IntervalSet
        The epoch on which tuning curves are computed
    nb_bins: int
        Number of bins in the tuning curves
    minmax: tuple or list, optional
        The min and max boundaries of the tuning curves given as:
        (minx, maxx, miny, maxy)
        If None, the boundaries are inferred from the target variable
    
    Returns
    -------
    numpy.ndarray
        Stacked array of the tuning curves with dimensions (n, nb_bins, nb_bins).
        n is the number of object in the input group. 
    list
        bins center in the two dimensions

    """
    if type(group) is dict:
        group = nap.TsGroup(group, time_support = ep)

    if feature.shape[1] != 2:
        raise RuntimeError("Variable is not 2 dimensional.")

    cols = list(feature.columns)

    groups_value = {}
    binsxy = {}
    for i, c in enumerate(cols):
        groups_value[c] = group.value_from(feature[c], ep)
        if minmax is None:
            bins = np.linspace(np.min(feature[c]), np.max(feature[c]), nb_bins)
        else:
            bins = np.linspace(minmax[i+i%2], minmax[i+1+i%2], nb_bins)
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
        count[np.isnan(count)] = 0.0
        tc[n] = count * feature.rate

    xy = [binsxy[c][0:-1] + np.diff(binsxy[c])/2 for c in binsxy.keys()]
    
    return tc, xy