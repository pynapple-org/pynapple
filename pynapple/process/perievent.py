# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-30 22:59:00
# @Last Modified by:   gviejo
# @Last Modified time: 2022-02-16 18:31:48

import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
from scipy import signal
from .. import core as nap


@jit(nopython=True)
def align_to_event(times, data, tref, windowsize):
    """
    Helper function compiled with numba for aligning times.
    See compute_perievent for using this function

    Parameters
    ----------
    times : numpy.ndarray
        The timestamps to align
    data : numpy.ndarray
        The data to align
    tref : numpy.ndarray
        The reference times
    windowsize : tuple
        Start and end of the window size around tref
    
    Returns
    -------
    list
        The align times and data
    """
    nt2 = len(tref)

    x = []    
    y = []

    for i in range(nt2):
        lbound = tref[i] - windowsize[0]
        rbound = tref[i] + windowsize[1]
        left = times > lbound
        right = times < rbound

        idx = np.logical_and(left, right)

        x.append(times[idx] - tref[i])
        y.append(data[idx])

    return x, y

def compute_perievent(data, tref,  minmax, time_unit = 's'):
    """
    Center ts/tsd/tsgroup object around the timestamps given by the tref argument.
    minmax indicates the start and end of the window.
    
    Parameters
    ----------
    data : Ts/Tsd/TsGroup
        The data to align to tref. 
        If Ts/Tsd, returns a TsGroup. 
        If TsGroup, returns a dictionnary of TsGroup
    tref : Ts/Tsd
        The timestamps of the event to align to
    minmax : tuple or int or float
        The window size. Can be unequal on each side i.e. (-500, 1000).
    time_unit : str, optional
        Time units of the minmax ('s' [default], 'ms', 'us').
    
    Returns
    -------
    dict
        A TsGroup if data is a Ts/Tsd or
        a dictionnary of TsGroup if data is a TsGroup.
    
    Raises
    ------
    RuntimeError
        if tref is not a Ts/Tsd object or if data is not a Ts/Tsd or TsGroup
    """
    if not isinstance(tref, nap.Tsd):
        raise RuntimeError("tref should be a Tsd object.")

    if isinstance(minmax, float) or isinstance(minmax, int):
        minmax = (minmax, minmax)

    window = np.abs(nap.TimeUnits.format_timestamps(np.array(minmax), time_unit))

    time_support = nap.IntervalSet(start = -window[0], end = window[1])

    if isinstance(data, nap.TsGroup):
        toreturn = {}
        for n in data.keys():
            toreturn[n] = compute_perievent(data[n], tref, minmax, time_unit)

        return toreturn

    elif isinstance(data, (nap.Ts, nap.Tsd)):
        
        xt, yd = align_to_event(data.index.values, data.values, tref.index.values, window)

        group = {}
        for i, (x, y) in enumerate(zip(xt, yd)):
            group[i] = nap.Tsd(
                t = x,
                d = y,
                time_support = time_support
                )

        group = nap.TsGroup(group, time_support = time_support)
        group.set_info(ref_times = tref.as_units('s').index.values)

    else: 
        raise RuntimeError("Unknown format for data")
    
    return group

