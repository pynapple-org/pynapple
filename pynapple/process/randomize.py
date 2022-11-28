# -*- coding: utf-8 -*-
# @Author: dspalla
# @Date:   2022-01-30 22:59:00
# @Last Modified by:   dspalla
# @Last Modified time: 2022-11-17 17:16:16

import numpy as np

from .. import core as nap


def shift_timestamps(ts,min_shift=0.0,max_shift=None):
    """
    Shifts all the time stamps of a random amount between min_shift and max_shift, wrapping the
    end of the time support to the beginning.


    Parameters
    ----------
    timestamps : Ts or TsGroup
        The timestamps to shift. If TsGroup, shifts all Ts in the group independently.
    min_shift : float, optional
        minimum shift (default: 0 )
    max_shift : float, optional
        maximum shift, (default: length of time support)

    Returns
    -------
    Ts or TsGroup
        The randomly shifted timestamps
    """
    strategies = {nap.core.time_series.Ts:_shift_ts,
                  nap.core.ts_group.TsGroup:_shift_tsgroup,
                  }
    # checks input type              
    if type(ts) not in strategies.keys():
        raise TypeError('Invalid input type, should be Ts or TsGroup')

    # checks max shift < len of time support  
    if max_shift > abs(ts.end_time - ts.start_time):
        raise ValueError('Invalid max_shift, max_shift > length of time support')

    strategy = strategies[type(ts)]
    return strategy(ts,min_shift,max_shift)


def _shift_ts(ts,min_shift=0,max_shift=None):
    """
    Shifts all the time stamps of a random amount between min_shift and max_shift, wrapping the
    end of the time support to the beginning.


    Parameters
    ----------
    timestamps : Ts 
        The timestamps to shift.
    min_shift : float, optional
        minimum shift (default: 0 )
    max_shift : float, optional
        maximum shift, (default: length of time support)

    Returns
    -------
    Ts 
        The randomly shifted timestamps
    """

    if max_shift == None:
        max_shift = ts.end_time() - ts.start_time()
    shift = np.random.uniform(min_shift,max_shift)
    shifted_timestamps = (ts.times() + shift) % ts.end_time() + ts.start_time()
    shifted_ts = nap.Ts(t=np.sort(shifted_timestamps))
    return shifted_ts


def _shift_tsgroup(tsgroup,min_shift=0,max_shift=None):
    """
    Shifts each Ts in the Ts group independently.


    Parameters
    ----------
    timestamps : TsGroup
        The collection of Ts to shift.
    min_shift : float, optional
        minimum shift (default: 0 )
    max_shift : float, optional
        maximum shift, (default: length of time support)

    Returns
    -------
    TsGroup
        The TSGroup with randomly shifted timestamps
    """

    start_time = tsgroup.time_support.start[0]
    end_time = tsgroup.time_support.end[0]

    if max_shift == None:
        max_shift = end_time - start_time

    shifted_tsgroup = {}
    for k in tsgroup.keys():
        shift = np.random.uniform(min_shift,max_shift)
        shifted_timestamps = (tsgroup[k].times() + shift) % end_time + start_time
        shifted_tsgroup[k] = nap.Ts(t=np.sort(shifted_timestamps))
    return nap.TsGroup(shifted_tsgroup)
