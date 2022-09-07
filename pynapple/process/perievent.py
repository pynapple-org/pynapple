# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-30 22:59:00
# @Last Modified by:   gviejo
# @Last Modified time: 2022-09-06 14:56:55

import numpy as np
from numba import jit
from scipy.linalg import hankel

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


def compute_perievent(data, tref, minmax, time_unit="s"):
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

    time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if isinstance(data, nap.TsGroup):
        toreturn = {}
        for n in data.keys():
            toreturn[n] = compute_perievent(data[n], tref, minmax, time_unit)

        return toreturn

    elif isinstance(data, (nap.Ts, nap.Tsd)):

        xt, yd = align_to_event(
            data.index.values, data.values, tref.index.values, window
        )

        group = {}
        for i, (x, y) in enumerate(zip(xt, yd)):
            group[i] = nap.Tsd(t=x, d=y, time_support=time_support)

        group = nap.TsGroup(group, time_support=time_support)
        group.set_info(ref_times=tref.as_units("s").index.values)

    else:
        raise RuntimeError("Unknown format for data")

    return group


def compute_event_trigger_average(
    group, feature, binsize, windowsize, ep, time_units="s"
):
    """
    Bin the spike train in binsize and compute the
    Spike Trigger Average (STA) within windowsize.
    If C is the spike count matrix and feature is a Tsd array, the function
    computes the Hankel matrix H from windowsize=(-t1,+t2) by offseting the Tsd array.

    The STA is then defined as the dot product between H and C divided by
    the number of spikes.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects that hold the trigger time.
    feature : Tsd
        The 1-dimensional feature to average
    binsize : float
        The bin size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which STA are computed
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').

    Returns
    -------
    TsdFrame
        A TsdFrame of Spike-Trigger Average. Each column is an element from the group.

    Raises
    ------
    RuntimeError
        if group is not a Ts/Tsd or TsGroup
    """
    if type(group) is not nap.TsGroup:
        raise RuntimeError("Unknown format for group")

    binsize = nap.TimeUnits.format_timestamps(binsize, time_units)[0]
    start = np.abs(nap.TimeUnits.format_timestamps(windowsize[0], time_units)[0])
    end = np.abs(nap.TimeUnits.format_timestamps(windowsize[1], time_units)[0])
    idx1 = -np.arange(0, start + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, end + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))

    # Bin the spike train
    #    count = newgroup.count(binsize)

    sta = []
    n_spikes = np.zeros(len(group))

    for i in ep.index:
        bins = np.arange(ep.start[i], ep.end[i] + binsize, binsize)
        bins = np.round(bins, 9)
        idx = np.digitize(feature.index.values, bins) - 1
        tmp = feature.groupby(idx).mean()
        if tmp.index.values[0] == -1.0:
            tmp = tmp.iloc[1:]
        if tmp.index.values[-1] == len(bins) - 1:
            tmp = tmp.iloc[:-1]

        if len(tmp) < len(bins) - 1:  # dirty
            tmp2 = np.zeros((len(bins) - 1))
            tmp2[tmp.index.values.astype("int")] = tmp.values
            tmp = tmp2

        tidx = nap.Tsd(t=bins[0:-1] + np.diff(bins) / 2, d=np.arange(len(tmp)))
        tmp = tmp[tidx.restrict(ep.loc[[i]]).values]

        # count_e = count.restrict(ep.loc[[i]]).values
        count = group.count(binsize, ep.loc[[i]])

        # Build the Hankel matrix
        n_p = len(idx1)
        n_f = len(idx2)
        pad_tmp = np.pad(tmp, (n_p, n_f))
        offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]

        n_spikes += count.sum(0).values

        sta.append(np.dot(offset_tmp.T, count))

    sta = np.array(sta).sum(0)

    sta = sta / n_spikes

    sta = nap.TsdFrame(t=time_idx, d=sta, columns=list(group.keys()))

    return sta
