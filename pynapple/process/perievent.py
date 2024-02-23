# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-30 22:59:00
# @Last Modified by:   gviejo
# @Last Modified time: 2024-02-20 22:27:23

import numpy as np

from .. import core as nap


def _align_tsd(tsd, tref, window, time_support):
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
    lbounds = np.searchsorted(tsd.index, tref.index - window[0])
    rbounds = np.searchsorted(tsd.index, tref.index + window[1])

    group = {}

    if isinstance(tsd, nap.Ts):
        for i in range(len(tref)):
            tmp = tsd.index[lbounds[i] : rbounds[i]] - tref.index[i]
            group[i] = nap.Ts(t=tmp, time_support=time_support)
    else:
        for i in range(len(tref)):
            tmp = tsd.index[lbounds[i] : rbounds[i]] - tref.index[i]
            tmp2 = tsd.values[lbounds[i] : rbounds[i]]
            group[i] = nap.Tsd(t=tmp, d=tmp2, time_support=time_support)

    group = nap.TsGroup(group, time_support=time_support, bypass_check=True)
    group.set_info(ref_times=tref.index)

    return group


def compute_perievent(data, tref, minmax, time_unit="s"):
    """
    Center the timestamps of a time series object or a time series group around the timestamps given by the `tref` argument.
    `minmax` indicates the start and end of the window. If `minmax=(-5, 10)`, the window will be from -5 second to 10 second.
    If `minmax=10`, the window will be from -10 second to 10 second.

    To center continuous time series around a set of timestamps, you can use `compute_perievent_continuous`.

    Parameters
    ----------
    data : Ts, Tsd or TsGroup
        The data to align to tref.
        If Ts/Tsd, returns a TsGroup.
        If TsGroup, returns a dictionnary of TsGroup
    tref : Ts or Tsd
        The timestamps of the event to align to
    minmax : tuple, int or float
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
    assert isinstance(tref, (nap.Ts, nap.Tsd)), "tref should be a Ts or Tsd object."
    assert isinstance(
        data, (nap.Ts, nap.Tsd, nap.TsGroup)
    ), "data should be a Ts, Tsd or TsGroup."
    assert isinstance(
        minmax, (float, int, tuple)
    ), "minmax should be a tuple or int or float."
    assert isinstance(time_unit, str), "time_unit should be a str."
    assert time_unit in ["s", "ms", "us"], "time_unit should be 's', 'ms' or 'us'"

    if isinstance(minmax, float) or isinstance(minmax, int):
        minmax = np.array([minmax, minmax], dtype=np.float64)

    window = np.abs(nap.TsIndex.format_timestamps(np.array(minmax), time_unit))

    time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if isinstance(data, nap.TsGroup):
        toreturn = {}

        for n in data.index:
            toreturn[n] = _align_tsd(data[n], tref, window, time_support)

        return toreturn

    else:
        return _align_tsd(data, tref, window, time_support)


def compute_perievent_continuous(data, tref, minmax, ep=None, time_unit="s"):
    """
    Center continuous time series around the timestamps given by the 'tref' argument.
    `minmax` indicates the start and end of the window. If `minmax=(-5, 10)`, the window will be from -5 second to 10 second.
    If `minmax=10`, the window will be from -10 second to 10 second.

    To realign timestamps around a set of timestamps, you can use `compute_perievent_continuous`.

    This function assumes a constant sampling rate of the time series.

    Parameters
    ----------
    data : Tsd, TsdFrame or TsdTensor
        The data to align to tref.
    tref : Ts or Tsd
        The timestamps of the event to align to
    minmax : tuple or int or float
        The window size. Can be unequal on each side i.e. (-500, 1000).
    ep : IntervalSet, optional
        The epochs to perform the operation. If None, the default is the time support of the data.
    time_unit : str, optional
        Time units of the minmax ('s' [default], 'ms', 'us').

    Returns
    -------
    TsdFrame, TsdTensor
        If `data` is a one-dimensional Tsd, the output is a TsdFrame. Each column is one timestamps from `tref`.
        If `data` is a TsdFrame or TsdTensor, the output is a TsdTensor with one more dimension. The first dimension is always time and the second dimension is the 'tref' timestamps.

    Raises
    ------
    RuntimeError
        if tref is not a Ts/Tsd object or if data is not a Tsd/TsdFrame/TsdTensor object.
    """

    assert isinstance(tref, (nap.Ts, nap.Tsd)), "tref should be a Ts or Tsd object."
    assert isinstance(
        data, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)
    ), "data should be a Tsd, TsdFrame or TsdTensor."
    assert isinstance(
        minmax, (float, int, tuple)
    ), "minmax should be a tuple or int or float."
    assert isinstance(time_unit, str), "time_unit should be a str."
    assert time_unit in ["s", "ms", "us"], "time_unit should be 's', 'ms' or 'us'"

    if ep is None:
        ep = data.time_support
    else:
        assert isinstance(ep, (nap.IntervalSet)), "ep should be an IntervalSet object."

    if isinstance(minmax, float) or isinstance(minmax, int):
        minmax = np.array([minmax, minmax], dtype=np.float64)

    window = np.abs(nap.TsIndex.format_timestamps(np.array(minmax), time_unit))

    time_array = data.index.values
    data_array = data.values
    time_target_array = tref.index.values
    starts = ep.start
    ends = ep.end

    binsize = time_array[1] - time_array[0]
    idx1 = -np.arange(0, window[0] + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, window[1] + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    windowsize = np.array([idx1.shape[0], idx2.shape[0]])

    new_data_array = nap._jitted_functions.jitcontinuous_perievent(
        time_array, data_array, time_target_array, starts, ends, windowsize
    )

    time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if new_data_array.ndim == 2:
        return nap.TsdFrame(t=time_idx, d=new_data_array, time_support=time_support)
    else:
        return nap.TsdTensor(t=time_idx, d=new_data_array, time_support=time_support)


def compute_event_trigger_average(
    group,
    feature,
    binsize,
    windowsize=None,
    ep=None,
    time_unit="s",
):
    """
    Bin the event timestamps within binsize and compute the Event Trigger Average (ETA) within windowsize.
    If C is the event count matrix and `feature` is a Tsd array, the function computes
    the Hankel matrix H from windowsize=(-t1,+t2) by offseting the Tsd array.

    The ETA is then defined as the dot product between H and C divided by the number of events.

    The object feature can be any dimensions.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects that hold the trigger time.
    feature : Tsd, TsdFrame or TsdTensor
        The feature to average.
    binsize : float or int
        The bin size. Default is second.
        If different, specify with the parameter time_unit ('s' [default], 'ms', 'us').
    windowsize : tuple of float/int or float/int
        The window size. Default is second. For example windowsize = (-1, 1) is equivalent to windowsize = 1
        If different, specify with the parameter time_unit ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epochs on which the average is computed
    time_unit : str, optional
        The time unit of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    """
    assert isinstance(group, nap.TsGroup), "group should be a TsGroup."
    assert isinstance(
        feature, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)
    ), "Feature should be a Tsd, TsdFrame or TsdTensor"
    assert isinstance(binsize, (float, int)), "binsize should be int or float."
    assert isinstance(time_unit, str), "time_unit should be a str."
    assert time_unit in ["s", "ms", "us"], "time_unit should be 's', 'ms' or 'us'"

    if windowsize is not None:
        if isinstance(windowsize, tuple):
            assert (
                len(windowsize) == 2
            ), "windowsize should be a tuple of 2 elements (-t, +t)"
            assert all(
                [isinstance(t, (float, int)) for t in windowsize]
            ), "windowsize should be a tuple of int/float"
        else:
            assert isinstance(
                windowsize, (float, int)
            ), "windowsize should be a tuple of int/float or int/float."
            windowsize = (windowsize, windowsize)
    else:
        windowsize = (0.0, 0.0)

    if ep is not None:
        assert isinstance(ep, (nap.IntervalSet)), "ep should be an IntervalSet object."
    else:
        ep = feature.time_support

    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_unit
    )[0]
    start = np.abs(
        nap.TsIndex.format_timestamps(
            np.array([windowsize[0]], dtype=np.float64), time_unit
        )[0]
    )
    end = np.abs(
        nap.TsIndex.format_timestamps(
            np.array([windowsize[1]], dtype=np.float64), time_unit
        )[0]
    )

    idx1 = -np.arange(0, start + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, end + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))

    eta = np.zeros((time_idx.shape[0], len(group), *feature.shape[1:]))

    windows = np.array([len(idx1), len(idx2)])

    # Bin the spike train
    count = group.count(binsize, ep)

    time_array = np.round(count.index.values - (binsize / 2), 9)
    count_array = count.values
    starts = ep.start
    ends = ep.end

    time_target_array = feature.index.values
    data_target_array = feature.values

    if data_target_array.ndim == 1:
        eta = nap._jitted_functions.jitperievent_trigger_average(
            time_array,
            count_array,
            time_target_array,
            np.expand_dims(data_target_array, -1),
            starts,
            ends,
            windows,
            binsize,
        )
        eta = np.squeeze(eta, -1)
    else:
        eta = nap._jitted_functions.jitperievent_trigger_average(
            time_array,
            count_array,
            time_target_array,
            data_target_array,
            starts,
            ends,
            windows,
            binsize,
        )

    if eta.ndim == 2:
        return nap.TsdFrame(t=time_idx, d=eta, columns=group.index)
    else:
        return nap.TsdTensor(t=time_idx, d=eta)
