# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-30 22:59:00
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-08 17:42:20

import numpy as np
from scipy.linalg import hankel

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
    starts = ep.start.values
    ends = ep.end.values

    binsize = time_array[1] - time_array[0]
    idx1 = -np.arange(0, window[0] + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, window[1] + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    windowsize = np.array([idx1.shape[0], idx2.shape[0]])

    new_data_array = nap.jitted_functions.jitcontinuous_perievent(
        time_array, data_array, time_target_array, starts, ends, windowsize
    )

    time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if new_data_array.ndim == 2:
        return nap.TsdFrame(t=time_idx, d=new_data_array, time_support=time_support)
    else:
        return nap.TsdTensor(t=time_idx, d=new_data_array, time_support=time_support)


def compute_event_trigger_average(
    group, feature, binsize, windowsize, ep, time_unit="s", fill_method="forward"
):
    """
    Bin the spike train in binsize and compute the Event Trigger Average (ETA) within windowsize.
    If C is the spike count matrix and `feature` is a Tsd array, the function computes
    the Hankel matrix H from windowsize=(-t1,+t2) by offseting the Tsd array.

    The ETA is then defined as the dot product between H and C divided by the number of events.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects that hold the trigger time.
    feature : Tsd
        The 1-dimensional feature to average. Can be a TsdFrame with one column only.
    binsize : float or int
        The bin size. Default is second.
        If different, specify with the parameter time_unit ('s' [default], 'ms', 'us').
    windowsize : tuple or list of float
        The window size. Default is second. For example (-1, 1).
        If different, specify with the parameter time_unit ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which ETA are computed
    time_unit : str, optional
        The time unit of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    fill_method : str, optional
        The method for adding feature values if the resolution determined by the binsize is larger than the feature rate.
        ('forward' [default], 'backward', 'closest')

    Returns
    -------
    TsdFrame
        A TsdFrame of Event-Trigger Average. Each column is an element from the group.

    Raises
    ------
    RuntimeError
        if group is not a Ts/Tsd or TsGroup
    """
    assert isinstance(group, nap.TsGroup), "group should be a TsGroup."
    assert isinstance(
        feature, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)
    ), "Feature should be a Tsd, TsdFrame or TsdTensor"
    assert isinstance(
        windowsize, (float, int, tuple)
    ), "windowsize should be a tuple or int or float."
    assert isinstance(binsize, (float, int)), "binsize should be int or float."
    assert isinstance(time_unit, str), "time_unit should be a str."
    assert time_unit in ["s", "ms", "us"], "time_unit should be 's', 'ms' or 'us'"
    assert isinstance(ep, (nap.IntervalSet)), "ep should be an IntervalSet object."
    assert fill_method in ["forward", "backward", "closest"], "fill_method should be 'forward', 'backward', 'closest'"

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

    if feature.rate > 1/binsize

        tmp = feature.bin_average(binsize, ep)

        # Check for any NaNs in feature
        if np.any(np.isnan(tmp)):
            tmp = tmp.dropna()
            ep = tmp.time_support

        # count = group.count(binsize, ep)
        n_p = len(idx1)
        n_f = len(idx2)
        pad_tmp = np.pad(tmp, (n_p, n_f))
        offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]

        eta = np.dot(offset_tmp.T, count.values)

        eta = eta / np.sum(count, 0)

        eta = nap.TsdFrame(t=time_idx, d=eta, columns=group.index)        

    else:
        
        time_array = feature.index.values
        data_array = feature.values
        starts = ep.start.values
        ends = ep.end.values
        windows = np.hstack((time_idx - binsize/2, [time_idx[-1] + binsize]))

        for i, n in enumerate(group.keys()):
            
            time_target_array = group[n].index.values

            eta[:,i] = nap.jitted_functions.jitperievent_trigger_average(
                time_array, data_array, time_target_array, starts, ends, windows, fill_method
                )

        eta[:,i] = new_data_array.mean(1)



    # if tmp.ndim == 1:
    #     # # Build the Hankel matrix
    #     # pad_tmp = np.pad(tmp, (n_p, n_f))
    #     # offset_tmp = hankel(pad_tmp, pad_tmp[-(n_p + n_f + 1) :])[0 : len(tmp)]

    #     # sta = np.dot(offset_tmp.T, count.values)

    #     # sta = sta / np.sum(count, 0)

    #     # sta = nap.TsdFrame(t=time_idx, d=sta, columns=group.index)




    #     eta = jit_compute_eta(tmp.values, count, n_p, n_f)
    #     eta = nap.TsdFrame(t=time_idx, d=eta, columns=group.index)

    # else:
        
    #     tmp = tmp.values.reshape(tmp.shape[0], np.prod(tmp.shape[1:]))

    #     sta = np.zeros((time_idx.shape[0], count.shape[1], tmp.shape[1]))

    #     for i in range(tmp.shape[1]):
    #         sta[:,:,i] = jit_compute_eta(tmp[:,i], count, n_p, n_f)

    #     # for i in range(n_p, len(count)-n_f-1):
    #     #     a = tmp[np.maximum(0, i-n_p):np.minimum(tmp.shape[0], i+n_f+1)].values
    #     #     b = count[np.maximum(0, i-n_p):np.minimum(tmp.shape[0], i+n_f+1)].values

    #     #     sta += np.einsum('ikl,ij->ijkl', a, b)

    if eta.ndim == 2:
        return nap.TsdFrame(t=time_idx, d=eta, columns=group.index)
    else:
        return nap.TsdTensor(t=time_idx, d=eta)


    return eta
