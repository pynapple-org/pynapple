"""
Functions to realign time series relative to a reference time.
"""

import inspect
from functools import wraps
from numbers import Number

import numpy as np

from .. import core as nap
from ._process_functions import (
    _perievent_continuous,
    _perievent_trigger_average,
)


def _validate_perievent_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        kwargs = sig.bind_partial(*args, **kwargs).arguments

        parameters_type = {
            "timestamps": (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup),
            "timeseries": (nap.Tsd, nap.TsdFrame, nap.TsdTensor),
            "tref": (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor),
            "group": (nap.TsGroup,),
            "ep": (nap.IntervalSet,),
            "feature": (nap.Tsd, nap.TsdFrame, nap.TsdTensor),
            "binsize": (Number,),
            "windowsize": (tuple, Number),
            "minmax": (tuple, Number),
            "time_unit": (str,),
        }
        for param, param_type in parameters_type.items():
            if param in kwargs:
                if not isinstance(kwargs[param], param_type):
                    raise TypeError(
                        f"Invalid type. Parameter {param} must be of type {[p.__name__ for p in param_type]}."
                    )

        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


def _align_tsd(tsd, tref, window, new_time_support):
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
    window : tuple
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
            group[i] = nap.Ts(t=tmp, time_support=new_time_support)
    else:
        for i in range(len(tref)):
            tmp = tsd.index[lbounds[i] : rbounds[i]] - tref.index[i]
            tmp2 = tsd.values[lbounds[i] : rbounds[i]]
            group[i] = nap.Tsd(t=tmp, d=tmp2, time_support=new_time_support)

    group = nap.TsGroup(group, time_support=new_time_support, bypass_check=True)
    group.set_info(ref_times=tref.index)

    return group


@_validate_perievent_inputs
def compute_perievent(timestamps, tref, minmax, time_unit="s", **kwargs):
    """
    Center the timestamps of a time series object or a time series group around the timestamps given by the `tref` argument.
    `minmax` indicates the start and end of the window. If `minmax=(-5, 10)`, the window will be from -5 second to 10 second.
    If `minmax=10`, the window will be from -10 second to 10 second.

    To center the values of a time series around a set of timestamps, you can use `compute_perievent_continuous`.

    Parameters
    ----------
    timestamps : Ts, Tsd, TsdFrame, TsdTensor or TsGroup
        The timestamps to align to tref.
        If Ts/Tsd/TsdFrame/TsdTensor, returns a TsGroup.
        If TsGroup, returns a dictionary of TsGroup
    tref : Ts, Tsd, TsdFrame or TsdTensor.
        The time reference of the event to align to
    minmax : tuple of int/float or int or float
        The window size. Can be unequal on each side i.e. (-500, 1000).
    time_unit : str, optional
        Time units of the minmax ('s' [default], 'ms', 'us').

    Returns
    -------
    dict
        A TsGroup if timestamps is a Ts/Tsd/TsdFrame/TsdTensor or
        a dictionary of TsGroup if timestamps is a TsGroup.

    Raises
    ------
    RuntimeError
        If `time_unit` not in ["s", "ms", "us"]
        If `minmax` is wrongly defined
    """
    if time_unit not in ["s", "ms", "us"]:
        raise RuntimeError("time_unit should be 's', 'ms' or 'us'")

    if isinstance(minmax, Number):
        minmax = np.array([minmax, minmax], dtype=np.float64)

    if len(minmax) != 2:
        raise RuntimeError("minmax should be a tuple of 2 numbers or a single number.")

    if not all([isinstance(x, Number) for x in minmax]):
        raise RuntimeError("minmax should be a tuple of 2 numbers or a single number.")

    window = np.abs(nap.TsIndex.format_timestamps(np.array(minmax), time_unit))

    new_time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if isinstance(timestamps, nap.TsGroup):
        toreturn = {}
        for n in timestamps.index:
            toreturn[n] = _align_tsd(timestamps[n], tref, window, new_time_support)
        return toreturn
    else:
        return _align_tsd(timestamps, tref, window, new_time_support)


@_validate_perievent_inputs
def compute_perievent_continuous(
    timeseries, tref, minmax, ep=None, time_unit="s", **kwargs
):
    """
    Center continuous time series around the timestamps given by the 'tref' argument.
    `minmax` indicates the start and end of the window. If `minmax=(-5, 10)`, the window will be from -5 second to 10 second.
    If `minmax=10`, the window will be from -10 second to 10 second.

    To realign timestamps around a set of timestamps, you can use `compute_perievent`.

    This function assumes a constant sampling rate of the time series.

    Parameters
    ----------
    timeseries : Tsd, TsdFrame or TsdTensor
        The time series to align to tref.
    tref : Ts, Tsd, TsdFrame or TsdTensor
        The time reference of the event to align to
    minmax : tuple of int/float or int or float
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
        If `time_unit` not in ["s", "ms", "us"]
    """
    if time_unit not in ["s", "ms", "us"]:
        raise RuntimeError("time_unit should be 's', 'ms' or 'us'")

    if isinstance(minmax, Number):
        minmax = np.array([minmax, minmax], dtype=np.float64)

    if len(minmax) != 2:
        raise RuntimeError("minmax should be a tuple of 2 numbers or a single number.")

    if not all([isinstance(x, Number) for x in minmax]):
        raise RuntimeError("minmax should be a tuple of 2 numbers or a single number.")

    if ep is None:
        ep = timeseries.time_support

    window = np.abs(nap.TsIndex.format_timestamps(np.array(minmax), time_unit))

    time_array = timeseries.index.values
    data_array = timeseries.values
    time_target_array = tref.index.values
    starts = ep.start
    ends = ep.end

    bin_size = time_array[1] - time_array[0]
    idx1 = -np.arange(0, window[0] + bin_size, bin_size)[::-1][:-1]
    idx2 = np.arange(0, window[1] + bin_size, bin_size)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    minmax = np.array([idx1.shape[0], idx2.shape[0]])

    new_data_array = _perievent_continuous(
        time_array, data_array, time_target_array, starts, ends, minmax
    )

    time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if new_data_array.ndim == 2:
        return nap.TsdFrame(t=time_idx, d=new_data_array, time_support=time_support)
    else:
        return nap.TsdTensor(t=time_idx, d=new_data_array, time_support=time_support)


@_validate_perievent_inputs
def compute_event_trigger_average(
    group,
    feature,
    binsize,
    windowsize=0,
    ep=None,
    time_unit="s",
):
    """
    Bin the event timestamps within bin_size and compute the Event-Triggered Average (ETA) within `windowsize`.
    If C is the event count matrix and `feature` is a Tsd array, the function computes
    the Hankel matrix H from windowsize=(-t1,+t2) by offseting the Tsd array.

    The ETA is then defined as the dot product between H and C divided by the number of events.

    The object `feature` can be any dimensions.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects that hold the trigger time.
    feature : Tsd, TsdFrame or TsdTensor
        The feature to average.
    binsize : float or int
        The bin size. Default is second.
        If different, specify with the parameter time_unit ('s' [default], 'ms', 'us').
    windowsize : tuple of float/int or float/int, optional
        The window size. Default is second. For example windowsize = (-1, 1) is equivalent to windowsize = 1
        If different, specify with the parameter time_unit ('s' [default], 'ms', 'us').
        Default is (0, 0)
    ep : IntervalSet, optional
        The epochs on which the average is computed. If None, the time support of the feature is used.
    time_unit : str, optional
        The time unit of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    """
    if time_unit not in ["s", "ms", "us"]:
        raise RuntimeError("time_unit should be 's', 'ms' or 'us'")

    if isinstance(windowsize, Number):
        windowsize = np.array([windowsize, windowsize], dtype=np.float64)

    if len(windowsize) != 2:
        raise RuntimeError(
            "windowsize should be a tuple of 2 numbers or a single number."
        )

    if not all([isinstance(x, Number) for x in windowsize]):
        raise RuntimeError(
            "windowsize should be a tuple of 2 numbers or a single number."
        )

    if ep is None:
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

    # eta = np.zeros((time_idx.shape[0], len(group), *feature.shape[1:]))

    windows = np.array([len(idx1), len(idx2)])

    # Bin the spike train
    count = group.count(binsize, ep)

    time_target_array = np.round(count.index.values - (binsize / 2), 9)
    count_array = count.values
    starts = ep.start
    ends = ep.end

    time_array = feature.index.values
    data_array = feature.values

    eta = _perievent_trigger_average(
        time_target_array,
        count_array,
        time_array,
        data_array,
        starts,
        ends,
        windows,
        binsize,
    )

    if eta.ndim == 2:
        return nap.TsdFrame(t=time_idx, d=eta, columns=group.index)
    else:
        return nap.TsdTensor(t=time_idx, d=eta)
