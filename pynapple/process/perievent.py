"""
Functions to realign time series relative to a reference time.
"""

from numbers import Number

import numpy as np

from .. import core as nap
from ._process_functions import _perievent_continuous, _perievent_trigger_average


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

    if not all(isinstance(x, Number) for x in windowsize):
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


def compute_perievent(data, events, window, time_unit="s", epochs=None):
    """
    Perievent alignment, handles both discrete and continuous data.

    This function automatically detects the data type, then applies the appropriate
    alignment strategy:

    - **discrete data:**
        - a single ``Ts``: returns a ``TsGroup`` with one element per event
        - multiple ``Ts`` in a ``TsGroup``: returns a dictionary with a ``TsGroup`` per unit
    - **continuous data:**
        - regularly sampled ``Tsd``/``TsdFrame``/``TsdTensor``: returns aligned object with
          events as columns and uniform time sampling
        - irregularly sampled data is not supported; interpolate or NaN-pad to a regular
          grid before calling this function

    Parameters
    ----------
    data : Ts, Tsd, TsdFrame, TsdTensor, TsGroup
        The timeseries to align.
    events : Ts, Tsd, TsdFrame, TsdTensor
        The events to align to.
        If not a ``Ts``, we simply take the timestamps of the object.
    window : int, float, tuple
        The alignment window, which can be unequal on each side, e.g. ``(-500, 1000)``
    time_unit : str, optional
        Time units of the window ('s' [default], 'ms', 'us').
    epochs : IntervalSet, optional
        The epochs to perform the operation over. If None, uses the data's ``time_support``.

    Returns
    -------
    TsGroup, TsdFrame, TsdTensor, dict
        The aligned timeseries.

    Raises
    ------
    RuntimeError
        If time_unit not in ["s", "ms", "us"], window format is invalid, or continuous
        data is not regularly sampled.

    Examples
    --------
    Align discrete data (``Ts``) to events:

    >>> import pynapple as nap
    >>> spikes = nap.Ts(t=[0.5, 1.2, 2.1, 3.5, 4.8])
    >>> events = nap.Ts(t=[1.0, 3.0])
    >>> result = nap.compute_perievent(spikes, events, window=1.0)
    >>> result
      Index    rate
    -------  ------
          0       1
          1       1
    >>> result[0]
    Time (s)
    -0.5
    0.2
    shape: 2
    >>> result[1]
    Time (s)
    -0.9
    0.5
    shape: 2

    Align data to events within specific epochs:

    >>> spikes = nap.Ts(t=[0.5, 1.2, 2.1, 3.5, 4.8])
    >>> events = nap.Ts(t=[1.0, 3.0, 5.0])
    >>> epochs = nap.IntervalSet(start=[0, 2.5], end=[2.4, 4.5])
    >>> result = nap.compute_perievent(spikes, events, window=1.0, epochs=epochs)
    >>> result # only 2 events are included!
      Index    rate
    -------  ------
          0       1
          1       1

    Align with unequal window:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> spikes = nap.Ts(t=[0.5, 1.2, 2.1, 3.5])
    >>> events = nap.Ts(t=[1.0, 3.0])
    >>> result = nap.compute_perievent(spikes, events, window=(-0.5, 1.5))
    >>> result[0]  # window from -0.5 to +1.5 relative to first event
    Time (s)
    -0.5
    0.2
    1.1
    shape: 3

    Align multiple discrete objects (``TsGroup``):

    >>> unit1 = nap.Ts(t=[0.5, 1.2, 2.1, 3.5])
    >>> unit2 = nap.Ts(t=[0.8, 1.5, 3.2, 4.1])
    >>> spikes = nap.TsGroup({0: unit1, 1: unit2})
    >>> events = nap.Ts(t=[1.0, 3.0])
    >>> result = nap.compute_perievent(spikes, events, window=1.0)
    >>> result
    {0:   Index    rate
    -------  ------
          0       1
          1       1, 1:   Index    rate
    -------  ------
          0     1
          1     0.5}
    >>> result[0] # aligned unit 0
      Index    rate
    -------  ------
          0       1
          1       1
    >>> result[1] # aligned unit 1
      Index    rate
    -------  ------
          0     1
          1     0.5

    Align regularly sampled continuous data (``Tsd``) to events, returning a ``TsdFrame``:

    >>> times = np.arange(0, 10, 0.5)
    >>> values = np.sin(times)
    >>> data = nap.Tsd(t=times, d=values)
    >>> events = nap.Ts(t=[2.0, 5.0, 8.0])
    >>> result = nap.compute_perievent(data, events, window=1.0)
    >>> result
    Time (s)           0          1         2
    ----------  --------  ---------  --------
    -1          0.841471  -0.756802  0.656987
    -0.5        0.997495  -0.97753   0.938
    0           0.909297  -0.958924  0.989358
    0.5         0.598472  -0.70554   0.798487
    1           0.14112   -0.279415  0.412118
    dtype: float64, shape: (5, 3)

    Align a regularly sampled continuous data (``TsdFrame``) to events, returning a ``TsdTensor``:

    >>> values = np.column_stack([np.sin(times), np.cos(times)])
    >>> data = nap.TsdFrame(t=times, d=values, columns=["sin", "cos"])
    >>> result = nap.compute_perievent(data, events, window=1.0)
    >>> result
    Time (s)
    ----------  ----------------------------
    -1          [[0.841471, 0.540302] ...]
    -0.5        [[0.997495, 0.070737] ...]
    0           [[ 0.909297, -0.416147] ...]
    0.5         [[ 0.598472, -0.801144] ...]
    1           [[ 0.14112 , -0.989992] ...]
    dtype: float64, shape: (5, 3, 2)
    """
    if time_unit not in ["s", "ms", "us"]:
        raise RuntimeError("time_unit should be 's', 'ms' or 'us'")

    if isinstance(window, Number):
        window = np.array([window, window], dtype=np.float64)
    if len(window) != 2 or not all(isinstance(x, Number) for x in window):
        raise RuntimeError("window should be a tuple of 2 numbers or a single number.")

    window = np.abs(nap.TsIndex.format_timestamps(np.array(window), time_unit))

    # Call recursively if data is a TsGroup
    if isinstance(data, nap.TsGroup):
        return {
            i: compute_perievent(data[i], events, window, time_unit, epochs)
            for i in data
        }

    if not isinstance(data, (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError(f"data should be a time series object: {type(data)}")

    if not isinstance(events, (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError(f"events should be a time series object: {type(events)}")

    if epochs is None:
        epochs = data.time_support

    events = events.restrict(epochs)
    new_time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if isinstance(data, nap.Ts):
        return _align_discrete(data, events, window, new_time_support)

    if not _is_regularly_sampled(data):
        raise RuntimeError(
            "Continuous data must be regularly sampled. "
            "Please interpolate or NaN-pad your data to a uniform time grid before "
            "calling compute_perievent."
        )

    return _align_regular(data, events, window, new_time_support)


def _is_regularly_sampled(data, tolerance=1e-6):
    """
    Check if a timeseries has regular sampling.

    Parameters
    ----------
    data : Ts, Tsd, TsdFrame, TsdTensor
        The timeseries to check
    tolerance : float
        Relative tolerance for bin size variation

    Returns
    -------
    bool
        True if sampling is regular (constant bin size)
    """
    if len(data) < 2:
        return True

    time_diffs = np.diff(data.t)
    bin_size = time_diffs[0]

    relative_variation = np.abs(time_diffs - bin_size) / bin_size
    return np.all(relative_variation < tolerance)


def _align_discrete(data, events, window, new_time_support):
    """
    Align a ``Ts`` to events, returning a TsGroup.

    Parameters
    ----------
    data : Ts
        Data to align
    events : Ts or other timeseries
        Event timestamps (pre-filtered by epochs)
    window : array-like
        [before, after] window
    new_time_support : IntervalSet
        Time support for output

    Returns
    -------
    TsGroup
        Aligned data, indexed by event number
    """
    aligned = {}
    event_times = events.times()

    for event_idx, event_time in enumerate(event_times):
        start_time = event_time - window[0]
        end_time = event_time + window[1]

        mask = (data.t >= start_time) & (data.t <= end_time)
        shifted_times = data.t[mask] - event_time

        aligned[event_idx] = nap.Ts(shifted_times, time_support=new_time_support)

    return nap.TsGroup(aligned)


def _align_regular(data, events, window, new_time_support):
    """
    Align regularly-sampled continuous data to events using optimized matrix approach.

    Returns a single TsdFrame/TsdTensor where each column is one event with uniform
    time sampling.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        Regularly-sampled continuous data
    events : Ts or other timeseries
        Event timestamps (pre-filtered by epochs)
    window : array-like
        [before, after] window
    new_time_support : IntervalSet
        Time support for output

    Returns
    -------
    TsdFrame or TsdTensor
        Aligned data with columns/slice 0 as events
    """
    epochs = data.time_support
    bin_size = data.t[1] - data.t[0]

    idx1 = -np.arange(0, window[0] + bin_size, bin_size)[::-1][:-1]
    idx2 = np.arange(0, window[1] + bin_size, bin_size)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    windowsize = np.array([idx1.shape[0], idx2.shape[0]], dtype=np.int64)

    new_data_array = _perievent_continuous(
        data.t, data.values, events.times(), epochs.start, epochs.end, windowsize
    )

    if new_data_array.ndim == 2:
        return nap.TsdFrame(t=time_idx, d=new_data_array, time_support=new_time_support)
    else:
        return nap.TsdTensor(
            t=time_idx, d=new_data_array, time_support=new_time_support
        )
