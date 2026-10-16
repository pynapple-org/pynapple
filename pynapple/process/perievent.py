"""
Functions to realign time series relative to a reference time.
"""

from __future__ import annotations

from numbers import Number
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numba import jit, prange

from .. import core as nap
from ._process_functions import _perievent_continuous, _perievent_triggered_average


def compute_event_triggered_average(
    data: Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor],
    events: Union[
        nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup, nap.IntervalSet
    ],
    binsize: float,
    window: Union[float, Tuple[float, float]] = 0,
    time_unit: str = "s",
    epochs: Optional[nap.IntervalSet] = None,
) -> Union[nap.TsdFrame, nap.TsdTensor]:
    """
    Bin the event timestamps and compute the Event-Triggered Average (ETA).

    Notes
    -----
    If `C` is the event count matrix and ``feature`` is a `Tsd`, then the function computes
    the Hankel matrix H from ``window=(-t1,+t2)`` by offsetting the `Tsd`.
    The ETA is then defined as the dot product between `H` and `C` divided by the number of events.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The continuous timeseries to use as the feature (must be regularly sampled).
    events : Ts, Tsd, TsdFrame, TsdTensor, TsGroup, or IntervalSet
        The events (or spike trains) to align to. If a ``TsGroup``, each unit's
        spikes are binned and used as a separate event count vector.
        If an ``IntervalSet``, the start of each interval is used as an event.
        For other time-series objects, only the timestamps are used.
    binsize : float or int
        The bin size.
    window : int, float, or tuple
        The alignment window. Can be unequal on each side, e.g. ``(-2, 1)``.
    time_unit : str, optional
        Time units of the ``window`` and ``binsize`` ('s' [default], 'ms', 'us').
    epochs : IntervalSet, optional
        The epochs to perform the operation over. If None, uses the data's ``time_support``.

    Returns
    -------
    TsdFrame or TsdTensor
        The event-triggered average.

    Raises
    ------
    RuntimeError
        If ``time_unit`` is invalid, ``window`` format is invalid, or ``data`` is not
        regularly sampled.
    TypeError
        If ``data`` or ``events`` have unsupported types.

    Examples
    --------
    Compute the ETA of a continuous feature triggered on a single event train:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> times = np.arange(0, 10, 0.1)
    >>> feature = nap.Tsd(t=times, d=np.sin(times))
    >>> events = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])
    >>> eta = nap.compute_event_triggered_average(feature, events, binsize=0.1, window=0.5)
    >>> eta
    Time (s)            0
    ----------  ---------
    -0.5        0.0788719
    -0.4        0.0994986
    -0.3        0.119131
    -0.2        0.137573
    -0.1        0.154641
    0           0.170163
    0.1         0.183986
    0.2         0.19597
    0.3         0.205995
    0.4         0.213963
    0.5         0.219793
    dtype: float64, shape: (11, 1)

    You can restrict the computation to certain parts of the data by passing ``epochs``:

    >>> epochs = nap.IntervalSet(start=[0, 5], end=[4, 9])
    >>> eta = nap.compute_event_triggered_average(feature, events, binsize=0.1, window=0.5, epochs=epochs)
    >>> eta
    Time (s)           0
    ----------  --------
    -0.5        0.323254
    -0.4        0.347921
    -0.3        0.369112
    -0.2        0.386614
    -0.1        0.400254
    0           0.170163
    0.1         0.183986
    0.2         0.19597
    0.3         0.205995
    0.4         0.213963
    0.5         0.219793
    dtype: float64, shape: (11, 1)

    Compute the ETA triggered on multiple spike trains (``TsGroup``), returning one
    column per unit:

    >>> unit1 = nap.Ts(t=[1.0, 3.0, 5.0])
    >>> unit2 = nap.Ts(t=[2.0, 4.0, 6.0])
    >>> spikes = nap.TsGroup({0: unit1, 1: unit2})
    >>> eta = nap.compute_event_triggered_average(feature, spikes, binsize=0.1, window=0.5)
    >>> eta
    Time (s)              0           1
    ----------  -----------  ----------
    -0.5         0.0334559   -0.0196095
    -0.4         0.0288176   -0.0247378
    -0.3         0.0238914   -0.029619
    -0.2         0.0187265   -0.0342041
    -0.1         0.0133745   -0.0384476
    0            0.00788891  -0.0423069
    0.1          0.00232445  -0.0457434
    0.2         -0.00326324  -0.0487229
    0.3         -0.00881832  -0.0512156
    0.4         -0.0142853   -0.0531966
    0.5         -0.0196095   -0.054646
    dtype: float64, shape: (11, 2)

    Compute the ETA of a multiple features (``TsdFrame``), returning a ``TsdTensor``
    with shape ``(n_time_bins, n_events, n_features)``:

    >>> values = np.column_stack([np.sin(times), np.cos(times)])
    >>> features = nap.TsdFrame(t=times, d=values, columns=["sin", "cos"])
    >>> eta = nap.compute_event_triggered_average(features, events, binsize=0.1, window=0.5)
    >>> eta
    Time (s)
    ----------  --------------------------
    -0.5        [[0.078872, 0.210558] ...]
    -0.4        [[0.099499, 0.201632] ...]
    -0.3        [[0.119131, 0.190691] ...]
    -0.2        [[0.137573, 0.177845] ...]
    -0.1        [[0.154641, 0.163222] ...]
    0           [[0.170163, 0.146969] ...]
    0.1         [[0.183986, 0.129246] ...]
    0.2         [[0.19597 , 0.110233] ...]
    0.3         [[0.205995, 0.090118] ...]
    0.4         [[0.213963, 0.069102] ...]
    0.5         [[0.219793, 0.047396] ...]
    dtype: float64, shape: (11, 1, 2)
    """
    if time_unit not in ["s", "ms", "us"]:
        raise RuntimeError("time_unit should be 's', 'ms' or 'us'")

    if isinstance(window, Number):
        window = np.array([window, window], dtype=np.float64)
    if len(window) != 2 or not all(isinstance(x, Number) for x in window):
        raise RuntimeError("window should be a tuple of 2 numbers or a single number.")

    if not isinstance(data, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError(
            f"data should be a continuous time series (Tsd, TsdFrame, or TsdTensor): {type(data)}"
        )

    if isinstance(events, nap.IntervalSet):
        events = nap.Ts(events.start)

    if not isinstance(
        events, (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup)
    ):
        raise TypeError(
            f"events should be a time series object or an IntervalSet: {type(events)}"
        )

    if epochs is None:
        epochs = data.time_support

    if not nap.utils._is_regularly_sampled(data):
        raise RuntimeError(
            "Continuous data must be regularly sampled. "
            "Please interpolate or NaN-pad your data to a uniform time grid before "
            "calling compute_event_triggered_average."
        )

    window = np.abs(nap.TsIndex.format_timestamps(np.array(window), time_unit))
    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_unit
    )[0]

    # Build the time index for the output
    idx1 = -np.arange(0, window[0] + binsize, binsize)[::-1][:-1]
    idx2 = np.arange(0, window[1] + binsize, binsize)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    windows = np.array([len(idx1), len(idx2)], dtype=np.int64)

    # Bin the events
    count = events.count(binsize, epochs)

    time_target_array = np.round(count.index.values - (binsize / 2), 9)
    count_array = count.values
    if count_array.ndim == 1:
        count_array = count_array[:, np.newaxis]
    starts = epochs.start
    ends = epochs.end

    time_array = data.index.values
    data_array = data.values

    eta = _perievent_triggered_average(
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
        columns = events.index if isinstance(events, nap.TsGroup) else None
        return nap.TsdFrame(t=time_idx, d=eta, columns=columns)
    else:
        return nap.TsdTensor(t=time_idx, d=eta)


def compute_spike_triggered_average(
    data: Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor],
    spikes: Union[nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup],
    binsize: float,
    window: Union[float, Tuple[float, float]] = 0,
    time_unit: str = "s",
    epochs: Optional[nap.IntervalSet] = None,
) -> Union[nap.TsdFrame, nap.TsdTensor]:
    """
    Alias for :func:`~pynapple.process.perievent.compute_event_triggered_average` with ``spikes`` as the events argument.

    Parameters
    ----------
    data : Tsd, TsdFrame, or TsdTensor
        The continuous timeseries to use as the feature (must be regularly sampled).
    spikes : Ts, Tsd, TsdFrame, TsdTensor, or TsGroup
        The spike train(s) to align to.
    binsize : float or int
        The bin size.
    window : int, float, or tuple
        The alignment window. Can be unequal on each side, e.g. ``(-2, 1)``.
    time_unit : str, optional
        Time units of the ``window`` and ``binsize`` ('s' [default], 'ms', 'us').
    epochs : IntervalSet, optional
        The epochs to perform the operation over. If None, uses the data's ``time_support``.

    Returns
    -------
    TsdFrame or TsdTensor
        The spike-triggered average.

    Raises
    ------
    RuntimeError
        If ``time_unit`` is invalid, ``window`` format is invalid, or ``data`` is not
        regularly sampled.
    TypeError
        If ``data`` or ``spikes`` are not valid timeseries objects.

    See Also
    --------
    :func:`~pynapple.process.perievent.compute_event_triggered_average`
    """
    return compute_event_triggered_average(
        data=data,
        events=spikes,
        binsize=binsize,
        window=window,
        time_unit=time_unit,
        epochs=epochs,
    )


def compute_perievent(
    data: Union[nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup],
    events: Union[nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor],
    window: Union[float, Tuple[float, float]],
    time_unit: str = "s",
    epochs: Optional[nap.IntervalSet] = None,
) -> Union[
    nap.TsGroup,
    nap.TsdFrame,
    nap.TsdTensor,
    Dict[int, Union[nap.TsGroup, nap.TsdFrame, nap.TsdTensor]],
]:
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
        The alignment window, which can be unequal on each side, e.g. ``(-2, 1)``
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
      Index    rate    events
    -------  ------  --------
          0       1         1
          1       1         3
    >>> result[0]
    Time (s)
    -0.5
    0.19999999999999996
    shape: 2
    >>> result[1]
    Time (s)
    -0.8999999999999999
    0.5
    shape: 2

    Align data to events within specific epochs:

    >>> spikes = nap.Ts(t=[0.5, 1.2, 2.1, 3.5, 4.8])
    >>> events = nap.Ts(t=[1.0, 3.0, 5.0])
    >>> epochs = nap.IntervalSet(start=[0, 2.5], end=[2.4, 4.5])
    >>> result = nap.compute_perievent(spikes, events, window=1.0, epochs=epochs)
    >>> result # only 2 events are included!
      Index    rate    events
    -------  ------  --------
          0       1         1
          1       1         3

    Align with unequal window:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> spikes = nap.Ts(t=[0.5, 1.2, 2.1, 3.5])
    >>> events = nap.Ts(t=[1.0, 3.0])
    >>> result = nap.compute_perievent(spikes, events, window=(-0.5, 1.5))
    >>> result[0]  # window from -0.5 to +1.5 relative to first event
    Time (s)
    -0.5
    0.19999999999999996
    1.1
    shape: 3

    Align multiple discrete objects (``TsGroup``):

    >>> unit1 = nap.Ts(t=[0.5, 1.2, 2.1, 3.5])
    >>> unit2 = nap.Ts(t=[0.8, 1.5, 3.2, 4.1])
    >>> spikes = nap.TsGroup({0: unit1, 1: unit2})
    >>> events = nap.Ts(t=[1.0, 3.0])
    >>> result = nap.compute_perievent(spikes, events, window=1.0)
    >>> result
    {0:   Index    rate    events
    -------  ------  --------
          0       1         1
          1       1         3, 1:   Index    rate    events
    -------  ------  --------
          0     1           1
          1     0.5         3}
    >>> result[0] # aligned unit 0
      Index    rate    events
    -------  ------  --------
          0       1         1
          1       1         3
    >>> result[1] # aligned unit 1
      Index    rate    events
    -------  ------  --------
          0     1           1
          1     0.5         3

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

    if isinstance(data, nap.TsGroup):
        # Members with values align as continuous data: one unit at a time
        if data._data is not None and np.any(data._is_tsd):
            return {
                i: compute_perievent(data[i], events, window, time_unit, epochs)
                for i in data
            }
    elif not isinstance(data, (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError(f"data should be a time series object: {type(data)}")

    if not isinstance(events, (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        raise TypeError(f"events should be a time series object: {type(events)}")

    if epochs is None:
        epochs = data.time_support

    window = np.abs(nap.TsIndex.format_timestamps(np.array(window), time_unit))
    events = events.restrict(epochs)
    new_time_support = nap.IntervalSet(start=-window[0], end=window[1])

    if isinstance(data, nap.TsGroup):
        return _align_group(data, events, window)

    if isinstance(data, nap.Ts):
        # a Ts aligns as a group of one unit
        event_times = events.times()
        relative_times, event_idx, _ = _align_sorted(
            data.t, np.zeros(len(data), dtype=np.int64), 1, event_times, window
        )
        return _aligned_tsgroup(
            relative_times, event_idx, event_times, new_time_support
        )

    if not nap.utils._is_regularly_sampled(data):
        raise RuntimeError(
            "Continuous data must be regularly sampled. "
            "Please interpolate or NaN-pad your data to a uniform time grid before "
            "calling compute_perievent."
        )

    return _align_regular(data, events, window, new_time_support)


def _align_sorted(times, unit_pos, n_units, event_times, window):
    """
    Align the spikes of all the units to the events in one pass.

    ``times`` holds the spikes of every unit in one sorted array, as a
    ``TsGroup`` stores them. For each event, two binary searches find the
    spikes in ``[event - window[0], event + window[1]]``. Both bounds are
    included. The subtraction ``t - event`` can give a relative time just
    outside ``[-window[0], window[1]]``, by the smallest step between two
    floats. These times are dropped, as a ``Ts`` on the output support (the
    window) drops them. Spikes in overlapping windows are aligned to each of
    their events.

    Parameters
    ----------
    times : numpy.ndarray
        Spike times of all the units, sorted.
    unit_pos : numpy.ndarray
        Position of the unit of each spike, in ``0..n_units-1``.
    n_units : int
        Number of units.
    event_times : numpy.ndarray
        Event times.
    window : numpy.ndarray
        ``[before, after]`` window, both positive.

    Returns
    -------
    relative_times : numpy.ndarray
        Spike times relative to their event (``t - event``), sorted by unit,
        then by relative time, then by event.
    event_idx : numpy.ndarray
        Index in ``event_times`` of the event of each spike.
    offsets : numpy.ndarray
        ``relative_times[offsets[i]:offsets[i + 1]]`` are the spikes of unit ``i``.
    """
    relative_times, event_idx, offsets = _jitalign_sorted(
        times, unit_pos, n_units, event_times, window[0], window[1]
    )
    _jitsort_units(relative_times, event_idx, offsets)
    return relative_times, event_idx, offsets


@jit(nopython=True, cache=True)
def _jitalign_sorted(times, unit_pos, n_units, event_times, before, after):
    """Spikes in the window of each event, grouped by unit. In each unit,
    the spikes are sorted by event, then by time. See ``_align_sorted``."""
    lo = np.searchsorted(times, event_times - before, side="left")
    hi = np.searchsorted(times, event_times + after, side="right")

    # `t - event` can fall just outside the window, by the smallest step
    # between two floats (e.g. 1.4 - 1.7 gives -0.30000000000000004, not
    # -0.3). The output support is the window, and a Ts on that support
    # drops these times: drop them here too.

    # first pass: number of spikes of each unit
    counts = np.zeros(n_units, dtype=np.int64)
    for j in range(len(event_times)):
        for k in range(lo[j], hi[j]):
            relative_time = times[k] - event_times[j]
            if relative_time >= -before and relative_time <= after:
                counts[unit_pos[k]] += 1
    offsets = np.zeros(n_units + 1, dtype=np.int64)
    for i in range(n_units):
        offsets[i + 1] = offsets[i] + counts[i]

    # second pass: write each spike in the block of its unit
    relative_times = np.empty(offsets[-1], dtype=np.float64)
    event_idx = np.empty(offsets[-1], dtype=np.int64)
    pos = offsets[:-1].copy()
    for j in range(len(event_times)):
        for k in range(lo[j], hi[j]):
            relative_time = times[k] - event_times[j]
            if relative_time >= -before and relative_time <= after:
                u = unit_pos[k]
                relative_times[pos[u]] = relative_time
                event_idx[pos[u]] = j
                pos[u] += 1
    return relative_times, event_idx, offsets


@jit(nopython=True, cache=True, parallel=True)
def _jitsort_units(relative_times, event_idx, offsets):
    """Sort the block of each unit by relative time, in place. The sort is
    stable: spikes with the same relative time stay in event order. The
    units are independent, thus they are sorted in parallel."""
    for i in prange(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        order = np.argsort(relative_times[a:b], kind="mergesort") + a
        relative_times[a:b] = relative_times[order]
        event_idx[a:b] = event_idx[order]


def _event_windows(event_times, window):
    """
    The windows ``[event - window[0], event + window[1]]`` as an IntervalSet,
    for ``TsGroup._load_in_memory``: a lazy group then reads only the spikes
    that ``_align_sorted`` aligns.

    Overlapping or touching windows are merged first: an IntervalSet takes
    1 microsecond off bounds that touch. The merged bounds are the same
    floats as in ``_align_sorted``. An IntervalSet drops windows of zero
    length. Thus a zero window gives None, to read all the spikes.

    Spikes outside ``epochs`` but in a window are read, as in the regular
    group, which does not restrict the spikes to ``epochs``.
    """
    if window[0] + window[1] == 0:
        return None
    if len(event_times) == 0:
        return nap.IntervalSet(start=[], end=[])
    starts = event_times - window[0]
    ends = event_times + window[1]
    # events are sorted, thus starts and ends are sorted too: a window
    # starts a new block when it starts after the end of the previous one
    new_block = np.concatenate(([True], starts[1:] > ends[:-1]))
    last_of_block = np.concatenate((new_block[1:], [True]))
    return nap.IntervalSet(start=starts[new_block], end=ends[last_of_block])


def _align_group(data, events, window):
    """
    Align every unit of a ``TsGroup`` to events, with one ``_align_sorted``
    pass over the merged spikes.

    Gives the same output as aligning each unit as a ``Ts``, without
    building one ``Ts`` per unit and per event.

    Parameters
    ----------
    data : TsGroup
        Group with no values (all members are ``Ts``). A lazy group reads
        only the spikes in the event windows.
    events : Ts or other timeseries
        Event timestamps (pre-filtered by epochs)
    window : array-like
        [before, after] window

    Returns
    -------
    dict
        One TsGroup per unit, indexed by event number
    """
    event_times = events.times()
    data = data._load_in_memory(_event_windows(event_times, window))
    relative_times, event_idx, offsets = _align_sorted(
        data._times, data._cluster_positions, len(data.index), event_times, window
    )
    aligned = {}
    for i, key in enumerate(data):
        start, end = offsets[i], offsets[i + 1]
        aligned[key] = _aligned_tsgroup(
            relative_times[start:end],
            event_idx[start:end],
            event_times,
            nap.IntervalSet(start=-window[0], end=window[1]),
        )
    return aligned


def _aligned_tsgroup(relative_times, event_idx, event_times, time_support):
    """
    The aligned spikes of one unit as a TsGroup, with one member per event.

    Parameters
    ----------
    relative_times : numpy.ndarray
        Spike times relative to their event, sorted (from ``_align_sorted``).
    event_idx : numpy.ndarray
        Index in ``event_times`` of the event of each spike.
    event_times : numpy.ndarray
        Event times, kept in the ``events`` metadata column.
    time_support : IntervalSet
        The window, ``[-before, after]``.

    Returns
    -------
    TsGroup
        Aligned spikes, indexed by event number
    """
    n_events = len(event_times)
    return nap.TsGroup._from_arrays(
        relative_times,
        event_idx,
        None,
        np.zeros(n_events, dtype=bool),
        np.arange(n_events),
        time_support,
        metadata={"events": event_times},
    )


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
    bin_size = np.round(np.median(data.time_diff(align="start").values), 9)

    idx1 = -np.arange(0, window[0] + bin_size, bin_size)[::-1][:-1]
    idx2 = np.arange(0, window[1] + bin_size, bin_size)[1:]
    time_idx = np.hstack((idx1, np.zeros(1), idx2))
    windowsize = np.array([idx1.shape[0], idx2.shape[0]], dtype=np.int64)

    new_data_array = _perievent_continuous(
        data.t, data.values, events.times(), epochs.start, epochs.end, windowsize
    )

    if new_data_array.size == 0:
        raise ValueError(
            "No overlap between input and epochs, perievent matrix is empty..."
        )
    if new_data_array.ndim == 2:
        columns = events.index if isinstance(events, nap.TsGroup) else None
        return nap.TsdFrame(
            t=time_idx, d=new_data_array, time_support=new_time_support, columns=columns
        )
    else:
        return nap.TsdTensor(
            t=time_idx,
            d=new_data_array,
            time_support=new_time_support,
        )
