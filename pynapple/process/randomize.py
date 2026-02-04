"""
Functions to shuffle timestamps to create surrogate datasets.
"""

import warnings

import numpy as np

from .. import core as nap


def shift_timestamps(data, min_shift=0.0, max_shift=None, mode="drop"):
    """
    Shifts all the time stamps of a random amount between a minimum and maximum shift, wrapping the
    end of the time support to the beginning.

    Notes
    -----
    If the time support of the input has multiple epochs, some timepoints will fall outside of
    those epochs after shifting. In ``mode='drop'``, those timepoints are dropped. In
    ``mode='wrap'``, timestamps are wrapped circularly using the full time support.
    However, if there are multiple epochs and ``mode='wrap'``,
    timepoints falling outside the epochs in the middle are still dropped.

    Parameters
    ----------
    data : Ts, TsGroup
        The timeseries object whose timestamps to shift.
        If TsGroup, shifts all objects in the group independently.
    min_shift : float, optional
        minimum shift (default: 0)
    max_shift : float, optional
        maximum shift, (default: length of time support)
    mode : ``'drop'`` or ``'wrap'``, optional
        How to handle timestamps that fall outside the time support after shifting.

        * ``'drop'``: (default): drop those timestamps
        * ``'wrap'``: circularly wrap timestamps within the time support

    Returns
    -------
    Ts or TsGroup
        The randomly shifted timestamps

    Examples
    --------
    Fixed shift with default ``mode='drop'``:

        >>> import pynapple as nap
        >>> ts = nap.Ts([25, 27, 33.3, 34.5])
        >>> shifted_ts = nap.shift_timestamps(ts, min_shift=1, max_shift=1, mode="drop")
        >>> shifted_ts
        Time (s)
        26.0
        28.0
        34.3
        shape: 4

    The last timepoint falls outside the time support, so it is dropped.

    With multiple epochs, timestamps falling outside the support anywhere are dropped:

        >>> epochs = nap.IntervalSet(start=[25, 30], end=[27, 34.5])
        >>> ts = nap.Ts([25, 27, 33.3, 34.5], time_support=epochs)
        >>> shifted_ts = nap.shift_timestamps(ts, min_shift=1, max_shift=1, mode="drop")
        >>> shifted_ts
        Time (s)
        26.0
        34.3
        shape: 3

    Using ``mode='wrap'`` to circularly wrap timestamps within the full time support:

        >>> epochs = nap.IntervalSet(start=0, end=40)
        >>> ts = nap.Ts([38, 39.5], time_support=epochs)
        >>> shifted_ts = nap.shift_timestamps(ts, min_shift=5, max_shift=5, mode="wrap")
        >>> shifted_ts
        Time (s)
        3.0
        4.5
        shape: 2

    When ``mode='wrap'`` and there are multiple epochs, the start and end are circular, but everything in between is not:

        >>> import pynapple as nap
        >>> epochs = nap.IntervalSet(start=[25, 30], end=[27, 34.5])
        >>> ts = nap.Ts([25, 27, 33.3, 34.5], time_support=epochs)
        >>> shifted_ts = nap.shift_timestamps(ts, min_shift=1, max_shift=1, mode="wrap")
        >>> shifted_ts
        Time (s)
        26.0
        26.0
        34.3
        shape: 3

    """
    if not isinstance(min_shift, (int, float)):
        raise TypeError("min_shift should be a number.")

    if max_shift is not None and not isinstance(max_shift, (int, float)):
        raise TypeError("max_shift should be a number.")

    if mode not in ("drop", "wrap"):
        raise ValueError("mode must be either 'drop' or 'wrap'.")

    if not isinstance(data, (nap.Ts, nap.TsGroup)):
        raise TypeError("Invalid input, data should be a Ts or TsGroup.")

    time_support = data.time_support

    if max_shift is None:
        max_shift = time_support.tot_length()

    def _shift(data):
        shift = np.random.uniform(min_shift, max_shift)
        times = data.times()
        shifted = times + shift

        if mode == "wrap":
            start = time_support.start[0]
            end = time_support.end[-1]
            period = end - start
            shifted = start + ((shifted - start) % period)

        shifted = np.sort(shifted)
        return nap.Ts(t=shifted, time_support=time_support)

    if isinstance(data, nap.TsGroup):
        shifted = {}
        for k in data:
            if not isinstance(data[k], nap.Ts):
                warnings.warn(
                    f"TsGroup entry {k} was not a Ts, but treating it as one!",
                    UserWarning,
                )
            shifted[k] = _shift(data[k])
        return nap.TsGroup(shifted, time_support=time_support)
    else:
        return _shift(data)


# Random shuffle intervals between timestamps


def shuffle_ts_intervals(ts):
    """
    Randomizes the timestamps by shuffling the intervals between them.


    Parameters
    ----------
    ts : Ts or TsGroup
        The timestamps to randomize. If TsGroup, randomizes all Ts in the group independently.

    Returns
    -------
    Ts or TsGroup
        The randomized timestamps, with shuffled intervals
    """
    strategies = {
        nap.time_series.Ts: _shuffle_intervals_ts,
        nap.ts_group.TsGroup: _shuffle_intervals_tsgroup,
    }
    # checks input type
    if type(ts) not in strategies.keys():
        raise TypeError("Invalid input type, should be Ts or TsGroup")

    strategy = strategies[type(ts)]
    return strategy(ts)


# Random Jitter


def jitter_timestamps(ts, max_jitter=None, keep_tsupport=False):
    """
    Jitters each time stamp independently of random amounts uniformly drawn between -max_jitter and max_jitter.


    Parameters
    ----------
    ts : Ts or TsGroup
        The timestamps to jitter. If TsGroup, jitter is applied to each element of the group.
    max_jitter : float
        maximum jitter
    keep_tsupport: bool, optional
        If True, keep time support of the input. The number of timestamps will not be conserved.
        If False, the time support is inferred from the jittered timestamps. The number of tmestamps
        is conserved. (default: False)

    Returns
    -------
    Ts or TsGroup
        The jittered timestamps
    """
    strategies = {
        nap.time_series.Ts: _jitter_ts,
        nap.ts_group.TsGroup: _jitter_tsgroup,
    }
    # checks input type
    if type(ts) not in strategies.keys():
        raise TypeError("Invalid input type, should be Ts or TsGroup")

    if max_jitter is None:
        raise TypeError("missing required argument: max_jitter ")

    strategy = strategies[type(ts)]
    return strategy(ts, max_jitter, keep_tsupport)


# Random resample


def resample_timestamps(ts):
    """
    Resamples the timestamps in the time support, with uniform distribution.


    Parameters
    ----------
    ts : Ts or TsGroup
        The timestamps to resample. If TsGroup, each Ts object in the group is independently
        resampled, in the time support of the whole group.


    Returns
    -------
    Ts or TsGroup
        The resampled timestamps
    """
    strategies = {
        nap.time_series.Ts: _resample_ts,
        nap.ts_group.TsGroup: _resample_tsgroup,
    }
    # checks input type
    if type(ts) not in strategies.keys():
        raise TypeError("Invalid input type, should be Ts or TsGroup")

    strategy = strategies[type(ts)]
    return strategy(ts)


# Helper functions


def _jitter_ts(ts, max_jitter=None, keep_tsupport=False):
    """
    Parameters
    ----------
    ts : Ts
        The timestamps to jitter.
    max_jitter : float
        maximum jitter
    keep_tsupport: bool, optional
        If True, keep time support of the input. The number of timestamps will not be conserved.
        If False, the time support is inferred from the jittered timestamps. The number of tmestamps
        is conserved. (default: False)

    Returns
    -------
    Ts
        The jittered timestamps
    """
    jittered_timestamps = ts.times() + np.random.uniform(
        -max_jitter, max_jitter, len(ts)
    )
    if keep_tsupport:
        jittered_ts = nap.Ts(
            t=np.sort(jittered_timestamps), time_support=ts.time_support
        )
    else:
        jittered_ts = nap.Ts(t=np.sort(jittered_timestamps))

    return jittered_ts


def _jitter_tsgroup(tsgroup, max_jitter=None, keep_tsupport=False):
    """
    Jitters each time stamp independently, for each element in the TsGroup
    of random amounts uniformly drawn between -max_jitter and max_jitter.

    Parameters
    ----------
    tsgroup : TsGroup
        The timestamps to jitter, the jitter is applied to each element of the group.
    max_jitter : float
        maximum jitter
    keep_tsupport: bool, optional
        If True, keep time support of the input. The number of timestamps will not be conserved.
        If False, the time support is inferred from the jittered timestamps. The number of tmestamps
        is conserved. (default: False)

    Returns
    -------
    TsGroup
        The jittered timestamps
    """

    jittered_tsgroup = {}
    for k in tsgroup.keys():
        jittered_timestamps = tsgroup[k].times() + np.random.uniform(
            -max_jitter, max_jitter, len(tsgroup[k])
        )
        jittered_tsgroup[k] = nap.Ts(t=np.sort(jittered_timestamps))

    if keep_tsupport:
        jittered_tsgroup = nap.TsGroup(
            jittered_tsgroup, time_support=tsgroup.time_support
        )
    else:
        jittered_tsgroup = nap.TsGroup(jittered_tsgroup)

    return jittered_tsgroup


def _resample_ts(ts):
    """
    Resamples the timestamps in the time support, with uniform distribution.

    Parameters
    ----------
    ts : Ts
        The timestamps to resample.
    Returns
    -------
    Ts
        The resampled timestamps
    """
    resampled_timestamps = np.random.uniform(ts.start_time(), ts.end_time(), len(ts))
    resampled_ts = nap.Ts(t=np.sort(resampled_timestamps), time_support=ts.time_support)

    return resampled_ts


def _resample_tsgroup(tsgroup):
    """
    Resamples the each timestamp series in the group, with uniform distribution and on the time
    support of the whole group.

    Parameters
    ----------
    tsgroup : TsGroup
        The TsGroup to resample, each Ts object in the group is independently
        resampled, in the time support of the whole group.

    Returns
    -------
    TsGroup
        The resampled TsGroup
    """
    start_time = tsgroup.time_support.start[0]
    end_time = tsgroup.time_support.end[0]

    resampled_tsgroup = {}
    for k in tsgroup.keys():
        resampled_timestamps = np.random.uniform(start_time, end_time, len(tsgroup[k]))
        resampled_tsgroup[k] = nap.Ts(t=np.sort(resampled_timestamps))

    return nap.TsGroup(resampled_tsgroup, time_support=tsgroup.time_support)


def _shuffle_intervals_ts(ts):
    """
    Randomizes the timestamps by shuffling the intervals between them.

    Parameters
    ----------
    ts : Ts
        The timestamps to randomize.
    Returns
    -------
    Ts
        The timestamps with shuffled intervals
    """
    intervals = np.diff(ts.times())
    shuffled_intervals = np.random.permutation(intervals)
    start_time = ts.times()[0]
    randomized_timestamps = np.hstack(
        [start_time, start_time + np.cumsum(shuffled_intervals)]
    )
    randomized_ts = nap.Ts(t=randomized_timestamps)

    return randomized_ts


def _shuffle_intervals_tsgroup(tsgroup):
    """
    Randomizes the timestamps by shuffling the intervals between them.
    Each Ts in the group is randomized independently

    Parameters
    ----------
    tsgroup : TsGroup
        The TsGroup to randomize.
    Returns
    -------
    tsGroup
        The TsGroup with shuffled intervals.
    """
    randomized_tsgroup = {k: _shuffle_intervals_ts(tsgroup[k]) for k in tsgroup.keys()}

    return nap.TsGroup(randomized_tsgroup)
