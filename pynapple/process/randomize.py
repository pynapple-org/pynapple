import numpy as np

from .. import core as nap

# Random shift


def shift_timestamps(ts, min_shift=0.0, max_shift=None):
    """
    Shifts all the time stamps of a random amount between min_shift and max_shift, wrapping the
    end of the time support to the beginning.


    Parameters
    ----------
    ts : Ts or TsGroup
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
    strategies = {
        nap.time_series.Ts: _shift_ts,
        nap.ts_group.TsGroup: _shift_tsgroup,
    }
    # checks input type
    if type(ts) not in strategies.keys():
        raise TypeError("Invalid input type, should be Ts or TsGroup")

    strategy = strategies[type(ts)]
    return strategy(ts, min_shift, max_shift)


# Random shuffle intervals between timestamps


def shuffle_ts_intervals(ts, min_shift=0.0, max_shift=None):
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


def _shift_ts(ts, min_shift=0, max_shift=None):
    """
    Shifts all the time stamps of a random amount between min_shift and max_shift, wrapping the
    end of the time support to the beginning.


    Parameters
    ----------
    ts : Ts
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

    if max_shift is None:
        max_shift = ts.end_time() - ts.start_time()
    shift = np.random.uniform(min_shift, max_shift)
    shifted_timestamps = (ts.times() + shift) % ts.end_time() + ts.start_time()
    shifted_ts = nap.Ts(t=np.sort(shifted_timestamps), time_support=ts.time_support)
    return shifted_ts


def _shift_tsgroup(tsgroup, min_shift=0, max_shift=None):
    """
    Shifts each Ts in the Ts group independently.


    Parameters
    ----------
    tsgroup : TsGroup
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

    if max_shift is None:
        max_shift = end_time - start_time

    shifted_tsgroup = {}
    for k in tsgroup.keys():
        shift = np.random.uniform(min_shift, max_shift)
        shifted_timestamps = (tsgroup[k].times() + shift) % end_time + start_time
        shifted_tsgroup[k] = nap.Ts(t=np.sort(shifted_timestamps))
    return nap.TsGroup(shifted_tsgroup, time_support=tsgroup.time_support)


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
