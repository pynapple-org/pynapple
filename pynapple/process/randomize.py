"""
Functions to shuffle timestamps to create surrogate datasets.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

from .. import core as nap
from ..core._core_functions import _restrict_arrays, _sort_merged


def shift_timestamps(
    data: Union[nap.Ts, nap.TsGroup],
    min_shift: float = 0.0,
    max_shift: Optional[float] = None,
    mode: str = "drop",
) -> Union[nap.Ts, nap.TsGroup]:
    """
    Shift all the timestamps by one random amount.

    The amount comes from a uniform distribution between ``min_shift`` and
    ``max_shift``. All the timestamps of a Ts get the same amount. For a
    TsGroup, each unit gets a different amount.

    The random numbers come from the global numpy generator. To get the same
    result again, call ``np.random.seed`` before the function.

    Parameters
    ----------
    data : Ts or TsGroup
        The timestamps to shift.
    min_shift : float, optional
        The minimum amount, in seconds. The default is 0.
    max_shift : float, optional
        The maximum amount, in seconds. The default is the total length of
        the epochs of the time support.
    mode : ``'drop'`` or ``'wrap'``, optional
        What to do with the timestamps that move out of the time support:

        * ``'drop'`` (default): remove them.
        * ``'wrap'``: move them back by the span of the time support. The
          span goes from the start of the first epoch to the end of the last
          epoch.

    Returns
    -------
    Ts or TsGroup
        The shifted timestamps, on the time support of ``data``. A TsGroup
        output has the same keys as ``data``, and no metadata other than
        ``rate``.

    Raises
    ------
    TypeError
        - If ``data`` is not a Ts or a TsGroup.
        - If ``min_shift`` or ``max_shift`` is not a number.
    ValueError
        If ``mode`` is not ``'drop'`` or ``'wrap'``.

    Warns
    -----
    UserWarning
        For each unit of a TsGroup that holds values. The function shifts its
        timestamps and does not keep its values.

    Notes
    -----
    If the time support has more than one epoch, a timestamp can move into
    a gap between two epochs. The function removes these timestamps in the
    two modes, because ``'wrap'`` uses the span of the time support and not
    its epochs.

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
        shape: 3

    The last timepoint falls outside the time support, so it is dropped.

    With multiple epochs, timestamps falling outside the support anywhere are dropped:

        >>> epochs = nap.IntervalSet(start=[25, 30], end=[27, 34.5])
        >>> ts = nap.Ts([25, 27, 33.3, 34.5], time_support=epochs)
        >>> shifted_ts = nap.shift_timestamps(ts, min_shift=1, max_shift=1, mode="drop")
        >>> shifted_ts
        Time (s)
        26.0
        34.3
        shape: 2

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

    if isinstance(data, nap.Ts):
        shifted = data.times() + np.random.uniform(min_shift, max_shift)
    else:
        # a lazy group reads all its spikes here: every spike is needed
        data = data._load_in_memory()
        for key in data.index[data._is_tsd]:
            warnings.warn(
                f"TsGroup entry {key} was not a Ts, but treating it as one!",
                UserWarning,
            )
        order, offsets = data._ragged_index
        unit_sizes = np.diff(offsets)
        # one shift per unit, in key order, as for a loop over the units
        shifts = np.random.uniform(min_shift, max_shift, len(data.index))
        shifted = data._times[order] + np.repeat(shifts, unit_sizes)

    if mode == "wrap":
        start = time_support.start[0]
        period = time_support.end[-1] - start
        shifted = start + ((shifted - start) % period)

    if isinstance(data, nap.Ts):
        # the timestamps out of the time support are dropped
        return nap.Ts(t=np.sort(shifted), time_support=time_support)

    times, clusters = _sort_merged(shifted, np.repeat(data.index, unit_sizes))
    times, clusters = _restrict_arrays(
        times, time_support.start, time_support.end, clusters
    )
    return nap.TsGroup._from_arrays(
        times,
        clusters,
        None,
        np.zeros(len(data.index), dtype=bool),
        data.index,
        time_support,
    )


# Random shuffle intervals between timestamps


def shuffle_ts_intervals(
    ts: Union[nap.Ts, nap.TsGroup],
) -> Union[nap.Ts, nap.TsGroup]:
    """
    Shuffle the intervals between consecutive timestamps.

    The function puts the epochs of the time support end to end, without the
    gaps between them. On this axis, it computes the intervals between
    consecutive timestamps, puts them in a random order, and adds them again
    from the first timestamp. Then it puts the timestamps back on the epochs.

    The intervals move freely across all the epochs. An interval that crosses
    a gap counts only the time in the epochs. No timestamp falls in a gap. The
    output keeps the first timestamp, the number of timestamps and the
    distribution of the intervals.

    The random numbers come from the global numpy generator. To get the same
    result again, call ``np.random.seed`` before the function.

    Parameters
    ----------
    ts : Ts or TsGroup
        The timestamps to shuffle. For a TsGroup, the function shuffles the
        intervals of each unit separately.

    Returns
    -------
    Ts or TsGroup
        The shuffled timestamps, on the time support of ``ts``. A TsGroup
        output has the same keys as ``ts``, and no metadata other than
        ``rate``.

    Raises
    ------
    TypeError
        If ``ts`` is not a Ts or a TsGroup.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> ts = nap.Ts([1.0, 2.0, 4.0, 8.0])
    >>> np.random.seed(0)
    >>> shuffled = nap.shuffle_ts_intervals(ts)
    >>> float(shuffled.t[0]), float(shuffled.t[-1])
    (1.0, 8.0)
    >>> np.sort(np.diff(shuffled.t))
    array([1., 2., 4.])

    With two epochs, the gap between the epochs is not an interval:

    >>> epochs = nap.IntervalSet(start=[0, 100], end=[10, 110])
    >>> ts = nap.Ts([1.0, 2.0, 3.0, 101.0, 102.0], time_support=epochs)
    >>> shuffled = nap.shuffle_ts_intervals(ts)
    >>> len(shuffled.restrict(epochs)) == len(ts)
    True
    """
    if not isinstance(ts, (nap.Ts, nap.TsGroup)):
        raise TypeError("Invalid input type, should be Ts or TsGroup")

    if isinstance(ts, nap.Ts):
        times = ts.times()
        offsets = np.array([0, len(times)])
    else:
        # a lazy group reads all its spikes here: every spike is needed
        ts = ts._load_in_memory()
        order, offsets = ts._ragged_index
        # the spikes of each unit one after the other, in time order
        times = ts._times[order]

    # position of each time on the epochs put end to end
    time_support = ts.time_support
    starts, ends = time_support.start, time_support.end
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(ends - starts)])
    epochs = np.searchsorted(starts, times, side="right") - 1
    positions = times - starts[epochs] + cumulative_lengths[epochs]

    # one permutation per unit, in key order, as for a loop over the units.
    # The spikes of a unit are at positions[unit_start:unit_stop]
    for unit_start, unit_stop in zip(offsets[:-1], offsets[1:]):
        if unit_stop > unit_start:
            intervals = np.random.permutation(np.diff(positions[unit_start:unit_stop]))
            positions[unit_start + 1 : unit_stop] = positions[unit_start] + np.cumsum(
                intervals
            )

    # back on the epochs. Rounding can move a time past the end of its epoch
    # by the smallest step between two floats
    epochs = np.searchsorted(cumulative_lengths, positions, side="right") - 1
    epochs = np.minimum(epochs, len(starts) - 1)
    times = np.minimum(
        starts[epochs] + (positions - cumulative_lengths[epochs]), ends[epochs]
    )

    if isinstance(ts, nap.Ts):
        return nap.Ts(t=times, time_support=time_support)

    times, clusters = _sort_merged(times, np.repeat(ts.index, np.diff(offsets)))
    return nap.TsGroup._from_arrays(
        times,
        clusters,
        None,
        np.zeros(len(ts.index), dtype=bool),
        ts.index,
        time_support,
    )


# Random Jitter


def jitter_timestamps(
    ts: Union[nap.Ts, nap.TsGroup],
    max_jitter: Optional[float] = None,
    keep_tsupport: bool = False,
) -> Union[nap.Ts, nap.TsGroup]:
    """
    Move each timestamp by a random amount.

    Each amount comes from a uniform distribution between ``-max_jitter`` and
    ``max_jitter``. Each timestamp gets a different amount. The function sorts
    the timestamps again after the move.

    The random numbers come from the global numpy generator. To get the same
    result again, call ``np.random.seed`` before the function.

    Parameters
    ----------
    ts : Ts or TsGroup
        The timestamps to move. For a TsGroup, the function moves the
        timestamps of all the units.
    max_jitter : float
        The maximum amount of a move, in seconds.
    keep_tsupport : bool, optional
        - False (default): the output gets a new time support. For a Ts, the
          time support goes from the first timestamp to the last timestamp.
          For a TsGroup, the time support is the union of these spans for
          all the units.
        - True: the output keeps the time support of ``ts``. The function
          removes the timestamps that move out of this time support.

    Returns
    -------
    Ts or TsGroup
        The moved timestamps. A TsGroup output has the same keys as ``ts``,
        and no metadata other than ``rate``.

    Raises
    ------
    TypeError
        - If ``ts`` is not a Ts or a TsGroup.
        - If ``max_jitter`` is not given.
    RuntimeError
        If ``ts`` is a TsGroup, ``keep_tsupport`` is False, and no unit has
        two timestamps or more. The new time support is then empty.

    Notes
    -----
    With ``keep_tsupport=False``, a TsGroup can lose timestamps. A unit with
    one timestamp has no span. If its timestamp is not in the span of
    another unit, the function removes it.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> ts = nap.Ts([1.0, 2.0, 3.0])
    >>> np.random.seed(0)
    >>> jittered = nap.jitter_timestamps(ts, max_jitter=0.1)
    >>> len(jittered)
    3

    The same seed gives the same result:

    >>> np.random.seed(0)
    >>> again = nap.jitter_timestamps(ts, max_jitter=0.1)
    >>> np.array_equal(jittered.t, again.t)
    True
    """
    if not isinstance(ts, (nap.Ts, nap.TsGroup)):
        raise TypeError("Invalid input type, should be Ts or TsGroup")

    if max_jitter is None:
        raise TypeError("missing required argument: max_jitter ")

    if isinstance(ts, nap.Ts):
        times = np.sort(
            ts.times() + np.random.uniform(-max_jitter, max_jitter, len(ts))
        )
        if keep_tsupport:
            return nap.Ts(t=times, time_support=ts.time_support)
        return nap.Ts(t=times)

    # a lazy group reads all its spikes here: every spike is needed
    tsgroup = ts._load_in_memory()
    times = tsgroup._times + np.random.uniform(
        -max_jitter, max_jitter, len(tsgroup._times)
    )
    times, clusters = _sort_merged(times, tsgroup._clusters)
    if keep_tsupport:
        # from_sorted_arrays computes a support from the spans of the units,
        # which fails if no unit has two spikes: build on the input support
        time_support = tsgroup.time_support
        times, clusters = _restrict_arrays(
            times, time_support.start, time_support.end, clusters
        )
        return nap.TsGroup._from_arrays(
            times,
            clusters,
            None,
            np.zeros(len(tsgroup.index), dtype=bool),
            tsgroup.index,
            time_support,
        )
    return nap.TsGroup.from_sorted_arrays(times, clusters, keys=tsgroup.index)


# Random resample


def resample_timestamps(
    ts: Union[nap.Ts, nap.TsGroup],
) -> Union[nap.Ts, nap.TsGroup]:
    """
    Replace each timestamp by a random time in the time support.

    The random times come from a uniform distribution over all the epochs of
    the time support. An epoch gets a number of times that is proportional to
    its length, on average. The gaps between the epochs get no time. The
    output has the same number of timestamps as ``ts``.

    The random numbers come from the global numpy generator. To get the same
    result again, call ``np.random.seed`` before the function.

    Parameters
    ----------
    ts : Ts or TsGroup
        The timestamps to replace. For a TsGroup, each unit keeps its number
        of timestamps, and all the units use the time support of the group.

    Returns
    -------
    Ts or TsGroup
        The new timestamps, on the time support of ``ts``. A TsGroup output has
        the same keys as ``ts``, and no metadata other than ``rate``.

    Raises
    ------
    TypeError
        If ``ts`` is not a Ts or a TsGroup.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> epochs = nap.IntervalSet(start=[0, 20], end=[10, 30])
    >>> ts = nap.Ts([1.0, 2.0, 3.0, 25.0], time_support=epochs)
    >>> np.random.seed(0)
    >>> resampled = nap.resample_timestamps(ts)
    >>> len(resampled)
    4
    >>> bool(np.all(resampled.restrict(epochs).t == resampled.t))
    True
    """
    if not isinstance(ts, (nap.Ts, nap.TsGroup)):
        raise TypeError("Invalid input type, should be Ts or TsGroup")

    if isinstance(ts, nap.TsGroup):
        # a lazy group reads all its spikes here: every spike is needed
        ts = ts._load_in_memory()
        n_times = len(ts._times)
    else:
        n_times = len(ts)

    # Draw a position on the epochs put end to end, then find the epoch of
    # each position and its time in that epoch
    time_support = ts.time_support
    starts, ends = time_support.start, time_support.end
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(ends - starts)])
    positions = np.random.uniform(0.0, cumulative_lengths[-1], n_times)
    epochs = np.searchsorted(cumulative_lengths, positions, side="right") - 1
    epochs = np.minimum(epochs, len(starts) - 1)
    # rounding can move a time past the end of its epoch by the smallest
    # step between two floats
    times = np.minimum(
        starts[epochs] + (positions - cumulative_lengths[epochs]), ends[epochs]
    )

    if isinstance(ts, nap.Ts):
        return nap.Ts(t=np.sort(times), time_support=time_support)

    # each draw replaces one spike, so each unit keeps its number of spikes
    times, clusters = _sort_merged(times, ts._clusters)
    return nap.TsGroup._from_arrays(
        times,
        clusters,
        None,
        np.zeros(len(ts.index), dtype=bool),
        ts.index,
        time_support,
    )
