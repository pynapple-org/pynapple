"""
Functions to compute correlograms of timestamps data.
"""

from __future__ import annotations

import inspect
from functools import wraps
from itertools import combinations, product
from numbers import Number
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import jit, prange

from .. import core as nap


def _validate_correlograms_inputs(func: Callable) -> Callable:
    """
    Decorator to validate input types for correlogram functions.

    Validates that group is a TsGroup (or tuple/list of TsGroups for crosscorrelogram),
    and checks types for binsize, windowsize, ep, norm, time_units, and event parameters.

    Parameters
    ----------
    func : Callable
        The function to wrap with input validation.

    Returns
    -------
    Callable
        The wrapped function with input validation.

    Raises
    ------
    TypeError
        If any parameter has an invalid type.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        kwargs = sig.bind_partial(*args, **kwargs).arguments

        # Only TypeError here
        if getattr(func, "__name__") == "compute_crosscorrelogram" and isinstance(
            kwargs["group"], (tuple, list)
        ):
            if (
                not all([isinstance(g, nap.TsGroup) for g in kwargs["group"]])
                or len(kwargs["group"]) != 2
            ):
                raise TypeError(
                    "Invalid type. Parameter group must be of type TsGroup or a tuple/list of (TsGroup, TsGroup)."
                )
        else:
            if not isinstance(kwargs["group"], nap.TsGroup):
                msg = "Invalid type. Parameter group must be of type TsGroup"
                if getattr(func, "__name__") == "compute_crosscorrelogram":
                    msg = msg + " or a tuple/list of (TsGroup, TsGroup)."
                raise TypeError(msg)

        parameters_type = {
            "binsize": Number,
            "windowsize": Number,
            "ep": nap.IntervalSet,
            "norm": bool,
            "time_units": str,
            "reverse": bool,
            "event": (nap.Ts, nap.Tsd),
        }
        for param, param_type in parameters_type.items():
            if param in kwargs:
                if not isinstance(kwargs[param], param_type):
                    raise TypeError(
                        f"Invalid type. Parameter {param} must be of type {param_type}."
                    )

        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


@jit(nopython=True, cache=True, parallel=True)
def _cross_correlograms(
    times1, offsets1, times2, offsets2, ref, target, binsize, windowsize, precision
):
    """Compute the cross-correlograms of pairs of units, with the pairs in
    parallel.

    The function uses integer times. Thus it has no floating-point error, and
    a lag exactly on a bin edge always goes into the bin on its right.

    Parameters
    ----------
    times1, times2 : numpy.ndarray
        The spike times of the reference units and of the target units, in
        integer units of the time precision (nanoseconds by default). The
        spikes of unit ``u`` are the sorted block
        ``times[offsets[u]:offsets[u + 1]]``. For pairs in one group, give the
        same arrays twice.
    offsets1, offsets2 : numpy.ndarray
        The start of the block of each unit, and the end of the last block.
    ref, target : numpy.ndarray
        The position of the reference unit and of the target unit of each
        pair.
    binsize, windowsize : float
        The bin size and the window size, in seconds.
    precision : float
        The number of time units in one second
        (``10**nap_config.time_index_precision``).

    Returns
    -------
    numpy.ndarray
        ``(n_pairs, n_bins)``. The rate of the target unit around the spikes
        of the reference unit, in Hz. For each spike at ``t`` of the reference
        unit, bin ``j`` counts the target spikes in
        ``[t - w + j * binsize, t - w + (j + 1) * binsize)``, with
        ``w = n_bins * binsize / 2``. The bin size is ``binsize`` rounded to
        the time precision.
    numpy.ndarray
        The center of each bin, in seconds.
    """
    # an odd number of bins, so that the center bin is the lag 0
    nbins = int((windowsize * 2) // binsize)
    if nbins % 2 == 0:
        nbins = nbins + 1

    # The bin size in integer time units. The bins and the lags use this value.
    binsize_int = int(np.round(binsize * precision))
    # The center of each bin. The division of two exact integers gives the
    # nearest float, so the lags have no floating-point error.
    lags = ((np.arange(nbins) - nbins // 2) * binsize_int) / precision

    # The lags are doubled, so that the half-window width / 2 needs no
    # rounding.
    width = nbins * binsize_int
    bin_width = 2 * binsize_int

    n_pairs = len(ref)
    rates = np.zeros((n_pairs, nbins))
    # Each pair writes only its own row: the pairs can run in parallel.
    for p in prange(n_pairs):
        # the block of the reference unit and the block of the target unit
        start1, stop1 = offsets1[ref[p]], offsets1[ref[p] + 1]
        start2, stop2 = offsets2[target[p]], offsets2[target[p] + 1]
        # the target spikes in [t - w, t + w) are times2[lo:hi].
        # t increases, so lo and hi only move forward.
        lo = start2
        hi = start2
        for i in range(start1, stop1):
            # the reference spike time, doubled
            t_ref = 2 * times1[i]
            # move lo past the target spikes before t - w
            while lo < stop2 and 2 * times2[lo] < t_ref - width:
                lo += 1
            if hi < lo:
                hi = lo
            # move hi past the target spikes before t + w
            while hi < stop2 and 2 * times2[hi] < t_ref + width:
                hi += 1
            # add each target spike in the window to the bin of its lag
            # Faster than in vectorized form
            for k in range(lo, hi):
                # the lag of the target spike, doubled, in [-width, width)
                lag = 2 * times2[k] - t_ref
                # shift the lag to [0, 2 * width)
                shifted = lag + width
                # the bin of the lag, from 0 to nbins - 1
                b = shifted // bin_width
                rates[p, b] += 1
        # change the counts to a rate in Hz per reference spike
        rates[p] /= (stop1 - start1) * binsize_int / precision
    return rates, lags


def _bin_parameters(binsize, windowsize, time_units):
    """
    Convert the bin size and the window size to seconds and check the bin size.

    Parameters
    ----------
    binsize : float
        The bin size, in time_units.
    windowsize : float
        The window size, in time_units.
    time_units : str
        The time units of binsize and windowsize ('s', 'ms' or 'us').

    Returns
    -------
    binsize : float
        The bin size in seconds.
    windowsize : float
        The window size in seconds.
    precision : float
        The number of time units in one second (10**time_index_precision).

    Raises
    ------
    ValueError
        If binsize is smaller than the time precision of pynapple.
    """
    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_units
    )[0]
    windowsize = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]

    digits = nap.nap_config.time_index_precision
    precision = 10.0**digits
    if np.round(binsize * precision) < 1:
        raise ValueError(
            f"binsize must be at least 1e-{digits} s, the time precision of "
            f"pynapple (nap_config.time_index_precision = {digits})."
        )
    return binsize, windowsize, precision


def _integer_ragged_times(group, precision):
    """
    Return the spike times of a TsGroup as a ragged array of integers.

    Parameters
    ----------
    group : TsGroup
        The group, loaded in memory.
    precision : float
        The number of time units in one second (10**time_index_precision).

    Returns
    -------
    times : numpy.ndarray
        The spike times in integer time units, one sorted block per unit.
    offsets : numpy.ndarray
        The block of unit u is times[offsets[u]:offsets[u + 1]].
    """
    order, offsets = group._ragged_index
    times = np.round(group._times[order] * precision).astype(np.int64)
    return times, offsets


@_validate_correlograms_inputs
def compute_autocorrelogram(
    group: nap.TsGroup,
    binsize: float,
    windowsize: float,
    ep: Optional[nap.IntervalSet] = None,
    norm: bool = True,
    time_units: str = "s",
) -> pd.DataFrame:
    """
    Compute the autocorrelogram of each unit of a group.

    For each spike of a unit, the function counts the other spikes of the same
    unit in the bins of a window around the spike. Then it divides the counts
    by the number of spikes and by the bin size. The result is a rate in Hz.

    Parameters
    ----------
    group : TsGroup
        The units.
    binsize : float
        The width of one bin, in ``time_units``. The function rounds it to the
        time precision of pynapple (1 ns by default).
    windowsize : float
        The largest lag, in ``time_units``. The bins cover the lags from about
        ``-windowsize`` to ``windowsize``.
    ep : IntervalSet, optional
        The epochs to use. The function restricts the group to ``ep``. If None
        (default), the function uses the time support of the group.
    norm : bool, optional
        If True (default), divide the autocorrelogram of each unit by the mean
        rate of the unit. Then the value 1 is equal to the mean rate. If False,
        the values are rates in Hz.
    time_units : str, optional
        The time units of ``binsize`` and ``windowsize``: ``'s'`` (default),
        ``'ms'`` or ``'us'``.

    Returns
    -------
    pandas.DataFrame
        One column for each unit, with the unit labels as column names. The
        index is the center of each bin (the lag), in seconds.

        The bin at lag 0 is always 0. Without this, each spike counts itself
        in this bin. Thus the bin at lag 0 also does not count the other
        spikes of the unit in ``[-binsize / 2, binsize / 2)``.

        If a unit has no spikes, its column is NaN. If the group has no units,
        the DataFrame is empty.

    Raises
    ------
    TypeError
        If ``group`` is not a TsGroup, or if a parameter has an incorrect type.
    ValueError
        If ``binsize`` is smaller than the time precision of pynapple.

    See Also
    --------
    compute_crosscorrelogram : The correlograms of pairs of units.
    compute_eventcorrelogram : The correlograms of units with events.

    Notes
    -----
    The number of bins is ``2 * windowsize // binsize``, plus 1 if this number
    is even. Thus the number of bins is always odd, and the center bin has the
    lag 0. The bin with the center ``lag`` counts the lags in
    ``[lag - binsize / 2, lag + binsize / 2)``. A lag exactly on a bin edge
    goes into the bin on its right.

    The function counts with integer times, in units of the time precision of
    pynapple. Thus the bins have no floating-point error.

    Examples
    --------
    A unit with a spike every 100 ms, and a unit with a spike every 200 ms:

    >>> import numpy as np
    >>> import pynapple as nap
    >>> group = nap.TsGroup({
    ...     0: nap.Ts(t=np.arange(0, 10, 0.1)),
    ...     1: nap.Ts(t=np.arange(0.02, 10, 0.2)),
    ... })
    >>> nap.compute_autocorrelogram(group, binsize=0.05, windowsize=0.2, norm=False)
              0     1
    -0.20  19.6  19.6
    -0.15   0.0   0.0
    -0.10  19.8   0.0
    -0.05   0.0   0.0
     0.00   0.0   0.0
     0.05   0.0   0.0
     0.10  19.8   0.0
     0.15   0.0   0.0
     0.20  19.6  19.6

    The same bins, with the sizes in milliseconds:

    >>> autocorr = nap.compute_autocorrelogram(
    ...     group, binsize=50, windowsize=200, time_units="ms"
    ... )
    >>> autocorr.index.values
    array([-0.2 , -0.15, -0.1 , -0.05,  0.  ,  0.05,  0.1 ,  0.15,  0.2 ])
    """
    binsize, windowsize, precision = _bin_parameters(binsize, windowsize, time_units)

    if isinstance(ep, nap.IntervalSet):
        newgroup = group.restrict(ep)
    else:
        newgroup = group._load_in_memory()

    if len(newgroup) == 0:
        return pd.DataFrame().astype("float")

    # each unit is the reference and the target of its own pair
    times, offsets = _integer_ragged_times(newgroup, precision)
    units = np.arange(len(newgroup))
    rates, lags = _cross_correlograms(
        times, offsets, times, offsets, units, units, binsize, windowsize, precision
    )
    # The number of bins is odd, so the center bin is always the lag 0.
    # In this bin, each spike counts itself: set it to 0.
    rates[:, len(lags) // 2] = 0.0
    if norm:
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = rates / newgroup.rates[:, None]

    autocorrs = pd.DataFrame(rates.T, index=lags, columns=newgroup.index)

    return autocorrs.astype("float")


@_validate_correlograms_inputs
def compute_crosscorrelogram(
    group: Union[nap.TsGroup, tuple[nap.TsGroup, nap.TsGroup], list[nap.TsGroup]],
    binsize: float,
    windowsize: float,
    ep: Optional[nap.IntervalSet] = None,
    norm: bool = True,
    time_units: str = "s",
    reverse: bool = False,
) -> pd.DataFrame:
    """
    Compute the cross-correlograms of pairs of units.

    Each pair has a reference unit and a target unit. For each spike of the
    reference unit, the function counts the spikes of the target unit in the
    bins of a window around the spike. Then it divides the counts by the
    number of reference spikes and by the bin size. The result is a rate in Hz.

    The pairs come from ``group``:

    - **One TsGroup:** each pair of two different units, from
      ``itertools.combinations``. For the units ``[0, 1, 2]``, the pairs are
      ``(0, 1)``, ``(0, 2)`` and ``(1, 2)``. The first unit of a pair is the
      reference. Set ``reverse=True`` to use the second unit as the reference.
    - **Two TsGroups** ``(group1, group2)``: each unit of ``group1`` with each
      unit of ``group2``. The unit of ``group1`` is the reference.

    Parameters
    ----------
    group : TsGroup, or tuple or list of two TsGroups
        The units.
    binsize : float
        The width of one bin, in ``time_units``. The function rounds it to the
        time precision of pynapple (1 ns by default).
    windowsize : float
        The largest lag, in ``time_units``. The bins cover the lags from about
        ``-windowsize`` to ``windowsize``.
    ep : IntervalSet, optional
        The epochs to use. The function restricts each group to ``ep``. If None
        (default), the function uses the time support of each group.
    norm : bool, optional
        If True (default), divide the cross-correlogram of each pair by the
        mean rate of the target unit. Then the value 1 is equal to the mean
        rate of the target unit. If False, the values are rates in Hz.
    time_units : str, optional
        The time units of ``binsize`` and ``windowsize``: ``'s'`` (default),
        ``'ms'`` or ``'us'``.
    reverse : bool, optional
        If True, use the second unit of each pair as the reference. Only for
        one TsGroup. Default is False.

    Returns
    -------
    pandas.DataFrame
        One column for each pair, with the tuple ``(reference, target)`` of
        unit labels as column name. The index is the center of each bin (the
        lag), in seconds. A positive lag is a target spike after a reference
        spike.

        If the reference unit has no spikes, the column is NaN. With
        ``norm=True``, the column is also NaN if the target unit has no spikes.
        If there are no pairs, the DataFrame is empty.

    Raises
    ------
    TypeError
        If ``group`` is not a TsGroup or a tuple or list of two TsGroups, or if
        a parameter has an incorrect type.
    ValueError
        If ``binsize`` is smaller than the time precision of pynapple.

    See Also
    --------
    compute_autocorrelogram : The correlogram of each unit with itself.
    compute_eventcorrelogram : The correlograms of units with events.

    Notes
    -----
    The number of bins is ``2 * windowsize // binsize``, plus 1 if this number
    is even. Thus the number of bins is always odd, and the center bin has the
    lag 0. The bin with the center ``lag`` counts the lags in
    ``[lag - binsize / 2, lag + binsize / 2)``. A lag exactly on a bin edge
    goes into the bin on its right.

    The function counts with integer times, in units of the time precision of
    pynapple. Thus the bins have no floating-point error.

    Examples
    --------
    A unit with a spike every 100 ms, and a unit with a spike every 200 ms,
    20 ms after the spikes of unit 0:

    >>> import numpy as np
    >>> import pynapple as nap
    >>> group = nap.TsGroup({
    ...     0: nap.Ts(t=np.arange(0, 10, 0.1)),
    ...     1: nap.Ts(t=np.arange(0.02, 10, 0.2)),
    ... })
    >>> nap.compute_crosscorrelogram(group, binsize=0.05, windowsize=0.2, norm=False)
              0
              1
    -0.20   9.8
    -0.15   0.0
    -0.10  10.0
    -0.05   0.0
     0.00  10.0
     0.05   0.0
     0.10   9.8
     0.15   0.0
     0.20   9.8

    With ``reverse=True``, unit 1 is the reference:

    >>> crosscorr = nap.compute_crosscorrelogram(
    ...     group, binsize=0.05, windowsize=0.2, reverse=True
    ... )
    >>> crosscorr.columns.tolist()
    [(1, 0)]

    With two groups, the units of the first group are the references:

    >>> crosscorr = nap.compute_crosscorrelogram(
    ...     (group[[0]], group[[1]]), binsize=0.05, windowsize=0.2
    ... )
    >>> crosscorr.columns.tolist()
    [(0, 1)]
    """
    binsize, windowsize, precision = _bin_parameters(binsize, windowsize, time_units)

    def _load(g):
        if isinstance(ep, nap.IntervalSet):
            return g.restrict(ep)
        return g._load_in_memory()

    if isinstance(group, (tuple, list)):
        group1 = _load(group[0])
        group2 = _load(group[1])
        pairs = list(product(group1.keys(), group2.keys()))
    else:
        group1 = group2 = _load(group)
        pairs = list(combinations(group1.keys(), 2))
        if reverse:
            pairs = [(j, i) for i, j in pairs]

    if len(pairs) == 0:
        return pd.DataFrame().astype("float")

    times1, offsets1 = _integer_ragged_times(group1, precision)
    if group2 is group1:
        times2, offsets2 = times1, offsets1
    else:
        times2, offsets2 = _integer_ragged_times(group2, precision)

    # the position of each unit in its group
    ref = np.searchsorted(group1.index, [i for i, _ in pairs])
    target = np.searchsorted(group2.index, [j for _, j in pairs])

    rates, lags = _cross_correlograms(
        times1, offsets1, times2, offsets2, ref, target, binsize, windowsize, precision
    )
    if norm:
        # divide each pair by the mean rate of its target unit
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = rates / group2.rates[target][:, None]

    crosscorrs = pd.DataFrame(
        rates.T, index=lags, columns=pd.MultiIndex.from_tuples(pairs)
    )
    return crosscorrs.astype("float")


@_validate_correlograms_inputs
def compute_eventcorrelogram(
    group: nap.TsGroup,
    event: Union[nap.Ts, nap.Tsd],
    binsize: float,
    windowsize: float,
    ep: Optional[nap.IntervalSet] = None,
    norm: bool = True,
    time_units: str = "s",
) -> pd.DataFrame:
    """
    Compute the correlogram of each unit of a group with events.

    The events are the reference. For each event, the function counts the
    spikes of each unit in the bins of a window around the event. Then it
    divides the counts by the number of events and by the bin size. The result
    is a rate in Hz.

    Parameters
    ----------
    group : TsGroup
        The units.
    event : Ts or Tsd
        The event times.
    binsize : float
        The width of one bin, in ``time_units``. The function rounds it to the
        time precision of pynapple (1 ns by default).
    windowsize : float
        The largest lag, in ``time_units``. The bins cover the lags from about
        ``-windowsize`` to ``windowsize``.
    ep : IntervalSet, optional
        The epochs to use. The function restricts the events and the group to
        ``ep``. If None (default), the function uses the time support of
        ``event``, and restricts the group to it.
    norm : bool, optional
        If True (default), divide the correlogram of each unit by the mean rate
        of the unit. Then the value 1 is equal to the mean rate. If False, the
        values are rates in Hz.
    time_units : str, optional
        The time units of ``binsize`` and ``windowsize``: ``'s'`` (default),
        ``'ms'`` or ``'us'``.

    Returns
    -------
    pandas.DataFrame
        One column for each unit, with the unit labels as column names. The
        index is the center of each bin (the lag), in seconds. A positive lag
        is a spike after an event.

        If there are no events, all the columns are NaN. With ``norm=True``,
        the column of a unit with no spikes is also NaN. If the group has no
        units, the DataFrame is empty.

    Raises
    ------
    TypeError
        If ``group`` is not a TsGroup, if ``event`` is not a Ts or a Tsd, or if
        a parameter has an incorrect type.
    ValueError
        If ``binsize`` is smaller than the time precision of pynapple.

    See Also
    --------
    compute_autocorrelogram : The correlogram of each unit with itself.
    compute_crosscorrelogram : The correlograms of pairs of units.

    Notes
    -----
    The number of bins is ``2 * windowsize // binsize``, plus 1 if this number
    is even. Thus the number of bins is always odd, and the center bin has the
    lag 0. The bin with the center ``lag`` counts the lags in
    ``[lag - binsize / 2, lag + binsize / 2)``. A lag exactly on a bin edge
    goes into the bin on its right.

    The function counts with integer times, in units of the time precision of
    pynapple. Thus the bins have no floating-point error.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> group = nap.TsGroup({
    ...     0: nap.Ts(t=np.arange(0, 10, 0.1)),
    ...     1: nap.Ts(t=np.arange(0.02, 10, 0.2)),
    ... })
    >>> event = nap.Ts(t=np.array([1.0, 3.0, 5.0, 7.0, 9.0]))
    >>> nap.compute_eventcorrelogram(
    ...     group, event, binsize=0.05, windowsize=0.2, norm=False
    ... )
              0     1
    -0.20  16.0  16.0
    -0.15   0.0   0.0
    -0.10  16.0   0.0
    -0.05   0.0   0.0
     0.00  20.0  16.0
     0.05   0.0   0.0
     0.10  16.0   0.0
     0.15   0.0   0.0
     0.20  16.0  16.0
    """
    if ep is None:
        ep = event.time_support
        tsd1 = event.index
    else:
        tsd1 = event.restrict(ep).index

    newgroup = group.restrict(ep)

    if len(newgroup) == 0:
        return pd.DataFrame().astype("float")

    binsize, windowsize, precision = _bin_parameters(binsize, windowsize, time_units)

    # the event is the reference of each pair, as a ragged array with one unit
    times1 = np.round(np.asarray(tsd1) * precision).astype(np.int64)
    offsets1 = np.array([0, len(times1)])
    times2, offsets2 = _integer_ragged_times(newgroup, precision)
    units = np.arange(len(newgroup))
    rates, lags = _cross_correlograms(
        times1,
        offsets1,
        times2,
        offsets2,
        np.zeros_like(units),
        units,
        binsize,
        windowsize,
        precision,
    )
    if norm:
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = rates / newgroup.rates[:, None]

    crosscorrs = pd.DataFrame(rates.T, index=lags, columns=newgroup.index)

    return crosscorrs.astype("float")


def compute_isi_distribution(
    data: Union[nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup],
    bins: Union[int, list, npt.NDArray] = 10,
    log_scale: bool = False,
    epochs: Optional[nap.IntervalSet] = None,
) -> pd.DataFrame:
    """
    Computes the interspike interval distribution.

    Parameters
    ----------
    data : Ts, TsGroup, Tsd, TsdFrame or TsdTensor
        The Ts, TsGroup, Tsd, TsdFrame or TsdTensor to compute the interspike interval distribution for.
    bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins in the given range (10, by default).
        If bins is a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform bin widths.
    log_scale: bool, optional
        If True, the computed ISI's are log-transformed. Default is False.
    epochs : IntervalSet, optional
        The epochs on which interspike intervals are computed.
        If None, the time support of the input is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame to hold the distribution data.

    Raises
    ------
    TypeError
        If data is not a Ts, TsGroup, Tsd, TsdFrame, or TsdTensor.
    TypeError
        If bins is not an int, list, or np.ndarray.
    TypeError
        If log_scale is not a bool.
    ValueError
        If bins is less than 1 (when int) or not monotonically increasing (when array).

    Examples
    --------
    >>> import numpy as np; np.random.seed(42)
    >>> import pynapple as nap
    >>> ts1 = nap.Ts(t=np.sort(np.random.uniform(0, 1000, 2000)), time_units="s")
    >>> ts2 = nap.Ts(t=np.sort(np.random.uniform(0, 1000, 1000)), time_units="s")
    >>> epochs = nap.IntervalSet(start=0, end=1000, time_units="s")
    >>> ts_group = nap.TsGroup({0: ts1, 1: ts2}, time_support=epochs)
    >>> isi_distribution = nap.compute_isi_distribution(data=ts_group, bins=10, epochs=epochs)
    >>> isi_distribution
                 0    1
    0.322415  1474  477
    0.966402   378  237
    1.610388   100  140
    2.254375    34   67
    2.898362    12   39
    3.542349     1   17
    4.186335     0   12
    4.830322     0    6
    5.474309     0    2
    6.118296     0    2
    """
    if not isinstance(data, (nap.base_class._Base, nap.TsGroup)):
        raise TypeError("data should be a Ts, TsGroup, Tsd, TsdFrame, TsdTensor.")

    if not isinstance(bins, (int, list, np.ndarray)):
        raise TypeError("bins should be either int, list or np.ndarray.")

    if not isinstance(log_scale, bool):
        raise TypeError("log_scale should be of type bool.")

    if epochs is None:
        epochs = data.time_support

    time_diffs = data.time_diff(epochs=epochs)
    if not isinstance(time_diffs, dict):
        time_diffs = {0: time_diffs}

    if log_scale:
        time_diffs = {k: np.log(v) for k, v in time_diffs.items()}

    if np.ndim(bins) == 0:
        if bins < 1:
            raise ValueError("`bins` must be positive, when an integer")
        all_time_diffs = np.hstack([time_diff.d for time_diff in time_diffs.values()])
        min_isi, max_isi = np.min(all_time_diffs), np.max(all_time_diffs)
        histogram_range = (min_isi, max_isi)
        bin_edges = np.histogram_bin_edges(
            all_time_diffs, bins=bins, range=histogram_range
        )
        histograms = {
            i: np.histogram(
                time_diffs[i].values,
                bins=bins,
                range=histogram_range,
            )[0]
            for i in time_diffs
        }
    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError("`bins` must increase monotonically, when an array")
        histograms = {
            i: np.histogram(time_diffs[i].values, bin_edges)[0] for i in time_diffs
        }
    else:
        raise ValueError("`bins` must be 1d, when an array")

    return pd.DataFrame(
        index=(bin_edges[:-1] + bin_edges[1:]) / 2,
        data=histograms,
    )
