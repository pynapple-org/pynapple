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


@jit(nopython=True, cache=True)
def _cross_correlogram(
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    binsize: float,
    windowsize: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Performs the discrete cross-correlogram of two time series.
    The units should be in s for all arguments.
    Return the firing rate of the series t2 relative to the timings of t1.
    See compute_crosscorrelogram, compute_autocorrelogram and compute_eventcorrelogram
    for wrappers of this function.

    Parameters
    ----------
    t1 : numpy.ndarray
        The timestamps of the reference time series (in seconds)
    t2 : numpy.ndarray
        The timestamps of the target time series (in seconds)
    binsize : float
        The bin size (in seconds)
    windowsize : float
        The window size (in seconds)

    Returns
    -------
    numpy.ndarray
        The cross-correlogram
    numpy.ndarray
        Center of the bins (in s)

    """
    # nbins = ((windowsize//binsize)*2)

    nt1 = len(t1)
    nt2 = len(t2)

    nbins = int((windowsize * 2) // binsize)
    if np.floor(nbins / 2) * 2 == nbins:
        nbins = nbins + 1

    w = (nbins / 2) * binsize
    C = np.zeros(nbins)
    i2 = 0

    for i1 in range(nt1):
        lbound = t1[i1] - w
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2 + 1
        while i2 > 0 and t2[i2 - 1] > lbound:
            i2 = i2 - 1

        rbound = lbound
        leftb = i2
        for j in range(nbins):
            k = 0
            rbound = rbound + binsize
            while leftb < nt2 and t2[leftb] < rbound:
                leftb = leftb + 1
                k = k + 1

            C[j] += k

    C = C / (nt1 * binsize)

    m = -w + binsize / 2
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m + j * binsize

    return C, B


@jit(nopython=True, cache=True, parallel=True)
def _crosscorrelogram_counts(
    times1, order1, offsets1, times2, order2, offsets2, ref, target, nbins, binsize
):
    """Cross-correlogram counts of pairs of units, pairs in parallel.

    Parameters
    ----------
    times1, times2 : numpy.ndarray
        Merged spike times of the reference and target groups, as integer
        nanoseconds (the same array when both come from one group).
    order1, offsets1, order2, offsets2 : numpy.ndarray
        ``order[offsets[u]:offsets[u + 1]]`` are the positions in ``times`` of
        unit ``u``'s spikes, in time order.
    ref, target : numpy.ndarray
        Positions of the reference and target unit of each pair.
    nbins : int
        Number of bins, odd.
    binsize : int
        Bin size in nanoseconds.

    Returns
    -------
    numpy.ndarray
        ``(n_pairs, nbins)`` counts. Bin ``j`` of a reference spike at ``t`` is
        ``[t - w + j * binsize, t - w + (j + 1) * binsize)`` with
        ``w = nbins * binsize / 2``, computed exactly in integers.
    """
    n_pairs = len(ref)
    counts = np.zeros((n_pairs, nbins), dtype=np.int64)
    # lags are taken at double scale so that w = width / 2 needs no rounding
    width = nbins * binsize
    bin_width = 2 * binsize
    # pairs are independent and each writes only its own row
    for p in prange(n_pairs):
        start1, stop1 = offsets1[ref[p]], offsets1[ref[p] + 1]
        start2, stop2 = offsets2[target[p]], offsets2[target[p] + 1]
        # target spikes in [t - w, t + w), moving forward with t
        lo = start2
        hi = start2
        for i in range(start1, stop1):
            t2 = 2 * times1[order1[i]]
            while lo < stop2 and 2 * times2[order2[lo]] < t2 - width:
                lo += 1
            if hi < lo:
                hi = lo
            while hi < stop2 and 2 * times2[order2[hi]] < t2 + width:
                hi += 1
            for k in range(lo, hi):
                counts[p, (2 * times2[order2[k]] - t2 + width) // bin_width] += 1
    return counts


def _correlogram_bins(binsize, windowsize):
    """Number of bins, half-window and bin size in integer nanoseconds.

    Spike times are binned in integer nanoseconds (the precision of the time
    index), so a lag falling exactly on a bin edge always lands in the bin on
    its right, with no floating-point rounding either way.

    Raises a ValueError if ``binsize`` (in seconds) is smaller than one unit of
    the time precision, ``10**-nap_config.time_index_precision`` seconds.
    """
    precision = nap.nap_config.time_index_precision
    # Check first: a bin size of 0 makes the number of bins infinite. Compare
    # at 6 decimals: binsize * 10**precision is not exact in floats.
    if np.round(binsize * 10.0**precision, 6) < 1:
        raise ValueError(
            f"binsize must be at least 1e-{precision} s, the time precision of "
            f"pynapple (nap_config.time_index_precision = {precision}). "
            f"Got {binsize} s after the conversion to seconds."
        )
    nbins = int((windowsize * 2) // binsize)
    if nbins % 2 == 0:
        nbins = nbins + 1
    w = (nbins / 2) * binsize
    binsize_ns = int(np.round(binsize * 10.0**precision))
    return nbins, w, binsize_ns


def _times_ns(times):
    """Times in seconds as integer nanoseconds (the time index precision)."""
    precision = 10.0**nap.nap_config.time_index_precision
    return np.round(np.asarray(times, dtype=np.float64) * precision).astype(np.int64)


def _ragged_spikes(group):
    """Give the spikes of a group in the layout of the correlogram kernel.

    Returns
    -------
    times, order, offsets : numpy.ndarray
        ``times`` holds the spike times in integer nanoseconds.
        ``order[offsets[u]:offsets[u + 1]]`` are the positions in ``times`` of
        the spikes of unit ``u``, in time order.
    """
    order, offsets = group._ragged_index
    return _times_ns(group._times), order, offsets


def _correlogram_counts(
    reference, target, ref_units, target_units, binsize, windowsize
):
    """Count the lags of the pairs of units in the bins of a correlogram.

    The three public correlogram functions call this function. It computes the
    bins, then calls the kernel ``_crosscorrelogram_counts``, with the pairs in
    parallel.

    Parameters
    ----------
    reference, target : tuple of numpy.ndarray
        The spikes of the reference units and of the target units, as
        ``_ragged_spikes`` gives them: ``(times, order, offsets)``. Use the same
        tuple twice for pairs in one group.
    ref_units, target_units : array-like of int
        The position of the reference unit and of the target unit of each
        pair.
    binsize, windowsize : float
        The bin size and the window size, in seconds.

    Returns
    -------
    counts : numpy.ndarray
        ``(n_pairs, n_bins)`` counts. For each spike at ``t`` of a reference
        unit, bin ``j`` counts the target spikes in
        ``[t - w + j * binsize, t - w + (j + 1) * binsize)``, with
        ``w = n_bins * binsize / 2``.
    lags : numpy.ndarray
        The center of each bin, in seconds.
    """
    nbins, w, binsize_ns = _correlogram_bins(binsize, windowsize)
    counts = _crosscorrelogram_counts(
        *reference,
        *target,
        np.asarray(ref_units, dtype=np.int64),
        np.asarray(target_units, dtype=np.int64),
        nbins,
        binsize_ns,
    )
    lags = -w + binsize / 2 + np.arange(nbins) * binsize
    return counts, lags


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
    Computes the autocorrelogram of a group of Ts/Tsd objects.
    The group can be passed directly as a TsGroup object.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects to auto-correlate
    binsize : float
        The bin size, in ``time_units`` (seconds by default). It must be at
        least the time precision of pynapple:
        ``10**-nap_config.time_index_precision`` seconds (1 ns by default).
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which auto-corrs are computed.
        If None, the epoch is the time support of the group.
    norm : bool, optional
         If True, autocorrelograms are normalized to baseline (i.e. divided by the average rate)
         If False, autocorrelograms are returned as the rate (Hz) of the time series (relative to itself)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').

    Returns
    -------
    pandas.DataFrame
        DataFrame with time lags as index and unit IDs as columns.
        Values represent the firing rate (or normalized rate if norm=True).

    Raises
    ------
    TypeError
        If group is not a TsGroup, or if binsize, windowsize, ep, norm, or time_units
        have invalid types.

    ValueError
        If ``binsize`` is smaller than the time precision of pynapple
        (``10**-nap_config.time_index_precision`` seconds, 1 ns by default).

    Notes
    -----
    The function computes the lags in integer units of the time precision of
    pynapple (1 ns by default). Thus it has no floating-point error, and a lag
    exactly on a bin edge always goes into the bin on its right.

    Examples
    --------
    >>> import pynapple as nap
    >>> import numpy as np
    >>> ts_group = nap.TsGroup({0: nap.Ts(t=np.sort(np.random.uniform(0, 10, 100)))})
    >>> autocorr = nap.compute_autocorrelogram(ts_group, binsize=0.01, windowsize=0.1)
    """
    if isinstance(ep, nap.IntervalSet):
        newgroup = group.restrict(ep)
    else:
        newgroup = group._load_in_memory()

    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_units
    )[0]
    windowsize = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]

    # The autocorrelogram of a unit is its cross-correlogram with itself: the
    # pairs (u, u). The kernel also counts each spike with itself, at lag 0.
    # Remove these counts: the center bin is lag 0 (the number of bins is odd).
    spikes = _ragged_spikes(newgroup)
    n_spikes = np.diff(spikes[2])
    units = np.arange(len(n_spikes))
    counts, lags = _correlogram_counts(
        spikes, spikes, units, units, binsize, windowsize
    )
    counts[:, len(lags) // 2] -= n_spikes

    with np.errstate(divide="ignore", invalid="ignore"):
        autocorrs = counts.T / (n_spikes * binsize)
    autocorrs = pd.DataFrame(autocorrs, index=np.round(lags, 6), columns=newgroup.index)

    if norm:
        autocorrs = autocorrs / newgroup.get_info("rate")

    # Bug here
    if 0 in autocorrs.index:
        autocorrs.loc[0] = 0.0

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
    Computes all the pairwise cross-correlograms for TsGroup or list/tuple of two TsGroup.

    If input is TsGroup only, the reference Ts/Tsd and target are chosen based on the builtin itertools.combinations function.
    For example if indexes are [0,1,2], the function computes cross-correlograms
    for the pairs (0,1), (0, 2), and (1, 2). The left index gives the reference time series.
    To reverse the order, set reverse=True.

    If input is tuple/list of TsGroup, for example group=(group1, group2), the reference for each pairs comes from group1.

    Parameters
    ----------
    group : TsGroup or tuple/list of two TsGroups
        The group(s) of Ts/Tsd objects to cross-correlate. If a single TsGroup,
        computes pairwise cross-correlograms within the group. If a tuple/list
        of two TsGroups, computes cross-correlograms between all pairs from
        group1 (reference) and group2 (target).
    binsize : float
        The bin size, in ``time_units`` (seconds by default). It must be at
        least the time precision of pynapple:
        ``10**-nap_config.time_index_precision`` seconds (1 ns by default).
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which cross-corrs are computed.
        If None, the epoch is the time support of the group.
    norm : bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-correlograms are returned as the rate (Hz) of the target time series (relative to the reference time series)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    reverse : bool, optional
        To reverse the pair order if input is TsGroup

    Returns
    -------
    pandas.DataFrame
        DataFrame with time lags as index and pair tuples (i, j) as columns.
        Values represent the firing rate of unit j relative to unit i
        (or normalized rate if norm=True).

    Raises
    ------
    TypeError
        If group is not a TsGroup or tuple/list of two TsGroups, or if binsize,
        windowsize, ep, norm, time_units, or reverse have invalid types.

    ValueError
        If ``binsize`` is smaller than the time precision of pynapple
        (``10**-nap_config.time_index_precision`` seconds, 1 ns by default).

    Notes
    -----
    The function computes the lags in integer units of the time precision of
    pynapple (1 ns by default). Thus it has no floating-point error, and a lag
    exactly on a bin edge always goes into the bin on its right.

    Examples
    --------
    >>> import pynapple as nap
    >>> import numpy as np
    >>> ts_group = nap.TsGroup({
    ...     0: nap.Ts(t=np.sort(np.random.uniform(0, 10, 100))),
    ...     1: nap.Ts(t=np.sort(np.random.uniform(0, 10, 100)))
    ... })
    >>> crosscorr = nap.compute_crosscorrelogram(ts_group, binsize=0.01, windowsize=0.1)
    """
    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_units
    )[0]
    windowsize = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]

    if isinstance(group, (tuple, list)):
        if isinstance(ep, nap.IntervalSet):
            newgroup = [group[i].restrict(ep) for i in range(2)]
        else:
            newgroup = [g._load_in_memory() for g in group]
        pairs = list(product(newgroup[0].keys(), newgroup[1].keys()))
    else:
        if isinstance(ep, nap.IntervalSet):
            newgroup = group.restrict(ep)
        else:
            newgroup = group._load_in_memory()
        pairs = list(combinations(newgroup.keys(), 2))
        if reverse:
            pairs = list(map(lambda n: (n[1], n[0]), pairs))
        newgroup = [newgroup, newgroup]

    if len(pairs) == 0:
        return pd.DataFrame().astype("float")

    ref = np.searchsorted(newgroup[0].index, [i for i, _ in pairs])
    target = np.searchsorted(newgroup[1].index, [j for _, j in pairs])

    spikes1 = _ragged_spikes(newgroup[0])
    spikes2 = spikes1 if newgroup[1] is newgroup[0] else _ragged_spikes(newgroup[1])
    counts, lags = _correlogram_counts(
        spikes1, spikes2, ref, target, binsize, windowsize
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        crosscorrs = counts.T / (np.diff(spikes1[2])[ref] * binsize)
        if norm:
            # rate of the target of each pair
            crosscorrs = crosscorrs / newgroup[1].rates[target]
    crosscorrs = pd.DataFrame(
        crosscorrs, index=lags, columns=pd.MultiIndex.from_tuples(pairs)
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
    Computes the correlograms of a group of Ts/Tsd objects with another single Ts/Tsd object
    The time of reference is the event times.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects to correlate with the event
    event : Ts/Tsd
        The event to correlate the each of the time series in the group with.
    binsize : float
        The bin size, in ``time_units`` (seconds by default). It must be at
        least the time precision of pynapple:
        ``10**-nap_config.time_index_precision`` seconds (1 ns by default).
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which cross-corrs are computed.
        If None, the epoch is the time support of the event.
    norm : bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-correlograms are returned as the rate (Hz) of the target time series (relative to the event time series)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').

    Returns
    -------
    pandas.DataFrame
        DataFrame with time lags as index and unit IDs as columns.
        Values represent the firing rate of each unit relative to the event times
        (or normalized rate if norm=True).

    Raises
    ------
    TypeError
        If group is not a TsGroup, if event is not a Ts or Tsd, or if binsize,
        windowsize, ep, norm, or time_units have invalid types.

    ValueError
        If ``binsize`` is smaller than the time precision of pynapple
        (``10**-nap_config.time_index_precision`` seconds, 1 ns by default).

    Notes
    -----
    The function computes the lags in integer units of the time precision of
    pynapple (1 ns by default). Thus it has no floating-point error, and a lag
    exactly on a bin edge always goes into the bin on its right.

    Examples
    --------
    >>> import pynapple as nap
    >>> import numpy as np
    >>> ts_group = nap.TsGroup({0: nap.Ts(t=np.sort(np.random.uniform(0, 10, 100)))})
    >>> event = nap.Ts(t=np.array([1.0, 3.0, 5.0, 7.0, 9.0]))
    >>> eventcorr = nap.compute_eventcorrelogram(ts_group, event, binsize=0.01, windowsize=0.1)
    """
    if ep is None:
        ep = event.time_support
        tsd1 = event.index
    else:
        tsd1 = event.restrict(ep).index

    newgroup = group.restrict(ep)

    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_units
    )[0]
    windowsize = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]

    if len(newgroup.index) == 0:
        return pd.DataFrame().astype("float")

    # The events are a single reference unit, paired with every unit of the
    # group.
    n_events = len(tsd1)
    n_units = len(newgroup.index)
    events = (_times_ns(tsd1), np.arange(n_events), np.array([0, n_events]))
    counts, lags = _correlogram_counts(
        events,
        _ragged_spikes(newgroup),
        np.zeros(n_units, dtype=np.int64),
        np.arange(n_units),
        binsize,
        windowsize,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        crosscorrs = counts.T / (n_events * binsize)
        if norm:
            crosscorrs = crosscorrs / newgroup.rates
    crosscorrs = pd.DataFrame(crosscorrs, index=lags, columns=newgroup.index)

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
