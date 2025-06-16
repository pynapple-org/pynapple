"""
Functions to compute correlograms of timestamps data.
"""

import inspect
from functools import wraps
from itertools import combinations, product
from numbers import Number

import numpy as np
import pandas as pd
from numba import jit

from .. import core as nap


def _validate_correlograms_inputs(func):
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
def _cross_correlogram(t1, t2, binsize, windowsize):
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


@_validate_correlograms_inputs
def compute_autocorrelogram(
    group, binsize, windowsize, ep=None, norm=True, time_units="s"
):
    """
    Computes the autocorrelogram of a group of Ts/Tsd objects.
    The group can be passed directly as a TsGroup object.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd objects to auto-correlate
    binsize : float
        The bin size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which auto-corrs are computed.
        If None, the epoch is the time support of the group.
    norm : bool, optional
         If True, autocorrelograms are normalized to baseline (i.e. divided by the average rate)
         If False, autoorrelograms are returned as the rate (Hz) of the time series (relative to itself)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').

    Returns
    -------
    pandas.DataFrame
        _

    Raises
    ------
    RuntimeError
        group must be TsGroup
    """
    if isinstance(ep, nap.IntervalSet):
        newgroup = group.restrict(ep)
    else:
        newgroup = group

    autocorrs = {}

    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_units
    )[0]
    windowsize = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]

    for n in newgroup.keys():
        spk_time = newgroup[n].index
        auc, times = _cross_correlogram(spk_time, spk_time, binsize, windowsize)
        autocorrs[n] = pd.Series(index=np.round(times, 6), data=auc, dtype="float")

    autocorrs = pd.DataFrame.from_dict(autocorrs)

    if norm:
        autocorrs = autocorrs / newgroup.get_info("rate")

    # Bug here
    if 0 in autocorrs.index:
        autocorrs.loc[0] = 0.0

    return autocorrs.astype("float")


@_validate_correlograms_inputs
def compute_crosscorrelogram(
    group, binsize, windowsize, ep=None, norm=True, time_units="s", reverse=False
):
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

    binsize : float
        The bin size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which cross-corrs are computed.
        If None, the epoch is the time support of the group.
    norm : bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-orrelograms are returned as the rate (Hz) of the target time series ((relative to the reference time series)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').
    reverse : bool, optional
        To reverse the pair order if input is TsGroup

    Returns
    -------
    pandas.DataFrame
        _

    Raises
    ------
    RuntimeError
        group must be TsGroup or tuple/list of two TsGroups

    """
    crosscorrs = {}

    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_units
    )[0]
    windowsize = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]

    if isinstance(group, tuple):
        if isinstance(ep, nap.IntervalSet):
            newgroup = [group[i].restrict(ep) for i in range(2)]
        else:
            newgroup = group

        pairs = product(list(newgroup[0].keys()), list(newgroup[1].keys()))

        for i, j in pairs:
            spk1 = newgroup[0][i].index
            spk2 = newgroup[1][j].index
            auc, times = _cross_correlogram(spk1, spk2, binsize, windowsize)
            if norm:
                auc /= newgroup[1][j].rate
            crosscorrs[(i, j)] = pd.Series(index=times, data=auc, dtype="float")

        crosscorrs = pd.DataFrame.from_dict(crosscorrs)
    else:
        if isinstance(ep, nap.IntervalSet):
            newgroup = group.restrict(ep)
        else:
            newgroup = group
        neurons = list(newgroup.keys())
        pairs = list(combinations(neurons, 2))
        if reverse:
            pairs = list(map(lambda n: (n[1], n[0]), pairs))

        for i, j in pairs:
            spk1 = newgroup[i].index
            spk2 = newgroup[j].index
            auc, times = _cross_correlogram(spk1, spk2, binsize, windowsize)
            crosscorrs[(i, j)] = pd.Series(index=times, data=auc, dtype="float")

        crosscorrs = pd.DataFrame.from_dict(crosscorrs)

        if norm:
            freq = pd.Series(
                index=newgroup.metadata_index, data=newgroup.get_info("rate")
            )
            freq2 = pd.Series(
                index=pairs, data=list(map(lambda n: freq.loc[n[1]], pairs))
            )
            crosscorrs = crosscorrs / freq2

    return crosscorrs.astype("float")


@_validate_correlograms_inputs
def compute_eventcorrelogram(
    group, event, binsize, windowsize, ep=None, norm=True, time_units="s"
):
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
        The bin size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    windowsize : float
        The window size. Default is second.
        If different, specify with the parameter time_units ('s' [default], 'ms', 'us').
    ep : IntervalSet
        The epoch on which cross-corrs are computed.
        If None, the epoch is the time support of the event.
    norm : bool, optional
        If True (default), cross-correlograms are normalized to baseline (i.e. divided by the average rate of the target time series)
        If False, cross-orrelograms are returned as the rate (Hz) of the target time series (relative to the event time series)
    time_units : str, optional
        The time units of the parameters. They have to be consistent for binsize and windowsize.
        ('s' [default], 'ms', 'us').

    Returns
    -------
    pandas.DataFrame
        _

    Raises
    ------
    RuntimeError
        group must be TsGroup

    """
    if ep is None:
        ep = event.time_support
        tsd1 = event.index
    else:
        tsd1 = event.restrict(ep).index

    newgroup = group.restrict(ep)

    crosscorrs = {}

    binsize = nap.TsIndex.format_timestamps(
        np.array([binsize], dtype=np.float64), time_units
    )[0]
    windowsize = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]

    for n in newgroup.keys():
        spk_time = newgroup[n].index
        auc, times = _cross_correlogram(tsd1, spk_time, binsize, windowsize)
        crosscorrs[n] = pd.Series(index=times, data=auc, dtype="float")

    crosscorrs = pd.DataFrame.from_dict(crosscorrs)

    if norm:
        crosscorrs = crosscorrs / newgroup.get_info("rate")

    return crosscorrs.astype("float")


def compute_isi_distribution(
    data,
    bins=10,
    log_scale=False,
    epochs=None,
):
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

    Examples
    -------
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
        bin_edges = np.linspace(min_isi, max_isi, bins + 1)
    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError("`bins` must increase monotonically, when an array")
    else:
        raise ValueError("`bins` must be 1d, when an array")

    return pd.DataFrame(
        index=(bin_edges[:-1] + bin_edges[1:]) / 2,
        data={i: np.histogram(time_diffs[i].values, bin_edges)[0] for i in time_diffs},
    )
