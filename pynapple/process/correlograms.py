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
from numba import jit

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


@jit(nopython=True, cache=True)
def _lagged_crosscorrelation(
    data1: npt.NDArray[np.float64],
    data2: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    pairs1: npt.NDArray[np.int64],
    pairs2: npt.NDArray[np.int64],
    max_lag: int,
) -> npt.NDArray[np.float64]:
    """Compute Pearson correlations for lagged column pairs within epochs."""
    n_lags = 2 * max_lag + 1
    n_pairs = len(pairs1)
    correlations = np.full((n_lags, n_pairs), np.nan)

    # Shift each stream before the online updates to preserve small variations.
    anchors1 = np.empty(data1.shape[1])
    anchors2 = np.empty(data2.shape[1])
    means1 = np.empty(data1.shape[1])
    means2 = np.empty(data2.shape[1])
    sums_of_squares1 = np.empty(data1.shape[1])
    sums_of_squares2 = np.empty(data2.shape[1])
    deltas1 = np.empty(data1.shape[1])
    residuals2 = np.empty(data2.shape[1])
    sums_of_products = np.empty(n_pairs)

    for lag_index in range(n_lags):
        lag = lag_index - max_lag
        means1.fill(0.0)
        means2.fill(0.0)
        sums_of_squares1.fill(0.0)
        sums_of_squares2.fill(0.0)
        sums_of_products.fill(0.0)
        n_observations = 0
        epoch_start = 0

        for count in counts:
            n_valid = count - abs(lag)
            if n_valid > 0:
                if lag >= 0:
                    start1 = epoch_start
                    start2 = epoch_start + lag
                else:
                    start1 = epoch_start - lag
                    start2 = epoch_start

                for sample in range(n_valid):
                    if n_observations == 0:
                        anchors1[:] = data1[start1 + sample]
                        anchors2[:] = data2[start2 + sample]
                    n_observations += 1

                    for column in range(data1.shape[1]):
                        value = data1[start1 + sample, column] - anchors1[column]
                        delta = value - means1[column]
                        means1[column] += delta / n_observations
                        sums_of_squares1[column] += delta * (value - means1[column])
                        deltas1[column] = delta

                    for column in range(data2.shape[1]):
                        value = data2[start2 + sample, column] - anchors2[column]
                        delta = value - means2[column]
                        means2[column] += delta / n_observations
                        residual = value - means2[column]
                        sums_of_squares2[column] += delta * residual
                        residuals2[column] = residual

                    for pair in range(n_pairs):
                        sums_of_products[pair] += (
                            deltas1[pairs1[pair]] * residuals2[pairs2[pair]]
                        )

            epoch_start += count

        if n_observations < 2:
            continue
        for pair in range(n_pairs):
            denominator = np.sqrt(
                sums_of_squares1[pairs1[pair]] * sums_of_squares2[pairs2[pair]]
            )
            if denominator > 0.0:
                correlation = sums_of_products[pair] / denominator
                correlations[lag_index, pair] = min(1.0, max(-1.0, correlation))

    return correlations


def compute_lagged_crosscorrelation(
    data: Union[
        nap.TsdFrame,
        tuple[nap.TsdFrame, nap.TsdFrame],
        list[nap.TsdFrame],
    ],
    windowsize: float,
    epochs: Optional[nap.IntervalSet] = None,
    time_units: str = "s",
) -> pd.DataFrame:
    """
    Compute lagged Pearson correlations between columns of continuous data.

    If ``data`` is one ``TsdFrame``, correlations are computed for every unique
    pair of columns. If ``data`` is a tuple or list of two ``TsdFrame`` objects,
    correlations are computed for every pair of columns across the two frames.
    Positive lags indicate that the second signal follows the first signal.

    Valid observations are pooled across epochs for each lag. Pairs never cross
    an epoch boundary.

    Parameters
    ----------
    data : TsdFrame or tuple/list of two TsdFrames
        The regularly sampled continuous signals to correlate. Two frames must
        have identical timestamps.
    windowsize : float
        Maximum lag duration on either side of zero.
    epochs : IntervalSet, optional
        Epochs over which correlations are computed. If None, the shared time
        support of the input data is used.
    time_units : str, optional
        Unit of ``windowsize`` (``"s"`` [default], ``"ms"``, or ``"us"``).

    Returns
    -------
    pandas.DataFrame
        Correlations indexed by lag in seconds, with signal pairs as columns.

    Raises
    ------
    TypeError
        If the inputs have invalid types.
    ValueError
        If ``windowsize`` is negative, ``time_units`` is invalid, or two input
        frames do not have identical timestamps.
    RuntimeError
        If the input is not regularly sampled or its sampling interval cannot
        be determined.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> t = np.arange(100) / 10
    >>> frame = nap.TsdFrame(t=t, d=np.column_stack((np.sin(t), np.cos(t))))
    >>> correlation = nap.compute_lagged_crosscorrelation(frame, windowsize=0.5)
    """
    if isinstance(data, (tuple, list)):
        if len(data) != 2 or not all(isinstance(d, nap.TsdFrame) for d in data):
            raise TypeError(
                "data must be a TsdFrame or a tuple/list of two TsdFrame objects."
            )
        data1, data2 = data
        if not np.array_equal(data1.index.values, data2.index.values):
            raise ValueError("The two TsdFrame objects must have identical timestamps.")
        if data1.shape[1] == 0 or data2.shape[1] == 0:
            raise ValueError("Each TsdFrame must contain at least one column.")
        time_support = data1.time_support.intersect(data2.time_support)
        pair_indices = list(product(range(data1.shape[1]), range(data2.shape[1])))
        pair_labels = list(product(data1.columns, data2.columns))
    else:
        if not isinstance(data, nap.TsdFrame):
            raise TypeError(
                "data must be a TsdFrame or a tuple/list of two TsdFrame objects."
            )
        data1 = data2 = data
        if data.shape[1] < 2:
            raise ValueError("A single TsdFrame must contain at least two columns.")
        time_support = data.time_support
        pair_indices = list(combinations(range(data.shape[1]), 2))
        pair_labels = list(combinations(data.columns, 2))

    if not isinstance(windowsize, Number):
        raise TypeError("windowsize must be a number.")
    if not np.isfinite(windowsize) or windowsize < 0:
        raise ValueError("windowsize must be finite and non-negative.")
    if epochs is not None and not isinstance(epochs, nap.IntervalSet):
        raise TypeError("epochs must be an IntervalSet or None.")
    if not isinstance(time_units, str):
        raise TypeError("time_units must be a string.")

    if np.iscomplexobj(data1.values) or np.iscomplexobj(data2.values):
        raise TypeError("Lagged cross-correlation requires real-valued data.")

    if epochs is not None:
        time_support = time_support.intersect(epochs)

    if not nap.utils._is_regularly_sampled(data1):
        raise RuntimeError("Lagged cross-correlation requires regularly sampled data.")
    time_differences = data1.time_diff().values
    if len(time_differences) == 0:
        raise RuntimeError("The sampling interval could not be determined.")
    sampling_interval = time_differences[0]
    if not np.isfinite(sampling_interval) or sampling_interval <= 0:
        raise RuntimeError("The sampling interval must be finite and positive.")

    window_seconds = nap.TsIndex.format_timestamps(
        np.array([windowsize], dtype=np.float64), time_units
    )[0]
    max_lag = int(np.floor(window_seconds / sampling_interval + 1e-12))

    indices, counts = nap._jitted_functions.jitrestrict_with_count(
        data1.index.values, time_support.start, time_support.end
    )
    pairs1 = np.asarray([pair[0] for pair in pair_indices], dtype=np.int64)
    pairs2 = np.asarray([pair[1] for pair in pair_indices], dtype=np.int64)

    correlations = _lagged_crosscorrelation(
        np.asarray(data1.values[indices], dtype=np.float64),
        np.asarray(data2.values[indices], dtype=np.float64),
        counts,
        pairs1,
        pairs2,
        max_lag,
    )
    lags = np.arange(-max_lag, max_lag + 1) * sampling_interval
    columns = pd.MultiIndex.from_tuples(pair_labels)

    return pd.DataFrame(correlations, index=lags, columns=columns, dtype=float)


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
    crosscorrs = {}

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
