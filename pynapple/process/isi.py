"""
Interspike interval functions: ISI distribution and burst detection.
"""

from __future__ import annotations

import inspect
import math
import numbers
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import jit

from .. import core as nap


def compute_isi_distribution(
    data: Union[nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup],
    bins: Union[int, list, npt.NDArray] = 10,
    log_scale: bool = False,
    epochs: Optional[nap.IntervalSet] = None,
) -> pd.DataFrame:
    """
    Compute the distribution of interspike intervals.

    Parameters
    ----------
    data : Ts, TsGroup, Tsd, TsdFrame or TsdTensor
        Timestamps to compute the intervals from. A TsGroup gives one distribution per unit.
    bins : int or array-like, optional
        Number of equal-width bins, or the bin edges. Default is 10.
    log_scale : bool, optional
        If True, the natural logarithm of the intervals is taken before binning.
        Default is False.
    epochs : IntervalSet, optional
        Epochs within which intervals are computed. Default is the time support of `data`.

    Returns
    -------
    pandas.DataFrame
        Counts per bin, with one column per unit and the bin centres as index, in seconds
        (or in log seconds if `log_scale` is True). The column of a unit without any
        interval is NaN.

    Raises
    ------
    TypeError
        If `data` is not a Ts, TsGroup, Tsd, TsdFrame or TsdTensor.
        If `bins` is not an int, list or array.
        If `log_scale` is not a bool.
    ValueError
        If `bins` is less than 1, or the bin edges are not increasing.

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

    if isinstance(bins, bool) or not isinstance(bins, (int, list, np.ndarray)):
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

    time_diffs = {k: np.asarray(v.values) for k, v in time_diffs.items()}

    if np.ndim(bins) == 0:
        if bins < 1:
            raise ValueError("`bins` must be positive, when an integer")
        all_time_diffs = np.hstack(list(time_diffs.values()))
        # the log of an interval of 0 is -inf, which cannot bound the bins
        all_time_diffs = all_time_diffs[np.isfinite(all_time_diffs)]
        # with no intervals at all, numpy falls back on the range (0, 1)
        histogram_range = (
            (np.min(all_time_diffs), np.max(all_time_diffs))
            if len(all_time_diffs)
            else None
        )
        bin_edges = np.histogram_bin_edges(
            all_time_diffs, bins=bins, range=histogram_range
        )
    elif np.ndim(bins) == 1:
        bin_edges = np.asarray(bins)
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError("`bins` must increase monotonically, when an array")
    else:
        raise ValueError("`bins` must be 1d, when an array")

    # a unit without intervals has no distribution: its column is NaN
    histograms = {
        i: (
            np.histogram(time_diffs[i], bins=bin_edges)[0]
            if len(time_diffs[i])
            else np.full(len(bin_edges) - 1, np.nan)
        )
        for i in time_diffs
    }

    return pd.DataFrame(
        index=(bin_edges[:-1] + bin_edges[1:]) / 2,
        data=histograms,
    )


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


def _check_spikes(spikes):
    if not isinstance(spikes, (nap.Ts, nap.TsGroup)):
        raise TypeError(f"`spikes` must be `Ts` or `TsGroup`, got {type(spikes)}")


def _check_positive(name: str, value: numbers.Real, strict: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"`{name}` must be a number, got {type(value)}")
    if not math.isfinite(value):
        raise ValueError(f"`{name}` must be finite")
    if strict and value <= 0:
        raise ValueError(f"`{name}` must be > 0")
    if not strict and value < 0:
        raise ValueError(f"`{name}` must be >= 0")


def _check_min_spikes(name: str, value: numbers.Integral) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"`{name}` must be an integer, got {type(value)}")
    if value < 2:
        raise ValueError(f"`{name}` must be >= 2, a burst has at least two spikes")


# ---------------------------------------------------------------------------
# Shared machinery: runs of spikes
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def _jitfind_runs(times, max_isi_start, max_isi_burst):
    """
    Phase 1: runs of spikes. A run starts at the first of two spikes closer than
    `max_isi_start` and ends at the last spike before an interval longer than
    `max_isi_burst`. Returns the first and the last spike index of each run,
    both inclusive.
    """
    n = len(times)
    # a run has at least two spikes and runs do not overlap
    first = np.empty(n // 2, dtype=np.int64)
    last = np.empty(n // 2, dtype=np.int64)
    count = 0
    start = -1
    for i in range(1, n):
        isi = times[i] - times[i - 1]
        if start < 0:
            if isi < max_isi_start:
                start = i - 1
        elif isi > max_isi_burst:
            first[count] = start
            last[count] = i - 1
            count += 1
            start = -1
    if start >= 0:
        first[count] = start
        last[count] = n - 1
        count += 1
    return first[:count], last[:count]


@jit(nopython=True, cache=True)
def _jitmerge_close_runs(times, first, last, min_inter_burst_interval):
    """
    Phase 2: two consecutive runs separated by less than `min_inter_burst_interval`
    (last spike of one to first spike of the next) become one run.
    """
    merged_first = np.empty(len(first), dtype=np.int64)
    merged_last = np.empty(len(first), dtype=np.int64)
    count = 0
    for k in range(len(first)):
        if (
            count > 0
            and times[first[k]] - times[merged_last[count - 1]]
            < min_inter_burst_interval
        ):
            merged_last[count - 1] = last[k]
        else:
            merged_first[count] = first[k]
            merged_last[count] = last[k]
            count += 1
    return merged_first[:count], merged_last[:count]


@jit(nopython=True, cache=True)
def _jitdrop_small_runs(times, first, last, min_burst_duration, min_spikes_in_burst):
    """
    Phase 3: a run shorter than `min_burst_duration` or with fewer than
    `min_spikes_in_burst` spikes is not a burst.
    """
    keep = np.empty(len(first), dtype=np.bool_)
    for k in range(len(first)):
        keep[k] = (
            times[last[k]] - times[first[k]] >= min_burst_duration
            and last[k] - first[k] + 1 >= min_spikes_in_burst
        )
    return first[keep], last[keep]


def _no_runs():
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def _detect_per_epoch(spikes, find_bursts):
    """
    Run `find_bursts(times)` on the spikes of each epoch of the time support,
    so no burst spans a gap in it, and return the bursts as one IntervalSet.
    `find_bursts` returns the first and the last spike index of each burst.
    """
    t = spikes.t
    # the spikes lie within the time support: those of an epoch run from its
    # first spike to the first spike of the next epoch
    bounds = np.append(np.searchsorted(t, spikes.time_support.start), len(t))
    starts, ends, n_spikes = [np.empty(0)], [np.empty(0)], [np.empty(0, dtype=int)]
    for k in range(len(bounds) - 1):
        times = t[bounds[k] : bounds[k + 1]]
        first, last = find_bursts(times)
        if len(first) == 0:
            continue
        # Skip identical spike times (artifacts)
        keep = times[last] != times[first]
        starts.append(times[first[keep]])
        ends.append(times[last[keep]])
        n_spikes.append(last[keep] - first[keep] + 1)
    return nap.IntervalSet(
        start=np.concatenate(starts),
        end=np.concatenate(ends),
        metadata={"n_spikes": np.concatenate(n_spikes)},
    )


def _apply(spikes, find_bursts):
    if isinstance(spikes, nap.TsGroup):
        return {k: _detect_per_epoch(spikes[k], find_bursts) for k in spikes}
    return _detect_per_epoch(spikes, find_bursts)


# ---------------------------------------------------------------------------
# MaxInterval
# ---------------------------------------------------------------------------


def _detect_bursts_max_interval(
    spikes,
    max_isi_start=0.17,
    max_isi_burst=0.3,
    min_inter_burst_interval=0.2,
    min_burst_duration=0.01,
    min_spikes_in_burst=3,
):
    """MaxInterval method, see `detect_spike_bursts`."""
    _check_positive("max_isi_start", max_isi_start)
    _check_positive("max_isi_burst", max_isi_burst)
    _check_positive("min_inter_burst_interval", min_inter_burst_interval, strict=False)
    _check_positive("min_burst_duration", min_burst_duration, strict=False)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)

    # floats, so that numba compiles the loops once
    max_isi_start, max_isi_burst = float(max_isi_start), float(max_isi_burst)
    min_inter_burst_interval = float(min_inter_burst_interval)
    min_burst_duration = float(min_burst_duration)

    def find_bursts(times):
        runs = _jitfind_runs(times, max_isi_start, max_isi_burst)
        runs = _jitmerge_close_runs(times, *runs, min_inter_burst_interval)
        return _jitdrop_small_runs(
            times, *runs, min_burst_duration, min_spikes_in_burst
        )

    return _apply(spikes, find_bursts)


# ---------------------------------------------------------------------------
# Instantaneous firing rate
# ---------------------------------------------------------------------------


def _detect_bursts_instantaneous_rate(
    spikes, rate_threshold=125.0, min_spikes_in_burst=2
):
    """Instantaneous firing rate threshold, see `detect_spike_bursts`."""
    _check_positive("rate_threshold", rate_threshold)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)
    max_isi = 1.0 / rate_threshold

    def find_bursts(times):
        runs = _jitfind_runs(times, max_isi, max_isi)
        return _jitdrop_small_runs(times, *runs, 0.0, min_spikes_in_burst)

    return _apply(spikes, find_bursts)


# ---------------------------------------------------------------------------
# Poisson surprise
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def _jitpoisson_surprise(times, first, n_spikes, mean_isi):
    """
    Surprise of the `n_spikes` spikes starting at index `first`: -log of the
    probability that a Poisson process with the mean rate of the train has at
    least `n_spikes - 1` spikes in their duration (sjemea's `surprise`,
    ``ppois(n - 2, upper)``).
    """
    # number of spikes expected in the duration of the group
    expected = (times[first + n_spikes - 1] - times[first]) / mean_isi
    if expected <= 0:
        return np.inf
    k = n_spikes - 1
    # The probability of `k` or more spikes is a sum of Poisson terms. The terms
    # decrease away from `expected`, so the sum runs over the shorter side.
    if expected <= k:
        # The first term can be too small for a float, so it stays in log space
        # and the other terms are summed relative to it.
        log_term = k * math.log(expected) - expected - math.lgamma(k + 1.0)
        term = 1.0
        tail = 1.0
        j = k
        while term > 1e-17 * tail:
            j += 1
            term *= expected / j
            tail += term
        return -log_term - math.log(tail)
    j = k - 1
    term = math.exp(j * math.log(expected) - expected - math.lgamma(j + 1.0))
    below = term
    while j > 0 and term > 1e-17 * below:
        term *= j / expected
        j -= 1
        below += term
    return -math.log1p(-below)


@jit(nopython=True, cache=True)
def _jitpoisson_surprise_bursts(times, mean_isi, isi_thresh):
    """
    Bursts of the Poisson surprise method, translated from `PS_method.R` of
    Cotterill & Eglen (2018) (`si.find.bursts.thresh` and
    `si.find.burst.thresh2`), with 0-based indices. `first` and `n_spikes` are
    `n` and `i` of the R code. Returns the first spike index, the last spike
    index and the surprise of each burst.
    """
    n_total = len(times)
    start_isi = mean_isi / 2
    # a burst has at least three spikes and bursts do not overlap
    burst_first = np.empty(n_total // 3, dtype=np.int64)
    burst_last = np.empty(n_total // 3, dtype=np.int64)
    burst_surprise = np.empty(n_total // 3, dtype=np.float64)
    count = 0
    first = 0
    # Unlike the R code (`n < nspikes - 2`, 1-based), the last three spikes are
    # also tested as the start of a burst.
    while first + 2 < n_total:
        isi_1 = times[first + 1] - times[first]
        isi_2 = times[first + 2] - times[first + 1]
        # A burst starts at three spikes whose two ISIs are below `start_isi`
        # and no longer than `isi_thresh`, so that no ISI of a burst exceeds
        # `isi_thresh`. The second test only matters when `isi_thresh` is set
        # below `start_isi` (the R code does not have it).
        if not (
            isi_1 < start_isi
            and isi_2 < start_isi
            and isi_1 <= isi_thresh
            and isi_2 <= isi_thresh
        ):
            first += 1
            continue

        # Phase 1: add spikes to the end while the surprise increases, looking up
        # to 10 spikes ahead, but not past an ISI longer than `isi_thresh`.
        n_spikes = 3
        s = _jitpoisson_surprise(times, first, n_spikes, mean_isi)
        while True:
            n_spikes_cur = n_spikes
            check = min(10, n_total - (n_spikes + first))
            improved = False
            while check > 0:
                check -= 1
                # `isi_thresh` takes precedence over the surprise: a spike further
                # than this from the last spike of the burst is never added, even
                # if it would increase the surprise. This departs from the R code,
                # which tests the ISI only after the surprise failed to increase,
                # and reads it at spike `i` of the train rather than of the burst.
                if times[first + n_spikes] - times[first + n_spikes - 1] > isi_thresh:
                    break
                n_spikes += 1
                s_new = _jitpoisson_surprise(times, first, n_spikes, mean_isi)
                if s_new > s:
                    improved = True
                    break
            if not improved:
                n_spikes = n_spikes_cur
                break
            s = s_new

        # Phase 2: drop spikes from the start while the surprise increases.
        while n_spikes > 3:
            s_new = _jitpoisson_surprise(times, first + 1, n_spikes - 1, mean_isi)
            if not s_new > s:
                break
            first, n_spikes, s = first + 1, n_spikes - 1, s_new

        burst_first[count] = first
        burst_last[count] = first + n_spikes - 1
        burst_surprise[count] = s
        count += 1
        first += n_spikes
    return burst_first[:count], burst_last[:count], burst_surprise[:count]


def _interspike_intervals(spikes):
    """
    The intervals (in seconds) between consecutive spikes, within each epoch
    of the time support only.
    """
    # time_diff() cannot read a train with no spikes at all.
    if len(spikes) == 0:
        return np.empty(0)
    return spikes.time_diff().values


def _poisson_surprise_single(spikes, min_surprise, max_isi_burst):
    isi = _interspike_intervals(spikes)
    if len(isi) == 0:
        return _detect_per_epoch(spikes, lambda times: _no_runs())
    mean_isi = float(isi.mean())
    isi_thresh = 2 * mean_isi if max_isi_burst is None else float(max_isi_burst)

    def find_bursts(times):
        first, last, surprise = _jitpoisson_surprise_bursts(times, mean_isi, isi_thresh)
        keep = surprise >= min_surprise
        return first[keep], last[keep]

    return _detect_per_epoch(spikes, find_bursts)


def _detect_bursts_poisson_surprise(spikes, min_surprise=5.0, max_isi_burst=None):
    """Poisson surprise method of Legendy & Salcman (1985), see `detect_spike_bursts`."""
    _check_positive("min_surprise", min_surprise, strict=False)
    if max_isi_burst is not None:
        _check_positive("max_isi_burst", max_isi_burst)
    if isinstance(spikes, nap.TsGroup):
        return {
            k: _poisson_surprise_single(spikes[k], min_surprise, max_isi_burst)
            for k in spikes
        }
    return _poisson_surprise_single(spikes, min_surprise, max_isi_burst)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_BURST_METHODS = {
    "poisson_surprise": _detect_bursts_poisson_surprise,
    "max_interval": _detect_bursts_max_interval,
    "instantaneous_rate": _detect_bursts_instantaneous_rate,
}


def detect_spike_bursts(spikes, method="poisson_surprise", **params):
    """
    Detect bursts in spike trains.

    A burst is a group of spikes much closer to each other than to the spikes around them.
    Bursts are detected within each epoch of the time support, so they never span a gap in it.
    Method can be:

    - `"poisson_surprise"` (default): Poisson surprise method [1]_, as implemented by [2]_.
      A burst starts at three spikes with ISIs below half the mean ISI and is grown, then
      trimmed, to maximise its surprise: -log of the probability of that many spikes in that
      time under a Poisson process at the mean rate of the train. Parameters: `min_surprise`
      (float, default 5), the smallest surprise of a burst; `max_isi_burst` (float or None,
      default None, i.e. twice the mean ISI), in seconds, the longest ISI within a burst.
    - `"max_interval"`: MaxInterval method [2]_ with fixed thresholds, in seconds:
      `max_isi_start` (default 0.17), `max_isi_burst` (0.3), `min_inter_burst_interval` (0.2),
      `min_burst_duration` (0.01) and `min_spikes_in_burst` (int, default 3).
    - `"instantaneous_rate"`: runs of spikes whose instantaneous firing rate (1 / ISI) stays
      above `rate_threshold` (float, default 125 Hz) [3]_, with at least `min_spikes_in_burst`
      spikes (int, default 2).

    Parameters
    ----------
    spikes : Ts or TsGroup
        Spike times.
    method : {"poisson_surprise", "max_interval", "instantaneous_rate"}, optional
        Detection method. Default is "poisson_surprise".
    **params
        Parameters of the chosen method, listed above.

    Returns
    -------
    IntervalSet or dict
        Bursts, from first to last spike, with the number of spikes in the metadata column
        `n_spikes`. A dictionary keyed by unit for a TsGroup.

    Raises
    ------
    TypeError
        If `spikes` is not a Ts or TsGroup.
        If a parameter has the wrong type or does not belong to the chosen method.
    ValueError
        If `method` is unknown or a parameter is out of range.

    See Also
    --------
    compute_isi_distribution : Distribution of the interspike intervals.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> t = np.concatenate([s + np.arange(5) * 0.01 for s in np.arange(200) * 1.5])
    >>> spikes = nap.Ts(t=t)
    >>> bursts = nap.detect_spike_bursts(spikes)
    >>> len(bursts), int(bursts.n_spikes.min()), int(bursts.n_spikes.max())
    (200, 5, 5)

    Other methods are selected with `method` and tuned with their own parameters:

    .. plot::
        :include-source: True
        :caption: Spikes (black) and the detected bursts (shaded)

        >>> import pynapple as nap
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> spikes = nap.Ts(t=[0.0, 0.1, 0.2, 0.3, 2.0, 2.05, 2.1, 5.0])
        >>> bursts = nap.detect_spike_bursts(
        ...     spikes, method="max_interval", max_isi_start=0.15
        ... )
        >>> bursts
          index    start    end    n_spikes
              0        0    0.3           4
              1        2    2.1           3
        shape: (2, 2), time unit: sec.
        >>> fig, ax = plt.subplots(figsize=(8, 2))
        >>> _ = ax.eventplot(spikes.t, color="k")
        >>> for start, end in zip(bursts.start, bursts.end):
        ...     _ = ax.axvspan(start, end, color="tab:orange", alpha=0.3)
        >>> _ = ax.set(xlabel="Time (s)", yticks=[])
        >>> plt.show()

    References
    ----------
    .. [1] Legendy, C. R., & Salcman, M. (1985). Bursts and recurrences of bursts
       in the spike trains of spontaneously active striate cortex neurons.
       Journal of Neurophysiology, 53(4), 926-939.
       https://doi.org/10.1152/jn.1985.53.4.926
    .. [2] Cotterill, E., & Eglen, S. J. (2018). Burst detection methods.
       arXiv:1802.01287. https://arxiv.org/abs/1802.01287
    .. [3] Leonardo, A., & Fee, M. S. (2005). Ensemble coding of vocal control
       in birdsong. Journal of Neuroscience, 25(3), 652-661.
       https://doi.org/10.1523/JNEUROSCI.3036-04.2005
    """
    _check_spikes(spikes)
    if not isinstance(method, str) or method not in _BURST_METHODS:
        raise ValueError(
            f"`method` must be one of {list(_BURST_METHODS)}, got {method!r}"
        )
    func = _BURST_METHODS[method]
    valid = [p for p in inspect.signature(func).parameters if p != "spikes"]
    unknown = sorted(set(params) - set(valid))
    if unknown:
        raise TypeError(
            f"Unknown parameter(s) {unknown} for method {method!r}. "
            f"Valid parameters are {valid}"
        )
    return func(spikes, **params)
