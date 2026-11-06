"""
Functions based on interspike intervals: ISI distribution and burst detection.

The :doc:`user guide </user_guide/15_isi_bursts>` illustrates the burst detection methods
and their parameters with figures.
"""

from __future__ import annotations

import inspect
import numbers
import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

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
        If True, the intervals are log-transformed before binning. Default is False.
    epochs : IntervalSet, optional
        Epochs within which intervals are computed. Default is the time support of `data`.

    Returns
    -------
    pandas.DataFrame
        Counts per bin, with bin centres as index and one column per unit.

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


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------


def _check_spikes(spikes):
    if not isinstance(spikes, (nap.Ts, nap.TsGroup)):
        raise TypeError(f"`spikes` must be `Ts` or `TsGroup`, got {type(spikes)}")


def _check_positive(name, value, strict=True):
    if not isinstance(value, numbers.Real):
        raise TypeError(f"`{name}` must be a number, got {type(value)}")
    if strict and value <= 0:
        raise ValueError(f"`{name}` must be > 0")
    if not strict and value < 0:
        raise ValueError(f"`{name}` must be >= 0")


def _check_min_spikes(name, value):
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"`{name}` must be an integer, got {type(value)}")
    if value < 2:
        raise ValueError(f"`{name}` must be >= 2, a burst has at least two spikes")


# ---------------------------------------------------------------------------
# Shared machinery: runs of spikes
# ---------------------------------------------------------------------------


def _find_runs(times, max_isi_start, max_isi_burst):
    """
    Phase 1: runs of spikes. A run starts at the first of two spikes closer than
    `max_isi_start` and ends at the last spike before an interval longer than
    `max_isi_burst`. Returns (first, last) spike indices, both inclusive.
    """
    runs = []
    start = None
    for i in range(1, len(times)):
        isi = times[i] - times[i - 1]
        if start is None:
            if isi < max_isi_start:
                start = i - 1
        elif isi > max_isi_burst:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(times) - 1))
    return runs


def _merge_close_runs(times, runs, min_inter_burst_interval):
    """
    Phase 2: two consecutive runs separated by less than `min_inter_burst_interval`
    (last spike of one to first spike of the next) become one run.
    """
    merged = []
    for first, last in runs:
        if merged and times[first] - times[merged[-1][1]] < min_inter_burst_interval:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))
    return merged


def _drop_small_runs(times, runs, min_burst_duration, min_spikes_in_burst):
    """
    Phase 3: a run shorter than `min_burst_duration` or with fewer than
    `min_spikes_in_burst` spikes is not a burst.
    """
    return [
        (first, last)
        for first, last in runs
        if times[last] - times[first] >= min_burst_duration
        and last - first + 1 >= min_spikes_in_burst
    ]


def _extend_to_related_spikes(times, cores, threshold):
    """
    Extend each core to the run of spikes with consecutive intervals no longer
    than `threshold` that contains it. Cores inside the same run become one burst.
    """
    runs = _find_runs(times, threshold, threshold)
    extended = []
    for first, last in cores:
        container = next(
            (r for r in runs if r[0] <= first and last <= r[1]), (first, last)
        )
        if container not in extended:
            extended.append(container)
    return extended


def _detect_per_epoch(spikes, find_bursts):
    """
    Run `find_bursts(times)` on the spikes of each epoch of the time support,
    so no burst spans a gap in it, and return the bursts as one IntervalSet.
    """
    starts, ends, n_spikes = [], [], []
    for epoch in spikes.time_support:
        times = spikes.restrict(epoch).t
        for first, last in find_bursts(times):
            # Skip identical spike times (artifacts)
            if times[last] == times[first]:
                continue
            starts.append(times[first])
            ends.append(times[last])
            n_spikes.append(last - first + 1)
    return nap.IntervalSet(start=starts, end=ends, metadata={"n_spikes": n_spikes})


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

    def find_bursts(times):
        runs = _find_runs(times, max_isi_start, max_isi_burst)
        runs = _merge_close_runs(times, runs, min_inter_burst_interval)
        return _drop_small_runs(times, runs, min_burst_duration, min_spikes_in_burst)

    return _apply(spikes, find_bursts)


# ---------------------------------------------------------------------------
# logISI
# ---------------------------------------------------------------------------


def _interspike_intervals(spikes):
    """
    The intervals (in seconds) between consecutive spikes, within each epoch
    of the time support only.
    """
    # time_diff() cannot read a train with no spikes at all.
    if len(spikes) == 0:
        return np.empty(0)
    return spikes.time_diff().values


def _log_isi_histogram(spikes, isi_ms):
    """
    Histogram of the interspike intervals on a logarithmic axis, computed with
    `compute_isi_distribution`: 10 bin edges per decade from 1 ms to the decade
    above the longest interval, intervals under 1 ms left out, counts
    normalised to sum to one. Returns (left edges in ms, normalised counts).
    """
    decades = max(1, int(np.ceil(np.log10(isi_ms.max()))))
    edges_ms = np.logspace(0, decades, 10 * decades)
    counts = compute_isi_distribution(spikes, bins=edges_ms / 1000)[0].values
    return edges_ms[:-1], counts / counts.sum()


def _find_peaks(density):
    """
    Bins strictly higher than every bin within two bins of them, excluding the
    first and last bin.
    """
    peaks = []
    for k in range(1, len(density) - 1):
        neighbours = np.delete(density[max(0, k - 2) : k + 3], min(k, 2))
        if np.all(density[k] > neighbours):
            peaks.append(k)
    return peaks


def _log_isi_valley(spikes, max_cutoff):
    """
    The threshold of Pasquale et al. (2010): the ISI at the first valley that
    separates the intra-burst peak of the log-ISI histogram from a later peak
    with a void parameter of at least 0.7.

    Returns (has_intraburst_peak, threshold in seconds or None).
    """
    isi_ms = _interspike_intervals(spikes) * 1000
    if len(isi_ms) == 0 or not np.any(isi_ms >= 1):
        return False, None
    edges, density = _log_isi_histogram(spikes, isi_ms)
    peaks = _find_peaks(density)
    intraburst = [k for k in peaks if edges[k] < max_cutoff * 1000]
    if not intraburst:
        return False, None
    peak = max(intraburst, key=lambda k: density[k])
    for later in peaks:
        if later <= peak:
            continue
        between = density[peak : later + 1]
        valley = peak + int(np.argmin(between))
        void = 1 - between.min() / np.sqrt(density[peak] * density[later])
        if void >= 0.7:
            return True, float(edges[valley] / 1000)
    return True, None


def compute_log_isi_threshold(spikes, max_cutoff=0.1):
    """
    Compute the burst threshold of the logISI method.

    The threshold is the ISI at the minimum of the log-ISI histogram between the
    intra-burst peak (below `max_cutoff`) and the next well separated peak,
    following Pasquale et al. (2010). See
    :func:`detect_spike_bursts` for the reference.

    Parameters
    ----------
    spikes : Ts or TsGroup
        Spike times.
    max_cutoff : float, optional
        Largest ISI, in seconds, at which the intra-burst peak may lie. Default is 0.1.

    Returns
    -------
    float, None or dict
        Threshold in seconds, or None if the histogram has no intra-burst peak or no clear minimum.
        A dictionary keyed by unit for a TsGroup.

    Raises
    ------
    TypeError
        If `spikes` is not a Ts or TsGroup.
        If `max_cutoff` is not a number.
    ValueError
        If `max_cutoff` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> t = np.concatenate([s + np.arange(5) * 0.01 for s in np.arange(200) * 1.5])
    >>> spikes = nap.Ts(t=t)
    >>> round(nap.compute_log_isi_threshold(spikes), 4)
    0.0106
    """
    _check_spikes(spikes)
    _check_positive("max_cutoff", max_cutoff)
    if isinstance(spikes, nap.TsGroup):
        return {k: compute_log_isi_threshold(spikes[k], max_cutoff) for k in spikes}
    return _log_isi_valley(spikes, max_cutoff)[1]


def _log_isi_bursts_single(spikes, max_cutoff, min_spikes_in_burst, fallback):
    """logISI bursts of one Ts. Returns (bursts, whether a threshold was missing)."""
    has_peak, threshold = _log_isi_valley(spikes, max_cutoff)
    no_valley = threshold is None or threshold >= 1.0

    def find_bursts(times):
        if not has_peak or (no_valley and not fallback):
            return []
        if no_valley:
            runs = _find_runs(times, max_cutoff, max_cutoff)
            return _drop_small_runs(times, runs, 0, min_spikes_in_burst)
        if threshold > max_cutoff:
            cores = _find_runs(times, max_cutoff, max_cutoff)
            cores = _merge_close_runs(times, cores, threshold)
            cores = _drop_small_runs(times, cores, 0, min_spikes_in_burst)
            return _extend_to_related_spikes(times, cores, threshold)
        runs = _find_runs(times, threshold, threshold)
        return _drop_small_runs(times, runs, 0, min_spikes_in_burst)

    return (
        _detect_per_epoch(spikes, find_bursts),
        has_peak and no_valley and not fallback,
    )


def _detect_bursts_log_isi(
    spikes, max_cutoff=0.1, min_spikes_in_burst=3, fallback=False
):
    """logISI method of Pasquale et al. (2010), see `detect_spike_bursts`."""
    _check_positive("max_cutoff", max_cutoff)
    _check_min_spikes("min_spikes_in_burst", min_spikes_in_burst)
    if not isinstance(fallback, bool):
        raise TypeError(f"`fallback` must be a bool, got {type(fallback)}")

    if isinstance(spikes, nap.TsGroup):
        results = {
            k: _log_isi_bursts_single(
                spikes[k], max_cutoff, min_spikes_in_burst, fallback
            )
            for k in spikes
        }
        missing = [k for k, (_, no_threshold) in results.items() if no_threshold]
        if missing:
            warnings.warn(
                f"No clear minimum in the log-ISI histogram of unit(s) {missing}: "
                "no bursts returned. Set `fallback=True` to use `max_cutoff` as the "
                "threshold instead, as in the published method.",
                UserWarning,
                stacklevel=3,
            )
        return {k: bursts for k, (bursts, _) in results.items()}

    bursts, no_threshold = _log_isi_bursts_single(
        spikes, max_cutoff, min_spikes_in_burst, fallback
    )
    if no_threshold:
        warnings.warn(
            "No clear minimum in the log-ISI histogram: no bursts returned. "
            "Set `fallback=True` to use `max_cutoff` as the threshold instead, "
            "as in the published method.",
            UserWarning,
            stacklevel=3,
        )
    return bursts


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
        runs = _find_runs(times, max_isi, max_isi)
        return _drop_small_runs(times, runs, 0, min_spikes_in_burst)

    return _apply(spikes, find_bursts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_BURST_METHODS = {
    "log_isi": _detect_bursts_log_isi,
    "max_interval": _detect_bursts_max_interval,
    "instantaneous_rate": _detect_bursts_instantaneous_rate,
}


def detect_spike_bursts(spikes, method="log_isi", **params):
    """
    Detect bursts in spike trains.

    A burst is a group of spikes much closer to each other than to the spikes around them.
    Bursts are detected within each epoch of the time support, so they never span a gap in it.
    Method can be:

    - `"log_isi"` (default): logISI method [1]_. The threshold between intra- and inter-burst
      intervals is read off the log-ISI histogram of each spike train, see
      :func:`compute_log_isi_threshold`. Parameters: `max_cutoff` (float, default 0.1), the
      largest ISI in seconds at which the intra-burst peak may lie; `min_spikes_in_burst`
      (int, default 3); `fallback` (bool, default False), whether to use `max_cutoff` as the
      threshold when no clear minimum is found, as the published method does. Otherwise no
      bursts are returned and a warning is raised.
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
    method : {"log_isi", "max_interval", "instantaneous_rate"}, optional
        Detection method. Default is "log_isi".
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
    compute_log_isi_threshold : Threshold of the logISI method.
    compute_isi_distribution : Histogram the threshold is read from.

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

    >>> spikes = nap.Ts(t=[0.0, 0.1, 0.2, 0.3, 2.0, 2.05, 2.1, 5.0])
    >>> nap.detect_spike_bursts(spikes, method="max_interval", max_isi_start=0.15)
      index    start    end    n_spikes
          0        0    0.3           4
          1        2    2.1           3
    shape: (2, 2), time unit: sec.

    References
    ----------
    .. [1] Pasquale, V., Martinoia, S., & Chiappalone, M. (2010). A self-adapting
       approach for the detection of bursts and network bursts in neuronal
       cultures. Journal of Computational Neuroscience, 29(1-2), 213-229.
       https://doi.org/10.1007/s10827-009-0175-1
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
