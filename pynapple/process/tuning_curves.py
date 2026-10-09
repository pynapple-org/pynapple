"""
Functions to compute n-dimensional tuning curves.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterable
from functools import wraps
from typing import TYPE_CHECKING, Dict, Hashable, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import jit

from .. import core as nap
from ..core._core_functions import _concat_ranges
from ..core._jitted_functions import jitvaluefrom

if TYPE_CHECKING:
    import xarray as xr


@jit(nopython=True, cache=True)
def _jitcount_spikes(idx, units, feature_bins, n_units, n_flat):
    """Count the spikes of each unit, in total and in each feature bin.

    Parameters
    ----------
    idx : ndarray[int64]
        The matched feature sample of each spike, or -1 if none.
    units : ndarray[int]
        The position of the unit of each spike, in ``range(n_units)``.
    feature_bins : ndarray[intp]
        The flat bin of each feature sample, in ``range(n_flat)``.
    n_units, n_flat : int
        The number of units and of flat bins.

    Returns
    -------
    counts : ndarray[float64]
        ``(n_units, n_flat)``. The matched spikes of each unit in each bin.
    n_spikes : ndarray[int64]
        ``(n_units,)``. All the spikes of each unit, matched or not.
    """
    counts = np.zeros((n_units, n_flat))
    n_spikes = np.zeros(n_units, dtype=np.int64)
    for i in range(idx.shape[0]):
        n_spikes[units[i]] += 1
        if idx[i] >= 0:
            counts[units[i], feature_bins[idx[i]]] += 1
    return counts, n_spikes


def _match_and_count_spikes(group, features, feature_bins, n_flat, bin_edges, epochs):
    """Match each spike to a feature sample, and count the spikes of each unit
    in each feature bin. Also return the rate of each unit.

    Equivalent to restricting ``group`` to ``epochs``, matching it against the
    feature's bin index with `value_from` and histogramming each unit, without
    building either intermediate group.

    Returns
    -------
    counts : ndarray
        ``(n_units, *n_bins)`` float64 spike counts per unit and bin.
    rates : ndarray
        ``(n_units,)`` rates within ``epochs``, as ``group.restrict(epochs).rates``.
    """
    # a lazy group reads only the spikes in the span of `epochs`
    group = group._load_in_memory(epochs)
    times = group._times
    target_times = features.index.values

    in_start = np.searchsorted(times, epochs.start, side="left")
    in_stop = np.searchsorted(times, epochs.end, side="right")
    # the matched feature sample of each spike in the epochs, -1 if none
    idx = jitvaluefrom(
        times,
        target_times,
        in_start,
        in_stop,
        np.searchsorted(target_times, epochs.start, side="left"),
        np.searchsorted(target_times, epochs.end, side="right"),
        1,  # closest
    )
    # the unit of each spike in the epochs
    units = _concat_ranges(group._cluster_positions, in_start, in_stop, copy=False)
    n_units = len(group.index)
    counts, n_in_epochs = _jitcount_spikes(idx, units, feature_bins, n_units, n_flat)
    padded_shape = [n_units, *[len(edges) + 1 for edges in bin_edges]]
    interior = (slice(None), *[slice(1, -1) for _ in bin_edges])
    counts = counts.reshape(padded_shape)[interior]

    duration = np.sum(epochs.end - epochs.start)
    if duration > 0:
        rates = n_in_epochs / duration
    else:
        rates = np.full(n_units, np.nan)
    return counts, rates


def _bin_edges(sample, bins, range):
    """The bin edges ``np.histogramdd(sample, bins=bins, range=range)`` would use.

    Without a ``range``, ``histogramdd`` derives edges from each feature's minimum
    and maximum only, so histogramming those two rows gives the same edges (and
    raises the same errors) without binning the whole sample.
    """
    sample = np.asarray(sample)
    if sample.ndim == 1:
        sample = sample[:, None]
    if len(sample):
        sample = np.stack([sample.min(axis=0), sample.max(axis=0)])
    return np.histogramdd(sample, bins=bins, range=range)[1]


def _flat_bin_index(sample, bin_edges):
    """Assign each sample to a bin of the ``histogramdd`` grid, once.

    ``np.histogramdd`` re-derives this assignment on every call. When many
    histograms share one set of edges -- one per unit or per column -- computing it
    once and reusing it turns each subsequent histogram into a single
    :func:`numpy.bincount`.

    The grid is the padded one numpy uses internally: each dimension gets two extra
    bins holding the samples that fall outside the edges, so out-of-range samples
    (and NaN, which sorts to the far end) are carried along and dropped later by
    :func:`_histogram_from_bin_index` rather than needing a mask here.

    Parameters
    ----------
    sample : array_like
        ``(n_samples,)`` or ``(n_samples, n_features)``. A 1-D array is read as
        many samples of one feature.
    bin_edges : sequence of ndarray
        One monotonically increasing edge array per feature, as returned by
        :func:`numpy.histogramdd`.

    Returns
    -------
    flat_index : ndarray
        ``(n_samples,)`` index into the flattened padded grid.
    n_flat : int
        Size of the flattened padded grid.
    """
    sample = np.asarray(sample)
    if sample.ndim == 1:
        # (n_samples,) is n samples of one feature, never one sample of n
        sample = sample[:, None]

    n_bins = np.empty(len(bin_edges), dtype=np.intp)
    per_dim = []
    for i, edges in enumerate(bin_edges):
        n_bins[i] = len(edges) + 1
        index = np.searchsorted(edges, sample[:, i], side="right")
        # numpy puts samples sitting exactly on the rightmost edge in the last
        # real bin rather than in the outlier bin above it
        index[sample[:, i] == edges[-1]] -= 1
        per_dim.append(index)

    return np.ravel_multi_index(per_dim, n_bins), int(n_bins.prod())


def _histogram_from_bin_index(flat_index, n_flat, bin_edges, weights=None):
    """Histogram from a precomputed :func:`_flat_bin_index`, dropping outliers.

    Equivalent to ``np.histogramdd(sample, bins=bin_edges, weights=weights)[0]``
    for the sample the index was built from.
    """
    padded_shape = [len(edges) + 1 for edges in bin_edges]
    counts = np.bincount(flat_index, weights=weights, minlength=n_flat)
    if weights is None:
        # bincount counts as int64, histogramdd always returns float64
        counts = counts.astype(np.float64)
    interior = tuple(slice(1, -1) for _ in bin_edges)
    return counts.reshape(padded_shape)[interior]


def compute_tuning_curves(
    data: Union[nap.TsGroup, nap.Ts, nap.TsdFrame, nap.Tsd],
    features: Union[nap.Tsd, nap.TsdFrame],
    bins: Union[int, Sequence[int], Sequence[npt.ArrayLike], npt.ArrayLike] = 10,
    range: Optional[
        Union[Tuple[float, float], Sequence[Optional[Tuple[float, float]]]]
    ] = None,
    epochs: Optional[nap.IntervalSet] = None,
    fs: Optional[float] = None,
    feature_names: Optional[Sequence[str]] = None,
    return_pandas: bool = False,
    return_counts: bool = False,
) -> Union[xr.DataArray, pd.DataFrame]:
    """
    Compute the tuning curve of each unit of ``data`` as a function of one or
    more features.

    A tuning curve gives the mean response of a unit for each value of a
    feature. An example is the firing rate of a neuron for each head
    direction. The function divides the feature space into bins, and gives one
    value for each unit and each bin.

    Let :math:`x(t)` be the feature samples, at the sampling rate :math:`f_s`,
    and let :math:`B` be a bin. The **occupancy** :math:`\\mathrm{occ}(B)` is
    the number of feature samples in :math:`B`. Thus
    :math:`\\mathrm{occ}(B) / f_s` is the time spent in :math:`B`.

    **Spike times** (TsGroup or Ts). The function matches each spike to the
    closest feature sample. Let :math:`N_u(B)` be the number of spikes of the
    unit :math:`u` whose matched sample is in :math:`B`. The tuning curve is the
    firing rate in :math:`B`, in Hz:

    .. math::

        \\lambda_u(B) = \\frac{N_u(B)}{\\mathrm{occ}(B) / f_s}
        = f_s \\frac{N_u(B)}{\\mathrm{occ}(B)}

    **Continuous values** (TsdFrame or Tsd), for example calcium imaging. The
    function matches each sample :math:`y_u(t)` of the data to the closest
    feature sample. Let :math:`T(B)` be the set of data timestamps whose matched
    sample is in :math:`B`. The tuning curve is the mean value in :math:`B`:

    .. math::

        \\bar{y}_u(B) = \\frac{1}{|T(B)|} \\sum_{t \\in T(B)} y_u(t)

    The function uses only the spikes, the data samples and the feature samples
    in ``epochs``. A spike or a data sample matches only a feature sample of the
    same epoch. A feature value outside the bin edges is not in any bin.

    Empty bins:

    - If :math:`\\mathrm{occ}(B) = 0`, the tuning curve is NaN in :math:`B`.
    - For continuous values, if :math:`\\mathrm{occ}(B) > 0` but :math:`T(B)` is
      empty, the tuning curve is 0 in :math:`B`.

    Parameters
    ----------
    data : TsGroup, Ts, TsdFrame or Tsd
        The responses: spike times (a TsGroup, or a Ts for one unit), or
        continuous values (a TsdFrame, or a Tsd for one unit). A Ts or a Tsd
        becomes the unit 0. The function uses a TsGroup member that is a Tsd as
        a Ts, and gives a warning.
    features : Tsd or TsdFrame
        The features, with one column for each feature. Examples are the
        position, the head direction or the speed.
    bins : int, sequence of int, or sequence of arrays, optional
        The bins of the features:

        - An int: the number of bins of each feature. Default is 10.
        - A sequence of int: the number of bins of each feature, in order.
        - A sequence of arrays: the increasing bin edges of each feature. For
          one feature, one array of edges is also accepted.

        Each bin includes its left edge. The last bin also includes its right
        edge.
    range : sequence of (float, float), optional
        The lower edge and the upper edge of each feature, when ``bins`` does
        not give the edges. An entry of None uses the minimum and the maximum of
        the feature. For one feature, one tuple is also accepted. If None
        (default), the function uses the minimum and the maximum of each
        feature.
    epochs : IntervalSet, optional
        The epochs to use. If None (default), the function uses the time
        support of ``features``.
    fs : float, optional
        The sampling rate :math:`f_s` of the features, in Hz. If None
        (default), the function uses 1 divided by the mean interval between two
        feature samples in ``epochs``. Give the exact value when you know it.
    feature_names : list of str, optional
        The name of each feature, used as the dimension names of the result. If
        None (default), the function uses the column names of ``features``, or
        ``"0"`` for a Tsd.
    return_pandas : bool, optional
        If True, return a pandas.DataFrame with one row for each bin and one
        column for each unit. Only for one feature. The DataFrame does not
        keep the attributes. Default is False.
    return_counts : bool, optional
        Only for spike times. If True, return the spike counts
        :math:`N_u(B)`, not divided by the occupancy. The attributes keep the
        occupancy and :math:`f_s`, so that you can divide later. Default is
        False.

    Returns
    -------
    xarray.DataArray
        The tuning curves. The first dimension is ``"unit"``, then one
        dimension for each feature, with the bin centers as coordinates. The
        attributes are:

        - ``occupancy``: :math:`\\mathrm{occ}(B)`, with one value for each bin.
        - ``bin_edges``: one array of edges for each feature.
        - ``fs``: the sampling rate :math:`f_s`.
        - ``rates``: only for spike times. The mean firing rate of each unit
          in ``epochs``: its number of spikes divided by the total duration of
          the epochs.
    pandas.DataFrame
        If ``return_pandas`` is True.

    Raises
    ------
    TypeError
        If ``data`` or ``features`` has an incorrect type, if
        ``feature_names`` is not a list of str, if ``epochs`` is not an
        IntervalSet, if ``fs`` is not a number, or if ``return_pandas`` or
        ``return_counts`` is not a boolean.
    ValueError
        If ``feature_names`` or ``bins`` does not have one entry for each
        feature, or if ``range`` is one tuple with more than one feature.

    See Also
    --------
    compute_response_per_epoch : The mean response of each unit in each epoch.
    compute_mutual_information : The information that tuning curves give.
    decode_bayes : Decode the features from spikes and tuning curves.

    Examples
    --------
    Spike times and one feature:

        >>> import pynapple as nap
        >>> import numpy as np; np.random.seed(42)
        >>> group = nap.TsGroup({
        ...     1: nap.Ts(np.arange(0, 100, 0.1)),
        ...     2: nap.Ts(np.arange(0, 100, 0.2))
        ... })
        >>> feature = nap.Tsd(d=np.arange(0, 100, 0.1) % 1, t=np.arange(0, 100, 0.1))
        >>> tcs = nap.compute_tuning_curves(group, feature, bins=10)
        >>> tcs
        <xarray.DataArray (unit: 2, 0: 10)> Size: 160B
        array([[10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
               [10.,  0., 10.,  0., 10.,  0., 10.,  0., 10.,  0.]])
        Coordinates:
          * unit     (unit) int64 16B 1 2
          * 0        (0) float64 80B 0.045 0.135 0.225 0.315 ... 0.585 0.675 0.765 0.855
        Attributes:
            occupancy:  [100. 100. 100. 100. 100. 100. 100. 100. 100. 100.]
            bin_edges:  [array([0.  , 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72,...
            fs:         10.0
            rates:      [10.01001001  5.00500501]

    With more than one feature, the tuning curves have more than one dimension.
    ``bins`` can give the number of bins of each feature:

        >>> features = nap.TsdFrame(
        ...     d=np.stack(
        ...         [
        ...             np.arange(0, 100, 0.1) % 1,
        ...             np.arange(0, 100, 0.1) % 2
        ...         ],
        ...         axis=1
        ...     ),
        ...     t=np.arange(0, 100, 0.1)
        ... )
        >>> tcs = nap.compute_tuning_curves(group, features, bins=[5, 3])
        >>> tcs
        <xarray.DataArray (unit: 2, 0: 5, 1: 3)> Size: 240B
        array([[[10., 10., nan],
                [10., 10., 10.],
                [10., nan, 10.],
                [10., 10., 10.],
                [nan, 10., 10.]],
        ...
               [[ 5.,  5., nan],
                [ 5., 10.,  0.],
                [ 5., nan,  5.],
                [10.,  0.,  5.],
                [nan,  5.,  5.]]])
        Coordinates:
          * unit     (unit) int64 16B 1 2
          * 0        (0) float64 40B 0.09 0.27 0.45 0.63 0.81
          * 1        (1) float64 24B 0.3167 0.95 1.583
        Attributes:
            occupancy:  [[100. 100.   0.]\\n [100.  50.  50.]\\n [100.   0. 100.]\\n [ 5...
            bin_edges:  [array([0.  , 0.18, 0.36, 0.54, 0.72, 0.9 ]), array([0.      ...
            fs:         10.0
            rates:      [10.01001001  5.00500501]

    ``bins`` can also give the bin edges of each feature:

        >>> tcs = nap.compute_tuning_curves(
        ...     group,
        ...     features,
        ...     bins=[np.linspace(0, 1, 5), np.linspace(0, 2, 3)]
        ... )
        >>> tcs
        <xarray.DataArray (unit: 2, 0: 4, 1: 2)> Size: 128B
        array([[[10.        , 10.        ],
                [10.        , 10.        ],
                [10.        , 10.        ],
                [10.        , 10.        ]],
        ...
               [[ 6.66666667,  6.66666667],
                [ 5.        ,  5.        ],
                [ 3.33333333,  3.33333333],
                [ 5.        ,  5.        ]]])
        Coordinates:
          * unit     (unit) int64 16B 1 2
          * 0        (0) float64 32B 0.125 0.375 0.625 0.875
          * 1        (1) float64 16B 0.5 1.5
        Attributes:
            occupancy:  [[150. 150.]\\n [100. 100.]\\n [150. 150.]\\n [100. 100.]]
            bin_edges:  [array([0.  , 0.25, 0.5 , 0.75, 1.  ]), array([0., 1., 2.])]
            fs:         10.0
            rates:      [10.01001001  5.00500501]

    With continuous values (for example calcium imaging), the tuning curves are
    the mean values in each bin:

        >>> frame = nap.TsdFrame(d=np.random.rand(2000, 3), t=np.arange(0, 100, 0.05))
        >>> tcs = nap.compute_tuning_curves(frame, feature, bins=10)
        >>> tcs
        <xarray.DataArray (unit: 3, 0: 10)> Size: 240B
        array([[0.50188524, 0.47039876, 0.52645761, 0.50672026, 0.51754981,
                0.49333603, 0.50215856, 0.51170549, 0.53723666, 0.51711987],
               [0.49673344, 0.45840472, 0.49148886, 0.49107241, 0.53621963,
                0.52414748, 0.49547209, 0.47547469, 0.4688882 , 0.49582293],
               [0.47057384, 0.48928271, 0.48659081, 0.49216857, 0.50806709,
                0.50675432, 0.47783004, 0.51557003, 0.41351384, 0.52013083]])
        Coordinates:
          * unit     (unit) int64 24B 0 1 2
          * 0        (0) float64 80B 0.045 0.135 0.225 0.315 ... 0.585 0.675 0.765 0.855
        Attributes:
            occupancy:  [100. 100. 100. 100. 100. 100. 100. 100. 100. 100.]
            bin_edges:  [array([0.  , 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72,...
            fs:         10.0
    """
    import xarray as xr

    # check data
    if not isinstance(data, (nap.TsdFrame, nap.TsGroup, nap.Ts, nap.Tsd)):
        raise TypeError("data should be a TsdFrame, TsGroup, Ts, or Tsd.")

    # check features
    if not isinstance(features, (nap.TsdFrame, nap.Tsd)):
        raise TypeError("features should be a Tsd or TsdFrame.")

    # check feature names
    if feature_names is None:
        feature_names = (
            features.columns if isinstance(features, nap.TsdFrame) else ["0"]
        )
    else:
        if (
            not hasattr(feature_names, "__len__")
            or isinstance(feature_names, str)
            or not all(isinstance(n, str) for n in feature_names)
        ):
            raise TypeError("feature_names should be a list of strings.")
        if len(feature_names) != (
            1 if isinstance(features, nap.Tsd) else features.shape[-1]
        ):
            raise ValueError("feature_names should match the number of features.")

    # check bins
    n_features = 1 if features.ndim == 1 else features.shape[1]
    if n_features == 1 and np.ndim(bins) == 1 and len(bins) > 1:
        bins = [bins]
    try:
        n_bin_specs = len(bins)
    except TypeError:
        n_bin_specs = None
    if n_bin_specs is not None and n_bin_specs != n_features:
        raise ValueError(
            "bins should contain one specification per feature "
            f"(expected {n_features}, got {n_bin_specs}). To use explicit bin "
            "edges with multiple features, pass one array per feature."
        )

    # check epochs
    if epochs is None:
        epochs = features.time_support
    elif isinstance(epochs, nap.IntervalSet):
        features = features.restrict(epochs)
    else:
        raise TypeError("epochs should be an IntervalSet.")
    if isinstance(data, nap.Ts):
        data = nap.TsGroup({0: data})
    if not isinstance(data, nap.TsGroup):
        # spikes are restricted while being matched, see `_match_and_count_spikes`
        data = data.restrict(epochs)

    # check fs
    if fs is None:
        fs = 1 / np.mean(features.time_diff(epochs=epochs).values)
    if not isinstance(fs, (int, float)):
        raise TypeError("fs should be a number (int or float)")

    # check range
    if range is not None and isinstance(range, tuple):
        if features.ndim == 1 or features.shape[1] == 1:
            range = [range]
        else:
            raise ValueError(
                "range should be a sequence of tuples, one for each feature."
            )

    # check return_pandas
    if (
        return_pandas != 1
        and return_pandas != 0
        and not isinstance(return_pandas, bool)
    ):
        raise TypeError("return_pandas should be a boolean.")

    # check return_counts
    if (
        return_counts != 1
        and return_counts != 0
        and not isinstance(return_counts, bool)
    ):
        raise TypeError("return_counts should be a boolean.")

    # occupancy
    # The feature is binned once: the same bin index gives the occupancy and,
    # for spikes, each spike's bin.
    bin_edges = _bin_edges(features, bins, range)
    feature_bins, n_flat = _flat_bin_index(features, bin_edges)
    occupancy = _histogram_from_bin_index(feature_bins, n_flat, bin_edges)

    # tuning curves
    # np.asarray drops any name carried by a pandas Index (TsdFrame.columns
    # loaded from NWB is named "id"), which xarray would otherwise use as the
    # coordinate dimension instead of "unit".
    keys = np.asarray(
        data.keys()
        if isinstance(data, nap.TsGroup)
        else data.columns if isinstance(data, nap.TsdFrame) else [0]
    )
    rates = None
    if isinstance(data, nap.TsGroup):
        # SPIKES
        for n in data.index[data._is_tsd]:
            warnings.warn(f"TsGroup entry {n} was not a Ts, but treating it as one!")

        # Every unit is matched against the same feature, so the whole group is
        # matched at once, and against the feature's bin index rather than its
        # values: the bin of each spike's matching sample is counted directly,
        # with exactly the matching and epochs `value_from` would have used on
        # the feature itself.
        tcs, rates = _match_and_count_spikes(
            data, features, feature_bins, n_flat, bin_edges, epochs
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            if not return_counts:
                tcs = (tcs / occupancy) * fs
    else:
        # RATES
        values = data.value_from(features)
        if isinstance(data, nap.Tsd):
            data = np.expand_dims(data.values, -1)
        else:
            # the raw array: `data[:, i]` below would rebuild a Tsd per column
            data = data.values
        # every column is histogrammed over the same samples and the same edges,
        # so the bin assignment is computed once and only the weights change
        flat_index, n_flat = _flat_bin_index(values, bin_edges)
        counts = _histogram_from_bin_index(flat_index, n_flat, bin_edges)
        counts[counts == 0] = np.nan
        tcs = np.stack(
            [
                _histogram_from_bin_index(
                    flat_index, n_flat, bin_edges, weights=data[:, i]
                )
                for i in np.arange(len(keys))
            ]
        )
        tcs /= counts
        tcs[np.isnan(tcs)] = 0.0
        tcs[:, occupancy == 0.0] = np.nan

    attrs = {"occupancy": occupancy, "bin_edges": bin_edges, "fs": fs}
    if rates is not None:
        attrs["rates"] = rates
    tcs = xr.DataArray(
        tcs,
        coords={
            "unit": keys,
            **{
                str(feature_name): e[:-1] + np.diff(e) / 2
                for feature_name, e in zip(feature_names, bin_edges)
            },
        },
        attrs=attrs,
    )
    if return_pandas:
        return tcs.to_pandas().T
    else:
        return tcs


def compute_response_per_epoch(
    data: Union[nap.TsGroup, nap.Ts, nap.TsdFrame, nap.Tsd],
    epochs_dict: Dict[Hashable, nap.IntervalSet],
    return_pandas: bool = False,
) -> Union[xr.DataArray, pd.DataFrame]:
    """
    Compute the mean response of each unit of ``data`` in each set of epochs.

    Use this function for discrete conditions, for example stimuli that are
    presented more than once. Each entry of ``epochs_dict`` is one condition,
    and its IntervalSet holds all the presentations of the condition.

    Let :math:`E` be the IntervalSet of one condition, and let :math:`|E|` be
    its total duration, in seconds.

    **Spike times** (TsGroup or Ts). Let :math:`N_u(E)` be the number of
    spikes of the unit :math:`u` in :math:`E`. The response is the firing
    rate in :math:`E`, in Hz:

    .. math::

        \\lambda_u(E) = \\frac{N_u(E)}{|E|}

    **Continuous values** (TsdFrame or Tsd), for example calcium imaging. Let
    :math:`T(E)` be the set of data timestamps in :math:`E`. The response is
    the mean value in :math:`E`:

    .. math::

        \\bar{y}_u(E) = \\frac{1}{|T(E)|} \\sum_{t \\in T(E)} y_u(t)

    A spike or a sample exactly on the start or on the end of an epoch is in
    the epoch. The IntervalSets of two conditions can overlap, but the epochs
    of one IntervalSet cannot overlap. For continuous values, a condition
    with no data sample gives NaN.

    Parameters
    ----------
    data : TsGroup, Ts, TsdFrame or Tsd
        The responses: spike times (a TsGroup, or a Ts for one unit), or
        continuous values (a TsdFrame, or a Tsd for one unit). A Ts or a Tsd
        becomes the unit 0.
    epochs_dict : dict of IntervalSet
        One entry for each condition. The keys are the names of the
        conditions, and the values are their epochs. The dict must not be
        empty.
    return_pandas : bool, optional
        If True, return a pandas.DataFrame with one row for each condition and
        one column for each unit. Default is False.

    Returns
    -------
    xarray.DataArray
        The responses, with the dimensions ``"unit"`` and ``"epochs"``. The
        coordinates are the unit labels and the keys of ``epochs_dict``.
    pandas.DataFrame
        If ``return_pandas`` is True.

    Raises
    ------
    TypeError
        If ``data`` has an incorrect type, if ``epochs_dict`` is not a
        non-empty dict of IntervalSets, or if ``return_pandas`` is not a
        boolean.

    See Also
    --------
    compute_tuning_curves : The response of each unit as a function of
        continuous features.

    Examples
    --------
    Spike times, with two conditions:

        >>> import pynapple as nap
        >>> import numpy as np; np.random.seed(42)
        >>> epochs_dict =  {
        ...     "stim0": nap.IntervalSet(start=0, end=30),
        ...     "stim1":nap.IntervalSet(start=60, end=90)
        ... }
        >>> group = nap.TsGroup({
        ...     1: nap.Ts(np.arange(0, 100, 0.1)),
        ...     2: nap.Ts(np.arange(0, 100, 0.2))
        ... })
        >>> tcs = nap.compute_response_per_epoch(group, epochs_dict)
        >>> tcs
        <xarray.DataArray (unit: 2, epochs: 2)> Size: 32B
        array([[10.03333333, 10.03333333],
               [ 5.03333333,  5.03333333]])
        Coordinates:
          * unit     (unit) int64 16B 1 2
          * epochs   (epochs) <U5 40B 'stim0' 'stim1'

    The same responses as a pandas.DataFrame:

        >>> nap.compute_response_per_epoch(
        ...     group, epochs_dict, return_pandas=True
        ... )  # doctest: +NORMALIZE_WHITESPACE
        unit            1         2
        epochs
        stim0   10.033333  5.033333
        stim1   10.033333  5.033333

    Continuous values (for example calcium imaging), with the mean value in
    each condition:

        >>> frame = nap.TsdFrame(d=np.random.rand(2000, 3), t=np.arange(0, 100, 0.05))
        >>> tcs = nap.compute_response_per_epoch(frame, epochs_dict)
        >>> tcs
        <xarray.DataArray (unit: 3, epochs: 2)> Size: 48B
        array([[0.50946668, 0.50897635],
               [0.48343249, 0.48191892],
               [0.50063158, 0.48748094]])
        Coordinates:
          * unit     (unit) int64 24B 0 1 2
          * epochs   (epochs) <U5 40B 'stim0' 'stim1'
    """
    import xarray as xr

    # check data
    if not isinstance(data, (nap.TsdFrame, nap.TsGroup, nap.Ts, nap.Tsd)):
        raise TypeError("data should be a TsdFrame, TsGroup, Ts, or Tsd.")

    # check epochs_dict
    if (
        not isinstance(epochs_dict, dict)
        or len(epochs_dict) == 0
        or not all(isinstance(epoch, nap.IntervalSet) for epoch in epochs_dict.values())
    ):
        raise TypeError("epochs_dict should be a dictionary of IntervalSets.")

    # check return_pandas
    if (
        return_pandas != 1
        and return_pandas != 0
        and not isinstance(return_pandas, bool)
    ):
        raise TypeError("return_pandas should be a boolean.")

    # tuning curves
    # See compute_tuning_curves: np.asarray drops the pandas Index name.
    keys = np.asarray(
        data.keys()
        if isinstance(data, nap.TsGroup)
        else data.columns if isinstance(data, nap.TsdFrame) else [0]
    )
    if isinstance(data, (nap.TsGroup, nap.Ts)):
        # SPIKES
        if isinstance(data, nap.Ts):
            data = nap.TsGroup({0: data}, time_support=data.time_support)
        tcs = np.stack(
            [
                data.restrict(epoch).count().values.sum(axis=0) / epoch.tot_length("s")
                for epoch in epochs_dict.values()
            ],
            axis=1,
        )
    else:
        # RATES
        if isinstance(data, nap.Tsd):
            data = nap.TsdFrame(
                d=np.expand_dims(data.values, -1),
                t=data.times(),
                time_support=data.time_support,
            )
        tcs = np.stack(
            [
                data.restrict(epoch).values.mean(axis=0)
                for epoch in epochs_dict.values()
            ],
            axis=1,
        )
    tcs = xr.DataArray(
        tcs,
        coords={
            "unit": keys,
            "epochs": list(epochs_dict.keys()),
        },
    )
    if return_pandas:
        return tcs.to_pandas().T
    else:
        return tcs


def compute_mutual_information(
    tuning_curves: xr.DataArray,
    rates: Optional[Union[list, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Compute the mutual information between the firing of each unit and the
    features, from n-dimensional tuning curves.

    The function uses the metric of Skaggs et al. [1]_. It gives how much
    information the spikes of a unit carry about the features, for example
    about the position.

    The mutual information in bits per second is given by:

    .. math::

        I_\\text{bits/s} = \\sum_x P(x) \\lambda(x) \\log_2 \\left( \\frac{\\lambda(x)}{\\bar{\\lambda}} \\right)

    where:

    - :math:`P(x)` is the probability of being in bin :math:`x` (occupancy),
    - :math:`\\lambda(x)` is the firing rate of the neuron in bin :math:`x`,
    - :math:`\\bar{\\lambda}` is the overall mean firing rate.

    The information per spike is computed by dividing the result by the mean firing rate:

    .. math::

        I_\\text{bits/spike} = \\frac{I}{\\bar{\\lambda}}

    :math:`P(x)` is the occupancy of the bin :math:`x` divided by the sum of
    the occupancy. A bin with :math:`\\lambda(x) = 0` adds 0 to the sum, and
    a bin with no occupancy (NaN) is not in the sum.

    Parameters
    ----------
    tuning_curves : xarray.DataArray
        The tuning curves, as :func:`~pynapple.process.tuning_curves.compute_tuning_curves`
        returns them. The attribute ``occupancy`` is necessary.
    rates : list or numpy.ndarray, optional
        The mean firing rate :math:`\\bar{\\lambda}` of each unit, in the
        order of the units. If None (default), the function uses the attribute
        ``rates`` of ``tuning_curves``: the mean firing rates in the epochs of
        the tuning curves. If ``tuning_curves`` has no attribute ``rates``,
        the function uses :math:`\\sum_x P(x) \\lambda(x)`, and gives a
        warning.

    Returns
    -------
    pandas.DataFrame
        One row for each unit, with the unit labels as index. The column
        ``"bits/sec"`` is :math:`I_\\text{bits/s}`, and the column
        ``"bits/spike"`` is :math:`I_\\text{bits/spike}`.

    Raises
    ------
    TypeError
        If ``tuning_curves`` is not an xarray.DataArray, or if ``rates`` is not
        a list or a numpy.ndarray.
    ValueError
        If ``rates`` does not have one value for each unit, or if
        ``tuning_curves`` has no attribute ``occupancy``.

    Warns
    -----
    UserWarning
        If ``rates`` is None and ``tuning_curves`` has no attribute ``rates``.

    See Also
    --------
    compute_tuning_curves : Compute the tuning curves and their occupancy.

    References
    ----------
    .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
           An information-theoretic approach to deciphering the hippocampal code.
           In Advances in neural information processing systems (pp. 1030-1037).

    Examples
    --------
    Two units, each with spikes in one bin of the feature:

        >>> import pynapple as nap
        >>> import numpy as np; np.random.seed(42)
        >>> epoch = nap.IntervalSet([0, 100])
        >>> t = np.arange(0, 100, 0.01)
        >>> feature = nap.Tsd(t=t, d=np.clip(t*0.01 + np.random.normal(0, 0.02, len(t)), 0, 1), time_support=epoch)
        >>> group = nap.TsGroup({
        ...     1: nap.Ts(t[(feature.values >= 0.2) & (feature.values < 0.3)]),
        ...     2: nap.Ts(t[(feature.values >= 0.7) & (feature.values < 0.8)])
        ... }, time_support=epoch)
        >>> tcs = nap.compute_tuning_curves(group, feature, bins=10)
        >>> tcs
        <xarray.DataArray (unit: 2, 0: 10)> Size: 160B
        array([[  0.,   0., 100.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
               [  0.,   0.,   0.,   0.,   0.,   0.,   0., 100.,   0.,   0.]])
        Coordinates:
          * unit     (unit) int64 16B 1 2
          * 0        (0) float64 80B 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95
        Attributes:
            occupancy:  [ 985. 1009. 1014.  996.  993. 1008.  991. 1008.  999.  997.]
            bin_edges:  [array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])]
            fs:         100.0
            rates:      [10.14 10.08]

    The mutual information of each unit:

        >>> MI = nap.compute_mutual_information(tcs)
        >>> MI
            bits/sec  bits/spike
        1  33.480966    3.301870
        2  33.369159    3.310432
    """
    import xarray as xr

    if not isinstance(tuning_curves, xr.DataArray):
        raise TypeError(
            "tuning_curves should be an xr.DataArray as computed by compute_tuning_curves."
        )

    if rates is not None:
        if not isinstance(rates, (list, np.ndarray)):
            raise TypeError("rates should be a list or array.")
        if tuning_curves.shape[0] != len(rates):
            raise ValueError(
                "dimension of rates should match that of the tuning curves."
            )

    if "occupancy" not in tuning_curves.attrs:
        raise ValueError("No occupancy found in tuning curves.")
    occupancy = tuning_curves.attrs["occupancy"]
    occupancy = occupancy / np.nansum(occupancy)  # (D1, D2, ..., Dn)

    fx = tuning_curves.values  # (N, D1, D2, ...Dn)
    fr = tuning_curves.attrs.get("rates") if rates is None else rates  # (N,)

    axes = tuple(range(1, fx.ndim))

    if fr is None:
        warnings.warn(
            "Estimating mean firing rates from tuning curves, "
            "they were not in the tuning curves nor passed.",
            UserWarning,
            stacklevel=2,
        )
        fr = np.nansum(fx * occupancy, axis=axes)  # (N,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxfr = fx / np.expand_dims(fr, axis=axes)  # (N, D1, D2, ..., Dn)
        logfx = np.log2(fxfr)  # (N, D1, D2, ..., Dn)
    logfx[~np.isfinite(logfx)] = 0.0

    MI_bits_per_sec = np.nansum(occupancy * fx * logfx, axis=axes)  # (N,)
    with np.errstate(divide="ignore", invalid="ignore"):
        MI_bits_per_spike = MI_bits_per_sec / fr  # (N,)

    return pd.DataFrame(
        data=np.stack([MI_bits_per_sec, MI_bits_per_spike], axis=1),
        index=tuning_curves.coords["unit"],
        columns=["bits/sec", "bits/spike"],
    )


# =====================================================================================
# OLD FUNCTIONS, DEPRECATED
# =====================================================================================


def _validate_tuning_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        kwargs = sig.bind_partial(*args, **kwargs).arguments

        if "feature" in kwargs:
            if not isinstance(kwargs["feature"], (nap.Tsd, nap.TsdFrame)):
                raise TypeError(
                    "feature should be a Tsd (or TsdFrame with 1 column only)"
                )
            if (
                isinstance(kwargs["feature"], nap.TsdFrame)
                and not kwargs["feature"].shape[1] == 1
            ):
                raise ValueError(
                    "feature should be a Tsd (or TsdFrame with 1 column only)"
                )
        if "features" in kwargs:
            if not isinstance(kwargs["features"], nap.TsdFrame):
                raise TypeError("features should be a TsdFrame with 2 columns")
            if not kwargs["features"].shape[1] == 2:
                raise ValueError("features should have 2 columns only.")
        if "nb_bins" in kwargs:
            if not isinstance(kwargs["nb_bins"], (int, tuple)):
                raise TypeError(
                    "nb_bins should be of type int (or tuple with (int, int) for 2D tuning curves)."
                )
        if "group" in kwargs:
            if not isinstance(kwargs["group"], nap.TsGroup):
                raise TypeError("group should be a TsGroup.")
        if "ep" in kwargs:
            if not isinstance(kwargs["ep"], nap.IntervalSet):
                raise TypeError("ep should be an IntervalSet")
        if "minmax" in kwargs:
            if not isinstance(kwargs["minmax"], Iterable):
                raise TypeError("minmax should be a tuple/list of 2 numbers")
        if "dict_ep" in kwargs:
            if not isinstance(kwargs["dict_ep"], dict):
                raise TypeError("dict_ep should be a dictionary of IntervalSet")
            if not all(
                isinstance(v, nap.IntervalSet) for v in kwargs["dict_ep"].values()
            ):
                raise TypeError("dict_ep argument should contain only IntervalSet.")
        if "tc" in kwargs:
            if not isinstance(kwargs["tc"], (pd.DataFrame, np.ndarray)):
                raise TypeError(
                    "Argument tc should be of type pandas.DataFrame or numpy.ndarray"
                )
        if "dict_tc" in kwargs:
            if not isinstance(kwargs["dict_tc"], (dict, np.ndarray)):
                raise TypeError(
                    "Argument dict_tc should be a dictionary of numpy.ndarray or numpy.ndarray."
                )
        if "bitssec" in kwargs:
            if not isinstance(kwargs["bitssec"], bool):
                raise TypeError("Argument bitssec should be of type bool")
        if "tsdframe" in kwargs:
            if not isinstance(kwargs["tsdframe"], (nap.Tsd, nap.TsdFrame)):
                raise TypeError("Argument tsdframe should be of type Tsd or TsdFrame.")
        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


@_validate_tuning_inputs
def compute_1d_tuning_curves(group, feature, nb_bins, ep=None, minmax=None):
    """
    .. deprecated:: 0.9.2
          `compute_1d_tuning_curves` will be removed in Pynapple 1.0.0, it is replaced by
          `compute_tuning_curves` because the latter works for N dimensions.
    """
    warnings.warn(
        "compute_1d_tuning_curves is deprecated and will be removed in a future version;"
        "use compute_tuning_curves instead.",
        FutureWarning,
        stacklevel=2,
    )
    return (
        compute_tuning_curves(
            group,
            feature,
            nb_bins,
            range=None if minmax is None else [minmax],
            epochs=ep,
        )
        .to_pandas()
        .T
    )


@_validate_tuning_inputs
def compute_1d_tuning_curves_continuous(
    tsdframe, feature, nb_bins, ep=None, minmax=None
):
    """
    .. deprecated:: 0.9.2
          `compute_1d_tuning_curves` will be removed in Pynapple 1.0.0, it is replaced by
          `compute_tuning_curves` because the latter works for N dimensions and continuous data.
    """
    warnings.warn(
        "compute_1d_tuning_curves_continuous is deprecated and will be removed in a future version;"
        "use compute_tuning_curves instead.",
        FutureWarning,
        stacklevel=2,
    )
    return (
        compute_tuning_curves(
            tsdframe,
            feature,
            nb_bins,
            range=None if minmax is None else [minmax],
            epochs=ep,
        )
        .to_pandas()
        .T
    )


@_validate_tuning_inputs
def compute_2d_tuning_curves(group, features, nb_bins, ep=None, minmax=None):
    """
    .. deprecated:: 0.9.2
          `compute_1d_tuning_curves` will be removed in Pynapple 1.0.0, it is replaced by
          `compute_tuning_curves` because the latter works for N dimensions.
    """
    warnings.warn(
        "compute_2d_tuning_curves is deprecated and will be removed in a future version;"
        "use compute_tuning_curves instead.",
        FutureWarning,
        stacklevel=2,
    )
    xarray = compute_tuning_curves(
        group,
        features,
        nb_bins,
        range=(
            None if minmax is None else [[minmax[0], minmax[1]], [minmax[2], minmax[3]]]
        ),
        epochs=ep,
    )
    tcs = {c: xarray.sel(unit=c).values for c in xarray.coords["unit"].values}
    bins = [xarray.coords[dim].values for dim in xarray.coords if dim != "unit"]
    return tcs, bins


@_validate_tuning_inputs
def compute_2d_tuning_curves_continuous(
    tsdframe, features, nb_bins, ep=None, minmax=None
):
    """
    .. deprecated:: 0.9.2
          `compute_1d_tuning_curves` will be removed in Pynapple 1.0.0, it is replaced by
          `compute_tuning_curves` because the latter works for N dimensions and continuous data.
    """
    warnings.warn(
        "compute_2d_tuning_curves_continuous is deprecated and will be removed in a future version;"
        "use compute_tuning_curves instead.",
        FutureWarning,
        stacklevel=2,
    )
    xarray = compute_tuning_curves(
        tsdframe,
        features,
        nb_bins,
        range=(
            None if minmax is None else [[minmax[0], minmax[1]], [minmax[2], minmax[3]]]
        ),
        epochs=ep,
    )
    tcs = {c: xarray.sel(unit=c).values for c in xarray.coords["unit"].values}
    bins = [xarray.coords[dim].values for dim in xarray.coords if dim != "unit"]
    return tcs, bins


@_validate_tuning_inputs
def compute_discrete_tuning_curves(group, dict_ep):
    """
    .. deprecated:: 0.9.2
          `compute_discrete_tuning_curves` will be removed in Pynapple 1.0.0, it is replaced by
          `compute_response_per_epoch`.
    """
    warnings.warn(
        "compute_discrete_tuning_curves is deprecated and will be removed in a future version;"
        "use compute_response_per_epoch instead.",
        FutureWarning,
        stacklevel=2,
    )

    return compute_response_per_epoch(group, dict_ep, return_pandas=True)


@_validate_tuning_inputs
def compute_2d_mutual_info(dict_tc, features, ep=None, minmax=None, bitssec=False):
    """
    .. deprecated:: 0.9.2
          `compute_2d_mutual_info` will be removed in Pynapple 1.0.0, it is replaced by
          `compute_mutual_information` because the latter works for N dimensions.
    """
    import xarray as xr

    warnings.warn(
        "compute_2d_mutual_info is deprecated and will be removed in a future version;"
        "use compute_mutual_information instead.",
        FutureWarning,
        stacklevel=2,
    )
    if type(dict_tc) is dict:
        tcs = xr.DataArray(
            np.array([dict_tc[i] for i in dict_tc.keys()]),
            coords={"unit": list(dict_tc.keys())},
            dims=["unit", "0", "1"],
        )
    else:
        tcs = xr.DataArray(
            dict_tc,
            coords={"unit": np.arange(len(dict_tc))},
            dims=["unit", "0", "1"],
        )

    nb_bins = (tcs.shape[1] + 1, tcs.shape[2] + 1)
    bins = []
    for i in range(2):
        if minmax is None:
            bins.append(
                np.linspace(
                    np.nanmin(features[:, i]), np.nanmax(features[:, i]), nb_bins[i]
                )
            )
        else:
            bins.append(
                np.linspace(minmax[i + i % 2], minmax[i + 1 + i % 2], nb_bins[i])
            )

    if isinstance(ep, nap.IntervalSet):
        features = features.restrict(ep)

    occupancy, _, _ = np.histogram2d(
        features[:, 0].values.flatten(),
        features[:, 1].values.flatten(),
        [bins[0], bins[1]],
    )
    occupancy = occupancy / occupancy.sum()

    tcs.attrs["occupancy"] = occupancy
    MI = compute_mutual_information(tcs)

    column = "bits/sec" if bitssec else "bits/spike"
    return MI[[column]].rename({column: "SI"}, axis=1)


@_validate_tuning_inputs
def compute_1d_mutual_info(tc, feature, ep=None, minmax=None, bitssec=False):
    """
    .. deprecated:: 0.9.2
          `compute_1d_mutual_info` will be removed in Pynapple 1.0.0, it is replaced by
          `compute_mutual_information` because the latter works for N dimensions.
    """
    import xarray as xr

    warnings.warn(
        "compute_1d_mutual_info is deprecated and will be removed in a future version;"
        "use compute_mutual_information instead.",
        FutureWarning,
        stacklevel=2,
    )
    if isinstance(tc, pd.DataFrame):
        tcs = xr.DataArray(
            tc.values.T, coords={"unit": tc.columns.values, "0": tc.index}
        )
    else:
        tcs = xr.DataArray(
            tc.T, coords={"unit": np.arange(tc.shape[1])}, dims=["unit", "0"]
        )

    nb_bins = tc.shape[0] + 1
    if minmax is None:
        bins = np.linspace(np.nanmin(feature), np.nanmax(feature), nb_bins)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins)

    if isinstance(ep, nap.IntervalSet):
        occupancy, _ = np.histogram(feature.restrict(ep).values, bins)
    else:
        occupancy, _ = np.histogram(feature.values, bins)
    occupancy = occupancy / occupancy.sum()
    tcs.attrs["occupancy"] = occupancy
    MI = compute_mutual_information(tcs)

    column = "bits/sec" if bitssec else "bits/spike"
    return MI[[column]].rename({column: "SI"}, axis=1)
