"""
Functions to compute n-dimensional tuning curves.
"""

import inspect
import warnings
from collections.abc import Iterable
from functools import wraps

import numpy as np
import pandas as pd

from .. import core as nap


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
    data,
    features,
    bins=10,
    range=None,
    epochs=None,
    fs=None,
    feature_names=None,
    return_pandas=False,
    return_counts=False,
):
    """
    Compute n-dimensional tuning curves describing how `data` varies as a
    function of one or more `features`.

    A tuning curve quantifies the relationship between a signal (e.g. neuronal
    spike rate or a continuous measurement) and one or more explanatory
    variables such as position, head direction, or speed. The feature space is
    discretised into bins, the data are accumulated within each bin, and the
    result is normalised by the occupancy of each bin (unless
    `return_counts=True`).

    If `epochs` is not provided, the tuning curves are computed over the time
    support of `features`.

    Parameters
    ----------
    data : TsGroup, TsdFrame, Ts, Tsd
        The data for which the tuning curves will be computed. This usually corresponds to the activity of the
        neurons, either as spike times (TsGroup or Ts) or continuous values (TsdFrame or Tsd).
    features : Tsd, TsdFrame
        The features (i.e. one column per feature). This usually corresponds to behavioral variables such as
        position, head direction, speed, etc.
    bins : sequence or int
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension. For a single feature, the edge sequence
          can be passed directly without wrapping it in another sequence.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).
    range : sequence, optional
        A sequence of entries per feature, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of D None values.
    epochs : IntervalSet, optional
        The epochs on which tuning curves are computed.
        If None, the epochs are the time support of the features.
    fs : float, optional
        The exact sampling frequency of the features used to normalise the tuning curves.
        Unit should match that of the features. If not passed, it is estimated.
    feature_names : list, optional
        A list of feature names. If not passed, the column names in `features` are used.
    return_pandas : bool, optional
        If True, the function returns a pandas.DataFrame instead of an xarray.DataArray.
        Note that this will not work if the features are not 1D and that occupancy and bin edges
        will not be stored as attributes.
    return_counts : bool, optional
        If True, does not divide the spike counts by occupancy, but returns the counts directly.
        The occupancy is stored in the xarray attributes, so the division can be performed after any
        particular processing steps.
        If the input is a TsdFrame, this does not do anything.

    Returns
    -------
    xarray.DataArray
        A tensor containing the tuning curves with labeled bin centres.
        The bin edges and occupancy are stored as attributes.

    Examples
    --------
    In the simplest case, we can pass a group of spikes per neuron and a single feature:

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

    The function can also take multiple features, in which case it computes n-dimensional tuning curves.
    We can specify the number of bins for each feature:

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

    Or even specify the bin edges directly:

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

    In all of these cases, it is also possible to pass continuous values instead of spikes (e.g. calcium imaging data), in that case the mean response is computed:

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
    occupancy, bin_edges = np.histogramdd(features, bins=bins, range=range)

    # tuning curves
    # np.asarray drops any name carried by a pandas Index (TsdFrame.columns
    # loaded from NWB is named "id"), which xarray would otherwise use as the
    # coordinate dimension instead of "unit".
    keys = np.asarray(
        data.keys()
        if isinstance(data, nap.TsGroup)
        else data.columns if isinstance(data, nap.TsdFrame) else [0]
    )
    tcs = np.zeros([len(keys), *occupancy.shape])
    if isinstance(data, (nap.TsGroup, nap.Ts)):
        # SPIKES
        if isinstance(data, nap.Ts):
            data = nap.TsGroup({0: data})

        # Each unit is matched against the same feature, so the feature's bin
        # assignment can be computed once and looked up per unit instead of being
        # re-derived from the matched values. `value_from` does the lookup: run
        # against a series carrying the bin indices on the feature's own
        # timestamps, it returns the bin of each spike directly, with exactly the
        # matching and epochs it would have used on the feature itself.
        #
        # Worth it only when more values are matched in total than the feature has
        # samples; below that, binning the whole feature costs more than binning
        # the matches. Measured to hold at both ends: 33.7 -> 23.2 ms at 400k
        # matches over 100k samples, but 44.7 -> 54.7 ms at 400k over 1M.
        prebin = sum(len(data[n]) for n in keys) > len(features)
        if prebin:
            feature_bins, n_flat = _flat_bin_index(features, bin_edges)
            # float64, not the integer index: an integer-valued target sends
            # `value_from` down a slower path
            feature_bins = nap.Tsd(
                t=features.index,
                d=feature_bins.astype(np.float64),
                time_support=features.time_support,
            )

        for i, n in enumerate(keys):
            if not isinstance(data[n], nap.Ts):
                warnings.warn(
                    f"TsGroup entry {n} was not a Ts, but treating it as one!"
                )
            if prebin:
                spike_bins = data[n].value_from(feature_bins).values
                # a spike with no target in its epoch comes back NaN; it has no bin
                # and is dropped, as an unmatched value would be by histogramdd
                if np.isnan(spike_bins).any():
                    spike_bins = spike_bins[~np.isnan(spike_bins)]
                tcs[i] = _histogram_from_bin_index(
                    spike_bins.astype(np.intp), n_flat, bin_edges
                )
            else:
                tcs[i] = np.histogramdd(
                    data[n].value_from(features),
                    bins=bin_edges,
                )[0]
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
        for i, n in enumerate(keys):
            tcs[i] = _histogram_from_bin_index(
                flat_index, n_flat, bin_edges, weights=data[:, i]
            )
        tcs /= counts
        tcs[np.isnan(tcs)] = 0.0
        tcs[:, occupancy == 0.0] = np.nan

    attrs = {"occupancy": occupancy, "bin_edges": bin_edges, "fs": fs}
    if isinstance(data, nap.TsGroup):
        attrs["rates"] = data.rates
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


def compute_response_per_epoch(data, epochs_dict, return_pandas=False):
    """
    Compute mean response per epoch, given a dictionary of epochs.

    Parameters
    ----------
    data : TsGroup, TsdFrame, Ts, Tsd
        The data for which the tuning curves will be computed.
    epochs_dict : dict
        Dictionary of IntervalSets.
    return_pandas : bool, optional
        If True, the function returns a pandas.DataFrame instead of an xarray.DataArray.

    Returns
    -------
    xarray.DataArray
        A tensor containing the tuning curves with labeled epochs.

    Examples
    --------
    This function is typically used for a set of discrete stimuli being presented for multiple epochs.
    The stimulus epochs can overlap, though note that epochs within an IntervalSet can not overlap.

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

    You can also pass a TsdFrame (e.g. calcium imaging data), in that case the response is computed:

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


def compute_mutual_information(tuning_curves, rates=None):
    """
    Computes mutual information from n-dimensional tuning curves.

    This function implements Skaggs et al.'s [1] metric to quantify
    the information content of a neuron's firing with respect to a variable
    (e.g., position), based on its tuning curve.

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

    References
    ----------
    .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
           An information-theoretic approach to deciphering the hippocampal code.
           In Advances in neural information processing systems (pp. 1030-1037).

    Parameters
    ----------
    tuning_curves : xarray.DataArray
        Tuning curves as computed by :func:`~pynapple.process.tuning_curves.compute_tuning_curves`.
    rates : list or numpy.ndarray, optional
        Mean firing rates of the units. By default :func:`~pynapple.process.tuning_curves.compute_tuning_curves` saves
        the mean firing rates over the epochs of the tuning curves in the tuning curve objects.
        This argument can be used to pass your own.

    Returns
    -------
    pandas.DataFrame
        A table containing the spatial information per unit, in both bits/sec and bits/spike.

    Examples
    --------
    We can compute the mutual information between a variable and a set of neurons' firing from the tuning curves:

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
