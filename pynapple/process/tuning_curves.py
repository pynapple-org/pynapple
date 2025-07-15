"""
Functions to compute tuning curves for features in 1 dimension or 2 dimension.

"""

import inspect
import warnings
from collections.abc import Iterable
from functools import wraps

import numpy as np
import pandas as pd
import xarray as xr

from .. import core as nap


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
                [isinstance(v, nap.IntervalSet) for v in kwargs["dict_ep"].values()]
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


def compute_tuning_curves(
    group,
    features,
    bins=10,
    range=None,
    epochs=None,
    fs=None,
    feature_names=None,
    return_pandas=False,
):
    """
    Computes n-dimensional tuning curves relative to n features.

    Parameters
    ----------
    group : TsGroup, TsdFrame or dict of Ts, Tsd objects.
        The group of Ts or Tsd for which the tuning curves will be computed
    features : Tsd, TsdFrame
        The features (i.e. one column per feature).
    bins : sequence or int
        The bin specification:
        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
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

    Returns
    -------
    xarray.DataArray
        An xarray.DataArray containing the tuning curves with bin centres as coordinates.
        The bin edges and occupancy are stored as attributes.

    Examples
    --------
    """

    # check group
    if isinstance(group, dict):
        group = nap.TsGroup(group)
    if isinstance(group, nap.Tsd):
        group = nap.TsdFrame(
            d=group.values,
            t=group.times(),
            time_support=group.time_support,
        )
    elif not isinstance(group, (nap.TsGroup, nap.TsdFrame)):
        raise TypeError("group should be a Tsd, TsdFrame, TsGroup, or dict.")

    # check features
    if isinstance(features, nap.Tsd):
        features = nap.TsdFrame(
            d=features.values,
            t=features.times(),
            time_support=features.time_support,
        )
    elif not isinstance(features, nap.TsdFrame):
        raise TypeError("features should be a Tsd or TsdFrame.")

    # check feature names
    if feature_names is None:
        feature_names = features.columns
    else:
        if not isinstance(feature_names, list) or not all(
            isinstance(n, str) for n in feature_names
        ):
            raise TypeError("feature_names should be a list of strings.")
        if len(feature_names) != features.shape[1]:
            raise ValueError("feature_names should match the number of features.")

    # check epochs
    if epochs is None:
        epochs = features.time_support
    elif isinstance(epochs, nap.IntervalSet):
        features = features.restrict(epochs)
    else:
        raise TypeError("epochs should be an IntervalSet.")
    group = group.restrict(epochs)

    # check fs
    if fs is None:
        fs = 1 / np.mean(features.time_diff(epochs=epochs).values)
    if not isinstance(fs, (int, float)):
        raise TypeError("fs should be a number (int or float)")

    # check range
    if range is not None and isinstance(range, tuple):
        if features.shape[1] == 1:
            range = [range]
        else:
            raise ValueError(
                "range should be a sequence of tuples, one for each feature."
            )

    # occupancy
    occupancy, bin_edges = np.histogramdd(features, bins=bins, range=range)

    # tunning curves
    keys = group.keys() if isinstance(group, nap.TsGroup) else group.columns
    tcs = np.zeros([len(keys), *occupancy.shape])
    if isinstance(group, nap.TsGroup):
        # SPIKES
        for i, n in enumerate(keys):
            tcs[i] = np.histogramdd(
                group[n].value_from(features, epochs),
                bins=bin_edges,
            )[0]
        occupancy[occupancy == 0.0] = np.nan
        tcs = (tcs / occupancy) * fs
    else:
        # RATES
        values = group.value_from(features, epochs)
        counts = np.histogramdd(values, bins=bin_edges)[0]
        counts[counts == 0] = np.nan
        for i, n in enumerate(keys):
            tcs[i] = np.histogramdd(
                values,
                weights=group.values[:, i],
                bins=bin_edges,
            )[0]
        tcs /= counts
        tcs[np.isnan(tcs)] = 0.0
        tcs[:, occupancy == 0.0] = np.nan

    if return_pandas and features.shape[1] == 1:
        return pd.DataFrame(
            tcs.T,
            index=bin_edges[0][:-1] + np.diff(bin_edges[0]) / 2,
            columns=keys,
        )
    else:
        return xr.DataArray(
            tcs,
            coords={
                "unit": keys,
                **{
                    str(feature_name): e[:-1] + np.diff(e) / 2
                    for feature_name, e in zip(feature_names, bin_edges)
                },
            },
            attrs={"occupancy": occupancy, "bin_edges": bin_edges},
        )


@_validate_tuning_inputs
def compute_1d_tuning_curves(group, feature, nb_bins, ep=None, minmax=None):
    warnings.warn(
        "compute_1d_tuning_curves is deprecated and will be removed in v1.0; "
        "use compute_tuning_curves instead.",
        DeprecationWarning,
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
    warnings.warn(
        "compute_1d_tuning_curves_continuous is deprecated and will be removed in v1.0; "
        "use compute_tuning_curves instead.",
        DeprecationWarning,
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
    warnings.warn(
        "compute_2d_tuning_curves is deprecated and will be removed in v1.0; "
        "use compute_tuning_curves instead.",
        DeprecationWarning,
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
    warnings.warn(
        "compute_2d_tuning_curves_continuous is deprecated and will be removed in v1.0; "
        "use compute_tuning_curves instead.",
        DeprecationWarning,
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
    Compute discrete tuning curves of a TsGroup using a dictionary of epochs.
    The function returns a pandas DataFrame with each row being a key of the dictionary of epochs
    and each column being a neurons.

       This function can typically being used for a set of stimulus being presented for multiple epochs.
    An example of the dictionary is :

        >>> dict_ep =  {
                "stim0": nap.IntervalSet(start=0, end=1),
                "stim1":nap.IntervalSet(start=2, end=3)
            }
    In this case, the function will return a pandas DataFrame :

        >>> tc
                   neuron0    neuron1    neuron2
        stim0        0 Hz       1 Hz       2 Hz
        stim1        3 Hz       4 Hz       5 Hz


    Parameters
    ----------
    group : nap.TsGroup
        The group of Ts/Tsd for which the tuning curves will be computed
    dict_ep : dict
        Dictionary of IntervalSets

    Returns
    -------
    pandas.DataFrame
        Table of firing rate for each neuron and each IntervalSet

    Raises
    ------
    RuntimeError
        If group is not a TsGroup object.
    """
    idx = np.sort(list(dict_ep.keys()))
    tuning_curves = pd.DataFrame(index=idx, columns=list(group.keys()), data=0.0)

    for k in dict_ep.keys():
        for n in group.keys():
            tuning_curves.loc[k, n] = float(len(group[n].restrict(dict_ep[k])))

        tuning_curves.loc[k] = tuning_curves.loc[k] / dict_ep[k].tot_length("s")

    return tuning_curves


@_validate_tuning_inputs
def compute_1d_mutual_info(tc, feature, ep=None, minmax=None, bitssec=False):
    """
    Mutual information of a tuning curve computed from a 1-d feature.

    See:

    Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
    An information-theoretic approach to deciphering the hippocampal code.
    In Advances in neural information processing systems (pp. 1030-1037).

    Parameters
    ----------
    tc : pandas.DataFrame or numpy.ndarray
        Tuning curves in columns
    feature : Tsd (or TsdFrame with 1 column only)
        The 1-dimensional target feature (e.g. head-direction)
    ep : IntervalSet, optional
        The epoch over which the tuning curves were computed
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target feature
    bitssec : bool, optional
        By default, the function return bits per spikes.
        Set to true for bits per seconds

    Returns
    -------
    pandas.DataFrame
        Spatial Information (default is bits/spikes)
    """
    if isinstance(tc, pd.DataFrame):
        columns = tc.columns.values
        fx = np.atleast_2d(tc.values)
    else:
        fx = np.atleast_2d(tc)
        columns = np.arange(tc.shape[1])

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
    occupancy = occupancy[:, np.newaxis]

    fr = np.sum(fx * occupancy, 0)
    fxfr = fx / fr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logfx = np.log2(fxfr)
    logfx[np.isinf(logfx)] = 0.0
    SI = np.sum(occupancy * fx * logfx, 0)

    if bitssec:
        SI = pd.DataFrame(index=columns, columns=["SI"], data=SI)
        return SI
    else:
        SI = SI / fr
        SI = pd.DataFrame(index=columns, columns=["SI"], data=SI)
        return SI


@_validate_tuning_inputs
def compute_2d_mutual_info(dict_tc, features, ep=None, minmax=None, bitssec=False):
    """
    Mutual information of a tuning curve computed from 2-d features.

    See:

    Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
    An information-theoretic approach to deciphering the hippocampal code.
    In Advances in neural information processing systems (pp. 1030-1037).

    Parameters
    ----------
    dict_tc : dict of numpy.ndarray or numpy.ndarray
        If array, first dimension should be the neuron
    features : TsdFrame
        The 2 columns features that were used to compute the tuning curves
    ep : IntervalSet, optional
        The epoch over which the tuning curves were computed
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target features
    bitssec : bool, optional
        By default, the function return bits per spikes.
        Set to true for bits per seconds

    Returns
    -------
    pandas.DataFrame
        Spatial Information (default is bits/spikes)
    """
    # A bit tedious here
    if type(dict_tc) is dict:
        fx = np.array([dict_tc[i] for i in dict_tc.keys()])
        idx = list(dict_tc.keys())
    else:
        fx = dict_tc
        idx = np.arange(len(dict_tc))

    nb_bins = (fx.shape[1] + 1, fx.shape[2] + 1)

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

    fr = np.nansum(fx * occupancy, (1, 2))
    fr = fr[:, np.newaxis, np.newaxis]
    fxfr = fx / fr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logfx = np.log2(fxfr)
    logfx[np.isinf(logfx)] = 0.0
    SI = np.nansum(occupancy * fx * logfx, (1, 2))

    if bitssec:
        SI = pd.DataFrame(index=idx, columns=["SI"], data=SI)
        return SI
    else:
        SI = SI / fr[:, 0, 0]
        SI = pd.DataFrame(index=idx, columns=["SI"], data=SI)
        return SI
