"""
Functions to compute tuning curves for features in 1 dimension or 2 dimension.

"""

import inspect
import warnings
from collections.abc import Iterable
from functools import wraps

import numpy as np
import pandas as pd

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
def compute_1d_tuning_curves(group, feature, nb_bins, ep=None, minmax=None):
    """
    Computes 1-dimensional tuning curves relative to a 1d feature.

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd for which the tuning curves will be computed
    feature : Tsd (or TsdFrame with 1 column only)
        The 1-dimensional target feature (e.g. head-direction)
    nb_bins : int
        Number of bins in the tuning curve
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target feature

    Returns
    -------
    pandas.DataFrame
        DataFrame to hold the tuning curves

    Raises
    ------
    RuntimeError
        If group is not a TsGroup object.

    """
    if minmax is not None and len(minmax) != 2:
        raise ValueError("minmax should be of length 2.")
    if ep is None:
        ep = feature.time_support

    if minmax is None:
        bins = np.linspace(np.nanmin(feature), np.nanmax(feature), nb_bins + 1)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins + 1)

    idx = bins[0:-1] + np.diff(bins) / 2

    tuning_curves = pd.DataFrame(index=idx, columns=list(group.keys()))

    group_value = group.value_from(feature, ep)

    occupancy, _ = np.histogram(feature.restrict(ep).values, bins)

    for k in group_value:
        count, _ = np.histogram(group_value[k].values, bins)
        count = count / occupancy
        tuning_curves[k] = count
        tuning_curves[k] = count * feature.rate

    return tuning_curves


@_validate_tuning_inputs
def compute_2d_tuning_curves(group, features, nb_bins, ep=None, minmax=None):
    """
    Computes 2-dimensional tuning curves relative to a 2d features

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd for which the tuning curves will be computed
    features : TsdFrame
        The 2d features (i.e. 2 columns features).
    nb_bins : int or tuple
        Number of bins in the tuning curves (separate for 2 feature dimensions if tuple provided)
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves given as:
        (minx, maxx, miny, maxy)
        If None, the boundaries are inferred from the target features

    Returns
    -------
    tuple
        A tuple containing: \n
        tc (dict): Dictionary of the tuning curves with dimensions (nb_bins, nb_bins).\n
        xy (list): List of bins center in the two dimensions

    Raises
    ------
    RuntimeError
        If group is not a TsGroup object or if features is not 2 columns only.

    """
    if minmax is not None and len(minmax) != 4:
        raise ValueError("minmax should be of length 4.")

    if isinstance(nb_bins, tuple) and len(nb_bins) != 2:
        raise ValueError(
            "nb_bins should be of type int (or tuple with (int, int) for 2D tuning curves)."
        )

    if isinstance(nb_bins, int):
        nb_bins = (nb_bins, nb_bins)

    if ep is None:
        ep = features.time_support
    else:
        features = features.restrict(ep)

    groups_value = {}
    binsxy = {}

    for i in range(2):
        groups_value[i] = group.value_from(features[:, i], ep)
        if minmax is None:
            bins = np.linspace(
                np.nanmin(features[:, i]), np.nanmax(features[:, i]), nb_bins[i] + 1
            )
        else:
            bins = np.linspace(minmax[i + i % 2], minmax[i + 1 + i % 2], nb_bins[i] + 1)
        binsxy[i] = bins

    occupancy, _, _ = np.histogram2d(
        features[:, 0].values.flatten(),
        features[:, 1].values.flatten(),
        [binsxy[0], binsxy[1]],
    )

    tc = {}
    for n in group.keys():
        count, _, _ = np.histogram2d(
            groups_value[0][n].values.flatten(),
            groups_value[1][n].values.flatten(),
            [binsxy[0], binsxy[1]],
        )
        count = count / occupancy
        tc[n] = count * features.rate

    xy = [binsxy[i][0:-1] + np.diff(binsxy[i]) / 2 for i in range(2)]

    return tc, xy


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


@_validate_tuning_inputs
def compute_1d_tuning_curves_continuous(
    tsdframe, feature, nb_bins, ep=None, minmax=None
):
    """
    Computes 1-dimensional tuning curves relative to a feature with continuous data.

    Parameters
    ----------
    tsdframe : Tsd or TsdFrame
        Input data (e.g. continuous calcium data
        where each column is the calcium activity of one neuron)
    feature : Tsd (or TsdFrame with 1 column only)
        The 1-dimensional target feature (e.g. head-direction)
    nb_bins : int
        Number of bins in the tuning curves
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        If None, the boundaries are inferred from the target feature

    Returns
    -------
    pandas.DataFrame to hold the tuning curves

    Raises
    ------
    RuntimeError
        If tsdframe is not a Tsd or a TsdFrame object.

    """
    if minmax is not None and len(minmax) != 2:
        raise ValueError("minmax should be of length 2.")

    feature = np.squeeze(feature)

    if isinstance(ep, nap.IntervalSet):
        feature = feature.restrict(ep)
        tsdframe = tsdframe.restrict(ep)
    else:
        tsdframe = tsdframe.restrict(feature.time_support)

    if isinstance(tsdframe, nap.Tsd):
        tsdframe = tsdframe[:, np.newaxis]

    if minmax is None:
        bins = np.linspace(np.nanmin(feature), np.nanmax(feature), nb_bins + 1)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins + 1)

    align_times = tsdframe.value_from(feature)
    idx = np.digitize(align_times.values, bins) - 1

    tc = np.zeros((len(bins) - 1, tsdframe.shape[1]))
    for i in range(0, nb_bins):
        tc[i] = np.mean(tsdframe.values[idx == i], axis=0)
    tc[np.isnan(tc)] = 0.0

    # Assigning nans if bin is not visited.
    occupancy, _ = np.histogram(feature, bins)
    tc[occupancy == 0.0] = np.nan

    tc = pd.DataFrame(
        index=bins[0:-1] + np.diff(bins) / 2, data=tc, columns=tsdframe.columns
    )
    return tc


@_validate_tuning_inputs
def compute_2d_tuning_curves_continuous(
    tsdframe, features, nb_bins, ep=None, minmax=None
):
    """
    Computes 2-dimensional tuning curves relative to a 2d feature with continuous data.

    Parameters
    ----------
    tsdframe : Tsd or TsdFrame
        Input data (e.g. continuous calcium data
        where each column is the calcium activity of one neuron)
    features : TsdFrame
        The 2d feature (two columns)
    nb_bins : int or tuple
        Number of bins in the tuning curves (separate for 2 feature dimensions if tuple provided)
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves.
        Should be a tuple of minx, maxx, miny, maxy
        If None, the boundaries are inferred from the target feature

    Returns
    -------
    tuple
        A tuple containing: \n
        tc (dict): Dictionary of the tuning curves with dimensions (nb_bins, nb_bins).\n
        xy (list): List of bins center in the two dimensions

    Raises
    ------
    RuntimeError
        If tsdframe is not a Tsd/TsdFrame or if features is not 2 columns

    """
    if minmax is not None and len(minmax) != 4:
        raise ValueError("minmax should be of length 4.")

    if isinstance(nb_bins, tuple) and len(nb_bins) != 2:
        raise ValueError(
            "nb_bins should be of type int (or tuple with (int, int) for 2D tuning curves)."
        )

    if isinstance(ep, nap.IntervalSet):
        features = features.restrict(ep)
        tsdframe = tsdframe.restrict(ep)
    else:
        tsdframe = tsdframe.restrict(features.time_support)

    if isinstance(tsdframe, nap.Tsd):
        tsdframe = tsdframe[:, np.newaxis]

    if isinstance(nb_bins, int):
        nb_bins = (nb_bins, nb_bins)

    binsxy = []
    idxs = []

    for i in range(2):
        if minmax is None:
            bins = np.linspace(
                np.nanmin(features[:, i]), np.nanmax(features[:, i]), nb_bins[i] + 1
            )
        else:
            bins = np.linspace(minmax[i + i % 2], minmax[i + 1 + i % 2], nb_bins[i] + 1)

        align_times = tsdframe.value_from(features[:, i], ep)
        idxs.append(np.digitize(align_times.values.flatten(), bins) - 1)
        binsxy.append(bins)

    idxs = np.transpose(np.array(idxs))

    tc = np.zeros((tsdframe.shape[1], nb_bins[0], nb_bins[1]))

    for i in range(nb_bins[0]):
        for j in range(nb_bins[1]):
            tc[:, i, j] = np.mean(
                tsdframe.values[np.logical_and(idxs[:, 0] == i, idxs[:, 1] == j)], 0
            )

    tc[np.isnan(tc)] = 0.0

    # Assigning nans if bin is not visited.
    occupancy, _, _ = np.histogram2d(
        features[:, 0].values.flatten(),
        features[:, 1].values.flatten(),
        [binsxy[0], binsxy[1]],
    )
    occupancy = occupancy[np.newaxis, :, :]
    occupancy = np.repeat(occupancy, len(tc), axis=0)
    tc[occupancy == 0.0] = np.nan

    xy = [binsxy[i][0:-1] + np.diff(binsxy[i]) / 2 for i in range(2)]

    tc = {c: tc[i] for i, c in enumerate(tsdframe.columns)}

    return tc, xy
