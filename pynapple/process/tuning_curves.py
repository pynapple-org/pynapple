# -*- coding: utf-8 -*-
"""
"""
# @Author: gviejo
# @Date:   2022-01-02 23:33:42
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-01-29 11:10:07

import warnings

import numpy as np
import pandas as pd

from .. import core as nap


def compute_discrete_tuning_curves(group, dict_ep):
    """
    Compute discrete tuning curves of a TsGroup using a dictionnary of epochs.
    The function returns a pandas DataFrame with each row being a key of the dictionnary of epochs
    and each column being a neurons.

       This function can typically being used for a set of stimulus being presented for multiple epochs.
    An example of the dictionnary is :

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
    assert isinstance(group, nap.TsGroup), "group should be a TsGroup."
    assert isinstance(dict_ep, dict), "dict_ep should be a dictionnary of IntervalSet"
    idx = np.sort(list(dict_ep.keys()))
    for k in idx:
        assert isinstance(
            dict_ep[k], nap.IntervalSet
        ), "dict_ep argument should contain only IntervalSet. Key {} in dict_ep is not an IntervalSet".format(
            k
        )

    tuning_curves = pd.DataFrame(index=idx, columns=list(group.keys()), data=0.0)

    for k in dict_ep.keys():
        for n in group.keys():
            tuning_curves.loc[k, n] = float(len(group[n].restrict(dict_ep[k])))

        tuning_curves.loc[k] = tuning_curves.loc[k] / dict_ep[k].tot_length("s")

    return tuning_curves


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
    assert isinstance(group, nap.TsGroup), "group should be a TsGroup."
    assert isinstance(
        feature, (nap.Tsd, nap.TsdFrame)
    ), "feature should be a Tsd (or TsdFrame with 1 column only)"
    if isinstance(feature, nap.TsdFrame):
        assert (
            feature.shape[1] == 1
        ), "feature should be a Tsd (or TsdFrame with 1 column only)"
    assert isinstance(nb_bins, int)

    if ep is None:
        ep = feature.time_support
    else:
        assert isinstance(ep, nap.IntervalSet), "ep should be an IntervalSet"

    if minmax is None:
        bins = np.linspace(np.min(feature), np.max(feature), nb_bins + 1)
    else:
        assert isinstance(minmax, tuple), "minmax should be a tuple of boundaries"
        bins = np.linspace(minmax[0], minmax[1], nb_bins + 1)

    idx = bins[0:-1] + np.diff(bins) / 2

    tuning_curves = pd.DataFrame(index=idx, columns=list(group.keys()))

    if isinstance(ep, nap.IntervalSet):
        group_value = group.value_from(feature, ep)
        occupancy, _ = np.histogram(feature.restrict(ep).values, bins)
    else:
        group_value = group.value_from(feature)
        occupancy, _ = np.histogram(feature.values, bins)

    for k in group_value:
        count, _ = np.histogram(group_value[k].values, bins)
        count = count / occupancy
        count[np.isnan(count)] = 0.0
        tuning_curves[k] = count
        tuning_curves[k] = count * feature.rate

    return tuning_curves


def compute_2d_tuning_curves(group, features, nb_bins, ep=None, minmax=None):
    """
    Computes 2-dimensional tuning curves relative to a 2d features

    Parameters
    ----------
    group : TsGroup
        The group of Ts/Tsd for which the tuning curves will be computed
    features : TsdFrame
        The 2d features (i.e. 2 columns features).
    nb_bins : int
        Number of bins in the tuning curves
    ep : IntervalSet, optional
        The epoch on which tuning curves are computed.
        If None, the epoch is the time support of the feature.
    minmax : tuple or list, optional
        The min and max boundaries of the tuning curves given as:
        (minx, maxx, miny, maxy)
        If None, the boundaries are inferred from the target variable

    Returns
    -------
    tuple
        A tuple containing: \n
        tc (dict): Dictionnary of the tuning curves with dimensions (nb_bins, nb_bins).\n
        xy (list): List of bins center in the two dimensions

    Raises
    ------
    RuntimeError
        If group is not a TsGroup object or if features is not 2 columns only.

    """
    assert isinstance(group, nap.TsGroup), "group should be a TsGroup."
    assert isinstance(
        features, nap.TsdFrame
    ), "features should be a TsdFrame with 2 columns"
    if isinstance(features, nap.TsdFrame):
        assert features.shape[1] == 2, "features should have 2 columns only."
    assert isinstance(nb_bins, int)

    if ep is None:
        ep = features.time_support
    else:
        assert isinstance(ep, nap.IntervalSet), "ep should be an IntervalSet"
        features = features.restrict(ep)

    cols = list(features.columns)
    groups_value = {}
    binsxy = {}

    for i, c in enumerate(cols):
        groups_value[c] = group.value_from(features.loc[c], ep)
        if minmax is None:
            bins = np.linspace(
                np.min(features.loc[c]), np.max(features.loc[c]), nb_bins + 1
            )
        else:
            assert isinstance(minmax, tuple), "minmax should be a tuple of 4 elements"
            bins = np.linspace(minmax[i + i % 2], minmax[i + 1 + i % 2], nb_bins + 1)
        binsxy[c] = bins

    occupancy, _, _ = np.histogram2d(
        features.loc[cols[0]].values.flatten(),
        features.loc[cols[1]].values.flatten(),
        [binsxy[cols[0]], binsxy[cols[1]]],
    )

    tc = {}
    for n in group.keys():
        count, _, _ = np.histogram2d(
            groups_value[cols[0]][n].values.flatten(),
            groups_value[cols[1]][n].values.flatten(),
            [binsxy[cols[0]], binsxy[cols[1]]],
        )
        count = count / occupancy
        # count[np.isnan(count)] = 0.0
        tc[n] = count * features.rate

    xy = [binsxy[c][0:-1] + np.diff(binsxy[c]) / 2 for c in binsxy.keys()]

    return tc, xy


def compute_1d_mutual_info(tc, feature, ep=None, minmax=None, bitssec=False):
    """
    Mutual information as defined in

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
    elif isinstance(tc, np.ndarray):
        fx = np.atleast_2d(tc)
        columns = np.arange(tc.shape[1])

    assert isinstance(
        feature, (nap.Tsd, nap.TsdFrame)
    ), "feature should be a Tsd (or TsdFrame with 1 column only)"
    if isinstance(feature, nap.TsdFrame):
        assert (
            feature.shape[1] == 1
        ), "feature should be a Tsd (or TsdFrame with 1 column only)"

    nb_bins = tc.shape[0] + 1
    if minmax is None:
        bins = np.linspace(np.min(feature), np.max(feature), nb_bins)
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


def compute_2d_mutual_info(tc, features, ep=None, minmax=None, bitssec=False):
    """
    Mutual information as defined in

    Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
    An information-theoretic approach to deciphering the hippocampal code.
    In Advances in neural information processing systems (pp. 1030-1037).

    Parameters
    ----------
    tc : dict or numpy.ndarray
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
    if type(tc) is dict:
        fx = np.array([tc[i] for i in tc.keys()])
        idx = list(tc.keys())
    elif type(tc) is np.ndarray:
        fx = tc
        idx = np.arange(len(tc))

    assert isinstance(
        features, nap.TsdFrame
    ), "features should be a TsdFrame with 2 columns"
    if isinstance(features, nap.TsdFrame):
        assert features.shape[1] == 2, "features should have 2 columns only."

    nb_bins = (fx.shape[1] + 1, fx.shape[2] + 1)

    cols = features.columns

    bins = []
    for i, c in enumerate(cols):
        if minmax is None:
            bins.append(
                np.linspace(
                    np.min(features.loc[c]), np.max(features.loc[c]), nb_bins[i]
                )
            )
        else:
            bins.append(
                np.linspace(minmax[i + i % 2], minmax[i + 1 + i % 2], nb_bins[i])
            )

    if isinstance(ep, nap.IntervalSet):
        features = features.restrict(ep)

    occupancy, _, _ = np.histogram2d(
        features.loc[cols[0]].values.flatten(),
        features.loc[cols[1]].values.flatten(),
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


def compute_1d_tuning_curves_continuous(
    tsdframe, feature, nb_bins, ep=None, minmax=None
):
    """
    Computes 1-dimensional tuning curves relative to a feature with continous data.

    Parameters
    ----------
    tsdframe : Tsd or TsdFrame
        Input data (e.g. continus calcium data
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
    pandas.DataFrame
        DataFrame to hold the tuning curves

    Raises
    ------
    RuntimeError
        If tsdframe is not a Tsd or a TsdFrame object.

    """
    if not isinstance(tsdframe, (nap.Tsd, nap.TsdFrame)):
        raise RuntimeError("Unknown format for tsdframe.")

    assert isinstance(
        feature, (nap.Tsd, nap.TsdFrame)
    ), "feature should be a Tsd (or TsdFrame with 1 column only)"
    if isinstance(feature, nap.TsdFrame):
        assert (
            feature.shape[1] == 1
        ), "feature should be a Tsd (or TsdFrame with 1 column only)"
        feature = np.squeeze(feature)

    if isinstance(ep, nap.IntervalSet):
        feature = feature.restrict(ep)
        tsdframe = tsdframe.restrict(ep)
    else:
        tsdframe = tsdframe.restrict(feature.time_support)

    if minmax is None:
        bins = np.linspace(np.min(feature), np.max(feature), nb_bins + 1)
    else:
        bins = np.linspace(minmax[0], minmax[1], nb_bins + 1)

    align_times = tsdframe.value_from(feature)
    idx = np.digitize(align_times.values, bins) - 1
    tmp = tsdframe.as_dataframe().groupby(idx).mean()
    tmp = tmp.reindex(np.arange(0, len(bins) - 1))
    tmp.index = pd.Index(bins[0:-1] + np.diff(bins) / 2)

    tmp = tmp.fillna(0)

    return pd.DataFrame(tmp)


def compute_2d_tuning_curves_continuous(
    tsdframe, features, nb_bins, ep=None, minmax=None
):
    """
    Computes 2-dimensional tuning curves relative to a 2d feature with continous data.

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
        tc (dict): Dictionnary of the tuning curves with dimensions (nb_bins, nb_bins).\n
        xy (list): List of bins center in the two dimensions

    Raises
    ------
    RuntimeError
        If tsdframe is not a Tsd/TsdFrame or if features is not 2 columns

    """
    if not isinstance(tsdframe, (nap.Tsd, nap.TsdFrame)):
        raise RuntimeError("Unknown format for tsdframe.")

    assert isinstance(
        features, nap.TsdFrame
    ), "features should be a TsdFrame with 2 columns"
    if isinstance(features, nap.TsdFrame):
        assert features.shape[1] == 2, "features should have 2 columns only."

    if isinstance(ep, nap.IntervalSet):
        features = features.restrict(ep)
        tsdframe = tsdframe.restrict(ep)
    else:
        tsdframe = tsdframe.restrict(features.time_support)

    if isinstance(nb_bins, int):
        nb_bins = (nb_bins, nb_bins)
    elif len(nb_bins) != 2:
        raise RuntimeError("nb_bins should be int or tuple of 2 ints")

    cols = list(features.columns)

    binsxy = {}
    idxs = {}

    for i, c in enumerate(cols):
        if minmax is None:
            bins = np.linspace(
                np.min(features.loc[c]), np.max(features.loc[c]), nb_bins[i] + 1
            )
        else:
            bins = np.linspace(minmax[i + i % 2], minmax[i + 1 + i % 2], nb_bins[i] + 1)

        align_times = tsdframe.value_from(features.loc[c], ep)
        idxs[c] = np.digitize(align_times.values.flatten(), bins) - 1
        binsxy[c] = bins

    idxs = pd.DataFrame(idxs)

    tc_np = np.zeros((tsdframe.shape[1], nb_bins[0], nb_bins[1])) * np.nan

    for k, tmp in idxs.groupby(cols):
        if (0 <= k[0] < nb_bins[0]) and (0 <= k[1] < nb_bins[1]):
            tc_np[:, k[0], k[1]] = np.mean(tsdframe[tmp.index].values, 0)

    tc_np[np.isnan(tc_np)] = 0.0

    xy = [binsxy[c][0:-1] + np.diff(binsxy[c]) / 2 for c in binsxy.keys()]

    tc = {c: tc_np[i] for i, c in enumerate(tsdframe.columns)}

    return tc, xy


# def compute_1d_poisson_glm(
#     group, feature, binsize, windowsize, ep, time_units="s", niter=100, tolerance=1e-5
# ):
#     """
#     Poisson GLM

#     Warning : this function is still experimental!

#     Parameters
#     ----------
#     group : TsGroup
#         Spike trains
#     feature : Tsd
#         The regressors
#     binsize : float
#         Bin size
#     windowsize : Float
#         The window for offsetting the regressors
#     ep : IntervalSet, optional
#         On which epoch to perfom the GLM
#     time_units : str, optional
#         Time units of binsize and windowsize
#     niter : int, optional
#         Number of iteration for fitting the GLM
#     tolerance : float, optional
#         Tolerance for stopping the IRLS

#     Returns
#     -------
#     tuple
#         regressors : TsdFrame\n
#         offset : pandas.Series\n
#         prediction : TsdFrame\n

#     Raises
#     ------
#     RuntimeError
#         if group is not a TsGroup

#     """
#     if type(group) is nap.TsGroup:
#         newgroup = group.restrict(ep)
#     else:
#         raise RuntimeError("Unknown format for group")

#     binsize = nap.TsIndex.format_timestamps(binsize, time_units)[0]
#     windowsize = nap.TsIndex.format_timestamps(windowsize, time_units)[0]

#     # Bin the spike train
#     count = newgroup.count(binsize)

#     # Downsample the feature to binsize
#     tidx = []
#     dfeat = []
#     for i in ep.index:
#         bins = np.arange(ep.start[i], ep.end[i] + binsize, binsize)
#         idx = np.digitize(feature.index.values, bins) - 1
#         tmp = feature.groupby(idx).mean()
#         tidx.append(bins[0:-1] + np.diff(bins) / 2)
#         dfeat.append(tmp)
#     dfeat = nap.Tsd(t=np.hstack(tidx), d=np.hstack(dfeat), time_support=ep)

#     # Build the Hankel matrix
#     nt = np.abs(windowsize // binsize).astype("int") + 1
#     X = hankel(
#         np.hstack((np.zeros(nt - 1), dfeat.values))[: -nt + 1], dfeat.values[-nt:]
#     )
#     X = np.hstack((np.ones((len(dfeat), 1)), X))

#     # Fitting GLM for each neuron
#     regressors = []
#     for i, n in enumerate(group.keys()):
#         print("Fitting Poisson GLM for unit %i" % n)
#         b = nap.jitted_functions.jit_poisson_IRLS(
#             X, count[n].values, niter=niter, tolerance=tolerance
#         )
#         regressors.append(b)

#     regressors = np.array(regressors).T
#     offset = regressors[0]
#     regressors = regressors[1:]
#     regressors = nap.TsdFrame(
#         t=np.arange(-nt + 1, 1) * binsize, d=regressors, columns=list(group.keys())
#     )
#     offset = pd.Series(index=group.keys(), data=offset)

#     prediction = nap.TsdFrame(
#         t=dfeat.index.values,
#         d=np.exp(np.dot(X[:, 1:], regressors.values) + offset.values) * binsize,
#     )

#     return (regressors, offset, prediction)
