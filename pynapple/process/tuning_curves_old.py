"""
Old functions to compute 1- and 2-dimensional tuning curves.
"""

import inspect
import warnings
from collections.abc import Iterable
from functools import wraps

import numpy as np
import pandas as pd
import xarray as xr

from .. import core as nap
from .tuning_curves import compute_mutual_information, compute_tuning_curves


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
def compute_discrete_tuning_curves(group, dict_ep):
    """
    Compute discrete tuning curves of a TsGroup using a dictionary of epochs.
    The function returns a pandas DataFrame with each row being a key of the dictionary of epochs
    and each column being a neurons.

    This function can typically being used for a set of stimulus being presented for multiple epochs.
    An example of the dictionary is:

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
    Deprecated, use `compute_tuning_curves` instead.
    """
    warnings.warn(
        "compute_1d_tuning_curves is deprecated and will be removed in a future version;"
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
    """
    Deprecated, use `compute_tuning_curves` instead.
    """
    warnings.warn(
        "compute_1d_tuning_curves_continuous is deprecated and will be removed in a future version;"
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
    """
    Deprecated, use `compute_tuning_curves` instead.
    """
    warnings.warn(
        "compute_2d_tuning_curves is deprecated and will be removed in a future version;"
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
    """
    Deprecated, use `compute_tuning_curves` instead.
    """
    warnings.warn(
        "compute_2d_tuning_curves_continuous is deprecated and will be removed in a future version;"
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
def compute_2d_mutual_info(dict_tc, features, ep=None, minmax=None, bitssec=False):
    warnings.warn(
        "compute_2d_mutual_info is deprecated and will be removed in a future version;"
        "use compute_mutual_information instead.",
        DeprecationWarning,
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
    warnings.warn(
        "compute_1d_mutual_info is deprecated and will be removed in a future version;"
        "use compute_mutual_information instead.",
        DeprecationWarning,
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
