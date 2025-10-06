"""
Functions to decode 1- and 2-dimensional features.
"""

import warnings

import numpy as np
import xarray as xr

from .. import core as nap
from .decoding import decode_bayes

# Deprecated functions for backward compatibility


def decode_1d(tuning_curves, group, ep, bin_size, time_units="s", feature=None):
    """
    Deprecated, use `decode` instead.
    """
    warnings.warn(
        "decode_1d is deprecated and will be removed in a future version; use decode_bayes instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Occupancy
    if feature is None:
        occupancy = np.ones(tuning_curves.shape[0])
    elif isinstance(feature, nap.Tsd):
        diff = np.diff(tuning_curves.index.values)
        bins = tuning_curves.index.values[:-1] - diff / 2
        bins = np.hstack(
            (bins, [bins[-1] + diff[-1], bins[-1] + 2 * diff[-1]])
        )  # assuming the size of the last 2 bins is equal
        occupancy, _ = np.histogram(feature.values, bins)
    else:
        raise RuntimeError("Unknown format for feature in decode_1d")
    return decode_bayes(
        xr.DataArray(
            data=tuning_curves.values.T,
            coords={
                "unit": tuning_curves.columns.values,
                "0": tuning_curves.index.values,
            },
            attrs={"occupancy": occupancy},
        ),
        nap.TsGroup(group) if isinstance(group, dict) else group,
        ep,
        bin_size,
        time_units=time_units,
        uniform_prior=feature is None,
    )


def decode_2d(tuning_curves, group, ep, bin_size, xy, time_units="s", features=None):
    """
    Deprecated, use `decode` instead.
    """
    warnings.warn(
        "decode_2d is deprecated and will be removed in a future version; use decode_bayes instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Occupancy
    indexes = list(tuning_curves.keys())
    if features is None:
        occupancy = np.ones_like(tuning_curves[indexes[0]]).flatten()
    else:
        binsxy = []
        for i in range(len(xy)):
            diff = np.diff(xy[i])
            bins = xy[i][:-1] - diff / 2
            bins = np.hstack(
                (bins, [bins[-1] + diff[-1], bins[-1] + 2 * diff[-1]])
            )  # assuming the size of the last 2 bins is equal
            binsxy.append(bins)

        occupancy, _, _ = np.histogram2d(
            features[:, 0].values, features[:, 1].values, [binsxy[0], binsxy[1]]
        )
        occupancy = occupancy.flatten()
    return decode_bayes(
        xr.DataArray(
            data=[tuning_curves[i] for i in indexes],
            coords={"unit": indexes, "0": xy[0], "1": xy[1]},
            attrs={"occupancy": occupancy},
        ),
        nap.TsGroup(group) if isinstance(group, dict) else group,
        ep,
        bin_size,
        time_units=time_units,
        uniform_prior=features is None,
    )
