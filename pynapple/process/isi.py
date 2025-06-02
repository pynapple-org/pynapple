"""
Functions to compute interspike (or event) interval distributions.
"""

import numpy as np
import pandas as pd

from .. import core as nap


def compute_isi_distribution(
    ts,
    nb_bins=None,
    log_scale=False,
    ep=None,
):
    """
    Computes the interspike interval distribution.

    Parameters
    ----------
    ts : Ts
        The Ts for which the distribution will be computed.
    nb_bins : int
        Number of bins in the distribution.
    log_scale=False,
        Whether or not to log transform the distribution.
    ep : IntervalSet, optional
        The epoch on which interspike intervals are computed.
        If None, the epoch is the time support of the input.

    Returns
    -------
    pandas.DataFrame
        DataFrame to hold the distribution data.

    """
    if not isinstance(ts, nap.Ts):
        raise TypeError("ts should be a Ts.")

    if not isinstance(nb_bins, int):
        raise TypeError("nb_bins should be of type int.")

    if not isinstance(log_scale, bool):
        raise TypeError("log_scale should be of type bool.")

    if ep is None:
        ep = ts.time_support

    time_diffs = ts.time_diff(ep=ep)
    if log_scale:
        time_diffs = np.log10(time_diffs)

    min, max = time_diffs.values.min(), time_diffs.values.max()

    bins = np.linspace(min, max, nb_bins + 1)

    idx = np.digitize(time_diffs.values, bins) - 1

    counts = np.bincount(idx, minlength=nb_bins)

    return pd.DataFrame(index=bins, data=counts, columns=["isi"])
