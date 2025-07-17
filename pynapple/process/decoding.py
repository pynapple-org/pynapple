"""
Decoding functions for timestamps data (spike times). The first argument is always a tuning curves object.
"""

import numpy as np
import xarray as xr

from .. import core as nap


def decode_1d(tuning_curves, group, ep, bin_size, time_units="s", feature=None):
    """
    Perform Bayesian decoding over a one dimensional feature.
    See:
    Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J.
    (1998). Interpreting neuronal population activity by
    reconstruction: unified framework with application to
    hippocampal place cells. Journal of neurophysiology, 79(2),
    1017-1044.

    Parameters
    ----------
    tuning_curves : pandas.DataFrame
        Each column is the tuning curve of one neuron relative to the feature.
        Index should be the center of the bin.
    group : TsGroup, TsdFrame or dict of Ts/Tsd object.
        A group of neurons with the same index as tuning curves column names.
        You may also pass a TsdFrame with smoothed rates (recommended).
    ep : IntervalSet
        The epoch on which decoding is computed
    bin_size : float
        Bin size. Default is second. Use the parameter time_units to change it.
    time_units : str, optional
        Time unit of the bin size ('s' [default], 'ms', 'us').
    feature : Tsd, optional
        The 1d feature used to compute the tuning curves. Used to correct for occupancy.
        If feature is not passed, the occupancy is uniform.

    Returns
    -------
    Tsd
        The decoded feature
    TsdFrame
        The probability distribution of the decoded feature for each time bin

    Raises
    ------
    RuntimeError
        If group is not a dict of Ts/Tsd or TsGroup.
        If different size of neurons for tuning_curves and group.
        If indexes don't match between tuning_curves and group.
    """
    if isinstance(group, nap.TsdFrame):
        newgroup = group.restrict(ep)

        if tuning_curves.shape[1] != newgroup.shape[1]:
            raise RuntimeError("Different shapes for tuning_curves and group")

        if not np.all(tuning_curves.columns.values == np.array(newgroup.columns)):
            raise RuntimeError("Different indices for tuning curves and group keys")

        count = group

    elif isinstance(group, nap.TsGroup):
        newgroup = group.restrict(ep)

        if tuning_curves.shape[1] != len(newgroup):
            raise RuntimeError("Different shapes for tuning_curves and group")

        if not np.all(tuning_curves.columns.values == np.array(newgroup.keys())):
            raise RuntimeError("Different indices for tuning curves and group keys")

        # Bin spikes
        count = newgroup.count(bin_size, ep, time_units)

    elif isinstance(group, dict):
        newgroup = nap.TsGroup(group, time_support=ep)
        count = newgroup.count(bin_size, ep, time_units)

    else:
        raise RuntimeError("Unknown format for group")

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

    # Transforming to pure numpy array
    tc = tuning_curves.values
    ct = count.values

    bin_size_s = nap.TsIndex.format_timestamps(
        np.array([bin_size], dtype=np.float64), time_units
    )[0]

    p1 = np.exp(-bin_size_s * tc.sum(1))
    p2 = occupancy / occupancy.sum()

    ct2 = np.tile(ct[:, np.newaxis, :], (1, tc.shape[0], 1))

    p3 = np.prod(tc**ct2, -1)

    p = p1 * p2 * p3
    p = p / p.sum(1)[:, np.newaxis]

    idxmax = np.argmax(p, 1)

    p = nap.TsdFrame(
        t=count.index, d=p, time_support=ep, columns=tuning_curves.index.values
    )

    decoded = nap.Tsd(
        t=count.index, d=tuning_curves.index.values[idxmax], time_support=ep
    )

    return decoded, p


def decode_2d(tuning_curves, group, ep, bin_size, xy, time_units="s", features=None):
    """
    Performs Bayesian decoding over 2 dimensional features.

    See:
    Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J.
    (1998). Interpreting neuronal population activity by
    reconstruction: unified framework with application to
    hippocampal place cells. Journal of neurophysiology, 79(2),
    1017-1044.

    Parameters
    ----------
    tuning_curves : dict
        Dictionary of 2d tuning curves (one for each neuron).
    group : TsGroup, TsdFrame or dict of Ts/Tsd object.
        A group of neurons with the same keys as tuning_curves dictionary.
        You may also pass a TsdFrame with smoothed rates (recommended).
    ep : IntervalSet
        The epoch on which decoding is computed
    bin_size : float
        Bin size. Default is second. Use the parameter time_units to change it.
    xy : tuple
        A tuple of bin positions for the tuning curves i.e. xy=(x,y)
    time_units : str, optional
        Time unit of the bin size ('s' [default], 'ms', 'us').
    features : TsdFrame
        The 2 columns features used to compute the tuning curves. Used to correct for occupancy.
        If feature is not passed, the occupancy is uniform.

    Returns
    -------
    Tsd
        The decoded feature in 2d
    numpy.ndarray
        The probability distribution of the decoded trajectory for each time bin

    Raises
    ------
    RuntimeError
        If group is not a dict of Ts/Tsd or TsGroup.
        If different size of neurons for tuning_curves and group.
        If indexes don't match between tuning_curves and group.

    """

    if type(group) is nap.TsdFrame:
        newgroup = group.restrict(ep)
        numcells = newgroup.shape[1]

        if len(tuning_curves) != numcells:
            raise RuntimeError("Different shapes for tuning_curves and group")

        if not np.all(
            np.array(list(tuning_curves.keys())) == np.array(newgroup.columns)
        ):
            raise RuntimeError("Different indices for tuning curves and group keys")

        count = group

    elif type(group) is nap.TsGroup:
        newgroup = group.restrict(ep)
        numcells = len(newgroup)

        if len(tuning_curves) != numcells:
            raise RuntimeError("Different shapes for tuning_curves and group")

        if not np.all(
            np.array(list(tuning_curves.keys())) == np.array(newgroup.keys())
        ):
            raise RuntimeError("Different indices for tuning curves and group keys")

        count = newgroup.count(bin_size, ep, time_units)

    elif type(group) is dict:
        newgroup = nap.TsGroup(group, time_support=ep)
        count = newgroup.count(bin_size, ep, time_units)

    else:
        raise RuntimeError("Unknown format for group")

    indexes = list(tuning_curves.keys())

    # Occupancy
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

    # Transforming to pure numpy array
    tc = np.array([tuning_curves[i] for i in tuning_curves.keys()])
    tc = tc.reshape(tc.shape[0], np.prod(tc.shape[1:]))
    tc = tc.T
    ct = count.values
    bin_size_s = nap.TsIndex.format_timestamps(
        np.array([bin_size], dtype=np.float64), time_units
    )[0]

    p1 = np.exp(-bin_size_s * np.nansum(tc, 1))
    p2 = occupancy / occupancy.sum()

    ct2 = np.tile(ct[:, np.newaxis, :], (1, tc.shape[0], 1))

    p3 = np.nanprod(tc**ct2, -1)

    p = p1 * p2 * p3
    p = p / p.sum(1)[:, np.newaxis]

    idxmax = np.argmax(p, 1)

    p = p.reshape(p.shape[0], len(xy[0]), len(xy[1]))

    idxmax2d = np.unravel_index(idxmax, (len(xy[0]), len(xy[1])))

    if features is not None:
        cols = features.columns
    else:
        cols = np.arange(2)

    decoded = nap.TsdFrame(
        t=count.index,
        d=np.vstack((xy[0][idxmax2d[0]], xy[1][idxmax2d[1]])).T,
        time_support=ep,
        columns=cols,
    )

    return decoded, p


def decode(tuning_curves, group, epochs, bin_size, time_units="s"):
    """
    Performs Bayesian decoding over n-dimensional features.

    See:
    Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J.
    (1998). Interpreting neuronal population activity by
    reconstruction: unified framework with application to
    hippocampal place cells. Journal of neurophysiology, 79(2),
    1017-1044.

    Parameters
    ----------
    tuning_curves : xr.DataArray
        Tuning curves as outputed by `compute_tuning_curves` (one for each unit).
    group : TsGroup, TsdFrame or dict of Ts/Tsd object.
        A group of neurons with the same keys as the tuning curves.
        You may also pass a TsdFrame with smoothed rates (recommended).
    epochs : IntervalSet
        The epochs on which decoding is computed
    bin_size : float
        Bin size. Default is second. Use the parameter time_units to change it.
    time_units : str, optional
        Time unit of the bin size ('s' [default], 'ms', 'us').

    Returns
    -------
    Tsd
        The decoded feature
    numpy.ndarray
        The probability distribution of the decoded trajectory for each time bin

    Raises
    ------
    RuntimeError
        If group is not a dict of Ts/Tsd or TsGroup.
        If different size of neurons for tuning_curves and group.
        If indexes don't match between tuning_curves and group.

    """

    # check tuning curves
    if not isinstance(tuning_curves, xr.DataArray):
        raise TypeError(
            "tuning_curves should be an xr.DataArray as outputed by compute_tuning_curves"
        )

    # check group
    if isinstance(group, (dict, nap.TsGroup)):
        numcells = len(group)

        if tuning_curves.sizes["unit"] != numcells:
            raise RuntimeError("Different shapes for tuning_curves and group")

        if not np.all(tuning_curves.coords["unit"] == np.array(list(group.keys()))):
            raise RuntimeError("Different indices for tuning curves and group keys")

        if isinstance(group, dict):
            group = nap.TsGroup(group, time_support=epochs)
        count = group.count(bin_size, epochs, time_units)
    elif isinstance(group, nap.TsdFrame):
        numcells = group.shape[1]

        if tuning_curves.sizes["unit"] != numcells:
            raise RuntimeError("Different shapes for tuning_curves and group")

        if not np.all(tuning_curves.coords["unit"] == group.columns):
            raise RuntimeError("Different indices for tuning curves and group keys")

        count = group
    else:
        raise RuntimeError("Unknown format for group")

    if "occupancy" in tuning_curves.dims:
        occupancy = tuning_curves.coords["occupancy"].values.flatten()
    else:
        occupancy = np.ones_like(tuning_curves[0]).flatten()

    # Transforming to pure numpy array
    tc = tuning_curves.values.reshape(tuning_curves.sizes["unit"], -1).T
    ct = count.values
    bin_size_s = nap.TsIndex.format_timestamps(
        np.array([bin_size], dtype=np.float64), time_units
    )[0]

    p1 = np.exp(-bin_size_s * np.nansum(tc, 1))
    p2 = occupancy / occupancy.sum()

    ct2 = np.tile(ct[:, np.newaxis, :], (1, tc.shape[0], 1))

    p3 = np.nanprod(tc**ct2, -1)

    p = p1 * p2 * p3
    p = p / p.sum(1)[:, np.newaxis]

    idxmax = np.argmax(p, 1)

    p = p.reshape(p.shape[0], *tuning_curves.shape[1:])
    p = getattr(nap, f"Tsd{'Tensor' if p.ndim > 2 else 'Frame'}")(
        t=count.index,
        d=p,
        time_support=epochs,
    )

    idxmax = np.unravel_index(idxmax, tuning_curves.shape[1:])

    if tuning_curves.ndim == 2:
        decoded = nap.Tsd(
            t=count.index,
            d=tuning_curves.coords[tuning_curves.dims[1]][idxmax[0]].values,
            time_support=epochs,
        )
    else:
        decoded = nap.TsdFrame(
            t=count.index,
            d=np.stack(
                [
                    tuning_curves.coords[dim][idxmax[i]]
                    for i, dim in enumerate(tuning_curves.dims[1:])
                ],
                axis=1,
            ),
            time_support=epochs,
            columns=tuning_curves.dims[1:],
        )

    return decoded, p
