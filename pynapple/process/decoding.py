"""
Decoding functions for timestamps data (spike times). The first argument is always a tuning curves object.
"""

import warnings

import numpy as np
import xarray as xr

from .. import core as nap


def decode_bayes(
    tuning_curves, data, epochs, bin_size, time_units="s", uniform_prior=True
):
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
    data : TsGroup, TsdFrame or dict of Ts, Tsd
        Neural activity with the same keys as the tuning curves.
        You may also pass a TsdFrame with smoothed rates (recommended).
    epochs : IntervalSet
        The epochs on which decoding is computed
    bin_size : float
        Bin size. Default is second. Use the parameter time_units to change it.
    time_units : str, optional
        Time unit of the bin size ('s' [default], 'ms', 'us').
    uniform_prior : bool, optional
        If True (default), uses a uniform distribution as a prior.
        If False, uses the occupancy from the tuning curves as a prior over the feature
        probability distribution.

    Returns
    -------
    Tsd
        The decoded feature
    TsdFrame, TsdTensor
        The probability distribution of the decoded trajectory for each time bin


    Examples
    --------
    In the simplest case, we can decode a single feature (e.g., position) from a group of neurons:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> group = nap.TsGroup({i: nap.Ts(t=np.arange(0, 50) + 50 * i) for i in range(2)})
    >>> feature = nap.Tsd(t=np.arange(0, 100, 1), d=np.repeat(np.arange(0, 2), 50))
    >>> tuning_curves = nap.compute_tuning_curves(group, feature, bins=2, range=(-.5, 1.5))
    >>> epochs = nap.IntervalSet([0, 100])
    >>> decoded, p = nap.decode_bayes(tuning_curves, group, epochs=epochs, bin_size=1)
    >>> decoded
    Time (s)
    ----------  --
    0.5          0
    1.5          0
    2.5          0
    3.5          0
    4.5          0
    5.5          0
    6.5          0
    ...
    93.5         1
    94.5         1
    95.5         1
    96.5         1
    97.5         1
    98.5         1
    99.5         1
    dtype: float64, shape: (100,)

    decode is a `Tsd` object containing the decoded feature for each time bin.

    >>> p
    Time (s)    0    1
    ----------  ---  ---
    0.5         1.0  0.0
    1.5         1.0  0.0
    2.5         1.0  0.0
    3.5         1.0  0.0
    4.5         1.0  0.0
    5.5         1.0  0.0
    6.5         1.0  0.0
    ...         ...  ...
    93.5        0.0  1.0
    94.5        0.0  1.0
    95.5        0.0  1.0
    96.5        0.0  1.0
    97.5        0.0  1.0
    98.5        0.0  1.0
    99.5        0.0  1.0
    dtype: float64, shape: (100, 2)

    p is a `TsdFrame` object containing the probability distribution for each time bin.

    The function also works for multiple features, in which case it does n-dimensional decoding:

    >>> features = nap.TsdFrame(
    ...     t=np.arange(0, 100, 1),
    ...     d=np.vstack((np.repeat(np.arange(0, 2), 50), np.tile(np.arange(0, 2), 50))).T,
    ... )
    >>> group = nap.TsGroup(
    ...     {
    ...         0: nap.Ts(np.arange(0, 50, 2)),
    ...         1: nap.Ts(np.arange(1, 51, 2)),
    ...         2: nap.Ts(np.arange(50, 100, 2)),
    ...         3: nap.Ts(np.arange(51, 101, 2)),
    ...     }
    ... )
    >>> tuning_curves = nap.compute_tuning_curves(group, features, bins=2, range=[(-.5, 1.5)]*2)
    >>> decoded, p = nap.decode_bayes(tuning_curves, group, epochs=epochs, bin_size=1)
    >>> decoded
    Time (s)    0    1
    ----------  ---  ---
    0.5         0.0  0.0
    1.5         0.0  1.0
    2.5         0.0  0.0
    3.5         0.0  1.0
    4.5         0.0  0.0
    5.5         0.0  1.0
    6.5         0.0  0.0
    ...         ...  ...
    93.5        1.0  1.0
    94.5        1.0  0.0
    95.5        1.0  1.0
    96.5        1.0  0.0
    97.5        1.0  1.0
    98.5        1.0  0.0
    99.5        1.0  1.0
    dtype: float64, shape: (100, 2)

    decoded is now a `TsdFrame` object containing the decoded features for each time bin.

    >>> p
    Time (s)
    ----------  --------------
    0.5         [[1., 0.] ...]
    1.5         [[0., 1.] ...]
    2.5         [[1., 0.] ...]
    3.5         [[0., 1.] ...]
    4.5         [[1., 0.] ...]
    5.5         [[0., 1.] ...]
    6.5         [[1., 0.] ...]
    ...
    93.5        [[0., 0.] ...]
    94.5        [[0., 0.] ...]
    95.5        [[0., 0.] ...]
    96.5        [[0., 0.] ...]
    97.5        [[0., 0.] ...]
    98.5        [[0., 0.] ...]
    99.5        [[0., 0.] ...]
    dtype: float64, shape: (100, 2, 2)

    and p is a `TsdTensor` object containing the probability distribution for each time bin.

    It is also possible to pass continuous values instead of spikes (e.g. smoothed spike counts):

    >>> frame = group.count(1).smooth(2)
    >>> tuning_curves = nap.compute_tuning_curves(frame, features, bins=2, range=[(-.5, 1.5)]*2)
    >>> decoded, p = nap.decode_bayes(tuning_curves, frame, epochs=epochs, bin_size=1)
    >>> decoded
    Time (s)    0    1
    ----------  ---  ---
    0.5         0.0  1.0
    1.5         0.0  1.0
    2.5         0.0  1.0
    3.5         0.0  1.0
    4.5         0.0  0.0
    5.5         0.0  0.0
    6.5         0.0  0.0
    ...         ...  ...
    92.5        1.0  0.0
    93.5        1.0  0.0
    94.5        1.0  0.0
    95.5        1.0  1.0
    96.5        1.0  1.0
    97.5        1.0  1.0
    98.5        1.0  1.0
    dtype: float64, shape: (98, 2)
    """

    # check tuning curves
    if not isinstance(tuning_curves, xr.DataArray):
        raise TypeError(
            "tuning_curves should be an xr.DataArray as outputed by compute_tuning_curves."
        )

    # check data
    if isinstance(data, (dict, nap.TsGroup)):
        numcells = len(data)

        if tuning_curves.sizes["unit"] != numcells:
            raise ValueError("Different shapes for tuning_curves and data.")

        if not np.all(tuning_curves.coords["unit"] == np.array(list(data.keys()))):
            raise ValueError("Different indices for tuning curves and data keys.")

        if isinstance(data, dict):
            data = nap.TsGroup(data, time_support=epochs)
        count = data.count(bin_size, epochs, time_units)
    elif isinstance(data, nap.TsdFrame):
        numcells = data.shape[1]

        if tuning_curves.sizes["unit"] != numcells:
            raise ValueError("Different shapes for tuning_curves and data.")

        if not np.all(tuning_curves.coords["unit"] == data.columns):
            raise ValueError("Different indices for tuning curves and data keys.")

        count = data
    else:
        raise TypeError("Unknown format for data.")

    if uniform_prior:
        occupancy = np.ones_like(tuning_curves[0]).flatten()
    else:
        if "occupancy" not in tuning_curves.attrs:
            raise ValueError(
                "uniform_prior set to False but no occupancy found in tuning curves."
            )
        occupancy = tuning_curves.attrs["occupancy"].flatten()

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
    if p.ndim > 2:
        p = nap.TsdTensor(
            t=count.index,
            d=p,
            time_support=epochs,
        )
    else:
        p = nap.TsdFrame(
            t=count.index,
            d=p,
            time_support=epochs,
            columns=tuning_curves.coords[tuning_curves.dims[1]].values,
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


def decode_1d(tuning_curves, group, ep, bin_size, time_units="s", feature=None):
    warnings.warn(
        "decode_1d is deprecated and will be removed in a future version; use decode instead.",
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
            data=tuning_curves.values,
            coords={
                "unit": tuning_curves.columns.values,
                "0": tuning_curves.index.values,
            },
            attrs={"occupancy": occupancy},
        ),
        group,
        ep,
        bin_size,
        time_units=time_units,
        uniform_prior=feature is None,
    )


def decode_2d(tuning_curves, group, ep, bin_size, xy, time_units="s", features=None):
    warnings.warn(
        "decode_2d is deprecated and will be removed in a future version; use decode instead.",
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
        group,
        ep,
        bin_size,
        time_units=time_units,
        uniform_prior=features is None,
    )
