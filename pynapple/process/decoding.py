"""
Functions to decode n-dimensional features.
"""

import inspect
import warnings
from functools import wraps

import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist

from .. import core as nap


def _format_decoding_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        kwargs = bound.arguments

        # check tuning curves
        tuning_curves = kwargs["tuning_curves"]
        if not isinstance(tuning_curves, xr.DataArray):
            raise TypeError(
                "tuning_curves should be an xarray.DataArray as computed by compute_tuning_curves."
            )

        # check data
        data = kwargs["data"]
        was_continuous = True
        if isinstance(data, nap.TsdFrame):
            # check match bin_size
            actual_bin_size = np.mean(data.time_diff().values)
            passed_bin_size = kwargs["bin_size"]
            if not isinstance(passed_bin_size, (int, float)):
                raise ValueError("bin_size should be a number.")
            if not np.isclose(
                actual_bin_size,
                nap.TsIndex.format_timestamps(
                    np.array([passed_bin_size], dtype=np.float64),
                    units=kwargs["time_units"],
                ),
            )[0]:
                warnings.warn("passed bin_size is different from actual data bin size.")
        elif isinstance(data, nap.TsGroup):
            data = data.count(
                kwargs["bin_size"], kwargs["epochs"], time_units=kwargs["time_units"]
            )
            was_continuous = False
        else:
            raise TypeError("Unknown format for data.")

        # check match tuning curves and data
        if tuning_curves.sizes["unit"] != data.shape[1]:
            raise ValueError("Different shapes for tuning_curves and data.")
        if not np.all(tuning_curves.coords["unit"] == data.columns.values):
            raise ValueError("Different indices for tuning curves and data keys.")

        if (
            "uniform_prior" in kwargs
            and not kwargs["uniform_prior"]
            and "occupancy" not in tuning_curves.attrs
        ):
            raise ValueError(
                "uniform_prior set to False but no occupancy found in tuning curves."
            )

        # smooth
        sliding_window = kwargs["sliding_window"]
        if sliding_window is not None:
            if not isinstance(sliding_window, int):
                raise ValueError("sliding_window should be a integer.")
            if sliding_window < 1:
                raise ValueError("sliding_window should be >= 1.")
            data = data.convolve(
                np.ones(sliding_window),
                ep=kwargs["epochs"],
            )
            if was_continuous:
                data = data / sliding_window
            else:
                bin_size = sliding_window * kwargs["bin_size"]
                kwargs["bin_size"] = bin_size

        kwargs["data"] = data

        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


def _format_decoding_outputs(dist, tuning_curves, data, epochs, greater_is_better):
    # Get the index of the decoded class
    filler = -np.inf if greater_is_better else np.inf
    filled = np.where(np.isnan(dist), filler, dist)
    idx = getattr(np, "argmax" if greater_is_better else "argmin")(filled, axis=1)

    # Replace with -1 where all values were NaN
    all_nan = np.isnan(dist).all(axis=1)
    idx[all_nan] = -1

    # Format probability/distance distribution
    dist = dist.reshape(dist.shape[0], *tuning_curves.shape[1:])
    if dist.ndim > 2:
        dist = nap.TsdTensor(
            t=data.index,
            d=dist,
            time_support=epochs,
        )
    else:
        dist = nap.TsdFrame(
            t=data.index,
            d=dist,
            time_support=epochs,
            columns=tuning_curves.coords[tuning_curves.dims[1]].values,
        )

    # Format decoded index
    shape = tuning_curves.shape[1:]
    valid = idx != -1

    if tuning_curves.ndim == 2:
        decoded_values = np.full(len(idx), np.nan)
        decoded_values[valid] = tuning_curves.coords[tuning_curves.dims[1]].values[
            idx[valid]
        ]
        decoded = nap.Tsd(
            t=data.index,
            d=decoded_values,
            time_support=epochs,
        )
    else:
        # unravel valid indices only
        unraveled = [np.full(len(idx), np.nan) for _ in shape]
        unraveled_indices = np.unravel_index(idx[valid], shape)
        for i in range(len(shape)):
            unraveled[i][valid] = tuning_curves.coords[
                tuning_curves.dims[1 + i]
            ].values[unraveled_indices[i]]

        decoded = nap.TsdFrame(
            t=data.index,
            d=np.stack(unraveled, axis=1),
            time_support=epochs,
            columns=tuning_curves.dims[1:],
        )

    return decoded, dist


@_format_decoding_inputs
def decode_bayes(
    tuning_curves,
    data,
    epochs,
    bin_size,
    sliding_window=None,
    time_units="s",
    uniform_prior=True,
):
    """
    Performs Bayesian decoding over n-dimensional features.

    The algorithm is based on Bayes' rule:

    .. math::

        P(x|n) \\propto P(n|x) P(x)

    where:

    - :math:`P(x|n)` is the **posterior probability** of the feature value given the observed neural activity.
    - :math:`P(n|x)` is the **likelihood** of the neural activity given the feature value.
    - :math:`P(x)` is the **prior** probability of the feature value.

    Mapping this to the function:

    - :math:`P(x|n)` is the estimated probability distribution over the decoded feature for each time bin.
      This is the output of the function. The decoded value is the one with the maximum posterior probability.
    - :math:`P(n|x)` is determined by the tuning curves. Assuming spikes follow a Poisson distribution and
      neurons are conditionally independent:

      .. math::

          P(n|x) = \\prod_{i=1}^{N} P(n_i|x) = \\prod_{i=1}^{N} \\frac{\\lambda_i^{n_i} e^{-\\lambda_i}}{n_i!}

      where :math:`\\lambda_i` is the expected firing rate of neuron :math:`i` at feature value :math:`x`,
      and :math:`n_i` is the spike count of neuron :math:`i`.

    - :math:`P(x)` depends on the value of the ``uniform_prior`` argument.
      If ``uniform_prior=True``, it is a uniform distribution over feature values.
      If ``uniform_prior=False``, it is based on the occupancy (i.e. the time spent in each feature bin during tuning curve estimation).

    References
    ----------
    .. [1] Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J.
           (1998). Interpreting neuronal population activity by
           reconstruction: unified framework with application to
           hippocampal place cells. Journal of neurophysiology, 79(2),
           1017-1044.

    Parameters
    ----------
    tuning_curves : xarray.DataArray
        Tuning curves as computed by :func:`~pynapple.process.tuning_curves.compute_tuning_curves`.
    data : TsGroup or TsdFrame
        Neural activity with the same keys as the tuning curves.
        You may also pass a TsdFrame with smoothed counts.
    epochs : IntervalSet
        The epochs on which decoding is computed
    bin_size : float
        Bin size. Default in seconds. Use ``time_units`` to change it.
    smoothing : str, optional
        Type of smoothing to apply to the binned spikes counts (``None`` [default], ``gaussian``, ``uniform``).
    smoothing_window : float, optional
        Size smoothing window. Default in seconds. Use ``time_units`` to change it.
    time_units : str, optional
        Time unit of the bin size (``s`` [default], ``ms``, ``us``).
    uniform_prior : bool, optional
        If True (default), uses a uniform distribution as a prior.
        If False, uses the occupancy from the tuning curves as a prior over the feature
        probability distribution.

    Returns
    -------
    Tsd
        The decoded feature.
    TsdFrame, TsdTensor
        The probability distribution of the decoded feature for each time bin.

    Examples
    --------
    In the simplest case, we can decode a single feature (e.g., position) from a group of neurons:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> data = nap.TsGroup({i: nap.Ts(t=np.arange(0, 50) + 50 * i) for i in range(2)})
    >>> feature = nap.Tsd(t=np.arange(0, 100, 1), d=np.repeat(np.arange(0, 2), 50))
    >>> tuning_curves = nap.compute_tuning_curves(data, feature, bins=2, range=(-.5, 1.5))
    >>> epochs = nap.IntervalSet([0, 100])
    >>> decoded, p = nap.decode_bayes(tuning_curves, data, epochs=epochs, bin_size=1)
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
    Time (s)    0.0    1.0
    ----------  -----  -----
    0.5         1.0    0.0
    1.5         1.0    0.0
    2.5         1.0    0.0
    3.5         1.0    0.0
    4.5         1.0    0.0
    5.5         1.0    0.0
    6.5         1.0    0.0
    ...         ...    ...
    93.5        0.0    1.0
    94.5        0.0    1.0
    95.5        0.0    1.0
    96.5        0.0    1.0
    97.5        0.0    1.0
    98.5        0.0    1.0
    99.5        0.0    1.0
    dtype: float64, shape: (100, 2)

    p is a `TsdFrame` object containing the probability distribution for each time bin.

    The function also works for multiple features, in which case it does n-dimensional decoding:

    >>> features = nap.TsdFrame(
    ...     t=np.arange(0, 100, 1),
    ...     d=np.vstack((np.repeat(np.arange(0, 2), 50), np.tile(np.arange(0, 2), 50))).T,
    ... )
    >>> data = nap.TsGroup(
    ...     {
    ...         0: nap.Ts(np.arange(0, 50, 2)),
    ...         1: nap.Ts(np.arange(1, 51, 2)),
    ...         2: nap.Ts(np.arange(50, 100, 2)),
    ...         3: nap.Ts(np.arange(51, 101, 2)),
    ...     }
    ... )
    >>> tuning_curves = nap.compute_tuning_curves(data, features, bins=2, range=[(-.5, 1.5)]*2)
    >>> decoded, p = nap.decode_bayes(tuning_curves, data, epochs=epochs, bin_size=1)
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

    >>> data = data.count(1).smooth(2)
    >>> tuning_curves = nap.compute_tuning_curves(data, features, bins=2, range=[(-.5, 1.5)]*2)
    >>> decoded, p = nap.decode_bayes(tuning_curves, data, epochs=epochs, bin_size=1)
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
    prior = (
        np.ones_like(tuning_curves[0]).flatten()
        if uniform_prior
        else tuning_curves.attrs["occupancy"].flatten()
    )
    prior = prior.astype(np.float64)
    prior /= prior.sum()

    rate_map = tuning_curves.values.reshape(tuning_curves.sizes["unit"], -1).T
    observed_counts = data.values
    bin_size_s = nap.TsIndex.format_timestamps(
        np.array([bin_size], dtype=np.float64), time_units
    )[0]
    observed_counts_expanded = np.tile(
        observed_counts[:, np.newaxis, :], (1, rate_map.shape[0], 1)
    )

    EPS = 1e-12
    log_likelihood = np.nansum(
        observed_counts_expanded * np.log(rate_map + EPS) - bin_size_s * rate_map,
        axis=-1,
    )

    log_posterior = log_likelihood + np.log(prior)
    posterior = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)

    return _format_decoding_outputs(
        posterior, tuning_curves, data, epochs, greater_is_better=True
    )


@_format_decoding_inputs
def decode_template(
    tuning_curves,
    data,
    epochs,
    bin_size,
    metric="correlation",
    sliding_window=None,
    time_units="s",
):
    """
    Performs template matching decoding over n-dimensional features.

    The algorithm decodes as follow:

    .. math::

        \\hat{x}(t) = \\arg\\min\\limits_{x} [dist(f(x), n(t))]

    where:

    - :math:`f(x)` is the the tuning curve function.
    - :math:`n(t)` is input neural activity at time :math:`t`.
    - :math:`dist` is a distance metric.

    The algorithm computes the distance between the observed neural activity and the tuning curves for every time bin.
    The decoded feature at each time bin corresponds to the tuning curve bin with the smallest distance.

    See :func:`scipy.spatial.distance.cdist` for available distance metrics and how they are computed.

    References
    ----------
    .. [1] Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J.
           (1998). Interpreting neuronal population activity by
           reconstruction: unified framework with application to
           hippocampal place cells. Journal of neurophysiology, 79(2),
           1017-1044.

    Parameters
    ----------
    tuning_curves : xarray.DataArray
        Tuning curves as computed by :func:`~pynapple.process.tuning_curves.compute_tuning_curves`.
    data : TsGroup or TsdFrame
        Neural activity with the same keys as the tuning curves.
        You may also pass a TsdFrame with smoothed counts.
    epochs : IntervalSet
        The epochs on which decoding is computed
    bin_size : float
        Bin size. Default is second. Use ``time_units`` to change it.
    metric : str or callable, optional
        The distance metric to use for template matching.

        If a string, passed to :func:`scipy.spatial.distance.cdist`, must be one of:
        ``braycurtis``, ``canberra``, ``chebyshev``, ``cityblock``, ``correlation``,
        ``cosine``, ``dice``, ``euclidean``, ``hamming``, ``jaccard``, ``jensenshannon``,
        ``kulczynski1``, ``mahalanobis``, ``matching``, ``minkowski``, ``rogerstanimoto``,
        ``russellrao``, ``seuclidean``, ``sokalmichener``, ``sokalsneath``,
        ``sqeuclidean`` or ``yule``.

        Default is ``correlation``.

        .. note::
           Some metrics may not be suitable for all types of data.
           For example, metrics such as ``hamming`` do not handle NaN values.

        If a callable, it must have the signature ``metric(u, v) -> float`` and
        return the distance between two 1D arrays.
    smoothing : str, optional
        Type of smoothing to apply to the binned spikes counts (``None`` [default], ``gaussian``, ``uniform``).
    smoothing_window : float, optional
        Size smoothing window. Default in seconds. Use ``time_units`` to change it.
    time_units : str, optional
        Time unit of the bin size (``s`` [default], ``ms``, ``us``).

    Returns
    -------
    Tsd
        The decoded feature
    TsdFrame or TsdTensor
        The distance matrix between the neural activity and the tuning curves for each time bin.

    Examples
    --------
    In the simplest case, we can decode a single feature (e.g., position) from a group of neurons:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> group = nap.TsGroup({i: nap.Ts(t=np.arange(0, 50) + 50 * i) for i in range(2)})
    >>> feature = nap.Tsd(t=np.arange(0, 100, 1), d=np.repeat(np.arange(0, 2), 50))
    >>> tuning_curves = nap.compute_tuning_curves(group, feature, bins=2, range=(-.5, 1.5))
    >>> epochs = nap.IntervalSet([0, 100])
    >>> decoded, dist = nap.decode_template(tuning_curves, group, epochs=epochs, bin_size=1)
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
    Time (s)    0.0    1.0
    ----------  -----  -----
    0.5         0.0    2.0
    1.5         0.0    2.0
    2.5         0.0    2.0
    3.5         0.0    2.0
    4.5         0.0    2.0
    5.5         0.0    2.0
    ...         ...    ...
    94.5        2.0    0.0
    95.5        2.0    0.0
    96.5        2.0    0.0
    97.5        2.0    0.0
    98.5        2.0    0.0
    99.5        2.0    0.0
    dtype: float64, shape: (100, 2)

    dist is a `TsdFrame` object containing the distances for each time bin.

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
    >>> decoded, dist = nap.decode_template(tuning_curves, group, epochs=epochs, bin_size=1)
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

    >>> dist
    Time (s)
    ----------  --------------------------
    0.5         [[0.      , 1.333333] ...]
    1.5         [[1.333333, 0.      ] ...]
    2.5         [[0.      , 1.333333] ...]
    3.5         [[1.333333, 0.      ] ...]
    4.5         [[0.      , 1.333333] ...]
    5.5         [[1.333333, 0.      ] ...]
    ...
    94.5        [[1.333333, 1.333333] ...]
    95.5        [[1.333333, 1.333333] ...]
    96.5        [[1.333333, 1.333333] ...]
    97.5        [[1.333333, 1.333333] ...]
    98.5        [[1.333333, 1.333333] ...]
    99.5        [[1.333333, 1.333333] ...]
    dtype: float64, shape: (100, 2, 2)

    and dist is a `TsdTensor` object containing the distances for each time bin.

    It is also possible to pass continuous values instead of spikes (e.g. calcium imaging):

    >>> time = np.arange(0,100, 0.1)
    >>> group = nap.TsdFrame(t=time, d=np.stack([time % 0.5, time %1], axis=1))
    >>> tuning_curves = nap.compute_tuning_curves(group, features, bins=2, range=[(-.5, 1.5)]*2)
    >>> decoded, dist = nap.decode_template(tuning_curves, group, epochs=epochs, bin_size=1)
    >>> decoded
    Time (s)    0    1
    ----------  ---  ---
    0.0         0.0  0.0
    0.1         0.0  0.0
    0.2         0.0  0.0
    0.3         0.0  0.0
    0.4         0.0  0.0
    0.5         1.0  1.0
    0.6         1.0  1.0
    ...         ...  ...
    99.3        0.0  0.0
    99.4        0.0  0.0
    99.5        1.0  1.0
    99.6        1.0  1.0
    99.7        1.0  1.0
    99.8        1.0  1.0
    99.9        1.0  1.0
    dtype: float64, shape: (1000, 2)
    """
    tc = tuning_curves.values.reshape(tuning_curves.sizes["unit"], -1)
    ct = data.values

    return _format_decoding_outputs(
        cdist(ct, tc.T, metric=metric),
        tuning_curves,
        data,
        epochs,
        greater_is_better=False,
    )


# -------------------------------------------------------------------------------------
# Deprecated functions for backward compatibility
# -------------------------------------------------------------------------------------


def decode_1d(tuning_curves, group, ep, bin_size, time_units="s", feature=None):
    """
    .. deprecated:: 0.9.2
          `decode_1d` will be removed in Pynapple 1.0.0, it is replaced by
          `decode_bayes` because the latter works for N dimensions.
    """
    warnings.warn(
        "decode_1d is deprecated and will be removed in a future version; use decode_bayes instead.",
        FutureWarning,
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
    .. deprecated:: 0.9.2
          `decode_2d` will be removed in Pynapple 1.0.0, it is replaced by
          `decode_bayes` because the latter works for N dimensions.
    """
    warnings.warn(
        "decode_2d is deprecated and will be removed in a future version; use decode_bayes instead.",
        FutureWarning,
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
