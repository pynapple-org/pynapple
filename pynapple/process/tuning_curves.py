"""
Functions to compute n-dimensional tuning curves.
"""

import inspect
import warnings
from collections.abc import Iterable
from functools import wraps

import numpy as np
import pandas as pd
import xarray as xr

from .. import core as nap


def compute_tuning_curves(
    data,
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
    data : TsGroup, TsdFrame, Ts, Tsd
        The data for which the tuning curves will be computed.
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
            occupancy:  [[100. 100.  nan]\\n [100.  50.  50.]\\n [100.  nan 100.]\\n [ 5...
            bin_edges:  [array([0.  , 0.18, 0.36, 0.54, 0.72, 0.9 ]), array([0.      ...

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

    In all of these cases, it is also possible to pass continuous values instead of spikes (e.g. calcium imaging data), in that case the mean response is computed:

        >>> frame = nap.TsdFrame(d=np.random.rand(2000, 3), t=np.arange(0, 100, 0.05))
        >>> tcs = nap.compute_tuning_curves(frame, feature, bins=10)
        >>> tcs
        <xarray.DataArray (unit: 3, 0: 10)> Size: 240B
        array([[0.49147343, 0.50190395, 0.50971339, 0.50128013, 0.54332711,
                0.49712328, 0.49594611, 0.5110517 , 0.52247351, 0.52057658],
               [0.51132036, 0.46410557, 0.47732505, 0.49830908, 0.53523019,
                0.53099429, 0.48668499, 0.44198555, 0.49222208, 0.47453398],
               [0.46591801, 0.50662914, 0.46875882, 0.48734997, 0.51836574,
                0.50722266, 0.48943577, 0.49730095, 0.47944075, 0.48623693]])
        Coordinates:
          * unit     (unit) int64 24B 0 1 2
          * 0        (0) float64 80B 0.045 0.135 0.225 0.315 ... 0.585 0.675 0.765 0.855
        Attributes:
            occupancy:  [100. 100. 100. 100. 100. 100. 100. 100. 100. 100.]
            bin_edges:  [array([0.  , 0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, 0.72,...
    """

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
    if not isinstance(return_pandas, bool):
        raise TypeError("return_pandas should be a boolean.")

    # occupancy
    occupancy, bin_edges = np.histogramdd(features, bins=bins, range=range)

    # tuning curves
    keys = (
        data.keys()
        if isinstance(data, nap.TsGroup)
        else data.columns if isinstance(data, nap.TsdFrame) else [0]
    )
    tcs = np.zeros([len(keys), *occupancy.shape])
    if isinstance(data, (nap.TsGroup, nap.Ts)):
        # SPIKES
        if isinstance(data, nap.Ts):
            data = {0: data}
        for i, n in enumerate(keys):
            tcs[i] = np.histogramdd(
                data[n].value_from(features, epochs),
                bins=bin_edges,
            )[0]
        occupancy[occupancy == 0.0] = np.nan
        tcs = (tcs / occupancy) * fs
    else:
        # RATES
        values = data.value_from(features, epochs)
        if isinstance(data, nap.Tsd):
            data = np.expand_dims(data.values, -1)
        counts = np.histogramdd(values, bins=bin_edges)[0]
        counts[counts == 0] = np.nan
        for i, n in enumerate(keys):
            tcs[i] = np.histogramdd(
                values,
                weights=data[:, i],
                bins=bin_edges,
            )[0]
        tcs /= counts
        tcs[np.isnan(tcs)] = 0.0
        tcs[:, occupancy == 0.0] = np.nan

    tcs = xr.DataArray(
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
    if return_pandas:
        return tcs.to_pandas().T
    else:
        return tcs


def compute_discrete_tuning_curves(data, epochs_dict, return_pandas=False):
    """
    Compute discrete tuning curves using a dictionary of epochs.

    Parameters
    ----------
    data : TsGroup, TsdFrame, Ts, Tsd
        The data for which the tuning curves will be computed.
    epochs_dict : dict
        Dictionary of IntervalSets.
    return_pandas : bool, optional
        If True, the function returns a pandas.DataFrame instead of an xarray.DataArray.

    Examples
    --------
    This function is typically used for a set of discrete stimuli being presented for multiple epochs.

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
        >>> tcs = nap.compute_discrete_tuning_curves(group, epochs_dict)
        >>> tcs
        <xarray.DataArray (unit: 2, epochs: 2)> Size: 32B
        array([[10.03333333, 10.03333333],
               [ 5.03333333,  5.03333333]])
        Coordinates:
          * unit     (unit) int64 16B 1 2
          * epochs   (epochs) <U5 40B 'stim0' 'stim1'

    You can also pass a TsdFrame (e.g. calcium imaging data), in that case the response is computed:

        >>> frame = nap.TsdFrame(d=np.random.rand(2000, 3), t=np.arange(0, 100, 0.05))
        >>> tcs = nap.compute_discrete_tuning_curves(frame, epochs_dict)
        >>> tcs
        <xarray.DataArray (unit: 3, epochs: 2)> Size: 48B
        array([[0.50946668, 0.50897635],
               [0.48343249, 0.48191892],
               [0.50063158, 0.48748094]])
        Coordinates:
          * unit     (unit) int64 24B 0 1 2
          * epochs   (epochs) <U5 40B 'stim0' 'stim1'

    Returns
    -------
    xarray.DataArray
        A tensor containing the tuning curves with labeled epochs.
    """
    # check data
    if not isinstance(data, (nap.TsdFrame, nap.TsGroup, nap.Ts, nap.Tsd)):
        raise TypeError("data should be a TsdFrame, TsGroup, Ts, or Tsd.")

    # check epochs_dict
    if not isinstance(epochs_dict, dict) or not all(
        isinstance(epoch, nap.IntervalSet) for epoch in epochs_dict.values()
    ):
        raise TypeError("epochs_dict should be a dictionary of IntervalSets.")

    # check return_pandas
    if not isinstance(return_pandas, bool):
        raise TypeError("return_pandas should be a boolean.")

    # tuning curves
    keys = (
        data.keys()
        if isinstance(data, nap.TsGroup)
        else data.columns if isinstance(data, nap.TsdFrame) else [0]
    )
    tcs = np.empty((len(keys), len(epochs_dict)))
    if isinstance(data, (nap.TsGroup, nap.Ts)):
        # SPIKES
        if isinstance(data, nap.Ts):
            data = {0: data}
        for epoch_idx, epoch in enumerate(epochs_dict.values()):
            for unit_idx, unit_label in enumerate(keys):
                tcs[unit_idx, epoch_idx] = float(len(data[unit_label].restrict(epoch)))
            tcs[:, epoch_idx] = tcs[:, epoch_idx] / epoch.tot_length("s")
    else:
        # RATES
        if isinstance(data, nap.Tsd):
            data = np.expand_dims(data.values, -1)
        for epoch_idx, epoch in enumerate(epochs_dict.values()):
            for unit_idx in range(len(keys)):
                tcs[unit_idx, epoch_idx] = np.mean(data[:, unit_idx].restrict(epoch))

    tcs = xr.DataArray(
        tcs,
        coords={"unit": keys, "epochs": list(epochs_dict.keys())},
    )
    if return_pandas:
        return tcs.to_pandas().T
    else:
        return tcs


def compute_mutual_information(tuning_curves):
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

    Parameters
    ----------
    tuning_curves : xarray.DataArray
        As outputted by `compute_tuning_curves`.

    Returns
    -------
    pandas.DataFrame
        A table containing the spatial information per unit, in both bits/sec and bits/spike.

    References
    ----------
    .. [1] Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993).
           An information-theoretic approach to deciphering the hippocampal code.
           In Advances in neural information processing systems (pp. 1030-1037).

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
        >>> MI = nap.compute_mutual_information(tcs)
        >>> MI
            bits/sec  bits/spike
        1  33.480966    3.301870
        2  33.369159    3.310432
    """
    if not isinstance(tuning_curves, xr.DataArray):
        raise TypeError(
            "tuning_curves should be an xr.DataArray as computed by compute_tuning_curves."
        )

    if "occupancy" not in tuning_curves.attrs:
        raise ValueError("No occupancy found in tuning curves.")
    occupancy = tuning_curves.attrs["occupancy"]
    occupancy = occupancy / np.nansum(occupancy)

    fx = tuning_curves.values
    axes = tuple(range(1, fx.ndim))
    fr_keepdims = np.nansum(fx * occupancy, axis=axes, keepdims=True)
    fr_scalar = np.squeeze(fr_keepdims, axis=axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxfr = fx / fr_keepdims
        logfx = np.log2(fxfr)
    logfx[~np.isfinite(logfx)] = 0.0
    MI_bits_per_sec = np.nansum(occupancy * fx * logfx, axis=axes)
    with np.errstate(divide="ignore", invalid="ignore"):
        MI_bits_per_spike = MI_bits_per_sec / fr_scalar

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
    """
    Deprecated, use `compute_mutual_information` instead.
    """
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
    """
    Deprecated, use `compute_mutual_information` instead.
    """
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
