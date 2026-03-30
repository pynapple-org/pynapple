"""
Functions to create trial-based tensors and warp times
"""

import inspect
from functools import wraps
from numbers import Number

import numpy as np

from .. import core as nap


def _validate_warping_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate each positional argument
        sig = inspect.signature(func)
        kwargs = sig.bind_partial(*args, **kwargs).arguments

        parameters_type = {
            "input": (nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup),
            "epochs": (nap.IntervalSet,),
            "bin_size": (Number,),
            "time_unit": (str,),
            "align": (str,),
            "padding_value": (Number,),
            "num_bins": (int,),
        }
        for param, param_type in parameters_type.items():
            if param in kwargs:
                if not isinstance(kwargs[param], param_type):
                    raise TypeError(
                        f"Invalid type. Parameter {param} must be of type {[p.__name__ for p in param_type]}."
                    )

        # Call the original function with validated inputs
        return func(**kwargs)

    return wrapper


@_validate_warping_inputs
def build_tensor(
    input, epochs, bin_size=None, align="start", padding_value=np.nan, time_unit="s"
):
    """
    Return trial-based tensor from an IntervalSet object.
    This is equivalent to calling `input.to_trial_tensor` for Tsd, TsdFrame, and TsdTensor objects,
    or is equivalent to calling `input.trial_count` for Ts and TsGroup objects.

    - If `input` is a `TsGroup`, returns a numpy array of shape (number of group elements, number of trials, number of time bins). The `bin_size` parameter determines the number of time bins.

    - If `input` is `Tsd`, `TsdFrame` or `TsdTensor`, returns a numpy array of shape (shape of time series, number of trials, number of  time points).

    The `align` parameter controls how the time series are aligned. If `align="start"`, the time
    series are aligned to the start of each trial. If `align="end"`, the time series are aligned
    to the end of each trial.

    If trials have uneven durations, the returned array is padded. The parameter `padding_value`
    determine which value is used to pad the array. Default is NaN.

    Parameters
    ----------
    input : Ts, Tsd, TsdFrame, TsdTensor or TsGroup
        Input to slice and align to the trials within the `ep` parameter.
    epochs : IntervalSet
        Epochs holding the trials. Each interval can be of unequal size.
    bin_size : Number, optional
        Size of the time bins for TsGroup and Ts objects.
    align: str, optional
        How to align the time series ('start' [default], 'end')
    padding_value: Number, optional
        How to pad the array if unequal intervals. Default is np.nan.
    time_unit : str, optional
        Time units of the bin_size parameter ('s' [default], 'ms', 'us').

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    RuntimeError
        If `time_unit` not in ["s", "ms", "us"]

    Examples
    --------
    >>> import pynapple as nap
    >>> import numpy as np
    >>> group = nap.TsGroup({0:nap.Ts(t=np.arange(0, 100))})
    >>> epochs = nap.IntervalSet(start=np.arange(20, 100, 20), end=np.arange(20, 100, 20) + np.arange(2, 10, 2))
    >>> print(epochs)
      index    start    end
          0       20     22
          1       40     44
          2       60     66
          3       80     88
    shape: (4, 2), time unit: sec.

    Create a trial-based tensor by counting events within 1 second bin for each interval of `epochs`.

    >>> tensor = nap.build_tensor(group, epochs, bin_size=1)
    >>> tensor
    array([[[ 1.,  1., nan, nan, nan, nan, nan, nan],
            [ 1.,  1.,  1.,  1., nan, nan, nan, nan],
            [ 1.,  1.,  1.,  1.,  1.,  1., nan, nan],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]])

    By default, the time series are aligned to the start of the epochs. The parameter `align` control this behavior.

    >>> tensor = nap.build_tensor(group, epochs, bin_size=1, align="end")
    >>> tensor
    array([[[nan, nan, nan, nan, nan, nan,  1.,  1.],
            [nan, nan, nan, nan,  1.,  1.,  1.,  1.],
            [nan, nan,  1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]])

    This function works for any time series.

    >>> tsdframe = nap.TsdFrame(t=np.arange(100), d=np.arange(200).reshape(2,100).T)
    >>> tensor = nap.build_tensor(tsdframe, epochs)
    >>> tensor
    array([[[ 20.,  21.,  22.,  nan,  nan,  nan,  nan,  nan,  nan],
            [ 40.,  41.,  42.,  43.,  44.,  nan,  nan,  nan,  nan],
            [ 60.,  61.,  62.,  63.,  64.,  65.,  66.,  nan,  nan],
            [ 80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.]],
           [[120., 121., 122.,  nan,  nan,  nan,  nan,  nan,  nan],
            [140., 141., 142., 143., 144.,  nan,  nan,  nan,  nan],
            [160., 161., 162., 163., 164., 165., 166.,  nan,  nan],
            [180., 181., 182., 183., 184., 185., 186., 187., 188.]]])

    """
    if time_unit not in ["s", "ms", "us"]:
        raise RuntimeError("time_unit should be 's', 'ms' or 'us'")
    if align not in ["start", "end"]:
        raise RuntimeError("align should be 'start' or 'end'")

    if isinstance(input, (nap.TsGroup, nap.Ts)):
        if not isinstance(bin_size, Number):
            raise RuntimeError(
                "When input is a TsGroup or Ts object, bin_size should be specified"
            )
        return input.trial_count(epochs, bin_size, align, padding_value, time_unit)
    else:
        return input.to_trial_tensor(epochs, align, padding_value)


def _warp_tensor_from_tsgroup(input, epochs, num_bins):
    if isinstance(input, nap.Ts):
        output = np.zeros(shape=(1, len(epochs), num_bins))
    else:
        output = np.zeros(shape=(len(input), len(epochs), num_bins))

    bin_sizes = (epochs.end - epochs.start) / num_bins

    for i in range(len(epochs)):
        tmp = input.count(bin_sizes[i], epochs[i])
        output[:, i, :] = np.transpose(tmp.values)

    if isinstance(input, nap.Ts):  # Removing first axis if Ts.
        output = output[0]

    return output


def _warp_tensor_from_tsd(input, epochs, num_bins):
    slices = [input.get_slice(s, e) for s, e in epochs.values]
    lengths = list(map(lambda sl: sl.stop - sl.start, slices))
    output = np.zeros(shape=(len(epochs), num_bins, *input.shape[1:]))
    for i, sl in enumerate(slices):
        if lengths[i] == 0:
            output[i] = np.full(output.shape[1:], np.nan)
        elif lengths[i] == num_bins:
            output[i] = input[sl].values
        elif lengths[i] > num_bins:  # Call bin_average
            output[i] = input[sl].bin_average(
                (epochs.end[i] - epochs.start[i]) / num_bins, epochs[i]
            )
        else:  # Call interpolate
            output[i] = input[sl].interpolate(
                ts=nap.Ts(t=np.linspace(epochs.start[i], epochs.end[i], num_bins)),
                ep=epochs[i],
            )

    if output.ndim > 2:
        output = np.moveaxis(output, source=[0, 1], destination=[-2, -1])

    return output


@_validate_warping_inputs
def warp_tensor(input, epochs, num_bins):
    """
    Return linearly time-warped trial-based tensor from an IntervalSet object.

    - If `input` is a `TsGroup`, returns a numpy array of shape (number of group element, number of trial, `num_bins`).

    - If `input` is `Tsd`, `TsdFrame` or `TsdTensor`, returns a numpy array of shape (shape of time series, number of trial, `num_bins`).

    Parameters
    ----------
    input : Ts , Tsd, TsdFrame, TsdTensor or TsGroup
        Input object
    epochs : IntervalSet
        Epochs holding the trials. Each interval can be of unequal size.
    num_bins : int

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import pynapple as nap
    >>> import numpy as np
    >>> group = nap.TsGroup({0:nap.Ts(t=np.arange(0, 100))})
    >>> epochs = nap.IntervalSet(start=np.arange(20, 100, 20), end=np.arange(20, 100, 20) + np.arange(2, 10, 2))
    >>> print(epochs)
      index    start    end
          0       20     22
          1       40     44
          2       60     66
          3       80     88
    shape: (4, 2), time unit: sec.

    Create a trial-based tensor by counting events within 10 bins between start and end of each interval of `epochs`.

    >>> tensor = nap.warp_tensor(group, epochs, num_bins=10)
    >>> tensor
    array([[[1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [1., 0., 1., 0., 0., 1., 0., 1., 0., 0.],
            [1., 1., 0., 1., 0., 1., 1., 0., 1., 0.],
            [1., 1., 1., 1., 0., 1., 1., 1., 1., 0.]]])

    This function works for any time series. Under the hood, the time series is either bin-averaged or interpolated depending on the number of bins.

    >>> tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    >>> tensor = nap.warp_tensor(tsd, epochs, num_bins=3)
    >>> tensor
    array([[20. , 21. , 22. ],
           [40.5, 42. , 43. ],
           [60.5, 62.5, 64.5],
           [81. , 84. , 87. ]])
    """
    if num_bins <= 0:
        raise RuntimeError("num_bins should be positive integer.")

    if isinstance(input, (nap.TsGroup, nap.Ts)):
        return _warp_tensor_from_tsgroup(input, epochs, num_bins)
    else:
        return _warp_tensor_from_tsd(input, epochs, num_bins)
