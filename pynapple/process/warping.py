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
            "ep": (nap.IntervalSet,),
            "binsize": (Number,),
            "time_unit": (str,),
            "align": (str,),
            "padding_value": (Number,),
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


def _build_tensor_from_tsgroup(input, ep, binsize, align, padding_value):
    # Determine size of tensor
    n_t = int(np.max(np.ceil((ep.end + binsize - ep.start) / binsize)))
    output = np.ones(shape=(len(input), len(ep), n_t)) * padding_value
    count = input.count(bin_size=binsize, ep=ep)

    if align == "start":
        for i in range(len(ep)):
            tmp = count.get(ep.start[i], ep.end[i]).values
            output[:, i, 0 : tmp.shape[0]] = np.transpose(tmp)
        if np.all(np.isnan(output[:, :, -1])):
            output = output[:, :, 0:-1]

    if align == "end":
        for i in range(len(ep)):
            tmp = count.get(ep.start[i], ep.end[i]).values
            output[:, i, -tmp.shape[0] :] = np.transpose(tmp)
        if np.all(np.isnan(output[:, :, 0])):
            output = output[:, :, 1:]

    return output


def _build_tensor_from_tsd(input, ep, align, padding_value):
    slices = [input.get_slice(s, e) for s, e in ep.values]
    lengths = list(map(lambda sl: sl.stop - sl.start, slices))
    n_t = max(lengths)
    output = np.ones(shape=(len(ep), n_t, *input.shape[1:])) * padding_value
    if align == "start":
        for i, sl in enumerate(slices):
            output[i, 0 : lengths[i]] = input[sl].values
    if align == "end":
        for i, sl in enumerate(slices):
            output[i, -lengths[i] :] = input[sl].values

    if output.ndim > 2:
        output = np.moveaxis(output, source=[0, 1], destination=[-2, -1])

    return output


@_validate_warping_inputs
def build_tensor(
    input, ep, binsize=None, align="start", padding_value=np.nan, time_unit="s"
):
    """
    Return trial-based tensor from an IntervalSet object.

    - If `input` is a `TsGroup`, returns a numpy array of shape (number of group element, number of trial, number of time bins). The `binsize` parameter determines the number of time bins.

    - If `input` is `Tsd`, `TsdFrame` or `TsdTensor`, returns a numpy array of shape (shape of time series, number of trial, number of  time points).

    The `align` parameter controls how the time series are aligned. If `align="start"`, the time
    series are aligned to the start of the trials. If `align="end"`, the time series are aligned
    to the end of the trials.

    If trials are uneven durations, the returned array is padded. The parameter `padding_value`
    determine which value is used to pad the array. Default is NaN.

    Parameters
    ----------
    input : Tsd, TsdFrame, TsdTensor or TsGroup
        Input to slice and align to the trials within the `ep` parameter.
    ep : IntervalSet
        Epochs holding the trials. Each interval can be of unequal size.
    binsize : Number, optional
    align: str, optional
        How to align the time series ('start' [default], 'end')
    padding_value: Number, optional
        How to pad the array if unequal intervals. Default is np.nan.
    time_unit : str, optional
        Time units of the binsize parameter ('s' [default], 'ms', 'us').

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    RuntimeError
        If `time_unit` not in ["s", "ms", "us"]


    Examples
    --------



    """
    if time_unit not in ["s", "ms", "us"]:
        raise RuntimeError("time_unit should be 's', 'ms' or 'us'")
    if align not in ["start", "end"]:
        raise RuntimeError("align should be 'start' or 'end'")

    if isinstance(input, nap.TsGroup):
        if not isinstance(binsize, Number):
            raise RuntimeError("When input is a TsGroup, binsize should be specified")
        return _build_tensor_from_tsgroup(input, ep, binsize, align, padding_value)

    if isinstance(input, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        return _build_tensor_from_tsd(input, ep, align, padding_value)


@_validate_warping_inputs
def warp_tensor(input, ep, num_bin=None, align="start"):
    """
    Return time-warped trial-based tensor from an IntervalSet object.

    - If `input` is a `TsGroup`, returns a numpy array of shape (number of group element, number of trial, number of time bins). The `binsize` parameter determines the number of time bins.

    - If `input` is `Tsd`, `TsdFrame` or `TsdTensor`, returns a numpy array of shape (shape of time series, number of trial, number of  time points).


    Parameters
    ----------
    input : Tsd, TsdFrame, TsdTensor or TsGroup
        Returns a numpy array.
    ep : IntervalSet
        Epochs holding the trials. Each interval can be of unequal size.
    binsize : Number, optional
    align: str, optional
        How to align the time series ('start' [default], 'end')
    padding_value: Number, optional
        How to pad the array if unequal intervals. Default is np.nan.
    time_unit : str, optional
        Time units of the binsize parameter ('s' [default], 'ms', 'us').

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    RuntimeError
        If `time_unit` not in ["s", "ms", "us"]


    Examples
    --------



    """
    pass
