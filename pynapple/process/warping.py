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

    for i in range(len(ep)):
        tmp = count.get(ep.start[i], ep.end[i]).values  # Time by neuron
        output[:, i, 0 : tmp.shape[0]] = np.transpose(tmp)

    return output


def _build_tensor_from_tsd(input, ep, binsize, align, padding_value):
    pass


@_validate_warping_inputs
def build_tensor(
    input, ep, binsize=None, align="start", padding_value=np.nan, time_unit="s"
):
    """
    Return trial-based tensor from an IntervalSet object.

    - if `input` is a `TsGroup`, returns a numpy array of shape (number of trial, number of group element, number of time bins).
        The `binsize` parameter determines the number of time bins.

    - if `input` is `Tsd`, `TsdFrame` or `TsdTensor`, returns a numpy array of shape
        (number of trial, shape of time series, number of  time points).
        If the parameter `binsize` is used, the data are "bin-averaged".


    Parameters
    ----------
    input : Tsd, TsdFrame, TsdTensor or TsGroup
        Returns a numpy array.
    ep : IntervalSet
        Epochs holding the trials. Each interval can be of unequal size.
    binsize : Number, optional
    align: str, optional
        How to align the time series ('start' [default], 'end', 'both')
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
    if align not in ["start", "end", "both"]:
        raise RuntimeError("align should be 'start', 'end' or 'both'")

    binsize = np.abs(nap.TsIndex.format_timestamps(np.array([binsize]), time_unit))[0]

    if isinstance(input, nap.TsGroup):
        return _build_tensor_from_tsgroup(input, ep, binsize, align, padding_value)

    if isinstance(input, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)):
        return _build_tensor_from_tsd(input, ep, binsize, align, padding_value)
