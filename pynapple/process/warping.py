"""
Functions to create trial-based tensors and warp times
"""

from __future__ import annotations

import inspect
from functools import wraps
from numbers import Number
from typing import Optional, Union

import numpy as np
from numba import jit, prange

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
    input: Union[nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup],
    epochs: nap.IntervalSet,
    bin_size: Optional[Number] = None,
    align: str = "start",
    padding_value: Number = np.nan,
    time_unit: str = "s",
) -> np.ndarray:
    """
    Make a tensor with one row for each trial.

    Each epoch of ``epochs`` is one trial. The function puts the data of each
    trial in one row of the tensor:

    - For a Ts or a TsGroup, it counts the timestamps in bins of ``bin_size``.
      This is the same as ``input.trial_count``.
    - For a Tsd, a TsdFrame or a TsdTensor, it takes the values in the trial.
      This is the same as ``input.to_trial_tensor``.

    The trials can have different lengths. The row of a short trial then
    gets ``padding_value`` in the bins or the time points that it does not
    have. ``align`` sets the side of the row that holds the data.

    Parameters
    ----------
    input : Ts, Tsd, TsdFrame, TsdTensor or TsGroup
        The data to put in the tensor.
    epochs : IntervalSet
        The trials. The trials can have different lengths.
    bin_size : Number, optional
        The size of the bins. Necessary for a Ts or a TsGroup, not used for
        the other types.
    align : str, optional
        - ``'start'`` (default): the data of each trial starts at the first
          bin or time point of its row.
        - ``'end'``: the data of each trial ends at the last bin or time point
          of its row.
    padding_value : Number, optional
        The value of the bins or time points that a trial does not have. The
        default is NaN.
    time_unit : str, optional
        The unit of ``bin_size``: ``'s'`` (default), ``'ms'`` or ``'us'``.

    Returns
    -------
    numpy.ndarray
        The tensor. The trials are on the second-last axis, and the bins or
        time points are on the last axis:

        - Ts: ``(n_trials, n_bins)``.
        - TsGroup: ``(n_units, n_trials, n_bins)``.
        - Tsd: ``(n_trials, n_time_points)``.
        - TsdFrame or TsdTensor: ``(*input.shape[1:], n_trials,
          n_time_points)``.

    Raises
    ------
    TypeError
        If an argument does not have the correct type.
    RuntimeError
        - If ``time_unit`` is not ``'s'``, ``'ms'`` or ``'us'``.
        - If ``align`` is not ``'start'`` or ``'end'``.
        - If ``input`` is a Ts or a TsGroup and ``bin_size`` is not given.

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

    Count the timestamps in bins of 1 second, for each trial of ``epochs``:

    >>> tensor = nap.build_tensor(group, epochs, bin_size=1)
    >>> tensor
    array([[[ 1.,  1., nan, nan, nan, nan, nan, nan],
            [ 1.,  1.,  1.,  1., nan, nan, nan, nan],
            [ 1.,  1.,  1.,  1.,  1.,  1., nan, nan],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]])

    By default, the data of each trial starts at the first bin of its row.
    With ``align="end"``, the data ends at the last bin of its row:

    >>> tensor = nap.build_tensor(group, epochs, bin_size=1, align="end")
    >>> tensor
    array([[[nan, nan, nan, nan, nan, nan,  1.,  1.],
            [nan, nan, nan, nan,  1.,  1.,  1.,  1.],
            [nan, nan,  1.,  1.,  1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]])

    For a TsdFrame, the function takes the values in each trial:

    >>> tsdframe = nap.TsdFrame(t=np.arange(100), d=np.arange(200).reshape(2,100).T)
    >>> tensor = nap.build_tensor(tsdframe, epochs)
    >>> tensor
    array([[[ 20.,  21.,  22.,  nan,  nan,  nan,  nan,  nan,  nan],
            [ 40.,  41.,  42.,  43.,  44.,  nan,  nan,  nan,  nan],
            [ 60.,  61.,  62.,  63.,  64.,  65.,  66.,  nan,  nan],
            [ 80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.]],
    <BLANKLINE>
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


@jit(nopython=True, cache=True, parallel=True)
def _jitwarp_count(times, unit_positions, starts, ends, num_bins, n_units):
    """
    Count the timestamps of each unit in ``num_bins`` bins of each trial.

    The bins of a trial have the size ``(end - start) / num_bins``. They follow
    the rules of ``jitcount``, so that the counts are the same as
    ``count(bin_size, trial)`` for each trial:

    - The bin edges are rounded to 9 decimals.
    - A bin includes its left edge and excludes its right edge.
    - A bin is kept only if its center is at or before the trial end.
    - A timestamp after the trial end is not counted.

    Parameters
    ----------
    times : ndarray
        The timestamps, sorted.
    unit_positions : ndarray
        The unit of each timestamp, from 0 to ``n_units - 1``.
    starts, ends : ndarray
        The start and the end of each trial.
    num_bins : int
        The number of bins of each trial.
    n_units : int
        The number of units.

    Returns
    -------
    ndarray
        The counts, with the shape ``(n_units, n_trials, num_bins)``.
    """
    n_times = len(times)
    n_trials = len(starts)
    output = np.zeros((n_units, n_trials, num_bins))
    # first timestamp of each trial
    first = np.searchsorted(times, starts, side="left")

    # each trial writes only output[:, trial, :], so the trials run in parallel
    for trial in prange(n_trials):
        start = starts[trial]
        end = ends[trial]
        bin_size = (end - start) / num_bins
        # number of bins that jitcount allocates for the trial
        if end - start > bin_size:
            max_bins = int(np.ceil((end + bin_size - start) / bin_size))
        else:
            max_bins = 1
        max_bins = min(max_bins, num_bins)

        t = first[trial]
        left_edge = start
        for b in range(max_bins):
            if left_edge + bin_size / 2 > end:
                break
            right_edge = np.round(left_edge + bin_size, 9)
            while t < n_times and times[t] < right_edge and times[t] <= end:
                output[unit_positions[t], trial, b] += 1
                t += 1
            left_edge = np.round(left_edge + bin_size, 9)

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
def warp_tensor(
    input: Union[nap.Ts, nap.Tsd, nap.TsdFrame, nap.TsdTensor, nap.TsGroup],
    epochs: nap.IntervalSet,
    num_bins: int,
) -> np.ndarray:
    """
    Make a tensor with one row of ``num_bins`` bins for each trial.

    Each epoch of ``epochs`` is one trial. The function divides each trial
    into ``num_bins`` bins of equal size. The bins of a long trial are
    larger than the bins of a short trial. Thus all the trials get the same
    number of bins, and the time in each trial is linearly warped.

    - For a Ts or a TsGroup, the function counts the timestamps in each
      bin. A bin includes its left edge and excludes its right edge. A
      timestamp at the end of a trial is not counted.
    - For a Tsd, a TsdFrame or a TsdTensor, the function changes the number
      of time points of each trial to ``num_bins``:

      * If the trial has more time points, it averages the values in each
        bin (``bin_average``).
      * If the trial has fewer time points, it interpolates the values at
        ``num_bins`` times from the start to the end of the trial
        (``interpolate``).
      * If the trial has no time point, its row is NaN.

    Parameters
    ----------
    input : Ts, Tsd, TsdFrame, TsdTensor or TsGroup
        The data to put in the tensor.
    epochs : IntervalSet
        The trials. The trials can have different lengths.
    num_bins : int
        The number of bins of each trial.

    Returns
    -------
    numpy.ndarray
        The tensor. The trials are on the second-last axis, and the bins are
        on the last axis:

        - Ts or Tsd: ``(n_trials, num_bins)``.
        - TsGroup: ``(n_units, n_trials, num_bins)``.
        - TsdFrame or TsdTensor: ``(*input.shape[1:], n_trials, num_bins)``.

    Raises
    ------
    TypeError
        If an argument does not have the correct type.
    RuntimeError
        If ``num_bins`` is not positive.

    Notes
    -----
    For a lazy TsGroup (e.g. the units of an NWB file), the function reads
    only the timestamps in the trials.

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

    Count the timestamps in 10 bins for each trial of ``epochs``:

    >>> tensor = nap.warp_tensor(group, epochs, num_bins=10)
    >>> tensor
    array([[[1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [1., 0., 1., 0., 0., 1., 0., 1., 0., 0.],
            [1., 1., 0., 1., 0., 1., 1., 0., 1., 0.],
            [1., 1., 1., 1., 0., 1., 1., 1., 1., 0.]]])

    For a Tsd, the function averages or interpolates the values of each trial
    to get 3 bins:

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

    if isinstance(input, nap.TsGroup):
        # a lazy group reads only the spikes in the trials
        input = input._load_in_memory(epochs)
        return _jitwarp_count(
            input._times,
            input._cluster_positions,
            epochs.start,
            epochs.end,
            num_bins,
            len(input.index),
        )
    if isinstance(input, nap.Ts):
        # a Ts counts as a group of one unit, without the unit axis
        times = input.t
        return _jitwarp_count(
            times,
            np.zeros(len(times), dtype=np.int64),
            epochs.start,
            epochs.end,
            num_bins,
            1,
        )[0]
    return _warp_tensor_from_tsd(input, epochs, num_bins)
