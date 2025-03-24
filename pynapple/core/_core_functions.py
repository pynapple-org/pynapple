"""
This module holds the core function of pynapple as well as
the dispatch between numba and jax.

If pynajax is installed and `nap.nap_config.backend` is set
to `jax`, the module will call the functions within pynajax.
Otherwise the module will call the functions within `_jitted_functions.py`.

"""

from typing import Literal

import numpy as np
from scipy import signal

from ._jitted_functions import (  # pjitconvolve,
    jitbin_array,
    jitcount,
    jitremove_nan,
    jitrestrict,
    jitrestrict_with_count,
    jitthreshold,
    jitvaluefrom,
)
from .utils import get_backend


def _restrict(time_array, starts, ends):
    return jitrestrict(time_array, starts, ends)


def _count(time_array, starts, ends, bin_size=None, dtype=None):
    if isinstance(bin_size, (float, int)):
        t, d = jitcount(time_array, starts, ends, bin_size, dtype)
    else:
        _, d = jitrestrict_with_count(time_array, starts, ends, dtype)
        t = starts + (ends - starts) / 2
    return t, d


def _value_from(
    time_array,
    time_target_array,
    data_target_array,
    starts,
    ends,
    mode: Literal["closest", "before", "after"] = "closest",
):
    idx_t, count = jitrestrict_with_count(time_array, starts, ends)
    idx_target, count_target = jitrestrict_with_count(time_target_array, starts, ends)
    # replace flag with int
    if mode == "closest":
        mode = 1
    else:
        mode = 0 if mode == "before" else 2

    idx = jitvaluefrom(
        time_array[idx_t],
        time_target_array[idx_target],
        count,
        count_target,
        starts,
        mode=mode,
    )

    new_time_array = time_array[idx_t]
    nan_idx = np.isnan(idx)

    # set the type as default
    use_type = data_target_array.dtype

    # if is already floating or all values are valid, keep type, otherwise use float
    use_type = (
        use_type if np.issubdtype(use_type, np.floating) or not any(nan_idx) else float
    )
    if not np.issubdtype(use_type, np.floating):
        new_data_array = np.zeros(
            (len(new_time_array), *data_target_array.shape[1:]),
            dtype=use_type,
        )
    else:
        new_data_array = np.full(
            (len(new_time_array), *data_target_array.shape[1:]),
            np.nan,
            dtype=use_type,
        )

    idx2 = ~np.isnan(idx)
    new_data_array[idx2] = data_target_array[idx_target][idx[idx2].astype(int)]

    return new_time_array, new_data_array


def _dropna(time_array, data_array, starts, ends, update_time_support, ndim):
    index_nan = np.asarray(np.any(np.isnan(data_array), axis=tuple(range(1, ndim))))
    if np.all(index_nan):  # In case it's only NaNs
        if update_time_support:
            starts = None
            ends = None
        return (
            np.array([]),
            np.empty(tuple([0] + [d for d in data_array.shape[1:]])),
            starts,
            ends,
        )
    elif np.any(index_nan):
        tokeep = np.where(~index_nan)[0]
        if update_time_support:
            starts, ends = jitremove_nan(time_array, index_nan)

            to_fix = starts == ends
            if np.any(to_fix):
                ends[to_fix] += 1e-6  # adding 1 millisecond in case of a single point
            return (time_array[tokeep], data_array[tokeep], starts, ends)
        else:
            return (time_array[tokeep], data_array[tokeep], starts, ends)
    else:
        return (time_array, data_array, starts, ends)


####################################
# Can call pynajax
####################################


def _convolve(time_array, data_array, starts, ends, array, trim="both"):
    if get_backend() == "jax":
        from pynajax.jax_core_convolve import convolve

        return convolve(time_array, data_array, starts, ends, array, trim)
    else:
        # reshape to 2d
        shape = data_array.shape
        data_array = np.reshape(data_array, (shape[0], -1))

        kshape = array.shape
        k = kshape[0]
        array = array.reshape(k, -1)

        new_data_array = np.zeros((shape[0], int(np.prod(shape[1:])), *array.shape[1:]))

        for s, e in zip(starts, ends):
            idx_s = np.searchsorted(time_array, s)
            idx_e = np.searchsorted(time_array, e, side="right")

            t = idx_e - idx_s
            if trim == "left":
                cut = (k - 1, t + k - 1)
            elif trim == "right":
                cut = (0, t)
            else:
                cut = ((k - 1) // 2, t + k - 1 - ((k - 1) // 2) - (1 - k % 2))

            for i in range(data_array.shape[1]):
                for j in range(array.shape[1]):
                    new_data_array[idx_s:idx_e, i, j] = signal.convolve(
                        data_array[idx_s:idx_e, i], array[:, j]
                    )[cut[0] : cut[1]]

        new_data_array = new_data_array.reshape((*shape, *kshape[1:]))

        return new_data_array


def _bin_average(time_array, data_array, starts, ends, bin_size):
    if get_backend() == "jax":
        from pynajax.jax_core_bin_average import bin_average

        return bin_average(time_array, data_array, starts, ends, bin_size)
    else:
        return jitbin_array(time_array, data_array, starts, ends, bin_size)


def _threshold(time_array, data_array, starts, ends, thr, method):
    if get_backend() == "jax":
        from pynajax.jax_core_threshold import threshold

        return threshold(time_array, data_array[:], starts, ends, thr, method)
    else:
        return jitthreshold(time_array, data_array[:], starts, ends, thr, method)
