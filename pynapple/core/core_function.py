"""
    This module holds the core function of pynapple as well as 
    the dispatch between numba and jax.

    If pynajax is installed and `nap.nap_config.backend` is set 
    to `jax`, the module will call the functions within pynajax.
    Otherwise the module will call the functions within `_jitted_functions.py`.

"""

import numpy as np
from scipy import signal

from ._jitted_functions import (    
    jitbin_array,
    jitcount,
    jitremove_nan,
    jitrestrict,
    jitthreshold,
    jittsrestrict_with_count,
    jitvaluefrom,
    pjitconvolve,
)
from .utils import get_backend


def _restrict(time_array, starts, ends):
    return jitrestrict(time_array, starts, ends)


def _count(time_array, starts, ends, bin_size=None):
    if isinstance(bin_size, (float, int)):
        return jitcount(time_array, starts, ends, bin_size)
    else:
        _, d = jittsrestrict_with_count(time_array, starts, ends)
        t = starts + (ends - starts) / 2
        return t, d


def _value_from(time_array, time_target_array, data_target_array, starts, ends):
    t, idx = jitvaluefrom(time_array, time_target_array, starts, ends)
    return t, data_target_array[idx]


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
        if update_time_support:
            starts, ends = jitremove_nan(time_array, index_nan)

            to_fix = starts == ends
            if np.any(to_fix):
                ends[to_fix] += 1e-6  # adding 1 millisecond in case of a single point

            return (time_array[~index_nan], data_array[~index_nan], starts, ends)
        else:
            return (time_array[~index_nan], data_array[~index_nan], starts, ends)
    else:
        return (time_array, data_array, starts, ends)


####################################
# Can call pynajax
####################################


def _convolve(time_array, data_array, starts, ends, array, trim="both"):
    if get_backend() == "jax":
        from pynajax.jax_core_convolve import convolve

        return convolve(time_array, data_array, starts, ends, array)
    else:
        if data_array.ndim == 1:
            new_data_array = np.zeros(data_array.shape)
            k = array.shape[0]
            for s, e in zip(starts, ends):
                idx_s = np.searchsorted(time_array, s)
                idx_e = np.searchsorted(time_array, e, side="right")

                t = idx_e - idx_s
                if trim == "left":
                    cut = (k - 1, t + k - 1)
                elif trim == "right":
                    cut = (0, t)
                else:
                    cut = ((1 - k % 2) + (k - 1) // 2, t + k - 1 - ((k - 1) // 2))
                # scipy is actually faster for Tsd
                new_data_array[idx_s:idx_e] = signal.convolve(
                    data_array[idx_s:idx_e], array
                )[cut[0] : cut[1]]

            return new_data_array
        else:
            new_data_array = np.zeros(data_array.shape)
            for s, e in zip(starts, ends):
                idx_s = np.searchsorted(time_array, s)
                idx_e = np.searchsorted(time_array, e, side="right")
                new_data_array[idx_s:idx_e] = pjitconvolve(
                    data_array[idx_s:idx_e], array, trim=trim
                )

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
        return threshold(time_array, data_array, starts, ends, thr, method)
    else:
        return jitthreshold(time_array, data_array, starts, ends, thr, method)

# def perievent

# def _sta():


# def _interp
