"""
    This module holds the core function of pynapple as well as 
    the dispatch between numba and jax.

    If pynajax is installed and `nap.nap_config.backend` is set 
    to `jax`, the module will call the functions within pynajax.
    Otherwise the module will call the functions within `_jitted_functions.py`.

"""

import numpy as np

from ._jitted_functions import (
    jitbin,
    jitbin_array,
    jitremove_nan,
    jitrestrict,
    jitthreshold,
    jittsrestrict,
    pjitconvolve,
)
from .utils import get_backend
from scipy import signal

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


def _restrict(time_array, data_array, starts, ends):
    if get_backend() == "jax":
        from pynajax.jax_core_restrict import restrict

        return restrict(time_array, data_array, starts, ends)
    else:
        if data_array is not None:
            return jitrestrict(time_array, data_array, starts, ends)
        else:
            return jittsrestrict(time_array, starts, ends)


def _value_from():
    pass


def _count():
    pass


def _bin_average():
    pass


def _interpolate():
    pass


def _threshold():
    pass