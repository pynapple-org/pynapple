"""
This module holds the core function of pynapple as well as
the dispatch between numba and jax.

If pynajax is installed and `nap.nap_config.backend` is set
to `jax`, the module will call the functions within pynajax.
Otherwise the module will call the functions within `_jitted_functions.py`.

"""

from typing import Literal

import numpy as np

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


def _use_searchsorted_restrict(n_intervals, n_samples):
    """Whether to restrict via searchsorted boundaries rather than a merge scan.

    ``searchsorted`` does ``n_intervals`` binary searches, each ~log2(n_samples)
    cache-missing memory jumps, then copies the selected ranges with numpy. Past
    a crossover (empirically ~n_samples/1000) a single sequential merge scan
    (:func:`jitrestrict`, O(n)) becomes faster. Realistic interval counts are far
    below that, so this stays on the searchsorted side up to ~n_samples/1024 and
    hands off to the scan beyond it, so the many-interval path never regresses.
    """
    return n_intervals * 1024 < n_samples


def _restrict_ranges(time_array, data_array, starts, ends):
    """Restrict to intervals via searchsorted boundaries + contiguous copies.

    Returns copied ``(new_time, new_data)``; ``new_data`` is None when
    ``data_array`` is None (timestamps-only objects). Assumes ``time_array`` is
    sorted and ``starts``/``ends`` are sorted and disjoint (guaranteed by
    IntervalSet), so the result is sorted and lies within the intervals. The
    inclusivity ``start <= t <= end`` matches :func:`jitrestrict`.

    The selected ranges are copied with plain numpy contiguous slice assignment
    (one memcpy per interval), which beats both a fancy-index gather and a numba
    kernel for the realistic (few-interval) regime this path handles.
    """
    il = np.searchsorted(time_array, starts, side="left")
    ir = np.searchsorted(time_array, ends, side="right")
    total = int(np.sum(ir - il))

    new_time = np.empty(total, dtype=time_array.dtype)
    new_data = (
        None
        if data_array is None
        else np.empty((total,) + data_array.shape[1:], dtype=data_array.dtype)
    )

    pos = 0
    for k in range(len(il)):
        count = ir[k] - il[k]
        new_time[pos : pos + count] = time_array[il[k] : ir[k]]
        if new_data is not None:
            new_data[pos : pos + count] = data_array[il[k] : ir[k]]
        pos += count

    return new_time, new_data


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

    new_time_array = time_array[idx_t]

    idx = jitvaluefrom(
        new_time_array,
        time_target_array[idx_target],
        count,
        count_target,
        starts,
        mode=mode,
    )

    # `idx` indexes the *restricted* target and uses NaN for unmatched timestamps.
    nan_mask = np.isnan(idx)
    has_nan = bool(nan_mask.any())

    # Composing `idx_target[idx]` gathers the values once, straight out of the full
    # target, instead of materializing the whole restricted target only to index
    # into it again. The composed indices are neither sorted nor unique, which
    # h5py/zarr datasets (``lazy_loading=True``) reject, so those keep the two-step
    # gather -- there the first, monotonic gather is what loads the data.
    can_compose = isinstance(data_target_array, np.ndarray)

    # keep the target dtype if it is floating or if every timestamp matched,
    # otherwise upcast to float to hold the NaNs
    use_type = data_target_array.dtype
    if has_nan and not np.issubdtype(use_type, np.floating):
        use_type = np.float64

    out_shape = (len(new_time_array), *data_target_array.shape[1:])

    if has_nan:
        # `use_type` is necessarily floating here
        new_data_array = np.full(out_shape, np.nan, dtype=use_type)
        valid = ~nan_mask
        local = idx[valid].astype(np.int64)
        if can_compose:
            new_data_array[valid] = data_target_array[idx_target[local]]
        else:
            new_data_array[valid] = data_target_array[idx_target][local]
    else:
        local = idx.astype(np.int64)
        if can_compose:
            new_data_array = np.empty(out_shape, dtype=use_type)
            np.take(data_target_array, idx_target[local], axis=0, out=new_data_array)
        else:
            new_data_array = data_target_array[idx_target][local]

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
    from scipy import signal

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
