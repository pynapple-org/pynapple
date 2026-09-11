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
    jitvaluefrom_ranges,
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

    new_time = _concat_ranges(time_array, il, ir, copy=True)
    new_data = (
        None if data_array is None else _concat_ranges(data_array, il, ir, copy=True)
    )

    return new_time, new_data


def _concat_ranges(array, range_starts, range_stops, copy):
    """Concatenate the slices ``array[range_starts[k]:range_stops[k]]`` along axis 0.

    The ranges must be sorted and non-overlapping. Each is copied with a plain
    contiguous slice assignment (one memcpy per range), which beats both a
    fancy-index gather and a numba kernel in the few-range regime.

    Parameters
    ----------
    array : ndarray
        Array to take the ranges from.
    range_starts, range_stops : ndarray[int]
        Half-open bounds of each range.
    copy : bool
        When False, ranges that happen to tile a single contiguous span are
        returned as a *view* of ``array`` rather than copied. Callers that hand the
        result to a user-facing object must weigh that aliasing; pass True to
        always copy.

    Returns
    -------
    ndarray
        The concatenated ranges.
    """
    counts = range_stops - range_starts
    total = int(np.sum(counts))

    if (
        not copy
        and total
        and (
            len(range_starts) == 1 or np.array_equal(range_stops[:-1], range_starts[1:])
        )
    ):
        return array[range_starts[0] : range_stops[-1]]

    out = np.empty((total,) + array.shape[1:], dtype=array.dtype)
    pos = 0
    for k in range(len(range_starts)):
        count = counts[k]
        out[pos : pos + count] = array[range_starts[k] : range_stops[k]]
        pos += count
    return out


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
    # replace flag with int
    if mode == "closest":
        mode = 1
    else:
        mode = 0 if mode == "before" else 2

    # Per-epoch slice boundaries, found with a handful of binary searches instead of
    # an O(n) scan over each array. IntervalSet guarantees sorted, disjoint and
    # non-touching epochs, so this reproduces jitrestrict's inclusive
    # ``start <= t <= end`` selection exactly, and nothing has to be gathered:
    # the kernel reads both full arrays through these bounds.
    in_start = np.searchsorted(time_array, starts, side="left")
    in_stop = np.searchsorted(time_array, ends, side="right")
    tg_start = np.searchsorted(time_target_array, starts, side="left")
    tg_stop = np.searchsorted(time_target_array, ends, side="right")

    new_time_array = _concat_ranges(time_array, in_start, in_stop, copy=False)

    # index into the *full* target for each kept timestamp, -1 where unmatched
    gather_idx = jitvaluefrom_ranges(
        time_array, time_target_array, in_start, in_stop, tg_start, tg_stop, mode
    )
    matched = gather_idx >= 0
    all_matched = bool(matched.all())

    # keep the target dtype if it is floating or if every timestamp matched,
    # otherwise upcast to float to hold the NaNs
    use_type = data_target_array.dtype
    if not (all_matched or np.issubdtype(use_type, np.floating)):
        use_type = np.float64

    out_shape = (len(new_time_array), *data_target_array.shape[1:])
    take_idx = gather_idx if all_matched else gather_idx[matched]

    if isinstance(data_target_array, np.ndarray):
        values = None  # gathered straight out of the target below
    else:
        # h5py/zarr datasets (``lazy_loading=True``) only accept fancy indices that
        # are strictly increasing, while `take_idx` is neither sorted nor unique
        # (any target matched by several timestamps repeats). Reading the distinct
        # targets once and expanding satisfies that, and touches strictly less of
        # the dataset than materializing the restricted target would.
        unique_idx, inverse = np.unique(take_idx, return_inverse=True)
        values = (
            data_target_array[unique_idx][inverse]
            if len(unique_idx)
            else np.empty((0,) + data_target_array.shape[1:], dtype=use_type)
        )

    if all_matched:
        new_data_array = np.empty(out_shape, dtype=use_type)
        if values is None:
            np.take(data_target_array, take_idx, axis=0, out=new_data_array)
        else:
            new_data_array[:] = values
    else:
        # `use_type` is necessarily floating here
        new_data_array = np.full(out_shape, np.nan, dtype=use_type)
        new_data_array[matched] = (
            data_target_array[take_idx] if values is None else values
        )

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
