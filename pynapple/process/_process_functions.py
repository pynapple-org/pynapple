"""
This module holds some process function of pynapple that can be
called with numba or pynajax as backend

If pynajax is installed and `nap.nap_config.backend` is set
to `jax`, the module will call the functions within pynajax.
Otherwise the module will call the functions within `_jitted_functions.py`.

"""

import numpy as np
from numba import jit

from .. import core as nap


@jit(nopython=True, cache=True)
def _jitcontinuous_perievent(time_array, time_target_array, starts, ends, windowsize):
    N_epochs = len(starts)
    count = np.zeros((N_epochs, 2), dtype=np.int64)

    idx, count[:, 1] = nap._jitted_functions.jitrestrict_with_count(
        time_target_array, starts, ends
    )
    time_target_array = time_target_array[idx]

    idx, count[:, 0] = nap._jitted_functions.jitrestrict_with_count(
        time_array, starts, ends
    )
    time_array = time_array[idx]

    N_target = len(time_target_array)

    slice_idx = np.zeros((N_target, 2), dtype=np.int64)
    start_w = np.zeros(N_target, dtype=np.int64)

    if np.any((count[:, 0] * count[:, 1]) > 0):
        for k in range(N_epochs):
            if count[k, 0] > 0 and count[k, 1] > 0:
                t = np.sum(count[0:k, 0])
                i = np.sum(count[0:k, 1])
                maxt = t + count[k, 0]
                maxi = i + count[k, 1]

                start_t = t

                while i < maxi:
                    interval = abs(time_array[t] - time_target_array[i])
                    t_pos = t
                    t += 1
                    while t < maxt:
                        new_interval = abs(time_array[t] - time_target_array[i])
                        if new_interval > interval:
                            break
                        else:
                            interval = new_interval
                            t_pos = t
                            t += 1

                    left = np.minimum(windowsize[0], t_pos - start_t)
                    right = np.minimum(windowsize[1], maxt - t_pos - 1)
                    # center = windowsize[0] + 1

                    slice_idx[i] = (t_pos - left, t_pos + right + 1)
                    start_w[i] = windowsize[0] - left

                    t -= 1
                    i += 1

    return idx, slice_idx, np.sum(count[:, 1]), start_w


@jit(nopython=True, cache=True)
def _jitperievent_trigger_average(
    time_array,
    count_array,
    time_target_array,
    data_target_array,
    starts,
    ends,
    windows,
    binsize,
):
    T = time_array.shape[0]
    N = count_array.shape[1]
    N_epochs = len(starts)

    idx, count = nap._jitted_functions.jitrestrict_with_count(
        time_target_array, starts, ends
    )
    time_target_array = time_target_array[idx]
    data_target_array = data_target_array[idx]

    max_count = np.cumsum(count)

    new_data_array = np.full(
        (int(windows.sum()) + 1, count_array.shape[1], *data_target_array.shape[1:]),
        0.0,
    )

    t = 0  # count events

    hankel_array = np.zeros((new_data_array.shape[0], *data_target_array.shape[1:]))

    for k in range(N_epochs):
        if count[k] > 0:
            t_start = t
            maxi = max_count[k]
            i = maxi - count[k]

            while t < T:
                lbound = time_array[t]
                rbound = np.round(lbound + binsize, 9)

                if time_target_array[i] < rbound:
                    i_start = i
                    i_stop = i

                    while i_stop < maxi:
                        if time_target_array[i_stop] < rbound:
                            i_stop += 1
                        else:
                            break

                    while i_start < i_stop - 1:
                        if time_target_array[i_start] < lbound:
                            i_start += 1
                        else:
                            break
                    v = np.sum(data_target_array[i_start:i_stop], 0) / float(
                        i_stop - i_start
                    )

                    checknan = np.sum(v)
                    if not np.isnan(checknan):
                        hankel_array[-1] = v

                if t - t_start >= windows[1]:
                    for n in range(N):
                        new_data_array[:, n] += (
                            hankel_array * count_array[t - windows[1], n]
                        )

                # hankel_array = np.roll(hankel_array, -1, axis=0)
                hankel_array[0:-1] = hankel_array[1:]
                hankel_array[-1] = 0.0

                t += 1

                i = i_start

                if t == T or time_array[t] > ends[k]:
                    if t - t_start > windows[1]:
                        for j in range(windows[1]):
                            for n in range(N):
                                new_data_array[:, n] += (
                                    hankel_array * count_array[t - windows[1] + j, n]
                                )

                            # hankel_array = np.roll(hankel_array, -1, axis=0)
                            hankel_array[0:-1] = hankel_array[1:]
                            hankel_array[-1] = 0.0

                    hankel_array *= 0.0
                    break

    total = np.sum(count_array, 0)
    for n in range(N):
        if total[n] > 0.0:
            new_data_array[:, n] /= total[n]

    return new_data_array


def _perievent_trigger_average(
    time_target_array,
    count_array,
    time_array,
    data_array,
    starts,
    ends,
    windows,
    binsize,
    batch_size=64,
):
    if nap.utils.get_backend() == "jax":
        from pynajax.jax_process_perievent import event_trigger_average

        return event_trigger_average(
            time_target_array,
            count_array,
            time_array,
            data_array[:],
            starts,
            ends,
            windows,
            binsize,
            batch_size,
        )

    else:
        if data_array.ndim == 1:
            eta = _jitperievent_trigger_average(
                time_target_array,
                count_array,
                time_array,
                np.expand_dims(data_array[:], -1),
                starts,
                ends,
                windows,
                binsize,
            )
            return np.squeeze(eta, -1)
        else:
            return _jitperievent_trigger_average(
                time_target_array,
                count_array,
                time_array,
                data_array[:],
                starts,
                ends,
                windows,
                binsize,
            )


def _perievent_continuous(
    time_array, data_array, time_target_array, starts, ends, windowsize
):
    idx, slice_idx, N_target, w_starts = _jitcontinuous_perievent(
        time_array, time_target_array, starts, ends, windowsize
    )

    data_array = data_array[idx]

    if nap.utils.get_backend() == "jax":
        from pynajax.jax_process_perievent import perievent_continuous

        return perievent_continuous(
            data_array, np.sum(windowsize) + 1, N_target, slice_idx, w_starts
        )
    else:
        new_data_array = np.full(
            (np.sum(windowsize) + 1, N_target, *data_array.shape[1:]), np.nan
        )

        w_sizes = slice_idx[:, 1] - slice_idx[:, 0]  # Different sizes

        all_w_sizes = np.unique(w_sizes)
        all_w_start = np.unique(w_starts)

        for w_size in all_w_sizes:
            for w_start in all_w_start:
                col_idx = w_sizes == w_size
                new_idx = np.zeros((w_size, np.sum(col_idx)), dtype=int)
                for i, tmp in enumerate(slice_idx[col_idx]):
                    new_idx[:, i] = np.arange(tmp[0], tmp[1])

                new_data_array[w_start : w_start + w_size, col_idx] = data_array[
                    new_idx
                ]

        return new_data_array
