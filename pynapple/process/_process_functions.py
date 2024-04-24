"""
    This module holds some process function of pynapple that can be
    called with numba or pynajax as backend    

    If pynajax is installed and `nap.nap_config.backend` is set 
    to `jax`, the module will call the functions within pynajax.
    Otherwise the module will call the functions within `_jitted_functions.py`.

"""

import numpy as np
from numba import jit, njit, prange

from .. import core as nap


@njit(parallel=True)
def _jitcontinuous_perievent(
    time_array, data_array, time_target_array, starts, ends, windowsize
):
    N_samples = len(time_array)
    N_target = len(time_target_array)
    N_epochs = len(starts)
    count = np.zeros((N_epochs, 2), dtype=np.int64)
    start_t = np.zeros((N_epochs, 2), dtype=np.int64)

    k = 0  # Epochs
    t = 0  # Samples
    i = 0  # Target

    while ends[k] < time_array[t] and ends[k] < time_target_array[i]:
        k += 1

    while k < N_epochs:
        # Outside
        while t < N_samples:
            if time_array[t] >= starts[k]:
                break
            t += 1

        while i < N_target:
            if time_target_array[i] >= starts[k]:
                break
            i += 1

        if time_array[t] <= ends[k]:
            start_t[k, 0] = t

        if time_target_array[i] <= ends[k]:
            start_t[k, 1] = i

        # Inside
        while t < N_samples:
            if time_array[t] > ends[k]:
                break
            else:
                count[k, 0] += 1
            t += 1

        while i < N_target:
            if time_target_array[i] > ends[k]:
                break
            else:
                count[k, 1] += 1
            i += 1

        k += 1

        if k == N_epochs:
            break
        if t == N_samples:
            break
        if i == N_target:
            break

    new_data_array = np.full(
        (np.sum(windowsize) + 1, np.sum(count[:, 1]), *data_array.shape[1:]), np.nan
    )

    if np.any((count[:, 0] * count[:, 1]) > 0):
        for k in prange(N_epochs):
            if count[k, 0] > 0 and count[k, 1] > 0:
                t = start_t[k, 0]
                i = start_t[k, 1]
                maxt = t + count[k, 0]
                maxi = i + count[k, 1]
                cnt_i = np.sum(count[0:k, 1])

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

                    left = np.minimum(windowsize[0], t_pos - start_t[k, 0])
                    right = np.minimum(windowsize[1], maxt - t_pos - 1)
                    center = windowsize[0] + 1
                    new_data_array[center - left - 1 : center + right, cnt_i] = (
                        data_array[t_pos - left : t_pos + right + 1]
                    )

                    t -= 1
                    i += 1
                    cnt_i += 1

    return new_data_array


@jit(nopython=True)
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
    time_array,
    count_array,
    time_target_array,
    data_target_array,
    starts,
    ends,
    windows,
    binsize,
    batch_size=64,
):
    if nap.utils.get_backend() == "jax":
        from pynajax.jax_process_perievent import event_trigger_average

        return event_trigger_average(
            time_array,
            count_array,
            time_target_array,
            np.expand_dims(data_target_array, -1),
            starts,
            ends,
            windows,
            binsize,
            batch_size,
        )

    else:
        if data_target_array.ndim == 1:
            eta = _jitperievent_trigger_average(
                time_array,
                count_array,
                time_target_array,
                np.expand_dims(data_target_array, -1),
                starts,
                ends,
                windows,
                binsize,
            )
            return np.squeeze(eta, -1)
        else:
            return _jitperievent_trigger_average(
                time_array,
                count_array,
                time_target_array,
                data_target_array,
                starts,
                ends,
                windows,
                binsize,
            )


def _perievent_continuous(
    time_array, data_array, time_target_array, starts, ends, windowsize
):
    if nap.utils.get_backend() == "jax":
        from pynajax.jax_process_perievent import perievent_continuous

        return perievent_continuous(
            time_array, data_array, time_target_array, starts, ends, windowsize
        )
    else:
        return _jitcontinuous_perievent(
            time_array, data_array, time_target_array, starts, ends, windowsize
        )
