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


@jit(nopython=True)
def _cross_correlogram(t1, t2, binsize, windowsize):
    """
    Performs the discrete cross-correlogram of two time series.
    The units should be in s for all arguments.
    Return the firing rate of the series t2 relative to the timings of t1.
    See compute_crosscorrelogram, compute_autocorrelogram and compute_eventcorrelogram
    for wrappers of this function.

    Parameters
    ----------
    t1 : numpy.ndarray
        The timestamps of the reference time series (in seconds)
    t2 : numpy.ndarray
        The timestamps of the target time series (in seconds)
    binsize : float
        The bin size (in seconds)
    windowsize : float
        The window size (in seconds)

    Returns
    -------
    numpy.ndarray
        The cross-correlogram
    numpy.ndarray
        Center of the bins (in s)

    """
    # nbins = ((windowsize//binsize)*2)

    nt1 = len(t1)
    nt2 = len(t2)

    nbins = int((windowsize * 2) // binsize)
    if np.floor(nbins / 2) * 2 == nbins:
        nbins = nbins + 1

    w = (nbins / 2) * binsize
    C = np.zeros(nbins)
    i2 = 0

    for i1 in range(nt1):
        lbound = t1[i1] - w
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2 + 1
        while i2 > 0 and t2[i2 - 1] > lbound:
            i2 = i2 - 1

        rbound = lbound
        leftb = i2
        for j in range(nbins):
            k = 0
            rbound = rbound + binsize
            while leftb < nt2 and t2[leftb] < rbound:
                leftb = leftb + 1
                k = k + 1

            C[j] += k

    C = C / (nt1 * binsize)

    m = -w + binsize / 2
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m + j * binsize

    return C, B


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
):
    if nap.utils.get_backend() == "jax":
        from pynajax.jax_process_eta import event_trigger_average

        return event_trigger_average(
            time_array,
            count_array,
            time_target_array,
            np.expand_dims(data_target_array, -1),
            starts,
            ends,
            windows,
            binsize,
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
