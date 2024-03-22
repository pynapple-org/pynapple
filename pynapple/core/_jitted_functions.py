import numpy as np
from numba import jit, njit, prange


@jit(nopython=True)
def jitrestrict(time_array, data_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.bool_)

    k = 0
    t = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                # ix[t] = True
                # t += 1
                break
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                break
            else:
                ix[t] = True
            t += 1

        if k == m:
            break
        if t == n:
            break

    new_time_array = time_array[ix]
    new_data_array = data_array[ix]
    return (new_time_array, new_data_array)


@jit(nopython=True)
def jittsrestrict(time_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.bool_)

    k = 0
    t = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                # ix[t] = True
                # t += 1
                break
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                break
            else:
                ix[t] = True
            t += 1

        if k == m:
            break
        if t == n:
            break

    new_time_array = time_array[ix]
    return new_time_array


@jit(nopython=True)
def jitrestrict_with_count(time_array, data_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.bool_)
    count = np.zeros(m, dtype=np.int64)

    k = 0
    t = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                # ix[t] = True
                # count[k] += 1
                # t += 1
                break
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                break
            else:
                ix[t] = True
                count[k] += 1
            t += 1

        if k == m:
            break
        if t == n:
            break

    new_time_array = time_array[ix]
    new_data_array = data_array[ix]
    return new_time_array, new_data_array, count


@jit(nopython=True)
def jittsrestrict_with_count(time_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.bool_)
    count = np.zeros(m, dtype=np.int64)

    k = 0
    t = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                # ix[t] = True
                # count[k] += 1
                # t += 1
                break
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                break
            else:
                ix[t] = True
                count[k] += 1
            t += 1

        if k == m:
            break
        if t == n:
            break

    new_time_array = time_array[ix]
    return new_time_array, count


@jit(nopython=True)
def jitthreshold(time_array, data_array, starts, ends, thr, method="above"):
    n = time_array.shape[0]

    if method == "above":
        ix = data_array > thr
    elif method == "below":
        ix = data_array < thr
    elif method == "aboveequal":
        ix = data_array >= thr
    elif method == "belowequal":
        ix = data_array <= thr

    k = 0
    t = 0

    ix_start = np.zeros(n, dtype=np.bool_)
    ix_end = np.zeros(n, dtype=np.bool_)
    new_start = np.zeros(n, dtype=np.float64)
    new_end = np.zeros(n, dtype=np.float64)

    while time_array[t] < starts[k]:
        k += 1

    if ix[t]:
        ix_start[t] = 1
        new_start[t] = time_array[t]

    t += 1

    while t < n - 1:
        # transition
        if time_array[t] > ends[k]:
            k += 1
            if ix[t - 1]:
                ix_end[t - 1] = 1
                new_end[t - 1] = time_array[t - 1]
            if ix[t]:
                ix_start[t] = 1
                new_start[t] = time_array[t]

        else:
            if not ix[t - 1] and ix[t]:
                ix_start[t] = 1
                new_start[t] = time_array[t] - (time_array[t] - time_array[t - 1]) / 2

            if ix[t - 1] and not ix[t]:
                ix_end[t] = 1
                new_end[t] = time_array[t] - (time_array[t] - time_array[t - 1]) / 2

        t += 1

    if ix[t] and ix[t - 1]:
        ix_end[t] = 1
        new_end[t] = time_array[t]

    if ix[t] and not ix[t - 1]:
        ix_start[t] = 1
        ix_end[t] = 1
        new_start[t] = time_array[t] - (time_array[t] - time_array[t - 1]) / 2
        new_end[t] = time_array[t]

    elif ix[t - 1] and not ix[t]:
        ix_end[t] = 1
        new_end[t] = time_array[t] - (time_array[t] - time_array[t - 1]) / 2

    new_time_array = time_array[ix]
    new_data_array = data_array[ix]
    new_starts = new_start[ix_start]
    new_ends = new_end[ix_end]

    return (new_time_array, new_data_array, new_starts, new_ends)


@jit(nopython=True)
def jitvaluefrom(time_array, time_target_array, data_target_array, starts, ends):
    time_array, count = jittsrestrict_with_count(time_array, starts, ends)
    time_target_array, data_target_array, count_target = jitrestrict_with_count(
        time_target_array, data_target_array, starts, ends
    )

    m = starts.shape[0]
    n = time_array.shape[0]
    d = time_target_array.shape[0]

    new_data_array = np.zeros(n, dtype=data_target_array.dtype)

    if n > 0 and d > 0:
        for k in range(m):
            if count[k] > 0 and count_target[k] > 0:
                t = np.sum(count[0:k])
                i = np.sum(count_target[0:k])
                maxt = t + count[k]
                maxi = i + count_target[k]
                while t < maxt:
                    interval = abs(time_array[t] - time_target_array[i])
                    new_data_array[t] = data_target_array[i]
                    i += 1
                    while i < maxi:
                        new_interval = abs(time_array[t] - time_target_array[i])
                        if new_interval > interval:
                            break
                        else:
                            new_data_array[t] = data_target_array[i]
                            interval = new_interval
                            i += 1
                    i -= 1
                    t += 1

    return (time_array, new_data_array, starts, ends)


@jit(nopython=True)
def jitvaluefromtensor(time_array, time_target_array, data_target_array, starts, ends):
    time_array, _, count = jitrestrict_with_count(
        time_array, np.zeros(time_array.shape[0]), starts, ends
    )
    time_target_array, data_target_array, count_target = jitrestrict_with_count(
        time_target_array, data_target_array, starts, ends
    )

    m = starts.shape[0]
    n = time_array.shape[0]
    d = time_target_array.shape[0]

    new_data_array = np.zeros(
        (n, *data_target_array.shape[1:]), dtype=data_target_array.dtype
    )

    if n > 0 and d > 0:
        for k in range(m):
            if count[k] > 0 and count_target[k] > 0:
                t = np.sum(count[0:k])
                i = np.sum(count_target[0:k])
                maxt = t + count[k]
                maxi = i + count_target[k]
                while t < maxt:
                    interval = abs(time_array[t] - time_target_array[i])
                    new_data_array[t] = data_target_array[i]
                    i += 1
                    while i < maxi:
                        new_interval = abs(time_array[t] - time_target_array[i])
                        if new_interval > interval:
                            break
                        else:
                            new_data_array[t] = data_target_array[i]
                            interval = new_interval
                            i += 1
                    i -= 1
                    t += 1

    return (time_array, new_data_array, starts, ends)


@jit(nopython=True)
def jitcount(time_array, starts, ends, bin_size):
    time_array, countin = jittsrestrict_with_count(time_array, starts, ends)

    m = starts.shape[0]

    nb_bins = np.zeros(m, dtype=np.int32)
    for k in range(m):
        if (ends[k] - starts[k]) > bin_size:
            nb_bins[k] = int(np.ceil((ends[k] + bin_size - starts[k]) / bin_size))
        else:
            nb_bins[k] = 1

    nb = np.sum(nb_bins)
    bins = np.zeros(nb, dtype=np.float64)
    cnt = np.zeros(nb, dtype=np.int64)

    k = 0
    t = 0
    b = 0

    while k < m:
        maxb = b + nb_bins[k]
        maxt = t + countin[k]
        lbound = starts[k]

        while b < maxb:
            xpos = lbound + bin_size / 2
            if xpos > ends[k]:
                break
            else:
                bins[b] = xpos
                rbound = np.round(lbound + bin_size, 9)
                while t < maxt:
                    if time_array[t] < rbound:  # similar to numpy hisrogram
                        cnt[b] += 1
                        t += 1
                    else:
                        break

                lbound += bin_size
                b += 1
        t = maxt
        k += 1

    new_time_array = bins[0:b]
    new_data_array = cnt[0:b]

    return (new_time_array, new_data_array)


@jit(nopython=True)
def jitbin(time_array, data_array, starts, ends, bin_size):
    time_array, data_array, countin = jitrestrict_with_count(
        time_array, data_array, starts, ends
    )

    m = starts.shape[0]

    nb_bins = np.zeros(m, dtype=np.int32)
    for k in range(m):
        if (ends[k] - starts[k]) > bin_size:
            nb_bins[k] = int(np.ceil((ends[k] + bin_size - starts[k]) / bin_size))
        else:
            nb_bins[k] = 1

    nb = np.sum(nb_bins)
    bins = np.zeros(nb, dtype=np.float64)
    cnt = np.zeros(nb, dtype=np.float64)
    average = np.zeros(nb, dtype=np.float64)

    k = 0
    t = 0
    b = 0

    while k < m:
        maxb = b + nb_bins[k]
        maxt = t + countin[k]
        lbound = starts[k]

        while b < maxb:
            xpos = lbound + bin_size / 2
            if xpos > ends[k]:
                break
            else:
                bins[b] = xpos
                rbound = np.round(lbound + bin_size, 9)
                while t < maxt:
                    if time_array[t] < rbound:  # similar to numpy hisrogram
                        cnt[b] += 1.0
                        average[b] += data_array[t]
                        t += 1
                    else:
                        break

                lbound += bin_size
                b += 1
        t = maxt
        k += 1

    new_time_array = bins[0:b]
    new_data_array = average[0:b] / cnt[0:b]

    return (new_time_array, new_data_array)


@jit(nopython=True)
def jitbin_array(time_array, data_array, starts, ends, bin_size):
    time_array, data_array, countin = jitrestrict_with_count(
        time_array, data_array, starts, ends
    )

    m = starts.shape[0]
    f = data_array.shape[1:]

    nb_bins = np.zeros(m, dtype=np.int32)
    for k in range(m):
        if (ends[k] - starts[k]) > bin_size:
            nb_bins[k] = int(np.ceil((ends[k] + bin_size - starts[k]) / bin_size))
        else:
            nb_bins[k] = 1

    nb = np.sum(nb_bins)
    bins = np.zeros(nb, dtype=np.float64)
    cnt = np.zeros((nb, *f), dtype=np.float64)
    average = np.zeros((nb, *f), dtype=np.float64)

    k = 0
    t = 0
    b = 0

    while k < m:
        maxb = b + nb_bins[k]
        maxt = t + countin[k]
        lbound = starts[k]

        while b < maxb:
            xpos = lbound + bin_size / 2
            if xpos > ends[k]:
                break
            else:
                bins[b] = xpos
                rbound = np.round(lbound + bin_size, 9)
                while t < maxt:
                    if time_array[t] < rbound:  # similar to numpy hisrogram
                        cnt[b] += 1.0
                        average[b] += data_array[t]
                        t += 1
                    else:
                        break

                lbound += bin_size
                b += 1
        t = maxt
        k += 1

    new_time_array = bins[0:b]

    new_data_array = average[0:b] / cnt[0:b]

    return (new_time_array, new_data_array)


@jit(nopython=True)
def jitintersect(start1, end1, start2, end2):
    m = start1.shape[0]
    n = start2.shape[0]

    i = 0
    j = 0

    newstart = np.zeros(m + n, dtype=np.float64)
    newend = np.zeros(m + n, dtype=np.float64)
    ct = 0

    while i < m:
        while j < n:
            if end2[j] > start1[i]:
                break
            j += 1

        if j == n:
            break

        if start2[j] < end1[i]:
            newstart[ct] = max(start1[i], start2[j])
            newend[ct] = min(end1[i], end2[j])
            ct += 1
            if end2[j] < end1[i]:
                j += 1
            else:
                i += 1
        else:
            i += 1

    newstart = newstart[0:ct]
    newend = newend[0:ct]

    return (newstart, newend)


@jit(nopython=True)
def jitunion(start1, end1, start2, end2):
    m = start1.shape[0]
    n = start2.shape[0]

    i = 0
    j = 0

    newstart = np.zeros(m + n, dtype=np.float64)
    newend = np.zeros(m + n, dtype=np.float64)
    ct = 0

    while i < m:
        while j < n:
            if end2[j] > start1[i]:
                break
            newstart[ct] = start2[j]
            newend[ct] = end2[j]
            ct += 1
            j += 1

        if j == n:
            break

        # overlap
        if start2[j] < end1[i]:
            newstart[ct] = min(start1[i], start2[j])

            while i < m and j < n:
                newend[ct] = max(end1[i], end2[j])

                if end1[i] < end2[j]:
                    i += 1
                else:
                    j += 1

                if i == m:
                    j += 1
                    ct += 1
                    break

                if j == n:
                    i += 1
                    ct += 1
                    break

                if end2[j] < start1[i]:
                    j += 1
                    ct += 1
                    break
                elif end1[i] < start2[j]:
                    i += 1
                    ct += 1
                    break

        else:
            newstart[ct] = start1[i]
            newend[ct] = end1[i]
            ct += 1
            i += 1

    while i < m:
        newstart[ct] = start1[i]
        newend[ct] = end1[i]
        ct += 1
        i += 1

    while j < n:
        newstart[ct] = start2[j]
        newend[ct] = end2[j]
        ct += 1
        j += 1

    newstart = newstart[0:ct]
    newend = newend[0:ct]

    return (newstart, newend)


@jit(nopython=True)
def jitdiff(start1, end1, start2, end2):
    m = start1.shape[0]
    n = start2.shape[0]

    i = 0
    j = 0

    newstart = np.zeros(m + n, dtype=np.float64)
    newend = np.zeros(m + n, dtype=np.float64)
    ct = 0

    while i < m:
        while j < n:
            if end2[j] > start1[i]:
                break
            j += 1

        if j == n:
            break

        # overlap
        if start2[j] < end1[i]:
            if start2[j] < start1[i] and end1[i] < end2[j]:
                i += 1

            else:
                if start2[j] > start1[i]:
                    newstart[ct] = start1[i]
                    newend[ct] = start2[j]
                    ct += 1
                    j += 1

                else:
                    newstart[ct] = end2[j]
                    newend[ct] = end1[i]
                    j += 1

                while j < n:
                    if start2[j] < end1[i]:
                        newstart[ct] = end2[j - 1]
                        newend[ct] = start2[j]
                        ct += 1
                        j += 1
                    else:
                        break

                if end2[j - 1] < end1[i]:
                    newstart[ct] = end2[j - 1]
                    newend[ct] = end1[i]
                    ct += 1
                else:
                    j -= 1
                i += 1

        else:
            newstart[ct] = start1[i]
            newend[ct] = end1[i]
            ct += 1
            i += 1

    while i < m:
        newstart[ct] = start1[i]
        newend[ct] = end1[i]
        ct += 1
        i += 1

    newstart = newstart[0:ct]
    newend = newend[0:ct]

    return (newstart, newend)


@jit(nopython=True)
def jitunion_isets(starts, ends):
    idx = np.argsort(starts)
    starts = starts[idx]
    ends = ends[idx]

    n = starts.shape[0]
    new_start = np.zeros(n, dtype=np.float64)
    new_end = np.zeros(n, dtype=np.float64)

    ct = 0
    new_start[ct] = starts[0]
    e = ends[0]
    i = 1
    while i < n:
        if starts[i] > e:
            new_end[ct] = e
            ct += 1
            new_start[ct] = starts[i]
            e = ends[i]
        else:
            e = max(e, ends[i])
        i += 1

    new_end[ct] = e
    ct += 1
    new_start = new_start[0:ct]
    new_end = new_end[0:ct]
    return (new_start, new_end)


@jit(nopython=True)
def jitin_interval(time_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    data = np.ones(n, dtype=np.float64) * np.nan

    k = 0
    t = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                # data[t] = k
                # t += 1
                break
            # data[t] = np.nan
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                # data[t] = np.nan
                break
            else:
                data[t] = k
            t += 1

        if k == m:
            break
        if t == n:
            break

    return data


@jit(nopython=True)
def jitremove_nan(time_array, index_nan):
    n = len(time_array)
    ix_start = np.zeros(n, dtype=np.bool_)
    ix_end = np.zeros(n, dtype=np.bool_)

    if not index_nan[0]:  # First start
        ix_start[0] = True

    t = 1
    while t < n:
        if index_nan[t - 1] and not index_nan[t]:  # start
            ix_start[t] = True
        if not index_nan[t - 1] and index_nan[t]:  # end
            ix_end[t - 1] = True
        t += 1

    if not index_nan[-1]:  # Last stop
        ix_end[-1] = True

    starts = time_array[ix_start]
    ends = time_array[ix_end]
    return (starts, ends)


@jit(nopython=True)
def jitconvolve(d, a):
    return np.convolve(d, a)


@njit(parallel=True)
def pjitconvolve(data_array, array, trim="both"):
    shape = data_array.shape
    t = shape[0]
    k = array.shape[0]

    data_array = data_array.reshape(t, -1)
    new_data_array = np.zeros(data_array.shape)

    if trim == "both":
        cut = ((1 - k % 2) + (k - 1) // 2, t + k - 1 - ((k - 1) // 2))
    elif trim == "left":
        cut = (k - 1, t + k - 1)
    elif trim == "right":
        cut = (0, t)

    for i in prange(data_array.shape[1]):
        new_data_array[:, i] = jitconvolve(data_array[:, i], array)[cut[0] : cut[1]]

    new_data_array = new_data_array.reshape(shape)

    return new_data_array


@njit(parallel=True)
def jitcontinuous_perievent(
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
def jitperievent_trigger_average(
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

    time_target_array, data_target_array, count = jitrestrict_with_count(
        time_target_array, data_target_array, starts, ends
    )
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


# @jit(nopython=True)
# def jit_poisson_IRLS(X, y, niter=100, tolerance=1e-5):
#     y = y.astype(np.float64)
#     X = X.astype(np.float64)
#     n, d = X.shape
#     W = np.ones(n)
#     iXtWX = np.linalg.inv(np.dot(X.T * W, X))
#     XtWY = np.dot(X.T * W, y)
#     B = np.dot(iXtWX, XtWY)

#     for _ in range(niter):
#         B_ = B
#         L = np.exp(X.dot(B))  # Link function
#         Z = L.reshape((-1, 1)) * X  # partial derivatives
#         delta = np.dot(np.linalg.inv(np.dot(Z.T * W, Z)), np.dot(Z.T * W, y))
#         B = B + delta
#         tol = np.sum(np.abs((B - B_) / B_))
#         if tol < tolerance:
#             return B
#     return B


# @jit(nopython=True)
# def jitfind_gaps(time_array, starts, ends, min_gap):
#     """
#     Jitted version of find_gap

#     Parameters
#     ----------
#     time_array : numpy.ndarray
#         Description
#     data_array : numpy.ndarray
#         Description
#     starts : numpy.ndarray
#         Description
#     ends : numpy.ndarray
#         Description

#     Returns
#     -------
#     TYPE
#         Description
#     """
#     n = len(time_array)
#     m = len(starts)

#     new_start = np.zeros(n+m, dtype=np.float64)
#     new_end = np.zeros(n+m, dtype=np.float64)

#     k = 0
#     t = 0
#     i = 0

#     while k<m:

#         start = starts[k]

#         while t < n:

#             if time_array[t] > ends[k]:
#                 break

#             if (time_array[t] - start) > min_gap:
#                 new_start[i] = start+1e-6
#                 new_end[i] = time_array[t]-1e-6
#                 start = time_array[t]
#                 t += 1
#                 i += 1

#             else:
#                 start = time_array[t]
#                 t += 1


#         k += 1


#     new_start = new_start[0:i]
#     new_end = new_end[0:i]

#     return new_start, new_end
