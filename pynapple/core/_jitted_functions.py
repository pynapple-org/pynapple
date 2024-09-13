import numpy as np
from numba import jit  # , njit, prange


################################
# Time only functions
################################
@jit(nopython=True)
def jitrestrict(time_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.int64)

    k = 0
    t = 0
    x = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                break
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                break
            else:
                ix[x] = t
                x += 1
            t += 1

        if k == m:
            break
        if t == n:
            break

    return ix[0:x]


@jit(nopython=True)
def jitrestrict_with_count(time_array, starts, ends, dtype=np.int64):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.int64)
    count = np.zeros(m, dtype=dtype)

    k = 0
    t = 0
    x = 0

    while ends[k] < time_array[t]:
        k += 1

    while k < m:
        # Outside
        while t < n:
            if time_array[t] >= starts[k]:
                break
            t += 1

        # Inside
        while t < n:
            if time_array[t] > ends[k]:
                k += 1
                break
            else:
                ix[x] = t
                count[k] += 1
                x += 1
            t += 1

        if k == m:
            break
        if t == n:
            break

    return ix[0:x], count


@jit(nopython=True)
def jitvaluefrom(time_array, time_target_array, count, count_target, starts, ends):
    m = starts.shape[0]
    n = time_array.shape[0]
    d = time_target_array.shape[0]

    idx = np.full(n, np.nan)

    if n > 0 and d > 0:
        for k in range(m):
            if count[k] > 0 and count_target[k] > 0:
                t = np.sum(count[0:k])
                i = np.sum(count_target[0:k])
                maxt = t + count[k]
                maxi = i + count_target[k]
                while t < maxt:
                    interval = abs(time_array[t] - time_target_array[i])
                    idx[t] = float(i)

                    i += 1
                    while i < maxi:
                        new_interval = abs(time_array[t] - time_target_array[i])
                        if new_interval > interval:
                            break
                        else:
                            idx[t] = float(i)
                            interval = new_interval
                            i += 1
                    i -= 1
                    t += 1

    return idx


@jit(nopython=True)
def jitcount(time_array, starts, ends, bin_size, dtype):
    idx, countin = jitrestrict_with_count(time_array, starts, ends)
    time_array = time_array[idx]

    m = starts.shape[0]

    nb_bins = np.zeros(m, dtype=np.int32)
    for k in range(m):
        if (ends[k] - starts[k]) > bin_size:
            nb_bins[k] = int(np.ceil((ends[k] + bin_size - starts[k]) / bin_size))
        else:
            nb_bins[k] = 1

    nb = np.sum(nb_bins)
    bins = np.zeros(nb, dtype=np.float64)
    cnt = np.zeros(nb, dtype=dtype)

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
                lbound = np.round(lbound, 9)
                b += 1
        t = maxt
        k += 1

    new_time_array = bins[0:b]
    new_data_array = cnt[0:b]

    return (new_time_array, new_data_array)


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


################################
# Time Data functions
################################
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


def jitbin_array(time_array, data_array, starts, ends, bin_size):
    """Slice first for compatibility with lazy loading."""
    idx, countin = jitrestrict_with_count(time_array, starts, ends)
    return _jitbin_array(
        countin, time_array[idx], data_array[idx], starts, ends, bin_size
    )


@jit(nopython=True)
def _jitbin_array(countin, time_array, data_array, starts, ends, bin_size):
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
                lbound = np.round(lbound, 9)
                b += 1
        t = maxt
        k += 1

    new_time_array = bins[0:b]

    new_data_array = average[0:b] / cnt[0:b]

    return (new_time_array, new_data_array)


# @jit(nopython=True)
# def jitconvolve(d, a):
#     return np.convolve(d, a)


# @njit(parallel=True)
# def pjitconvolve(data_array, array, trim="both"):
#     shape = data_array.shape
#     t = shape[0]
#     k = array.shape[0]

#     data_array = data_array.reshape(t, -1)
#     new_data_array = np.zeros(data_array.shape)

#     if trim == "both":
#         cut = ((k - 1) // 2, t + k - 1 - ((k - 1) // 2) - (1 - k % 2))
#     elif trim == "left":
#         cut = (k - 1, t + k - 1)
#     elif trim == "right":
#         cut = (0, t)

#     for i in prange(data_array.shape[1]):
#         new_data_array[:, i] = jitconvolve(data_array[:, i], array)[cut[0] : cut[1]]

#     new_data_array = new_data_array.reshape(shape)

#     return new_data_array


################################
# IntervalSet functions
################################
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
def _jitfix_iset(start, end):
    """
    0 - > "Some starts and ends are equal. Removing 1 microsecond!",
    1 - > "Some ends precede the relative start. Dropping them!",
    2 - > "Some starts precede the previous end. Joining them!",
    3 - > "Some epochs have no duration"

    Parameters
    ----------
    start : numpy.ndarray
        Description
    end : numpy.ndarray
        Description

    Returns
    -------
    TYPE
        Description
    """
    to_warn = np.zeros(4, dtype=np.bool_)
    m = start.shape[0]
    data = np.zeros((m, 2), dtype=np.float64)
    i = 0
    ct = 0

    while i < m:
        newstart = start[i]
        newend = end[i]

        while i < m:
            if end[i] == start[i]:
                to_warn[3] = True
                i += 1
            else:
                newstart = start[i]
                newend = end[i]
                break

        while i < m:
            if end[i] < start[i]:
                to_warn[1] = True
                i += 1
            else:
                newstart = start[i]
                newend = end[i]
                break

        if i >= m:
            break

        while i < m - 1:
            if start[i + 1] < end[i]:
                to_warn[2] = True
                i += 1
                newend = max(end[i - 1], end[i])
            else:
                break

        if i < m - 1:
            if newend == start[i + 1]:
                to_warn[0] = True
                newend -= 1.0e-6

        data[ct, 0] = newstart
        data[ct, 1] = newend

        ct += 1
        i += 1

    data = data[0:ct]

    return (data, to_warn)
