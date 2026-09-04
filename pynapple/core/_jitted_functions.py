import numpy as np
from numba import jit  # , njit, prange


################################
# Time only functions
################################
@jit(nopython=True, cache=True)
def jitrestrict(time_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.int64)

    if n == 0 or m == 0:
        return np.empty(0, dtype=np.int64)

    k = 0
    t = 0
    x = 0

    while k < m and ends[k] < time_array[t]:
        k += 1
    if k == m:
        return np.empty(0, dtype=np.int64)

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


@jit(nopython=True, cache=True)
def jitrestrict_with_count(time_array, starts, ends, dtype=np.int64):
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.int64)
    count = np.zeros(m, dtype=dtype)

    if n == 0 or m == 0:
        return np.empty(0, dtype=np.int64), np.zeros(m, dtype=dtype)

    k = 0
    t = 0
    x = 0

    while k < m and ends[k] < time_array[t]:
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


# Within one epoch, matching `n` timestamps against `d` targets costs ~n*log2(d)
# cache-missing jumps by binary search, versus ~n+d sequential steps by merge scan,
# so binary search only pays when the input is far sparser than the target.
#
# The measured crossover is not a fixed ratio: it is d/n ~ 64 at d=1e4 and doubles
# per decade (128, 256, 512 at 1e5, 1e6, 1e7), because a probe gets more expensive
# as the target outgrows each cache level. A constant is used anyway -- it costs at
# most 1.23x versus always picking the better kernel, and 1.7% on average over that
# grid, whereas fitting the growth (4*d**0.3 tracks it almost exactly) would bake
# one machine's cache hierarchy into the source. The textbook n*log2(d) < n+d test
# is much worse than either, at 1.27x average and 3.6x worst case.
VALUE_FROM_BSEARCH_RATIO = 128


@jit(nopython=True, cache=True, inline="always")
def use_bsearch_match(n_in, n_tg):
    """Whether to match this epoch by binary search rather than a merge scan.

    Exposed (and jitted so the kernel can call it) so the threshold can be tested
    directly: the branch itself is inside compiled code and cannot be spied on, and
    picking the wrong one costs speed without changing any result.

    Parameters
    ----------
    n_in : int
        Input timestamps in the epoch.
    n_tg : int
        Targets in the epoch.

    Returns
    -------
    bool
        True to binary search, False to merge scan.
    """
    return n_in * VALUE_FROM_BSEARCH_RATIO < n_tg


@jit(nopython=True, cache=True, inline="always")
def _vf_first_of_run(time_target_array, index, start):
    """First index of the run of targets sharing ``time_target_array[index]``.

    pynapple accepts duplicate timestamps: they are neither rejected nor
    deduplicated on construction, and a unit conversion can create them (e.g.
    ``Ts(t=[1000.0, 1000.0], time_units="ms")``). So a run of identical target
    timestamps is reachable, and which member of it a mode resolves to is
    user-visible behaviour that must be preserved.

    Parameters
    ----------
    time_target_array : ndarray
        Full, sorted target time array.
    index : int
        Index somewhere inside the run.
    start : int
        First index of the epoch's target slice; the walk stops there.

    Returns
    -------
    int
        Lowest index ``>= start`` holding the same timestamp as ``index``.
    """
    while index > start and time_target_array[index - 1] == time_target_array[index]:
        index -= 1
    return index


@jit(nopython=True, cache=True, inline="always")
def _vf_last_of_run(time_target_array, index, stop):
    """Last index of the run of targets sharing ``time_target_array[index]``.

    See :func:`_vf_first_of_run` on why runs of equal timestamps are reachable.

    Parameters
    ----------
    time_target_array : ndarray
        Full, sorted target time array.
    index : int
        Index somewhere inside the run.
    stop : int
        One past the last index of the epoch's target slice; the walk stops there.

    Returns
    -------
    int
        Highest index ``< stop`` holding the same timestamp as ``index``.
    """
    while index + 1 < stop and time_target_array[index + 1] == time_target_array[index]:
        index += 1
    return index


@jit(nopython=True, cache=True, inline="always")
def _vf_match(time_target_array, timestamp, first_after, start, stop, mode):
    """Pick the target matching one timestamp, within one epoch's target slice.

    All three modes are derived from a single pivot, so the caller only has to
    locate that pivot once (by binary search or by a merge scan, whichever is
    cheaper) and this decides what it means.

    Parameters
    ----------
    time_target_array : ndarray
        Full, sorted target time array.
    timestamp : float
        The input timestamp to match.
    first_after : int
        Index of the first target in ``[start, stop)`` strictly greater than
        ``timestamp`` (``stop`` if there is none). Its predecessor is therefore the
        last target less than or equal to ``timestamp``.
    start, stop : int
        Half-open bounds of this epoch's slice of ``time_target_array``.
    mode : int
        0 before, 1 closest, 2 after.

    Returns
    -------
    int
        Index into ``time_target_array``, or -1 when no target qualifies.

    Notes
    -----
    Tie-breaking reproduces the pre-rewrite kernel exactly, including two rules that
    are inconsistent with each other but are existing, user-visible behaviour:

    - on a run of identical target timestamps, ``before`` and ``after`` resolve to
      the *first* of the run while ``closest`` resolves to the *last*;
    - when the two neighbours are exactly equidistant, ``closest`` resolves to the
      *later* one.
    """
    last_at_or_before = first_after - 1

    if mode == 0:  # before: last target <= timestamp
        if (
            last_at_or_before >= start
            and time_target_array[last_at_or_before] == timestamp
        ):
            return _vf_first_of_run(time_target_array, last_at_or_before, start)
        return last_at_or_before if last_at_or_before >= start else -1

    if mode == 2:  # after: first target >= timestamp
        if (
            last_at_or_before >= start
            and time_target_array[last_at_or_before] == timestamp
        ):
            return _vf_first_of_run(time_target_array, last_at_or_before, start)
        return first_after if first_after < stop else -1

    # closest: whichever neighbour is nearer
    if last_at_or_before < start:  # nothing at or below: only the upper neighbour
        if first_after >= stop:
            return -1
        return _vf_last_of_run(time_target_array, first_after, stop)
    if first_after >= stop:  # nothing above: only the lower neighbour
        return last_at_or_before

    upper = _vf_last_of_run(time_target_array, first_after, stop)
    # `<=`, not `<`: on an exact distance tie the reference kernel keeps walking
    # forward (its break test is `new_interval > interval`), landing on the later
    # target. A strict `<` here silently disagrees on every equidistant match.
    if (time_target_array[upper] - timestamp) <= (
        timestamp - time_target_array[last_at_or_before]
    ):
        return upper
    return last_at_or_before


@jit(nopython=True, cache=True)
def jitvaluefrom_ranges(
    time_array, time_target_array, starts_in, ends_in, starts_tg, ends_tg, mode
):
    """Compute value_from indices from per-epoch slice boundaries.

    Unlike the pre-rewrite kernel, this takes the *full* time arrays plus the
    half-open slice boundaries of each epoch, so neither array has to be restricted
    and gathered beforehand, and per-epoch offsets are accumulated rather than
    recomputed with an O(n_epochs^2) prefix sum.

    Parameters
    ----------
    time_array : ndarray
        Full, sorted input time array.
    time_target_array : ndarray
        Full, sorted target time array.
    starts_in, ends_in : ndarray[int64]
        Half-open [start, end) boundaries of each epoch within ``time_array``.
    starts_tg, ends_tg : ndarray[int64]
        Half-open [start, end) boundaries of each epoch within
        ``time_target_array``.
    mode : int
        0 before, 1 closest, 2 after.

    Returns
    -------
    ndarray[int64]
        For each in-epoch input timestamp, the index into ``time_target_array`` of
        the matching target, or -1 when none qualifies. Length is the total number
        of in-epoch input timestamps, epochs concatenated in order.
    """
    n_epochs = starts_in.shape[0]

    n_out = 0
    for k in range(n_epochs):
        n_out += ends_in[k] - starts_in[k]

    idx = np.full(n_out, -1, dtype=np.int64)

    out_offset = 0
    for k in range(n_epochs):
        in_start = starts_in[k]
        n_in = ends_in[k] - in_start
        tg_start = starts_tg[k]
        tg_stop = ends_tg[k]
        n_tg = tg_stop - tg_start
        if n_tg == 0:  # no target in this epoch: every timestamp stays unmatched
            out_offset += n_in
            continue

        if use_bsearch_match(n_in, n_tg):
            # input far sparser than the target: binary search each timestamp.
            # The search spans the whole epoch every time on purpose -- narrowing
            # the lower bound from the previous result measures ~2.8x slower, as
            # it breaks the otherwise predictable access pattern.
            for i in range(n_in):
                timestamp = time_array[in_start + i]
                left = tg_start
                right = tg_stop
                while left < right:  # upper bound: first target > timestamp
                    # equivalent to floor(left + right / 2)
                    # shifting binary numbers by one position gives
                    # the half (10 in binary is 1010 shifted is 0101, which is 5)
                    mid = (left + right) >> 1
                    if time_target_array[mid] <= timestamp:
                        left = mid + 1
                    else:
                        right = mid
                idx[out_offset + i] = _vf_match(
                    time_target_array, timestamp, left, tg_start, tg_stop, mode
                )
        else:
            # comparable sizes: one sequential merge pass. `first_after` never
            # rewinds, because the input timestamps are non-decreasing, so the whole
            # epoch costs n_in + n_tg steps rather than n_in binary searches.
            first_after = tg_start
            for i in range(n_in):
                timestamp = time_array[in_start + i]
                while (
                    first_after < tg_stop
                    and time_target_array[first_after] <= timestamp
                ):
                    first_after += 1
                idx[out_offset + i] = _vf_match(
                    time_target_array, timestamp, first_after, tg_start, tg_stop, mode
                )
        out_offset += n_in

    return idx


@jit(nopython=True, cache=True)
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


@jit(nopython=True, cache=True)
def jitin_interval(time_array, starts, ends):
    n = len(time_array)
    m = len(starts)
    data = np.ones(n, dtype=np.float64) * np.nan

    if n == 0 or m == 0:
        return data

    k = 0
    t = 0

    while k < m and ends[k] < time_array[t]:
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


@jit(nopython=True, cache=True)
def jitremove_nan(time_array, index_nan):
    n = len(time_array)
    ix_start = np.zeros(n, dtype=np.bool_)
    ix_end = np.zeros(n, dtype=np.bool_)

    if n == 0:
        return time_array[ix_start], time_array[ix_end]

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
@jit(nopython=True, cache=True)
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

    if n == 0:
        return (time_array[ix], data_array[ix], new_start[ix_start], new_end[ix_end])

    while k < len(starts) and time_array[t] < starts[k]:
        k += 1

    if ix[t]:
        ix_start[t] = 1
        new_start[t] = time_array[t]

    if n == 1:
        if ix[t]:
            ix_end[t] = 1
            new_end[t] = time_array[t]
        return (time_array[ix], data_array[ix], new_start[ix_start], new_end[ix_end])

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


@jit(nopython=True, cache=True)
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


# @jit(nopython=True, cache=True)
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
@jit(nopython=True, cache=True)
def jitintersect(start1, end1, start2, end2):
    m = start1.shape[0]  # number of intervals in set 1
    n = start2.shape[0]  # number of intervals in set 2

    i = 0  # interval index for set 1
    j = 0  # interval index for set 2

    newstart = np.zeros(m + n, dtype=np.float64)
    newend = np.zeros(m + n, dtype=np.float64)
    newmeta = np.zeros((m + n, 2), dtype=np.int32)
    ct = 0  # counter for number of new intervals

    while i < m:
        while j < n:  # set 2 interval ends before set 1 interval starts
            if end2[j] > start1[i]:
                break
            j += 1  # increment set 2 index

        if j == n:  # stop if no more intervals in set 2
            break

        if start2[j] < end1[i]:  # set 2 interval starts before set 1 interval ends
            newstart[ct] = max(
                start1[i], start2[j]
            )  # start of interval is whichever occurs last
            newend[ct] = min(
                end1[i], end2[j]
            )  # end of interval is whichever occurs first
            newmeta[ct] = [
                i,
                j,
            ]  # store indices of intervals in set 1 and set 2 for metadata
            ct += 1
            if end2[j] < end1[i]:
                j += 1  # increment set 2 index if set 2 interval ends first
            else:
                i += 1  # increment set 1 index if set 1 interval ends first
        else:
            i += 1

    newstart = newstart[0:ct]
    newend = newend[0:ct]
    newmeta = newmeta[0:ct]

    return (newstart, newend, newmeta)


@jit(nopython=True, cache=True)
def jitunion(start1, end1, start2, end2):
    m = start1.shape[0]  # number of intervals in set 1
    n = start2.shape[0]  # number of intervals in set 2

    i = 0  # interval index for set 1
    j = 0  # interval index for set 2

    newstart = np.zeros(m + n, dtype=np.float64)
    newend = np.zeros(m + n, dtype=np.float64)
    ct = 0

    while i < m:
        while j < n:  # all set 2 intervals that start before set 1 interval
            if end2[j] > start1[i]:
                break
            newstart[ct] = start2[j]  # add set 2 interval
            newend[ct] = end2[j]
            ct += 1
            j += 1  # increment set 2 index

        if j == n:
            break

        if start2[j] < end1[i]:  # overlap
            newstart[ct] = min(
                start1[i], start2[j]
            )  # start of interval is whichever occurs first

            while i < m and j < n:
                newend[ct] = max(
                    end1[i], end2[j]
                )  # end of interval is whichever occurs last

                if end1[i] < end2[j]:
                    i += 1  # increment set 1 index if it ends first
                else:
                    j += 1  # increment set 2 index if it ends first

                if i == m:  # stop if no more intervals in set 1
                    j += 1  # increment set 2 index
                    ct += 1
                    break

                if j == n:  # stop if no more intervals in set 2
                    i += 1  # increment set 1 index
                    ct += 1
                    break

                # stop if end of overlap
                if end2[j] < start1[i]:  # set 2 interval comes first
                    j += 1  # increment set 2 index
                    ct += 1
                    break
                elif end1[i] < start2[j]:  # set 1 interval comes first
                    i += 1  # increment set 1 index
                    ct += 1
                    break

        else:  # no overlap
            newstart[ct] = start1[i]  # add set 1 interval
            newend[ct] = end1[i]
            ct += 1
            i += 1  # increment set 1 index

    while i < m:  # add remaining intervals from set 1
        newstart[ct] = start1[i]
        newend[ct] = end1[i]
        ct += 1
        i += 1

    while j < n:  # add remaining intervals from set 2
        newstart[ct] = start2[j]
        newend[ct] = end2[j]
        ct += 1
        j += 1

    newstart = newstart[0:ct]
    newend = newend[0:ct]

    return (newstart, newend)


@jit(nopython=True, cache=True)
def jitdiff(start1, end1, start2, end2):
    m = start1.shape[0]  # number of intervals in set 1
    n = start2.shape[0]  # number of intervals in set 2

    i = 0  # interval index for set 1
    j = 0  # interval index for set 2

    newstart = np.zeros(m + n, dtype=np.float64)
    newend = np.zeros(m + n, dtype=np.float64)
    newmeta = np.zeros(m + n, dtype=np.int32)
    ct = 0

    while i < m:
        while j < n:  # for all set 2 intervals that end before set 1 interval starts
            if end2[j] > start1[i]:
                break
            j += 1  # increment set 2 index

        if j == n:  # stop if no more intervals in set 2
            break

        if start2[j] < end1[i]:  # overlap
            if (
                start2[j] < start1[i] and end1[i] < end2[j]
            ):  # if set 1 interval is completely within set 2 interval
                i += 1  # increment set 1 index

            else:
                if (
                    start2[j] > start1[i]
                ):  # if set 2 interval starts inside set 1 interval
                    newstart[ct] = start1[i]  # add interval between both starts
                    newend[ct] = start2[j]
                    newmeta[ct] = i  # store index of interval in set 1 for metadata
                    ct += 1
                    j += 1  # increment set 2 index

                else:  # if set 2 interval starts before set 1 interval
                    newstart[ct] = end2[j]  # add interval between both ends
                    newend[ct] = end1[i]
                    newmeta[ct] = i
                    j += 1  # increment set 2 index

                while j < n:
                    if (
                        start2[j] < end1[i]
                    ):  # space between adjacent set 2 intervals falls inside set 1 interval
                        newstart[ct] = end2[
                            j - 1
                        ]  # add interval for space between adjacent set 2 intervals
                        newend[ct] = start2[j]
                        newmeta[ct] = i
                        ct += 1
                        j += 1  # increment set 2 index
                    else:
                        break

                if (
                    end2[j - 1] < end1[i]
                ):  # previous set 2 interval ends before set 1 interval
                    newstart[ct] = end2[j - 1]  # add interval between both ends
                    newend[ct] = end1[i]
                    newmeta[ct] = i
                    ct += 1
                else:  # previous set 2 interval ends after set 1 interval
                    j -= 1  # decrement set 2 index
                i += 1  # increment set 1 index

        else:  # no overlap
            newstart[ct] = start1[i]  # add set 1 interval
            newend[ct] = end1[i]
            newmeta[ct] = i
            ct += 1
            i += 1  # increment set 1 index

    while i < m:  # add remaining intervals from set 1
        newstart[ct] = start1[i]
        newend[ct] = end1[i]
        newmeta[ct] = i
        ct += 1
        i += 1

    newstart = newstart[0:ct]
    newend = newend[0:ct]
    newmeta = newmeta[0:ct]

    return (newstart, newend, newmeta)


@jit(nopython=True, cache=True)
def jitunion_isets(starts, ends):
    idx = np.argsort(starts)
    starts = starts[idx]
    ends = ends[idx]

    n = starts.shape[0]
    new_start = np.zeros(n, dtype=np.float64)
    new_end = np.zeros(n, dtype=np.float64)

    if n == 0:
        return (new_start, new_end)

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


@jit(nopython=True, cache=True)
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
