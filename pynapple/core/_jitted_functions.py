import numpy as np
from numba import jit  # , njit, prange

from .utils import take


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


@jit(nopython=True, cache=True)
def jitcount_epochs(time_array, unit_pos, starts, ends, n_units, dtype=np.int64):
    """Count timestamps per epoch and per unit.

    ``time_array`` is sorted and may merge the timestamps of several units,
    with ``unit_pos`` the column ``0..n_units-1`` of each timestamp. Returns a
    ``(n_epochs, n_units)`` count matrix. Epoch ends are inclusive, as in
    :func:`jitrestrict_with_count`. For a single series, pass ``n_units=1``
    and ``unit_pos=np.broadcast_to(np.int64(0), len(time_array))``.
    """
    n = len(time_array)
    m = len(starts)
    count = np.zeros((m, n_units), dtype=dtype)

    if n == 0 or m == 0:
        return count

    k = 0
    t = 0

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
                count[k, unit_pos[t]] += 1
            t += 1

        if k == m:
            break
        if t == n:
            break

    return count


@jit(nopython=True, cache=True)
def jitcount(time_array, unit_pos, starts, ends, bin_size, n_units, dtype):
    """Count timestamps per bin of ``bin_size`` within each epoch, per unit.

    ``time_array`` is sorted and may merge the timestamps of several units,
    with ``unit_pos`` the column ``0..n_units-1`` of each timestamp. Returns
    the bin centers and a ``(n_bins, n_units)`` count matrix. For a single
    series, pass ``n_units=1`` and
    ``unit_pos=np.broadcast_to(np.int64(0), len(time_array))``.

    The kernel walks the timestamps and the bins together in one sweep. It
    does not restrict the timestamps first. It rounds the bin edges to 9
    decimals. Each bin includes its left edge and excludes its right edge,
    like ``np.histogram``. The kernel keeps a bin only if its center is at or
    before the epoch end.
    """
    n = time_array.shape[0]
    m = starts.shape[0]

    nb_bins = np.zeros(m, dtype=np.int32)
    for k in range(m):
        if (ends[k] - starts[k]) > bin_size:
            nb_bins[k] = int(np.ceil((ends[k] + bin_size - starts[k]) / bin_size))
        else:
            nb_bins[k] = 1

    nb = np.sum(nb_bins)
    bins = np.zeros(nb, dtype=np.float64)
    cnt = np.zeros((nb, n_units), dtype=dtype)

    t = 0
    b = 0

    for k in range(m):
        # Outside
        while t < n and time_array[t] < starts[k]:
            t += 1

        maxb = b + nb_bins[k]
        lbound = starts[k]

        while b < maxb:
            xpos = lbound + bin_size / 2
            if xpos > ends[k]:
                break
            else:
                bins[b] = xpos
                rbound = np.round(lbound + bin_size, 9)
                # similar to numpy histogram
                while t < n and time_array[t] < rbound and time_array[t] <= ends[k]:
                    cnt[b, unit_pos[t]] += 1
                    t += 1

                lbound += bin_size
                lbound = np.round(lbound, 9)
                b += 1

        # Inside the epoch but past its last kept bin
        while t < n and time_array[t] <= ends[k]:
            t += 1

    return (bins[0:b], cnt[0:b])


@jit(nopython=True, cache=True)
def jittimediff_grouped(time_array, unit_pos, starts, ends, n_units, alpha):
    """Differences between subsequent timestamps of each unit, within epochs.

    Forward sweeps over the merged, globally sorted ``time_array``, keeping the
    last timestamp seen per unit (and the epoch it was seen in): a difference is
    emitted whenever a unit is seen again within the same epoch. A first sweep
    only counts the differences of each unit, so that the second one writes
    each unit's differences into its own contiguous block, in time order --
    splitting the output by unit then needs no sort and no gather.

    Returns
    -------
    (new_t, new_d, offsets)
        ``new_t = last + alpha * diff`` and ``new_d = diff``, where
        ``[offsets[i]:offsets[i + 1]]`` is the block of unit ``i``.
    """
    n = len(time_array)
    m = len(starts)
    last_time = np.zeros(n_units, dtype=np.float64)
    last_epoch = np.full(n_units, -1, dtype=np.int64)
    offsets = np.zeros(n_units + 1, dtype=np.int64)

    for sweep in range(2):
        if sweep == 1:
            for u in range(n_units):
                offsets[u + 1] += offsets[u]
            fill = offsets[:-1].copy()
            new_t = np.empty(offsets[n_units], dtype=np.float64)
            new_d = np.empty(offsets[n_units], dtype=np.float64)
            last_epoch[:] = -1

        t = 0
        k = 0
        while k < m and t < n:
            while t < n and time_array[t] < starts[k]:
                t += 1
            while t < n and time_array[t] <= ends[k]:
                u = unit_pos[t]
                if last_epoch[u] == k:
                    if sweep == 0:
                        offsets[u + 1] += 1
                    else:
                        diff = time_array[t] - last_time[u]
                        new_d[fill[u]] = diff
                        new_t[fill[u]] = last_time[u] + alpha * diff
                        fill[u] += 1
                last_time[u] = time_array[t]
                last_epoch[u] = k
                t += 1
            k += 1

    return new_t, new_d, offsets


@jit(nopython=True, cache=True)
def jitgroup_by_unit(unit_pos, n_units):
    """Stable counting sort of dense unit positions.

    Returns ``(order, offsets)`` such that ``order[offsets[i]:offsets[i + 1]]``
    are the positions holding unit ``i``, in their original order. O(n), unlike
    a comparison-based stable argsort.
    """
    n = unit_pos.shape[0]
    offsets = np.zeros(n_units + 1, dtype=np.int64)
    for i in range(n):
        offsets[unit_pos[i] + 1] += 1
    for u in range(n_units):
        offsets[u + 1] += offsets[u]
    fill = offsets[:-1].copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        u = unit_pos[i]
        order[fill[u]] = i
        fill[u] += 1
    return order, offsets


@jit(nopython=True, cache=True)
def jitcount_clusters(clusters, lo, span):
    """Number of timestamps of each cluster key, in one pass.

    ``counts[k - lo]`` is the number of entries of ``clusters`` equal to ``k``,
    for ``lo <= k < lo + span``. Assumes dense keys: memory is O(span).
    """
    counts = np.zeros(span, dtype=np.int64)
    for i in range(clusters.shape[0]):
        counts[clusters[i] - lo] += 1
    return counts


# The helper below is called once per timestamp from the innermost loop, so it
# is declared inline="always". That makes numba paste its body into the caller
# instead of emitting a call. Consequences are that: there is no longer a
# function call per element, and because `mode` is the same on every iteration,
# the compiler can test it once before the loop rather than on every timestamp.


@jit(nopython=True, cache=True, inline="always")
def _valuefrom_match(time_target_array, timestamp, first_after, start, stop, mode):
    """Pick the target matching one timestamp, within one epoch's target slice.

    All three modes are derived from a single pivot, so the caller only has to
    locate that pivot once (with a merge scan) and this decides what it means.

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

    Runs of identical target timestamps are reachable: pynapple neither rejects
    nor deduplicates duplicate timestamps, and a unit conversion can create them
    (e.g. ``Ts(t=[1000.0, 1000.0], time_units="ms")``).
    """
    last_at_or_before = first_after - 1

    if mode != 1:
        # before and after: an exact hit resolves to the first target of its run
        if (
            last_at_or_before >= start
            and time_target_array[last_at_or_before] == timestamp
        ):
            index = last_at_or_before
            while (
                index > start
                and time_target_array[index - 1] == time_target_array[index]
            ):
                index -= 1
            return index
        if mode == 0:  # before: last target <= timestamp
            return last_at_or_before if last_at_or_before >= start else -1
        # after: first target >= timestamp
        return first_after if first_after < stop else -1

    # closest: whichever neighbour is nearer
    if first_after >= stop:  # nothing above: only the lower neighbour
        return last_at_or_before if last_at_or_before >= start else -1

    # the upper neighbour is the last target of its run
    upper = first_after
    while upper + 1 < stop and time_target_array[upper + 1] == time_target_array[upper]:
        upper += 1
    if last_at_or_before < start:  # nothing at or below: only the upper neighbour
        return upper
    # `<=`, not `<`: on an exact distance tie the reference kernel keeps walking
    # forward (its break test is `new_interval > interval`), landing on the later
    # target. A strict `<` here silently disagrees on every equidistant match.
    if (time_target_array[upper] - timestamp) <= (
        timestamp - time_target_array[last_at_or_before]
    ):
        return upper
    return last_at_or_before


@jit(nopython=True, cache=True)
def jitvaluefrom(
    time_array, time_target_array, starts_in, ends_in, starts_tg, ends_tg, mode
):
    """Match each input timestamp to a target timestamp, in each epoch.

    The function does one merge scan in each epoch. The input timestamps are
    sorted, so the pivot (the first target after the timestamp) only moves
    forward, and an epoch costs ``n_in + n_tg`` steps.

    The function reads the full arrays through the bounds of each epoch, so
    the caller does not have to restrict the arrays first.

    Parameters
    ----------
    time_array : ndarray
        The sorted input timestamps.
    time_target_array : ndarray
        The sorted target timestamps.
    starts_in, ends_in : ndarray[int64]
        The bounds ``[start, end)`` of each epoch in ``time_array``.
    starts_tg, ends_tg : ndarray[int64]
        The bounds ``[start, end)`` of each epoch in ``time_target_array``.
    mode : int
        0 before, 1 closest, 2 after.

    Returns
    -------
    ndarray[int64]
        For each input timestamp in the epochs, the index in
        ``time_target_array`` of its target, or -1 if no target matches. The
        epochs are in order.
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
        target_start = starts_tg[k]
        target_stop = ends_tg[k]
        if target_stop > target_start:  # with no target in the epoch, nothing matches
            first_after = target_start
            for i in range(n_in):
                timestamp = time_array[in_start + i]
                # move the pivot forward, to the first target after timestamp
                while (
                    first_after < target_stop
                    and time_target_array[first_after] <= timestamp
                ):
                    first_after += 1
                idx[out_offset + i] = _valuefrom_match(
                    time_target_array,
                    timestamp,
                    first_after,
                    target_start,
                    target_stop,
                    mode,
                )
        out_offset += n_in

    return idx


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
        countin, time_array[idx], take(data_array, idx), starts, ends, bin_size
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
