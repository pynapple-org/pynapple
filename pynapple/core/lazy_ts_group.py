"""
Lazy TsGroup: a TsGroup that keeps its spike times on disk.

A ``LazyTsGroup`` reads its spike times from a source only when an operation
needs them. It does not keep them in memory. Each operation reads only the
spike times that it needs into a temporary regular ``TsGroup``, and runs on
that group.

A source knows the layout of the spike times on disk and reads parts of them.
There are two sources:

- ``_RaggedArraySource`` reads a ragged layout, as in an NWB units table: the
  spike times of each unit, one unit after the other, and the end position of
  each unit. A selection of units is fast.
- ``_SortedArraySource`` reads a sorted layout: all the spike times in one
  sorted array, and the key of each spike in a second array. A selection of
  time is fast.
"""

import warnings
from numbers import Number

import numpy as np

from ._core_functions import _restrict_arrays
from ._jitted_functions import jitunion_isets
from .interval_set import IntervalSet
from .metadata_class import _MetadataMixin
from .time_index import TsIndex
from .ts_group import TsGroup

_CLOSED_MESSAGE = (
    "The file is closed. Thus this group cannot read its spike times. Select "
    "the units or the time that you need (e.g. units[[0, 1]] or "
    "units.restrict(ep)) before you close the file."
)


def _search(array, lo, hi, t, right):
    """Find the position of ``t`` in the sorted values ``array[lo:hi]``, as
    ``np.searchsorted`` does. For a file, each step of the binary search reads
    one value."""
    if isinstance(array, np.ndarray):
        side = "right" if right else "left"
        return lo + int(np.searchsorted(array[lo:hi], t, side=side))
    while lo < hi:
        mid = (lo + hi) // 2
        value = array[mid]
        if value < t or (right and value == t):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _tsgroup_from_state(state):
    """Make a regular TsGroup from a saved state, for pickle (see
    ``LazyTsGroup.__reduce_ex__``)."""
    obj = TsGroup.__new__(TsGroup)
    obj.__setstate__(state)
    return obj


def _support_and_counts(first, last, counts):
    """Compute the time support and the number of spikes of each unit, as an
    eager construction does.

    Parameters
    ----------
    first, last : ndarray
        The first and the last spike of each unit that is not empty.
    counts : ndarray
        The number of spikes of each unit.

    Returns
    -------
    time_support : IntervalSet
        The union of the ``[first, last]`` spans of the units. IntervalSet
        drops a span with no duration (a unit with one spike).
    counts : ndarray
        ``counts``, with 0 for a unit with one spike outside the time support.
        An eager construction drops that spike.
    """
    nonempty = counts > 0
    spans = last > first
    ts_start, ts_end = jitunion_isets(first[spans], last[spans])
    if len(ts_start) == 0:
        raise RuntimeError(
            "Union of time supports is empty. Consider passing a time support as argument."
        )

    # Only a unit with one spike can have a spike outside the time support.
    # That spike is `first`.
    is_single = ~spans & (counts[nonempty] == 1)
    if np.any(is_single):
        single = np.flatnonzero(nonempty)[is_single]
        t = first[is_single]
        k = np.searchsorted(ts_start, t, side="right") - 1
        outside = (k < 0) | (t > ts_end[np.maximum(k, 0)])
        counts = counts.copy()
        counts[single[outside]] = 0

    return IntervalSet(ts_start, ts_end), counts


class _RaggedArraySource:
    """Spike times in a ragged array, e.g. an NWB units table.

    ``ragged_array`` holds the spike times of each unit, one unit after the
    other. The spike times of each unit are sorted. ``ragged_array_index``
    holds the end position of each unit in ``ragged_array``. The arrays can be
    numpy arrays, h5py datasets or other array-likes that support slices.

    The constructor reads only ``ragged_array_index`` and the first and last
    spike of each unit. These values give the keys, the counts and the time
    support.

    Attributes
    ----------
    keys : ndarray
        The keys of the units, sorted.
    counts : ndarray
        The number of spikes of each unit, in the order of ``keys``.
    time_support : IntervalSet
        The union of the ``[first, last]`` spans of the units.
    keep_alive : object
        An object that the source keeps, so that the file stays open (e.g. the
        ``NWBFile`` that holds the arrays).
    closed : bool
        True after ``close()``.
    """

    def __init__(self, ragged_array, ragged_array_index, keys, keep_alive=None):
        """
        Parameters
        ----------
        ragged_array : array-like
            The spike times of all the units.
        ragged_array_index : array-like
            The end position of each unit in ``ragged_array`` (one position
            after its last spike).
        keys : array-like of int
            The key of each unit, in the order of ``ragged_array_index``.
        keep_alive : object, optional
            An object that the source keeps, so that the file stays open.

        Raises
        ------
        ValueError
            - If ``keys`` and ``ragged_array_index`` do not have the same
              length, or if two keys are equal.
            - If ``ragged_array_index`` decreases, or if it goes past the end
              of ``ragged_array``.
        """
        keys = np.asarray(keys, dtype=np.int64)
        stops = np.asarray(ragged_array_index, dtype=np.int64)
        starts = np.concatenate([[0], stops[:-1]]).astype(np.int64)
        counts = stops - starts

        if not hasattr(ragged_array, "shape"):  # a list in memory
            ragged_array = np.asarray(ragged_array, dtype=np.float64)

        if len(keys) != len(stops):
            raise ValueError(
                f"keys must have one key for each unit. There are {len(stops)} "
                f"units in ragged_array_index and {len(keys)} keys."
            )
        if len(np.unique(keys)) != len(keys):
            raise ValueError("Two keys have the same value.")
        if np.any(counts < 0):
            raise ValueError("ragged_array_index must not decrease.")
        if len(stops) and stops[-1] > len(ragged_array):
            raise ValueError(
                f"ragged_array_index goes past the end of ragged_array: the "
                f"last unit ends at {stops[-1]}, and ragged_array has "
                f"{len(ragged_array)} values."
            )

        # The first and the last spike of each unit that is not empty. h5py
        # needs increasing positions for a point selection.
        nonempty = counts > 0
        first = np.asarray(ragged_array[starts[nonempty]], dtype=np.float64)
        last = np.asarray(ragged_array[stops[nonempty] - 1], dtype=np.float64)

        time_support, counts = _support_and_counts(first, last, counts)

        # The arrays below follow the order of the units in the ragged array.
        self._ragged_array = ragged_array
        self._starts = starts
        self._stops = stops
        self._first = np.full(len(keys), np.nan)
        self._first[nonempty] = first
        self._last = np.full(len(keys), np.nan)
        self._last[nonempty] = last
        # `_sort_index[i]` is the position in the ragged array of `keys[i]`.
        self._sort_index = np.argsort(keys, kind="stable")

        self.keys = keys[self._sort_index]
        self.counts = counts[self._sort_index]
        self.time_support = time_support
        self.keep_alive = keep_alive
        self.closed = False

    def close(self):
        """Mark the file as closed. After this, a read raises a RuntimeError."""
        self.closed = True

    def _rows(self, keys):
        """Get the positions in the ragged array of the units ``keys``.
        ``keys`` must be sorted keys of the source."""
        return self._sort_index[np.searchsorted(self.keys, keys)]

    def _search(self, lo, hi, t, right):
        """Find the position of ``t`` in the sorted spikes
        ``ragged_array[lo:hi]`` of one unit."""
        return _search(self._ragged_array, lo, hi, t, right)

    def _read_rows(self, keys, lo, hi):
        """Read ``ragged_array[lo[i]:hi[i]]`` for each unit ``keys[i]``.

        ``keys`` must be sorted keys of the source.

        Returns
        -------
        times, clusters : ndarray
            The spike times, sorted by time and then by key, and the key of
            each spike.
        """
        if self.closed:
            raise RuntimeError(_CLOSED_MESSAGE)
        lengths = hi - lo
        try:
            array = self._ragged_array
            n_total = int(self._stops[-1]) if len(self._stops) else 0
            if len(lengths) and np.sum(lengths) == n_total:
                # For all the spikes, one read is faster than one read for
                # each unit.
                array = np.asarray(array[:n_total])
            parts = [
                np.asarray(array[a:b], dtype=np.float64)
                for a, b in zip(lo, hi)
                if b > a
            ]
        except (ValueError, OSError, KeyError) as err:
            raise RuntimeError("Cannot read the spike times from the file.") from err
        times = np.concatenate(parts) if parts else np.zeros(0)
        clusters = np.repeat(np.asarray(keys, dtype=np.int64), lengths)

        # The time support and the binary search assume that each unit is
        # sorted.
        ends = np.cumsum(lengths)
        unsorted = np.setdiff1d(np.flatnonzero(times[1:] < times[:-1]) + 1, ends)
        if len(unsorted):
            bad = np.unique(clusters[unsorted]).tolist()
            warnings.warn(
                f"Spike times of units {bad} are not sorted in the file. The "
                "lazy TsGroup assumes sorted spike times. Thus its time "
                "support, its rates and its time windows can be wrong. Load "
                "the spike times in memory instead (for an NWB file, use "
                "lazy_loading=False).",
                stacklevel=4,
            )

        # Sort by time, then by key for equal times (as TsGroup does).
        order = np.lexsort((clusters, times))
        return times[order], clusters[order]

    def read(self, keys=None, start=-np.inf, end=np.inf):
        """Read the spike times of the units ``keys``.

        The function reads only the spikes ``t`` with ``start <= t <= end``
        (in seconds). In a unit that is only partly in ``[start, end]``, a
        binary search in the file finds these spikes.

        Parameters
        ----------
        keys : array-like of int, optional
            Sorted keys of the source. The default is all the keys.
        start, end : float, optional
            The time window, in seconds. The default is all the time.

        Returns
        -------
        times, clusters : ndarray
            The spike times, sorted by time and then by key, and the key of
            each spike.
        """
        keys = self.keys if keys is None else np.asarray(keys, dtype=np.int64)
        rows = self._rows(keys)
        lo = self._starts[rows].copy()
        hi = self._stops[rows].copy()
        first = self._first[rows]
        last = self._last[rows]

        # Units with no spike in [start, end]. An empty unit has NaN as
        # `first`.
        if start > end:
            outside = np.ones(len(rows), dtype=bool)
        else:
            outside = ~(first <= end) | ~(last >= start)
        hi[outside] = lo[outside]
        for i in np.flatnonzero(~outside & (first < start)):
            lo[i] = self._search(lo[i], hi[i], start, right=False)
        for i in np.flatnonzero(~outside & (last > end)):
            hi[i] = self._search(lo[i], hi[i], end, right=True)
        # With unsorted spikes, the binary search can give hi < lo.
        hi = np.maximum(hi, lo)

        return self._read_rows(keys, lo, hi)

    def read_closest(self, t):
        """Read the two spikes around ``t`` (in seconds) in each unit.

        The closest spike of each unit is one of these two spikes.

        Returns
        -------
        times, clusters : ndarray
            As ``read`` returns them.
        """
        rows = self._rows(self.keys)
        lo, hi = self._starts[rows], self._stops[rows]
        pos = np.array(
            [self._search(a, b, t, right=False) for a, b in zip(lo, hi)],
            dtype=np.int64,
        )
        return self._read_rows(
            self.keys, np.maximum(lo, pos - 1), np.minimum(hi, pos + 1)
        )


class _SortedArraySource:
    """Spike times in two arrays: all the spike times in one sorted array, and
    the key of each spike in a second array.

    This is the layout of ``TsGroup.save`` (``t`` and ``index``), and of
    ``TsGroup.to_tsd()`` (its timestamps and its values). The arrays can be
    numpy arrays, h5py datasets, zarr arrays or other array-likes that support
    slices.

    The constructor reads all of ``clusters`` one time, in chunks of
    ``chunk_size`` values. This scan gives the keys, the counts, and the
    position of the first and last spike of each unit. Then the constructor
    reads only the times at these positions, for the time support.

    A selection of time is fast: two binary searches in ``times``, then one
    slice. A selection of units reads all of ``clusters`` again, in chunks, and
    reads ``times`` only in the chunks that hold selected spikes.

    Attributes
    ----------
    keys : ndarray
        The keys of the units, sorted.
    counts : ndarray
        The number of spikes of each unit, in the order of ``keys``.
    time_support : IntervalSet
        The union of the ``[first, last]`` spans of the units.
    keep_alive : object
        An object that the source keeps, so that the file stays open.
    closed : bool
        True after ``close()``.
    """

    def __init__(
        self, times, clusters, keys=None, keep_alive=None, chunk_size=1_000_000
    ):
        """
        Parameters
        ----------
        times : array-like
            The spike times of all the units, sorted.
        clusters : array-like of int
            The key of each spike.
        keys : array-like of int, optional
            The keys of the units. Use it to add units with no spike. The
            default is the keys in ``clusters``.
        keep_alive : object, optional
            An object that the source keeps, so that the file stays open.
        chunk_size : int, optional
            The number of values that a scan of ``clusters`` reads at a time.

        Raises
        ------
        ValueError
            - If ``times`` and ``clusters`` do not have the same length.
            - If two keys are equal, or if ``clusters`` holds a key that is
              not in ``keys``.
        """
        if not hasattr(times, "shape"):  # a list in memory
            times = np.asarray(times, dtype=np.float64)
        if not hasattr(clusters, "shape"):
            clusters = np.asarray(clusters, dtype=np.int64)
        if len(times) != len(clusters):
            raise ValueError(
                f"times and clusters must have the same length. times has "
                f"{len(times)} values and clusters has {len(clusters)}."
            )
        self._times = times
        self._clusters = clusters
        self._n = len(times)
        self._chunk_size = int(chunk_size)
        self.keep_alive = keep_alive
        self.closed = False

        found, counts, first_pos, last_pos = self._scan_clusters()
        if keys is None:
            keys = found
        else:
            given = np.asarray(keys, dtype=np.int64)
            keys = np.unique(given)
            if len(keys) != len(given):
                raise ValueError("Two keys have the same value.")
            missing = np.setdiff1d(found, keys)
            if len(missing):
                raise ValueError(
                    f"clusters holds keys that are not in keys: {missing.tolist()}."
                )

        # A key with no spike has a count of 0 and a position of -1.
        pos = np.searchsorted(keys, found)
        all_counts = np.zeros(len(keys), dtype=np.int64)
        all_counts[pos] = counts
        self._first_pos = np.full(len(keys), -1, dtype=np.int64)
        self._first_pos[pos] = first_pos
        self._last_pos = np.full(len(keys), -1, dtype=np.int64)
        self._last_pos[pos] = last_pos

        values = self._read_points(np.concatenate([first_pos, last_pos]))
        first, last = values[: len(found)], values[len(found) :]
        time_support, all_counts = _support_and_counts(first, last, all_counts)

        self.keys = keys
        self.counts = all_counts
        self.time_support = time_support

    def close(self):
        """Mark the file as closed. After this, a read raises a RuntimeError."""
        self.closed = True

    #################################
    # Low-level reads
    #################################

    def _chunks(self, lo, hi):
        """Read ``clusters[lo:hi]`` in chunks. Give the start position of each
        chunk and its keys."""
        for a in range(lo, hi, self._chunk_size):
            b = min(a + self._chunk_size, hi)
            yield a, np.asarray(self._clusters[a:b], dtype=np.int64)

    def _scan_clusters(self):
        """Read all of ``clusters`` one time.

        Returns
        -------
        found, counts, first_pos, last_pos : ndarray
            The keys in ``clusters`` (sorted), the number of spikes of each
            key, and the position of its first and last spike.
        """
        parts = []
        for a, c in self._chunks(0, self._n):
            u, i_first, n = np.unique(c, return_index=True, return_counts=True)
            _, i_last = np.unique(c[::-1], return_index=True)
            parts.append((u, n, a + i_first, a + len(c) - 1 - i_last))
        if not parts:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, empty

        u, n, first, last = (np.concatenate(p) for p in zip(*parts))
        found = np.unique(u)
        idx = np.searchsorted(found, u)
        counts = np.zeros(len(found), dtype=np.int64)
        np.add.at(counts, idx, n)
        first_pos = np.full(len(found), self._n, dtype=np.int64)
        np.minimum.at(first_pos, idx, first)
        last_pos = np.full(len(found), -1, dtype=np.int64)
        np.maximum.at(last_pos, idx, last)
        return found, counts, first_pos, last_pos

    def _read_points(self, positions):
        """Read the times at ``positions``. h5py needs unique and increasing
        positions for a point selection."""
        if len(positions) == 0:
            return np.zeros(0)
        unique, inverse = np.unique(positions, return_inverse=True)
        return np.asarray(self._times[unique], dtype=np.float64)[inverse]

    def _ordered(self, times, clusters):
        """Sort the spikes by time and then by key, only if necessary.

        A slice of ``times`` is already sorted. A sort is necessary if the
        times are not sorted in the file, or if equal times are not in the
        order of the keys.
        """
        dt = np.diff(times)
        if np.any(dt < 0):
            warnings.warn(
                "Spike times are not sorted in the file. The lazy TsGroup "
                "assumes sorted spike times. Thus its time support, its rates "
                "and its time windows can be wrong. Load the spike times in "
                "memory instead.",
                stacklevel=4,
            )
        elif not np.any((dt == 0) & (clusters[1:] < clusters[:-1])):
            return times, clusters
        order = np.lexsort((clusters, times))
        return times[order], clusters[order]

    #################################
    # Reads for LazyTsGroup
    #################################

    def read(self, keys=None, start=-np.inf, end=np.inf):
        """Read the spike times of the units ``keys``.

        The function reads only the spikes ``t`` with ``start <= t <= end``
        (in seconds). Two binary searches in ``times`` find these spikes.

        Parameters
        ----------
        keys : array-like of int, optional
            Sorted keys of the source. The default is all the keys.
        start, end : float, optional
            The time window, in seconds. The default is all the time.

        Returns
        -------
        times, clusters : ndarray
            The spike times, sorted by time and then by key, and the key of
            each spike.
        """
        if self.closed:
            raise RuntimeError(_CLOSED_MESSAGE)
        empty = np.zeros(0), np.zeros(0, dtype=np.int64)
        if start > end or (keys is not None and len(keys) == 0):
            return empty
        try:
            lo = (
                0
                if start == -np.inf
                else _search(self._times, 0, self._n, start, False)
            )
            hi = (
                self._n
                if end == np.inf
                else _search(self._times, 0, self._n, end, True)
            )
            if keys is None or np.array_equal(keys, self.keys):
                times = np.asarray(self._times[lo:hi], dtype=np.float64)
                clusters = np.asarray(self._clusters[lo:hi], dtype=np.int64)
            else:
                times, clusters = self._read_units(np.asarray(keys), lo, hi)
        except (ValueError, OSError, KeyError) as err:
            raise RuntimeError("Cannot read the spike times from the file.") from err
        return self._ordered(times, clusters)

    def _read_units(self, keys, lo, hi):
        """Read the spikes of the units ``keys`` in ``[lo, hi)``. Read ``times``
        only in the chunks that hold selected spikes."""
        parts_t, parts_c = [], []
        for a, c in self._chunks(lo, hi):
            mask = np.isin(c, keys)
            if np.any(mask):
                t = np.asarray(self._times[a : a + len(c)], dtype=np.float64)
                parts_t.append(t[mask])
                parts_c.append(c[mask])
        if not parts_t:
            return np.zeros(0), np.zeros(0, dtype=np.int64)
        return np.concatenate(parts_t), np.concatenate(parts_c)

    def read_closest(self, t):
        """Read the spikes around ``t`` (in seconds) in each unit: the last
        spike before ``t`` and the first spike after ``t``.

        The closest spike of each unit is one of these two spikes. The
        function reads ``clusters`` in a window around ``t``. The window grows
        until it holds these two spikes for each unit. Then the function reads
        only the times of these spikes.

        Returns
        -------
        times, clusters : ndarray
            As ``read`` returns them.
        """
        if self.closed:
            raise RuntimeError(_CLOSED_MESSAGE)
        try:
            p = _search(self._times, 0, self._n, t, False)
            before, after = self._window_around(p)
            # The last spike of each unit before `p`, and its first spike from
            # `p`.
            k_before, i_before = np.unique(before[::-1], return_index=True)
            k_after, i_after = np.unique(after, return_index=True)
            positions = np.concatenate([p - 1 - i_before, p + i_after])
            clusters = np.concatenate([k_before, k_after])
            order = np.argsort(positions)
            positions, clusters = positions[order], clusters[order]
            times = self._read_points(positions)
        except (ValueError, OSError, KeyError) as err:
            raise RuntimeError("Cannot read the spike times from the file.") from err
        return self._ordered(times, clusters)

    def _window_around(self, p):
        """Read ``clusters`` before and after position ``p``. The window grows
        until it holds a spike before ``p`` of each unit that has one, and a
        spike from ``p`` of each unit that has one."""
        need_before = self.keys[(self._first_pos >= 0) & (self._first_pos < p)]
        need_after = self.keys[self._last_pos >= p]
        width = self._chunk_size
        while True:
            lo, hi = max(0, p - width), min(self._n, p + width)
            c = np.asarray(self._clusters[lo:hi], dtype=np.int64)
            before, after = c[: p - lo], c[p - lo :]
            done = np.all(np.isin(need_before, before)) and np.all(
                np.isin(need_after, after)
            )
            if done or (lo == 0 and hi == self._n):
                return before, after
            width *= 4


class LazyTsGroup(TsGroup):
    """TsGroup that keeps its spike times on disk. It reads them only when an
    operation needs them, and it does not keep them in memory.

    Make a LazyTsGroup with ``TsGroup.from_ragged_arrays`` or
    ``TsGroup.from_sorted_arrays``. ``nap.load_file`` also gives a LazyTsGroup
    for the units table of an NWB file.

    A source reads the spike times from the disk (see ``_RaggedArraySource``
    and ``_SortedArraySource``). The constructor uses only the keys, the
    counts and the time support of the source. These give the index, the
    rates and the time support of the group.

    Each operation reads only the spike times that it needs into a temporary
    regular TsGroup, and runs on that group:

    - A selection of units (``units[k]``, ``units[[k1, k2]]``, ``getby_*``)
      reads only the selected units.
    - ``restrict(ep)``, ``get(start, end)`` and the operations with an epoch
      argument (``count``, ``value_from``, ``time_diff``) read only the spikes
      from the start of the first epoch to the end of the last epoch.
    - The other operations (``to_tsd``, ``count`` without ``ep``, iteration,
      ``copy``, ...) read all the spikes. They drop the spikes after the
      operation.

    A selection gives a regular TsGroup, not a lazy TsGroup.

    After ``source.close()``, an operation that reads spikes raises a
    RuntimeError.
    """

    def __init__(self, source, metadata=None):
        """
        Parameters
        ----------
        source : _RaggedArraySource or _SortedArraySource
            The source of the spike times.
        metadata : dict, optional
            The metadata columns, in the order of ``source.keys``.
        """
        self.__dict__["_initialized"] = False

        self.index = source.keys
        _MetadataMixin.__init__(self)
        self.time_support = source.time_support

        self._source = source
        # The group holds spike times only: no values and no column names.
        self._data = None
        self._is_tsd = np.zeros(len(source.keys), dtype=bool)
        self._columns = None

        self._finalize(metadata)

    def _unit_counts(self):
        # The source gives the counts. Thus the rates need no spike read.
        return self._source.counts

    #################################
    # Read from the source
    #################################

    def _group(self, keys, times, clusters):
        """Make a regular TsGroup with the units ``keys`` and the spikes
        ``times`` and ``clusters``. The function restricts the spikes to the
        time support of this group."""
        ts = self.time_support
        times, clusters = _restrict_arrays(times, ts.start, ts.end, clusters)
        return TsGroup._from_arrays(
            times,
            clusters,
            None,
            np.zeros(len(keys), dtype=bool),
            keys,
            ts,
            metadata=self._metadata.loc[keys].copy().drop("rate"),
        )

    def _read(self, keys=None, start=-np.inf, end=np.inf):
        """Read the units ``keys`` (default: all) into a regular TsGroup.

        The function reads only the spikes ``t`` with ``start <= t <= end``
        (in seconds).
        """
        if keys is None:
            keys = self.index
        keys = np.unique(np.asarray(keys, dtype=np.int64))
        times, clusters = self._source.read(keys, start, end)
        return self._group(keys, times, clusters)

    def _loaded(self, ep=None):
        """Read the spikes in the span of ``ep`` into a regular TsGroup. If
        ``ep`` is None, read all the spikes."""
        if ep is None:
            return self._read()
        if not isinstance(ep, IntervalSet):
            # Read no spike. The method of the regular group then raises its
            # error.
            return self._read(keys=[])
        if len(ep) == 0:
            return self._read(start=np.inf, end=-np.inf)
        return self._read(start=ep.start[0], end=ep.end[-1])

    #################################
    # Selection of units
    #################################

    def _take(self, keys):
        return self._read(keys)

    def _get_member(self, key):
        return self._read([key])._get_member(key)

    def _members(self):
        return self._read()._members()

    #################################
    # Selection of time
    #################################

    def restrict(self, ep):
        return self._loaded(ep).restrict(ep)

    def get(self, start, end=None, time_units="s"):
        for name, value in (("start", start), ("end", end)):
            if value is not None and not isinstance(value, Number):
                raise ValueError(
                    f"'{name}' must be an int or a float. Type {type(value)} "
                    "provided instead!"
                )
        if end is not None:
            s, e = TsIndex.format_timestamps(np.array([start, end]), time_units)
            return self._read(start=s, end=e).get(start, end, time_units)

        # The closest spike of each unit is one of the two spikes around
        # `start`.
        t = TsIndex.format_timestamps(np.array([start]), time_units)[0]
        times, clusters = self._source.read_closest(t)
        return self._group(self.index, times, clusters).get(start, end, time_units)

    def count(self, bin_size=None, ep=None, time_units="s", dtype=None):
        return self._loaded(ep).count(bin_size, ep, time_units, dtype)

    def value_from(self, tsd, ep=None, mode="closest"):
        # Without `ep`, `value_from` uses the time support of `tsd`.
        window = ep if ep is not None else getattr(tsd, "time_support", None)
        group = self._read(keys=[]) if window is None else self._loaded(window)
        return group.value_from(tsd, ep, mode)

    def time_diff(self, align="center", epochs=None):
        return self._loaded(epochs).time_diff(align, epochs)

    #################################
    # Operations on all the spikes
    #################################

    def to_tsd(self, *args):
        return self._read().to_tsd(*args)

    def subsample(self, fraction, seed=None):
        return self._read().subsample(fraction, seed)

    def save(self, filename):
        return self._read().save(filename)

    def merge(self, *tsgroups, **kwargs):
        return self._read().merge(*tsgroups, **kwargs)

    def __eq__(self, other):
        return self._read() == other

    __hash__ = None

    def __reduce_ex__(self, protocol):
        # Pickle and deepcopy cannot copy a file dataset. Thus give a regular
        # TsGroup with the spikes.
        return (_tsgroup_from_state, (self._read().__getstate__(),))

    # Some code reads the merged arrays directly. These properties read all
    # the spikes for each access, and do not keep them. The code in pynapple
    # calls `_loaded` first.
    @property
    def _times(self):
        return self._read()._times

    @property
    def _clusters(self):
        return self._read()._clusters

    @property
    def _cluster_positions(self):
        return self._read()._cluster_positions

    @property
    def _ragged_index(self):
        return self._read()._ragged_index
