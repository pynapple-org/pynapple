"""

The class `TsGroup` helps group objects with different timestamps
(i.e. timestamps of spikes of a population of neurons).

"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from functools import cached_property
from numbers import Number
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy
import numpy as np
import pandas as pd
from tabulate import tabulate

from ._core_functions import (
    _count,
    _count_clusters,
    _group_by_unit,
    _is_dense_index,
    _restrict_arrays,
    _time_diff_grouped,
    _value_from,
)
from ._jitted_functions import jitunion, jitunion_isets
from .base_class import _Base
from .config import nap_config
from .interval_set import IntervalSet
from .metadata_class import (
    _Metadata,
    _MetadataMixin,
    add_meta_docstring,
    add_or_convert_metadata,
)
from .time_index import TsIndex, trusted_construction
from .time_series import Ts, Tsd, TsdFrame, TsdTensor, _BaseTsd, is_array_like
from .utils import (
    _convert_iter_to_str,
    _get_terminal_size,
    check_filename,
    convert_to_numpy_array,
)


def _union_intervals(i_sets):
    """
    Helper to merge intervals from ts_group
    """
    n = len(i_sets)

    if n == 1:
        return i_sets[0]

    new_start = np.zeros(0)
    new_end = np.zeros(0)

    if n == 2:
        new_start, new_end = jitunion(
            i_sets[0].start,
            i_sets[0].end,
            i_sets[1].start,
            i_sets[1].end,
        )

    if n > 2:
        sizes = np.array([i_sets[i].shape[0] for i in range(n)])
        startends = np.zeros((np.sum(sizes), 2))
        ct = 0
        for i in range(sizes.shape[0]):
            startends[ct : ct + sizes[i], :] = i_sets[i].values
            ct += sizes[i]
        new_start, new_end = jitunion_isets(startends[:, 0], startends[:, 1])

    return IntervalSet(new_start, new_end)


def _concat_values(parts, lengths):
    """Concatenate per-part values into one array aligned with merged times.

    Parameters
    ----------
    parts : list of array-like or None
        Values of each part (a member, or a whole group); None for parts
        without values.
    lengths : list of int
        Number of timestamps of each part.

    Returns
    -------
    ndarray or None
        Values cast to the common dtype of the valued parts, the timestamps of
        parts without values filled with NaN (0 for non-floating dtypes). None
        when no part holds values.

    Raises
    ------
    ValueError
        If valued parts do not share the same shape after the time axis.
    """
    valued = [np.asarray(p) for p in parts if p is not None]
    if not valued:
        return None
    trailing = {a.shape[1:] for a in valued}
    if len(trailing) > 1:
        raise ValueError(
            "All Tsd, TsdFrame or TsdTensor objects in a TsGroup must have the "
            f"same shape after the time axis. Got shapes {sorted(trailing)}."
        )
    dtype = np.result_type(*[a.dtype for a in valued])
    fill = np.nan if np.issubdtype(dtype, np.inexact) else 0
    values = np.empty((int(np.sum(lengths)),) + valued[0].shape[1:], dtype=dtype)
    valued = iter(valued)
    pos = 0
    for p, n in zip(parts, lengths):
        values[pos : pos + n] = fill if p is None else next(valued)
        pos += n
    return values


def _build_sorted_arrays(data):
    """Merge per-unit Ts/Tsd objects into a TsGroup's flat sorted arrays.

    Parameters
    ----------
    data : dict
        ``{key: Ts/Tsd/TsdFrame/TsdTensor}``, keys being integers.

    Returns
    -------
    times : ndarray[float64]
        Every unit's timestamps merged into one sorted array. Ties keep the
        dict order (stable sort), and each unit keeps its own timestamp order.
    clusters : ndarray[int64]
        The key of the unit each timestamp belongs to.
    values : ndarray or None
        Values aligned with ``times`` (shape ``(n, *trailing)``), cast to the
        common dtype of the valued members. Timestamps of bare ``Ts`` members are
        filled with NaN (0 for non-floating dtypes) and flagged by ``is_tsd``.
        None when no member holds values.
    is_tsd : ndarray[bool]
        Per unit (dict order), whether the member holds values.

    Raises
    ------
    ValueError
        If valued members do not share the same shape after the time axis.
    """
    keys = np.fromiter(data.keys(), dtype=np.int64, count=len(data))
    lengths = np.array([len(m) for m in data.values()], dtype=np.int64)
    is_tsd = np.array([isinstance(m, _BaseTsd) for m in data.values()], dtype=bool)

    if len(data):
        times = np.concatenate([m.index.values for m in data.values()]).astype(
            np.float64, copy=False
        )
    else:
        times = np.empty(0, dtype=np.float64)
    clusters = np.repeat(keys, lengths)

    values = _concat_values(
        [m.values if v else None for m, v in zip(data.values(), is_tsd)], lengths
    )

    if len(times) > 1 and np.any(times[1:] < times[:-1]):
        order = np.argsort(times, kind="stable")
        times = times[order]
        clusters = clusters[order]
        if values is not None:
            values = values[order]

    return times, clusters, values, is_tsd


def _shared_columns(members):
    """Column names shared by every TsdFrame among ``members``.

    The merged layout stores one set of column names for the whole group. None
    when there is no TsdFrame, or (with a warning) when their columns differ.
    """
    frames = [m for m in members if isinstance(m, TsdFrame)]
    if not frames:
        return None
    columns = frames[0].columns
    if all(np.array_equal(f.columns, columns) for f in frames[1:]):
        return columns
    warnings.warn(
        "TsdFrame objects in a TsGroup have different columns: "
        "columns are reset to default.",
        stacklevel=3,
    )
    return None


# A member of a TsGroup, as returned by `tsgroup[key]`
_Member = Union[Ts, Tsd, TsdFrame, TsdTensor]
_TimeUnits = Literal["s", "ms", "us"]


class _TsGroupDictView(Mapping):
    """Read-only ``key -> Ts/Tsd`` view of a TsGroup.

    Stands in for the dict ``TsGroup.data`` used to be: members are built from
    the group's merged arrays on access, never stored.
    """

    def __init__(self, group):
        self._group = group

    def __getitem__(self, key):
        if key not in self._group:
            raise KeyError(key)
        return self._group._get_member(key)

    def __iter__(self):
        return iter(self._group.index.tolist())

    def __len__(self):
        return len(self._group.index)

    def __contains__(self, key):
        return key in self._group

    def __repr__(self):
        """One line for each element: its key, its type and its number of
        timestamps. The repr does not build the elements. Thus it stays fast
        for a large group, and it does not read the spike times of a lazy NWB
        group. A group with more than 10 elements shows the first 5 and the
        last 5."""
        group = self._group
        counts = group._unit_counts()
        if group._data is None:
            kinds = ["Ts"] * len(group.index)
        else:
            kind = {1: "Tsd", 2: "TsdFrame"}.get(group._data.ndim, "TsdTensor")
            kinds = [kind if is_tsd else "Ts" for is_tsd in group._is_tsd]
        rows = [
            f"  {key}: {kind}, {n} timestamp{'' if n == 1 else 's'}"
            for key, kind, n in zip(group.index.tolist(), kinds, counts)
        ]
        if len(rows) > 10:
            rows = rows[:5] + ["  ..."] + rows[-5:]
        n_elements = len(group.index)
        header = (
            f"{type(self).__name__}: {n_elements} "
            f"element{'' if n_elements == 1 else 's'} (read-only)"
        )
        return "\n".join([header] + rows)


class TsGroup(_MetadataMixin):
    """
    A group of timestamp objects with different timestamps, e.g. the spike
    times of a population of neurons.

    Each element of the group is a Ts, Tsd, TsdFrame or TsdTensor with an
    integer key. The group works like a dictionary: ``tsgroup[key]`` gives
    one element. All the elements share one time support. Each element has
    one row of metadata. The ``rate`` metadata column is always present.

    Parameters
    ----------
    data : dict or iterable
        The elements: Ts, Tsd, TsdFrame or TsdTensor objects.

        - dict: the keys must be integers, or values that convert to an
          integer (e.g. ``"2"``). The group sorts the keys.
        - Other iterable: the keys are ``0..n_elements-1``.

        An element can also be a list or a numpy.ndarray of timestamps. The
        group then makes a Ts with ``time_units``, and gives a warning.
    time_support : IntervalSet, optional
        The time support of the group. The group drops the timestamps
        outside this time support. If None (default), the time support is
        the union of the time supports of the elements.
    time_units : {"s", "ms", "us"}, optional
        The time unit of the elements given as a list or a numpy.ndarray.
        The default is ``"s"``.
    metadata : pandas.DataFrame or dict, optional
        One row of metadata for each element, in the order of ``data``.
        The column names come from the DataFrame columns or the dict keys.
        The index of a DataFrame must be the keys of the group.

    Raises
    ------
    TypeError
        - If ``time_support`` is not an IntervalSet.
        - If an element is not a Ts, Tsd, TsdFrame, TsdTensor, list or
          numpy.ndarray.
    ValueError
        - If ``time_units`` is not ``"s"``, ``"ms"`` or ``"us"``.
        - If a key does not convert to an integer, or has a decimal part
          (e.g. ``1.5``).
        - If two keys convert to the same integer (e.g. ``1`` and ``"1"``).
        - If the index of the metadata DataFrame is not the keys of the
          group.
        - If the Tsd, TsdFrame or TsdTensor elements do not have the same
          shape after the time axis.
    RuntimeError
        If ``time_support`` is None and the union of the time supports of
        the elements is empty. For example, each element has only one
        timestamp.

    Notes
    -----
    The group does not store one object for each element. It stores all the
    timestamps in one sorted array, with the key of the element of each
    timestamp, and the value of each timestamp for Tsd, TsdFrame and
    TsdTensor elements. Operations on the whole group (e.g. ``count``,
    ``restrict``, ``value_from``) thus go through this array one time, not
    one time for each element. This has these results:

    - ``tsgroup[key]``, ``values()``, ``items()`` and iteration make a new
      Ts, Tsd, TsdFrame or TsdTensor for each call.
    - All the values are in one array, with one dtype. The group casts the
      values of all the elements to a common dtype.
    - All the TsdFrame elements share one set of column names. If the
      column names are different, the group gives a warning and uses the
      default column names.
    - In a group with values, a Ts element stays a Ts.
    - When several elements have the same timestamp, the order of these
      timestamps follows the keys.

    Examples
    --------
    Make a TsGroup from a dict of Ts objects:

    >>> import pynapple as nap
    >>> import numpy as np
    >>> import pandas as pd
    >>> data = {
    ...    0: nap.Ts(np.arange(100)),
    ...    1: nap.Ts(np.arange(0, 100, 2)),
    ...    2: nap.Ts(np.arange(0, 100, 3)),
    ... }
    >>> tsgroup = nap.TsGroup(data)
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Make the same TsGroup from a list. The keys are 0, 1 and 2:

    >>> tsgroup = nap.TsGroup(list(data.values()))
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Make the same TsGroup from a list of numpy arrays. The group makes one
    Ts for each array, and gives a UserWarning:

    >>> tsgroup = nap.TsGroup([np.arange(100), np.arange(0, 100, 2), np.arange(0, 100, 3)])
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Add metadata with a dict:

    >>> tsgroup = nap.TsGroup(data, metadata={"label": ["A", "B", "C"]})
    >>> tsgroup
      Index     rate  label
    -------  -------  -------
          0  1.0101   A
          1  0.50505  B
          2  0.34343  C

    Add metadata with a pandas DataFrame:

    >>> metadata = pd.DataFrame(data=["A", "B", "C"], columns=["label"])
    >>> tsgroup = nap.TsGroup(data, metadata=metadata)
    >>> tsgroup
      Index     rate  label
    -------  -------  -------
          0  1.0101   A
          1  0.50505  B
          2  0.34343  C

    Get one element with its key:

    >>> tsgroup[1]
    Time (s)
    0.0
    2.0
    4.0
    6.0
    8.0
    ...
    90.0
    92.0
    94.0
    96.0
    98.0
    shape: 50
    """

    index: np.ndarray
    """The keys of the elements, sorted."""

    time_support: IntervalSet
    """The time support of the group. All the elements share it."""

    nap_class: str
    """The pynapple class name"""

    # merged sorted arrays backing the group (see `_set_arrays`)
    _times: np.ndarray
    _clusters: np.ndarray
    _data: Optional[np.ndarray]
    _is_tsd: np.ndarray
    _columns: Optional[np.ndarray]

    def __init__(
        self,
        data: Union[Mapping[Any, Any], Iterable[Any]],
        time_support: Optional[IntervalSet] = None,
        time_units: _TimeUnits = "s",
        metadata: Optional[Union[pd.DataFrame, dict]] = None,
    ) -> None:
        # Check input type
        if time_units not in ["s", "ms", "us"]:
            raise ValueError("Argument time_units should be 's', 'ms' or 'us'")
        passed_time_support = False

        if isinstance(time_support, IntervalSet):
            passed_time_support = True
        else:
            if time_support is not None:
                raise TypeError("Argument time_support should be of type IntervalSet")
            else:
                passed_time_support = False

        # set directly in __dict__ to avoid infinite recursion in __setattr__
        self.__dict__["_initialized"] = False

        if not isinstance(data, dict):
            data = dict(enumerate(data))

        # convert all keys to integer
        try:
            keys = [int(k) for k in data.keys()]
        except Exception:
            raise ValueError("All keys must be convertible to integer.")

        # check that there were no floats with decimal points in keys.
        # i.e. 0.5 is not a valid key
        if not all(np.allclose(keys[j], float(k)) for j, k in enumerate(data.keys())):
            raise ValueError("All keys must have integer value!}")

        # check that we have the same num of unique keys
        # {"0":val, 0:val} would be a problem...
        if len(keys) != len(np.unique(keys)):
            raise ValueError("Two dictionary keys contain the same integer value!")

        # Re-key data with the integer keys, ordered the same as the index
        order = np.argsort(keys)
        values = list(data.values())
        self.index = np.asarray(keys)[order]
        data = {keys[i]: values[i] for i in order}

        # Also sort metadata if more than one key
        if len(keys) > 1:
            if (metadata is not None) and (len(metadata) > 0):
                # one value per element, checked before the reorder below
                for name, value in metadata.items():
                    if isinstance(value, str) or not hasattr(value, "__len__"):
                        n_values = 1
                    else:
                        n_values = len(value)
                    if n_values != len(keys):
                        raise ValueError(
                            f"Metadata '{name}' must have {len(keys)} values, "
                            f"one for each element. It has {n_values}."
                        )
                if hasattr(metadata, "index") and np.all(metadata.index != keys):
                    # check that index matches before sort if index exists
                    raise ValueError(
                        "Metadata index does not match the index of the TsGroup."
                    )
                metadata = {
                    key: np.array(value)[order] for key, value in metadata.items()
                }

        # initialize metadata
        _MetadataMixin.__init__(self)
        # to test compatibility with pandas
        # self._metadata = pd.DataFrame(index=self.metadata_index)

        # Transform elements to Ts/Tsd objects
        for k in self.index:
            if not isinstance(data[k], _Base):
                if isinstance(data[k], list) or is_array_like(data[k]):
                    warnings.warn(
                        "Elements should not be passed as {}. Default time units is seconds when creating the Ts object.".format(
                            type(data[k])
                        ),
                        stacklevel=2,
                    )
                    data[k] = Ts(
                        t=convert_to_numpy_array(data[k], "key {}".format(k)),
                        time_support=time_support,
                        time_units=time_units,
                    )

        # before the union of time supports, which reads `.time_support`
        for k in self.index:
            if not isinstance(data[k], _Base):
                raise TypeError(
                    f"Element {k} of TsGroup should be a Ts, Tsd, TsdFrame or "
                    f"TsdTensor. {type(data[k])} provided instead."
                )

        if not passed_time_support:
            # Do the union of all time supports
            time_support = _union_intervals([data[k].time_support for k in self.index])
            if len(time_support) == 0:
                raise RuntimeError(
                    "Union of time supports is empty. Consider passing a time support as argument."
                )
        self.time_support = time_support

        # Every unit is merged into one sorted array, restricted once to the
        # time support (instead of once per unit).
        times, clusters, values, is_tsd = _build_sorted_arrays(data)
        times, clusters, values = _restrict_arrays(
            times, time_support.start, time_support.end, clusters, values
        )
        self._set_arrays(
            times, clusters, values, is_tsd, _shared_columns(data.values())
        )

        self._finalize(metadata)

    def _set_arrays(
        self,
        times: np.ndarray,
        clusters: np.ndarray,
        values: Optional[np.ndarray],
        is_tsd: np.ndarray,
        columns: Optional[np.ndarray] = None,
    ) -> None:
        """Set the merged sorted arrays backing the group.

        ``_times`` holds every unit's timestamps in one sorted array,
        ``_clusters`` the cluster of each timestamp (the key of its unit), ``_data`` the
        matching values (None when no member holds values), ``_is_tsd`` which
        units hold values and ``_columns`` the column names of 2-dimensional
        members (None for the default ones).
        """
        self._times = times
        self._clusters = clusters
        self._data = values
        self._is_tsd = np.asarray(is_tsd, dtype=bool)
        self._columns = columns

    def _finalize(self, metadata: Optional[Any] = None) -> None:
        """Compute rates, freeze the object and set metadata."""
        self._metadata["rate"] = self._compute_rates()
        self.nap_class = self.__class__.__name__
        # grab current attributes before adding metadata
        self._class_attributes = self.__dir__()
        self._class_attributes.append("_class_attributes")  # add this property

        # Making the TsGroup non mutable
        self._initialized = True

        self.set_info(metadata)

    def _compute_rates(self) -> np.ndarray:
        duration = self.time_support.tot_length()
        counts = self._unit_counts()
        if duration > 0:
            rates = counts / duration
        else:
            rates = np.full(len(counts), np.nan)
        # read-only, as `set_info` makes the other metadata columns
        rates.setflags(write=False)
        return rates

    def _unit_counts(self) -> np.ndarray:
        """Number of timestamps of each unit, following ``self.index``."""
        return _count_clusters(self._clusters, self.index)

    @cached_property
    def _cluster_positions(self) -> np.ndarray:
        """Map the cluster of each timestamp to its position in ``self.index``.

        ``_clusters`` holds the key of the unit of each timestamp (e.g. 3, 7,
        12). Kernels that output one column per unit (``count``, ``time_diff``,
        ...) need a dense column number in ``0..n_units-1`` instead. With
        ``self.index = [3, 7, 12]``, clusters ``[7, 3, 12, 7]`` map to
        ``[1, 0, 2, 1]``.

        When the keys are dense enough (see ``_is_dense_index``), the mapping
        uses a lookup table indexed by ``key - min(key)``, which is O(n). For
        sparse keys (e.g. ``{0, 10**12}``), where such a table would be too big,
        it uses a binary search in ``self.index`` instead, which is
        O(n log n_units).

        Returns
        -------
        numpy.ndarray
            int64 array with the same length as ``_times``. Computed once and
            cached: ``index`` and ``_clusters`` cannot change once the group
            is built. Not pickled (see ``__getstate__``).
        """
        index = np.asarray(self.index, dtype=np.int64)
        if len(index) == 0:
            return np.zeros(len(self._clusters), dtype=np.int64)
        if not _is_dense_index(index):
            return np.searchsorted(index, self._clusters)
        lo = index[0]
        key_to_position = np.empty(index[-1] - lo + 1, dtype=np.int64)
        key_to_position[index - lo] = np.arange(len(index))
        return key_to_position[self._clusters - lo]

    @classmethod
    def _from_arrays(
        cls,
        times: np.ndarray,
        clusters: np.ndarray,
        values: Optional[np.ndarray],
        is_tsd: np.ndarray,
        index: np.ndarray,
        time_support: IntervalSet,
        metadata: Optional[Any] = None,
        columns: Optional[np.ndarray] = None,
    ) -> TsGroup:
        """Build a TsGroup directly from its merged arrays, without validation.

        Bypasses ``__init__``, which only builds from a dict of per-unit objects:
        callers already holding merged arrays would otherwise split them into
        one Ts per unit just to have them merged again.

        ``times`` must be sorted, ``clusters`` hold keys of ``index``, and
        ``index`` be sorted; ``metadata`` rows must follow ``index``. ``columns``
        names the columns of 2-dimensional members.
        """
        obj = cls.__new__(cls)
        obj.__dict__["_initialized"] = False
        obj.index = np.asarray(index, dtype=np.int64)
        _MetadataMixin.__init__(obj)
        obj.time_support = time_support
        obj._set_arrays(times, clusters, values, is_tsd, columns)
        obj._finalize(metadata)
        return obj

    @cached_property
    def _ragged_index(self) -> tuple[np.ndarray, np.ndarray]:
        """``(order, offsets)`` such that ``order[offsets[i]:offsets[i + 1]]``
        are the positions in ``_times`` of unit ``self.index[i]``, in time order.

        Computed once on first per-unit access and cached, so that iterating
        over members costs one counting sort overall rather than a full scan per
        member.
        """
        return _group_by_unit(self._cluster_positions, len(self.index))

    @cached_property
    def _keys_set(self) -> set[int]:
        """Keys as a set, for O(1) membership tests."""
        return set(self.index.tolist())

    def _get_member(self, key: int) -> _Member:
        """Return the object of the unit ``key``, as ``tsgroup[key]`` does.

        Finds the position of ``key`` in ``self.index`` and takes the block of
        that unit from ``_ragged_index``. The type of the object depends on
        the stored values:

        - ``Ts`` if the unit holds no values (``_is_tsd`` is False),
        - ``Tsd``, ``TsdFrame`` or ``TsdTensor`` if ``_data`` is 1-, 2- or
          N-dimensional. A ``TsdFrame`` gets the shared ``_columns``.

        The object is built on each call (nothing is cached) and shares the
        group's ``time_support``.

        The caller must check that ``key`` is in the group. This method does
        not: for an unknown key it returns a different unit, or fails with an
        IndexError.

        Parameters
        ----------
        key : int
            Key of the unit, a value of ``self.index``.

        Returns
        -------
        Ts, Tsd, TsdFrame or TsdTensor
            The unit ``key``.
        """
        i = np.searchsorted(self.index, key)
        order, offsets = self._ragged_index
        idx = order[offsets[i] : offsets[i + 1]]
        t = self._times[idx]
        # timestamps come sorted out of the merged arrays
        with trusted_construction():
            if self._data is None or not self._is_tsd[i]:
                return Ts(t=t, time_support=self.time_support)
            d = self._data[idx]
            if d.ndim == 1:
                return Tsd(t=t, d=d, time_support=self.time_support)
            if d.ndim == 2:
                return TsdFrame(
                    t=t, d=d, time_support=self.time_support, columns=self._columns
                )
            return TsdTensor(t=t, d=d, time_support=self.time_support)

    def _members(self) -> list[_Member]:
        """All members, in index order."""
        return [self._get_member(k) for k in self.index]

    def _loaded(self, ep: Optional[IntervalSet] = None) -> TsGroup:
        """The group with its spikes in memory, for code that reads the merged
        arrays directly.

        A regular TsGroup returns itself. A lazy group (e.g. the units of an
        NWB file) returns a regular TsGroup with the spikes in ``ep``, or all
        the spikes if ``ep`` is None.
        """
        return self

    def _take(self, keys: Iterable[int]) -> TsGroup:
        """New TsGroup holding only the units ``keys``.

        ``keys`` must be keys of the group. They are sorted and duplicates are
        dropped.
        """
        keys = np.unique(np.asarray(keys, dtype=np.int64))
        mask = np.isin(self._clusters, keys)
        sel = np.searchsorted(self.index, keys)
        return TsGroup._from_arrays(
            self._times[mask],
            self._clusters[mask],
            None if self._data is None else self._data[mask],
            self._is_tsd[sel],
            keys,
            self.time_support,
            metadata=self._metadata.loc[keys].copy().drop("rate"),
            columns=self._columns,
        )

    @property
    def data(self) -> Mapping[int, _Member]:
        """Read-only mapping from each key to its Ts/Tsd, built on access."""
        return _TsGroupDictView(self)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        # derived and O(n_timestamps): recomputed on demand after unpickling
        state.pop("_ragged_index", None)
        state.pop("_cluster_positions", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore a TsGroup from the state saved by ``__getstate__``.

        Called by ``pickle`` and ``copy`` on an empty object (``__init__``
        does not run). Two kinds of state are accepted:

        - State from this version: it holds the merged arrays and is copied
          as is. ``_ragged_index`` and ``_cluster_positions`` are not in it
          and are built again when first needed.
        - State from a version older than the merged arrays: it holds a
          ``data`` dict of Ts/Tsd objects instead. The merged arrays are
          built from that dict with ``_build_sorted_arrays``.

        The attributes are written directly to ``__dict__``. Normal assignment
        cannot be used: once ``_initialized`` is restored, ``__setattr__``
        refuses reserved names and treats other names as metadata.

        Parameters
        ----------
        state : dict
            The saved ``__dict__`` of the TsGroup.
        """
        state = dict(state)
        # objects pickled before the merged-array layout held a dict of members
        members = state.pop("data", None)
        self.__dict__.update(state)
        if "_times" not in state and members is not None:
            members = {k: members[k] for k in self.index}
            times, clusters, values, is_tsd = _build_sorted_arrays(members)
            self.__dict__.update(
                _times=times,
                _clusters=clusters,
                _data=values,
                _is_tsd=is_tsd,
                _columns=_shared_columns(members.values()),
            )
        # unpickled arrays are writable: make the metadata read-only again
        metadata = self.__dict__.get("_metadata")
        if isinstance(metadata, _Metadata):
            for value in metadata.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)

    """
    Base functions
    """

    def __setattr__(self, name: str, value: Any) -> None:
        # necessary setter to allow metadata to be set as an attribute
        if self._initialized:
            if name in self._class_attributes:
                raise AttributeError(
                    f"Cannot set attribute: '{name}' is a reserved attribute. Use 'set_info()' to set '{name}' as metadata."
                )
            else:
                _MetadataMixin.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    @add_or_convert_metadata
    def __getattr__(self, name: str) -> Any:
        # Necessary for backward compatibility with pickle

        # avoid infinite recursion when pickling due to
        # self._metadata.column having attributes '__reduce__', '__reduce_ex__'
        if name in ("__getstate__", "__setstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)

        # try:
        #     metadata = self._metadata
        # except Exception:
        #     metadata = pd.DataFrame(index=self.index)
        metadata = self._metadata

        if name == "_metadata":
            return metadata
        elif name in metadata.columns:
            return _MetadataMixin.__getattr__(self, name)
        else:
            return super().__getattr__(name)

    @add_or_convert_metadata
    def __getitem__(
        self, key: Union[int, str, list, np.ndarray]
    ) -> Union[_Member, TsGroup, pd.Series, pd.DataFrame]:
        # Standard dict keys are Hashable
        if isinstance(key, Hashable):
            if self.__contains__(key):
                return self._get_member(key)
            elif key in self._metadata.columns:
                return _MetadataMixin.__getitem__(self, key)
            else:
                raise KeyError(r"Key {} not in group index.".format(key))
        elif (
            isinstance(key, list) and len(key) and all(isinstance(k, str) for k in key)
        ):
            # index multiple metadata columns
            return _MetadataMixin.__getitem__(self, key)

        # array boolean are transformed into indices
        # note that raw boolean are hashable, and won't be
        # tsd == tsg.to_tsd()
        elif np.asarray(key).dtype == bool:
            key = np.asarray(key)
            if key.ndim != 1:
                raise IndexError("Only 1-dimensional boolean indices are allowed!")
            if len(key) != self.__len__():
                raise IndexError(
                    "Boolean index length must be equal to the number of units in the group! "
                    f"The number of units is {self.__len__()}, but the boolean array"
                    f"has length {len(key)} instead!"
                )
            key = self.index[key]

        keys_not_in = list(filter(lambda x: x not in self.index, key))

        if len(keys_not_in):
            raise KeyError(r"Key {} not in group index.".format(keys_not_in))

        return self._take(key)

    def __eq__(self, other: object) -> bool:
        """Two TsGroup are equal when they hold the same keys, time support,
        timestamps (and values) per key, and metadata."""
        if not isinstance(other, TsGroup):
            return NotImplemented
        if not (
            np.array_equal(self.index, other.index)
            and np.array_equal(self.time_support.values, other.time_support.values)
            and np.array_equal(self._times, other._times)
            and np.array_equal(self._clusters, other._clusters)
            and np.array_equal(self._is_tsd, other._is_tsd)
            and self._metadata == other._metadata
        ):
            return False
        if (self._data is None) != (other._data is None):
            return False
        if self._data is not None and not np.array_equal(
            self._data, other._data, equal_nan=self._data.dtype.kind in "fc"
        ):
            return False
        if (self._columns is None) != (other._columns is None):
            return False
        return self._columns is None or np.array_equal(self._columns, other._columns)

    # Equal groups compare by content, which a hash could not follow: unhashable.
    __hash__ = None

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[int]:
        return iter(self.index.tolist())

    def __contains__(self, key: object) -> bool:
        try:
            return key in self._keys_set
        except TypeError:  # unhashable
            return False

    def __repr__(self) -> str:
        # Start by determining how many columns and rows.
        # This can be unique for each object
        cols, rows = _get_terminal_size()
        max_cols = np.maximum(cols // 12, 5)
        max_rows = np.maximum(rows - 10, 2)

        # By default, the first three columns should always show.
        # Adding an extra column between actual values and metadata
        try:
            col_names = self._metadata.columns
        except Exception:
            # Necessary for backward compatibility when saving IntervalSet as pickle
            col_names = []

        if len(col_names) and "rate" in col_names:
            col_names.remove("rate")

        col_to_show = col_names[0:max_cols]

        headers = ["Index", "rate"] + col_to_show
        end = ["..."] if len(headers) > max_cols else []
        headers += end

        if len(self) == 0:
            return tabulate(tabular_data=[], headers=headers)

        if len(self) > max_rows:
            n_rows = max_rows // 2
            ends = np.array([end] * n_rows)
            if len(col_to_show):
                try:
                    mt_top = np.array(
                        [
                            _convert_iter_to_str(self._metadata[c][0:n_rows])
                            for c in col_to_show
                        ]
                    ).T
                    mt_bot = np.array(
                        [
                            _convert_iter_to_str(self._metadata[c][-n_rows:])
                            for c in col_to_show
                        ]
                    ).T
                except Exception:
                    mt_top = np.ndarray(shape=(n_rows, 0))
                    mt_bot = np.ndarray(shape=(n_rows, 0))
            else:
                mt_top = np.ndarray(shape=(n_rows, 0))
                mt_bot = np.ndarray(shape=(n_rows, 0))

            table = np.vstack(
                (
                    np.hstack(
                        (
                            self.index[0:n_rows, None],
                            np.round(self._metadata["rate"], 5)[0:n_rows, None],
                            mt_top,
                            ends,
                        ),
                        dtype=object,
                    ),
                    np.array(
                        [["..." for _ in range(2 + len(col_to_show))] + end],
                        dtype=object,
                    ),
                    np.hstack(
                        (
                            self.index[-n_rows:, None],
                            np.round(self._metadata["rate"], 5)[-n_rows:, None],
                            mt_bot,
                            ends,
                        ),
                        dtype=object,
                    ),
                )
            )
        else:
            ends = np.array([end] * len(self))
            if len(col_to_show):
                mt = np.array(
                    [_convert_iter_to_str(self._metadata[c]) for c in col_to_show]
                ).T
            else:
                mt = np.ndarray(shape=(len(self), 0))

            table = np.hstack(
                (
                    self.index[:, None],
                    np.round(self._metadata["rate"], 5)[:, None],
                    mt,
                    ends,
                ),
                dtype=object,
            )

        return tabulate(table, headers=headers)

    def __str__(self) -> str:
        # Show all columns and all rows (no truncation).
        try:
            col_names = self._metadata.columns
        except Exception:
            col_names = []

        if len(col_names) and "rate" in col_names:
            col_names.remove("rate")

        headers = ["Index", "rate"] + col_names

        if len(self) == 0:
            return tabulate(tabular_data=[], headers=headers)

        if len(col_names):
            try:
                mt = np.array(
                    [_convert_iter_to_str(self._metadata[c]) for c in col_names]
                ).T
            except Exception:
                mt = np.ndarray(shape=(len(self), 0))
        else:
            mt = np.ndarray(shape=(len(self), 0))

        table = np.hstack(
            (
                self.index[:, None],
                np.round(self._metadata["rate"], 5)[:, None],
                mt,
            ),
            dtype=object,
        )

        return tabulate(table, headers=headers)

    def keys(self) -> list[int]:
        """
        Return index/keys of TsGroup

        Returns
        -------
        list
            List of keys
        """
        return self.index.tolist()

    def items(self) -> list[tuple[int, _Member]]:
        """
        Return a list of key/object.

        Returns
        -------
        list
            List of tuples
        """
        return list(zip(self.index.tolist(), self._members()))

    def values(self) -> list[_Member]:
        """
        Return a list of all the Ts/Tsd objects in the TsGroup

        Returns
        -------
        list
            List of Ts/Tsd objects
        """
        return self._members()

    @property
    def rates(self) -> np.ndarray:
        """
        The mean rate of each element, in Hz.

        The rate of an element is its number of timestamps divided by the
        total duration of the time support, in seconds. The values are the
        same as the ``rate`` metadata column. The order of the values follows
        ``tsgroup.index``.

        Each function that returns a new TsGroup (e.g. ``restrict``, ``get``
        or ``subsample``) computes the rates again, on the time support of
        the new TsGroup.

        Returns
        -------
        numpy.ndarray
            One rate for each element.

        Examples
        --------
        >>> import pynapple as nap
        >>> tsgroup = nap.TsGroup(
        ...     {0: nap.Ts(t=[1, 2, 7]), 5: nap.Ts(t=[2, 4])},
        ...     time_support=nap.IntervalSet(start=[0, 6], end=[2, 8]),
        ... )

        The time support lasts 4 seconds. Element 0 has 3 timestamps in the
        time support. Element 5 has 1 timestamp in the time support, because
        the time 4 is outside the time support:

        >>> tsgroup.rates
        array([0.75, 0.25])
        """
        return self._metadata["rate"]

    def copy(self) -> TsGroup:
        """
        Make a deep copy of the TsGroup.

        The copy holds its own copy of the timestamps, values, time support
        and metadata. Thus a change to the copy does not change the original
        TsGroup.

        Returns
        -------
        TsGroup
            A TsGroup equal to the original TsGroup.

        Notes
        -----
        For a TsGroup that ``NWBFile`` loads with lazy loading, the function
        reads the spike times from the file first. The copy is a regular
        TsGroup.

        Examples
        --------
        >>> import pynapple as nap
        >>> tsgroup = nap.TsGroup(
        ...     {0: nap.Ts(t=[1, 2, 3]), 1: nap.Ts(t=[2, 3, 4])},
        ...     metadata={"label": ["a", "b"]},
        ... )
        >>> tsgroup_copy = tsgroup.copy()
        >>> tsgroup_copy == tsgroup
        True

        A change to the metadata of the copy does not change the original:

        >>> tsgroup_copy.set_info(label=["x", "y"])
        >>> tsgroup
          Index    rate  label
        -------  ------  -------
              0       1  a
              1       1  b
        """
        import copy

        return copy.deepcopy(self)

    #################################
    # Groups from arrays on disk
    #################################

    @staticmethod
    def _metadata_in_order(metadata, n_units, order):
        """Check that each metadata column has one value for each unit, then
        put the values in the order ``order``."""
        if metadata is None:
            return None
        columns = {}
        for name, value in metadata.items():
            value = np.asarray(value)
            if value.ndim == 0 or len(value) != n_units:
                n_values = 1 if value.ndim == 0 else len(value)
                raise ValueError(
                    f"Metadata '{name}' must have {n_units} values, one for "
                    f"each unit. It has {n_values}."
                )
            columns[name] = value[order]
        return columns

    @classmethod
    def from_ragged_arrays(
        cls,
        ragged_array: Any,
        ragged_array_index: Any,
        keys: Optional[Iterable[int]] = None,
        metadata: Optional[Union[pd.DataFrame, dict]] = None,
        lazy: bool = True,
    ) -> TsGroup:
        """
        Make a TsGroup from spike times in a ragged layout.

        ``ragged_array`` holds the spike times of each unit, one unit after the
        other. The spike times of each unit must be sorted.
        ``ragged_array_index`` holds the end position of each unit in
        ``ragged_array``. This is the layout of the units table of an NWB file
        (``spike_times`` and ``spike_times_index``).

        The arrays can be numpy arrays, memory-mapped arrays, h5py datasets or
        other array-likes that support slices.

        Parameters
        ----------
        ragged_array : array-like
            The spike times of all the units, in seconds.
        ragged_array_index : array-like of int
            The end position of each unit in ``ragged_array`` (one position
            after its last spike). Unit ``i`` holds
            ``ragged_array[ragged_array_index[i - 1]:ragged_array_index[i]]``.
        keys : array-like of int, optional
            The key of each unit, in the order of ``ragged_array_index``. The
            default is ``0..n_units-1``.
        metadata : dict or pandas.DataFrame, optional
            One value for each unit, in the order of ``ragged_array_index``.
        lazy : bool, optional
            - True (default): return a LazyTsGroup. It keeps the spike times
              on disk and reads them only when an operation needs them.
            - False: read all the spike times and return a regular TsGroup.

        Returns
        -------
        LazyTsGroup or TsGroup
            The time support is the union of the spans of the units. The span
            of a unit goes from its first spike to its last spike.

        Raises
        ------
        ValueError
            - If ``keys`` does not have one key for each unit, or if two keys
              are equal.
            - If ``ragged_array_index`` decreases, or if it goes past the end
              of ``ragged_array``.
            - If a metadata column does not have one value for each unit.

        See Also
        --------
        from_sorted_arrays : Make a TsGroup from spike times in a sorted layout.

        Notes
        -----
        A LazyTsGroup reads only the spike times that each operation needs:

        - A selection of units (``units[[0, 1]]``) reads only these units. It
          is fast.
        - ``restrict(ep)``, ``get(start, end)`` and the operations with an
          epoch argument read only the spikes in the epochs. They do a binary
          search in each unit, with one read for each step. This is fast for
          h5py, but slow for zarr. With many epochs, the read merges the
          epochs that have the smallest gaps between them, so that it reads
          at most 16 time windows.
        - The other operations read all the spike times, each time that they
          run.

        A selection gives a regular TsGroup in memory. The arrays must stay
        readable while the LazyTsGroup is in use (e.g. keep the h5py file
        open).

        Examples
        --------
        >>> import numpy as np
        >>> import pynapple as nap
        >>> ragged_array = np.array([0.5, 1.5, 2.0, 4.0, 1.0, 3.0])
        >>> ragged_array_index = np.array([4, 6])
        >>> units = nap.TsGroup.from_ragged_arrays(
        ...     ragged_array, ragged_array_index, metadata={"area": ["CA1", "PFC"]}
        ... )
        >>> units
          Index     rate  area
        -------  -------  ------
              0  1.14286  CA1
              1  0.57143  PFC

        A selection of units reads only these units:

        >>> units[1]
        Time (s)
        1.0
        3.0
        shape: 2

        With h5py, the arrays stay in the file:

        .. code-block:: python

            import h5py

            f = h5py.File("spikes.h5", "r")
            units = nap.TsGroup.from_ragged_arrays(
                f["spike_times"], f["spike_times_index"][:]
            )
        """
        from .lazy_ts_group import LazyTsGroup, _RaggedArraySource

        n_units = len(ragged_array_index)
        if keys is None:
            keys = np.arange(n_units)
        source = _RaggedArraySource(ragged_array, ragged_array_index, keys)
        metadata = cls._metadata_in_order(
            metadata, n_units, np.argsort(np.asarray(keys), kind="stable")
        )
        group = LazyTsGroup(source, metadata=metadata)
        return group if lazy else group._read()

    @classmethod
    def from_sorted_arrays(
        cls,
        times: Any,
        clusters: Any,
        keys: Optional[Iterable[int]] = None,
        metadata: Optional[Union[pd.DataFrame, dict]] = None,
        lazy: bool = True,
    ) -> TsGroup:
        """
        Make a TsGroup from spike times in a sorted layout.

        ``times`` holds the spike times of all the units, sorted. ``clusters``
        holds the key of each spike. This is the layout of
        ``TsGroup.to_tsd()``: its timestamps are ``times``, and its values are
        the keys.

        The arrays can be numpy arrays, memory-mapped arrays, h5py datasets,
        zarr arrays or other array-likes that support slices.

        Parameters
        ----------
        times : array-like
            The spike times of all the units, sorted, in seconds.
        clusters : array-like of int
            The key of each spike.
        keys : array-like of int, optional
            The keys of the units. Use it to add units with no spike. The
            default is the keys in ``clusters``.
        metadata : dict or pandas.DataFrame, optional
            One value for each unit, in the order of ``keys``. Without
            ``keys``, in the order of the sorted keys in ``clusters``.
        lazy : bool, optional
            - True (default): return a LazyTsGroup. It keeps the spike times
              on disk and reads them only when an operation needs them.
            - False: read all the spike times and return a regular TsGroup.

        Returns
        -------
        LazyTsGroup or TsGroup
            The time support is the union of the spans of the units. The span
            of a unit goes from its first spike to its last spike.

        Raises
        ------
        ValueError
            - If ``times`` and ``clusters`` do not have the same length.
            - If two keys are equal, or if ``clusters`` holds a key that is not
              in ``keys``.
            - If a metadata column does not have one value for each unit.

        See Also
        --------
        from_ragged_arrays : Make a TsGroup from spike times in a ragged layout.

        Notes
        -----
        The construction reads all of ``clusters`` one time, to find the keys
        and the number of spikes of each unit. It reads only some values of
        ``times``.

        A LazyTsGroup reads only the spike times that each operation needs:

        - ``restrict(ep)``, ``get(start, end)`` and the operations with an
          epoch argument read only the spikes in the epochs. For each epoch,
          they do two binary searches in ``times``, then read one slice. This
          is fast. With many epochs, the read merges the epochs that have the
          smallest gaps between them, so that it reads at most 256 time
          windows.
        - A selection of units (``units[[0, 1]]``) reads all of ``clusters``,
          and ``times`` only where the selected units have spikes.
        - The other operations read all the spike times, each time that they
          run.

        A selection gives a regular TsGroup in memory. The arrays must stay
        readable while the LazyTsGroup is in use (e.g. keep the h5py file
        open).

        Examples
        --------
        >>> import numpy as np
        >>> import pynapple as nap
        >>> times = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        >>> clusters = np.array([0, 1, 0, 0, 1, 0])
        >>> units = nap.TsGroup.from_sorted_arrays(
        ...     times, clusters, metadata={"area": ["CA1", "PFC"]}
        ... )
        >>> units
          Index     rate  area
        -------  -------  ------
              0  1.14286  CA1
              1  0.57143  PFC

        A selection of time reads only the spikes in that time:

        >>> units.restrict(nap.IntervalSet(1, 2))
          Index    rate  area
        -------  ------  ------
              0       2  CA1
              1       1  PFC

        Save a TsGroup in zarr, then make a lazy TsGroup from the saved
        arrays:

        .. code-block:: python

            import zarr

            tsd = tsgroup.to_tsd()
            root = zarr.open("spikes.zarr", mode="w")
            root["times"] = tsd.t
            root["clusters"] = tsd.values.astype(np.int64)

            root = zarr.open("spikes.zarr", mode="r")
            units = nap.TsGroup.from_sorted_arrays(root["times"], root["clusters"])
        """
        from .lazy_ts_group import LazyTsGroup, _SortedArraySource

        source = _SortedArraySource(times, clusters, keys=keys)
        # The metadata follow `keys`, or the sorted keys in `clusters`.
        if keys is None:
            order = np.arange(len(source.keys))
        else:
            order = np.argsort(np.asarray(keys), kind="stable")
        metadata = cls._metadata_in_order(metadata, len(source.keys), order)
        group = LazyTsGroup(source, metadata=metadata)
        return group if lazy else group._read()

    #################################
    # Generic functions of Tsd objects
    #################################
    def restrict(self, ep: IntervalSet) -> TsGroup:
        """
        Keep the timestamps of each element that are in the epochs of ``ep``.

        The result keeps each timestamp ``t`` with ``start <= t <= end`` for
        an epoch of ``ep``. Tsd, TsdFrame and TsdTensor elements keep the
        values of their kept timestamps.

        Parameters
        ----------
        ep : IntervalSet
            The epochs.

        Returns
        -------
        TsGroup
            A TsGroup with the same keys and metadata. Its time support is
            ``ep``, also if ``ep`` is larger than the time support of the
            original TsGroup. An element with no timestamp in ``ep`` is
            empty. The function computes the ``rate`` metadata again, on the
            total duration of ``ep``.

        Raises
        ------
        TypeError
            If ``ep`` is not an IntervalSet.

        See Also
        --------
        get : Keep the timestamps between two times, with the same time
            support.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')
        >>> newtsgroup = tsgroup.restrict(ep)

        The time support of the result is ``ep``. Each element of the result
        also has ``ep`` as its time support:

        >>> newtsgroup.time_support
          index    start    end
              0        0    100
        shape: (1, 2), time unit: sec.
        >>> newtsgroup[0].time_support
          index    start    end
              0        0    100
        shape: (1, 2), time unit: sec.

        The rates use the 100 seconds of ``ep``. Each epoch includes its end,
        thus element 0 keeps 101 timestamps:

        >>> newtsgroup
          Index    rate
        -------  ------
              0    1.01
              1    2.01
              2    4.01
        """
        if not isinstance(ep, IntervalSet):
            raise TypeError("Argument should be IntervalSet")
        times, clusters, values = _restrict_arrays(
            self._times, ep.start, ep.end, self._clusters, self._data
        )
        cols = self._metadata.columns[1:]  # .drop("rate")

        return TsGroup._from_arrays(
            times,
            clusters,
            values,
            self._is_tsd,
            self.index,
            ep,
            metadata=self._metadata[cols],
            columns=self._columns,
        )

    def value_from(
        self,
        tsd: Union[Tsd, TsdFrame, TsdTensor],
        ep: Optional[IntervalSet] = None,
        mode: Literal["closest", "before", "after"] = "closest",
    ) -> TsGroup:
        """
        Give each timestamp of the group a value taken from ``tsd``.

        For every timestamp of every unit, the matching sample of ``tsd`` is
        found and its value is assigned to the timestamp. A typical use is to
        get the position of the animal at each spike.

        The match uses only the samples of ``tsd`` in the same epoch of
        ``ep`` as the timestamp. A timestamp with no matching sample in its
        epoch gets NaN (integer values are then converted to float).

        The returned TsGroup:

        - holds only the timestamps inside ``ep``,
        - has ``ep`` as its time support, with the rates computed again on it,
        - holds members of the same type as ``tsd``: ``Tsd``, ``TsdFrame``
          (with the columns of ``tsd``) or ``TsdTensor``. Values held before
          by the group are replaced.
        - keeps the keys and the metadata of the group.

        Parameters
        ----------
        tsd : Tsd, TsdFrame or TsdTensor
            The object that holds the values to assign.
        ep : IntervalSet, optional
            The epochs in which the timestamps are kept and matched. If None,
            the time support of ``tsd`` is used.
        mode : {'closest', 'before', 'after'}, optional
            How a timestamp is matched to a sample of ``tsd``:

            - ``'closest'`` (default): the nearest sample.
            - ``'before'``: the last sample at or before the timestamp.
            - ``'after'``: the first sample at or after the timestamp.

        Returns
        -------
        TsGroup
            A new TsGroup whose members hold the values from ``tsd``.

        Raises
        ------
        TypeError
            If ``tsd`` is not a Tsd, TsdFrame or TsdTensor, or if ``ep`` is not
            an IntervalSet.
        ValueError
            If ``mode`` is not 'closest', 'before' or 'after'.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsgroup = nap.TsGroup({0: nap.Ts(t=[1.0, 2.5, 6.0]), 3: nap.Ts(t=[2.0, 4.6])})

        ``tsd`` holds the values to assign, for example the position of the
        animal sampled every second:

        >>> tsd = nap.Tsd(t=np.arange(0.0, 6.0), d=np.arange(0.0, 60.0, 10.0))
        >>> newtsgroup = tsgroup.value_from(tsd)
        >>> newtsgroup[0]
        Time (s)
        ----------  --
        1           10
        2.5         30
        dtype: float64, shape: (2,)

        The timestamp at 6.0 s is outside the time support of ``tsd`` and is
        dropped. With ``mode="before"``, 2.5 s takes the sample at 2 s:

        >>> tsgroup.value_from(tsd, mode="before")[0]
        Time (s)
        ----------  --
        1           10
        2.5         20
        dtype: float64, shape: (2,)

        """
        if not isinstance(tsd, _BaseTsd):
            raise TypeError(
                "First argument should be an instance of Tsd, TsdFrame or TsdTensor"
            )
        if ep is None:
            ep = tsd.time_support
        if not isinstance(ep, IntervalSet):
            raise TypeError("Argument ep should be of type IntervalSet or None")
        if mode not in ("closest", "before", "after"):
            raise ValueError(
                f"Argument mode should be 'closest', 'before', or 'after'. {mode} provided instead."
            )

        starts = ep.start
        ends = ep.end
        # matching depends only on each timestamp, not on its unit: one pass
        # over the merged array covers every unit
        times, values, clusters = _value_from(
            self._times,
            tsd.index.values,
            tsd.values,
            starts,
            ends,
            self._clusters,
            mode=mode,
        )

        cols = self._metadata.columns[1:]  # .drop("rate")
        return TsGroup._from_arrays(
            times,
            clusters,
            values,
            np.ones(len(self.index), dtype=bool),
            self.index,
            IntervalSet(start=starts, end=ends),
            metadata=self._metadata[cols],
            columns=tsd.columns if isinstance(tsd, TsdFrame) else None,
        )

    @add_or_convert_metadata
    def count(
        self,
        bin_size: Optional[float] = None,
        ep: Optional[IntervalSet] = None,
        time_units: _TimeUnits = "s",
        dtype: Optional[Union[str, type, np.dtype]] = None,
    ) -> TsdFrame:
        """
        Count the timestamps of each unit in time bins.

        There are two ways to define the bins:

        - With ``bin_size``: each epoch of ``ep`` is cut into bins of
          ``bin_size``, starting at the start of the epoch. The last bin of an
          epoch can be shorter than ``bin_size``. It is kept only if its
          center is at or before the end of the epoch. Otherwise, its
          timestamps are not counted.
        - Without ``bin_size``: each epoch of ``ep`` is one bin.

        Timestamps outside ``ep`` are not counted. The time of each bin is its
        center.

        Typical calls:

        - ``tsgroup.count(0.1)``: bins of 0.1 s over the time support.
        - ``tsgroup.count(100, time_units="ms")``: bins of 100 ms over the
          time support.
        - ``tsgroup.count(0.1, ep=epochs)``: bins of 0.1 s inside each epoch
          of ``epochs``.
        - ``tsgroup.count(ep=epochs)``: one count per epoch of ``epochs``.
        - ``tsgroup.count()``: one count per epoch of the time support.

        Parameters
        ----------
        bin_size : float or int, optional
            Size of the bins, in ``time_units``. If None (default), each epoch
            of ``ep`` is one bin.
        ep : IntervalSet, optional
            The epochs to count in. If None (default), the time support of the
            group is used.
        time_units : {'s', 'ms', 'us'}, optional
            Unit of ``bin_size``. Default is 's'.
        dtype : str, type or np.dtype, optional
            Data type of the counts. Default is np.int64.

        Returns
        -------
        TsdFrame
            The counts, with one row per bin and one column per unit. The
            columns are the keys of the group, the time support is ``ep``,
            and the metadata of the group (without ``rate``) is attached to
            the columns.

        Raises
        ------
        TypeError
            If ``bin_size`` is not a float or an int, or if ``ep`` is not an
            IntervalSet.
        ValueError
            If ``time_units`` is not 's', 'ms' or 'us', or if ``dtype`` is not
            a valid numpy dtype.

        Examples
        --------
        Count the timestamps in bins of 1 second over the first 100 seconds:

        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')
        >>> bincount = tsgroup.count(1, ep)
        >>> bincount
        Time (s)      0    1    2
        ----------  ---  ---  ---
        0.5           1    2    4
        1.5           1    2    4
        2.5           1    2    4
        3.5           1    2    4
        4.5           1    2    4
        5.5           1    2    4
        6.5           1    2    4
        ...
        93.5          1    2    4
        94.5          1    2    4
        95.5          1    2    4
        96.5          1    2    4
        97.5          1    2    4
        98.5          1    2    4
        99.5          1    2    4
        dtype: int64, shape: (100, 3)

        Without ``bin_size``, each epoch is one bin, centered on the epoch:

        >>> tsgroup.count(ep=nap.IntervalSet(start=[0, 100], end=[10, 150]))
        Time (s)      0    1    2
        ----------  ---  ---  ---
        5            11   21   41
        125          51  101  201
        dtype: int64, shape: (2, 3)

        """
        if bin_size is not None:
            if isinstance(bin_size, int):
                bin_size = float(bin_size)
            if not isinstance(bin_size, float):
                raise TypeError("bin_size argument should be float or int.")

        if not isinstance(time_units, str) or time_units not in ["s", "ms", "us"]:
            raise ValueError("time_units argument should be 's', 'ms' or 'us'.")

        if ep is None:
            ep = self.time_support
        if not isinstance(ep, IntervalSet):
            raise TypeError("ep argument should be of type IntervalSet")

        if dtype is None:
            dtype = np.dtype(np.int64)
        else:
            try:
                dtype = np.dtype(dtype)
            except Exception:
                raise ValueError(f"{dtype} is not a valid numpy dtype.")

        starts = ep.start
        ends = ep.end

        if isinstance(bin_size, (float, int)):
            bin_size = TsIndex.format_timestamps(np.array([bin_size]), time_units)[0]

        time_index, count = _count(
            self._times,
            starts,
            ends,
            bin_size,
            dtype=dtype,
            cluster_pos=self._cluster_positions,
            n_units=len(self.index),
        )

        metadata = self._metadata.copy()
        # drop rate
        metadata.drop("rate")
        return TsdFrame(
            t=time_index,
            d=count,
            time_support=ep,
            columns=self.index,
            metadata=metadata,
        )

    def to_tsd(self, *args: Union[str, list, np.ndarray, pd.Series]) -> Tsd:
        """
        Merge all the elements of the TsGroup into a single Tsd.

        Each timestamp of the group becomes one timestamp of the Tsd, sorted
        in time. Its value identifies the element it comes from: by default
        the element's key, otherwise a value per element taken from a
        metadata column or passed directly.

        Parameters
        ----------
        *args : str, list, numpy.ndarray or pandas.Series, optional
            The value of each element. Only the first argument is used.

            - Nothing (default): the key of the element.
            - str: the name of a numeric metadata column (e.g. ``"rate"``).
            - list or numpy.ndarray: one value per element, in the order of
              ``tsgroup.index``.
            - pandas.Series: one value per element, indexed by the keys of the
              TsGroup (same keys, in the same order as ``tsgroup.index``).

        Returns
        -------
        Tsd
            A float64 Tsd with one row per timestamp of the group, on the time
            support of the TsGroup.

        Raises
        ------
        RuntimeError
            - "Index are not equals": the index of the pandas.Series does not
              match ``tsgroup.index``.
            - "Values is not the same length.": the list or numpy.ndarray does
              not have one value per element.
            - "Key ... not in metadata of TsGroup": the string is not the name
              of a metadata column.
            - "Unknown argument format...": the argument is not a str, list,
              numpy.ndarray or pandas.Series. The message lists the numeric
              metadata columns.
        ValueError
            If the values cannot be converted to float (e.g. a metadata
            column of strings).

        Notes
        -----
        - Timestamps that are equal in several elements appear once per
          element, in the order of the keys.
        - ``to_tsd`` does not keep the values of Tsd, TsdFrame or TsdTensor
          elements. It uses only the key of the element, or the value given
          for that element.
        - ``Tsd.to_tsgroup`` does the reverse operation.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsgroup = nap.TsGroup({0:nap.Ts(t=np.array([0, 1])), 5:nap.Ts(t=np.array([2, 3]))})

        By default, the value of each timestamp is the key of its element:

        >>> tsgroup.to_tsd()
        Time (s)
        ----------  --
        0            0
        1            0
        2            5
        3            5
        dtype: float64, shape: (4,)

        The values can come from a metadata column, given by its name:

        >>> tsgroup.set_info( phase=np.array([np.pi, 2*np.pi]) ) # assigning a phase to my 2 elements of the TsGroup
        >>> tsgroup.to_tsd("phase")
        Time (s)
        ----------  -------
        0           3.14159
        1           3.14159
        2           6.28319
        3           6.28319
        dtype: float64, shape: (4,)

        The values can also be passed directly, one per element:

        >>> tsgroup.to_tsd([-1, 1])
        Time (s)
        ----------  --
        0           -1
        1           -1
        2            1
        3            1
        dtype: float64, shape: (4,)

        With the default values, ``Tsd.to_tsgroup`` gives back the TsGroup:

        >>> my_tsd = tsgroup.to_tsd()
        >>> my_tsd.to_tsgroup()
          Index    rate
        -------  ------
              0       1
              5       1
        """
        if len(args):
            if isinstance(args[0], pd.Series):
                if np.array_equal(self._metadata.index, args[0].index):
                    _values = args[0].values.flatten()
                else:
                    raise RuntimeError("Index are not equals")
            elif isinstance(args[0], (np.ndarray, list)):
                if self._metadata.shape[0] == len(args[0]):
                    _values = np.array(args[0])
                else:
                    raise RuntimeError("Values is not the same length.")
            elif isinstance(args[0], str):
                if args[0] in self._metadata.columns:
                    _values = self._metadata[args[0]]
                else:
                    raise RuntimeError(
                        "Key {} not in metadata of TsGroup".format(args[0])
                    )
            else:
                possible_keys = []
                for k, d in self._metadata.dtypes.items():
                    if "int" in str(d) or "float" in str(d):
                        possible_keys.append(k)
                raise RuntimeError(
                    "Unknown argument format. Must be pandas.Series, numpy.ndarray or a string from one of the following values : [{}]".format(
                        ", ".join(possible_keys)
                    )
                )
        else:
            _values = self.index

        data = np.zeros(len(self._times))
        if len(data):
            data[:] = np.asarray(_values)[self._cluster_positions]

        return Tsd(t=self._times, d=data, time_support=self.time_support)

    @add_or_convert_metadata
    def trial_count(
        self,
        ep: IntervalSet,
        bin_size: float,
        align: Literal["start", "end"] = "start",
        padding_value: float = np.nan,
        time_unit: _TimeUnits = "s",
    ) -> np.ndarray:
        """
        Count the timestamps of each element in time bins, trial by trial.

        Each interval of ``ep`` is one trial. The function divides each trial
        into bins of ``bin_size`` and counts the timestamps of each element in
        each bin. The result is a 3-d array with the shape
        ``(n_units, n_trials, n_bins)``.

        Trials can have different durations. A short trial has fewer bins than
        the longest trial. ``padding_value`` fills the bins that a trial does
        not have. ``align`` sets which side of the array these bins are on.

        Parameters
        ----------
        ep : IntervalSet
            The trials, one per interval. The intervals can have different
            durations.
        bin_size : int or float
            The size of the time bins, in ``time_unit``.
        align : {"start", "end"}, optional
            - ``"start"`` (default): bin 0 starts at the start of each trial.
              The padding is at the end of the last axis.
            - ``"end"``: the last bin ends at the end of each trial. The
              padding is at the start of the last axis.
        padding_value : float, optional
            The value for the bins that a trial does not have. The default is
            ``np.nan``.
        time_unit : {"s", "ms", "us"}, optional
            The time unit of ``bin_size``. The default is ``"s"``.

        Returns
        -------
        numpy.ndarray
            A float64 array with the shape ``(n_elements, n_trials, n_bins)``.

            - Axis 0 follows the order of ``tsgroup.index``.
            - Axis 1 follows the order of the intervals in ``ep``.
            - ``n_bins`` is the number of bins of the longest trial.

        Raises
        ------
        RuntimeError
            - If ``ep`` is not an IntervalSet.
            - If ``time_unit`` is not ``"s"``, ``"ms"`` or ``"us"``.
            - If ``align`` is not ``"start"`` or ``"end"``.
            - If ``bin_size`` is not a number.

        Notes
        -----
        The bins are the bins of ``TsGroup.count(bin_size, ep)``. The function
        keeps a bin when its center is at or before the end of the trial. Thus
        the last bin of a trial can end after the end of the trial. For
        example, a trial of 2.5 s with 1 s bins has 3 bins.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> group = nap.TsGroup({0:nap.Ts(t=np.arange(0, 100))})
        >>> ep = nap.IntervalSet(start=np.arange(20, 100, 20), end=np.arange(20, 100, 20) + np.arange(2, 10, 2))
        >>> print(ep)
          index    start    end
              0       20     22
              1       40     44
              2       60     66
              3       80     88
        shape: (4, 2), time unit: sec.

        Count the timestamps in bins of 1 second, in each interval of ``ep``.
        The trials last 2, 4, 6 and 8 seconds. NaN fills the bins that a
        trial does not have:

        >>> tensor = group.trial_count(ep, bin_size=1)
        >>> tensor
        array([[[ 1.,  1., nan, nan, nan, nan, nan, nan],
                [ 1.,  1.,  1.,  1., nan, nan, nan, nan],
                [ 1.,  1.,  1.,  1.,  1.,  1., nan, nan],
                [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]])

        With ``align="end"``, the trials align on their end. The padding moves
        to the start:

        >>> tensor = group.trial_count(ep, bin_size=1, align="end")
        >>> tensor
        array([[[nan, nan, nan, nan, nan, nan,  1.,  1.],
                [nan, nan, nan, nan,  1.,  1.,  1.,  1.],
                [nan, nan,  1.,  1.,  1.,  1.,  1.,  1.],
                [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]])
        """
        if not isinstance(ep, IntervalSet):
            raise RuntimeError("Argument ep should be of type IntervalSet")
        if time_unit not in ["s", "ms", "us"]:
            raise RuntimeError("time_unit should be 's', 'ms' or 'us'")
        if align not in ["start", "end"]:
            raise RuntimeError("align should be 'start' or 'end'")
        if not isinstance(bin_size, Number):
            raise RuntimeError("bin_size should be of type int or float")
        # Determine size of tensor
        bin_size = float(TsIndex.format_timestamps(np.array([bin_size]), time_unit)[0])
        n_t = int(np.max(np.ceil((ep.end + bin_size - ep.start) / bin_size)))
        count = self.count(bin_size=bin_size, ep=ep)

        output = np.ones(shape=(count.shape[1], len(ep), n_t)) * padding_value

        n_ep = np.zeros(len(ep), dtype="int")  # To trim to the minimum length

        if align == "start":
            for i in range(len(ep)):
                tmp = count.get(ep.start[i], ep.end[i]).values
                n_ep[i] = tmp.shape[0]
                output[:, i, 0 : tmp.shape[0]] = np.transpose(tmp)
            output = output[:, :, 0 : np.max(n_ep)]

        if align == "end":
            for i in range(len(ep)):
                tmp = count.get(ep.start[i], ep.end[i]).values
                n_ep[i] = tmp.shape[0]
                output[:, i, -tmp.shape[0] :] = np.transpose(tmp)
            output = output[:, :, -np.max(n_ep) :]

        return output

    def time_diff(
        self,
        align: Literal["start", "center", "end"] = "center",
        epochs: Optional[IntervalSet] = None,
    ) -> dict[int, Tsd]:
        """
        Compute the time between consecutive timestamps of each element.

        For spike trains, these differences are the inter-spike intervals.
        The function computes the differences separately for each element and
        for each epoch. It does not compute a difference between the last
        timestamp of one epoch and the first timestamp of the next epoch.

        Parameters
        ----------
        align : {"start", "center", "end"}, optional
            The timestamp of each difference, between the two timestamps
            ``t[i]`` and ``t[i + 1]``:

            - ``"start"``: ``t[i]``.
            - ``"center"`` (default): ``(t[i] + t[i + 1]) / 2``.
            - ``"end"``: ``t[i + 1]``.
        epochs : IntervalSet, optional
            The epochs in which the function computes the differences. It
            ignores the timestamps outside these epochs. The default is the
            time support of the TsGroup.

        Returns
        -------
        dict of int to Tsd
            One Tsd per element, with the keys in the order of
            ``tsgroup.index``. Each Tsd holds the differences, in the time
            unit of the timestamps, and has ``epochs`` as its time support.
            An element with fewer than 2 timestamps in an epoch has no
            difference in that epoch. Its Tsd can be empty.

        Raises
        ------
        RuntimeError
            If ``align`` is not ``"start"``, ``"center"`` or ``"end"``.
        TypeError
            If ``epochs`` is not an IntervalSet.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=[1, 3, 5, 6, 8, 12], time_units='s'),1:nap.Ts(t=[2, 8, 9, 13, 14, 17], time_units='s'), 2:nap.Ts(t=[1, 2, 5, 7, 9, 12], time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp)
        >>> epochs = nap.IntervalSet(start=2, end=9, time_units='s')

        In the epoch [2, 9], element 1 has the timestamps 2, 8 and 9. Its
        differences are 6 and 1, at the centers 5 and 8.5:

        >>> time_diffs = tsgroup.time_diff(align="center", epochs=epochs)
        >>> time_diffs
        {0: Time (s)
        ----------  --
        4            2
        5.5          1
        7            2
        dtype: float64, shape: (3,), 1: Time (s)
        ----------  --
        5            6
        8.5          1
        dtype: float64, shape: (2,), 2: Time (s)
        ----------  --
        3.5          3
        6            2
        8            2
        dtype: float64, shape: (3,)}
        """
        if align not in ["start", "center", "end"]:
            raise RuntimeError("align should be 'start', 'center' or 'end'")

        if epochs is None:
            epochs = self.time_support
        elif not isinstance(epochs, IntervalSet):
            raise TypeError("epochs should be an object of type IntervalSet")

        alpha = 0.0 if align == "start" else 0.5 if align == "center" else 1.0
        new_t, new_d, offsets = _time_diff_grouped(
            self._times,
            self._cluster_positions,
            len(self.index),
            epochs.start,
            epochs.end,
            alpha,
        )

        out = {}
        for i, k in enumerate(self.index.tolist()):
            sl = slice(offsets[i], offsets[i + 1])
            # differences are emitted per-epoch in order -> sorted and within `epochs`
            with trusted_construction():
                out[k] = Tsd(t=new_t[sl], d=new_d[sl], time_support=epochs)
        return out

    def get(
        self,
        start: float,
        end: Optional[float] = None,
        time_units: _TimeUnits = "s",
    ) -> TsGroup:
        """
        Keep the timestamps of each element between ``start`` and ``end``.

        - With ``end``: the result keeps every timestamp ``t`` with
          ``start <= t <= end``, in every element.
        - Without ``end``: the result keeps the timestamp of each element that
          is closest to ``start``. If two timestamps are at the same distance
          from ``start``, the result keeps the later timestamp.

        The time support does not change. To change the time support, use
        ``restrict``. The ``rate`` metadata of the result uses the new number
        of timestamps and the time support that did not change. The other
        metadata do not change.

        Parameters
        ----------
        start : int or float
            The start of the slice. Without ``end``, the function keeps the
            timestamp closest to this time.
        end : int or float, optional
            The end of the slice. The default is None.
        time_units : {"s", "ms", "us"}, optional
            The time unit of ``start`` and ``end``. The default is ``"s"``.

        Returns
        -------
        TsGroup
            A TsGroup with the same keys and the same time support. An element
            with no timestamp in the slice is empty.

        Raises
        ------
        ValueError
            - If ``start`` or ``end`` is not an int or a float.
            - If ``start`` is after ``end``.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsgroup = nap.TsGroup({0: nap.Ts(t=[1, 3, 5]), 2: nap.Ts(t=[2, 6])})

        Keep the timestamps between 2 and 5 seconds. The result is a TsGroup
        with the same keys and the same time support:

        >>> new_tsgroup = tsgroup.get(2, 5)
        >>> new_tsgroup
          Index    rate
        -------  ------
              0     0.4
              2     0.2
        >>> new_tsgroup.time_support
          index    start    end
              0        1      6
        shape: (1, 2), time unit: sec.

        ``to_tsd`` shows the timestamps that the result keeps. The value of
        each timestamp is the key of its element:

        >>> new_tsgroup.to_tsd()
        Time (s)
        ----------  --
        2            2
        3            0
        5            0
        dtype: float64, shape: (3,)

        Keep the timestamp of each element that is closest to 2.6 seconds:

        >>> new_tsgroup = tsgroup.get(2.6)
        >>> new_tsgroup.to_tsd()
        Time (s)
        ----------  --
        2            2
        3            0
        dtype: float64, shape: (2,)
        """
        cols = self._metadata.columns[1:]  # .drop("rate")

        if end is None:
            # closest timestamp of each unit: inherently per unit
            newgr = {k: m.get(start, end, time_units) for k, m in self.items()}
            return TsGroup(
                newgr,
                time_support=self.time_support,
                metadata=self._metadata[cols],
            )

        # `start <= t <= end` does not depend on the unit: slice the merged
        # array once. Same validation as `Ts.get`, but bounds come from a plain
        # searchsorted since several units may share the `end` timestamp.
        for name, value in (("start", start), ("end", end)):
            if not isinstance(value, Number):
                raise ValueError(
                    f"'{name}' must be an int or a float. Type {type(value)} provided instead!"
                )
        start, end = TsIndex.format_timestamps(np.array([start, end]), time_units)
        if start > end:
            raise ValueError("'start' should not precede 'end'.")
        sl = slice(
            np.searchsorted(self._times, start, side="left"),
            np.searchsorted(self._times, end, side="right"),
        )

        return TsGroup._from_arrays(
            self._times[sl],
            self._clusters[sl],
            None if self._data is None else self._data[sl],
            self._is_tsd,
            self.index,
            self.time_support,
            metadata=self._metadata[cols],
            columns=self._columns,
        )

    #################################
    # Special slicing of metadata
    #################################

    def getby_threshold(
        self, key: str, thr: float, op: Literal[">", "<", ">=", "<="] = ">"
    ) -> TsGroup:
        """
        Select the elements whose metadata value passes a threshold.

        The function compares the value of the metadata column ``key`` of each
        element with ``thr``, with the operator ``op``. It returns a TsGroup
        with only the elements that pass the comparison.

        Parameters
        ----------
        key : str
            The name of a metadata column, e.g. ``"rate"``.
        thr : float
            The threshold.
        op : {">", "<", ">=", "<="}, optional
            The comparison operator. An element passes when
            ``value op thr`` is true. The default is ``">"``.

        Returns
        -------
        TsGroup
            A TsGroup with the elements that pass, with their keys and
            metadata. The time support does not change. If no element passes,
            the TsGroup is empty.

        Raises
        ------
        KeyError
            If ``key`` is not the name of a metadata column.
        RuntimeError
            If ``op`` is not ``">"``, ``"<"``, ``">="`` or ``"<="``.

        See Also
        --------
        getby_intervals : Select the elements by bins of a metadata value.
        getby_category : Select the elements by category of a metadata value.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp)
        >>> tsgroup
          Index     rate
        -------  -------
              0  0.66722
              1  1.33445
              2  4.00334

        Keep the elements with a rate above 1 Hz:

        >>> newtsgroup = tsgroup.getby_threshold('rate', 1, op='>')
        >>> newtsgroup
          Index     rate
        -------  -------
              1  1.33445
              2  4.00334

        """
        if op == ">":
            ix = list(self._metadata.index[self._metadata[key] > thr])
            return self[ix]
        elif op == "<":
            ix = list(self._metadata.index[self._metadata[key] < thr])
            return self[ix]
        elif op == ">=":
            ix = list(self._metadata.index[self._metadata[key] >= thr])
            return self[ix]
        elif op == "<=":
            ix = list(self._metadata.index[self._metadata[key] <= thr])
            return self[ix]
        else:
            raise RuntimeError("Operation {} not recognized.".format(op))

    def getby_intervals(
        self, key: str, bins: Union[list, np.ndarray]
    ) -> tuple[list[TsGroup], np.ndarray]:
        """
        Split the elements into bins of a metadata value.

        The function puts each element into a bin, by the value of its
        metadata column ``key``. It returns one TsGroup for each bin that
        has at least one element.

        Parameters
        ----------
        key : str
            The name of a numeric metadata column.
        bins : numpy.ndarray or list
            The bin edges, in increasing order. ``n`` edges give ``n - 1``
            bins. Each bin includes its left edge and excludes its right
            edge: an element is in bin ``i`` if
            ``bins[i] <= value < bins[i + 1]``.

        Returns
        -------
        groups : list of TsGroup
            One TsGroup for each bin that has at least one element, in the
            order of the bins. The function skips empty bins. Each TsGroup
            keeps the keys, the metadata and the time support of its
            elements.
        bin_centers : numpy.ndarray
            The center of each bin in ``groups``, in the same order.

        Raises
        ------
        KeyError
            If ``key`` is not the name of a metadata column.

        See Also
        --------
        getby_threshold : Select the elements by a threshold on a metadata value.
        getby_category : Select the elements by category of a metadata value.

        Notes
        -----
        The function drops the elements with a value outside all bins. This
        includes a value equal to the last edge.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp, metadata={"alpha": np.arange(3)})

        Split the elements into the bins [0, 1) and [1, 2) of ``alpha``.
        Element 2 has ``alpha = 2``, which is equal to the last edge. Thus
        it is not in a bin:

        >>> groups, bin_centers = tsgroup.getby_intervals('alpha', [0, 1, 2])
        >>> groups[0]
          Index     rate    alpha
        -------  -------  -------
              0  0.66722        0
        >>> groups[1]
          Index     rate    alpha
        -------  -------  -------
              1  1.33445        1
        >>> bin_centers
        array([0.5, 1.5])
        """
        idx = np.digitize(self._metadata[key], bins) - 1
        groups = {k: self._metadata.index[idx == k] for k in np.unique(idx)}
        ix = np.unique(list(groups.keys()))
        ix = ix[ix >= 0]
        ix = ix[ix < len(bins) - 1]
        xb = bins[0:-1] + np.diff(bins) / 2
        sliced = [self[list(groups[i])] for i in ix]
        return sliced, xb[ix]

    def getby_category(self, key: str) -> dict[Any, TsGroup]:
        """
        Split the elements into groups by the value of a metadata column.

        The function puts the elements with the same value of the metadata
        column ``key`` into the same TsGroup.

        Parameters
        ----------
        key : str or list of str
            The name of a metadata column. With a list of names, the function
            groups the elements by each combination of values.

        Returns
        -------
        dict
            One TsGroup for each value, with the value as the dict key, in
            sorted order. With a list of names, each dict key is a tuple of
            values. Each TsGroup keeps the keys, the metadata and the time
            support of its elements.

        Raises
        ------
        ValueError
            If ``key`` is not the name of a metadata column.

        See Also
        --------
        groupby : Get the keys of the elements in each group.
        getby_threshold : Select the elements by a threshold on a metadata value.
        getby_intervals : Select the elements by bins of a metadata value.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp, metadata={"group": [0, 1, 1]})

        Split the elements by the value of ``group``. Element 0 has the value
        0. Elements 1 and 2 have the value 1:

        >>> groups = tsgroup.getby_category('group')
        >>> groups[0]
          Index     rate    group
        -------  -------  -------
              0  0.66722        0
        >>> groups[1]
          Index     rate    group
        -------  -------  -------
              1  1.33445        1
              2  4.00334        1
        """
        groups = self.groupby(key)
        sliced = {k: self[list(groups[k])] for k in groups.keys()}
        return sliced

    @staticmethod
    @add_or_convert_metadata
    def merge_group(
        *tsgroups: TsGroup,
        reset_index: bool = False,
        reset_time_support: bool = False,
        ignore_metadata: bool = False,
    ) -> TsGroup:
        """
        Merge several TsGroup objects into one TsGroup.

        The result holds all the elements of all the input TsGroups. By
        default, the function makes three checks before the merge:

        - The keys of the TsGroups do not overlap.
        - The TsGroups have the same time support.
        - The TsGroups have the same metadata columns.

        Each parameter below removes one check.

        Parameters
        ----------
        *tsgroups : TsGroup
            The TsGroups to merge.
        reset_index : bool, optional
            - False (default): the result keeps the keys of the elements. The
              keys must not overlap.
            - True: the result gets the new keys ``0..n_elements-1``. The new
              keys follow the order of the TsGroups, then the order of the
              keys in each TsGroup.
        reset_time_support : bool, optional
            - False (default): the TsGroups must have the same time support.
              The result keeps this time support.
            - True: the time support of the result is the union of the time
              supports of the TsGroups.
        ignore_metadata : bool, optional
            - False (default): the TsGroups must have the same metadata
              columns. The result concatenates the metadata.
            - True: the result has no metadata column other than ``rate``.

        Returns
        -------
        TsGroup
            The merged TsGroup. The function computes the ``rate`` metadata
            again, on the time support of the result.

        Raises
        ------
        TypeError
            If an input is not a TsGroup.
        ValueError
            - If ``reset_index=False`` and the keys overlap.
            - If ``reset_time_support=False`` and the time supports are not
              the same.
            - If ``ignore_metadata=False`` and the metadata columns are not
              the same.
            - If the Tsd, TsdFrame or TsdTensor elements do not have the same
              shape after the time axis.

        See Also
        --------
        merge : Merge other TsGroups into this TsGroup.

        Notes
        -----
        - With only one TsGroup, the function prints a message and returns
          the same object, not a copy.
        - The result keeps the values of the Tsd, TsdFrame and TsdTensor
          elements. A TsGroup stores all the values in one array. Thus the
          values of all the elements must have the same shape after the time
          axis. The function casts the values to a common dtype.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> time_support = nap.IntervalSet(0, 4)
        >>> tsgroup1 = nap.TsGroup(
        ...     {5: nap.Ts(t=[1, 2]), 3: nap.Ts(t=[1.5, 2.5, 3.5])},
        ...     time_support=time_support,
        ...     metadata={"label": ["a", "b"]},
        ... )
        >>> tsgroup2 = nap.TsGroup(
        ...     {0: nap.Ts(t=[1, 3])},
        ...     time_support=time_support,
        ...     metadata={"label": ["c"]},
        ... )

        The keys do not overlap. Thus the result can keep them:

        >>> nap.TsGroup.merge_group(tsgroup1, tsgroup2)
          Index    rate  label
        -------  ------  -------
              0    0.5   c
              3    0.75  b
              5    0.5   a

        With ``reset_index=True``, the new keys follow the order of the
        TsGroups. Keys 3 and 5 of ``tsgroup1`` become 0 and 1. Key 0 of
        ``tsgroup2`` becomes 2:

        >>> nap.TsGroup.merge_group(tsgroup1, tsgroup2, reset_index=True)
          Index    rate  label
        -------  ------  -------
              0    0.75  b
              1    0.5   a
              2    0.5   c
        """
        is_tsgroup = [isinstance(tsg, TsGroup) for tsg in tsgroups]
        if not all(is_tsgroup):
            not_tsgroup_index = [i + 1 for i, boo in enumerate(is_tsgroup) if not boo]
            raise TypeError(f"Input at positions {not_tsgroup_index} are not TsGroup!")

        if len(tsgroups) == 1:
            print("Only one TsGroup object provided, no merge needed.")
            return tsgroups[0]

        # the merge reads the merged arrays of every group
        tsgroups = [tsg._loaded() for tsg in tsgroups]

        tsg1 = tsgroups[0]
        keys = set(tsg1.keys())
        metadata = tsg1._metadata.copy()

        for i, tsg in enumerate(tsgroups[1:]):
            if not ignore_metadata:
                if tsg1.metadata_columns != tsg.metadata_columns:
                    raise ValueError(
                        f"TsGroup at position {i + 2} has different metadata columns from previous TsGroup objects. "
                        "Set `ignore_metadata=True` to bypass the check."
                    )
                metadata.merge(tsg._metadata)

            if not reset_index:
                key_overlap = keys.intersection(tsg.keys())
                if key_overlap:
                    raise ValueError(
                        f"TsGroup at position {i + 2} has overlapping keys {key_overlap} with previous TsGroup objects. "
                        "Set `reset_index=True` to bypass the check."
                    )
                keys.update(tsg.keys())

            if reset_time_support:
                time_support = None
            else:
                if not np.allclose(
                    tsg1.time_support.as_units("s").to_numpy(),
                    tsg.time_support.as_units("s").to_numpy(),
                    atol=10 ** (-nap_config.time_index_precision),
                    rtol=0,
                ):
                    raise ValueError(
                        f"TsGroup at position {i + 2} has different time support from previous TsGroup objects. "
                        "Set `reset_time_support=True` to bypass the check."
                    )
                time_support = tsg1.time_support

        if time_support is None:
            time_support = _union_intervals([tsg.time_support for tsg in tsgroups])

        # Concatenate the merged arrays of every group. With `reset_index`, keys
        # become 0..n-1 following the groups' order then each group's key order.
        clusters = []
        offset = 0
        for tsg in tsgroups:
            if reset_index:
                clusters.append(tsg._cluster_positions + offset)
                offset += len(tsg.index)
            else:
                clusters.append(tsg._clusters)
        clusters = np.concatenate(clusters).astype(np.int64, copy=False)
        if reset_index:
            index = np.arange(offset, dtype=np.int64)
        else:
            index = np.concatenate([tsg.index for tsg in tsgroups])
        times = np.concatenate([tsg._times for tsg in tsgroups])
        is_tsd = np.concatenate([tsg._is_tsd for tsg in tsgroups])

        # groups without values contribute fill values, flagged by `is_tsd`
        values = _concat_values(
            [tsg._data for tsg in tsgroups], [len(tsg._times) for tsg in tsgroups]
        )

        order = np.argsort(times, kind="stable")
        times = times[order]
        clusters = clusters[order]
        if values is not None:
            values = values[order]

        times, clusters, values = _restrict_arrays(
            times, time_support.start, time_support.end, clusters, values
        )

        # keep the index sorted, metadata rows and `is_tsd` following it
        sort_index = np.argsort(index, kind="stable")
        index = index[sort_index]
        is_tsd = is_tsd[sort_index]

        if ignore_metadata:
            metadata = None
        else:
            if reset_index:
                metadata.reset_index()
            metadata.drop("rate")
            metadata = metadata.loc[index]

        columns = tsgroups[0]._columns
        if any(
            tsg._columns is None
            or columns is None
            or not np.array_equal(tsg._columns, columns)
            for tsg in tsgroups[1:]
            if tsg._data is not None
        ):
            columns = None

        return TsGroup._from_arrays(
            times,
            clusters,
            values,
            is_tsd,
            index,
            time_support,
            metadata=metadata,
            columns=columns,
        )

    def merge(
        self,
        *tsgroups: TsGroup,
        reset_index: bool = False,
        reset_time_support: bool = False,
        ignore_metadata: bool = False,
    ) -> TsGroup:
        """
        Merge this TsGroup with other TsGroups.

        The result holds all the elements of this TsGroup and of the other
        TsGroups. For example, use this method to add more neurons or
        channels, or to add more trials. This method calls ``merge_group``
        with this TsGroup first. It makes the same checks:

        - The keys of the TsGroups do not overlap.
        - The TsGroups have the same time support.
        - The TsGroups have the same metadata columns.

        Each parameter below removes one check.

        Parameters
        ----------
        *tsgroups : TsGroup
            The TsGroups to merge with this TsGroup.
        reset_index : bool, optional
            - False (default): the result keeps the keys of the elements. The
              keys must not overlap.
            - True: the result gets the new keys ``0..n_elements-1``. The new
              keys follow the order of the TsGroups (this TsGroup first), then
              the order of the keys in each TsGroup.
        reset_time_support : bool, optional
            - False (default): the TsGroups must have the same time support.
              The result keeps this time support.
            - True: the time support of the result is the union of the time
              supports of the TsGroups.
        ignore_metadata : bool, optional
            - False (default): the TsGroups must have the same metadata
              columns. The result concatenates the metadata.
            - True: the result has no metadata column other than ``rate``.

        Returns
        -------
        TsGroup
            The merged TsGroup. The function computes the ``rate`` metadata
            again, on the time support of the result.

        Raises
        ------
        TypeError
            If an input is not a TsGroup.
        ValueError
            - If ``reset_index=False`` and the keys overlap.
            - If ``reset_time_support=False`` and the time supports are not
              the same.
            - If ``ignore_metadata=False`` and the metadata columns are not
              the same.
            - If the Tsd, TsdFrame or TsdTensor elements do not have the same
              shape after the time axis.

        See Also
        --------
        merge_group : Merge several TsGroups into one TsGroup.

        Examples
        --------
        >>> import pynapple as nap
        >>> time_support_a = nap.IntervalSet(start=-1, end=1, time_units='s')
        >>> time_support_b = nap.IntervalSet(start=-5, end=5, time_units='s')
        >>> tsgroup1 = nap.TsGroup({0: nap.Ts(t=[-1, 0, 1])}, time_support=time_support_a)
        >>> tsgroup2 = nap.TsGroup({10: nap.Ts(t=[-1, 0, 1])}, time_support=time_support_a)
        >>> tsgroup3 = nap.TsGroup({0: nap.Ts(t=[-.1, 0, .1])}, time_support=time_support_a)
        >>> tsgroup4 = nap.TsGroup({10: nap.Ts(t=[-1, 0, 1])}, time_support=time_support_b)

        ``tsgroup1`` and ``tsgroup2`` have the same time support and different
        keys. Thus the default options work:

        >>> tsgroup1.merge(tsgroup2)
          Index    rate
        -------  ------
              0     1.5
             10     1.5

        ``tsgroup1`` and ``tsgroup3`` both have the key 0. Use
        ``reset_index=True`` to give new keys to the elements:

        >>> tsgroup1.merge(tsgroup3, reset_index=True)
          Index    rate
        -------  ------
              0     1.5
              1     1.5

        ``tsgroup1`` and ``tsgroup4`` have different time supports. Use
        ``reset_time_support=True`` to use the union of the time supports.
        The rates use the new time support of 10 seconds:

        >>> tsgroup_14 = tsgroup1.merge(tsgroup4, reset_time_support=True)
        >>> tsgroup_14
          Index    rate
        -------  ------
              0     0.3
             10     0.3
        >>> tsgroup_14.time_support
          index    start    end
              0       -5      5
        shape: (1, 2), time unit: sec.
        """
        return TsGroup.merge_group(
            self,
            *tsgroups,
            reset_index=reset_index,
            reset_time_support=reset_time_support,
            ignore_metadata=ignore_metadata,
        )

    @add_or_convert_metadata
    def save(self, filename: Union[str, Path]) -> None:
        """
        Save the TsGroup in a npz file.

        Use ``nap.load_file`` to load the file again. This function is for
        small and medium TsGroups. The file stores the TsGroup as flat
        arrays: all the timestamps in one sorted array, with the key of the
        element of each timestamp. For example, this TsGroup:

        .. code-block:: python

            TsGroup({
                0: Tsd(t=[0, 2, 4], d=[1, 2, 3]),
                1: Tsd(t=[1, 5], d=[5, 6]),
            })

        gives a npz file with these keys:

        .. code-block:: python

            {
                "t": [0, 1, 2, 4, 5],     # all the timestamps, sorted
                "d": [1, 5, 2, 3, 6],     # the value of each timestamp
                "index": [0, 1, 0, 0, 1], # the key of each timestamp
                "keys": [0, 1],           # the keys of the TsGroup
                "start": [0],             # the time support
                "end": [5],
                "type": "TsGroup",
                "_metadata": {...},       # the metadata, without "rate"
            }

        The file has the key ``"d"`` only if at least one element holds
        values.

        Parameters
        ----------
        filename : str or Path
            The name of the file. The function sets the suffix to ``.npz``.
            If the name has a different suffix, the function replaces it.

        Raises
        ------
        TypeError
            If ``filename`` is not a str or a Path.
        RuntimeError
            - If ``filename`` is a directory.
            - If the parent directory of ``filename`` does not exist.

        See Also
        --------
        pynapple.io.misc.load_file : Load a npz file saved by pynapple.

        Notes
        -----
        The file stores the metadata in the ``"_metadata"`` key as a pickled
        object. ``nap.load_file`` computes ``rate`` again when it loads the
        file.

        Some information is lost:

        - The file stores the values as float64, with NaN for the elements
          without values. Thus integer values come back as float64.
        - If at least one element holds values, every element comes back as
          a Tsd, TsdFrame or TsdTensor. A Ts element comes back with NaN
          values.
        - The column names of TsdFrame elements are not saved.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsgroup = nap.TsGroup(
        ...     {0: nap.Ts(t=np.array([0.0, 2.0, 4.0])),
        ...      6: nap.Ts(t=np.array([1.0, 5.0]))},
        ...     metadata={"group": np.array([0, 1]),
        ...               "location": np.array(['right foot', 'left foot'])}
        ... )
        >>> tsgroup
          Index    rate    group  location
        -------  ------  -------  ----------
              0     0.6        0  right foot
              6     0.4        1  left foot
        >>> tsgroup.save("my_tsgroup.npz")

        Load the file with ``nap.load_file``:

        >>> tsgroup = nap.load_file("my_tsgroup.npz")
        >>> tsgroup
          Index    rate    group  location
        -------  ------  -------  ----------
              0     0.6        0  right foot
              6     0.4        1  left foot
        """
        filename = check_filename(filename)

        dicttosave = {"type": np.array(["TsGroup"], dtype=np.str_)}
        # don't save rate in metadata since it will be re-added when loading
        dicttosave["_metadata"] = dict(self._metadata.copy().drop("rate"))

        # are these things that still need to be enforced?
        # for k in self._metadata.columns:
        #     if k not in ["t", "d", "start", "end", "index", "keys"]:
        #         tmp = self._metadata[k].values
        #         if tmp.dtype == np.dtype("O"):
        #             tmp = tmp.astype(np.str_)
        #         dicttosave[k] = tmp

        # The merged arrays already are the flattened layout saved on disk.
        dicttosave["t"] = self._times
        dicttosave["index"] = self._clusters
        if self._data is not None:
            data = np.full(self._data.shape, np.nan)
            valued = self._is_tsd[self._cluster_positions]
            data[valued] = self._data[valued]
            if not np.all(np.isnan(data)):
                dicttosave["d"] = data
        dicttosave["keys"] = np.array(self.keys())
        dicttosave["start"] = self.time_support.start
        dicttosave["end"] = self.time_support.end

        np.savez(filename, **dicttosave)

        return

    @classmethod
    def _from_npz_reader(cls, file: Mapping[str, Any]) -> TsGroup:
        """
        Load a Tsd object from a npz file.

        Parameters
        ----------
        file : str
            The opened npz file

        Returns
        -------
        Tsd
            The Tsd object
        """

        times = file["t"]
        index = file["index"]
        has_data = "d" in file.keys()
        time_support = IntervalSet(file["start"], file["end"])

        if has_data:
            data = file["d"]

        if "keys" in file.keys():
            keys = np.asarray(file["keys"], dtype=np.int64)
        else:
            keys = np.unique(index)
        keys = np.sort(keys)

        times = np.asarray(times, dtype=np.float64)
        index = np.asarray(index, dtype=np.int64)
        values = data if has_data else None
        if len(times) > 1 and np.any(times[1:] < times[:-1]):
            order = np.argsort(times, kind="stable")
            times = times[order]
            index = index[order]
            if has_data:
                values = values[order]

        tsgroup = cls._from_arrays(
            times,
            index,
            values,
            np.full(len(keys), has_data, dtype=bool),
            keys,
            time_support,
        )

        if "_metadata" in file:  # load metadata if it exists
            if file["_metadata"]:  # check that metadata is not empty
                metainfo = file["_metadata"].item()
                # check if first field is a dictionary, meaning it was saved from a pandas.DataFrame
                if isinstance(next(iter(metainfo.values())), dict):
                    metainfo = pd.DataFrame.from_dict(metainfo)
                tsgroup.set_info(metainfo)

        metainfo = {}
        not_info_keys = {
            "start",
            "end",
            "t",
            "index",
            "d",
            "rate",
            "keys",
            "_metadata",
            "type",
        }

        for k in set(file.keys()) - not_info_keys:
            tmp = file[k]
            if len(tmp) == len(tsgroup):
                metainfo[k] = tmp

        tsgroup.set_info(**metainfo)

        return tsgroup

    @add_meta_docstring("set_info")
    def set_info(
        self, metadata: Optional[Union[pd.DataFrame, dict]] = None, **kwargs: Any
    ) -> None:
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,200), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        ... }
        >>> tsgroup = nap.TsGroup(tmp)

        To add metadata with a pandas.DataFrame:

        >>> import pandas as pd
        >>> structs = pd.DataFrame(index = [0,1,2], data=['pfc','pfc','ca1'], columns=['struct'])
        >>> tsgroup.set_info(structs)
        >>> tsgroup
          Index     rate  struct
        -------  -------  --------
              0  0.66722  pfc
              1  1.33445  pfc
              2  4.00334  ca1

        To add metadata with a dictionary:

        >>> coords = {"coords": [[0,0],[0,1],[1,0]]}
        >>> tsgroup.set_info(coords)
        >>> tsgroup
          Index     rate  struct    coords
        -------  -------  --------  --------
              0  0.66722  pfc       [0 0]
              1  1.33445  pfc       [0 1]
              2  4.00334  ca1       [1 0]

        To add metadata with a keyword argument (pd.Series, numpy.ndarray, list or tuple):

        >>> hd = pd.Series(index = [0,1,2], data = [0,1,1])
        >>> tsgroup.set_info(hd=hd)
        >>> tsgroup
          Index     rate  struct    coords      hd
        -------  -------  --------  --------  ----
              0  0.66722  pfc       [0 0]        0
              1  1.33445  pfc       [0 1]        1
              2  4.00334  ca1       [1 0]        1

        To add metadata as an attribute:

        >>> tsgroup.label = ["a", "b", "c"]
        >>> tsgroup
          Index     rate  struct    coords      hd  label
        -------  -------  --------  --------  ----  -------
              0  0.66722  pfc       [0 0]        0  a
              1  1.33445  pfc       [0 1]        1  b
              2  4.00334  ca1       [1 0]        1  c

        To add metadata as a key:

        >>> tsgroup["type"] = ["multi", "multi", "single"]
        >>> tsgroup
          Index     rate  struct    coords      hd  label    type    ...
        -------  -------  --------  --------  ----  -------  ------  -----
              0  0.66722  pfc       [0 0]        0  a        multi   ...
              1  1.33445  pfc       [0 1]        1  b        multi   ...
              2  4.00334  ca1       [1 0]        1  c        single  ...

        Metadata can be overwritten:

        >>> tsgroup.set_info(label=["x", "y", "z"])
        >>> tsgroup
          Index     rate  struct    coords      hd  label    type    ...
        -------  -------  --------  --------  ----  -------  ------  -----
              0  0.66722  pfc       [0 0]        0  x        multi   ...
              1  1.33445  pfc       [0 1]        1  y        multi   ...
              2  4.00334  ca1       [1 0]        1  z        single  ...

        """
        _MetadataMixin.set_info(self, metadata, **kwargs)

    @add_meta_docstring("get_info")
    def get_info(self, key: Union[str, list[str]]) -> Union[pd.Series, pd.DataFrame]:
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,200), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        ... }
        >>> metadata = {"l1": [1, 2, 3], "l2": ["x", "x", "y"]}
        >>> tsgroup = nap.TsGroup(tmp,metadata=metadata)
        >>> print(tsgroup)
          Index     rate    l1  l2
        -------  -------  ----  ----
              0  0.66722     1  x
              1  1.33445     2  x
              2  4.00334     3  y

        To access a single metadata column:

        >>> tsgroup.get_info("l1")
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        To access multiple metadata columns:

        >>> tsgroup.get_info(["l1", "l2"])
           l1 l2
        0   1  x
        1   2  x
        2   3  y

        To access metadata as a key:

        >>> tsgroup["l1"]
        0    1
        1    2
        2    3
        Name: l1, dtype: int64

        Multiple metadata columns can be accessed as keys:

        >>> tsgroup[["l1", "l2"]]
           l1 l2
        0   1  x
        1   2  x
        2   3  y
        """
        return _MetadataMixin.get_info(self, key)

    @add_meta_docstring("drop_info")
    def drop_info(self, key: Union[str, list[str]]) -> None:
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,200), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        ... }
        >>> metadata = {"l1": [1, 2, 3], "l2": ["x", "x", "y"], "l3": [4, 5, 6]}
        >>> tsgroup = nap.TsGroup(tmp,metadata=metadata)
        >>> print(tsgroup)
          Index     rate    l1  l2      l3
        -------  -------  ----  ----  ----
              0  0.66722     1  x        4
              1  1.33445     2  x        5
              2  4.00334     3  y        6

        To drop a single metadata column:

        >>> tsgroup.drop_info("l1")
        >>> tsgroup
          Index     rate  l2      l3
        -------  -------  ----  ----
              0  0.66722  x        4
              1  1.33445  x        5
              2  4.00334  y        6

        To drop multiple metadata columns:

        >>> tsgroup.drop_info(["l2", "l3"])
        >>> tsgroup
          Index     rate
        -------  -------
              0  0.66722
              1  1.33445
              2  4.00334
        """
        return _MetadataMixin.drop_info(self, key)

    @add_meta_docstring("restrict_info")
    def restrict_info(self, key: Union[str, list[str]]) -> None:
        """
        Note
        ----
        The `rate` column is always kept in the metadata, even if it is not specified in `key`.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,200), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        ... }
        >>> metadata = {"l1": [1, 2, 3], "l2": ["x", "x", "y"], "l3": [4, 5, 6]}
        >>> tsgroup = nap.TsGroup(tmp,metadata=metadata)
        >>> print(tsgroup)
          Index     rate    l1  l2      l3
        -------  -------  ----  ----  ----
              0  0.66722     1  x        4
              1  1.33445     2  x        5
              2  4.00334     3  y        6

        To restrict to multiple metadata columns:

        >>> tsgroup.restrict_info(["l2", "l3"])
        >>> tsgroup
          Index     rate  l2      l3
        -------  -------  ----  ----
              0  0.66722  x        4
              1  1.33445  x        5
              2  4.00334  y        6

        To restrict to a single metadata column:

        >>> tsgroup.drop_info("l2")
        >>> tsgroup
          Index     rate    l3
        -------  -------  ----
              0  0.66722     4
              1  1.33445     5
              2  4.00334     6
        """
        return _MetadataMixin.restrict_info(self, key)

    @add_or_convert_metadata
    @add_meta_docstring("groupby")
    def groupby(
        self, by: Union[str, list[str]], get_group: Optional[Any] = None
    ) -> Union[dict[Any, np.ndarray], TsGroup]:
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,40), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,40,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,40,0.25), time_units='s'),
        ... }
        >>> metadata = {"l1": [1, 2, 2], "l2": ["x", "x", "y"]}
        >>> tsgroup = nap.TsGroup(tmp,metadata=metadata)
        >>> print(tsgroup)
          Index     rate    l1  l2
        -------  -------  ----  ----
              0  1.00629     1  x
              1  2.01258     2  x
              2  4.02516     2  y

        Grouping by a single column:

        >>> tsgroup.groupby("l2")
        {'x': array([0, 1]), 'y': array([2])}

        Grouping by multiple columns:

        >>> tsgroup.groupby(["l1","l2"])
        {(1, 'x'): array([0]), (2, 'x'): array([1]), (2, 'y'): array([2])}

        Filtering to a specific group using the output dictionary:

        >>> groups = tsgroup.groupby("l2")
        >>> tsgroup[groups["x"]]
          Index     rate    l1  l2
        -------  -------  ----  ----
              0  1.00629     1  x
              1  2.01258     2  x

        Filtering to a specific group using the get_group argument:

        >>> tsgroup.groupby("l2", get_group="x")
          Index     rate    l1  l2
        -------  -------  ----  ----
              0  1.00629     1  x
              1  2.01258     2  x
        """
        return _MetadataMixin.groupby(self, by, get_group)

    @add_meta_docstring("groupby_apply")
    def groupby_apply(
        self,
        by: Union[str, list[str]],
        func: Callable[..., Any],
        input_key: Optional[str] = None,
        **func_kwargs: Any,
    ) -> dict[Any, Any]:
        """
        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0:nap.Ts(t=np.arange(0,40), time_units='s'),
        ... 1:nap.Ts(t=np.arange(0,40,0.5), time_units='s'),
        ... 2:nap.Ts(t=np.arange(0,40,0.25), time_units='s'),
        ... }
        >>> metadata = {"l1": [1, 2, 2], "l2": ["x", "x", "y"]}
        >>> tsgroup = nap.TsGroup(tmp,metadata=metadata)
        >>> print(tsgroup)
          Index     rate    l1  l2
        -------  -------  ----  ----
              0  1.00629     1  x
              1  2.01258     2  x
              2  4.02516     2  y

        Apply a custom function:

        >>> tsgroup.groupby_apply("l2", lambda x: x.to_tsd().shape[0])
        {'x': 120, 'y': 160}

        Apply a function with additional arguments:

        >>> feature = nap.Tsd(
        ...     t=np.arange(40),
        ...     d=np.concatenate([np.zeros(20), np.ones(20)]),
        ...     time_support=nap.IntervalSet(np.array([[0, 5], [10, 12], [20, 33]])),
        ... )
        >>> print(tsgroup.groupby_apply("l2", nap.compute_tuning_curves, features=feature, bins=2))
        {'x': <xarray.DataArray (unit: 2, 0: 2)> Size: 32B
        array([[1.        , 1.        ],
               [1.77777778, 1.92857143]])
        Coordinates:
          * unit     (unit) int64 16B 0 1
          * 0        (0) float64 16B 0.25 0.75
        Attributes:
            occupancy:  [ 9. 14.]
            bin_edges:  [array([0. , 0.5, 1. ])]
            fs:         1.0
            rates:      [1.15 2.15], 'y': <xarray.DataArray (unit: 1, 0: 2)> Size: 16B
        array([[3.33333333, 3.78571429]])
        Coordinates:
          * unit     (unit) int64 8B 2
          * 0        (0) float64 16B 0.25 0.75
        Attributes:
            occupancy:  [ 9. 14.]
            bin_edges:  [array([0. , 0.5, 1. ])]
            fs:         1.0
            rates:      [4.15]}
        """
        return _MetadataMixin.groupby_apply(self, by, func, input_key, **func_kwargs)

    def subsample(self, fraction: float, seed: Optional[int] = None) -> TsGroup:
        """
        Keep a random fraction of the timestamps of each element.

        For each element with ``n`` timestamps, the function keeps
        ``round(n * fraction)`` timestamps, selected at random. ``round``
        rounds a half to the nearest even integer: with ``fraction=0.5``, 3
        timestamps give 2, and 5 timestamps give 2.

        Parameters
        ----------
        fraction : float
            The fraction of timestamps to keep, from 0 to 1.
        seed : int, optional
            The seed of the random number generator. The same seed gives the
            same timestamps. The default is None: each call gives different
            timestamps.

        Returns
        -------
        TsGroup
            A TsGroup with the same keys, time support and metadata. The
            function computes the ``rate`` metadata again. The kept
            timestamps stay in time order. Tsd, TsdFrame and TsdTensor
            elements keep the values of their kept timestamps.

        Raises
        ------
        TypeError
            If ``fraction`` is not a number.
        ValueError
            If ``fraction`` is not from 0 to 1.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {
        ...     0: nap.Ts(t=np.arange(0, 100)),
        ...     1: nap.Ts(t=np.arange(0, 100, 0.5)),
        ...     2: nap.Ts(t=np.arange(0, 100, 0.25)),
        ... }
        >>> tsgroup = nap.TsGroup(tmp)
        >>> tsgroup
          Index     rate
        -------  -------
              0  1.00251
              1  2.00501
              2  4.01003

        Keep 50% of the timestamps of each element. The rates are half of
        the rates before:

        >>> subsampled = tsgroup.subsample(0.5, seed=42)
        >>> subsampled
          Index     rate
        -------  -------
              0  0.50125
              1  1.00251
              2  2.00501
        """
        if not isinstance(fraction, Number):
            raise TypeError("fraction must be a number.")
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be between 0 and 1.")

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Keep exactly round(n * fraction) timestamps per unit: draw one random
        # value per timestamp of the unit and keep the n_keep smallest
        # (argpartition: O(n), no sort). Units are drawn in key order, only when
        # a choice is needed, so a given seed selects the same timestamps as
        # when units were stored as separate objects.
        order, offsets = self._ragged_index
        counts = np.diff(offsets)
        keep = np.zeros(len(self._times), dtype=bool)
        for i in range(len(self.index)):
            idx = order[offsets[i] : offsets[i + 1]]
            n_keep = int(np.round(counts[i] * fraction))
            if n_keep >= counts[i]:
                keep[idx] = True
            elif n_keep > 0:
                random_values = rng.random(counts[i])
                keep[idx[np.argpartition(random_values, n_keep)[:n_keep]]] = True

        cols = self._metadata.columns[1:]  # drop "rate"
        return TsGroup._from_arrays(
            self._times[keep],
            self._clusters[keep],
            None if self._data is None else self._data[keep],
            self._is_tsd,
            self.index,
            self.time_support,
            metadata=self._metadata[cols],
            columns=self._columns,
        )
