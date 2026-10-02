"""

The class `TsGroup` helps group objects with different timestamps
(i.e. timestamps of spikes of a population of neurons).

"""

import warnings
from collections import UserDict
from collections.abc import Hashable, Mapping
from functools import cached_property
from numbers import Number

import numpy
import numpy as np
import pandas as pd
from tabulate import tabulate

from ._core_functions import (
    _concat_ranges,
    _count,
    _count_grouped,
    _group_by_unit,
    _restrict_grouped,
    _time_diff_grouped,
    _value_from,
)
from ._jitted_functions import jitunion, jitunion_isets
from .base_class import _Base
from .config import nap_config
from .interval_set import IntervalSet
from .metadata_class import (
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
    unit_index : ndarray[int64]
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
    members = list(data.values())
    lengths = np.array([len(m) for m in members], dtype=np.int64)
    is_tsd = np.array([isinstance(m, _BaseTsd) for m in members], dtype=bool)

    if len(members):
        times = np.concatenate([m.index.values for m in members]).astype(
            np.float64, copy=False
        )
    else:
        times = np.empty(0, dtype=np.float64)
    unit_index = np.repeat(keys, lengths)

    values = _concat_values(
        [m.values if v else None for m, v in zip(members, is_tsd)], lengths
    )

    if len(times) > 1 and np.any(times[1:] < times[:-1]):
        order = np.argsort(times, kind="stable")
        times = times[order]
        unit_index = unit_index[order]
        if values is not None:
            values = values[order]

    return times, unit_index, values, is_tsd


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


class _TsGroupMembers(Mapping):
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


class TsGroup(UserDict, _MetadataMixin):
    """
    Dictionary-like object to group objects with different timestamps (for example timestamps of spikes of a population of neurons).

    Parameters
    ----------
    data : dict or iterable
        Dictionary or iterable of Ts/Tsd objects. The keys should be integer-convertible; if a non-dict iterator is
        passed, its values will be used to create a dict with integer keys.
    time_support : IntervalSet, optional
        The time support of the TsGroup. Ts/Tsd objects will be restricted to the time support if passed.
        If no time support is specified, TsGroup will merge time supports from all the Ts/Tsd objects in data.
    time_units : str, optional
        Time units if data does not contain Ts/Tsd objects ('us', 'ms', 's' [default]).
    bypass_check: bool, optional
        To avoid checking that each element is within time_support.
        Useful to speed up initialization of TsGroup when Ts/Tsd objects have already been restricted beforehand
    metadata: pd.DataFrame or dict, optional
        Metadata associated with each Ts/Tsd object. Metadata names are pulled from DataFrame columns or dictionary keys.
        The length of the metadata should match the number of Ts/Tsd objects.
    **kwargs
        Meta-info about the Ts/Tsd objects. Can be either pandas.Series, numpy.ndarray, list or tuple
        The index should match the index of the input dictionary if pandas Series.
        NOTE: This method of initializing metadata is deprecated and will be removed in a future version of Pynapple.

    Raises
    ------
    RuntimeError
        Raise error if the union of time support of Ts/Tsd object is empty.
    ValueError
        - If a key cannot be converted to integer.
        - If a key was a floating point with non-negligible decimal part.
        - If the converted keys are not unique, i.e. {1: ts_2, "2": ts_2} is valid,
            {1: ts_2, "1": ts_2}  is invalid.
        - If Tsd/TsdFrame/TsdTensor objects have different shapes after the time axis.

    Notes
    -----
    The group does not store one object per element: every timestamp is merged
    into one sorted array, along with the key of its element (and its value for
    Tsd, TsdFrame or TsdTensor elements). Operations over the whole group run once
    over that array instead of once per element. Consequently:

    - Accessing an element (``tsgroup[k]``, ``values()``, iteration) builds a new
      Ts/Tsd object on each call.
    - Values of all elements share one array, cast to a common dtype; TsdFrame
      elements share one set of column names (reset to default, with a warning,
      if they differ).
    - Timestamps shared by several elements are ordered by key.

    Examples
    --------
    Initialize a TsGroup as a dictionary of Ts/Tsd objects:

    >>> import pynapple as nap
    >>> import numpy as np
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

    Initialize a TsGroup as a list of Ts/Tsd objects:

    >>> data = [
    ...    nap.Ts(np.arange(100)),
    ...    nap.Ts(np.arange(0, 100, 2)),
    ...    nap.Ts(np.arange(0, 100, 3)),
    ... ]
    >>> tsgroup = nap.TsGroup(data)
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Initialize a TsGroup as a list of array (throws UserWarning):

    >>> data = [
    ...    np.arange(100),
    ...    np.arange(0, 100, 2),
    ...    np.arange(0, 100, 3),
    ... ]
    >>> tsgroup = nap.TsGroup(data)
    >>> tsgroup
      Index     rate
    -------  -------
          0  1.0101
          1  0.50505
          2  0.34343

    Initialize a TsGroup with metadata:

    >>> data = {
    ...    0: nap.Ts(np.arange(100)),
    ...    1: nap.Ts(np.arange(0, 100, 2)),
    ...    2: nap.Ts(np.arange(0, 100, 3)),
    ... }
    >>> metadata = {"label": ["A", "B", "C"]}
    >>> tsgroup = nap.TsGroup(data, metadata=metadata)
    >>> tsgroup
      Index     rate  label
    -------  -------  -------
          0  1.0101   A
          1  0.50505  B
          2  0.34343  C

    Initialize a TsGroup with metadata as a pandas DataFrame:

    >>> data = {
    ...    0: nap.Ts(np.arange(100)),
    ...    1: nap.Ts(np.arange(0, 100, 2)),
    ...    2: nap.Ts(np.arange(0, 100, 3)),
    ... }
    >>> metadata = pd.DataFrame(data=["A", "B", "C"], columns=["label"])
    >>> tsgroup = nap.TsGroup(data, metadata=metadata)
    >>> tsgroup
      Index     rate  label
    -------  -------  -------
          0  1.0101   A
          1  0.50505  B
          2  0.34343  C

    """

    index: np.ndarray
    """The index of the TsGroup, indicating the keys of each member"""

    time_support: IntervalSet
    """The time support of the TsGroup, indicating the time intervals where the TsGroup is defined"""

    nap_class: str
    """The pynapple class name"""

    def __init__(
        self,
        data,
        time_support=None,
        time_units="s",
        bypass_check=False,
        metadata=None,
        **kwargs,
    ):
        # Check input type
        if time_units not in ["s", "ms", "us"]:
            raise ValueError("Argument time_units should be 's', 'ms' or 'us'")
        if not isinstance(bypass_check, bool):
            raise TypeError("Argument bypass_check should be of type bool")
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

        data = {keys[j]: data[k] for j, k in enumerate(data.keys())}
        self.index = np.sort(keys)

        # Make sure data dict and index are ordered the same
        data = {k: data[k] for k in self.index}

        # Also sort metadata if more than one key
        if len(keys) > 1:
            sort_index = np.argsort(keys)
            if (metadata is not None) and (len(metadata) > 0):
                if hasattr(metadata, "index") and np.all(metadata.index != keys):
                    # check that index matches before sort if index exists
                    raise ValueError(
                        "Metadata index does not match the index of the TsGroup."
                    )
                metadata = {
                    key: np.array(value)[sort_index] for key, value in metadata.items()
                }
            if kwargs:
                # this should also check for index within individual kwargs,
                # but we should just deprecate this in the future
                kwargs = {
                    key: np.array(value)[sort_index] for key, value in kwargs.items()
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

        if not passed_time_support:
            # Otherwise do the union of all time supports
            time_support = _union_intervals([data[k].time_support for k in self.index])
            if len(time_support) == 0:
                raise RuntimeError(
                    "Union of time supports is empty. Consider passing a time support as argument."
                )
        self.time_support = time_support

        for k in self.index:
            if not isinstance(data[k], _Base):
                raise TypeError(
                    f"Element {k} of TsGroup should be a Ts, Tsd, TsdFrame or "
                    f"TsdTensor. {type(data[k])} provided instead."
                )

        # Every unit is merged into one sorted array, restricted once to the
        # time support (instead of once per unit).
        times, unit_index, values, is_tsd = _build_sorted_arrays(data)
        if not bypass_check:
            times, unit_index, values = _restrict_grouped(
                times, unit_index, values, time_support.start, time_support.end
            )
        self._set_arrays(
            times, unit_index, values, is_tsd, _shared_columns(data.values())
        )

        self._finalize(metadata, kwargs)

    @classmethod
    def _from_arrays(
        cls,
        times,
        unit_index,
        values,
        is_tsd,
        index,
        time_support,
        metadata=None,
        columns=None,
    ):
        """Build a TsGroup directly from its merged arrays, without validation.

        Bypasses ``__init__``, which only builds from a dict of per-unit objects:
        callers already holding merged arrays would otherwise split them into
        one Ts per unit just to have them merged again.

        ``times`` must be sorted, ``unit_index`` hold keys of ``index``, and
        ``index`` be sorted; ``metadata`` rows must follow ``index``. ``columns``
        names the columns of 2-dimensional members.
        """
        obj = cls.__new__(cls)
        obj.__dict__["_initialized"] = False
        obj.index = np.asarray(index, dtype=np.int64)
        _MetadataMixin.__init__(obj)
        obj.time_support = time_support
        obj._set_arrays(times, unit_index, values, is_tsd, columns)
        obj._finalize(metadata)
        return obj

    def _set_arrays(self, times, unit_index, values, is_tsd, columns=None):
        """Set the merged sorted arrays backing the group.

        ``_times`` holds every unit's timestamps in one sorted array,
        ``_unit_index`` the (real) key of each timestamp's unit, ``_data`` the
        matching values (None when no member holds values), ``_is_tsd`` which
        units hold values and ``_columns`` the column names of 2-dimensional
        members (None for the default ones).
        """
        self._times = times
        self._unit_index = unit_index
        self._data = values
        self._is_tsd = np.asarray(is_tsd, dtype=bool)
        self._columns = columns

    def _finalize(self, metadata=None, kwargs=None):
        """Compute rates, freeze the object and set metadata."""
        self._metadata["rate"] = self._compute_rates()
        self.nap_class = self.__class__.__name__
        # grab current attributes before adding metadata
        self._class_attributes = self.__dir__()
        self._class_attributes.append("_class_attributes")  # add this property

        # Making the TsGroup non mutable
        self._initialized = True

        # Trying to add argument as metainfo
        kwargs = kwargs or {}
        if len(kwargs):
            warnings.warn(
                "initializing metadata with variable keyword arguments may be unsupported in a future version of Pynapple. Instead, initialize using the metadata argument.",
                FutureWarning,
            )
        self.set_info(metadata, **kwargs)

    def _unit_counts(self):
        """Number of timestamps of each unit, following ``self.index``."""
        return np.bincount(self._unit_pos(), minlength=len(self.index))

    def _compute_rates(self):
        ts = self.time_support.values
        duration = np.sum(ts[:, 1] - ts[:, 0])
        counts = self._unit_counts()
        if duration > 0:
            return counts / duration
        return np.full(len(counts), np.nan)

    def _unit_pos(self, unit_index=None):
        """Dense ``0..n_units-1`` column of each timestamp's unit.

        Transient (never stored): only kernels whose output has one column per
        unit need it.
        """
        if unit_index is None:
            unit_index = self._unit_index
        if len(self.index) == 0:
            return np.zeros(len(unit_index), dtype=np.int64)
        # A dense key -> position table makes this one gather instead of a
        # binary search per timestamp (~3x faster than searchsorted on
        # time-sorted, hence key-shuffled, input). Fall back to searchsorted
        # when keys are too sparse for the table to stay small.
        lo = self.index[0]
        span = int(self.index[-1] - lo) + 1
        if span > 16 * len(self.index) + 4096:
            return np.searchsorted(self.index, unit_index)
        lut = np.zeros(span, dtype=np.int64)
        lut[self.index - lo] = np.arange(len(self.index))
        return lut[unit_index - lo]

    @cached_property
    def _unit_order(self):
        """``(order, offsets)`` such that ``order[offsets[i]:offsets[i + 1]]``
        are the positions in ``_times`` of unit ``self.index[i]``, in time order.

        Computed once on first per-unit access and cached, so that iterating
        over members costs one counting sort overall rather than a full scan per
        member.
        """
        return _group_by_unit(self._unit_pos(), len(self.index))

    @cached_property
    def _keys_set(self):
        """Keys as a set, for O(1) membership tests."""
        return set(self.index.tolist())

    def _make_member(self, i, idx):
        """Ts/Tsd/TsdFrame/TsdTensor of unit ``self.index[i]`` from the positions
        ``idx`` of its timestamps."""
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

    def _get_member(self, key):
        i = int(np.searchsorted(self.index, key))
        order, offsets = self._unit_order
        return self._make_member(i, order[offsets[i] : offsets[i + 1]])

    def _members(self):
        """All members, in index order."""
        order, offsets = self._unit_order
        return [
            self._make_member(i, order[offsets[i] : offsets[i + 1]])
            for i in range(len(self.index))
        ]

    def _take(self, keys):
        """New TsGroup holding only the units ``keys`` (sorted, unique)."""
        mask = np.isin(self._unit_index, keys)
        sel = np.searchsorted(self.index, keys)
        return TsGroup._from_arrays(
            self._times[mask],
            self._unit_index[mask],
            None if self._data is None else self._data[mask],
            self._is_tsd[sel],
            keys,
            self.time_support,
            metadata=self._metadata.loc[keys].copy().drop("rate"),
            columns=self._columns,
        )

    @property
    def data(self):
        """Read-only mapping from each key to its Ts/Tsd, built on access."""
        return _TsGroupMembers(self)

    def __getstate__(self):
        state = dict(self.__dict__)
        # derived and O(n_timestamps): recomputed on demand after unpickling
        state.pop("_unit_order", None)
        return state

    def __setstate__(self, state):
        state = dict(state)
        # objects pickled before the merged-array layout held a dict of members
        members = state.pop("data", None)
        self.__dict__.update(state)
        if "_times" not in state and members is not None:
            members = {k: members[k] for k in self.index}
            times, unit_index, values, is_tsd = _build_sorted_arrays(members)
            self.__dict__.update(
                _times=times,
                _unit_index=unit_index,
                _data=values,
                _is_tsd=is_tsd,
                _columns=_shared_columns(members.values()),
            )

    """
    Base functions
    """

    def __setattr__(self, name, value):
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
    def __getattr__(self, name):
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

    def __setitem__(self, key, value):
        if not self._initialized:
            # self._metadata.loc[int(key), "rate"] = float(value.rate)
            super().__setitem__(int(key), value)
        else:
            _MetadataMixin.__setitem__(self, key, value)

    @add_or_convert_metadata
    def __getitem__(self, key):
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
                    "Boolean index length must be equal to the number of Ts in the group! "
                    f"The number of Ts is {self.__len__()}, but the boolean array"
                    f"has length {len(key)} instead!"
                )
            key = self.index[key]

        keys_not_in = list(filter(lambda x: x not in self.index, key))

        if len(keys_not_in):
            raise KeyError(r"Key {} not in group index.".format(keys_not_in))

        return self._ts_group_from_keys(key)

    def _ts_group_from_keys(self, keys):
        return self._take(np.unique(np.asarray(keys, dtype=np.int64)))

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        return iter(self.index.tolist())

    def __contains__(self, key):
        try:
            return key in self._keys_set
        except TypeError:  # unhashable
            return False

    def __repr__(self):
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

    def __str__(self):
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

    def keys(self):
        """
        Return index/keys of TsGroup

        Returns
        -------
        list
            List of keys
        """
        return self.index.tolist()

    def items(self):
        """
        Return a list of key/object.

        Returns
        -------
        list
            List of tuples
        """
        return list(zip(self.index.tolist(), self._members()))

    def values(self):
        """
        Return a list of all the Ts/Tsd objects in the TsGroup

        Returns
        -------
        list
            List of Ts/Tsd objects
        """
        return self._members()

    @property
    def rates(self):
        """
        Return the rates of each element of the group in Hz
        """
        return self._metadata["rate"]

    def copy(self):
        """
        Return an exact copy of the TsGroup
        """
        import copy

        return copy.deepcopy(self)

    #################################
    # Generic functions of Tsd objects
    #################################
    def restrict(self, ep):
        """
        Restricts a TsGroup object to a set of time intervals delimited by an IntervalSet object

        Parameters
        ----------
        ep : IntervalSet
            the IntervalSet object

        Returns
        -------
        TsGroup
            TsGroup object restricted to ep

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

        All objects within the TsGroup automatically inherit the epochs defined by ep.

        >>> newtsgroup.time_support
          index    start    end
              0        0    100
        shape: (1, 2), time unit: sec.
        >>> newtsgroup[0].time_support
          index    start    end
              0        0    100
        shape: (1, 2), time unit: sec.
        """
        if not isinstance(ep, IntervalSet):
            raise TypeError("Argument should be IntervalSet")
        times, unit_index, values = _restrict_grouped(
            self._times, self._unit_index, self._data, ep.start, ep.end
        )
        cols = self._metadata.columns[1:]  # .drop("rate")

        return TsGroup._from_arrays(
            times,
            unit_index,
            values,
            self._is_tsd,
            self.index,
            ep,
            metadata=self._metadata[cols],
            columns=self._columns,
        )

    def value_from(self, tsd, ep=None, mode="closest"):
        """
        Replace the value of each Ts/Tsd object within the Ts group with the closest value from tsd argument

        Parameters
        ----------
        tsd : Tsd
            The Tsd object holding the values to replace
        ep : IntervalSet (optional)
            The IntervalSet object to restrict the operation.
            If None, the time support of the tsd input object is used.
        mode: literal, either 'closest', 'before', 'after'
            If closest, replace value with value from Tsd/TsdFrame/TsdTensor, if before gets the
            first value before, if after the first value after.

        Returns
        -------
        out : TsGroup
            TsGroup object with the new values

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp)
        >>> ep = nap.IntervalSet(start=0, end=100, time_units='s')

        The variable tsd is a time series object containing the values to assign, for example the tracking data:

        >>> tsd = nap.Tsd(t=np.arange(0,100), d=np.random.rand(100), time_units='s')
        >>> ep = nap.IntervalSet(start = 0, end = 100, time_units = 's')
        >>> newtsgroup = tsgroup.value_from(tsd, ep)

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
        times, values = _value_from(
            self._times, tsd.index.values, tsd.values, starts, ends, mode=mode
        )
        unit_index = _concat_ranges(
            self._unit_index,
            np.searchsorted(self._times, starts, side="left"),
            np.searchsorted(self._times, ends, side="right"),
            copy=True,
        )

        cols = self._metadata.columns[1:]  # .drop("rate")
        return TsGroup._from_arrays(
            times,
            unit_index,
            values,
            np.ones(len(self.index), dtype=bool),
            self.index,
            IntervalSet(start=starts, end=ends),
            metadata=self._metadata[cols],
            columns=tsd.columns if isinstance(tsd, TsdFrame) else None,
        )

    @add_or_convert_metadata
    def count(self, bin_size=None, ep=None, time_units="s", dtype=None):
        """
        Count occurrences of events within bin_size or within a set of bins defined as an IntervalSet.
        You can call this function in multiple ways :

        1. *tsgroup.count(bin_size=1, time_units = 'ms')*
        -> Count occurrence of events within a 1 ms bin defined on the time support of the object.

        2. *tsgroup.count(1, ep=my_epochs)*
        -> Count occurent of events within a 1 second bin defined on the IntervalSet my_epochs.

        3. *tsgroup.count(ep=my_bins)*
        -> Count occurent of events within each epoch of the intervalSet object my_bins

        4. *tsgroup.count()*
        -> Count occurent of events within each epoch of the time support.

        bin_size should be seconds unless specified.
        If bin_size is used and no epochs is passed, the data will be binned based on the time support of the object.

        Parameters
        ----------
        bin_size : None or float, optional
            The bin size (default is second)
        ep : None or IntervalSet, optional
            IntervalSet to restrict the operation
        time_units : str, optional
            Time units of bin size ('us', 'ms', 's' [default])
        dtype: type, optional
            Data type for the count. Default is np.int64.

        Returns
        -------
        out: TsdFrame
            A TsdFrame with the columns being the index of each item in the TsGroup.

        Examples
        --------
        This example shows how to count events within bins of 0.1 second for the first 100 seconds.

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

        if len(self) >= 1:
            time_index, count = _count_grouped(
                self._times,
                self._unit_pos(),
                len(self.index),
                starts,
                ends,
                bin_size,
                dtype=dtype,
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
        else:
            time_index, _ = _count(np.array([]), starts, ends, bin_size, dtype=dtype)
            return TsdFrame(
                t=time_index,
                d=np.empty((len(time_index), 0)),
                time_support=ep,
                metadata=self._metadata.copy().drop("rate"),
            )

    def to_tsd(self, *args):
        """
        Convert TsGroup to a Tsd. The timestamps of the TsGroup are merged together and sorted.

        Parameters
        ----------
        *args
            string, list, numpy.ndarray or pandas.Series

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tsgroup = nap.TsGroup({0:nap.Ts(t=np.array([0, 1])), 5:nap.Ts(t=np.array([2, 3]))})


        By default, the values of the Tsd is the index of the timestamp in the TsGroup:

        >>> tsgroup.to_tsd()
        Time (s)
        ----------  --
        0            0
        1            0
        2            5
        3            5
        dtype: float64, shape: (4,)

        Values can be inherited from the metadata of the TsGroup by giving the key of the corresponding columns.

        >>> tsgroup.set_info( phase=np.array([np.pi, 2*np.pi]) ) # assigning a phase to my 2 elements of the TsGroup
        >>> tsgroup.to_tsd("phase")
        Time (s)
        ----------  -------
        0           3.14159
        1           3.14159
        2           6.28319
        3           6.28319
        dtype: float64, shape: (4,)

        Values can also be passed directly to the function from a list, numpy.ndarray or pandas.Series of values as long as the length matches :

        >>> tsgroup.to_tsd([-1, 1])
        Time (s)
        ----------  --
        0           -1
        1           -1
        2            1
        3            1
        dtype: float64, shape: (4,)

        The reverse operation can be done with the Tsd.to_tsgroup function :

        >>> my_tsd = tsgroup.to_tsd()
        >>> my_tsd
        Time (s)
        ----------  --
        0            0
        1            0
        2            5
        3            5
        dtype: float64, shape: (4,)
        >>> my_tsd.to_tsgroup()
          Index    rate
        -------  ------
              0       1
              5       1

        Returns
        -------
        Tsd

        Raises
        ------
        RuntimeError
            "Metadata indices do not match" : if pandas.Series indexes don't match the TsGroup indexes
            "Values is not the same length" : if numpy.ndarray/list object is not the same size as the TsGroup object
            "Key not in metadata of TsGroup" : if string argument does not match any column names of the metadata,
            "Unknown argument format" ; if argument is not a string, list, numpy.ndarray or pandas.Series

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
            data[:] = np.asarray(_values)[self._unit_pos()]

        return Tsd(t=self._times, d=data, time_support=self.time_support)

    @add_or_convert_metadata
    def trial_count(
        self, ep, bin_size, align="start", padding_value=np.nan, time_unit="s"
    ):
        """
        Return trial-based count tensor from an IntervalSet object. The shape of the tensor array is
        (number of group elements, number of trials, number of time bins).

        The `bin_size` parameter determines the number of time bins.

        The `align` parameter controls how the time series are aligned. If `align="start"`, the time
        series are aligned to the start of each trial. If `align="end"`, the time series are aligned
        to the end of each trial.

        If trials have uneven durations, the returned array is padded. The parameter `padding_value`
        determine which value is used to pad the array. Default is NaN.

        Parameters
        ----------
        ep : IntervalSet
            Epochs holding the trials. Each interval can be of unequal size.
        bin_size : Number
            The size of the time bins.
        align: str, optional
            How to align the time series ('start' [default], 'end')
        padding_value: Number, optional
            How to pad the array if unequal intervals. Default is np.nan.
        time_unit : str, optional
            Time units of the bin_size parameter ('s' [default], 'ms', 'us').

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        RuntimeError
            If `time_unit` not in ["s", "ms", "us"]

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

        Create a trial-based tensor by counting events within 1 second bin for each interval of `ep`.

        >>> tensor = group.trial_count(ep, bin_size=1)
        >>> tensor
        array([[[ 1.,  1., nan, nan, nan, nan, nan, nan],
                [ 1.,  1.,  1.,  1., nan, nan, nan, nan],
                [ 1.,  1.,  1.,  1.,  1.,  1., nan, nan],
                [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]])

        By default, the time series are aligned to the start of the epochs. The parameter `align` control this behavior.

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

    def time_diff(self, align="center", epochs=None):
        """
        Computes the differences between subsequent timestamps.

        Parameters
        ----------
        align: str, optional
            Determines the time index of the resulting time differences:
             - "start" : the start of the interval between two timestamps.
             - "center" [default]: the center of the interval between two timestamps.
             - "end" : the end of the interval between two timestamps.
        epochs : IntervalSet, optional
            The epochs on which interspike intervals are computed.
            If None, the time support of the input is used.

        Returns
        -------
        dict
            A dictionary of Tsd containing the time differences for each Ts in the group.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=[1, 3, 5, 6, 8, 12], time_units='s'),1:nap.Ts(t=[2, 8, 9, 13, 14, 17], time_units='s'), 2:nap.Ts(t=[1, 2, 5, 7, 9, 12], time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp)
        >>> epochs = nap.IntervalSet(start=2, end=9, time_units='s')
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
            self._unit_pos(),
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

    def get(self, start, end=None, time_units="s"):
        """Slice the `TsGroup` object from `start` to `end` such that all the timestamps within the group satisfy `start<=t<=end`.
        If `end` is None, only the timepoint closest to `start` is returned.

        By default, the time support doesn't change. If you want to change the time support, use the `restrict` function.

        Parameters
        ----------
        start : float or int
            The start (or closest time point if `end` is None)
        end : float or int or None
            The end
        """
        cols = self._metadata.columns[1:]  # .drop("rate")

        if end is None:
            # closest timestamp of each unit: inherently per unit
            newgr = {k: m.get(start, end, time_units) for k, m in self.items()}
            return TsGroup(
                newgr,
                time_support=self.time_support,
                bypass_check=True,
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
            self._unit_index[sl],
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

    def getby_threshold(self, key, thr, op=">"):
        """
        Return a TsGroup with all Ts/Tsd objects with values above threshold for metainfo under key.

        Parameters
        ----------
        key : str
            One of the metainfo columns name
        thr : float
            THe value for thresholding
        op : str, optional
            The type of operation. Possibilities are '>', '<', '>=' or '<='.

        Returns
        -------
        TsGroup
            The new TsGroup

        Raises
        ------
        RuntimeError
            Raise error is operation is not recognized.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp)

        This example shows how to get a new TsGroup with all elements for which the rate is above 1.

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

    def getby_intervals(self, key, bins):
        """
        Return a list of TsGroup binned.

        Parameters
        ----------
        key : str
            One of the metainfo columns name
        bins : numpy.ndarray or list
            The bin intervals

        Returns
        -------
        list
            A list of TsGroup

        Examples
        --------

        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp, metadata={"alpha": np.arange(3)})

        This example shows how to bin the TsGroup according to one metainfo key.

        >>> newtsgroup, bincenter = tsgroup.getby_intervals('alpha', [0, 1, 2])
        >>> newtsgroup[0]
          Index     rate    alpha
        -------  -------  -------
              0  0.66722        0
        >>> newtsgroup[1]
          Index     rate    alpha
        -------  -------  -------
              1  1.33445        1

        By default, the function returns the center of the bins.

        >>> bincenter
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

    def getby_category(self, key):
        """
        Return a list of TsGroup grouped by category.

        Parameters
        ----------
        key : str
            One of the metainfo columns name

        Returns
        -------
        dict
            A dictionary of TsGroup

        Examples
        --------

        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = {0: nap.Ts(t=np.arange(0, 200), time_units='s'),
        ...        1: nap.Ts(t=np.arange(0, 200, 0.5), time_units='s'),
        ...        2: nap.Ts(t=np.arange(0, 300, 0.25), time_units='s')}
        >>> tsgroup = nap.TsGroup(tmp, metadata={"group": [0, 1, 1]})

        This example shows how to group the TsGroup according to one metainfo key.

        >>> newtsgroup = tsgroup.getby_category('group')
        >>> newtsgroup[0]
          Index     rate    group
        -------  -------  -------
              0  0.66722        0
        >>> newtsgroup[1]
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
        *tsgroups, reset_index=False, reset_time_support=False, ignore_metadata=False
    ):
        """
        Merge multiple TsGroup objects into a single TsGroup object.

        Parameters
        ----------
        *tsgroups : TsGroup
            The TsGroup objects to merge
        reset_index : bool, optional
            If True, the keys will be reset to range(len(data))
            If False, the keys of the TsGroup objects should be non-overlapping and will be preserved
        reset_time_support : bool, optional
            If True, the merged TsGroup will merge time supports from all the Ts/Tsd objects in data
            If False, the time support of the TsGroup objects should be the same and will be preserved
        ignore_metadata : bool, optional
            If True, the merged TsGroup will not have any metadata columns other than 'rate'
            If False, all metadata columns should be the same and all metadata will be concatenated

        Returns
        -------
        TsGroup
            A TsGroup of merged objects

        Raises
        ------
        TypeError
            If the input objects are not TsGroup objects
        ValueError
            If `ignore_metadata=False` but metadata columns are not the same
            If `reset_index=False` but keys overlap
            If `reset_time_support=False` but time supports are not the same

        """
        is_tsgroup = [isinstance(tsg, TsGroup) for tsg in tsgroups]
        if not all(is_tsgroup):
            not_tsgroup_index = [i + 1 for i, boo in enumerate(is_tsgroup) if not boo]
            raise TypeError(f"Input at positions {not_tsgroup_index} are not TsGroup!")

        if len(tsgroups) == 1:
            print("Only one TsGroup object provided, no merge needed.")
            return tsgroups[0]

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
        unit_index = []
        offset = 0
        for tsg in tsgroups:
            if reset_index:
                unit_index.append(tsg._unit_pos() + offset)
                offset += len(tsg.index)
            else:
                unit_index.append(tsg._unit_index)
        unit_index = np.concatenate(unit_index).astype(np.int64, copy=False)
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
        unit_index = unit_index[order]
        if values is not None:
            values = values[order]

        times, unit_index, values = _restrict_grouped(
            times, unit_index, values, time_support.start, time_support.end
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
            metadata = metadata.iloc[sort_index]

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
            unit_index,
            values,
            is_tsd,
            index,
            time_support,
            metadata=metadata,
            columns=columns,
        )

    def merge(
        self,
        *tsgroups,
        reset_index=False,
        reset_time_support=False,
        ignore_metadata=False,
    ):
        """
        Merge the TsGroup object with other TsGroup objects.
        Common uses include adding more neurons/channels (supposing each Ts/Tsd corresponds to data from a neuron/channel) or adding more trials (supposing each Ts/Tsd corresponds to data from a trial).

        Parameters
        ----------
        *tsgroups : TsGroup
            The TsGroup objects to merge with
        reset_index : bool, optional
            If True, the keys will be reset to range(len(data))
            If False, the keys of the TsGroup objects should be non-overlapping and will be preserved
        reset_time_support : bool, optional
            If True, the merged TsGroup will merge time supports from all the Ts/Tsd objects in data
            If False, the time support of the TsGroup objects should be the same and will be preserved
        ignore_metadata : bool, optional
            If True, the merged TsGroup will not have any metadata columns other than 'rate'
            If False, all metadata columns should be the same and all metadata will be concatenated

        Returns
        -------
        TsGroup
            A TsGroup of merged objects

        Raises
        ------
        TypeError
            If the input objects are not TsGroup objects
        ValueError
            If `ignore_metadata=False` but metadata columns are not the same
            If `reset_index=False` but keys overlap
            If `reset_time_support=False` but time supports are not the same

        Examples
        --------

        >>> import pynapple as nap
        >>> time_support_a = nap.IntervalSet(start=-1, end=1, time_units='s')
        >>> time_support_b = nap.IntervalSet(start=-5, end=5, time_units='s')

        >>> dict1 = {0: nap.Ts(t=[-1, 0, 1], time_units='s')}
        >>> tsgroup1 = nap.TsGroup(dict1, time_support=time_support_a)

        >>> dict2 = {10: nap.Ts(t=[-1, 0, 1], time_units='s')}
        >>> tsgroup2 = nap.TsGroup(dict2, time_support=time_support_a)

        >>> dict3 = {0: nap.Ts(t=[-.1, 0, .1], time_units='s')}
        >>> tsgroup3 = nap.TsGroup(dict3, time_support=time_support_a)

        >>> dict4 = {10: nap.Ts(t=[-1, 0, 1], time_units='s')}
        >>> tsgroup4 = nap.TsGroup(dict2, time_support=time_support_b)

        Merge with default options if have the same time support and non-overlapping indexes:

        >>> tsgroup_12 = tsgroup1.merge(tsgroup2)
        >>> tsgroup_12
          Index    rate
        -------  ------
              0     1.5
             10     1.5

        Set `reset_index=True` if indexes are overlapping:

        >>> tsgroup_13 = tsgroup1.merge(tsgroup3, reset_index=True)
        >>> tsgroup_13
          Index    rate
        -------  ------
              0     1.5
              1     1.5

        Set `reset_time_support=True` if time supports are different:

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

        See Also
        --------
        [`TsGroup.merge_group`](./#pynapple.core.ts_group.TsGroup.merge_group)
        """
        return TsGroup.merge_group(
            self,
            *tsgroups,
            reset_index=reset_index,
            reset_time_support=reset_time_support,
            ignore_metadata=ignore_metadata,
        )

    @add_or_convert_metadata
    def save(self, filename):
        """
        Save TsGroup object in npz format. The file will contain the timestamps,
        the data (if group of Tsd), group index, the time support and the metadata

        The main purpose of this function is to save small/medium sized TsGroup
        objects.

        The function will "flatten" the TsGroup by sorting all the timestamps
        and assigning to each the corresponding index. Typically, a TsGroup like
        this :

        TsGroup({
            0 : Tsd(t=[0, 2, 4], d=[1, 2, 3])
            1 : Tsd(t=[1, 5], d=[5, 6])})

        will be saved as npz with the following keys:

        {
            't' : [0, 1, 2, 4, 5],
            'd' : [1, 5, 2, 3, 5],
            'index' : [0, 1, 0, 0, 1],
            'start' : [0],
            'end' : [5],
            'keys' : [0, 1],
            'type' : 'TsGroup'
        }

        Metadata are saved by columns with the column name as the npz key. To avoid
        potential conflicts, make sure the columns name of the metadata are different
        from ['t', 'd', 'start', 'end', 'index', 'keys']

        You can load the object with `nap.load_file`. Default keys are 't', 'd'(optional),
        'start', 'end', 'index', 'keys' and 'type'.
        See the example below.

        Parameters
        ----------
        filename : str
            The filename

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

        To get back to pynapple, you can use the `nap.load_file` function :

        >>> tsgroup = nap.load_file("my_tsgroup.npz")
        >>> tsgroup
          Index    rate    group  location
        -------  ------  -------  ----------
              0     0.6        0  right foot
              6     0.4        1  left foot

        Raises
        ------
        RuntimeError
            If filename is not str, path does not exist or filename is a directory.
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
        dicttosave["index"] = self._unit_index
        if self._data is not None:
            data = np.full(self._data.shape, np.nan)
            valued = self._is_tsd[self._unit_pos()]
            data[valued] = self._data[valued]
            if not np.all(np.isnan(data)):
                dicttosave["d"] = data
        dicttosave["keys"] = np.array(self.keys())
        dicttosave["start"] = self.time_support.start
        dicttosave["end"] = self.time_support.end

        np.savez(filename, **dicttosave)

        return

    @classmethod
    def _from_npz_reader(cls, file):
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
    def set_info(self, metadata=None, **kwargs):
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
    def get_info(self, key):
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
    def drop_info(self, key):
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
    def restrict_info(self, key):
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
    def groupby(self, by, get_group=None):
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
    def groupby_apply(self, by, func, input_key=None, **func_kwargs):
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

    def subsample(self, fraction, seed=None):
        """
        Randomly subsample timestamps in each element of the TsGroup.

        This function randomly selects a fraction of the timestamps from each
        Ts/Tsd object in the TsGroup using a fast mask-based approach.

        Parameters
        ----------
        fraction : float
            The fraction of timestamps to keep. Must be between 0 and 1.
        seed : int, optional
            Random seed for reproducibility. If None, no seed is set.

        Returns
        -------
        TsGroup
            A new TsGroup with subsampled timestamps.

        Raises
        ------
        ValueError
            If fraction is not between 0 and 1.
        TypeError
            If fraction is not a number.

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

        Subsample to keep exactly 50% of the timestamps:

        >>> subsampled = tsgroup.subsample(0.5, seed=42)

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
        # value per timestamp and keep, within each unit, the n_keep smallest
        # (argpartition: O(n) per unit, no sort).
        order, offsets = self._unit_order
        counts = np.diff(offsets)
        n_keep = np.round(counts * fraction).astype(np.int64)
        random_values = rng.random(len(self._times))
        keep = np.zeros(len(self._times), dtype=bool)
        for i in range(len(self.index)):
            idx = order[offsets[i] : offsets[i + 1]]
            if n_keep[i] >= counts[i]:
                keep[idx] = True
            elif n_keep[i] > 0:
                part = np.argpartition(random_values[idx], n_keep[i])
                keep[idx[part[: n_keep[i]]]] = True

        cols = self._metadata.columns[1:]  # drop "rate"
        return TsGroup._from_arrays(
            self._times[keep],
            self._unit_index[keep],
            None if self._data is None else self._data[keep],
            self._is_tsd,
            self.index,
            self.time_support,
            metadata=self._metadata[cols],
            columns=self._columns,
        )
