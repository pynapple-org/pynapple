"""
Pynapple class to interface with NWB files.
Data are always lazy-loaded.
Object behaves like dictionary.
"""

import errno
import importlib
import os
import warnings
from collections import UserDict
from numbers import Number
from pathlib import Path

import numpy as np
from tabulate import tabulate

from .. import core as nap
from ..core._core_functions import _restrict_arrays
from ..core._jitted_functions import jitunion_isets
from ..core.metadata_class import _MetadataMixin
from ..core.ts_group import TsGroup


def _get_unique_identifier(full_path_to_key):
    out, count = np.unique(list(full_path_to_key.values()), return_counts=True)
    if len(out) != len(full_path_to_key):
        key_to_change = out[count > 1]
        # Filter for ambiguous keys only
        update_dict = {
            key: val
            for key, val in full_path_to_key.items()
            if full_path_to_key[key] in key_to_change
        }
        for full_path, key in update_dict.items():
            # Adding the most immediate parent path until disambiguation
            base_parts = full_path.split("/")
            relative_parts = key.split("/")
            new_key = "/".join(base_parts[-len(relative_parts) - 1 :])
            if new_key.startswith("/"):
                new_key = new_key[1:]
            update_dict[full_path] = new_key
        update_dict = _get_unique_identifier(update_dict)
        full_path_to_key.update(update_dict)
    return full_path_to_key


def _get_full_path(path, obj):
    if hasattr(obj, "parent"):  # Better be safe here
        if obj.parent is None:
            return "/" + path
        else:
            if hasattr(obj.parent, "name"):  # and extra safe
                if obj.parent.name == "root":
                    return "/" + path
                else:
                    return _get_full_path(obj.parent.name + "/" + path, obj.parent)
            else:
                return "/" + path
    else:
        return "/" + path


def iterate_over_nwb(nwbfile):
    pynwb = importlib.import_module("pynwb")
    for oid, obj in nwbfile.objects.items():
        if isinstance(obj, pynwb.misc.DynamicTable) and any(
            i.name.endswith("_times_index") for i in obj.columns
        ):
            # data["units"] = {"id": oid, "type": "TsGroup"}
            yield obj, {"id": oid, "type": "TsGroup"}

        elif isinstance(obj, pynwb.epoch.TimeIntervals):
            # Supposedly IntervalsSets
            yield obj, {"id": oid, "type": "IntervalSet"}

        elif isinstance(obj, pynwb.misc.DynamicTable) and any(
            i.name.endswith("_times") for i in obj.columns
        ):
            # Supposedly Timestamps
            yield obj, {"id": oid, "type": "Ts"}

        elif isinstance(obj, pynwb.misc.AnnotationSeries):
            # Old timestamps version
            yield obj, {"id": oid, "type": "Ts"}

        elif isinstance(obj, pynwb.misc.TimeSeries):
            if len(obj.data.shape) > 2:
                yield obj, {"id": oid, "type": "TsdTensor"}

            elif len(obj.data.shape) == 2:
                yield obj, {"id": oid, "type": "TsdFrame"}

            elif len(obj.data.shape) == 1:
                yield obj, {"id": oid, "type": "Tsd"}


def _extract_compatible_data_from_nwbfile(nwbfile):
    """Extract all the NWB objects that can be converted to a pynapple object. If two objects have the same names, they
    are distinguished by adding their module name to their path.

    Parameters
    ----------
    nwbfile : pynwb.file.NWBFile
        Instance of NWB file

    Returns
    -------
    dict
        Dictionary containing all the object found and their type in pynapple.
    """
    return {
        _get_full_path(obj.name, obj): out for obj, out in iterate_over_nwb(nwbfile)
    }


def _make_interval_set(obj, **kwargs):
    """Helper function to make IntervalSet

    Parameters
    ----------
    obj : pynwb.epoch.TimeIntervals
        NWB object

    Returns
    -------
    IntervalSet or dict of IntervalSet or pandas.DataFrame
        If contains multiple epochs, a dictionary of IntervalSet is returned.
        It too many metadata, the function returns the output of nwbfile.trials.to_dataframe()
    """
    if hasattr(obj, "to_dataframe"):
        df = obj.to_dataframe()

        if hasattr(df, "start_time") and hasattr(df, "stop_time"):
            df = df.rename(columns={"start_time": "start", "stop_time": "end"})
            # create from full dataframe to ensure that metadata is associated correctly
            data = nap.IntervalSet(df)
            return data

    else:
        return obj


def _make_tsd(obj, lazy_loading=True):
    """Helper function to make Tsd

    Parameters
    ----------
    obj : pynwb.misc.TimeSeries
        NWB object
    lazy_loading: bool
        If True return a memory-view of the data, load otherwise.

    Returns
    -------
    Tsd

    """

    d = obj.data
    if not lazy_loading:
        d = d[:]

    if obj.timestamps is not None:
        t = obj.timestamps[:]
    else:
        t = obj.starting_time + np.arange(obj.num_samples) / obj.rate

    data = nap.Tsd(t=t, d=d, load_array=not lazy_loading)

    return data


def _make_tsd_tensor(obj, lazy_loading=True):
    """Helper function to make TsdTensor

    Parameters
    ----------
    obj : pynwb.misc.TimeSeries
        NWB object
    lazy_loading: bool
        If True return a memory-view of the data, load otherwise.

    Returns
    -------
    Tsd

    """

    d = obj.data
    if not lazy_loading:
        d = d[:]

    if obj.timestamps is not None:
        t = obj.timestamps[:]
    else:
        t = obj.starting_time + np.arange(obj.num_samples) / obj.rate

    data = nap.TsdTensor(t=t, d=d, load_array=not lazy_loading)

    return data


def _extract_dynamic_table_metadata(region):
    """Helper function to extract metadata from a DynamicTableRegion

    Columns holding one array per row (e.g. the `image_mask` of a
    `PlaneSegmentation`) are skipped based on the table schema, so they are
    never loaded only to be dropped. They can be orders of magnitude larger
    than the metadata itself.

    Parameters
    ----------
    region : hdmf.common.table.DynamicTableRegion
        The `electrodes` of an ElectricalSeries or the `rois` of a
        RoiResponseSeries.

    Returns
    -------
    pandas.DataFrame
        One row per element of the region, indexed by table id.

    """
    vector_index = importlib.import_module("hdmf.common.table").VectorIndex

    exclude = set()
    for name in region.table.colnames:
        column = region.table[name]
        if isinstance(column, vector_index):  # ragged, e.g. pixel_mask
            exclude.add(name)
        elif len(column.data) and np.ndim(column.data[0]):  # e.g. image_mask
            exclude.add(name)

    return (
        region.to_dataframe(exclude=exclude)
        .convert_dtypes()
        .select_dtypes(exclude="object")
    )


def _make_tsd_frame(obj, lazy_loading=True):
    """Helper function to make TsdFrame

    Parameters
    ----------
    obj : pynwb.misc.TimeSeries
        NWB object
    lazy_loading: bool
        If True return a memory-view of the data, load otherwise.

    Returns
    -------
    Tsd

    """
    pynwb = importlib.import_module("pynwb")

    d = obj.data
    metadata = {}
    if not lazy_loading:
        d = d[:]

    if obj.timestamps is not None:
        t = obj.timestamps[:]
    else:
        t = obj.starting_time + np.arange(obj.num_samples) / obj.rate

    if isinstance(obj, pynwb.behavior.SpatialSeries):
        if obj.data.shape[1] == 2:
            columns = ["x", "y"]
        elif obj.data.shape[1] == 3:
            columns = ["x", "y", "z"]
        else:
            columns = np.arange(obj.data.shape[1])

    elif isinstance(obj, pynwb.ecephys.ElectricalSeries):
        # (channel mapping)
        try:
            metadata = _extract_dynamic_table_metadata(obj.electrodes)
            columns = metadata.index
        except Exception:
            columns = np.arange(obj.data.shape[1])

    elif isinstance(obj, pynwb.ophys.RoiResponseSeries):
        # (cell number)
        try:
            metadata = _extract_dynamic_table_metadata(obj.rois)
            columns = metadata.index
        except Exception:
            columns = np.arange(obj.data.shape[1])

    else:
        columns = np.arange(obj.data.shape[1])

    if len(columns) >= d.shape[1]:  # Weird sometimes if background ID added
        columns = columns[0 : obj.data.shape[1]]
        if len(metadata):
            metadata = metadata.iloc[0 : obj.data.shape[1]]
    else:
        # Columns fell back to a range index, so the metadata no longer applies
        columns = np.arange(obj.data.shape[1])
        metadata = {}

    data = nap.TsdFrame(
        t=t,
        d=d,
        columns=columns,
        load_array=not lazy_loading,
        metadata=metadata,
    )

    return data


def _tsgroup_from_state(state):
    """Unpickle a regular TsGroup (see ``_NWBLazyTsGroup.__reduce_ex__``)."""
    obj = TsGroup.__new__(TsGroup)
    obj.__setstate__(state)
    return obj


def _support_and_counts(first, last, counts):
    """Time support and per-unit counts of a units table, as eager construction
    gives them.

    Parameters
    ----------
    first, last : ndarray
        Earliest and latest spike of each non-empty unit, in table order.
    counts : ndarray
        Number of spikes of each unit, in table order.

    Returns
    -------
    time_support : IntervalSet
        Union of each unit's ``[first, last]`` span. A zero-length span (single
        spike) is dropped, as IntervalSet does.
    counts : ndarray
        ``counts``, with 0 for single-spike units whose spike falls outside the
        time support (eager construction drops it).
    """
    nonempty = counts > 0
    spans = last > first
    ts_start, ts_end = jitunion_isets(first[spans], last[spans])
    if len(ts_start) == 0:
        raise RuntimeError(
            "Union of time supports is empty. Consider passing a time support as argument."
        )

    # spikes outside the time support are dropped: only possible for
    # single-spike units, whose spike is `first`
    is_single = ~spans & (counts[nonempty] == 1)
    if np.any(is_single):
        single = np.flatnonzero(nonempty)[is_single]
        t = first[is_single]
        k = np.searchsorted(ts_start, t, side="right") - 1
        outside = (k < 0) | (t > ts_end[np.maximum(k, 0)])
        counts = counts.copy()
        counts[single[outside]] = 0

    return nap.IntervalSet(ts_start, ts_end), counts


class _NWBLazyTsGroup(TsGroup):
    """TsGroup over an NWB units table. It reads the spike times only when an
    operation needs them, and it never keeps them.

    The units table stores the spike times of all the units in one ragged
    array on disk. The NWB column ``spike_times`` (here ``ragged_array``)
    holds the spike times of each unit, one unit after the other. The NWB
    column ``spike_times_index`` (here ``ragged_array_index``) holds the end
    position of each unit in that array.

    The constructor reads only the index and the first and last spike of each
    unit. These give the keys, the time support and the rates.

    Each operation reads only what it needs into a temporary regular TsGroup,
    and runs on that group:

    - A selection of units (``units[k]``, ``units[[k1, k2]]``, ``getby_*``)
      reads only the selected units.
    - ``restrict(ep)``, ``get(start, end)`` and the operations with an epoch
      argument (``count``, ``value_from``, ``time_diff``) read, in each unit,
      only the spikes from the start of the first epoch to the end of the
      last epoch. A binary search in the file finds these positions.
    - The other operations (``to_tsd``, ``count`` without ``ep``, iteration,
      ``copy``, ...) read all the spikes, and drop them after the operation.

    A selection gives a regular TsGroup, not a lazy one.

    The time support and the binary search assume that the spikes of each
    unit are sorted in the file. A read that finds unsorted spikes gives a
    warning.

    After ``NWBFile.close()``, an operation that reads spikes raises a
    RuntimeError.
    """

    def __init__(self, ragged_array, ragged_array_index, ids, metadata=None):
        """
        Parameters
        ----------
        ragged_array : array-like
            Flat spike times of every unit (h5py dataset for a file on disk).
        ragged_array_index : array-like
            One past the last spike of each unit in ``ragged_array``.
        ids : array-like of int
            Unit ids, in table order.
        metadata : dict, optional
            Metadata columns, in table order.
        """
        self.__dict__["_initialized"] = False

        ids = np.asarray(ids, dtype=np.int64)
        stops = np.asarray(ragged_array_index, dtype=np.int64)
        starts = np.concatenate([[0], stops[:-1]]).astype(np.int64)
        counts = stops - starts

        if not hasattr(ragged_array, "shape"):  # in-memory lists
            ragged_array = np.asarray(ragged_array, dtype=np.float64)

        # first and last spike of each non-empty unit (increasing positions, as
        # h5py point selection requires)
        nonempty = counts > 0
        first = np.asarray(ragged_array[starts[nonempty]], dtype=np.float64)
        last = np.asarray(ragged_array[stops[nonempty] - 1], dtype=np.float64)

        time_support, counts = _support_and_counts(first, last, counts)

        sort_index = np.argsort(ids, kind="stable")
        self.index = ids[sort_index]
        _MetadataMixin.__init__(self)
        self.time_support = time_support

        # everything below is in table order, except `_counts`
        self._ragged_array = ragged_array
        self._file_closed = False
        self._table_starts = starts
        self._table_stops = stops
        self._table_first = np.full(len(ids), np.nan)
        self._table_first[nonempty] = first
        self._table_last = np.full(len(ids), np.nan)
        self._table_last[nonempty] = last
        self._sort_index = sort_index
        self._counts = counts[sort_index]
        # spike times only: no values, no columns
        self._data = None
        self._is_tsd = np.zeros(len(ids), dtype=bool)
        self._columns = None

        if metadata:
            metadata = {k: np.asarray(v)[sort_index] for k, v in metadata.items()}
        self._finalize(metadata)

    def _unit_counts(self):
        # known from the ragged index: rates need no spike read
        return self._counts

    #################################
    # Read from the file
    #################################

    def _search(self, lo, hi, t, right):
        """Position of ``t`` in the sorted spikes ``ragged_array[lo:hi]``, as
        ``np.searchsorted`` gives it. In a file, it reads one spike for each
        step of a binary search."""
        array = self._ragged_array
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

    def _read_rows(self, keys, lo, hi):
        """Read ``ragged_array[lo[i]:hi[i]]`` for each unit ``keys[i]`` into a
        regular TsGroup with the time support of this group.

        ``keys`` must be sorted keys of the group.
        """
        if self.__dict__.get("_file_closed", False):
            raise RuntimeError(
                "The NWB file is closed: the spike times of this units group "
                "cannot be read. Select the units or the time that you need "
                "(e.g. units[[0, 1]] or units.restrict(ep)) before you close "
                "the file."
            )
        lengths = hi - lo
        try:
            array = self._ragged_array
            n_total = int(self._table_stops[-1]) if len(self._table_stops) else 0
            if len(lengths) and np.sum(lengths) == n_total:
                # every spike: one read is faster than one read for each unit
                array = np.asarray(array[:n_total])
            parts = [
                np.asarray(array[a:b], dtype=np.float64)
                for a, b in zip(lo, hi)
                if b > a
            ]
        except (ValueError, OSError, KeyError) as err:
            raise RuntimeError(
                "Cannot read the spike times from the NWB file."
            ) from err
        times = np.concatenate(parts) if parts else np.zeros(0)
        clusters = np.repeat(np.asarray(keys, dtype=np.int64), lengths)

        # the time support and the binary search assume sorted units
        ends = np.cumsum(lengths)
        unsorted = np.setdiff1d(np.flatnonzero(times[1:] < times[:-1]) + 1, ends)
        if len(unsorted):
            bad = np.unique(clusters[unsorted]).tolist()
            warnings.warn(
                f"Spike times of units {bad} are not sorted in the NWB file. "
                "The lazy units group assumes sorted spike times: its time "
                "support, its rates and its time windows can be wrong. Load "
                "the file with lazy_loading=False.",
                stacklevel=3,
            )

        # sorted by time, then by key for equal times (as TsGroup does)
        order = np.lexsort((clusters, times))
        ts = self.time_support
        times, clusters = _restrict_arrays(
            times[order], ts.start, ts.end, clusters[order]
        )
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

        Only the spikes ``t`` with ``start <= t <= end`` (in seconds) are read.
        A binary search in the file finds them in the units that are only
        partly in ``[start, end]``.
        """
        if keys is None:
            keys = self.index
        keys = np.unique(np.asarray(keys, dtype=np.int64))
        rows = self._sort_index[np.searchsorted(self.index, keys)]
        lo = self._table_starts[rows].copy()
        hi = self._table_stops[rows].copy()
        first = self._table_first[rows]
        last = self._table_last[rows]

        # units with no spike in [start, end] (an empty unit has a NaN first)
        if start > end:
            outside = np.ones(len(rows), dtype=bool)
        else:
            outside = ~(first <= end) | ~(last >= start)
        hi[outside] = lo[outside]
        for i in np.flatnonzero(~outside & (first < start)):
            lo[i] = self._search(lo[i], hi[i], start, right=False)
        for i in np.flatnonzero(~outside & (last > end)):
            hi[i] = self._search(lo[i], hi[i], end, right=True)
        # with unsorted spikes, the binary search can give hi < lo
        hi = np.maximum(hi, lo)

        return self._read_rows(keys, lo, hi)

    def _loaded(self, ep=None):
        """Read the spikes in the span of ``ep`` (default: all the spikes)."""
        if ep is None:
            return self._read()
        if not isinstance(ep, nap.IntervalSet):
            # read nothing: the method of the regular group raises its error
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
            s, e = nap.TsIndex.format_timestamps(np.array([start, end]), time_units)
            return self._read(start=s, end=e).get(start, end, time_units)

        # the closest spike of each unit is one of the two spikes around `start`
        t = nap.TsIndex.format_timestamps(np.array([start]), time_units)[0]
        keys = self.index
        rows = self._sort_index
        lo, hi = self._table_starts[rows], self._table_stops[rows]
        pos = np.array([self._search(a, b, t, right=False) for a, b in zip(lo, hi)])
        group = self._read_rows(keys, np.maximum(lo, pos - 1), np.minimum(hi, pos + 1))
        return group.get(start, end, time_units)

    def count(self, bin_size=None, ep=None, time_units="s", dtype=None):
        return self._loaded(ep).count(bin_size, ep, time_units, dtype)

    def value_from(self, tsd, ep=None, mode="closest"):
        # without `ep`, `value_from` uses the time support of `tsd`
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
        # the h5py dataset cannot be pickled or deep-copied: hand over a regular
        # TsGroup holding the spikes instead
        return (_tsgroup_from_state, (self._read().__getstate__(),))

    # Code that reads the merged arrays directly gets them from a full read,
    # which is not kept. Callers in pynapple use `_loaded` first.
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

    def _close_file(self):
        """Mark the file as closed: later reads raise a RuntimeError."""
        self.__dict__["_file_closed"] = True


def _make_tsgroup(obj, lazy_loading=True, **kwargs):
    """Helper function to make TsGroup

    Parameters
    ----------
    obj : pynwb.misc.Units
        NWB object

    Returns
    -------
    TsGroup

    """
    pynwb = importlib.import_module("pynwb")
    index = obj.id[:]

    N = len(index)
    metainfo = {}
    for coln in obj.colnames:
        if coln == "electrode_group":
            for e in [
                "location",
                "x",
                "y",
                "z",
                "imp",
                "filtering",
                "rel_x",
                "rel_y",
                "rel_z",
                "reference",
            ]:
                tmp = [eg.__getattribute__(e) for eg in obj[coln] if hasattr(eg, e)]
                if len(tmp) == N:
                    metainfo[e] = np.array(tmp)

        if coln not in ["spike_times_index", "spike_times", "electrode_group"]:
            col = obj[coln]
            if len(col) == N:
                if hasattr(col, "to_dataframe"):
                    # df rows are already in the correct unit order
                    df = col.to_dataframe()
                    for k in df.columns:
                        column_not_yet_set = k not in metainfo
                        if column_not_yet_set and not isinstance(
                            df[k].values[0],
                            (
                                list,
                                tuple,
                                dict,
                                set,
                                pynwb.ecephys.ElectrodeGroup,
                            ),
                        ):
                            metainfo[k] = df[k].values
                # elif not isinstance(col[0], (np.ndarray, list, tuple, dict, set)):
                elif isinstance(col[0], (Number, str)):
                    metainfo[coln] = np.array(col[:])
                else:
                    pass

    if not lazy_loading:
        # read every unit now, as a regular TsGroup
        units = {
            i: nap.Ts(t=np.array(t)) for i, t in zip(index, obj.spike_times_index[:])
        }
        return nap.TsGroup(units, metadata=metainfo)

    # spike times are only read when an operation needs them
    return _NWBLazyTsGroup(
        obj.spike_times.data,
        obj.spike_times_index.data[:],
        index,
        metadata=metainfo,
    )


def _make_ts(obj, **kwargs):
    """Helper function to make Ts

    Parameters
    ----------
    obj : pynwb.misc.AnnotationSeries or pynwb.misc.DynamicTable
        NWB object

    Returns
    -------
    Ts or dict of Ts

    """
    if hasattr(obj, "timestamps"):
        data = nap.Ts(obj.timestamps[:])
    else:
        df = obj.to_dataframe()
        data = {}
        for k in df.columns:
            if isinstance(k, str):
                if k.endswith("_times"):
                    data[k] = nap.Ts(df[k].values)
        if len(data) == 1:
            data = data[list(data.keys())[0]]

    return data


class NWBFile(UserDict):
    """Class for reading NWB Files.


    Examples
    --------
    >>> import pynapple as nap
    >>> data = nap.load_file("my_file.nwb")
    >>> data["units"]
      Index    rate  location      group
    -------  ------  ----------  -------
          0    1.0  brain        0
          1    1.0  brain        0
          2    1.0  brain        0

    """

    _f_eval = {
        "IntervalSet": _make_interval_set,
        "Tsd": _make_tsd,
        "Ts": _make_ts,
        "TsdFrame": _make_tsd_frame,
        "TsdTensor": _make_tsd_tensor,
        "TsGroup": _make_tsgroup,
    }

    def __init__(self, file, lazy_loading=True):
        """
        Parameters
        ----------
        file : str or pynwb.file.NWBFile
            Valid file to a NWB file
        lazy_loading: bool
            If True return a memory-view of the data, load otherwise.

        Raises
        ------
        FileNotFoundError
            If path is invalid
        RuntimeError
            If file is not an instance of NWBFile
        """
        # TODO: do we really need to have instantiation from file and object in the same place?
        pynwb = importlib.import_module("pynwb")
        NWBHDF5IO = pynwb.NWBHDF5IO
        if isinstance(file, pynwb.file.NWBFile):
            self.nwb = file
            self.name = self.nwb.session_id
        else:
            path = Path(file)

            if path.exists():
                self.path = path
                self.name = path.stem
                self.io = NWBHDF5IO(path, "r")
                self.nwb = self.io.read()
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)

        # Get a dictionary with full_path -> {'id', 'type'}
        self.data = _extract_compatible_data_from_nwbfile(self.nwb)

        # Need to check if some object names are doublons
        self.full_path_to_key = _get_unique_identifier(
            {p: os.path.basename(p) for p in self.data.keys()}
        )

        # Creating the reverse mapping for the user : key -> full_path and key -> {'id', 'type'}
        self.key_to_full_path = {v: k for k, v in self.full_path_to_key.items()}
        self.data = {self.full_path_to_key[p]: self.data[p] for p in self.data.keys()}

        # Mapping unique path identifier to id
        self.key_to_id = {k: self.data[k]["id"] for k in self.data.keys()}

        self._view = [[k, self.data[k]["type"]] for k in self.data.keys()]

        self._lazy_loading = lazy_loading

        UserDict.__init__(self, self.data)

    def __str__(self):
        title = self.name if isinstance(self.name, str) else "-"
        headers = ["Keys", "Type"]
        return (
            title
            + "\n"
            + tabulate(self._view, headers=headers, tablefmt="mixed_outline")
        )

        # self._view = Table(title=self.name)
        # self._view.add_column("Keys", justify="left", style="cyan", no_wrap=True)
        # self._view.add_column("Type", style="green")
        # for k in self.data.keys():
        #     self._view.add_row(
        #         k,
        #         self.data[k]["type"],
        #     )

        # """View of the object"""
        # with Console() as console:
        #     console.print(self._view)
        # return ""

    def __repr__(self):
        """View of the object"""
        return self.__str__()

    def __getitem__(self, key):
        """Get object from NWB

        Parameters
        ----------
        key : str


        Returns
        -------
        (Ts, Tsd, TsdFrame, TsGroup, IntervalSet or dict of IntervalSet)


        Raises
        ------
        KeyError
            If key is not in the dictionary
        """
        if key.__hash__:
            if key.startswith("/"):  # allow user to specify the full path to the object
                if key in self.full_path_to_key:
                    return self[self.full_path_to_key[key]]
                else:
                    raise KeyError("Can't find key {} in group index.".format(key))

            if self.__contains__(key):
                if isinstance(self.data[key], dict) and "id" in self.data[key]:
                    obj = self.nwb.objects[self.data[key]["id"]]
                    try:
                        data = self._f_eval[self.data[key]["type"]](
                            obj, lazy_loading=self._lazy_loading
                        )
                    except Exception:
                        warnings.warn(
                            "Failed to build {}.\n Returning the NWB object for manual inspection".format(
                                self.data[key]["type"]
                            ),
                            stacklevel=2,
                        )
                        data = obj

                    if isinstance(data, _NWBLazyTsGroup):
                        # keep the file open while the group can read it
                        data.__dict__["_nwb_file"] = self
                    self.data[key] = data
                    return data
                else:
                    return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))

    def close(self):
        """Close the NWB file"""
        # lazy units groups cannot read their spike times after this
        for data in self.data.values():
            if isinstance(data, _NWBLazyTsGroup):
                data._close_file()
        self.io.close()

    def keys(self):
        """
        Return keys of NWBFile

        Returns
        -------
        list
            List of keys
        """
        return list(self.data.keys())

    def items(self):
        """
        Return a list of key/object.

        Returns
        -------
        list
            List of tuples
        """
        return list(self.data.items())

    def values(self):
        """
        Return a list of all the objects

        Returns
        -------
        list
            List of objects
        """
        return list(self.data.values())
