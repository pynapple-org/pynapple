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


def _extract_compatible_data_from_nwbfile(nwbfile):
    """Extract all the NWB objects that can be converted to a pynapple object.

    Parameters
    ----------
    nwbfile : pynwb.file.NWBFile
        Instance of NWB file

    Returns
    -------
    dict
        Dictionary containing all the object found and their type in pynapple.
    """
    pynwb = importlib.import_module("pynwb")
    data = {}

    for oid, obj in nwbfile.objects.items():
        if isinstance(obj, pynwb.misc.DynamicTable) and any(
            [i.name.endswith("_times_index") for i in obj.columns]
        ):
            data["units"] = {"id": oid, "type": "TsGroup"}

        elif isinstance(obj, pynwb.epoch.TimeIntervals):
            # Supposedly IntervalsSets
            data[obj.name] = {"id": oid, "type": "IntervalSet"}

        elif isinstance(obj, pynwb.misc.DynamicTable) and any(
            [i.name.endswith("_times") for i in obj.columns]
        ):
            # Supposedly Timestamps
            data[obj.name] = {"id": oid, "type": "Ts"}

        elif isinstance(obj, pynwb.misc.AnnotationSeries):
            # Old timestamps version
            data[obj.name] = {"id": oid, "type": "Ts"}

        elif isinstance(obj, pynwb.misc.TimeSeries):
            if len(obj.data.shape) > 2:
                data[obj.name] = {"id": oid, "type": "TsdTensor"}

            elif len(obj.data.shape) == 2:
                data[obj.name] = {"id": oid, "type": "TsdFrame"}

            elif len(obj.data.shape) == 1:
                data[obj.name] = {"id": oid, "type": "Tsd"}

    return data


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
            df = obj.electrodes.to_dataframe()
            if hasattr(df, "label"):
                columns = df["label"].values
            else:
                columns = df.index.values
        except Exception:
            columns = np.arange(obj.data.shape[1])

    elif isinstance(obj, pynwb.ophys.RoiResponseSeries):
        # (cell number)
        try:
            columns = obj.rois["id"][:]
        except Exception:
            columns = np.arange(obj.data.shape[1])

    else:
        columns = np.arange(obj.data.shape[1])

    if len(columns) >= d.shape[1]:  # Weird sometimes if background ID added
        columns = columns[0 : obj.data.shape[1]]
    else:
        columns = np.arange(obj.data.shape[1])

    data = nap.TsdFrame(t=t, d=d, columns=columns, load_array=not lazy_loading)

    return data


def _make_tsgroup(obj, **kwargs):
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
    tsgroup = {}
    for i, gr in zip(index, obj.spike_times_index[:]):
        # if np.min(np.diff(gr))<0.0:
        #     break
        tsgroup[i] = nap.Ts(t=np.array(gr))

    N = len(tsgroup)
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
                    df = col.to_dataframe()
                    df = df.sort_index()
                    for k in df.columns:
                        if not isinstance(
                            df[k].values[0],
                            (list, tuple, dict, set, pynwb.ecephys.ElectrodeGroup),
                        ):
                            metainfo[k] = df[k].values
                # elif not isinstance(col[0], (np.ndarray, list, tuple, dict, set)):
                elif isinstance(col[0], (Number, str)):
                    metainfo[coln] = np.array(col[:])
                else:
                    pass

    tsgroup = nap.TsGroup(tsgroup, metadata=metainfo)

    return tsgroup


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

        self.data = _extract_compatible_data_from_nwbfile(self.nwb)
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

                    self.data[key] = data
                    return data
                else:
                    return self.data[key]
            else:
                raise KeyError("Can't find key {} in group index.".format(key))

    def close(self):
        """Close the NWB file"""
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
