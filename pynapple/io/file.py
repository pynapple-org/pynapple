#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-05 16:03:25
# @Last Modified by:   gviejo
# @Last Modified time: 2023-07-28 17:05:48

"""
File classes help to validate and load pynapple objects or NWB files.
Data are always lazy-loaded.
Both classes behaves like dictionnary.
"""


import errno
import os
import warnings
from collections import UserDict

import numpy as np
import pynwb
from pynwb import NWBHDF5IO
from rich.console import Console
from rich.table import Table

from .. import core as nap



class NPZFile(object):
    """Class that points to a NPZ file that can be loaded as a pynapple object. 
    Objects have a save function in npz format as well as the Folder class.
    
    Examples
    --------
    >>> import pynapple as nap    
    >>> tsd = nap.load_file("path/to/my_tsd.npz")
    >>> tsd
    Time (s)
    0.0    0
    0.1    1
    0.2    2
    dtype: int64

    """

    def __init__(self, path):
        """Initialization of the NPZ file

        Parameters
        ----------
        path : str
            Valid path to a NPZ file
        """
        self.path = path
        self.name = os.path.basename(path)
        self.file = np.load(self.path, allow_pickle=True)
        self.type = ""

        # First check if type is explicitely defined
        possible = ["Ts", "Tsd", "TsdFrame", "TsGroup", "IntervalSet"]
        if "type" in self.file.keys():
            if len(self.file["type"]) == 1:
                if isinstance(self.file["type"][0], np.str_):
                    if self.file["type"] in possible:
                        self.type = self.file["type"][0]

        # Second check manually
        if self.type == "":
            k = set(self.file.keys())
            if {"t", "start", "end", "index"}.issubset(k):
                self.type = "TsGroup"
            elif {"t", "d", "start", "end", "columns"}.issubset(k):
                self.type = "TsdFrame"
            elif {"t", "d", "start", "end"}.issubset(k):
                self.type = "Tsd"
            elif {"t", "start", "end"}.issubset(k):
                self.type = "Ts"
            elif {"start", "end"}.issubset(k):
                self.type = "IntervalSet"
            else:
                self.type = "npz"

    def load(self):
        """Load the NPZ file

        Returns
        -------
        (Tsd, Ts, TsdFrame, TsGroup, IntervalSet)
            A pynapple object
        """
        if self.type == "npz":
            return self.file
        else:
            time_support = nap.IntervalSet(self.file["start"], self.file["end"])
            if self.type == "TsGroup":
                tsd = nap.Tsd(
                    t=self.file["t"], d=self.file["index"], time_support=time_support
                )
                tsgroup = tsd.to_tsgroup()
                if "d" in self.file.keys():
                    print("TODO")

                metainfo = {}
                for k in set(self.file.keys()) - {
                    "start",
                    "end",
                    "t",
                    "index",
                    "d",
                    "rate",
                }:
                    tmp = self.file[k]
                    if len(tmp) == len(tsgroup):
                        metainfo[k] = tmp
                tsgroup.set_info(**metainfo)
                return tsgroup

            elif self.type == "TsdFrame":
                return nap.TsdFrame(
                    t=self.file["t"],
                    d=self.file["d"],
                    time_support=time_support,
                    columns=self.file["columns"],
                )

            elif self.type == "Tsd":
                return nap.Tsd(
                    t=self.file["t"], d=self.file["d"], time_support=time_support
                )
            elif self.type == "Ts":
                return nap.Ts(t=self.file["t"], time_support=time_support)
            elif self.type == "IntervalSet":
                return time_support
            else:
                return self.file


def _extract_compatible_data_from_nwbfile(nwbfile):
    """Extract all the NWB objects that can be converted to a pynapple object.

    Parameters
    ----------
    nwbfile : pynwb.file.NWBFile
        Instance of NWB file

    Returns
    -------
    dict
        Dictionnary containing all the object found and their type in pynapple.
    """
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
            if len(obj.data.shape) == 2:
                data[obj.name] = {"id": oid, "type": "TsdFrame"}

            elif len(obj.data.shape) == 1:
                data[obj.name] = {"id": oid, "type": "Tsd"}

    return data


def _make_interval_set(obj):
    """Helper function to make IntervalSet

    Parameters
    ----------
    obj : pynwb.epoch.TimeIntervals
        NWB object

    Returns
    -------
    IntervalSet or dict of IntervalSet
        If contains multiple epochs, a dictionary of IntervalSet is returned.
    """
    start_time = obj.start_time.data[:]
    stop_time = obj.stop_time.data[:]
    if hasattr(obj, "tags"):
        tags = obj.tags.data[:]
        categories = np.unique(tags)
        if len(categories) > 1:
            data = {}
            for c in categories:
                data[c] = nap.IntervalSet(
                    start=start_time[tags == c], end=stop_time[tags == c]
                )
        else:
            data = nap.IntervalSet(start=start_time, end=stop_time)
    else:
        data = nap.IntervalSet(start=start_time, end=stop_time)

    return data


def _make_tsd(obj):
    """Helper function to make Tsd

    Parameters
    ----------
    obj : pynwb.misc.TimeSeries
        NWB object

    Returns
    -------
    Tsd

    """

    d = obj.data[:]
    if obj.timestamps is not None:
        t = obj.timestamps[:]
    else:
        t = obj.starting_time + np.arange(obj.num_samples) / obj.rate

    data = nap.Tsd(t=t, d=d)

    return data


def _make_tsd_frame(obj):
    """Helper function to make TsdFrame

    Parameters
    ----------
    obj : pynwb.misc.TimeSeries
        NWB object

    Returns
    -------
    Tsd

    """

    d = obj.data[:]
    if obj.timestamps is not None:
        t = obj.timestamps[:]
    else:
        t = obj.starting_time + np.arange(obj.num_samples) / obj.rate

    if isinstance(obj, pynwb.behavior.SpatialSeries):
        if obj.data.shape[1] == 2:
            columns = ["x", "y"]
        elif obj.data.shape[1] == 3:
            columns = ["x", "y", "z"]
    elif isinstance(obj, pynwb.ecephys.ElectricalSeries):
        print("TODO")
        columns = np.arange(obj.data.shape[1])
        # (channel mapping)
    elif isinstance(obj, pynwb.ophys.RoiResponseSeries):
        print("TODO")
        columns = np.arange(obj.data.shape[1])
        # (cell number)
    else:
        columns = np.arange(obj.data.shape[1])

    data = nap.TsdFrame(t=t, d=d, columns=columns)

    return data


def _make_tsgroup(obj):
    """Helper function to make TsGroup

    Parameters
    ----------
    obj : pynwb.misc.Units
        NWB object

    Returns
    -------
    TsGroup

    """

    index = obj.id[:]
    tsgroup = {}
    for i, gr in zip(index, obj.spike_times_index[:]):
        # if np.min(np.diff(gr))<0.0:
        #     break
        tsgroup[i] = nap.Ts(t=gr)

    N = len(tsgroup)
    metainfo = {}
    for colname, col in zip(obj.colnames, obj.columns):
        if colname not in ["spike_times_index", "spike_times"]:
            if len(col) > 0:
                if len(col) == N:
                    if not isinstance(col[0], (np.ndarray, list, tuple, dict, set)):
                        metainfo[colname] = col[:]

    tsgroup = nap.TsGroup(tsgroup, **metainfo)

    return tsgroup


def _make_ts(obj):
    """Helper function to make Ts

    Parameters
    ----------
    obj : pynwb.misc.AnnotationSeries
        NWB object

    Returns
    -------
    Ts

    """
    data = nap.Ts(obj.timestamps[:])
    return data


class NWBFile(UserDict):
    """Class for reading NWB Files.


    Examples
    --------
    >>> import pynapple as nap
    >>> data = nap.load_file("my_file.nwb")
    >>> data["units"]

    """

    _f_eval = {
        "IntervalSet": _make_interval_set,
        "Tsd": _make_tsd,
        "Ts": _make_ts,
        "TsdFrame": _make_tsd_frame,
        "TsGroup": _make_tsgroup,
    }

    def __init__(self, file):
        """
        Parameters
        ----------
        file : str or pynwb.file.NWBFile
            Valid file to a NWB file

        Raises
        ------
        FileNotFoundError
            If path is invalid
        RuntimeError
            If file is not an instance of NWBFile
        """
        if isinstance(file, str):
            if os.path.exists(file):
                self.path = file
                self.name = os.path.basename(file).split(".")[0]
                self.io = NWBHDF5IO(file, "r")
                self.nwb = self.io.read()
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)
        elif isinstance(file, pynwb.file.NWBFile):
            self.nwb = file
            self.name = self.nwb.subject.subject_id

        else:
            raise RuntimeError(
                "unrecognized argument. Please provide path to a valid NWB file or open NWB file."
            )

        self.data = _extract_compatible_data_from_nwbfile(self.nwb)
        self.key_to_id = {k: self.data[k]["id"] for k in self.data.keys()}

        self._view = Table(title=self.name)
        self._view.add_column("Keys", justify="left", style="cyan", no_wrap=True)
        self._view.add_column("Type", style="green")
        # self._view.add_column("NWB module", justify="right", style="magenta")

        for k in self.data.keys():
            self._view.add_row(
                k,
                self.data[k]["type"],
                # self.data[k]['top_module']
            )

        UserDict.__init__(self, self.data)

    def __str__(self):
        """View of the object"""
        return self.__repr__()

    def __repr__(self):
        """View of the object"""
        console = Console()
        console.print(self._view)
        return ""

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
            If key is not in the dictionnary
        """
        if key.__hash__:
            if self.__contains__(key):
                if isinstance(self.data[key], dict):
                    obj = self.nwb.objects[self.data[key]["id"]]
                    try:
                        data = self._f_eval[self.data[key]["type"]](obj)
                    except:
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
