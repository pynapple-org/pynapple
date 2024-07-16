#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-05 16:03:25
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-04-02 14:32:25


import os

import numpy as np

from .. import core as nap

#
EXPECTED_ENTRIES = {
    "TsGroup": {"t", "start", "end", "index"},
    "TsdFrame": {"t", "d", "start", "end", "columns"},
    "TsdTensor": {"t", "d", "start", "end"},
    "Tsd": {"t", "d", "start", "end"},
    "Ts": {"t", "start", "end"},
    "IntervalSet": {"start", "end"},
}


def _find_class_from_variables(file_variables, data_ndims=None):
    if data_ndims is not None:
        # either TsdTensor or Tsd:
        assert EXPECTED_ENTRIES["Tsd"].issubset(file_variables)

        return "Tsd" if data_ndims == 1 else "TsdTensor"

    for possible_type, espected_variables in EXPECTED_ENTRIES.items():
        if espected_variables.issubset(file_variables):
            return possible_type

    return "npz"


class LazyNPZLoader:
    """Class that lazily loads an NPZ file."""

    def __init__(self, file_path, lazy_loading=False):
        self.lazy_loading = lazy_loading
        self.file_path = file_path
        self.npz_file = np.load(
            file_path, allow_pickle=True, mmap_mode="r" if lazy_loading else None
        )
        self.data = {key: None for key in self.npz_file.keys()}

    def __getitem__(self, key):
        if key not in self.data:
            raise KeyError(f"{key} not found in the NPZ file")

        if self.data[key] is None:
            self.data[key] = self._load_array(key)

        return self.data[key]

    def _load_array(self, key):
        if self.lazy_loading:
            array_info = self.npz_file.zip.read(
                self.npz_file.zip.NameToInfo[key].filename
            )
            np_array = np.frombuffer(
                array_info, dtype=self.npz_file[key].dtype
            ).reshape(self.npz_file[key].shape)
            return np.memmap(
                self.npz_file.filename,
                dtype=np_array.dtype,
                mode="r",
                shape=np_array.shape,
            )
        else:
            return self.npz_file[key]

    def keys(self):
        return self.npz_file.keys()


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

    # valid_types = ["Ts", "Tsd", "TsdFrame", "TsdTensor", "TsGroup", "IntervalSet"]

    def __init__(self, path, lazy_loading=False):
        """Initialization of the NPZ file

        Parameters
        ----------
        path : str
            Valid path to a NPZ file
        """
        self.path = path
        self.name = os.path.basename(path)
        self.file = LazyNPZLoader(
            path, lazy_loading=lazy_loading
        )  # np.load(self.path, allow_pickle=True)
        type_ = ""

        # First check if type is explicitely defined in the file:
        try:
            type_ = self.file["type"][0]
            assert type_ in EXPECTED_ENTRIES.keys()

        # if not, use heuristics:
        except (KeyError, IndexError, AssertionError):
            file_variables = set(self.file.keys())
            data_ndims = self.file["d"].ndim if "d" in file_variables else None

            type_ = _find_class_from_variables(file_variables, data_ndims)

        self.type = type_

    def load(self):
        """Load the NPZ file

        Returns
        -------
        (Tsd, Ts, TsdFrame, TsdTensor, TsGroup, IntervalSet)
            A pynapple object
        """
        if self.type == "npz":
            return self.file.npz_file

        return getattr(nap, self.type)._from_npz_reader(self.file)
