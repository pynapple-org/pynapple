#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-05 16:03:25
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-09 16:30:09

import os

import numpy as np

from .. import core as nap


class NPZFile(object):
    """

    """

    def __init__(self, path):
        """Summary

        Parameters
        ----------
        path : str
            Valid path to a NPZ file
        """
        self.path = path
        self.name = os.path.basename(path)
        self.file = np.load(self.path, allow_pickle=True)
        self.type = ""
        if "index" in self.file.keys():
            self.type = "TsGroup"
        elif "columns" in self.file.keys():
            self.type = "TsdFrame"
        elif "d" in self.file.keys():
            self.type = "Tsd"
        elif "t" in self.file.keys():
            self.type = "Ts"
        else:
            self.type = "IntervalSet"

    def load(self):
        time_support = nap.IntervalSet(self.file["start"], self.file["end"])
        if "index" in self.file.keys():
            tsd = nap.Tsd(
                t=self.file["t"], d=self.file["index"], time_support=time_support
            )
            tsgroup = tsd.to_tsgroup()
            tsgroup.set_info(group=self.file["group"], location=self.file["location"])
            return tsgroup
        elif "columns" in self.file.keys():
            return nap.TsdFrame(
                t=self.file["t"],
                d=self.file["d"],
                time_support=time_support,
                columns=self.file["columns"],
            )
        elif "d" in self.file.keys():
            return nap.Tsd(
                t=self.file["t"], d=self.file["d"], time_support=time_support
            )
        elif "t" in self.file.keys():
            return nap.Ts(t=self.file["t"], time_support=time_support)
        else:
            return time_support


class NWBFile(object):
    def __init__(self, path):
        """Summary

        Parameters
        ----------
        path : str
            Valid path to a NWB file
        """
        self.path = path
        self.name = os.path.basename(path)

    def load(self):
        print("yo")
