#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-09-09 14:53:21
# @Last Modified by:   gviejo
# @Last Modified time: 2023-11-16 13:22:10

"""
> :warning: **DEPRECATED**: This will be removed in version 1.0.0. Check [nwbmatic](https://github.com/pynapple-org/nwbmatic) or [neuroconv](https://github.com/catalystneuro/neuroconv) instead.

Loader for Suite2P
https://github.com/MouseLand/suite2p

"""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from .. import core as nap
from .loader import BaseLoader


class Suite2P(BaseLoader):
    """Loader for data processed with Suite2P.

    Pynapple will try to look for data in this order :

    1. pynapplenwb/session_name.nwb

    2. suite2p/plane*/*.npy


    Attributes
    ----------
    F : TsdFrame
        Fluorescence traces (timepoints x ROIs) for all planes
    Fneu : TsdFrame
        Neuropil fluorescence traces (timepoints x ROIs) for all planes
    spks : TsdFrame
        Deconvolved traces (timepoints x ROIS) for all planes
    plane_info : pandas.DataFrame
        Contains plane identity of each cell
    stats : dict
        dictionnay of statistics from stat.npy for each planes only for the neurons that were classified as cells
        (Can be smaller when loading from the NWB file)
    ops : dict
        Parameters from Suite2p. (Can be smaller when loading from the NWB file)
    iscell : numpy.ndarray
        Cell classification
    """

    def __init__(self, path):
        """

        Parameters
        ----------
        path : str
            The path of the session
        """
        path = Path(path)
        self.basename = path.name

        super().__init__(path)

        self.load_suite2p_nwb(path)

    def load_suite2p_nwb(self, path):
        """
        Load suite2p data from NWB

        Parameters
        ----------
        path : str
            Path to the session
        """
        pynwb = importlib.import_module("pynwb")
        io = pynwb.NWBHDF5IO(self.nwbfilepath, "r")

        try:
            nwbfile = io.read()

            if "ophys" not in nwbfile.processing.keys():
                return False

            ophys = nwbfile.processing["ophys"]
            info = self._load_ophys_info(nwbfile, ophys)
            rois, multiplane = self._get_rois(ophys)
            self._populate_roi_stats(rois, info)

            fields = self._get_timeseries_fields(ophys)
            if len(fields) == 0:
                print(
                    "No " + " or ".join(["Fluorescence", "Neuropil", "Deconvolved"]),
                    "found in nwb {}".format(self.nwbfilepath),
                )
                return False

            data = self._load_timeseries(ophys, fields, info, multiplane)
            self._assign_timeseries(data)

            self.plane_info = pd.DataFrame(
                data=info["plane"][info["iscell"] == 1].values, columns=["plane"]
            )

            return True
        finally:
            io.close()

    def _load_ophys_info(self, nwbfile, ophys):
        dims = nwbfile.acquisition["TwoPhotonSeries"].dimension[:]
        self.ops = {"Ly": dims[0], "Lx": dims[1]}
        self.rate = nwbfile.acquisition["TwoPhotonSeries"].imaging_plane.imaging_rate

        self.stats = {0: {}}
        self.iscell = ophys["ImageSegmentation"]["PlaneSegmentation"]["iscell"].data[:]

        info = pd.DataFrame(data=self.iscell[:, 0].astype("int"), columns=["iscell"])
        info["plane"] = 0
        return info

    def _get_rois(self, ophys):
        try:
            rois = ophys["ImageSegmentation"]["PlaneSegmentation"]["pixel_mask"]
            return rois, False
        except Exception:
            rois = ophys["ImageSegmentation"]["PlaneSegmentation"]["voxel_mask"]
            return rois, True

    def _populate_roi_stats(self, rois, info):
        idx = np.where(self.iscell[:, 0])[0]

        for n in range(len(rois)):
            roi = pd.DataFrame(rois[n])
            pl = roi["z"][0] if "z" in roi.columns else 0

            info.loc[n, "plane"] = pl

            if pl not in self.stats.keys():
                self.stats[pl] = {}

            if n in idx:
                self.stats[pl][n] = {
                    "xpix": roi["y"].values,
                    "ypix": roi["x"].values,
                    "lam": roi["weight"].values,
                }

    def _get_timeseries_fields(self, ophys):
        return np.intersect1d(
            ["Fluorescence", "Neuropil", "Deconvolved"],
            list(ophys.fields["data_interfaces"].keys()),
        )

    def _load_timeseries(self, ophys, fields, info, multiplane):
        data = {}

        for k, name in zip(
            ["F", "Fneu", "spks"], ["Fluorescence", "Neuropil", "Deconvolved"]
        ):
            if name not in fields:
                continue

            data[k] = self._load_timeseries_field(ophys, name, info, multiplane)

        return data

    def _load_timeseries_field(self, ophys, name, info, multiplane):
        if multiplane:
            keys = ophys[name].roi_response_series.keys()
            planes = [int(k[-1]) for k in keys if "plane" in k]
        else:
            planes = [0]

        tmp = []
        timestamps = []

        for n in planes:
            if multiplane:
                pl = "plane{}".format(n)
            else:
                pl = name

            tokeep = info["iscell"][info["plane"] == n].values == 1
            d = np.transpose(ophys[name][pl].data[:][tokeep])

            if ophys[name][pl].timestamps is not None:
                t = ophys[name][pl].timestamps[:]
            else:
                t = (np.arange(0, len(d)) / self.rate) + ophys[name][pl].starting_time

            tmp.append(d)
            timestamps.append(t)

        return nap.TsdFrame(t=timestamps[0], d=np.hstack(tmp))

    def _assign_timeseries(self, data):
        if "F" in data.keys():
            self.F = data["F"]
        if "Fneu" in data.keys():
            self.Fneu = data["Fneu"]
        if "spks" in data.keys():
            self.spks = data["spks"]
