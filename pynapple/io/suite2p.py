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

import os

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

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
        self.basename = os.path.basename(path)

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
        self.nwb_path = os.path.join(path, "pynapplenwb")
        if not os.path.exists(self.nwb_path):
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))

        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if "nwb" in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, "r")
        nwbfile = io.read()

        if "ophys" in nwbfile.processing.keys():
            ophys = nwbfile.processing["ophys"]

            #################################################################
            # STATS, OPS and ISCELL
            #################################################################
            dims = nwbfile.acquisition["TwoPhotonSeries"].dimension[:]
            self.ops = {"Ly": dims[0], "Lx": dims[1]}
            self.rate = nwbfile.acquisition[
                "TwoPhotonSeries"
            ].imaging_plane.imaging_rate

            self.stats = {0: {}}
            self.iscell = ophys["ImageSegmentation"]["PlaneSegmentation"][
                "iscell"
            ].data[:]

            info = pd.DataFrame(
                data=self.iscell[:, 0].astype("int"), columns=["iscell"]
            )

            #################################################################
            # ROIS
            #################################################################
            try:
                rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                    "PlaneSegmentation"
                ]["pixel_mask"]
                multiplane = False
            except Exception:
                rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                    "PlaneSegmentation"
                ]["voxel_mask"]
                multiplane = True

            idx = np.where(self.iscell[:, 0])[0]
            info["plane"] = 0

            for n in range(len(rois)):
                roi = pd.DataFrame(rois[n])
                if "z" in roi.columns:
                    pl = roi["z"][0]
                else:
                    pl = 0

                info.loc[n, "plane"] = pl

                if pl not in self.stats.keys():
                    self.stats[pl] = {}

                if n in idx:
                    self.stats[pl][n] = {
                        "xpix": roi["y"].values,
                        "ypix": roi["x"].values,
                        "lam": roi["weight"].values,
                    }

            #################################################################
            # Time Series
            #################################################################
            fields = np.intersect1d(
                ["Fluorescence", "Neuropil", "Deconvolved"],
                list(ophys.fields["data_interfaces"].keys()),
            )

            if len(fields) == 0:
                print(
                    "No " + " or ".join(["Fluorescence", "Neuropil", "Deconvolved"]),
                    "found in nwb {}".format(self.nwbfilepath),
                )
                return False

            keys = ophys[fields[0]].roi_response_series.keys()

            planes = [int(k[-1]) for k in keys if "plane" in k]

            data = {}

            if multiplane:
                keys = ophys[fields[0]].roi_response_series.keys()
                planes = [int(k[-1]) for k in keys if "plane" in k]
            else:
                planes = [0]

            for k, name in zip(
                ["F", "Fneu", "spks"], ["Fluorescence", "Neuropil", "Deconvolved"]
            ):
                tmp = []
                timestamps = []

                for i, n in enumerate(planes):
                    if multiplane:
                        pl = "plane{}".format(n)
                    else:
                        pl = name  # This doesn't make sense

                    tokeep = info["iscell"][info["plane"] == n].values == 1

                    d = np.transpose(ophys[name][pl].data[:][tokeep])

                    if ophys[name][pl].timestamps is not None:
                        t = ophys[name][pl].timestamps[:]
                    else:
                        t = (np.arange(0, len(d)) / self.rate) + ophys[name][
                            pl
                        ].starting_time

                    tmp.append(d)
                    timestamps.append(t)

                data[k] = nap.TsdFrame(t=timestamps[0], d=np.hstack(tmp))

            if "F" in data.keys():
                self.F = data["F"]
            if "Fneu" in data.keys():
                self.Fneu = data["Fneu"]
            if "spks" in data.keys():
                self.spks = data["spks"]

            self.plane_info = pd.DataFrame(
                data=info["plane"][info["iscell"] == 1].values, columns=["plane"]
            )

            io.close()
            return True
        else:
            io.close()
            return False
