#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-09-09 14:53:21
# @Last Modified by:   gviejo
# @Last Modified time: 2022-09-15 14:40:10

"""
Loader for Suite2P
https://github.com/MouseLand/suite2p

"""

import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from pynwb.ophys import (
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)
from PyQt5.QtWidgets import QApplication

from .. import core as nap
from .loader import BaseLoader
from .ophys_gui import OphysGUI


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

        # Need to check if nwb file exists and if data are there
        loading_my_data = True
        if self.path is not None:
            nwb_path = os.path.join(self.path, "pynapplenwb")
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith(".nwb")]):
                    success = self.load_suite2p_nwb(path)
                    if success:
                        loading_my_data = False

        # Bypass if data have already been transfered to nwb
        if loading_my_data:

            app = QApplication([])
            window = OphysGUI(path=path)
            window.show()
            app.exec()
            if window.status:
                self.ophys_information = window.ophys_information
                self.load_suite2p(path)
                self.save_suite2p_nwb(path)

    def load_suite2p(self, path):
        """
        Looking for suite2/plane*

        Parameters
        ----------
        path : str
            The path of the session

        """
        self.path_suite2p = os.path.join(path, "suite2p")

        self.sampling_rate = float(
            self.ophys_information["ImagingPlane"]["imaging_rate"]
        )

        data = {
            "F": [],
            "Fneu": [],
            "spks": [],
        }
        plane_info = []

        self.stats = {}
        self.pops = {}
        self.iscells = {}

        self.planes = []

        if os.path.exists(self.path_suite2p):
            planes = glob.glob(os.path.join(self.path_suite2p, "plane*"))

            if len(planes):
                # count = 0
                for plane_dir in planes:
                    n = int(os.path.basename(plane_dir)[-1])
                    self.planes.append(n)
                    # Loading iscell.npy
                    try:
                        iscell = np.load(
                            os.path.join(plane_dir, "iscell.npy"), allow_pickle=True
                        )
                        idx = np.where(iscell.astype("int")[:, 0])[0]
                        plane_info.append(np.ones(len(idx), dtype="int") * n)

                    except OSError as e:
                        print(e)
                        sys.exit()

                    # Loading F.npy, Fneu.py and spks.npy
                    for obj in ["F.npy", "Fneu.npy", "spks.npy"]:
                        try:
                            name = obj.split(".")[0]
                            tmp = np.load(
                                os.path.join(plane_dir, obj), allow_pickle=True
                            )
                            data[name].append(tmp[idx])

                        except OSError as e:
                            print(e)
                            sys.exit()

                    # Loading stat.npy and ops.npy
                    try:
                        stat = np.load(
                            os.path.join(plane_dir, "stat.npy"), allow_pickle=True
                        )
                        ops = np.load(
                            os.path.join(plane_dir, "ops.npy"), allow_pickle=True
                        ).item()
                    except OSError as e:
                        print(e)
                        sys.exit()

                    # Saving stat, ops and iscell
                    self.stats[n] = stat
                    self.pops[n] = ops
                    self.iscells[n] = iscell

                    # count += len(idx)

            else:
                warnings.warn(
                    "Couldn't find planes in %s" % self.path_suite2p, stacklevel=2
                )
                sys.exit()
        else:
            warnings.warn("No suite2p folder in %s" % path, stacklevel=2)
            sys.exit()

        # Calcium transients
        data["F"] = np.transpose(np.vstack(data["F"]))
        data["Fneu"] = np.transpose(np.vstack(data["Fneu"]))
        data["spks"] = np.transpose(np.vstack(data["spks"]))

        time_index = np.arange(0, len(data["F"])) / self.sampling_rate

        self.F = nap.TsdFrame(t=time_index, d=data["F"])
        self.Fneu = nap.TsdFrame(t=time_index, d=data["Fneu"])
        self.spks = nap.TsdFrame(t=time_index, d=data["spks"])

        self.ops = self.pops[0]
        self.iscell = np.vstack([self.iscells[k] for k in self.iscells.keys()])

        # Metadata
        self.plane_info = pd.DataFrame.from_dict({"plane": np.hstack(plane_info)})
        return

    def save_suite2p_nwb(self, path):
        """
        Save the data to NWB. To ensure continuity, this function is based on :
        https://github.com/MouseLand/suite2p/blob/main/suite2p/io/nwb.py.

        Parameters
        ----------
        path : str
            The path of the session
        """
        self.nwb_path = os.path.join(path, "pynapplenwb")
        if not os.path.exists(self.nwb_path):
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if "nwb" in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        multiplane = True if len(self.planes) > 1 else False

        ops = self.pops[list(self.pops.keys())[0]]

        io = NWBHDF5IO(self.nwbfilepath, "r+")
        nwbfile = io.read()

        device = nwbfile.create_device(
            name=self.ophys_information["device"]["name"],
            description=self.ophys_information["device"]["description"],
            manufacturer=self.ophys_information["device"]["manufacturer"],
        )
        imaging_plane = nwbfile.create_imaging_plane(
            name=self.ophys_information["ImagingPlane"]["name"],
            optical_channel=OpticalChannel(
                name=self.ophys_information["OpticalChannel"]["name"],
                description=self.ophys_information["OpticalChannel"]["description"],
                emission_lambda=float(
                    self.ophys_information["OpticalChannel"]["emission_lambda"]
                ),
            ),
            imaging_rate=self.sampling_rate,
            description=self.ophys_information["ImagingPlane"]["description"],
            device=device,
            excitation_lambda=float(
                self.ophys_information["ImagingPlane"]["excitation_lambda"]
            ),
            indicator=self.ophys_information["ImagingPlane"]["indicator"],
            location=self.ophys_information["ImagingPlane"]["location"],
            grid_spacing=([2.0, 2.0, 30.0] if multiplane else [2.0, 2.0]),
            grid_spacing_unit="microns",
        )

        # link to external data
        image_series = TwoPhotonSeries(
            name="TwoPhotonSeries",
            dimension=[ops["Ly"], ops["Lx"]],
            external_file=(ops["filelist"] if "filelist" in ops else [""]),
            imaging_plane=imaging_plane,
            starting_frame=[0],
            format="external",
            starting_time=0.0,
            rate=ops["fs"] * ops["nplanes"],
        )
        nwbfile.add_acquisition(image_series)

        # processing
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name=self.ophys_information["PlaneSegmentation"]["name"],
            description=self.ophys_information["PlaneSegmentation"]["description"],
            imaging_plane=imaging_plane,
            # reference_images=image_series,
        )
        ophys_module = nwbfile.create_processing_module(
            name="ophys", description="optical physiology processed data"
        )
        ophys_module.add(img_seg)

        file_strs = ["F.npy", "Fneu.npy", "spks.npy"]
        traces = []
        ncells = np.zeros(len(self.pops), dtype=np.int_)
        Nfr = np.array([self.pops[k]["nframes"] for k in self.pops.keys()]).max()

        for iplane, ops in self.pops.items():
            if iplane == 0:
                iscell = self.iscells[iplane]
                for fstr in file_strs:
                    traces.append(np.load(os.path.join(ops["save_path"], fstr)))
                PlaneCellsIdx = iplane * np.ones(len(iscell))
            else:
                iscell = np.append(
                    iscell,
                    self.iscells[iplane],
                    axis=0,
                )
                for i, fstr in enumerate(file_strs):
                    trace = np.load(os.path.join(ops["save_path"], fstr))
                    if trace.shape[1] < Nfr:
                        fcat = np.zeros(
                            (trace.shape[0], Nfr - trace.shape[1]), "float32"
                        )
                        trace = np.concatenate((trace, fcat), axis=1)
                    traces[i] = np.append(traces[i], trace, axis=0)
                PlaneCellsIdx = np.append(
                    PlaneCellsIdx, iplane * np.ones(len(iscell) - len(PlaneCellsIdx))
                )

            stat = self.stats[iplane]
            ncells[iplane] = len(stat)

            for n in range(ncells[iplane]):
                if multiplane:
                    pixel_mask = np.array(
                        [
                            stat[n]["ypix"],
                            stat[n]["xpix"],
                            iplane * np.ones(stat[n]["npix"]),
                            stat[n]["lam"],
                        ]
                    )
                    ps.add_roi(voxel_mask=pixel_mask.T)
                else:
                    pixel_mask = np.array(
                        [stat[n]["ypix"], stat[n]["xpix"], stat[n]["lam"]]
                    )
                    ps.add_roi(pixel_mask=pixel_mask.T)

        ps.add_column("iscell", "two columns - iscell & probcell", iscell)

        rt_region = []
        for iplane, ops in self.pops.items():
            if iplane == 0:
                rt_region.append(
                    ps.create_roi_table_region(
                        region=list(
                            np.arange(0, ncells[iplane]),
                        ),
                        description=f"ROIs for plane{int(iplane)}",
                    )
                )
            else:
                rt_region.append(
                    ps.create_roi_table_region(
                        region=list(
                            np.arange(
                                np.sum(ncells[:iplane]),
                                ncells[iplane] + np.sum(ncells[:iplane]),
                            )
                        ),
                        description=f"ROIs for plane{int(iplane)}",
                    )
                )

        # FLUORESCENCE (all are required)
        name_strs = ["Fluorescence", "Neuropil", "Deconvolved"]

        for i, (fstr, nstr) in enumerate(zip(file_strs, name_strs)):
            for iplane, ops in self.pops.items():
                roi_resp_series = RoiResponseSeries(
                    name=f"plane{int(iplane)}",
                    data=traces[i][PlaneCellsIdx == iplane],
                    rois=rt_region[iplane],
                    unit="lumens",
                    rate=ops["fs"],
                )
                if iplane == 0:
                    fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
                else:
                    fl.add_roi_response_series(roi_response_series=roi_resp_series)
            ophys_module.add(fl)

        io.write(nwbfile)
        io.close()
        return

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
