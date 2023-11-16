# -*- coding: utf-8 -*-
"""
> :warning: **DEPRECATED**: This will be removed in version 1.0.0. Check [nwbmatic](https://github.com/pynapple-org/nwbmatic) or [neuroconv](https://github.com/catalystneuro/neuroconv) instead.

Loaders for calcium imaging data with miniscope.
Support CNMF-E in matlab, inscopix-cnmfe and minian.

"""
# @Author: gviejo
# @Date:   2022-02-17 11:07:00
# @Last Modified by:   gviejo
# @Last Modified time: 2023-11-16 13:14:54

import os

from pynwb import NWBHDF5IO

from .. import core as nap
from .loader import BaseLoader


class CNMF_E(BaseLoader):
    """Loader for data processed with matlab CNMF-E(https://github.com/zhoupc/CNMF_E).
    The path folder should contain a file ending in .mat
    when calling Source2d.save_neurons

    Attributes
    ----------
    A : numpy.ndarray
        Spatial footprints
    C : TsdFrame
        The calcium transients
    sampling_rate : float
        Sampling rate of the data (default is 30 Hz).

    """

    def __init__(self, path):
        """

        Parameters
        ----------
        path : str
            The path to the data.
        """
        self.basename = os.path.basename(path)

        super().__init__(path)

        self.load_cnmfe_nwb(path)

    def load_cnmfe_nwb(self, path):
        """
        Load the calcium transient and spatial footprint from nwb

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
            data = nwbfile.processing["ophys"]["Fluorescence"][
                "RoiResponseSeries"
            ].data[:]
            t = nwbfile.processing["ophys"]["Fluorescence"][
                "RoiResponseSeries"
            ].timestamps[:]
            self.C = nap.TsdFrame(t=t, d=data)
            self.A = nwbfile.processing["ophys"]["ImageSegmentation"][
                "PlaneSegmentation"
            ]["image_mask"].data[:]

            io.close()
            return True
        else:
            io.close()
            return False


class Minian(BaseLoader):
    """Loader for data processed with Minian (https://github.com/denisecailab/minian).
    The path folder should contain a subfolder name minian.

    Attributes
    ----------
    A : numpy.ndarray
        Spatial footprints
    C : TsdFrame
        The calcium transients
    sampling_rate : float
        Sampling rate of the data (default is 30 Hz).

    """

    def __init__(self, path):
        """

        Parameters
        ----------
        path : str
            The path to the data.
        """
        self.basename = os.path.basename(path)

        super().__init__(path)

        self.load_cnmfe_nwb(path)

    def load_cnmfe_nwb(self, path):
        """
        Load the calcium transient and spatial footprint from nwb

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
            data = nwbfile.processing["ophys"]["Fluorescence"][
                "RoiResponseSeries"
            ].data[:]
            t = nwbfile.processing["ophys"]["Fluorescence"][
                "RoiResponseSeries"
            ].timestamps[:]
            self.C = nap.TsdFrame(t=t, d=data)
            self.A = nwbfile.processing["ophys"]["ImageSegmentation"][
                "PlaneSegmentation"
            ]["image_mask"].data[:]

            io.close()
            return True
        else:
            io.close()
            return False


class InscopixCNMFE(BaseLoader):
    """Loader for Inscopix-cnmfe (https://github.com/inscopix/inscopix-cnmfe).
    The folder should contain a file ending with '_traces.csv'
    and a tiff file for spatial footprints.

    Attributes
    ----------
    A : np.ndarray
        The spatial footprints
    C : TsdFrame
        The calcium transients
    sampling_rate : float
        Sampling rate of the data (default is 30 Hz).

    """

    def __init__(self, path):
        """

        Parameters
        ----------
        path : str
            The path to the data.
        """
        self.basename = os.path.basename(path)

        super().__init__(path)

        self.load_cnmfe_nwb(path)

    def load_cnmfe_nwb(self, path):
        """
        Load the calcium transient and spatial footprint from nwb

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
            data = nwbfile.processing["ophys"]["Fluorescence"][
                "RoiResponseSeries"
            ].data[:]
            t = nwbfile.processing["ophys"]["Fluorescence"][
                "RoiResponseSeries"
            ].timestamps[:]
            self.C = nap.TsdFrame(t=t, d=data)
            self.A = nwbfile.processing["ophys"]["ImageSegmentation"][
                "PlaneSegmentation"
            ]["image_mask"].data[:]

            io.close()
            return True
        else:
            io.close()
            return False
