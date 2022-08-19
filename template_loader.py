# -*- coding: utf-8 -*-
"""
Template for building a data loader with pynapple
"""
# @Author: gviejo
# @Date:   2022-01-26 18:13:42
# @Last Modified by:   gviejo
# @Last Modified time: 2022-08-18 18:02:38

import os

from pynwb import NWBHDF5IO

from pynapple.io.loader import BaseLoader


class MyCustomIO(BaseLoader):
    def __init__(self, path):
        """

        Parameters
        ----------
        path : str
            The path to the data.
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
                    success = self.load_my_nwb(path)
                    if success:
                        loading_my_data = False

        # Bypass if data have already been transfered to nwb
        if loading_my_data:
            self.load_my_data(path)

            self.save_my_data_in_nwb(path)

    def load_my_data(self, path):
        """
        This load the raw data

        Parameters
        ----------
        path : str
            Path to the session
        """
        """
        Load Raw data here
        """
        print(path)
        return None

    def save_my_data_in_nwb(self, path):
        """
        Save the raw data to NWB

        Parameters
        ----------
        path : TYPE
            Description
        """
        self.nwb_path = os.path.join(path, "pynapplenwb")
        if not os.path.exists(self.nwb_path):
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if "nwb" in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, "r+")

        """
        Save data in NWB here
        """

        io.close()

        return

    def load_my_nwb(self, path):
        """
        This load the nwb that is already create by the base loader

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
        print(nwbfile)

        """
        Add code to write to nwb file here
        """

        io.close()


mydata = MyCustomIO(".")

print(type(mydata))
