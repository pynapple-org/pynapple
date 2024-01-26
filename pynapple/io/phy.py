"""
> :warning: **DEPRECATED**: This will be removed in version 1.0.0. Check [nwbmatic](https://github.com/pynapple-org/nwbmatic) or [neuroconv](https://github.com/catalystneuro/neuroconv) instead.

Class and functions for loading data processed with Phy2

@author: Sara Mahallati, Guillaume Viejo
"""

import os

import numpy as np
from pynwb import NWBHDF5IO

from .. import core as nap
from .loader import BaseLoader


class Phy(BaseLoader):
    """
    Loader for Phy data
    """

    def __init__(self, path):
        """
        Instantiate the data class from a Phy folder.

        Parameters
        ----------
        path : str or Path object
            The path to the data.
        """
        self.basename = os.path.basename(path)
        self.time_support = None

        super().__init__(path)

        self.load_nwb_spikes(path)

    def load_nwb_spikes(self, path):
        """Read the NWB spikes to extract the spike times.

        Returns
        -------
        TYPE
            Description
        """
        self.nwb_path = os.path.join(path, "pynapplenwb")
        if not os.path.exists(self.nwb_path):
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if "nwb" in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, "r")
        nwbfile = io.read()

        if nwbfile.units is None:
            io.close()
            return None
        else:
            units = nwbfile.units.to_dataframe()
            spikes = {
                n: nap.Ts(t=units.loc[n, "spike_times"], time_units="s")
                for n in units.index
            }

            self.spikes = nap.TsGroup(
                spikes,
                time_support=self.time_support,
                time_units="s",
                group=units["group"],
            )

            if ~np.all(units["location"] == ""):
                self.spikes.set_info(location=units["location"])

            io.close()
            return True

    def load_lfp(
        self,
        filename=None,
        channel=None,
        extension=".eeg",
        frequency=1250.0,
        precision="int16",
        bytes_size=2,
    ):
        """
        Load the LFP.

        Parameters
        ----------
        filename : str, optional
            The filename of the lfp file.
            It can be useful it multiple dat files are present in the data directory
        channel : int or list of int, optional
            The channel(s) to load. If None return a memory map of the dat file to avoid memory error
        extension : str, optional
            The file extenstion (.eeg, .dat, .lfp). Make sure the frequency match
        frequency : float, optional
            Default 1250 Hz for the eeg file
        precision : str, optional
            The precision of the binary file
        bytes_size : int, optional
            Bytes size of the lfp file

        Raises
        ------
        RuntimeError
            If can't find the lfp/eeg/dat file

        Returns
        -------
        Tsd or TsdFrame
            The lfp in a time series format
        """
        if filename is not None:
            filepath = self.path / filename
        else:
            try:
                filepath = list(self.path.glob(f"*{extension}"))[0]
            except IndexError:
                raise RuntimeError(f"Path {self.path} contains no {extension} files;")

        # is it possible that this is a leftover from neurosuite data?
        # This is not implemented for this class.
        self.load_neurosuite_xml(self.path)

        n_channels = int(self.nChannels)

        f = open(filepath, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        duration = n_samples / frequency
        f.close()
        fp = np.memmap(filepath, np.int16, "r", shape=(n_samples, n_channels))
        timestep = np.arange(0, n_samples) / frequency

        time_support = nap.IntervalSet(start=0, end=duration, time_units="s")

        if channel is None:
            return nap.TsdFrame(
                t=timestep, d=fp, time_units="s", time_support=time_support
            )
        elif type(channel) is int:
            return nap.Tsd(
                t=timestep, d=fp[:, channel], time_units="s", time_support=time_support
            )
        elif type(channel) is list:
            return nap.TsdFrame(
                t=timestep,
                d=fp[:, channel],
                time_units="s",
                time_support=time_support,
                columns=channel,
            )
