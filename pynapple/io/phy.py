"""
Class and functions for loading data processed with Phy2

@author: Sara Mahallati, Guillaume Viejo
"""
import os
import re
import sys
from xml.dom import minidom

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO, NWBFile
from pynwb.device import Device
from PyQt5.QtWidgets import QApplication

from .. import core as nap
from .ephys_gui import EphysGUI
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
        path : str
            The path to the data.
        """
        self.basename = os.path.basename(path)
        self.time_support = None

        super().__init__(path)

        # Need to check if nwb file exists and if data are there
        loading_phy = True
        if self.path is not None:
            nwb_path = os.path.join(self.path, "pynapplenwb")
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith(".nwb")]):
                    success = self.load_nwb_spikes(path)
                    if success:
                        loading_phy = False

        # Bypass if data have already been transfered to nwb
        if loading_phy:
            self.load_phy_params(path)
            app = QApplication([])
            window = EphysGUI(path=path, groups=self.channel_map)
            window.show()
            app.exec()
            if window.status:
                self.ephys_information = window.ephys_information
                self.load_phy_spikes(path, self.time_support)
                self.save_data(path)
            app.quit()

    def load_phy_params(self, path):
        """
        path should be the folder session containing the params.py file

        Function reads :
        1. the number of channels
        2. the sampling frequency of the dat file

        Parameters
        ----------
        path: str
            The path to the data

        Raises
        ------
        RuntimeError
            If path does not contain the params file or channel_map.npy
        """
        if os.path.exists(path):
            listdir = os.listdir(path)

        if os.path.isfile(os.path.join(path, "params.py")):
            sys.path.append(path)
            import params as params

            self.sample_rate = params.sample_rate
            self.n_channels_dat = params.n_channels_dat
        else:
            raise RuntimeError("Can't find params.py in path {};".format(path))

        if os.path.isfile(os.path.join(path, "channel_map.npy")):
            channel_map = np.load(os.path.join(path, "channel_map.npy"))
            self.channel_map = {i: channel_map[i] for i in range(len(channel_map))}
            self.ch_to_sh = pd.Series(
                index=channel_map.flatten(),
                data=np.hstack(
                    [
                        np.ones(len(channel_map[i]), dtype=int) * i
                        for i in range(len(channel_map))
                    ]
                ),
            )
        else:
            raise RuntimeError("Can't find channel_map.npy in path {};".format(path))

        return

    def load_phy_spikes(self, path, time_support=None):
        """
        Load Phy spike times and convert to NWB.
        Instantiate automatically a TsGroup object.
        The cluster group is taken first from cluster_info.tsv and second from cluster_group.tsv

        Parameters
        ----------
        path : str
            The path to the data
        time_support : IntevalSet, optional
            The time support of the data

        Raises
        ------
        RuntimeError
            If files are missing.
            The function needs :
            - cluster_info.tsv or cluster_group.tsv
            - spike_times.npy
            - spike_clusters.npy
            - channel_positions.npy
            - templates.npy

        """
        files = os.listdir(path)

        has_cluster_info = False
        if "cluster_info.tsv" in files:
            self.cluster_info = pd.read_csv(
                os.path.join(path, "cluster_info.tsv"), sep="\t", index_col="cluster_id"
            )
            cluster_id_good = self.cluster_info[
                self.cluster_info.group == "good"
            ].index.values
            has_cluster_info = True
        elif "cluster_group.tsv" in files:
            self.cluster_group = pd.read_csv(
                os.path.join(path, "cluster_group.tsv"),
                sep="\t",
                index_col="cluster_id",
            )
            cluster_id_good = self.cluster_group[
                self.cluster_group.group == "good"
            ].index.values
        else:
            raise RuntimeError(
                "Can't find cluster_info.tsv or cluster_group.tsv in {};".format(path)
            )

        spike_times = np.load(os.path.join(path, "spike_times.npy"))
        spike_clusters = np.load(os.path.join(path, "spike_clusters.npy"))

        spikes = {}
        for n in cluster_id_good:
            spikes[n] = nap.Ts(
                t=spike_times[spike_clusters == n] / self.sample_rate,
                time_support=time_support,
            )

        self.spikes = nap.TsGroup(spikes, time_support=time_support)

        # Adding the position of the electrodes in case
        self.channel_positions = np.load(os.path.join(path, "channel_positions.npy"))

        # Adding shank group info from cluster_info if present
        if has_cluster_info:
            group = self.cluster_info.loc[cluster_id_good, "sh"]
            self.spikes.set_info(group=group)
        else:
            template = np.load(os.path.join(path, "templates.npy"))
            template = template[cluster_id_good]
            ch = np.power(template, 2).max(1).argmax(1)
            group = pd.Series(index=cluster_id_good, data=self.ch_to_sh[ch].values)
            self.spikes.set_info(group=group)

        names = pd.Series(
            index=group.index,
            data=[self.ephys_information[group.loc[i]]["name"] for i in group.index],
        )
        if ~np.all(names.values == ""):
            self.spikes.set_info(name=names)

        locations = pd.Series(
            index=group.index,
            data=[
                self.ephys_information[group.loc[i]]["location"] for i in group.index
            ],
        )
        if ~np.all(locations.values == ""):
            self.spikes.set_info(location=locations)

        return

    def save_data(self, path):
        """
        Save the data to NWB format.

        Parameters
        ----------
        path : str
            The path to save the data

        """
        self.nwb_path = os.path.join(path, "pynapplenwb")
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if "nwb" in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, "r+")
        nwbfile = io.read()

        electrode_groups = {}

        for g in self.channel_map:

            device = nwbfile.create_device(
                name=self.ephys_information[g]["device"]["name"] + "-" + str(g),
                description=self.ephys_information[g]["device"]["description"],
                manufacturer=self.ephys_information[g]["device"]["manufacturer"],
            )

            if (
                len(self.ephys_information[g]["position"])
                and type(self.ephys_information[g]["position"]) is str
            ):
                self.ephys_information[g]["position"] = re.split(
                    ";|,| ", self.ephys_information[g]["position"]
                )
            elif self.ephys_information[g]["position"] == "":
                self.ephys_information[g]["position"] = None

            electrode_groups[g] = nwbfile.create_electrode_group(
                name="group" + str(g) + "_" + self.ephys_information[g]["name"],
                description=self.ephys_information[g]["description"],
                position=self.ephys_information[g]["position"],
                location=self.ephys_information[g]["location"],
                device=device,
            )

            for idx in self.channel_map[g]:
                nwbfile.add_electrode(
                    id=idx,
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    imp=0.0,
                    location=self.ephys_information[g]["location"],
                    filtering="none",
                    group=electrode_groups[g],
                )

        # Adding units
        nwbfile.add_unit_column("location", "the anatomical location of this unit")
        nwbfile.add_unit_column("group", "the group of the unit")
        for u in self.spikes.keys():
            nwbfile.add_unit(
                id=u,
                spike_times=self.spikes[u].as_units("s").index.values,
                electrode_group=electrode_groups[self.spikes.get_info("group").loc[u]],
                location=self.ephys_information[self.spikes.get_info("group").loc[u]][
                    "location"
                ],
                group=self.spikes.get_info("group").loc[u],
            )

        io.write(nwbfile)
        io.close()

        return

    def load_nwb_spikes(self, path):
        """
        Read the NWB spikes to extract the spike times.

        Parameters
        ----------
        path : str
            The path to the data

        Returns
        -------
        TYPE
            Description
        """
        self.nwb_path = os.path.join(path, "pynapplenwb")
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if "nwb" in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, "r")
        nwbfile = io.read()

        if nwbfile.units is None:
            io.close()
            return False
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
            filepath = os.path.join(self.path, filename)
        else:
            listdir = os.listdir(self.path)
            eegfile = [f for f in listdir if f.endswith(extension)]
            if not len(eegfile):
                raise RuntimeError(
                    "Path {} contains no {} files;".format(self.path, extension)
                )

            filepath = os.path.join(self.path, eegfile[0])

        self.load_neurosuite_xml(self.path)

        n_channels = int(self.nChannels)

        f = open(filepath, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        duration = n_samples / frequency
        interval = 1 / frequency
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
