"""
> :warning: **DEPRECATED**: This will be removed in version 1.0.0. Check [nwbmatic](https://github.com/pynapple-org/nwbmatic) or [neuroconv](https://github.com/catalystneuro/neuroconv) instead.

Class and functions for loading data processed with Phy2

@author: Sara Mahallati, Guillaume Viejo
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from .. import core as nap
from .ephys_gui import App, EphysGUI
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

        self.time_support = None

        self.sample_rate = None
        self.n_channels_dat = None
        self.channel_map = None
        self.ch_to_sh = None
        self.spikes = None
        self.channel_positions = None

        super().__init__(path)
        # This path stuff should happen only once in the parent class
        self.path = Path(path)
        self.basename = self.path.name
        self.nwb_path = self.path / "pynapplenwb"
        # from what I can see in the loading function, only one nwb file per folder:
        try:
            self.nwb_file = list(self.nwb_path.glob("*.nwb"))[0]
        except IndexError:
            self.nwb_file = None

        # Need to check if nwb file exists and if data are there
        # if self.path is not None:  -> are there any cases where this is None?
        if self.nwb_file is not None:
            loaded_spikes = self.load_nwb_spikes()
            if loaded_spikes is not None:
                return

        # Bypass if data have already been transferred to nwb
        self.load_phy_params()

        app = App()
        window = EphysGUI(app, path=path, groups=self.channel_map)
        app.mainloop()
        try:
            app.update()
        except Exception:
            pass

        if window.status:
            self.ephys_information = window.ephys_information
            self.load_phy_spikes(self.time_support)
            self.save_data()
        app.quit()

    def load_phy_params(self):
        """
        path should be the folder session containing the params.py file

        Function reads :
        1. the number of channels
        2. the sampling frequency of the dat file


        Raises
        ------
        AssertionError
            If path does not contain the params file or channel_map.npy
        """
        assert (
            self.path / "params.py"
        ).exists(), f"Can't find params.py in {self.path}"

        # It is strongly recommended not to conflate parameters and code! Also, there's a library called params.
        # I would recommend putting in the folder a file called params.json, or .txt, or .yml, but not .py!
        # In this way we just read the file, and we don't have to add to sys to import...
        # TODO maybe remove this
        sys.path.append(str(self.path))
        import params as params

        self.sample_rate = params.sample_rate
        self.n_channels_dat = params.n_channels_dat

        assert (
            self.path / "channel_map.npy"
        ).exists(), f"Can't find channel_map.npy in {self.path}"
        channel_map = np.load(self.path / "channel_map.npy")

        if (self.path / "channel_shanks.npy").exists():
            channel_shank = np.load(self.path / "channel_shanks.npy")
            n_shanks = len(np.unique(channel_shank))

            self.channel_map = {
                i: channel_map[channel_shank == i] for i in range(n_shanks)
            }
            self.ch_to_sh = pd.Series(
                index=channel_map.flatten(),
                data=channel_shank.flatten(),
            )
        else:
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

        return

    def load_phy_spikes(self, time_support=None):
        """
        Load Phy spike times and convert to NWB.
        Instantiate automatically a TsGroup object.
        The cluster group is taken first from cluster_info.tsv and second from cluster_group.tsv

        Parameters
        ----------
        path : Path object
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
        # Check if cluster_info.tsv or cluster_group.tsv exists. If both exist, cluster_info.tsv is used:
        has_cluster_info = False
        if (self.path / "cluster_info.tsv").exists():
            cluster_info_file = self.path / "cluster_info.tsv"
            has_cluster_info = True
        elif (self.path / "cluster_group.tsv").exists():
            cluster_info_file = self.path / "cluster_group.tsv"
        else:
            raise RuntimeError(
                "Can't find cluster_info.tsv or cluster_group.tsv in {};".format(
                    self.path
                )
            )

        cluster_info = pd.read_csv(cluster_info_file, sep="\t", index_col="cluster_id")
        # In my processed data with KiloSort 3.0, the column is named KSLabel
        if "group" in cluster_info.columns:
            cluster_id_good = cluster_info[cluster_info.group == "good"].index.values
        elif "KSLabel" in cluster_info.columns:
            cluster_id_good = cluster_info[cluster_info.KSLabel == "good"].index.values
        else:
            raise RuntimeError(
                "Can't find column group or KSLabel in {};".format(cluster_info_file)
            )

        spike_times = np.load(self.path / "spike_times.npy")
        spike_clusters = np.load(self.path / "spike_clusters.npy")

        spikes = {}
        for n in cluster_id_good:
            spikes[n] = nap.Ts(
                t=spike_times[spike_clusters == n] / self.sample_rate,
                time_support=time_support,
            )

        self.spikes = nap.TsGroup(spikes, time_support=time_support)

        # Adding the position of the electrodes in case
        self.channel_positions = np.load(self.path / "channel_positions.npy")

        # Adding shank group info from cluster_info if present
        if has_cluster_info:
            group = cluster_info.loc[cluster_id_good, "sh"]
            self.spikes.set_info(group=group)
        else:
            template = np.load(self.path / "templates.npy")
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

    def save_data(self):
        """Save the data to NWB format."""

        io = NWBHDF5IO(self.nwb_file, "r+")
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

    def load_nwb_spikes(self):
        """Read the NWB spikes to extract the spike times.

        Returns
        -------
        TYPE
            Description
        """

        io = NWBHDF5IO(self.nwb_file, "r")
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
