"""
> :warning: **DEPRECATED**: This will be removed in version 1.0.0. Check [nwbmatic](https://github.com/pynapple-org/nwbmatic) or [neuroconv](https://github.com/catalystneuro/neuroconv) instead.

Class and functions for loading data processed with the Neurosuite (Klusters, Neuroscope, NDmanager)

@author: Guillaume Viejo
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import core as nap
from .loader import BaseLoader


class NeuroSuite(BaseLoader):
    """
    Loader for kluster data
    """

    def __init__(self, path):
        """
        Instantiate the data class from a neurosuite folder.

        Parameters
        ----------
        path : str
            The path to the data.
        """
        path = Path(path)
        self.basename = path.name
        self.time_support = None

        super().__init__(path)

        self.load_nwb_spikes()

    def load_nwb_spikes(self):
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
        pynwb = importlib.import_module("pynwb")
        io = pynwb.NWBHDF5IO(self.nwbfilepath, "r")
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
            filepath = self.path / filename
        else:
            filepath = Path(filename)
            eegfile = list(filepath.glob(f"*{extension}"))

            if not len(eegfile):
                raise RuntimeError(
                    "Path {} contains no {} files;".format(self.path, extension)
                )

            filepath = eegfile[0]

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

    def read_neuroscope_intervals(self, name=None, path2file=None):
        """
        This function reads .evt files in which odd raws indicate the beginning
        of the time series and the even raws are the ends.
        If the file is present in the nwb, provide the just the name. If the file
        is not present in the nwb, it loads the events from the nwb directory.
        If just the path is provided but not the name, it takes the name from the file.

        Parameters
        ----------
        name: str
            name of the epoch in the nwb file, e.g. "rem" or desired name save
            the data in the nwb.

        path2file: str
            Path of the file you want to load.

        Returns
        -------
        IntervalSet
            Contains two columns corresponding to the start and end of the intervals.

        """
        # if name:
        #     isets = self.load_nwb_intervals(name)
        #     if isinstance(isets, nap.IntervalSet):
        #         return isets

        if name is not None and path2file is None:
            path2file = self.path / (self.basename + "." + name + ".evt")
        if path2file is not None:  # TODO maybe useless conditional?
            try:
                # df = pd.read_csv(path2file, delimiter=' ', usecols = [0], header = None)
                tmp = np.genfromtxt(path2file)[:, 0]
                df = tmp.reshape(len(tmp) // 2, 2)
            except ValueError:
                print("specify a valid name")
            isets = nap.IntervalSet(df[:, 0], df[:, 1], time_units="ms")
            if name is None:
                name = path2file.split(".")[-2]
                print("*** saving file in the nwb as", name)
            # self.save_nwb_intervals(isets, name)
        else:
            raise ValueError("specify a valid path")
        return isets

    def write_neuroscope_intervals(self, extension, isets, name):
        """Write events to load with neuroscope (e.g. ripples start and ends)

        Parameters
        ----------
        extension : str
            The extension of the file (e.g. basename.evt.py.rip)
        isets : IntervalSet
            The IntervalSet to write
        name : str
            The name of the events (e.g. Ripples)
        """
        start = isets.as_units("ms")["start"].values
        ends = isets.as_units("ms")["end"].values

        datatowrite = np.vstack((start, ends)).T.flatten()

        n = len(isets)

        texttowrite = np.vstack(
            (
                (np.repeat(np.array([name + " start"]), n)),
                (np.repeat(np.array([name + " end"]), n)),
            )
        ).T.flatten()

        evt_file = self.path / (self.basename + extension)

        f = open(evt_file, "w")
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()

        return

    def _normalize_waveform_window(self, waveform_window):
        if isinstance(waveform_window, nap.IntervalSet):
            return waveform_window
        return nap.IntervalSet(start=-0.5, end=1, time_units="ms")

    def _ensure_path_exists(self):
        if not self.path.exists():  # check if path exists
            print(f"The path {self.path} doesn't exist; Exiting ...")
            sys.exit()

    def _restrict_spikes_to_epoch(self, spikes, epoch, fs):
        if epoch is None:
            return spikes, None

        if type(epoch) is not nap.IntervalSet:
            print("Epoch must be an IntervalSet")
            sys.exit()

        print("Restricting spikes to epoch")
        restricted_spikes = spikes.restrict(epoch)
        epstart = int(epoch.as_units("s")["start"].values[0] * fs)
        epend = int(epoch.as_units("s")["end"].values[0] * fs)
        return restricted_spikes, (epstart, epend)

    def _load_dat_file(self, n_channels):
        file = next(self.path.glob("^[^.][^.]*.dat"))
        f = open(file, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        f.close()
        fp = np.memmap(file, np.int16, "r", shape=(n_samples, n_channels))
        return file, fp, n_samples, bytes_size

    def _sample_counted_spikes(self, sample_spikes, spike_count):
        sample_counted_spikes = {}
        for index, neuron in enumerate(sample_spikes):
            if len(sample_spikes[neuron]) >= spike_count:
                sample_counted_spikes[neuron] = np.array(
                    np.random.choice(list(sample_spikes[neuron]), spike_count)
                )
            elif len(sample_spikes[neuron]) < spike_count:
                print(
                    "Not enough spikes in neuron " + str(index) + "... using all spikes"
                )
                sample_counted_spikes[neuron] = sample_spikes[neuron]
        return sample_counted_spikes

    def _build_batches(self, n_samples, batch_size, overlap, epoch_bounds=None):
        windows = np.arange(0, n_samples, batch_size)
        if epoch_bounds is not None:
            print("Restricting dat file to epoch")
            windows = windows[(windows >= epoch_bounds[0]) & (windows <= epoch_bounds[1])]

        if not len(windows):
            return []

        batches = []
        for i in windows:
            if i == windows[-1]:
                batches.append([i, n_samples])
            else:
                batches.append([i, i + batch_size + overlap])
        return [np.int32(batch) for batch in batches]

    def _accumulate_waveforms(
        self,
        fp,
        batches,
        sample_counted_spikes,
        spikes,
        waveform_window,
        n_channels,
        group_to_channel,
        group,
    ):
        neuron_waveforms = {
            n: np.zeros([np.sum(waveform_window), len(group_to_channel[group[n]])])
            for n in sample_counted_spikes
        }

        spike_check = np.array(
            [
                int(spikes_neuron)
                for spikes_neuron in sample_counted_spikes[neuron]
                for neuron in sample_counted_spikes
            ]
        )

        for index, timestep in enumerate(batches):
            print(
                f"Extracting waveforms from dat file: window {index + 1} / {len(batches)}",
                end="\r",
            )

            if (
                len(
                    spike_check[
                        (timestep[0] < spike_check) & (timestep[1] > spike_check)
                    ]
                )
                == 0
            ):
                continue

            tmp = pd.DataFrame(
                data=fp[timestep[0] : timestep[1], :],
                columns=np.arange(n_channels),
                index=range(timestep[0], timestep[1]),
            )

            for neuron in sample_counted_spikes:
                neurontmp = sample_counted_spikes[neuron]
                tmp2 = neurontmp[(timestep[0] < neurontmp) & (timestep[1] > neurontmp)]
                if len(neurontmp) == 0:
                    continue

                tmpn = tmp[group_to_channel[group[neuron]]]

                for time in tmp2:
                    spikewindow = tmpn.loc[
                        time - waveform_window[0] : time + waveform_window[1] - 1
                    ]
                    try:
                        neuron_waveforms[neuron] += spikewindow.values
                    except Exception:
                        pass

        return neuron_waveforms

    def _build_mean_waveforms(
        self,
        neuron_waveforms,
        sample_counted_spikes,
        spike_count,
        spikes,
        group_to_channel,
        group,
        waveform_window,
        fs,
    ):
        return {
            n: pd.DataFrame(
                data=np.array(neuron_waveforms[n]) / spike_count,
                columns=np.arange(len(group_to_channel[group[n]])),
                index=np.array(np.arange(-waveform_window[0], waveform_window[1])) / fs,
            )
            for n in sample_counted_spikes
        }

    def _build_max_channels(self, meanwf, spikes):
        return pd.Series(
            data=[meanwf[n][meanwf[n].loc[0].idxmin()].name for n in meanwf],
            index=spikes.keys(),
        )

    def load_mean_waveforms(self, epoch=None, waveform_window=None, spike_count=1000):
        """
        Load the mean waveforms from a dat file.

        Parameters
        ----------
        epoch : IntervalSet
            default = None
            Restrict spikes to an epoch.
        waveform_window : IntervalSet
            default interval nap.IntervalSet(start = -0.0005, end = 0.001, time_units = 'ms')
            Limit waveform extraction before and after spike time
        spike_count : int
            default = 1000
            Number of spikes used per neuron for the calculation of waveforms

        Returns
        -------
        dictionary
            the waveforms for all neurons
        pandas.Series
            the channel with the maximum waveform for each neuron

        """
        waveform_window = self._normalize_waveform_window(waveform_window)

        self._ensure_path_exists()

        self.load_neurosuite_xml(self.path)
        n_channels = self.nChannels
        fs = self.fs_dat
        group_to_channel = self.group_to_channel
        spikes = self.spikes
        group = spikes.get_info("group")

        spikes, epoch_bounds = self._restrict_spikes_to_epoch(spikes, epoch, fs)

        _, fp, n_samples, bytes_size = self._load_dat_file(n_channels)

        # map to memory all samples for all channels, channels are numbered according to neuroscope number
        # convert spike times to spikes in sample number
        sample_spikes = {
            neuron: (spikes[neuron].as_units("s").index.values * fs).astype("int")
            for neuron in spikes
        }

        # prep for waveforms
        overlap = int(waveform_window.tot_length(time_units="s"))
        waveform_window = abs(np.array(waveform_window.as_units("s"))[0] * fs).astype(int)

        # divide dat file into batches that slightly overlap for faster loading
        batch_size = 3000000
        sample_counted_spikes = self._sample_counted_spikes(sample_spikes, spike_count)
        batches = self._build_batches(n_samples, batch_size, overlap, epoch_bounds)

        neuron_waveforms = self._accumulate_waveforms(
            fp,
            batches,
            sample_counted_spikes,
            spikes,
            waveform_window,
            n_channels,
            group_to_channel,
            group,
        )

        meanwf = self._build_mean_waveforms(
            neuron_waveforms,
            sample_counted_spikes,
            spike_count,
            spikes,
            group_to_channel,
            group,
            waveform_window,
            fs,
        )

        maxch = self._build_max_channels(meanwf, spikes)

        return meanwf, maxch
