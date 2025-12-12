import neo
import numpy as np
import pynapple as nap
from neo.io.proxyobjects import AnalogSignalProxy, SpikeTrainProxy
from neo.core.spiketrainlist import SpikeTrainList
import pathlib


class NEOSignalInterface:

    def __init__(self, signal, block, time_support, sig_num=None):
        self.time_support = time_support
        if isinstance(signal, (neo.AnalogSignal, AnalogSignalProxy)):
            self.is_analog = True
            self.nap_type = self._get_meta_analog(signal)
        elif isinstance(signal, (neo.SpikeTrain, SpikeTrainProxy)):
            self.nap_type = nap.Ts
            self.is_analog = False
        elif isinstance(signal, (list, SpikeTrainList)):
            self.nap_type = nap.TsGroup
            self.is_analog = False
        else:
            raise TypeError(f"signal type {type(signal)} not recognized.")
        self._block = block
        self._sig_num = sig_num

        if self.is_analog:
            self.dt = (1 / signal.sampling_rate).rescale("s").magnitude
            self.shape = signal.shape
        if not issubclass(self.nap_type, nap.TsGroup):
            self.start_time = signal.t_start.rescale("s").magnitude
            self.end_time = signal.t_stop.rescale("s").magnitude
        else:
            self.start_time = [s.t_start.rescale("s").magnitude for s in signal]
            self.end_time = [s.t_stop.rescale("s").magnitude for s in signal]

    @staticmethod
    def _get_meta_analog(signal):
        if len(signal.shape) == 1:
            nap_type = nap.Tsd
        elif len(signal.shape) == 2:
            nap_type = nap.TsdFrame
        else:
            nap_type = nap.TsdTensor
        return nap_type

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._get_from_slice(item)
        raise ValueError(f"Cannot get item {item}.")

    def get(self, start, stop):
        """Get data between start and stop times."""
        if self.is_analog:
            return self._get_analog(start, stop)
        elif issubclass(self.nap_type, nap.Ts):
            return self._get_ts(self._sig_num, start, stop)
        else:  # TsGroup
            return self._get_tsgroup(start, stop)

    def restrict(self, epoch):
        """Restrict data to epochs."""
        if self.is_analog:
            return self._restrict_analog(epoch)
        elif issubclass(self.nap_type, nap.Ts):
            return self._restrict_ts(epoch)
        else:  # TsGroup
            return self._restrict_tsgroup(epoch)

    def _get_from_slice(self, slc):
        start = slc.start if slc.start is not None else 0
        stop = slc.stop
        step = slc.step if slc.step is not None else 1

        if self.is_analog:
            if stop is None:
                stop = sum(s.analogsignals[self._sig_num].shape[0] for s in self._block.segments)
            return self._slice_segment_analog(start, stop, step)
        elif issubclass(self.nap_type, nap.Ts):
            if stop is None:
                stop = sum(len(s.spiketrains[self._sig_num]) for s in self._block.segments)
            return self._slice_segment_ts(start, stop, step)
        else:
            raise ValueError("Cannot slice a TsGroup.")

    def _instantiate_nap(self, time, data, time_support):
        return self.nap_type(
            t=time,
            d=data,
            time_support=time_support,
        )

    def _concatenate_array(self, time_list, data_list):
        if len(data_list) == 0:
            return np.array([]), np.array([]).reshape((0, *self.shape[1:]) if len(self.shape) > 1 else (0, 1))
        else:
            return np.concatenate(time_list), np.concatenate(data_list, axis=0)

    # ========== Analog Signal Methods ==========

    def _get_analog(self, start, stop, return_array=False):
        """Get analog signal between start and stop times."""
        data = []
        time = []

        for i, seg in enumerate(self._block.segments):
            signal = seg.analogsignals[self._sig_num]

            # Get segment boundaries
            seg_start = self.time_support.start[i]
            seg_stop = self.time_support.end[i]

            # Check if requested time overlaps with this segment
            if start >= seg_stop or stop <= seg_start:
                continue  # No overlap, skip this segment

            # Clip to segment bounds
            chunk_start = max(start, seg_start)
            chunk_stop = min(stop, seg_stop)

            chunk = signal.time_slice(chunk_start, chunk_stop)

            if chunk.shape[0] > 0:  # Has data
                data.append(chunk.magnitude)
                time.append(chunk.times.rescale("s").magnitude)

        time, data = self._concatenate_array(time, data)
        if not return_array:
            return self._instantiate_nap(time, data, time_support=self.time_support)
        else:
            return time, data

    def _restrict_analog(self, epoch):
        """Restrict analog signal to epochs."""
        time = []
        data = []

        for start, end in epoch.values:
            time_ep, data_ep = self._get_analog(start, end, return_array=True)
            time.append(time_ep)
            data.append(data_ep)

        time, data = self._concatenate_array(time, data)
        return self._instantiate_nap(time, data, self.time_support).restrict(epoch)

    def _slice_segment_analog(self, start_idx, stop_idx, step):
        """Load by exact indices from each segment."""
        data = []
        time = []

        for i, seg in enumerate(self._block.segments):
            signal = seg.analogsignals[self._sig_num]

            # Segment boundaries from time_support (already in seconds)
            seg_start_time = self.time_support.start[i]
            seg_end_time = self.time_support.end[i]
            seg_duration = seg_end_time - seg_start_time
            seg_n_samples = signal.shape[0]

            # Actual dt for this segment
            dt = seg_duration / seg_n_samples

            # Clip indices to segment bounds
            seg_start_idx = max(0, start_idx)
            seg_stop_idx = min(seg_n_samples, stop_idx)

            if seg_start_idx >= seg_stop_idx:
                continue  # No overlap with this segment

            # Load full segment and slice exactly
            try:
                signal_loaded = signal.load()
                chunk = signal_loaded[seg_start_idx:seg_stop_idx:step]

            except MemoryError:
                # Fallback: use time_slice
                chunk_start_time = seg_start_time + seg_start_idx * dt
                chunk_stop_time = seg_start_time + seg_stop_idx * dt
                chunk = signal.time_slice(chunk_start_time, chunk_stop_time)

                if step != 1:
                    chunk = chunk[::step]

            data.append(chunk.magnitude)
            time.append(chunk.times.rescale("s").magnitude)

        time, data = self._concatenate_array(time, data)
        return self._instantiate_nap(time, data, time_support=self.time_support)

    # ========== Spike Train (Ts) Methods ==========

    def _get_ts(self, unit_idx, start, stop, return_array=False):
        """Get spike times for a unit within time range."""
        spikes = []

        for i, seg in enumerate(self._block.segments):
            spiketrain = seg.spiketrains[unit_idx]

            # Get segment boundaries
            seg_start = self.time_support.start[i]
            seg_stop = self.time_support.end[i]

            # Check if requested time overlaps with this segment
            if start >= seg_stop or stop <= seg_start:
                continue  # No overlap

            # Clip to segment bounds
            chunk_start = max(start, seg_start)
            chunk_stop = min(stop, seg_stop)

            chunk = spiketrain.time_slice(chunk_start, chunk_stop)

            if len(chunk) > 0:  # Has spikes
                spikes.append(chunk.times.rescale("s").magnitude)

        spike_times = np.concatenate(spikes) if spikes else np.array([])

        if return_array:
            return spike_times
        else:
            return nap.Ts(t=spike_times, time_support=self.time_support)

    def _restrict_ts(self, epoch):
        """Restrict spike train to epochs."""
        spikes = []

        for start, end in epoch.values:
            spike_times = self._get_ts(self._sig_num, start, end, return_array=True)
            if len(spike_times) > 0:
                spikes.append(spike_times)

        spike_times = np.concatenate(spikes) if spikes else np.array([])
        return nap.Ts(t=spike_times, time_support=self.time_support).restrict(epoch)

    def _slice_segment_ts(self, start_idx, stop_idx, step):
        """Slice spike trains by spike index."""
        spikes = []

        for i, seg in enumerate(self._block.segments):
            spiketrain = seg.spiketrains[self._sig_num]

            # Get number of spikes in this segment
            n_spikes = len(spiketrain)

            # Clip indices to segment bounds
            seg_start_idx = max(0, start_idx)
            seg_stop_idx = min(n_spikes, stop_idx)

            if seg_start_idx >= seg_stop_idx:
                continue  # No overlap

            # Load and slice by spike index
            spiketrain_loaded = spiketrain.load() if hasattr(spiketrain, 'load') else spiketrain
            chunk = spiketrain_loaded[seg_start_idx:seg_stop_idx:step]

            spikes.append(chunk.times.rescale("s").magnitude)

        return nap.Ts(
            t=np.concatenate(spikes) if spikes else np.array([]),
            time_support=self.time_support
        )

    # ========== TsGroup Methods ==========

    def _get_tsgroup(self, start, stop):
        """Get TsGroup (all units) within time range."""
        n_units = len(self._block.segments[0].spiketrains)
        ts_dict = {}

        for unit_idx in range(n_units):
            spike_times = self._get_ts(unit_idx, start, stop, return_array=True)
            ts_dict[unit_idx] = nap.Ts(t=spike_times, time_support=self.time_support)

        return nap.TsGroup(ts_dict, time_support=self.time_support)

    def _restrict_tsgroup(self, epoch):
        """Restrict TsGroup to epochs."""
        n_units = len(self._block.segments[0].spiketrains)
        ts_dict = {}

        for unit_idx in range(n_units):
            spikes = []
            for start, end in epoch.values:
                spike_times = self._get_ts(unit_idx, start, end, return_array=True)
                if len(spike_times) > 0:
                    spikes.append(spike_times)

            spike_times = np.concatenate(spikes) if spikes else np.array([])
            ts_dict[unit_idx] = nap.Ts(t=spike_times, time_support=self.time_support)

        return nap.TsGroup(ts_dict, time_support=self.time_support).restrict(epoch)


class NEOExperimentInterface:
    def __init__(self, reader, lazy=False):
        # block, aka experiments (contains multiple segments, aka trials)
        self._reader = reader
        self._lazy = lazy
        self.experiment = self._collect_time_series_info()
        self._reader = reader

    def _collect_time_series_info(self):
        blocks = self._reader.read(lazy=self._lazy)

        experiments = {}
        for i, block in enumerate(blocks):
            name = f"block {i}"
            if block.name:
                name += ": " + block.name
            experiments[name] = {}
            # loop once to get the time support
            starts, ends = np.empty(len(block.segments)), np.empty(len(block.segments))
            for trial_num, segment in enumerate(block.segments):
                starts[trial_num] = segment.t_start.rescale("s").magnitude
                ends[trial_num] = segment.t_stop.rescale("s").magnitude

            iset = nap.IntervalSet(starts, ends)
            for trial_num, segment in enumerate(block.segments):
                # segment may contain epoch (potentially overlapping)
                # with fields: times, durations, labels. We may add them to metadata.

                # tsd/tsdFrame/TsdTensor
                for signal_num, signal in enumerate(segment.analogsignals):
                    if signal.name:
                        signame = f" {signal_num}: " + signal.name
                    else:
                        signame = f" {signal_num}"
                    signal_interface = NEOSignalInterface(signal, block, iset, sig_num=signal_num)
                    signame = signal_interface.nap_type.__name__ + signame
                    experiments[name][signame] = signal_interface

                if len(segment.spiketrains) == 1:
                    signal = segment.spiketrains[0]
                    signal_interface = NEOSignalInterface(signal, block, iset, sig_num=0)
                    signame = f"Ts" + ": " + signal.name if signal.name else "Ts"
                    experiments[name][signame] = signal_interface
                else:
                    signame = f"TsGroup"
                    experiments[name][signame] = NEOSignalInterface(segment.spiketrains, block, iset)
        return experiments

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.experiment[item]
        else:
            res = self.experiment
            for it in item:
                res = res[it]
            return res

    def keys(self):
        return [(k, k2) for k in self.experiment.keys() for k2 in self.experiment[k]]


def load_experiment(path: str | pathlib.Path, lazy: bool = True) -> NEOExperimentInterface:
    """
    Load a neural recording experiment.

    Parameters
    ----------
    path : str or Path
        Path to the recording file
    lazy : bool, default True
        Whether to lazy load the data

    Returns
    -------
    NEOExperimentInterface
    """
    path = pathlib.Path(path)
    reader = neo.io.get_io(path)

    return NEOExperimentInterface(reader, lazy=lazy)