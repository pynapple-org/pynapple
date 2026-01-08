"""
Pynapple interface for Neo (neural electrophysiology objects).

Neo is a Python package for working with electrophysiology data in Python,
supporting many file formats through a unified API.

Data are lazy-loaded by default. The interface behaves like a dictionary.

For more information on Neo, see: https://neo.readthedocs.io/

Neo to Pynapple Object Conversion
---------------------------------
The following Neo objects are converted to their pynapple equivalents:

- neo.AnalogSignal -> Tsd, TsdFrame, or TsdTensor (depending on shape)
- neo.IrregularlySampledSignal -> Tsd, TsdFrame, or TsdTensor (depending on shape)
- neo.SpikeTrain -> Ts
- neo.SpikeTrain (list) -> TsGroup
- neo.SpikeTrainList -> TsGroup
- neo.Epoch -> IntervalSet
- neo.Event -> Ts
"""

import warnings
from collections import UserDict
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import numpy as np

try:
    import neo
    from neo.io.proxyobjects import (
        AnalogSignalProxy,
        SpikeTrainProxy,
        EpochProxy,
        EventProxy,
    )
    from neo.core.spiketrainlist import SpikeTrainList

    HAS_NEO = True
except ImportError:
    HAS_NEO = False

try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from .. import core as nap


def _check_neo_installed():
    """Check if neo is installed and raise ImportError if not."""
    if not HAS_NEO:
        raise ImportError(
            "Neo is required for this functionality. "
            "Install it with: pip install neo"
        )


def _rescale_to_seconds(quantity):
    """Convert a neo quantity to seconds.

    Parameters
    ----------
    quantity : neo.core.baseneo.BaseNeo or quantities.Quantity
        A quantity with time units

    Returns
    -------
    float
        Value in seconds
    """
    return float(quantity.rescale("s").magnitude)


def _get_signal_type(signal) -> type:
    """Determine the appropriate pynapple type for a Neo signal.

    Parameters
    ----------
    signal : neo.AnalogSignal or neo.IrregularlySampledSignal
        The Neo signal object

    Returns
    -------
    type
        The pynapple type (Tsd, TsdFrame, or TsdTensor)
    """
    if len(signal.shape) == 1:
        return nap.Tsd
    elif len(signal.shape) == 2:
        if signal.shape[1] == 1:
            return nap.Tsd
        return nap.TsdFrame
    else:
        return nap.TsdTensor


def _extract_annotations(obj) -> Dict[str, Any]:
    """Extract annotations from a Neo object.

    Parameters
    ----------
    obj : neo.core.baseneo.BaseNeo
        Any Neo object with annotations

    Returns
    -------
    dict
        Dictionary of annotations
    """
    annotations = {}
    if hasattr(obj, "annotations") and obj.annotations:
        annotations.update(obj.annotations)
    if hasattr(obj, "name") and obj.name:
        annotations["neo_name"] = obj.name
    if hasattr(obj, "description") and obj.description:
        annotations["neo_description"] = obj.description
    return annotations


def _extract_array_annotations(obj) -> Dict[str, np.ndarray]:
    """Extract array annotations from a Neo object.

    Parameters
    ----------
    obj : neo.core.baseneo.BaseNeo
        Any Neo object with array_annotations

    Returns
    -------
    dict
        Dictionary of array annotations
    """
    if hasattr(obj, "array_annotations") and obj.array_annotations:
        return dict(obj.array_annotations)
    return {}


# =============================================================================
# Conversion functions: Neo -> Pynapple
# =============================================================================


def _make_intervalset_from_epoch(epoch, time_support: Optional[nap.IntervalSet] = None) -> nap.IntervalSet:
    """Convert a Neo Epoch to a pynapple IntervalSet.

    Parameters
    ----------
    epoch : neo.Epoch or neo.io.proxyobjects.EpochProxy
        Neo Epoch object
    time_support : IntervalSet, optional
        Time support for the IntervalSet

    Returns
    -------
    IntervalSet
        Pynapple IntervalSet
    """
    if hasattr(epoch, "load"):
        epoch = epoch.load()

    times = epoch.times.rescale("s").magnitude
    durations = epoch.durations.rescale("s").magnitude

    starts = times
    ends = times + durations

    # Extract labels as metadata if available
    metadata = {}
    if hasattr(epoch, "labels") and len(epoch.labels) > 0:
        metadata["label"] = np.array(epoch.labels)

    # Add any other annotations
    annotations = _extract_annotations(epoch)

    iset = nap.IntervalSet(start=starts, end=ends, metadata=metadata)

    return iset


def _make_ts_from_event(event, time_support: Optional[nap.IntervalSet] = None) -> nap.Ts:
    """Convert a Neo Event to a pynapple Ts.

    Parameters
    ----------
    event : neo.Event or neo.io.proxyobjects.EventProxy
        Neo Event object
    time_support : IntervalSet, optional
        Time support for the Ts

    Returns
    -------
    Ts
        Pynapple Ts object
    """
    if hasattr(event, "load"):
        event = event.load()

    times = event.times.rescale("s").magnitude

    return nap.Ts(t=times, time_support=time_support)


def _make_tsd_from_analog(
    signal,
    time_support: Optional[nap.IntervalSet] = None,
    column_names: Optional[List[str]] = None,
) -> Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor]:
    """Convert a Neo AnalogSignal to a pynapple Tsd/TsdFrame/TsdTensor.

    Parameters
    ----------
    signal : neo.AnalogSignal or AnalogSignalProxy
        Neo analog signal
    time_support : IntervalSet, optional
        Time support
    column_names : list of str, optional
        Column names for TsdFrame

    Returns
    -------
    Tsd, TsdFrame, or TsdTensor
        Appropriate pynapple time series object
    """
    if hasattr(signal, "load"):
        signal = signal.load()

    times = signal.times.rescale("s").magnitude
    data = signal.magnitude

    nap_type = _get_signal_type(signal)

    if nap_type == nap.Tsd:
        if len(data.shape) == 2:
            data = data.squeeze()
        return nap.Tsd(t=times, d=data, time_support=time_support)
    elif nap_type == nap.TsdFrame:
        if column_names is None:
            # Try to get channel names from annotations
            if hasattr(signal, "array_annotations"):
                channel_names = signal.array_annotations.get("channel_names", None)
                if channel_names is not None:
                    column_names = list(channel_names)
        return nap.TsdFrame(t=times, d=data, columns=column_names, time_support=time_support)
    else:
        return nap.TsdTensor(t=times, d=data, time_support=time_support)


def _make_tsd_from_irregular(
    signal,
    time_support: Optional[nap.IntervalSet] = None,
    column_names: Optional[List[str]] = None,
) -> Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor]:
    """Convert a Neo IrregularlySampledSignal to a pynapple Tsd/TsdFrame/TsdTensor.

    Parameters
    ----------
    signal : neo.IrregularlySampledSignal
        Neo irregularly sampled signal
    time_support : IntervalSet, optional
        Time support
    column_names : list of str, optional
        Column names for TsdFrame

    Returns
    -------
    Tsd, TsdFrame, or TsdTensor
        Appropriate pynapple time series object
    """
    if hasattr(signal, "load"):
        signal = signal.load()

    times = signal.times.rescale("s").magnitude
    data = signal.magnitude

    nap_type = _get_signal_type(signal)

    if nap_type == nap.Tsd:
        if len(data.shape) == 2:
            data = data.squeeze()
        return nap.Tsd(t=times, d=data, time_support=time_support)
    elif nap_type == nap.TsdFrame:
        return nap.TsdFrame(t=times, d=data, columns=column_names, time_support=time_support)
    else:
        return nap.TsdTensor(t=times, d=data, time_support=time_support)


def _make_ts_from_spiketrain(
    spiketrain, time_support: Optional[nap.IntervalSet] = None
) -> nap.Ts:
    """Convert a Neo SpikeTrain to a pynapple Ts.

    Parameters
    ----------
    spiketrain : neo.SpikeTrain or SpikeTrainProxy
        Neo spike train
    time_support : IntervalSet, optional
        Time support

    Returns
    -------
    Ts
        Pynapple Ts object
    """
    if hasattr(spiketrain, "load"):
        spiketrain = spiketrain.load()

    times = spiketrain.times.rescale("s").magnitude

    return nap.Ts(t=times, time_support=time_support)


def _make_tsgroup_from_spiketrains(
    spiketrains: Union[list, "SpikeTrainList"],
    time_support: Optional[nap.IntervalSet] = None,
) -> nap.TsGroup:
    """Convert a list of Neo SpikeTrains to a pynapple TsGroup.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain or SpikeTrainList
        List of spike trains
    time_support : IntervalSet, optional
        Time support

    Returns
    -------
    TsGroup
        Pynapple TsGroup
    """
    ts_dict = {}
    metadata = {}

    for i, st in enumerate(spiketrains):
        if hasattr(st, "load"):
            st = st.load()

        times = st.times.rescale("s").magnitude
        ts_dict[i] = nap.Ts(t=times, time_support=time_support)

        # Collect metadata from annotations
        for key, value in _extract_annotations(st).items():
            if key not in metadata:
                metadata[key] = []
            metadata[key].append(value)

    # Convert metadata lists to arrays
    meta_arrays = {}
    for key, values in metadata.items():
        try:
            meta_arrays[key] = np.array(values)
        except (ValueError, TypeError):
            # Skip metadata that can't be converted to array
            pass

    return nap.TsGroup(ts_dict, time_support=time_support, **meta_arrays)



# =============================================================================
# Signal Interface for lazy loading
# =============================================================================


class NeoSignalInterface:
    """Interface for lazy-loading Neo signals into pynapple objects.

    This class provides lazy access to Neo signals, loading data only when
    requested via `get()` or `restrict()` methods.

    Parameters
    ----------
    signal : neo signal object
        A Neo signal (AnalogSignal, SpikeTrain, etc.)
    block : neo.Block
        The parent block containing the signal
    time_support : IntervalSet
        Time support for the data
    sig_num : int, optional
        Index of the signal within the segment

    Attributes
    ----------
    nap_type : type
        The pynapple type this signal will be converted to
    is_analog : bool
        Whether this is an analog signal
    dt : float
        Sampling interval (for analog signals)
    shape : tuple
        Shape of the data
    start_time : float or list
        Start time(s)
    end_time : float or list
        End time(s)
    """

    def __init__(self, signal, block, time_support, sig_num=None):
        self.time_support = time_support
        self._block = block
        self._sig_num = sig_num

        # Determine signal type and pynapple mapping
        if isinstance(signal, (neo.AnalogSignal, AnalogSignalProxy)):
            self.is_analog = True
            self.nap_type = _get_signal_type(signal)
            self._signal_type = "analog"
        elif hasattr(neo, "IrregularlySampledSignal") and isinstance(
            signal, neo.IrregularlySampledSignal
        ):
            self.is_analog = False  # Irregularly sampled
            self.nap_type = _get_signal_type(signal)
            self._signal_type = "irregular"
        elif isinstance(signal, (neo.SpikeTrain, SpikeTrainProxy)):
            self.nap_type = nap.Ts
            self.is_analog = False
            self._signal_type = "spiketrain"
        elif isinstance(signal, (list, SpikeTrainList)):
            self.nap_type = nap.TsGroup
            self.is_analog = False
            self._signal_type = "tsgroup"
        elif isinstance(signal, (neo.Epoch,)) or (hasattr(neo.io, "proxyobjects") and isinstance(signal, EpochProxy)):
            self.nap_type = nap.IntervalSet
            self.is_analog = False
            self._signal_type = "epoch"
        elif isinstance(signal, (neo.Event,)) or (hasattr(neo.io, "proxyobjects") and isinstance(signal, EventProxy)):
            self.nap_type = nap.Ts
            self.is_analog = False
            self._signal_type = "event"
        else:
            raise TypeError(f"Signal type {type(signal)} not recognized.")

        # Store timing info
        if self.is_analog:
            self.dt = (1 / signal.sampling_rate).rescale("s").magnitude
            self.shape = signal.shape
        elif self._signal_type == "irregular":
            self.shape = signal.shape

        if self._signal_type not in ("tsgroup",):
            self.start_time = _rescale_to_seconds(signal.t_start)
            self.end_time = _rescale_to_seconds(signal.t_stop)
        else:
            self.start_time = [_rescale_to_seconds(s.t_start) for s in signal]
            self.end_time = [_rescale_to_seconds(s.t_stop) for s in signal]

    def __repr__(self):
        return f"<NeoSignalInterface: {self.nap_type.__name__}>"

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._get_from_slice(item)
        raise ValueError(f"Cannot get item {item}.")

    def get(self, start: float, stop: float):
        """Get data between start and stop times.

        Parameters
        ----------
        start : float
            Start time in seconds
        stop : float
            Stop time in seconds

        Returns
        -------
        pynapple object
            Data restricted to the time range
        """
        if self.is_analog:
            return self._get_analog(start, stop)
        elif self._signal_type == "irregular":
            return self._get_irregular(start, stop)
        elif self._signal_type == "spiketrain":
            return self._get_ts(self._sig_num, start, stop)
        elif self._signal_type == "tsgroup":
            return self._get_tsgroup(start, stop)
        elif self._signal_type == "epoch":
            return self._get_epoch(start, stop)
        elif self._signal_type == "event":
            return self._get_event(start, stop)

    def load(self):
        """Load all data.

        Returns
        -------
        pynapple object
            The fully loaded data
        """
        start = float(self.time_support.start[0])
        end = float(self.time_support.end[-1])
        return self.get(start, end)

    def restrict(self, epoch: nap.IntervalSet):
        """Restrict data to epochs.

        Parameters
        ----------
        epoch : IntervalSet
            Epochs to restrict to

        Returns
        -------
        pynapple object
            Data restricted to the epochs
        """
        if self.is_analog:
            return self._restrict_analog(epoch)
        elif self._signal_type == "irregular":
            return self._restrict_irregular(epoch)
        elif self._signal_type == "spiketrain":
            return self._restrict_ts(epoch)
        elif self._signal_type == "tsgroup":
            return self._restrict_tsgroup(epoch)
        elif self._signal_type == "epoch":
            return self._get_epoch(
                float(epoch.start[0]), float(epoch.end[-1])
            ).restrict(epoch)
        elif self._signal_type == "event":
            return self._get_event(
                float(epoch.start[0]), float(epoch.end[-1])
            ).restrict(epoch)

    def _get_from_slice(self, slc):
        start = slc.start if slc.start is not None else 0
        stop = slc.stop
        step = slc.step if slc.step is not None else 1

        if self.is_analog:
            if stop is None:
                stop = sum(
                    s.analogsignals[self._sig_num].shape[0]
                    for s in self._block.segments
                )
            return self._slice_segment_analog(start, stop, step)
        elif self._signal_type == "spiketrain":
            if stop is None:
                stop = sum(
                    len(s.spiketrains[self._sig_num]) for s in self._block.segments
                )
            return self._slice_segment_ts(start, stop, step)
        else:
            raise ValueError(f"Cannot slice a {self._signal_type}.")

    def _instantiate_nap(self, time, data, time_support):
        return self.nap_type(
            t=time,
            d=data,
            time_support=time_support,
        )

    def _concatenate_array(self, time_list, data_list):
        if len(data_list) == 0:
            shape = getattr(self, "shape", (0, 1))
            return np.array([]), np.array([]).reshape(
                (0, *shape[1:]) if len(shape) > 1 else (0,)
            )
        else:
            return np.concatenate(time_list), np.concatenate(data_list, axis=0)

    # ========== Analog Signal Methods ==========

    def _get_analog(self, start, stop, return_array=False):
        """Get analog signal between start and stop times."""
        data = []
        time = []

        for i, seg in enumerate(self._block.segments):
            signal = seg.analogsignals[self._sig_num]

            seg_start = self.time_support.start[i]
            seg_stop = self.time_support.end[i]

            if start >= seg_stop or stop <= seg_start:
                continue

            chunk_start = max(start, seg_start)
            chunk_stop = min(stop, seg_stop)

            chunk = signal.time_slice(chunk_start, chunk_stop)

            if chunk.shape[0] > 0:
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

            seg_start_time = self.time_support.start[i]
            seg_end_time = self.time_support.end[i]
            seg_duration = seg_end_time - seg_start_time
            seg_n_samples = signal.shape[0]

            dt = seg_duration / seg_n_samples

            seg_start_idx = max(0, start_idx)
            seg_stop_idx = min(seg_n_samples, stop_idx)

            if seg_start_idx >= seg_stop_idx:
                continue

            try:
                signal_loaded = signal.load()
                chunk = signal_loaded[seg_start_idx:seg_stop_idx:step]
            except MemoryError:
                chunk_start_time = seg_start_time + seg_start_idx * dt
                chunk_stop_time = seg_start_time + seg_stop_idx * dt
                chunk = signal.time_slice(chunk_start_time, chunk_stop_time)
                if step != 1:
                    chunk = chunk[::step]

            data.append(chunk.magnitude)
            time.append(chunk.times.rescale("s").magnitude)

        time, data = self._concatenate_array(time, data)
        return self._instantiate_nap(time, data, time_support=self.time_support)

    # ========== Irregularly Sampled Signal Methods ==========

    def _get_irregular(self, start, stop, return_array=False):
        """Get irregularly sampled signal between start and stop times."""
        data = []
        time = []

        for i, seg in enumerate(self._block.segments):
            signal = seg.irregularlysampledsignals[self._sig_num]

            seg_start = self.time_support.start[i]
            seg_stop = self.time_support.end[i]

            if start >= seg_stop or stop <= seg_start:
                continue

            chunk_start = max(start, seg_start)
            chunk_stop = min(stop, seg_stop)

            chunk = signal.time_slice(chunk_start, chunk_stop)

            if chunk.shape[0] > 0:
                data.append(chunk.magnitude)
                time.append(chunk.times.rescale("s").magnitude)

        if len(time) == 0:
            time = np.array([])
            data = np.array([])
        else:
            time = np.concatenate(time)
            data = np.concatenate(data, axis=0)

        if not return_array:
            if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
                if data.ndim == 2:
                    data = data.squeeze()
                return nap.Tsd(t=time, d=data, time_support=self.time_support)
            elif data.ndim == 2:
                return nap.TsdFrame(t=time, d=data, time_support=self.time_support)
            else:
                return nap.TsdTensor(t=time, d=data, time_support=self.time_support)
        else:
            return time, data

    def _restrict_irregular(self, epoch):
        """Restrict irregularly sampled signal to epochs."""
        time = []
        data = []

        for start, end in epoch.values:
            time_ep, data_ep = self._get_irregular(start, end, return_array=True)
            if len(time_ep) > 0:
                time.append(time_ep)
                data.append(data_ep)

        if len(time) == 0:
            return nap.Tsd(t=np.array([]), d=np.array([]), time_support=epoch)

        time = np.concatenate(time)
        data = np.concatenate(data, axis=0)

        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            if data.ndim == 2:
                data = data.squeeze()
            return nap.Tsd(t=time, d=data, time_support=self.time_support).restrict(
                epoch
            )
        elif data.ndim == 2:
            return nap.TsdFrame(
                t=time, d=data, time_support=self.time_support
            ).restrict(epoch)
        else:
            return nap.TsdTensor(
                t=time, d=data, time_support=self.time_support
            ).restrict(epoch)

    # ========== Spike Train (Ts) Methods ==========

    def _get_ts(self, unit_idx, start, stop, return_array=False):
        """Get spike times for a unit within time range."""
        spikes = []

        for i, seg in enumerate(self._block.segments):
            spiketrain = seg.spiketrains[unit_idx]

            seg_start = self.time_support.start[i]
            seg_stop = self.time_support.end[i]

            if start >= seg_stop or stop <= seg_start:
                continue

            chunk_start = max(start, seg_start)
            chunk_stop = min(stop, seg_stop)

            chunk = spiketrain.time_slice(chunk_start, chunk_stop)

            if len(chunk) > 0:
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

            n_spikes = len(spiketrain)

            seg_start_idx = max(0, start_idx)
            seg_stop_idx = min(n_spikes, stop_idx)

            if seg_start_idx >= seg_stop_idx:
                continue

            spiketrain_loaded = (
                spiketrain.load() if hasattr(spiketrain, "load") else spiketrain
            )
            chunk = spiketrain_loaded[seg_start_idx:seg_stop_idx:step]

            spikes.append(chunk.times.rescale("s").magnitude)

        return nap.Ts(
            t=np.concatenate(spikes) if spikes else np.array([]),
            time_support=self.time_support,
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

    # ========== Epoch Methods ==========

    def _get_epoch(self, start, stop):
        """Get epochs within time range."""
        all_starts = []
        all_ends = []
        all_labels = []

        for i, seg in enumerate(self._block.segments):
            for epoch in seg.epochs:
                if hasattr(epoch, "load"):
                    epoch = epoch.load()

                times = epoch.times.rescale("s").magnitude
                durations = epoch.durations.rescale("s").magnitude

                for t, d, lbl in zip(times, durations, epoch.labels):
                    ep_start = t
                    ep_end = t + d

                    # Check overlap with requested range
                    if ep_end > start and ep_start < stop:
                        all_starts.append(max(ep_start, start))
                        all_ends.append(min(ep_end, stop))
                        all_labels.append(lbl)

        if len(all_starts) == 0:
            return nap.IntervalSet(start=[], end=[])

        return nap.IntervalSet(
            start=np.array(all_starts),
            end=np.array(all_ends),
            metadata={"label": np.array(all_labels)} if all_labels else None,
        )

    # ========== Event Methods ==========

    def _get_event(self, start, stop):
        """Get events within time range."""
        all_times = []

        for i, seg in enumerate(self._block.segments):
            for event in seg.events:
                if hasattr(event, "load"):
                    event = event.load()

                times = event.times.rescale("s").magnitude

                mask = (times >= start) & (times <= stop)
                all_times.extend(times[mask])

        return nap.Ts(t=np.array(all_times), time_support=self.time_support)


# =============================================================================
# Main Interface Class
# =============================================================================


class NeoReader(UserDict):
    """Class for reading Neo-compatible files.

    This class provides a dictionary-like interface to Neo files, with
    lazy-loading support. It automatically detects the appropriate IO
    based on the file extension.

    Parameters
    ----------
    file : str or Path
        Path to the file to load
    lazy : bool, default True
        Whether to use lazy loading

    Examples
    --------
    >>> import pynapple as nap
    >>> data = nap.io.NeoReader("my_file.plx")
    >>> print(data)
    my_file
    +---------------------+----------+
    | Key                 | Type     |
    +=====================+==========+
    | TsGroup             | TsGroup  |
    | Tsd 0: LFP          | Tsd      |
    +---------------------+----------+

    >>> spikes = data["TsGroup"]
    >>> lfp = data["Tsd 0: LFP"]
    """

    def __init__(self, file: Union[str, Path], lazy: bool = True):
        _check_neo_installed()

        self.path = Path(file)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {file}")

        self.name = self.path.stem
        self._lazy = lazy

        # Get appropriate IO
        self._reader = neo.io.get_io(str(self.path))

        # Read blocks
        self._blocks = self._reader.read(lazy=lazy)

        # Build data dictionary
        self.data = {}
        self._data_info = {}  # Store type info for display
        self._interfaces = {}  # Store NeoSignalInterface objects

        self._collect_data()

        UserDict.__init__(self, self.data)

    def _collect_data(self):
        """Collect all data from Neo blocks into the dictionary."""
        for block_idx, block in enumerate(self._blocks):
            block_prefix = "" if len(self._blocks) == 1 else f"block{block_idx}/"

            # Build time support from segments
            starts = np.array(
                [_rescale_to_seconds(seg.t_start) for seg in block.segments]
            )
            ends = np.array(
                [_rescale_to_seconds(seg.t_stop) for seg in block.segments]
            )
            time_support = nap.IntervalSet(starts, ends)

            # Process first segment to get signal info
            # (assuming consistent structure across segments)
            if len(block.segments) > 0:
                seg = block.segments[0]

                # Analog signals
                for sig_idx, signal in enumerate(seg.analogsignals):
                    nap_type = _get_signal_type(signal)
                    name = signal.name if signal.name else f"signal{sig_idx}"
                    key = f"{block_prefix}{nap_type.__name__} {sig_idx}: {name}"

                    interface = NeoSignalInterface(
                        signal, block, time_support, sig_num=sig_idx
                    )
                    self._interfaces[key] = interface
                    self.data[key] = {"type": nap_type.__name__, "interface": interface}
                    self._data_info[key] = nap_type.__name__

                # Irregularly sampled signals
                for sig_idx, signal in enumerate(seg.irregularlysampledsignals):
                    nap_type = _get_signal_type(signal)
                    name = signal.name if signal.name else f"irregular{sig_idx}"
                    key = f"{block_prefix}{nap_type.__name__} (irregular) {sig_idx}: {name}"

                    interface = NeoSignalInterface(
                        signal, block, time_support, sig_num=sig_idx
                    )
                    self._interfaces[key] = interface
                    self.data[key] = {"type": nap_type.__name__, "interface": interface}
                    self._data_info[key] = nap_type.__name__

                # Spike trains
                if len(seg.spiketrains) == 1:
                    st = seg.spiketrains[0]
                    name = st.name if st.name else "spikes"
                    key = f"{block_prefix}Ts: {name}"

                    interface = NeoSignalInterface(
                        st, block, time_support, sig_num=0
                    )
                    self._interfaces[key] = interface
                    self.data[key] = {"type": "Ts", "interface": interface}
                    self._data_info[key] = "Ts"
                elif len(seg.spiketrains) > 1:
                    key = f"{block_prefix}TsGroup"

                    interface = NeoSignalInterface(
                        seg.spiketrains, block, time_support
                    )
                    self._interfaces[key] = interface
                    self.data[key] = {"type": "TsGroup", "interface": interface}
                    self._data_info[key] = "TsGroup"

                # Epochs
                for ep_idx, epoch in enumerate(seg.epochs):
                    name = epoch.name if hasattr(epoch, "name") and epoch.name else f"epoch{ep_idx}"
                    key = f"{block_prefix}IntervalSet {ep_idx}: {name}"

                    interface = NeoSignalInterface(
                        epoch, block, time_support, sig_num=ep_idx
                    )
                    self._interfaces[key] = interface
                    self.data[key] = {"type": "IntervalSet", "interface": interface}
                    self._data_info[key] = "IntervalSet"

                # Events
                for ev_idx, event in enumerate(seg.events):
                    name = event.name if hasattr(event, "name") and event.name else f"event{ev_idx}"
                    key = f"{block_prefix}Ts (event) {ev_idx}: {name}"

                    interface = NeoSignalInterface(
                        event, block, time_support, sig_num=ev_idx
                    )
                    self._interfaces[key] = interface
                    self.data[key] = {"type": "Ts", "interface": interface}
                    self._data_info[key] = "Ts"

    def __str__(self):
        """String representation showing available data."""
        title = self.name
        view = [[k, self._data_info[k]] for k in self.data.keys()]
        headers = ["Key", "Type"]

        if HAS_TABULATE:
            return title + "\n" + tabulate(view, headers=headers, tablefmt="mixed_outline")
        else:
            # Simple fallback without tabulate
            lines = [title, "-" * len(title)]
            for k, t in view:
                lines.append(f"  {k}: {t}")
            return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key: str):
        """Get data by key, loading if necessary.

        Parameters
        ----------
        key : str
            Key for the data item

        Returns
        -------
        pynapple object
            The requested data (Ts, Tsd, TsdFrame, TsGroup, IntervalSet, etc.)
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found. Available keys: {list(self.data.keys())}")

        item = self.data[key]

        # If already loaded, return it
        if not isinstance(item, dict):
            return item

        # Load via interface
        interface = item["interface"]
        loaded_data = interface.load()

        # Cache the loaded data
        self.data[key] = loaded_data

        return loaded_data

    def keys(self):
        """Return available data keys."""
        return list(self.data.keys())

    def items(self):
        """Return key-value pairs (loads data on access)."""
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        """Return all values (loads all data)."""
        return [self[k] for k in self.keys()]

    def get_time_support(self) -> nap.IntervalSet:
        """Get the time support from the first interface.

        Returns
        -------
        IntervalSet
            Time support covering all segments
        """
        if self._interfaces:
            return list(self._interfaces.values())[0].time_support
        return nap.IntervalSet(start=0, end=0)

    def close(self):
        """Close the underlying Neo reader if it supports closing."""
        if hasattr(self._reader, "close"):
            self._reader.close()


# =============================================================================
# Legacy Interface (for backward compatibility)
# =============================================================================


class NEOSignalInterface(NeoSignalInterface):
    """Legacy alias for NeoSignalInterface."""
    pass


class NEOExperimentInterface:
    """Legacy interface for Neo experiments.

    .. deprecated::
        Use :class:`NeoReader` instead.
    """

    def __init__(self, reader, lazy=False):
        warnings.warn(
            "NEOExperimentInterface is deprecated. Use NeoReader instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._reader = reader
        self._lazy = lazy
        self.experiment = self._collect_time_series_info()

    def _collect_time_series_info(self):
        blocks = self._reader.read(lazy=self._lazy)

        experiments = {}
        for i, block in enumerate(blocks):
            name = f"block {i}"
            if block.name:
                name += ": " + block.name
            experiments[name] = {}

            starts, ends = np.empty(len(block.segments)), np.empty(len(block.segments))
            for trial_num, segment in enumerate(block.segments):
                starts[trial_num] = segment.t_start.rescale("s").magnitude
                ends[trial_num] = segment.t_stop.rescale("s").magnitude

            iset = nap.IntervalSet(starts, ends)

            for trial_num, segment in enumerate(block.segments):
                # Analog signals
                for signal_num, signal in enumerate(segment.analogsignals):
                    if signal.name:
                        signame = f" {signal_num}: " + signal.name
                    else:
                        signame = f" {signal_num}"
                    signal_interface = NeoSignalInterface(
                        signal, block, iset, sig_num=signal_num
                    )
                    signame = signal_interface.nap_type.__name__ + signame
                    experiments[name][signame] = signal_interface

                # Spike trains
                if len(segment.spiketrains) == 1:
                    signal = segment.spiketrains[0]
                    signal_interface = NeoSignalInterface(
                        signal, block, iset, sig_num=0
                    )
                    signame = f"Ts" + ": " + signal.name if signal.name else "Ts"
                    experiments[name][signame] = signal_interface
                else:
                    signame = f"TsGroup"
                    experiments[name][signame] = NeoSignalInterface(
                        segment.spiketrains, block, iset
                    )

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


def load_file(path: Union[str, Path], lazy: bool = True) -> NeoReader:
    """Load a neural recording file using Neo.

    This function automatically detects the file format and uses the
    appropriate Neo IO to load the data.

    Parameters
    ----------
    path : str or Path
        Path to the recording file
    lazy : bool, default True
        Whether to use lazy loading (recommended for large files)

    Returns
    -------
    NeoReader
        Interface to the loaded data

    Examples
    --------
    >>> import pynapple as nap
    >>> data = nap.io.neo.load_file("recording.plx")
    >>> print(data)
    recording
    +---------------------+----------+
    | Key                 | Type     |
    +=====================+==========+
    | TsGroup             | TsGroup  |
    | Tsd 0: LFP          | Tsd      |
    +---------------------+----------+

    >>> spikes = data["TsGroup"]

    See Also
    --------
    NeoReader : Class for Neo file interface

    Notes
    -----
    Supported formats depend on your Neo installation. Common formats include:
    - Plexon (.plx, .pl2)
    - Blackrock (.nev, .ns*)
    - Spike2 (.smr)
    - Neuralynx (.ncs, .nse, .ntt)
    - OpenEphys
    - Intan (.rhd, .rhs)
    - And many more (see Neo documentation)
    """
    return NeoReader(path, lazy=lazy)


# Legacy alias
def load_experiment(path: Union[str, Path], lazy: bool = True) -> NEOExperimentInterface:
    """Load a neural recording experiment.

    .. deprecated::
        Use :func:`load_file` instead.

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
    import pathlib

    path = pathlib.Path(path)
    reader = neo.io.get_io(path)

    return NEOExperimentInterface(reader, lazy=lazy)


#
# # =============================================================================
# # Conversion functions: Pynapple -> Neo
# # =============================================================================
#
#
# def to_neo_analogsignal(
#     tsd: Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor],
#     units: str = "dimensionless",
#     **kwargs,
# ) -> "neo.AnalogSignal":
#     """Convert a pynapple Tsd/TsdFrame/TsdTensor to a Neo AnalogSignal.
#
#     Parameters
#     ----------
#     tsd : Tsd, TsdFrame, or TsdTensor
#         Pynapple time series object
#     units : str, default "dimensionless"
#         Units for the signal (e.g., "mV", "uV")
#     **kwargs
#         Additional arguments passed to neo.AnalogSignal
#
#     Returns
#     -------
#     neo.AnalogSignal
#         Neo analog signal object
#     """
#     _check_neo_installed()
#     import quantities as pq
#
#     times = tsd.times()
#     data = tsd.values
#
#     # Ensure 2D for AnalogSignal
#     if data.ndim == 1:
#         data = data.reshape(-1, 1)
#
#     # Calculate sampling rate from timestamps
#     if len(times) > 1:
#         dt = np.median(np.diff(times))
#         sampling_rate = 1.0 / dt
#     else:
#         sampling_rate = 1.0  # Default if only one sample
#
#     signal = neo.AnalogSignal(
#         data,
#         units=units,
#         sampling_rate=sampling_rate * pq.Hz,
#         t_start=times[0] * pq.s,
#         **kwargs,
#     )
#
#     return signal
#
#
# def to_neo_spiketrain(
#     ts: nap.Ts,
#     t_stop: Optional[float] = None,
#     units: str = "s",
#     **kwargs,
# ) -> "neo.SpikeTrain":
#     """Convert a pynapple Ts to a Neo SpikeTrain.
#
#     Parameters
#     ----------
#     ts : Ts
#         Pynapple Ts object
#     t_stop : float, optional
#         Stop time for the spike train. If None, uses the end of time_support
#     units : str, default "s"
#         Time units
#     **kwargs
#         Additional arguments passed to neo.SpikeTrain
#
#     Returns
#     -------
#     neo.SpikeTrain
#         Neo spike train object
#     """
#     _check_neo_installed()
#     import quantities as pq
#
#     times = ts.times()
#
#     if t_stop is None:
#         t_stop = float(ts.time_support.end[-1])
#
#     t_start = float(ts.time_support.start[0]) if len(times) == 0 else min(times[0], float(ts.time_support.start[0]))
#
#     spiketrain = neo.SpikeTrain(
#         times,
#         units=units,
#         t_start=t_start * pq.s,
#         t_stop=t_stop * pq.s,
#         **kwargs,
#     )
#
#     return spiketrain
#
#
# def to_neo_epoch(
#     iset: nap.IntervalSet,
#     labels: Optional[np.ndarray] = None,
#     **kwargs,
# ) -> "neo.Epoch":
#     """Convert a pynapple IntervalSet to a Neo Epoch.
#
#     Parameters
#     ----------
#     iset : IntervalSet
#         Pynapple IntervalSet
#     labels : array-like, optional
#         Labels for each epoch. If None, uses integers.
#     **kwargs
#         Additional arguments passed to neo.Epoch
#
#     Returns
#     -------
#     neo.Epoch
#         Neo epoch object
#     """
#     _check_neo_installed()
#     import quantities as pq
#
#     starts = iset.start
#     ends = iset.end
#     durations = ends - starts
#
#     if labels is None:
#         # Check if there's a 'label' column in metadata
#         if hasattr(iset, "label"):
#             labels = iset.label
#         else:
#             labels = np.arange(len(starts)).astype(str)
#
#     epoch = neo.Epoch(
#         times=starts * pq.s,
#         durations=durations * pq.s,
#         labels=labels,
#         **kwargs,
#     )
#
#     return epoch
#
#
# def to_neo_event(
#     ts: nap.Ts,
#     labels: Optional[np.ndarray] = None,
#     **kwargs,
# ) -> "neo.Event":
#     """Convert a pynapple Ts to a Neo Event.
#
#     Parameters
#     ----------
#     ts : Ts
#         Pynapple Ts object
#     labels : array-like, optional
#         Labels for each event. If None, uses integers.
#     **kwargs
#         Additional arguments passed to neo.Event
#
#     Returns
#     -------
#     neo.Event
#         Neo event object
#     """
#     _check_neo_installed()
#     import quantities as pq
#
#     times = ts.times()
#
#     if labels is None:
#         labels = np.arange(len(times)).astype(str)
#
#     event = neo.Event(
#         times=times * pq.s,
#         labels=labels,
#         **kwargs,
#     )
#
#     return event
