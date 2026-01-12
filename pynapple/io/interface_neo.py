"""
Pynapple interface for Neo (neural electrophysiology objects).

Neo is a Python package for working with electrophysiology data in Python,
supporting many file formats through a unified API.

The interface behaves like a dictionary.

For more information on Neo, see: https://neo.readthedocs.io/

Neo to Pynapple Object Conversion
---------------------------------
The following Neo objects are converted to their pynapple equivalents:

- 'neo.AnalogSignal' -> 'Tsd', `TsdFrame`, or `TsdTensor` (depending on shape) [lazy-loaded]
- neo.IrregularlySampledSignal -> Tsd, TsdFrame, or TsdTensor (depending on shape) [lazy-loaded]
- neo.SpikeTrain -> Ts
- neo.SpikeTrain (list) -> TsGroup
- neo.SpikeTrainList -> TsGroup
- neo.Epoch -> IntervalSet
- neo.Event -> Ts

Note: All data types support lazy loading. Data is only loaded when accessed
via __getitem__ (e.g., data["TsGroup"]).
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


def _make_intervalset_from_epoch_multiseg(
    block, ep_idx: int, time_support: Optional[nap.IntervalSet] = None
) -> nap.IntervalSet:
    """Convert Neo Epochs from multiple segments to a pynapple IntervalSet.

    Parameters
    ----------
    block : neo.Block
        The Neo block containing the segments
    ep_idx : int
        Index of the epoch in each segment
    time_support : IntervalSet, optional
        Time support for the IntervalSet

    Returns
    -------
    IntervalSet
        Pynapple IntervalSet
    """
    all_starts = []
    all_ends = []
    all_labels = []

    for seg in block.segments:
        if ep_idx >= len(seg.epochs):
            continue

        epoch = seg.epochs[ep_idx]
        if hasattr(epoch, "load"):
            epoch = epoch.load()

        times = epoch.times.rescale("s").magnitude
        durations = epoch.durations.rescale("s").magnitude

        all_starts.extend(times)
        all_ends.extend(times + durations)

        if hasattr(epoch, "labels") and len(epoch.labels) > 0:
            all_labels.extend(epoch.labels)

    if len(all_starts) == 0:
        return nap.IntervalSet(start=[], end=[])

    metadata = {}
    if all_labels:
        metadata["label"] = np.array(all_labels)

    return nap.IntervalSet(
        start=np.array(all_starts),
        end=np.array(all_ends),
        metadata=metadata if metadata else None,
    )


def _make_ts_from_event_multiseg(
    block, ev_idx: int, time_support: Optional[nap.IntervalSet] = None
) -> nap.Ts:
    """Convert Neo Events from multiple segments to a pynapple Ts.

    Parameters
    ----------
    block : neo.Block
        The Neo block containing the segments
    ev_idx : int
        Index of the event in each segment
    time_support : IntervalSet, optional
        Time support

    Returns
    -------
    Ts
        Pynapple Ts object
    """
    all_times = []

    for seg in block.segments:
        if ev_idx >= len(seg.events):
            continue

        event = seg.events[ev_idx]
        if hasattr(event, "load"):
            event = event.load()

        times = event.times.rescale("s").magnitude
        all_times.extend(times)

    return nap.Ts(t=np.array(all_times), time_support=time_support)


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


def _make_ts_from_spiketrain_multiseg(
    block, unit_idx: int, time_support: Optional[nap.IntervalSet] = None
) -> nap.Ts:
    """Convert a Neo SpikeTrain from multiple segments to a pynapple Ts.

    Parameters
    ----------
    block : neo.Block
        The Neo block containing the segments
    unit_idx : int
        Index of the spike train in each segment
    time_support : IntervalSet, optional
        Time support

    Returns
    -------
    Ts
        Pynapple Ts object
    """
    all_times = []

    for seg in block.segments:
        spiketrain = seg.spiketrains[unit_idx]
        if hasattr(spiketrain, "load"):
            spiketrain = spiketrain.load()

        times = spiketrain.times.rescale("s").magnitude
        all_times.append(times)

    spike_times = np.concatenate(all_times) if all_times else np.array([])
    return nap.Ts(t=spike_times, time_support=time_support)


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


def _make_tsgroup_from_spiketrains_multiseg(
    all_spiketrains: List[list],
    time_support: Optional[nap.IntervalSet] = None,
) -> nap.TsGroup:
    """Convert spike trains from multiple segments to a pynapple TsGroup.

    This function concatenates spike times across segments for each unit.

    Parameters
    ----------
    all_spiketrains : list of lists
        List of spike train lists, one per segment. Each inner list contains
        the spike trains for that segment.
    time_support : IntervalSet, optional
        Time support

    Returns
    -------
    TsGroup
        Pynapple TsGroup
    """
    if len(all_spiketrains) == 0:
        return nap.TsGroup({}, time_support=time_support)

    n_units = len(all_spiketrains[0])
    ts_dict = {}
    metadata = {}

    for unit_idx in range(n_units):
        all_times = []

        for seg_spiketrains in all_spiketrains:
            st = seg_spiketrains[unit_idx]
            if hasattr(st, "load"):
                st = st.load()

            times = st.times.rescale("s").magnitude
            all_times.append(times)

            # Collect metadata from first segment only
            if seg_spiketrains is all_spiketrains[0]:
                for key, value in _extract_annotations(st).items():
                    if key not in metadata:
                        metadata[key] = []
                    metadata[key].append(value)

        spike_times = np.concatenate(all_times) if all_times else np.array([])
        ts_dict[unit_idx] = nap.Ts(t=spike_times, time_support=time_support)

    # Convert metadata lists to arrays
    meta_arrays = {}
    for key, values in metadata.items():
        try:
            meta_arrays[key] = np.array(values)
        except (ValueError, TypeError):
            # Skip metadata that can't be converted to array
            pass

    return nap.TsGroup(ts_dict, time_support=time_support, **meta_arrays)

def _make_tsd_from_interface(interface) -> Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor]:
    """Convert a NeoSignalInterface to a pynapple Tsd/TsdFrame/TsdTensor.

    Parameters
    ----------
    interface : NeoSignalInterface
        The NeoSignalInterface object

    Returns
    -------
    Tsd, TsdFrame, or TsdTensor
        Appropriate pynapple time series object
    """
    nap_type = interface.nap_type

    # return nap_type(t=times, d=data, time_support=interface.time_support, load_array=False)
    return nap_type(t=interface.times, d=interface, load_array=False)


# =============================================================================
# Signal Interface for lazy loading
# =============================================================================


class NeoSignalInterface:
    """Interface for lazy-loading Neo analog signals into pynapple objects.

    This class provides lazy access to Neo analog signals (AnalogSignal,
    IrregularlySampledSignal), loading data only when requested. It acts as
    a pseudo memory-mapped array that can be passed directly to Tsd, TsdFrame,
    or TsdTensor initialization with `load_array=False`.

    The interface is array-like (has shape, dtype, ndim, supports indexing
    and iteration) so it can be used as a drop-in replacement for numpy arrays
    in pynapple time series constructors.

    Parameters
    ----------
    signal : neo signal object
        A Neo analog signal (AnalogSignal or IrregularlySampledSignal)
    block : neo.Block
        The parent block containing the signal
    time_support : IntervalSet
        Time support for the data
    sig_num : int, optional
        Index of the signal within the segment

    Attributes
    ----------
    nap_type : type
        The pynapple type this signal will be converted to (Tsd, TsdFrame, or TsdTensor)
    is_analog : bool
        Whether this is a regularly sampled analog signal
    dt : float
        Sampling interval (for analog signals)
    shape : tuple
        Shape of the data (total samples across all segments, channels, ...)
    dtype : numpy.dtype
        Data type of the signal
    ndim : int
        Number of dimensions
    times : numpy.ndarray
        Pre-loaded timestamps for all segments (in seconds)
    start_time : float
        Start time
    end_time : float
        End time

    Examples
    --------
    >>> interface = NeoSignalInterface(signal, block, time_support, sig_num=0)
    >>> # Use as array-like for lazy loading
    >>> tsd = nap.Tsd(t=interface.times, d=interface, load_array=False)
    >>> # Data is only loaded when accessed
    >>> chunk = tsd[0:1000]  # Loads only first 1000 samples
    """

    def __init__(self, signal, block, time_support=None, sig_num=0):
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
        else:
            raise TypeError(f"Signal type {type(signal)} not recognized.")

        # Store dtype from signal
        self.dtype = signal.dtype

        # Build segment info and compute total shape across all segments
        self._segment_offsets = []  # Cumulative sample counts per segment
        self._segment_n_samples = []  # Number of samples per segment
        self._times_list = []  # Pre-load timestamps per segment (small memory footprint)

        total_samples = 0
        for seg in block.segments:
            if self.is_analog:
                seg_signal = seg.analogsignals[sig_num]
            else:
                seg_signal = seg.irregularlysampledsignals[sig_num]

            n_samples = seg_signal.shape[0]
            self._segment_offsets.append(total_samples)
            self._segment_n_samples.append(n_samples)
            total_samples += n_samples

            # Pre-load timestamps (much smaller than data)
            if hasattr(seg_signal, "times"):
                self._times_list.append(seg_signal.times.rescale("s").magnitude)
            else:
                self._times_list.append(
                    np.linspace(
                        _rescale_to_seconds(seg_signal.t_start),
                        _rescale_to_seconds(seg_signal.t_stop),
                        n_samples,
                        endpoint=False,
                    )
                )

        self._segment_offsets = np.array(self._segment_offsets)
        self._segment_n_samples = np.array(self._segment_n_samples)

        # Concatenate all timestamps
        if self._times_list:
            self._times = np.concatenate(self._times_list)
        else:
            self._times = np.array([])

        # Compute total shape (first dimension is total samples)
        if len(signal.shape) == 1:
            self.shape = (int(total_samples),)
        else:
            self.shape = (int(total_samples),) + signal.shape[1:]

        # Store timing info
        if self.is_analog:
            self.dt = (1 / signal.sampling_rate).rescale("s").magnitude

        self.start_time = _rescale_to_seconds(signal.t_start)
        self.end_time = _rescale_to_seconds(signal.t_stop)

    def __repr__(self):
        return f"<NeoSignalInterface: {self.nap_type.__name__}, shape={self.shape}, dtype={self.dtype}>"

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def times(self):
        """Pre-loaded timestamps for all segments (in seconds)."""
        return self._times

    def __len__(self):
        """Return the number of samples (first dimension of shape)."""
        return self.shape[0]

    def __iter__(self):
        """Iterate over the first axis, loading data lazily."""
        for i in range(len(self)):
            yield self[i]

    def _find_segment_for_index(self, idx):
        """Find which segment contains the given global index.

        Returns
        -------
        seg_idx : int
            Index of the segment
        local_idx : int
            Index within that segment
        """
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for size {len(self)}")

        # Find segment using binary search on offsets
        seg_idx = np.searchsorted(self._segment_offsets, idx, side='right') - 1
        local_idx = idx - self._segment_offsets[seg_idx]
        return seg_idx, local_idx

    def _load_data_range(self, start_idx, stop_idx, step=1):
        """Load data for a range of global indices.

        Parameters
        ----------
        start_idx : int
            Start index (inclusive)
        stop_idx : int
            Stop index (exclusive)
        step : int
            Step size

        Returns
        -------
        numpy.ndarray
            The loaded data
        """
        if start_idx >= stop_idx:
            # Return empty array with correct shape
            if len(self.shape) == 1:
                return np.array([], dtype=self.dtype)
            else:
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)

        data_chunks = []

        for seg_idx, seg in enumerate(self._block.segments):
            seg_start = self._segment_offsets[seg_idx]
            seg_end = seg_start + self._segment_n_samples[seg_idx]

            # Check if this segment overlaps with requested range
            if stop_idx <= seg_start or start_idx >= seg_end:
                continue

            # Calculate local indices within this segment
            local_start = max(0, start_idx - seg_start)
            local_stop = min(self._segment_n_samples[seg_idx], stop_idx - seg_start)

            # Load data from this segment
            if self.is_analog:
                signal = seg.analogsignals[self._sig_num]
            else:
                signal = seg.irregularlysampledsignals[self._sig_num]

            # Try to load with indexing, fall back to time slicing
            try:
                if hasattr(signal, 'load'):
                    loaded = signal.load()
                    chunk = loaded[local_start:local_stop].magnitude
                else:
                    chunk = signal[local_start:local_stop].magnitude
            except (MemoryError, AttributeError):
                # Fall back to time slicing
                t_start = self._times_list[seg_idx][local_start]
                t_stop = self._times_list[seg_idx][min(local_stop, len(self._times_list[seg_idx]) - 1)]
                chunk = signal.time_slice(t_start, t_stop).magnitude

            data_chunks.append(chunk)

        if not data_chunks:
            if len(self.shape) == 1:
                return np.array([], dtype=self.dtype)
            else:
                return np.empty((0,) + self.shape[1:], dtype=self.dtype)

        result = np.concatenate(data_chunks, axis=0)

        # Apply step if needed
        if step != 1:
            result = result[::step]

        return result

    def __getitem__(self, item):
        """Get data by index, loading lazily from Neo signals.

        Supports integer indexing, slicing, and tuple indexing for
        multi-dimensional access.

        Parameters
        ----------
        item : int, slice, or tuple
            Index specification

        Returns
        -------
        numpy.ndarray or scalar
            The requested data
        """
        # Handle integer indexing
        if isinstance(item, (int, np.integer)):
            return self._load_data_range(item, item + 1)[0]

        # Handle slice indexing
        if isinstance(item, slice):
            start = item.start if item.start is not None else 0
            stop = item.stop if item.stop is not None else len(self)
            step = item.step if item.step is not None else 1

            # Handle negative indices
            if start < 0:
                start = len(self) + start
            if stop < 0:
                stop = len(self) + stop

            return self._load_data_range(start, stop, step)

        # Handle tuple indexing (e.g., interface[0:100, 0] for specific channel)
        if isinstance(item, tuple):
            # First index is for time dimension
            time_idx = item[0]
            rest = item[1:]

            # Get data for time dimension
            data = self[time_idx]

            # Apply remaining indices
            if rest:
                data = data[(slice(None),) + rest] if isinstance(time_idx, slice) else data[rest]

            return data

        # Handle numpy array or list indexing
        if isinstance(item, (np.ndarray, list)):
            indices = np.asarray(item)
            if indices.dtype == bool:
                # Boolean indexing
                indices = np.where(indices)[0]

            # Load each index and stack
            result = np.stack([self[int(i)] for i in indices])
            return result

        raise TypeError(f"Invalid index type: {type(item)}")



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

                # Analog signals - deferred loading via NeoSignalInterface
                for sig_idx, signal in enumerate(seg.analogsignals):
                    nap_type = _get_signal_type(signal)
                    name = signal.name if signal.name else f"signal{sig_idx}"
                    key = f"{block_prefix}{nap_type.__name__} {sig_idx}: {name}"

                    self.data[key] = {
                        "type": nap_type.__name__,
                        "loader": "analogsignal",
                        "block": block,
                        "sig_num": sig_idx,
                        "time_support": time_support,
                    }
                    self._data_info[key] = nap_type.__name__

                # Irregularly sampled signals
                for sig_idx, signal in enumerate(seg.irregularlysampledsignals):
                    nap_type = _get_signal_type(signal)
                    name = signal.name if signal.name else f"irregular{sig_idx}"
                    key = f"{block_prefix}{nap_type.__name__} (irregular) {sig_idx}: {name}"

                    self.data[key] = {
                        "type": nap_type.__name__,
                        "loader": "irregularsignal",
                        "block": block,
                        "sig_num": sig_idx,
                        "time_support": time_support,
                    }
                    self._data_info[key] = nap_type.__name__

                # Spike trains - deferred loading
                if len(seg.spiketrains) == 1:
                    st = seg.spiketrains[0]
                    name = st.name if st.name else "spikes"
                    key = f"{block_prefix}Ts: {name}"

                    # Store info for deferred loading
                    self.data[key] = {
                        "type": "Ts",
                        "loader": "spiketrain",
                        "block": block,
                        "unit_idx": 0,
                        "time_support": time_support,
                    }
                    self._data_info[key] = "Ts"
                elif len(seg.spiketrains) > 1:
                    key = f"{block_prefix}TsGroup"

                    # Store info for deferred loading
                    self.data[key] = {
                        "type": "TsGroup",
                        "loader": "tsgroup",
                        "block": block,
                        "time_support": time_support,
                    }
                    self._data_info[key] = "TsGroup"

                # Epochs - deferred loading
                for ep_idx, epoch in enumerate(seg.epochs):
                    name = epoch.name if hasattr(epoch, "name") and epoch.name else f"epoch{ep_idx}"
                    key = f"{block_prefix}IntervalSet {ep_idx}: {name}"

                    # Store info for deferred loading
                    self.data[key] = {
                        "type": "IntervalSet",
                        "loader": "epoch",
                        "block": block,
                        "ep_idx": ep_idx,
                        "time_support": time_support,
                    }
                    self._data_info[key] = "IntervalSet"

                # Events - deferred loading
                for ev_idx, event in enumerate(seg.events):
                    name = event.name if hasattr(event, "name") and event.name else f"event{ev_idx}"
                    key = f"{block_prefix}Ts (event) {ev_idx}: {name}"

                    # Store info for deferred loading
                    self.data[key] = {
                        "type": "Ts",
                        "loader": "event",
                        "block": block,
                        "ev_idx": ev_idx,
                        "time_support": time_support,
                    }
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
            The requested data (Ts, Tsd, TsdFrame, TsdTensor, TsGroup, IntervalSet)
        """
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found. Available keys: {list(self.data.keys())}")

        item = self.data[key]

        # If already loaded, return it
        if not isinstance(item, dict):
            return item

        # Load based on loader type
        loader = item.get("loader")

        if loader == "spiketrain":
            # Load single spike train from all segments
            loaded_data = _make_ts_from_spiketrain_multiseg(
                item["block"],
                unit_idx=item["unit_idx"],
                time_support=item["time_support"],
            )
        elif loader == "tsgroup":
            # Load TsGroup from all segments
            all_spiketrains = [s.spiketrains for s in item["block"].segments]
            loaded_data = _make_tsgroup_from_spiketrains_multiseg(
                all_spiketrains,
                time_support=item["time_support"],
            )
        elif loader == "epoch":
            # Load IntervalSet from all segments
            loaded_data = _make_intervalset_from_epoch_multiseg(
                item["block"],
                ep_idx=item["ep_idx"],
                time_support=item["time_support"],
            )
        elif loader == "event":
            # Load Ts (event) from all segments
            loaded_data = _make_ts_from_event_multiseg(
                item["block"],
                ev_idx=item["ev_idx"],
                time_support=item["time_support"],
            )
        elif loader in ["analogsignal", "irregularsignal"]:
            # Load via NeoSignalInterface (deferred loading)
            interface = NeoSignalInterface(
                signal=item["block"].segments[0].analogsignals[item["sig_num"]]
                if loader == "analogsignal"
                else item["block"].segments[0].irregularlysampledsignals[item["sig_num"]],
                block=item["block"],
                time_support=item["time_support"],
                sig_num=item["sig_num"],
            )
            loaded_data = _make_tsd_from_interface(interface)

        else:
            raise ValueError(f"Unknown loader type for key '{key}'")

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









#
#
# # =============================================================================
# # Legacy Interface (for backward compatibility)
# # =============================================================================
#
#
# class NEOSignalInterface(NeoSignalInterface):
#     """Legacy alias for NeoSignalInterface."""
#     pass
#
#
# class NEOExperimentInterface:
#     """Legacy interface for Neo experiments.
#
#     .. deprecated::
#         Use :class:`NeoReader` instead.
#     """
#
#     def __init__(self, reader, lazy=False):
#         warnings.warn(
#             "NEOExperimentInterface is deprecated. Use NeoReader instead.",
#             DeprecationWarning,
#             stacklevel=2,
#         )
#         self._reader = reader
#         self._lazy = lazy
#         self.experiment = self._collect_time_series_info()
#
#     def _collect_time_series_info(self):
#         blocks = self._reader.read(lazy=self._lazy)
#
#         experiments = {}
#         for i, block in enumerate(blocks):
#             name = f"block {i}"
#             if block.name:
#                 name += ": " + block.name
#             experiments[name] = {}
#
#             starts, ends = np.empty(len(block.segments)), np.empty(len(block.segments))
#             for trial_num, segment in enumerate(block.segments):
#                 starts[trial_num] = segment.t_start.rescale("s").magnitude
#                 ends[trial_num] = segment.t_stop.rescale("s").magnitude
#
#             iset = nap.IntervalSet(starts, ends)
#
#             for trial_num, segment in enumerate(block.segments):
#                 # Analog signals
#                 for signal_num, signal in enumerate(segment.analogsignals):
#                     if signal.name:
#                         signame = f" {signal_num}: " + signal.name
#                     else:
#                         signame = f" {signal_num}"
#                     signal_interface = NeoSignalInterface(
#                         signal, block, iset, sig_num=signal_num
#                     )
#                     signame = signal_interface.nap_type.__name__ + signame
#                     experiments[name][signame] = signal_interface
#
#                 # Spike trains
#                 if len(segment.spiketrains) == 1:
#                     signal = segment.spiketrains[0]
#                     signal_interface = NeoSignalInterface(
#                         signal, block, iset, sig_num=0
#                     )
#                     signame = f"Ts" + ": " + signal.name if signal.name else "Ts"
#                     experiments[name][signame] = signal_interface
#                 else:
#                     signame = f"TsGroup"
#                     experiments[name][signame] = NeoSignalInterface(
#                         segment.spiketrains, block, iset
#                     )
#
#         return experiments
#
#     def __getitem__(self, item):
#         if isinstance(item, str):
#             return self.experiment[item]
#         else:
#             res = self.experiment
#             for it in item:
#                 res = res[it]
#             return res
#
#     def keys(self):
#         return [(k, k2) for k in self.experiment.keys() for k2 in self.experiment[k]]
#
#
# def load_file(path: Union[str, Path], lazy: bool = True) -> NeoReader:
#     """Load a neural recording file using Neo.
#
#     This function automatically detects the file format and uses the
#     appropriate Neo IO to load the data.
#
#     Parameters
#     ----------
#     path : str or Path
#         Path to the recording file
#     lazy : bool, default True
#         Whether to use lazy loading (recommended for large files)
#
#     Returns
#     -------
#     NeoReader
#         Interface to the loaded data
#
#     Examples
#     --------
#     >>> import pynapple as nap
#     >>> data = nap.io.neo.load_file("recording.plx")
#     >>> print(data)
#     recording
#     +---------------------+----------+
#     | Key                 | Type     |
#     +=====================+==========+
#     | TsGroup             | TsGroup  |
#     | Tsd 0: LFP          | Tsd      |
#     +---------------------+----------+
#
#     >>> spikes = data["TsGroup"]
#
#     See Also
#     --------
#     NeoReader : Class for Neo file interface
#
#     Notes
#     -----
#     Supported formats depend on your Neo installation. Common formats include:
#     - Plexon (.plx, .pl2)
#     - Blackrock (.nev, .ns*)
#     - Spike2 (.smr)
#     - Neuralynx (.ncs, .nse, .ntt)
#     - OpenEphys
#     - Intan (.rhd, .rhs)
#     - And many more (see Neo documentation)
#     """
#     return NeoReader(path, lazy=lazy)
#
#
# # Legacy alias
# def load_experiment(path: Union[str, Path], lazy: bool = True) -> NEOExperimentInterface:
#     """Load a neural recording experiment.
#
#     .. deprecated::
#         Use :func:`load_file` instead.
#
#     Parameters
#     ----------
#     path : str or Path
#         Path to the recording file
#     lazy : bool, default True
#         Whether to lazy load the data
#
#     Returns
#     -------
#     NEOExperimentInterface
#     """
#     import pathlib
#
#     path = pathlib.Path(path)
#     reader = neo.io.get_io(path)
#
#     return NEOExperimentInterface(reader, lazy=lazy)
#
#
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
