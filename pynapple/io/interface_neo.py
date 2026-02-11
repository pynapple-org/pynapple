"""Pynapple interface to Neo for reading electrophysiology files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import UserDict
from pathlib import Path
from typing import Any

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
from .interface_neurosuite import NeuroSuiteIO


def _parse_openephys_electrode_positions(settings_xml_path):
    """Parse electrode (x, y) positions from an Open Ephys settings.xml.

    The positions are stored in ``<ELECTRODE_XPOS>`` / ``<ELECTRODE_YPOS>``
    elements inside ``<NP_PROBE>`` nodes.  Each attribute is named ``CH{n}``
    where *n* is the 0-based electrode index.

    Parameters
    ----------
    settings_xml_path : str or Path
        Path to the ``settings.xml`` file.

    Returns
    -------
    dict[str, dict[int, np.ndarray]]
        Mapping of probe serial number to a dict with keys ``"x"`` and
        ``"y"``, each an ndarray of shape ``(n_electrodes,)`` ordered by
        electrode index.  Returns an empty dict if no positions are found.
    """
    settings_xml_path = Path(settings_xml_path)
    if not settings_xml_path.exists():
        return {}

    tree = ET.parse(settings_xml_path)
    root = tree.getroot()

    probes = {}
    for np_probe in root.iter("NP_PROBE"):
        xpos_el = np_probe.find("ELECTRODE_XPOS")
        ypos_el = np_probe.find("ELECTRODE_YPOS")
        if xpos_el is None or ypos_el is None:
            continue

        x_dict = {int(k[2:]): float(v) for k, v in xpos_el.attrib.items()}
        y_dict = {int(k[2:]): float(v) for k, v in ypos_el.attrib.items()}

        n = max(max(x_dict.keys()), max(y_dict.keys())) + 1
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        for ch, val in x_dict.items():
            x[ch] = val
        for ch, val in y_dict.items():
            y[ch] = val

        serial = np_probe.get("probe_serial_number", "unknown")
        probes[serial] = {"x": x, "y": y}

    return probes


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


def _extract_annotations(obj) -> dict[str, Any]:
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


def _extract_array_annotations(obj) -> dict[str, np.ndarray]:
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


def _make_intervalset_from_epoch(
    epoch, time_support: nap.IntervalSet | None = None
) -> nap.IntervalSet:
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
    block, ep_idx: int, time_support: nap.IntervalSet | None = None
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
    block, ev_idx: int, time_support: nap.IntervalSet | None = None
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


def _make_ts_from_event(
    event, time_support: nap.IntervalSet | None = None
) -> nap.Ts:
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
    spiketrain, time_support: nap.IntervalSet | None = None
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
    block, unit_idx: int, time_support: nap.IntervalSet | None = None
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
    spiketrains: list | SpikeTrainList,
    time_support: nap.IntervalSet | None = None,
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

    return nap.TsGroup(ts_dict, time_support=time_support, metadata=meta_arrays)


def _make_tsgroup_from_spiketrains_multiseg(
    all_spiketrains: list[list],
    time_support: nap.IntervalSet | None = None,
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

    return nap.TsGroup(ts_dict, time_support=time_support, metadata=meta_arrays)


def _make_tsd_from_interface(interface) -> nap.Tsd | nap.TsdFrame | nap.TsdTensor:
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
        self._times_list = (
            []
        )  # Pre-load timestamps per segment (small memory footprint)

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
        seg_idx = np.searchsorted(self._segment_offsets, idx, side="right") - 1
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
                if hasattr(signal, "load"):
                    loaded = signal.load()
                    chunk = loaded[local_start:local_stop].magnitude
                else:
                    chunk = signal[local_start:local_stop].magnitude
            except (MemoryError, AttributeError):
                # Fall back to time slicing
                t_start = self._times_list[seg_idx][local_start]
                t_stop = self._times_list[seg_idx][
                    min(local_stop, len(self._times_list[seg_idx]) - 1)
                ]
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
                data = (
                    data[(slice(None),) + rest]
                    if isinstance(time_idx, slice)
                    else data[rest]
                )

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


class EphysReader(UserDict):
    """Read Neo-compatible electrophysiology files into pynapple objects.

    `Neo <https://neo.readthedocs.io/>`_ is a Python package for working with
    electrophysiology data, supporting many file formats through a unified API.

    This class provides a dictionary-like interface to Neo files, with
    lazy-loading support. It automatically detects the appropriate IO
    based on the file extension.

    Neo to Pynapple Object Conversion
    ---------------------------------
    The following Neo objects are converted to their pynapple equivalents:

    .. list-table::
       :header-rows: 1
       :widths: 40 40 20

       * - Neo Object
         - Pynapple Object
         - Notes
       * - `AnalogSignal <https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.AnalogSignal>`_
         - :py:class:`~pynapple.Tsd`, :py:class:`~pynapple.TsdFrame`, or :py:class:`~pynapple.TsdTensor`
         - Depends on shape; lazy-loaded
       * - `IrregularlySampledSignal <https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.IrregularlySampledSignal>`_
         - :py:class:`~pynapple.Tsd`, :py:class:`~pynapple.TsdFrame`, or :py:class:`~pynapple.TsdTensor`
         - Depends on shape; lazy-loaded
       * - `SpikeTrain <https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.SpikeTrain>`_
         - :py:class:`~pynapple.Ts`
         - Single unit
       * - `SpikeTrain <https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.SpikeTrain>`_ (list)
         - :py:class:`~pynapple.TsGroup`
         - Multiple units
       * - `Epoch <https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.Epoch>`_
         - :py:class:`~pynapple.IntervalSet`
         -
       * - `Event <https://neo.readthedocs.io/en/latest/api_reference.html#neo.core.Event>`_
         - :py:class:`~pynapple.Ts`
         -

    Note: All data types support lazy loading. Data is only loaded when accessed
    via ``__getitem__`` (e.g., ``data["TsGroup"]``).

    Parameters
    ----------
    path : str or Path
        Path to the file to load or directory containing the files.
    lazy : bool, default True
        Whether to use lazy loading
    format : str, type, or None, default None
        Specify the Neo IO format to use. Can be:

        - ``None``: Automatically detect the format using ``neo.io.get_io``
        - ``str``: Name of the IO class (e.g., ``"PlexonIO"``, ``"Plexon"``, ``"plexon"``)
        - ``type``: A class from ``neo.io.iolist`` (e.g., ``neo.io.PlexonIO``)

        When a string is provided, it is matched case-insensitively against IO class names.
        The "IO" suffix is optional.

    Examples
    --------
    >>> import pynapple as nap
    >>> data = nap.EphysReader("my_file.plx")
    >>> print(data)
    my_file
    ┍━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━┑
    │ Key                 │ Type     │
    ┝━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━┥
    │ TsGroup             │ TsGroup  │
    │ Tsd 0: LFP          │ Tsd      │
    ┕━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━┙

    >>> spikes = data["TsGroup"]
    >>> lfp = data["Tsd 0: LFP"]

    To explicitly specify the file format:

    >>> data = nap.EphysReader("my_file.plx", format="PlexonIO")
    """

    def __init__(
        self,
        path: str | Path,
        lazy: bool = True,
        format: str | type | None = None,
    ):
        _check_neo_installed()

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Files not found: {path}")

        self.name = self.path.stem

        self.data = {}
        # special case if format is NeuroscopeIO to check for binary files and set up deferred loading
        if self._is_neuroscope(self.path, format):
            self._collect_neuroscope(self.path)
        else:
            self._init_neo_reader(format)
            self._collect_data(lazy=lazy)  # This will read the Neo blocks and populate self.data with metadata and interfaces for lazy loading

        UserDict.__init__(self, self.data)

    def _is_neuroscope(self, path, format=None):
        """Check if the format specifies NeuroscopeIO.

        Parameters
        ----------
        path : Path
            The path to the file or directory being loaded.
        format : str, type, or None
            The format argument passed to EphysReader.

        Returns
        -------
        bool
            True if the format is NeuroscopeIO.
        """
        if isinstance(format, str) and format.lower().replace("io", "") == "neuroscope":
            return True
        if isinstance(format, type) and format.__name__.lower().replace("io", "") == "neuroscope":
            return True
        # Additional heuristic: check for presence of Neuroscope binary files
        binary_extensions = [".dat", ".lfp", ".eeg"]
        if path.is_dir():
            for ext in binary_extensions:
                if any(path.glob(f"*{ext}")):
                    # Check for an xml file that would indicate Neuroscope format
                    if any(path.glob("*.xml")):
                        return True
        elif path.suffix.lower() in binary_extensions:
            return True
        return False

    def _init_neo_reader(self, format):
        """Initialize the Neo reader

        Parameters
        ----------
        format : str, type, or None
            The format argument passed to EphysReader.

        """
        if format is None:
            self._reader = neo.io.get_io(str(self.path))
        elif isinstance(format, str):
            io_class = None
            format_lower = format.lower()
            for io in neo.io.iolist:
                io_name = io.__name__.lower()
                io_name_no_suffix = io_name.replace("io", "")
                if io_name == format_lower or io_name_no_suffix == format_lower:
                    io_class = io
                    break
            if io_class is None:
                available = [io.__name__ for io in neo.io.iolist]
                raise ValueError(
                    f"Format '{format}' not found in neo.io.iolist. "
                    f"Available formats: {available}"
                )
            self._reader = io_class(str(self.path))
        elif isinstance(format, type):
            if format not in neo.io.iolist:
                available = [io.__name__ for io in neo.io.iolist]
                raise ValueError(
                    f"Class {format.__name__} is not in neo.io.iolist. "
                    f"Available formats: {available}"
                )
            self._reader = format(str(self.path))
        else:
            raise TypeError(
                f"format must be None, a string, or a class from neo.io.iolist, "
                f"not {type(format).__name__}"
            )

    def _collect_neuroscope(self, path):
        """Scan a Neuroscope session and set up deferred loading entries.

        Uses :class:`~pynapple.io.interface_neurosuite.NeuroSuiteIO` to
        discover files and parse XML metadata.  Populates ``self.data``
        with entries for .dat, .eeg/.lfp, and .clu/.res files.

        Parameters
        ----------
        path : Path
            The path to the session directory (or a file inside it).
        """
        ns = NeuroSuiteIO(path)

        # --- .dat file (raw wideband) ---
        if ns.dat_files:
            dat_file = ns.dat_files[0]
            self.data[dat_file.name] = {
                "type": "TsdFrame",
                "loader": lambda f=dat_file: ns.load_binary(f),
            }

        # --- .eeg / .lfp file (LFP) ---
        if ns.lfp_files:
            lfp_file = ns.lfp_files[0]
            self.data[lfp_file.name] = {
                "type": "TsdFrame",
                "loader": lambda f=lfp_file: ns.load_binary(f),
            }

        # --- .clu.N / .res.N pairs (spike sorting) ---
        for shank in ns.spike_groups:
            clu_file, res_file = ns.spike_groups[shank]
            self.data[clu_file.name] = {
                "type": "TsGroup",
                "loader": lambda s=shank: ns.load_spikes(s),
            }

        self._ns = ns  # Store the NeuroSuiteIO instance for use in loaders

    def _get_electrode_positions(self):
        """Extract electrode positions from the reader's settings file.

        Currently supports Open Ephys binary format, where positions are
        stored in ``settings.xml`` under ``<ELECTRODE_XPOS>`` /
        ``<ELECTRODE_YPOS>`` elements.

        Returns
        -------
        dict
            Probe serial -> ``{"x": ndarray, "y": ndarray}`` mapping,
            or empty dict if not available.
        """
        if not hasattr(self._reader, "folder_structure"):
            return {}
        fs = self._reader.folder_structure
        # Walk to the first experiment's settings_file
        for node in fs.values():
            for exp in node.get("experiments", {}).values():
                settings_file = exp.get("settings_file")
                if settings_file is not None:
                    return _parse_openephys_electrode_positions(settings_file)
        return {}

    def _make_mmap_entry(self, proxy, block_idx, nap_type, time_support,
                         metadata=None):
        """Build a deferred-loading dict that memory-maps a raw binary file.

        Uses Neo's buffer description API to locate the file on disk and
        the column slice that corresponds to this particular stream.

        Parameters
        ----------
        proxy : AnalogSignalProxy
            The Neo proxy for this signal.
        block_idx : int
            Block index in the reader.
        nap_type : type
            Pynapple type (Tsd, TsdFrame, TsdTensor).
        time_support : IntervalSet
            Time support for the recording.
        metadata : dict or None
            Optional per-column metadata to attach to the resulting object.

        Returns
        -------
        dict
            Entry for ``self.data`` with a callable ``"loader"``.
        """
        reader = self._reader
        stream_idx = proxy._stream_index
        stream_info = reader.header["signal_streams"][stream_idx]
        stream_id = stream_info["id"]
        buffer_id = stream_info["buffer_id"]

        bd = reader.get_analogsignal_buffer_description(
            block_index=block_idx, seg_index=0, buffer_id=buffer_id,
        )
        file_path = bd["file_path"]
        dtype = np.dtype(bd["dtype"])
        buf_shape = tuple(bd["shape"])
        col_slice = reader._stream_buffer_slice[stream_id]

        # Pre-compute timestamps from the proxy metadata
        t_start = _rescale_to_seconds(proxy.t_start)
        sampling_rate = float(proxy.sampling_rate.rescale("Hz").magnitude)
        n_samples = proxy.shape[0]

        def _loader(
            _fp=file_path, _dt=dtype, _bs=buf_shape,
            _cs=col_slice, _t0=t_start, _sr=sampling_rate,
            _ns=n_samples, _nt=nap_type, _meta=metadata,
        ):
            fp = np.memmap(_fp, _dt, "r", shape=_bs)
            data = fp[:, _cs]
            timestamps = _t0 + np.arange(_ns) / _sr
            kwargs = {}
            if _meta is not None:
                kwargs["metadata"] = _meta
            return _nt(t=timestamps, d=data, load_array=False, **kwargs)

        return {
            "type": nap_type.__name__,
            "loader": _loader,
        }

    def _collect_data(self, lazy=True):
        """Collect all data from Neo blocks into the dictionary."""
        # Read blocks
        self._blocks = self._reader.read(lazy=lazy)

        # Build data dictionary
        for block_idx, block in enumerate(self._blocks):
            block_prefix = "" if len(self._blocks) == 1 else f"block{block_idx}/"

            # Build time support from segments
            starts = np.array(
                [_rescale_to_seconds(seg.t_start) for seg in block.segments]
            )
            ends = np.array([_rescale_to_seconds(seg.t_stop) for seg in block.segments])
            time_support = nap.IntervalSet(starts, ends)

            # Process first segment to get signal info
            # (assuming consistent structure across segments)
            if len(block.segments) > 0:
                seg = block.segments[0]

                # Analog signals - deferred loading via NeoSignalInterface
                # or memory-mapped when the reader exposes raw buffer info
                has_buffer_api = (
                    hasattr(self._reader, "has_buffer_description_api")
                    and self._reader.has_buffer_description_api()
                )

                # Parse electrode positions from settings.xml (OpenEphys)
                electrode_positions = self._get_electrode_positions()

                for sig_idx, signal in enumerate(seg.analogsignals):
                    nap_type = _get_signal_type(signal)
                    name = signal.name if signal.name else f"signal{sig_idx}"
                    key = f"{block_prefix}{sig_idx}:{name}"

                    if has_buffer_api and isinstance(signal, AnalogSignalProxy):
                        # Build per-column metadata from electrode positions
                        metadata = None
                        if electrode_positions:
                            n_ch = signal.shape[1] if len(signal.shape) > 1 else 1
                            # Use the first (and usually only) probe
                            probe = next(iter(electrode_positions.values()))
                            n_electrodes = np.sum(~np.isnan(probe["x"]))
                            if n_ch == n_electrodes:
                                mask = ~np.isnan(probe["x"])
                                metadata = {
                                    "x": probe["x"][mask],
                                    "y": probe["y"][mask],
                                }

                        self.data[key] = self._make_mmap_entry(
                            signal, block_idx, nap_type, time_support,
                            metadata=metadata,
                        )
                    else:
                        self.data[key] = {
                            "type": nap_type.__name__,
                            "loader": "analogsignal",
                            "block": block,
                            "sig_num": sig_idx,
                            "time_support": time_support,
                        }

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
                elif len(seg.spiketrains) > 1:
                    key = f"{block_prefix}TsGroup"

                    # Store info for deferred loading
                    self.data[key] = {
                        "type": "TsGroup",
                        "loader": "tsgroup",
                        "block": block,
                        "time_support": time_support,
                    }
                # Epochs - deferred loading
                for ep_idx, epoch in enumerate(seg.epochs):
                    name = (
                        epoch.name
                        if hasattr(epoch, "name") and epoch.name
                        else f"epoch{ep_idx}"
                    )
                    key = f"{block_prefix}IntervalSet {ep_idx}: {name}"

                    # Store info for deferred loading
                    self.data[key] = {
                        "type": "IntervalSet",
                        "loader": "epoch",
                        "block": block,
                        "ep_idx": ep_idx,
                        "time_support": time_support,
                    }
                # Events - deferred loading
                for ev_idx, event in enumerate(seg.events):
                    name = (
                        event.name
                        if hasattr(event, "name") and event.name
                        else f"event{ev_idx}"
                    )
                    key = f"{block_prefix}Ts (event) {ev_idx}: {name}"

                    # Store info for deferred loading
                    self.data[key] = {
                        "type": "Ts",
                        "loader": "event",
                        "block": block,
                        "ev_idx": ev_idx,
                        "time_support": time_support,
                    }

    def __str__(self):
        """String representation showing available data."""
        title = self.name
        view = []
        for k, v in self.data.items():
            if isinstance(v, dict):
                view.append([k, v.get("type", "")])
            else:
                view.append([k, type(v).__name__])
        headers = ["Key", "Type"]

        if HAS_TABULATE:
            return (
                title + "\n" + tabulate(view, headers=headers, tablefmt="mixed_outline")
            )
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
            raise KeyError(
                f"Key '{key}' not found. Available keys: {list(self.data.keys())}"
            )

        item = self.data[key]

        # If already loaded, return it
        if not isinstance(item, dict):
            return item

        # Load based on loader type
        loader = item.get("loader")

        if callable(loader):
            loaded_data = loader()
        elif loader == "spiketrain":
            loaded_data = _make_ts_from_spiketrain_multiseg(
                item["block"],
                unit_idx=item["unit_idx"],
                time_support=item["time_support"],
            )
        elif loader == "tsgroup":
            all_spiketrains = [s.spiketrains for s in item["block"].segments]
            loaded_data = _make_tsgroup_from_spiketrains_multiseg(
                all_spiketrains,
                time_support=item["time_support"],
            )
        elif loader == "epoch":
            loaded_data = _make_intervalset_from_epoch_multiseg(
                item["block"],
                ep_idx=item["ep_idx"],
                time_support=item["time_support"],
            )
        elif loader == "event":
            loaded_data = _make_ts_from_event_multiseg(
                item["block"],
                ev_idx=item["ev_idx"],
                time_support=item["time_support"],
            )
        elif loader in ["analogsignal", "irregularsignal"]:
            interface = NeoSignalInterface(
                signal=(
                    item["block"].segments[0].analogsignals[item["sig_num"]]
                    if loader == "analogsignal"
                    else item["block"]
                    .segments[0]
                    .irregularlysampledsignals[item["sig_num"]]
                ),
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

    def close(self):
        """Close the underlying Neo reader if it supports closing."""
        if hasattr(self._reader, "close"):
            self._reader.close()
        if hasattr(self, "_ns"):
            del self._ns  # Clean up NeuroSuiteIO instance