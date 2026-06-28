"""Reader for Neurosuite/Neuroscope formatted electrophysiology data.

Handles:
- XML metadata parsing (channel count, sampling rates, channel groups)
- Binary signal files (.dat, .eeg, .lfp) via memory mapping
- Spike sorting results (.clu.N / .res.N pairs)
- Event files (.evt) with automatic pairing of start/stop events into IntervalSets
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import core as nap

_START_RE = re.compile(r"\b(start|Start|START)\b")
_STOP_WORDS = ["stop", "Stop", "STOP", "end", "End", "END"]


def _pair_start_stop(starts, stops):
    """Pair each start with the first stop after it and before the next start."""
    paired_s, paired_e = [], []
    j = 0
    for i, t0 in enumerate(starts):
        t_next = starts[i + 1] if i + 1 < len(starts) else np.inf
        while j < len(stops) and stops[j] <= t0:
            j += 1
        if j < len(stops) and stops[j] < t_next:
            paired_s.append(t0)
            paired_e.append(stops[j])
            j += 1
    return np.array(paired_s), np.array(paired_e)


def _build_events(raw_events):
    """Convert {category: [timestamps_s]} into pynapple objects.

    Keys containing 'start'/'Start'/'START' are paired with their matching
    stop key to form an IntervalSet.  Any remaining timestamps whose events
    fall inside those intervals are attached as IntervalSet metadata.
    Unpaired keys are returned as Ts objects.
    """
    result = {}
    used = set()

    for key in list(raw_events.keys()):
        if key in used or not _START_RE.search(key):
            continue

        # Find corresponding stop key
        start_word = _START_RE.search(key).group(0)
        stop_key = None
        for stop_word in _STOP_WORDS:
            candidate = key.replace(start_word, stop_word)
            if candidate != key and candidate in raw_events:
                stop_key = candidate
                break
        if stop_key is None:
            continue

        starts = np.sort(np.array(raw_events[key]))
        stops = np.sort(np.array(raw_events[stop_key]))
        paired_s, paired_e = _pair_start_stop(starts, stops)
        if len(paired_s) == 0:
            continue

        iset = nap.IntervalSet(start=paired_s, end=paired_e)
        used.update([key, stop_key])

        # Attach remaining timestamps as metadata (one value per interval)
        metadata = {}
        for other_key, other_times in raw_events.items():
            if other_key in used:
                continue
            other_times = np.sort(np.array(other_times))
            col = []
            for i in range(len(iset)):
                mask = (other_times >= iset.start[i]) & (other_times <= iset.end[i])
                matches = other_times[mask]
                col.append(matches[0] if len(matches) > 0 else np.nan)
            metadata[other_key] = col
            used.add(other_key)

        if metadata:
            iset.set_info(**metadata)

        # Build a clean key: strip the start word and collapse extra spaces
        iset_key = re.sub(r"\s+", " ", _START_RE.sub("", key)).strip() or key
        result[iset_key] = iset

    # Remaining keys become plain Ts objects
    for key, times in raw_events.items():
        if key not in used:
            result[key] = nap.Ts(t=np.sort(np.array(times)))

    return result


def _text(parent, tag):
    """Safely extract text from a subelement."""
    if parent is None:
        return None
    el = parent.find(tag)
    if el is None or el.text is None:
        return None
    return el.text.strip()


def parse_neuroscope_xml(xml_path):
    """
    Parse a Neuroscope / ndManager XML file.

    Parameters
    ----------
    xml_path : str or Path

    Returns
    -------
    dict
        Structured representation of the XML.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    out = {}

    # --------------------
    # General info
    # --------------------
    gi = root.find("generalInfo")
    out["general_info"] = {
        "date": _text(gi, "date"),
        "experimenters": _text(gi, "experimenters"),
        "description": _text(gi, "description"),
        "notes": _text(gi, "notes"),
    }

    # --------------------
    # Acquisition system
    # --------------------
    acq = root.find("acquisitionSystem")
    out["acquisition"] = {
        "n_bits": int(_text(acq, "nBits")),
        "n_channels": int(_text(acq, "nChannels")),
        "sampling_rate": float(_text(acq, "samplingRate")),
        "voltage_range": float(_text(acq, "voltageRange")),
        "amplification": float(_text(acq, "amplification")),
        "offset": float(_text(acq, "offset")),
    }

    # --------------------
    # LFP
    # --------------------
    lfp = root.find("fieldPotentials")
    out["lfp"] = {"sampling_rate": float(_text(lfp, "lfpSamplingRate"))}

    # --------------------
    # Anatomical channel groups
    # --------------------
    anatomy_groups = []
    for group in root.find("anatomicalDescription/channelGroups").findall("group"):
        channels = []
        for ch in group.findall("channel"):
            channels.append(
                {"id": int(ch.text), "skip": bool(int(ch.attrib.get("skip", "0")))}
            )
        anatomy_groups.append(channels)

    out["anatomy"] = {"channel_groups": anatomy_groups}

    # --------------------
    # Spike detection groups
    # --------------------
    spike_groups = []
    for group in root.find("spikeDetection/channelGroups").findall("group"):
        channels = [int(ch.text) for ch in group.find("channels").findall("channel")]
        spike_groups.append(
            {
                "channels": channels,
                "n_samples": int(_text(group, "nSamples")),
                "n_features": int(_text(group, "nFeatures")),
                "peak_sample_index": int(_text(group, "peakSampleIndex")),
            }
        )

    out["spike_detection"] = {"channel_groups": spike_groups}

    # --------------------
    # Units (clusters)
    # --------------------
    units = []
    for unit in root.find("units").findall("unit"):
        units.append(
            {
                "group": int(_text(unit, "group")),
                "cluster": int(_text(unit, "cluster")),
                "structure": _text(unit, "structure"),
                "type": _text(unit, "type"),
                "isolation_distance": _text(unit, "isolationDistance"),
                "quality": _text(unit, "quality"),
                "notes": _text(unit, "notes"),
            }
        )

    out["units"] = units

    # --------------------
    # Neuroscope channel display info
    # --------------------
    channels = {}

    neuroscope_channels = root.find("neuroscope/channels")
    for cc in neuroscope_channels.findall("channelColors"):
        ch = int(_text(cc, "channel"))
        channels.setdefault(ch, {})
        channels[ch]["color"] = _text(cc, "color")
        channels[ch]["anatomy_color"] = _text(cc, "anatomyColor")
        channels[ch]["spike_color"] = _text(cc, "spikeColor")

    for co in neuroscope_channels.findall("channelOffset"):
        ch = int(_text(co, "channel"))
        channels.setdefault(ch, {})
        channels[ch]["offset"] = float(_text(co, "defaultOffset"))

    out["neuroscope"] = {"channels": channels}

    return out


@dataclass(frozen=True)
class _NeuroSuiteMetadata:
    xml_info: dict
    n_channels: int
    channel_order: np.ndarray
    skip: np.ndarray
    groups: np.ndarray
    binary_metadata: dict


@dataclass(frozen=True)
class _DiscoveredFiles:
    dat_files: list[Path]
    lfp_files: list[Path]
    spike_groups: dict
    evt_files: list[Path]


class NeuroSuiteIO:
    """Load data from a Neurosuite/Neuroscope session directory.

    A Neuroscope session directory typically contains:

    - ``basename.xml`` – session metadata (channels, sampling rates, groups)
    - ``basename.dat`` – raw wideband recording (int16 binary)
    - ``basename.eeg`` or ``basename.lfp`` – LFP signal (int16 binary)
    - ``basename.clu.N`` / ``basename.res.N`` – spike cluster IDs and times
      for electrode group *N*
    - ``basename.evt`` – event timestamps with category labels

    Parameters
    ----------
    path : str or Path
        Path to the session directory, or to a file inside it.

    Public properties
    ------------------
    session_dir : Path
        Resolved session directory.
    basename : str
        Session basename (used as prefix for all data files).
    n_channels : int
        Total number of channels (from XML).
    fs_dat : float
        Sampling rate of the .dat file (Hz).
    fs_lfp : float
        Sampling rate of the .eeg/.lfp file (Hz).
    channel_order : ndarray
        Channel ordering as read from the XML.
    groups : ndarray
        Channel group index per channel.
    dat_files : list of Path
        Detected .dat files.
    lfp_files : list of Path
        Detected .eeg / .lfp files.
    spike_groups : dict
        Mapping of shank number (str) to ``(clu_path, res_path)`` tuples.
    evt_files : list of Path
        Detected .evt files.
    """

    def __init__(self, path):
        path = Path(path)

        if path.is_dir():
            self.session_dir = path
            self.basename = path.name
        else:
            self.session_dir = path.parent
            self.basename = path.stem.split(".")[0]

        # Parse XML
        xml_info = parse_neuroscope_xml(self.session_dir / f"{self.basename}.xml")
        n_channels = xml_info["acquisition"]["n_channels"]
        channel_order = np.zeros(n_channels, dtype="int")
        skip = np.zeros(n_channels, dtype=bool)
        groups = np.zeros(n_channels, dtype="int")
        count = 0
        for group_idx, channels in enumerate(xml_info["anatomy"]["channel_groups"]):
            for ch in channels:
                ch_id = ch["id"]
                channel_order[count] = ch_id
                skip[ch_id] = ch["skip"]
                groups[ch_id] = group_idx
                count += 1
        binary_metadata = {
            "anatomy": np.argsort(channel_order),  # Different for pynaviz
            "skip": skip,
            "group": groups,
        }
        self._metadata = _NeuroSuiteMetadata(
            xml_info=xml_info,
            n_channels=n_channels,
            channel_order=channel_order,
            skip=skip,
            groups=groups,
            binary_metadata=binary_metadata,
        )

        # Discover files
        self._files = _DiscoveredFiles(
            dat_files=self._find_files(".dat"),
            lfp_files=self._find_files(".eeg") or self._find_files(".lfp"),
            spike_groups=self._find_spike_groups(),
            evt_files=self._find_evt_files(),
        )

    @property
    def xml_info(self):
        return self._metadata.xml_info

    @property
    def n_channels(self):
        return self._metadata.n_channels

    @property
    def channel_order(self):
        return self._metadata.channel_order

    @property
    def skip(self):
        return self._metadata.skip

    @property
    def groups(self):
        return self._metadata.groups

    @property
    def binary_metadata(self):
        return self._metadata.binary_metadata

    @property
    def fs_dat(self):
        return self._metadata.xml_info["acquisition"]["sampling_rate"]

    @property
    def fs_lfp(self):
        return self._metadata.xml_info["lfp"]["sampling_rate"]

    @property
    def dat_files(self):
        return self._files.dat_files

    @property
    def lfp_files(self):
        return self._files.lfp_files

    @property
    def spike_groups(self):
        return self._files.spike_groups

    @property
    def evt_files(self):
        return self._files.evt_files

    # ------------------------------------------------------------------
    # File discovery helpers
    # ------------------------------------------------------------------

    def _find_files(self, extension):
        """Return sorted list of files matching *basename.ext*, falling
        back to ``*.ext`` if none found."""
        files = sorted(self.session_dir.glob(f"{self.basename}{extension}"))
        if not files:
            files = sorted(self.session_dir.glob(f"*{extension}"))
        return files

    def _find_evt_files(self):
        """Return sorted list of *.evt, *.evt.*, and *.*.evt files for this session."""
        exact = sorted(self.session_dir.glob(f"{self.basename}.evt"))
        pre_tagged = sorted(self.session_dir.glob(f"{self.basename}.*.evt"))
        post_tagged = sorted(self.session_dir.glob(f"{self.basename}.evt.*"))
        if not exact and not pre_tagged and not post_tagged:
            exact = sorted(self.session_dir.glob("*.evt"))
            pre_tagged = sorted(self.session_dir.glob("*.*.evt"))
            post_tagged = sorted(self.session_dir.glob("*.evt.*"))
        return exact + pre_tagged + post_tagged

    def _find_spike_groups(self):
        """Find paired .clu.N / .res.N files and return a dict keyed by
        shank number (str)."""
        clu_files = sorted(self.session_dir.glob(f"{self.basename}.clu.*"))
        if not clu_files:
            clu_files = sorted(self.session_dir.glob("*.clu.*"))

        groups = {}
        for clu_file in clu_files:
            shank = clu_file.suffix.lstrip(".")
            res_file = clu_file.with_suffix("").with_suffix(f".res.{shank}")
            if res_file.exists():
                groups[shank] = (clu_file, res_file)
        return groups

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_binary(self, filepath, frequency=None):
        """Load a binary signal file (.dat, .eeg, .lfp) as a TsdFrame.

        The file is memory-mapped so that data is loaded on demand.

        Parameters
        ----------
        filepath : str or Path
            Path to the binary file.
        frequency : float, optional
            Sampling rate in Hz.  If *None*, inferred from the XML
            based on the file extension (.dat -> fs_dat, .eeg/.lfp -> fs_lfp).

        Returns
        -------
        TsdFrame
            Memory-mapped time series with one column per channel.
        """
        filepath = Path(filepath)

        if frequency is None:
            if filepath.suffix.lower() == ".dat":
                frequency = self.fs_dat
            else:
                frequency = self.fs_lfp

        n_channels = self.n_channels
        bytes_size = 2  # int16

        file_size = filepath.stat().st_size
        n_samples = int(file_size / n_channels / bytes_size)

        fp = np.memmap(filepath, np.int16, "r", shape=(n_samples, n_channels))
        timestep = np.arange(0, n_samples) / frequency

        return nap.TsdFrame(
            t=timestep, d=fp, load_array=False, metadata=self.binary_metadata
        )

    def load_spikes(self, shank):
        """Load spike data for a given shank from .clu / .res files.

        Clusters 0 (noise) and 1 (multi-unit artefact) are excluded by
        convention.

        Parameters
        ----------
        shank : str
            The shank identifier (e.g. ``"1"``).

        Returns
        -------
        TsGroup
            One :class:`~pynapple.Ts` per sorted unit, with a ``"group"``
            metadata column containing the original cluster ID.
        """
        if shank not in self.spike_groups:
            raise ValueError(
                f"Shank '{shank}' not found. "
                f"Available shanks: {list(self.spike_groups.keys())}"
            )

        clu_file, res_file = self.spike_groups[shank]

        res = np.loadtxt(str(res_file), dtype=np.int64)
        clu = np.loadtxt(str(clu_file), dtype=np.int64)

        # First line of clu is the total number of clusters
        clu = clu[1:]

        spike_times_s = res / self.fs_dat

        unit_ids = np.unique(clu)
        unit_ids = unit_ids[unit_ids > 1]  # skip 0 (noise) and 1 (MUA)

        ts_dict = {}
        for i, uid in enumerate(unit_ids):
            mask = clu == uid
            ts_dict[i] = nap.Ts(t=spike_times_s[mask])

        # Time support should be inferred from the recording duration
        if self.dat_files and len(self.dat_files) and self.dat_files[0].exists():
            dat_file = self.dat_files[0]
            file_size = dat_file.stat().st_size
            n_samples = int(file_size / self.n_channels / 2)  # int16
            duration_s = n_samples / self.fs_dat
            time_support = nap.IntervalSet(start=0, end=duration_s, time_units="s")
        elif self.lfp_files and len(self.lfp_files) and self.lfp_files[0].exists():
            lfp_file = self.lfp_files[0]
            file_size = lfp_file.stat().st_size
            n_samples = int(file_size / self.n_channels / 2)  # int16
            duration_s = n_samples / self.fs_lfp
            time_support = nap.IntervalSet(start=0, end=duration_s, time_units="s")
        else:
            time_support = None

        group = np.ones(len(unit_ids), dtype=int) * int(shank)

        return nap.TsGroup(
            ts_dict, time_support=time_support, metadata={"group": group}
        )

    def load_events(self, filepath=None):
        """Load a NeuroSuite .evt file and return a dict of pynapple objects.

        Keys containing 'start'/'Start'/'START' are paired with their
        matching stop key to produce an :class:`~pynapple.IntervalSet`.
        Timestamps from other categories that fall within those intervals
        are attached as IntervalSet metadata.  Any unpaired categories are
        returned as :class:`~pynapple.Ts` objects.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to the .evt file.  If *None*, the first discovered .evt file
            in the session directory is used.

        Returns
        -------
        dict
            Mapping of category/base-name to IntervalSet or Ts.
        """
        if filepath is None:
            if not self.evt_files:
                raise FileNotFoundError(f"No .evt files found in {self.session_dir}")
            filepath = self.evt_files[0]

        filepath = Path(filepath)
        raw_events = defaultdict(list)

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                raw_events[parts[1]].append(float(parts[0]) / 1000.0)  # ms -> s

        return _build_events(raw_events)
