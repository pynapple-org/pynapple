"""Reader for Neurosuite/Neuroscope formatted electrophysiology data.

Handles:
- XML metadata parsing (channel count, sampling rates, channel groups)
- Binary signal files (.dat, .eeg, .lfp) via memory mapping
- Spike sorting results (.clu.N / .res.N pairs)
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .. import core as nap


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
    out["lfp"] = {
        "sampling_rate": float(_text(lfp, "lfpSamplingRate"))
    }

    # --------------------
    # Anatomical channel groups
    # --------------------
    anatomy_groups = []
    for group in root.find("anatomicalDescription/channelGroups").findall("group"):
        channels = []
        for ch in group.findall("channel"):
            channels.append({
                "id": int(ch.text),
                "skip": bool(int(ch.attrib.get("skip", "0")))
            })
        anatomy_groups.append(channels)

    out["anatomy"] = {
        "channel_groups": anatomy_groups
    }

    # --------------------
    # Spike detection groups
    # --------------------
    spike_groups = []
    for group in root.find("spikeDetection/channelGroups").findall("group"):
        channels = [int(ch.text) for ch in group.find("channels").findall("channel")]
        spike_groups.append({
            "channels": channels,
            "n_samples": int(_text(group, "nSamples")),
            "n_features": int(_text(group, "nFeatures")),
            "peak_sample_index": int(_text(group, "peakSampleIndex")),
        })

    out["spike_detection"] = {
        "channel_groups": spike_groups
    }

    # --------------------
    # Units (clusters)
    # --------------------
    units = []
    for unit in root.find("units").findall("unit"):
        units.append({
            "group": int(_text(unit, "group")),
            "cluster": int(_text(unit, "cluster")),
            "structure": _text(unit, "structure"),
            "type": _text(unit, "type"),
            "isolation_distance": _text(unit, "isolationDistance"),
            "quality": _text(unit, "quality"),
            "notes": _text(unit, "notes"),
        })

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

    out["neuroscope"] = {
        "channels": channels
    }

    return out


class NeuroSuiteIO:
    """Load data from a Neurosuite/Neuroscope session directory.

    A Neuroscope session directory typically contains:

    - ``basename.xml`` – session metadata (channels, sampling rates, groups)
    - ``basename.dat`` – raw wideband recording (int16 binary)
    - ``basename.eeg`` or ``basename.lfp`` – LFP signal (int16 binary)
    - ``basename.clu.N`` / ``basename.res.N`` – spike cluster IDs and times
      for electrode group *N*

    Parameters
    ----------
    path : str or Path
        Path to the session directory, or to a file inside it.

    Attributes
    ----------
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
    channel_groups : dict
        Mapping of group index to list of channel numbers.
    dat_files : list of Path
        Detected .dat files.
    lfp_files : list of Path
        Detected .eeg / .lfp files.
    spike_groups : dict
        Mapping of shank number (str) to ``(clu_path, res_path)`` tuples.
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
        self.xml_info = parse_neuroscope_xml(self.session_dir / f"{self.basename}.xml")
        self.n_channels = self.xml_info["acquisition"]["n_channels"]
        self.channel_order = np.zeros(self.n_channels, dtype="int")
        self.skip = np.zeros(self.n_channels, dtype=bool)
        self.groups = np.zeros(self.n_channels, dtype="int")
        count = 0
        for group_idx, channels in enumerate(self.xml_info["anatomy"]["channel_groups"]):
            for ch in channels:
                ch_id = ch["id"]
                self.channel_order[count] = ch_id
                self.skip[count] = ch["skip"]
                self.groups[ch["id"]] = group_idx
                count += 1
        self.binary_metadata = {
            "anatomy": np.argsort(self.channel_order), # Different for pynaviz
            "skip": self.skip,
            "group": self.groups,
        }
        self.fs_dat = self.xml_info["acquisition"]["sampling_rate"]
        self.fs_lfp = self.xml_info["lfp"]["sampling_rate"]

        # Discover files
        self.dat_files = self._find_files(".dat")
        self.lfp_files = self._find_files(".eeg") or self._find_files(".lfp")
        self.spike_groups = self._find_spike_groups()

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

    def _find_spike_groups(self):
        """Find paired .clu.N / .res.N files and return a dict keyed by
        shank number (str)."""
        clu_files = sorted(
            self.session_dir.glob(f"{self.basename}.clu.*")
        )
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

        return nap.TsdFrame(t=timestep, d=fp, load_array=False, metadata=self.binary_metadata)

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

        return nap.TsGroup(ts_dict, time_support = time_support, metadata={"group": group})