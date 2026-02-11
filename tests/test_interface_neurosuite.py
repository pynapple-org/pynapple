"""Tests for pynapple.io.interface_neurosuite module."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import pynapple as nap
from pynapple.io.interface_neurosuite import NeuroSuiteIO, parse_neuroscope_xml

# ---------------------------------------------------------------------------
# Helpers to generate synthetic Neuroscope session files
# ---------------------------------------------------------------------------

N_CHANNELS = 4
FS_DAT = 20000.0
FS_LFP = 1250.0
N_SAMPLES_DAT = 200
N_SAMPLES_LFP = int(N_SAMPLES_DAT * FS_LFP / FS_DAT)
BASENAME = "session"


def _make_xml(
    session_dir, basename=BASENAME, n_channels=N_CHANNELS, fs_dat=FS_DAT, fs_lfp=FS_LFP
):
    """Write a minimal Neuroscope-compatible XML file."""
    root = ET.Element("parameters")

    # General info
    gi = ET.SubElement(root, "generalInfo")
    ET.SubElement(gi, "date").text = "2024-01-01"
    ET.SubElement(gi, "experimenters").text = "tester"
    ET.SubElement(gi, "description").text = "test session"
    ET.SubElement(gi, "notes").text = "synthetic"

    # Acquisition system
    acq = ET.SubElement(root, "acquisitionSystem")
    ET.SubElement(acq, "nBits").text = "16"
    ET.SubElement(acq, "nChannels").text = str(n_channels)
    ET.SubElement(acq, "samplingRate").text = str(fs_dat)
    ET.SubElement(acq, "voltageRange").text = "20"
    ET.SubElement(acq, "amplification").text = "1000"
    ET.SubElement(acq, "offset").text = "0"

    # LFP
    fp = ET.SubElement(root, "fieldPotentials")
    ET.SubElement(fp, "lfpSamplingRate").text = str(fs_lfp)

    # Anatomical channel groups: two groups of 2 channels
    anat = ET.SubElement(root, "anatomicalDescription")
    cg = ET.SubElement(anat, "channelGroups")
    for grp_chans in [[0, 1], [2, 3]]:
        g = ET.SubElement(cg, "group")
        for ch_id in grp_chans:
            ch_el = ET.SubElement(g, "channel", skip="0")
            ch_el.text = str(ch_id)

    # Spike detection groups: one group per anatomical group
    sd = ET.SubElement(root, "spikeDetection")
    sdcg = ET.SubElement(sd, "channelGroups")
    for grp_chans in [[0, 1], [2, 3]]:
        g = ET.SubElement(sdcg, "group")
        channels = ET.SubElement(g, "channels")
        for ch_id in grp_chans:
            ET.SubElement(channels, "channel").text = str(ch_id)
        ET.SubElement(g, "nSamples").text = "32"
        ET.SubElement(g, "nFeatures").text = "3"
        ET.SubElement(g, "peakSampleIndex").text = "16"

    # Units: two sorted units
    units_el = ET.SubElement(root, "units")
    for grp, cluster in [(0, 2), (1, 2)]:
        u = ET.SubElement(units_el, "unit")
        ET.SubElement(u, "group").text = str(grp)
        ET.SubElement(u, "cluster").text = str(cluster)
        ET.SubElement(u, "structure").text = "CA1"
        ET.SubElement(u, "type").text = "pyramidal"
        ET.SubElement(u, "isolationDistance").text = "15.0"
        ET.SubElement(u, "quality").text = "good"
        ET.SubElement(u, "notes").text = ""

    # Neuroscope display info
    ns = ET.SubElement(root, "neuroscope")
    ns_ch = ET.SubElement(ns, "channels")
    for ch_id in range(n_channels):
        cc = ET.SubElement(ns_ch, "channelColors")
        ET.SubElement(cc, "channel").text = str(ch_id)
        ET.SubElement(cc, "color").text = "#0000ff"
        ET.SubElement(cc, "anatomyColor").text = "#00ff00"
        ET.SubElement(cc, "spikeColor").text = "#ff0000"
        co = ET.SubElement(ns_ch, "channelOffset")
        ET.SubElement(co, "channel").text = str(ch_id)
        ET.SubElement(co, "defaultOffset").text = "0"

    tree = ET.ElementTree(root)
    xml_path = session_dir / f"{basename}.xml"
    tree.write(str(xml_path), xml_declaration=True, encoding="utf-8")
    return xml_path


def _make_dat(
    session_dir, basename=BASENAME, n_channels=N_CHANNELS, n_samples=N_SAMPLES_DAT
):
    """Write a synthetic int16 .dat file."""
    rng = np.random.default_rng(42)
    data = rng.integers(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)
    dat_path = session_dir / f"{basename}.dat"
    data.tofile(str(dat_path))
    return dat_path, data


def _make_eeg(
    session_dir, basename=BASENAME, n_channels=N_CHANNELS, n_samples=N_SAMPLES_LFP
):
    """Write a synthetic int16 .eeg file."""
    rng = np.random.default_rng(43)
    data = rng.integers(-500, 500, size=(n_samples, n_channels), dtype=np.int16)
    eeg_path = session_dir / f"{basename}.eeg"
    data.tofile(str(eeg_path))
    return eeg_path, data


def _make_clu_res(
    session_dir, basename=BASENAME, shank="1", fs=FS_DAT, n_samples_dat=N_SAMPLES_DAT
):
    """Write synthetic .clu.N / .res.N files.

    Creates 50 spikes spread across clusters 0, 1, 2, 3.
    Clusters 0 and 1 should be skipped by convention (noise, MUA).
    Spike sample indices are within ``[0, n_samples_dat)``.
    """
    rng = np.random.default_rng(44 + int(shank))
    n_spikes = 50
    n_clusters = 4  # 0=noise, 1=MUA, 2=unit, 3=unit
    cluster_ids = rng.choice(n_clusters, size=n_spikes)
    sample_times = np.sort(rng.integers(0, n_samples_dat, size=n_spikes))

    clu_path = session_dir / f"{basename}.clu.{shank}"
    res_path = session_dir / f"{basename}.res.{shank}"

    np.savetxt(str(clu_path), np.concatenate([[n_clusters], cluster_ids]), fmt="%d")
    np.savetxt(str(res_path), sample_times, fmt="%d")

    return clu_path, res_path, cluster_ids, sample_times


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_dir(tmp_path):
    """Create a complete synthetic Neuroscope session directory."""
    sdir = tmp_path / BASENAME
    sdir.mkdir()
    _make_xml(sdir)
    _make_dat(sdir)
    _make_eeg(sdir)
    _make_clu_res(sdir, shank="1")
    _make_clu_res(sdir, shank="2")
    return sdir


@pytest.fixture
def ns(session_dir):
    """A NeuroSuiteIO instance built from the synthetic session."""
    return NeuroSuiteIO(session_dir)


# ---------------------------------------------------------------------------
# Tests: parse_neuroscope_xml
# ---------------------------------------------------------------------------


class TestParseNeuroscopeXml:
    def test_general_info(self, session_dir):
        info = parse_neuroscope_xml(session_dir / f"{BASENAME}.xml")
        assert info["general_info"]["date"] == "2024-01-01"
        assert info["general_info"]["experimenters"] == "tester"

    def test_acquisition(self, session_dir):
        info = parse_neuroscope_xml(session_dir / f"{BASENAME}.xml")
        acq = info["acquisition"]
        assert acq["n_bits"] == 16
        assert acq["n_channels"] == N_CHANNELS
        assert acq["sampling_rate"] == FS_DAT
        assert acq["amplification"] == 1000

    def test_lfp(self, session_dir):
        info = parse_neuroscope_xml(session_dir / f"{BASENAME}.xml")
        assert info["lfp"]["sampling_rate"] == FS_LFP

    def test_anatomy_groups(self, session_dir):
        info = parse_neuroscope_xml(session_dir / f"{BASENAME}.xml")
        groups = info["anatomy"]["channel_groups"]
        assert len(groups) == 2
        assert [ch["id"] for ch in groups[0]] == [0, 1]
        assert [ch["id"] for ch in groups[1]] == [2, 3]
        assert all(ch["skip"] is False for g in groups for ch in g)

    def test_spike_detection_groups(self, session_dir):
        info = parse_neuroscope_xml(session_dir / f"{BASENAME}.xml")
        sg = info["spike_detection"]["channel_groups"]
        assert len(sg) == 2
        assert sg[0]["channels"] == [0, 1]
        assert sg[0]["n_samples"] == 32
        assert sg[0]["n_features"] == 3
        assert sg[0]["peak_sample_index"] == 16

    def test_units(self, session_dir):
        info = parse_neuroscope_xml(session_dir / f"{BASENAME}.xml")
        units = info["units"]
        assert len(units) == 2
        assert units[0]["group"] == 0
        assert units[0]["cluster"] == 2
        assert units[0]["structure"] == "CA1"

    def test_neuroscope_display(self, session_dir):
        info = parse_neuroscope_xml(session_dir / f"{BASENAME}.xml")
        channels = info["neuroscope"]["channels"]
        assert len(channels) == N_CHANNELS
        assert channels[0]["color"] == "#0000ff"


# ---------------------------------------------------------------------------
# Tests: NeuroSuiteIO.__init__ and file discovery
# ---------------------------------------------------------------------------


class TestNeuroSuiteIOInit:
    def test_from_directory(self, ns, session_dir):
        assert ns.session_dir == session_dir
        assert ns.basename == BASENAME

    def test_from_file(self, session_dir):
        dat_path = session_dir / f"{BASENAME}.dat"
        ns = NeuroSuiteIO(dat_path)
        assert ns.session_dir == session_dir
        assert ns.basename == BASENAME

    def test_n_channels(self, ns):
        assert ns.n_channels == N_CHANNELS

    def test_sampling_rates(self, ns):
        assert ns.fs_dat == FS_DAT
        assert ns.fs_lfp == FS_LFP

    def test_channel_order(self, ns):
        assert len(ns.channel_order) == N_CHANNELS
        assert set(ns.channel_order) == {0, 1, 2, 3}

    def test_groups(self, ns):
        assert ns.groups[0] == 0
        assert ns.groups[1] == 0
        assert ns.groups[2] == 1
        assert ns.groups[3] == 1

    def test_binary_metadata(self, ns):
        meta = ns.binary_metadata
        assert "anatomy" in meta
        assert "skip" in meta
        assert "group" in meta
        assert len(meta["group"]) == N_CHANNELS

    def test_dat_files_found(self, ns, session_dir):
        assert len(ns.dat_files) == 1
        assert ns.dat_files[0] == session_dir / f"{BASENAME}.dat"

    def test_lfp_files_found(self, ns, session_dir):
        assert len(ns.lfp_files) == 1
        assert ns.lfp_files[0] == session_dir / f"{BASENAME}.eeg"

    def test_spike_groups_found(self, ns):
        assert "1" in ns.spike_groups
        assert "2" in ns.spike_groups

    def test_lfp_fallback_to_lfp_extension(self, tmp_path):
        """When no .eeg file exists, .lfp should be used instead."""
        sdir = tmp_path / BASENAME
        sdir.mkdir()
        _make_xml(sdir)
        _make_dat(sdir)
        # Write an .lfp file instead of .eeg
        rng = np.random.default_rng(99)
        data = rng.integers(-500, 500, size=(N_SAMPLES_LFP, N_CHANNELS), dtype=np.int16)
        lfp_path = sdir / f"{BASENAME}.lfp"
        data.tofile(str(lfp_path))

        ns = NeuroSuiteIO(sdir)
        assert len(ns.lfp_files) == 1
        assert ns.lfp_files[0].suffix == ".lfp"


# ---------------------------------------------------------------------------
# Tests: load_binary
# ---------------------------------------------------------------------------


class TestLoadBinary:
    def test_load_dat(self, ns, session_dir):
        dat_path = session_dir / f"{BASENAME}.dat"
        tsd = ns.load_binary(dat_path)
        assert isinstance(tsd, nap.TsdFrame)
        assert tsd.shape == (N_SAMPLES_DAT, N_CHANNELS)

    def test_dat_is_memmap(self, ns, session_dir):
        dat_path = session_dir / f"{BASENAME}.dat"
        tsd = ns.load_binary(dat_path)
        assert isinstance(tsd.d, np.memmap)

    def test_dat_frequency_inferred(self, ns, session_dir):
        dat_path = session_dir / f"{BASENAME}.dat"
        tsd = ns.load_binary(dat_path)
        expected_duration = (N_SAMPLES_DAT - 1) / FS_DAT
        np.testing.assert_allclose(tsd.times()[-1], expected_duration, rtol=1e-6)

    def test_load_eeg(self, ns, session_dir):
        eeg_path = session_dir / f"{BASENAME}.eeg"
        tsd = ns.load_binary(eeg_path)
        assert isinstance(tsd, nap.TsdFrame)
        assert isinstance(tsd.d, np.memmap)
        assert tsd.shape == (N_SAMPLES_LFP, N_CHANNELS)

    def test_eeg_frequency_inferred(self, ns, session_dir):
        eeg_path = session_dir / f"{BASENAME}.eeg"
        tsd = ns.load_binary(eeg_path)
        expected_duration = (N_SAMPLES_LFP - 1) / FS_LFP
        np.testing.assert_allclose(tsd.times()[-1], expected_duration, rtol=1e-6)

    def test_explicit_frequency(self, ns, session_dir):
        dat_path = session_dir / f"{BASENAME}.dat"
        custom_fs = 10000.0
        tsd = ns.load_binary(dat_path, frequency=custom_fs)
        expected_duration = (N_SAMPLES_DAT - 1) / custom_fs
        np.testing.assert_allclose(tsd.times()[-1], expected_duration, rtol=1e-6)

    def test_dat_values(self, ns, session_dir):
        """Values from the TsdFrame should match the original array."""
        _, orig = _make_dat(session_dir)  # overwrites with same seed
        dat_path = session_dir / f"{BASENAME}.dat"
        tsd = ns.load_binary(dat_path)
        np.testing.assert_array_equal(np.array(tsd), orig)

    def test_metadata_attached(self, ns, session_dir):
        dat_path = session_dir / f"{BASENAME}.dat"
        tsd = ns.load_binary(dat_path)
        assert "anatomy" in tsd.metadata.keys()
        assert "skip" in tsd.metadata.keys()
        assert "group" in tsd.metadata.keys()


# ---------------------------------------------------------------------------
# Tests: load_spikes
# ---------------------------------------------------------------------------


class TestLoadSpikes:
    def test_returns_tsgroup(self, ns):
        tsgroup = ns.load_spikes("1")
        assert isinstance(tsgroup, nap.TsGroup)

    def test_skips_noise_and_mua(self, ns, session_dir):
        """Clusters 0 (noise) and 1 (MUA) should be excluded."""
        _, _, cluster_ids, _ = _make_clu_res(session_dir, shank="1")
        tsgroup = ns.load_spikes("1")

        valid_ids = np.unique(cluster_ids)
        valid_ids = valid_ids[valid_ids > 1]
        assert len(tsgroup) == len(valid_ids)

    def test_spike_times_in_seconds(self, ns, session_dir):
        # Read back the files that the fixture wrote
        clu_path = session_dir / f"{BASENAME}.clu.1"
        res_path = session_dir / f"{BASENAME}.res.1"
        clu = np.loadtxt(str(clu_path), dtype=np.int64)
        cluster_ids = clu[1:]  # first line is cluster count
        sample_times = np.loadtxt(str(res_path), dtype=np.int64)

        tsgroup = ns.load_spikes("1")

        # Collect all spike times from the TsGroup
        all_times = np.sort(
            np.concatenate([tsgroup[i].times() for i in tsgroup.keys()])
        )
        # Compute expected times for clusters > 1
        mask = cluster_ids > 1
        expected = np.sort(sample_times[mask] / FS_DAT)
        np.testing.assert_allclose(all_times, expected, rtol=1e-10)

    def test_invalid_shank_raises(self, ns):
        with pytest.raises(ValueError, match="Shank '99' not found"):
            ns.load_spikes("99")

    def test_group_metadata(self, ns):
        tsgroup = ns.load_spikes("1")
        assert "group" in tsgroup.metadata.keys()

    def test_multiple_shanks(self, ns):
        ts1 = ns.load_spikes("1")
        ts2 = ns.load_spikes("2")
        assert isinstance(ts1, nap.TsGroup)
        assert isinstance(ts2, nap.TsGroup)
