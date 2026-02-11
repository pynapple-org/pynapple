"""Tests for pynapple.io.interface_neo module."""

import xml.etree.ElementTree as ET
from pathlib import Path

import neo
import numpy as np
import quantities as pq
import pytest

import pynapple as nap
from pynapple.io.interface_neo import (
    EphysReader,
    NeoSignalInterface,
    _extract_annotations,
    _extract_array_annotations,
    _get_signal_type,
    # _make_intervalset_from_epoch,
    _make_intervalset_from_epoch_multiseg,
    _make_ts_from_event,
    _make_ts_from_event_multiseg,
    _make_ts_from_spiketrain,
    _make_ts_from_spiketrain_multiseg,
    _make_tsgroup_from_spiketrains,
    _make_tsgroup_from_spiketrains_multiseg,
    _make_tsd_from_interface,
    _parse_openephys_electrode_positions,
    _rescale_to_seconds,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic Neo objects
# ---------------------------------------------------------------------------


class _FakeProxy:
    """Wraps a real Neo object to simulate a lazy proxy with .load()."""

    def __init__(self, obj):
        self._obj = obj

    def load(self):
        return self._obj

    def __getattr__(self, name):
        return getattr(self._obj, name)


def _make_block(
    n_samples=100,
    n_channels=3,
    fs=1000.0,
    n_units=3,
    n_segments=1,
    with_epoch=True,
    with_event=True,
):
    """Build a synthetic Neo Block for testing."""
    rng = np.random.default_rng(0)
    block = neo.Block()

    for seg_i in range(n_segments):
        seg = neo.Segment()
        block.segments.append(seg)

        t_start = seg_i * (n_samples / fs)

        # Analog signal
        data = rng.standard_normal((n_samples, n_channels)).astype(np.float32)
        sig = neo.AnalogSignal(
            data,
            units="mV",
            sampling_rate=fs * pq.Hz,
            t_start=t_start * pq.s,
        )
        sig.name = "LFP"
        seg.analogsignals.append(sig)

        # Spike trains
        for u in range(n_units):
            times = np.sort(rng.uniform(t_start, t_start + n_samples / fs, size=20))
            st = neo.SpikeTrain(
                times,
                units="s",
                t_stop=(t_start + n_samples / fs) * pq.s,
            )
            st.name = f"unit{u}"
            seg.spiketrains.append(st)

        # Epoch
        if with_epoch:
            epoch = neo.Epoch(
                times=np.array([t_start + 0.01, t_start + 0.05]) * pq.s,
                durations=np.array([0.01, 0.02]) * pq.s,
                labels=np.array(["a", "b"]),
            )
            epoch.name = "trials"
            seg.epochs.append(epoch)

        # Event
        if with_event:
            event = neo.Event(
                times=np.array([t_start + 0.02, t_start + 0.04]) * pq.s,
                labels=np.array(["x", "y"]),
            )
            event.name = "stim"
            seg.events.append(event)

    return block


def _make_settings_xml(path, n_channels=4):
    """Write a minimal OpenEphys-style settings.xml with electrode positions."""
    root = ET.Element("SETTINGS")
    probe = ET.SubElement(root, "NP_PROBE", probe_serial_number="12345")
    xpos = ET.SubElement(probe, "ELECTRODE_XPOS")
    ypos = ET.SubElement(probe, "ELECTRODE_YPOS")
    for i in range(n_channels):
        xpos.set(f"CH{i}", str(float(i * 10)))
        ypos.set(f"CH{i}", str(float(i * 15)))
    tree = ET.ElementTree(root)
    tree.write(str(path), xml_declaration=True, encoding="utf-8")
    return path


# Neuroscope helpers (reused from test_interface_neurosuite)


def _make_neuroscope_xml(
    session_dir, basename, n_channels=4, fs_dat=20000.0, fs_lfp=1250.0
):
    """Write a minimal Neuroscope-compatible XML file."""
    root = ET.Element("parameters")
    gi = ET.SubElement(root, "generalInfo")
    for tag, text in [
        ("date", "2024-01-01"),
        ("experimenters", "tester"),
        ("description", "test"),
        ("notes", ""),
    ]:
        ET.SubElement(gi, tag).text = text
    acq = ET.SubElement(root, "acquisitionSystem")
    for tag, val in [
        ("nBits", "16"),
        ("nChannels", str(n_channels)),
        ("samplingRate", str(fs_dat)),
        ("voltageRange", "20"),
        ("amplification", "1000"),
        ("offset", "0"),
    ]:
        ET.SubElement(acq, tag).text = val
    fp = ET.SubElement(root, "fieldPotentials")
    ET.SubElement(fp, "lfpSamplingRate").text = str(fs_lfp)
    anat = ET.SubElement(root, "anatomicalDescription")
    cg = ET.SubElement(anat, "channelGroups")
    g = ET.SubElement(cg, "group")
    for ch_id in range(n_channels):
        el = ET.SubElement(g, "channel", skip="0")
        el.text = str(ch_id)
    sd = ET.SubElement(root, "spikeDetection")
    sdcg = ET.SubElement(sd, "channelGroups")
    sg = ET.SubElement(sdcg, "group")
    channels = ET.SubElement(sg, "channels")
    for ch_id in range(n_channels):
        ET.SubElement(channels, "channel").text = str(ch_id)
    ET.SubElement(sg, "nSamples").text = "32"
    ET.SubElement(sg, "nFeatures").text = "3"
    ET.SubElement(sg, "peakSampleIndex").text = "16"
    ET.SubElement(root, "units")
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def block():
    return _make_block()


@pytest.fixture
def neuroscope_dir(tmp_path):
    """A synthetic Neuroscope session with dat, eeg, clu, res files."""
    basename = "sess"
    sdir = tmp_path / basename
    sdir.mkdir()
    n_ch = 4
    n_dat = 200
    n_lfp = 12
    fs_dat = 20000.0

    _make_neuroscope_xml(sdir, basename, n_channels=n_ch)

    rng = np.random.default_rng(0)
    rng.integers(-1000, 1000, (n_dat, n_ch), dtype=np.int16).tofile(
        str(sdir / f"{basename}.dat")
    )
    rng.integers(-500, 500, (n_lfp, n_ch), dtype=np.int16).tofile(
        str(sdir / f"{basename}.eeg")
    )

    # clu/res for shank 1
    n_spikes = 30
    clu = np.concatenate([[3], rng.choice(4, n_spikes)])
    res = np.sort(rng.integers(0, n_dat, n_spikes))
    np.savetxt(str(sdir / f"{basename}.clu.1"), clu, fmt="%d")
    np.savetxt(str(sdir / f"{basename}.res.1"), res, fmt="%d")

    return sdir


@pytest.fixture
def raw_binary_file(tmp_path):
    """A synthetic raw binary file readable by RawBinarySignalIO.

    RawBinarySignalIO defaults: dtype=int16, nb_channel=2, sampling_rate=10000.
    """
    rng = np.random.default_rng(1)
    n_channels = 2  # matches RawBinarySignalIO default nb_channel
    n_samples = 200
    data = rng.integers(-1000, 1000, (n_samples, n_channels), dtype=np.int16)
    fpath = tmp_path / "test.raw"
    data.tofile(str(fpath))
    return fpath, data, n_samples, n_channels


# ===========================================================================
# Tests: helper functions
# ===========================================================================


class TestRescaleToSeconds:
    def test_basic(self):
        q = 500.0 * pq.ms
        assert _rescale_to_seconds(q) == pytest.approx(0.5)

    def test_microseconds(self):
        q = 1_000_000 * pq.us
        assert _rescale_to_seconds(q) == pytest.approx(1.0)


class TestGetSignalType:
    def test_1d(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 1)), units="mV", sampling_rate=1000 * pq.Hz
        )
        assert _get_signal_type(sig) is nap.TsdFrame

    def test_2d(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 3)), units="mV", sampling_rate=1000 * pq.Hz
        )
        assert _get_signal_type(sig) is nap.TsdFrame

    def test_1d_scalar_channel(self):
        """A signal that reports shape as 1-D should map to Tsd."""

        class FakeSignal:
            shape = (10,)

        assert _get_signal_type(FakeSignal()) is nap.Tsd

    def test_3d_tensor(self):
        """A signal with 3+ dims should map to TsdTensor."""

        class FakeSignal:
            shape = (10, 2, 3)

        assert _get_signal_type(FakeSignal()) is nap.TsdTensor


class TestExtractAnnotations:
    def test_with_name(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 1)), units="mV", sampling_rate=1000 * pq.Hz
        )
        sig.name = "mySignal"
        annot = _extract_annotations(sig)
        assert annot["neo_name"] == "mySignal"

    def test_with_description(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 1)), units="mV", sampling_rate=1000 * pq.Hz
        )
        sig.description = "some desc"
        annot = _extract_annotations(sig)
        assert annot["neo_description"] == "some desc"

    def test_empty(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 1)), units="mV", sampling_rate=1000 * pq.Hz
        )
        annot = _extract_annotations(sig)
        assert annot == {}

    def test_with_annotations_dict(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 1)), units="mV", sampling_rate=1000 * pq.Hz
        )
        sig.annotations["brain_area"] = "hippocampus"
        annot = _extract_annotations(sig)
        assert annot["brain_area"] == "hippocampus"


class TestExtractArrayAnnotations:
    def test_with_array_annotations(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 2)), units="mV", sampling_rate=1000 * pq.Hz
        )
        sig.array_annotations["channel"] = np.array([0, 1])
        aa = _extract_array_annotations(sig)
        np.testing.assert_array_equal(aa["channel"], [0, 1])

    def test_empty(self):
        sig = neo.AnalogSignal(
            np.zeros((10, 1)), units="mV", sampling_rate=1000 * pq.Hz
        )
        assert _extract_array_annotations(sig) == {}


class TestParseOpenephysElectrodePositions:
    def test_valid(self, tmp_path):
        xml_path = tmp_path / "settings.xml"
        _make_settings_xml(xml_path, n_channels=4)
        probes = _parse_openephys_electrode_positions(xml_path)
        assert "12345" in probes
        np.testing.assert_array_equal(probes["12345"]["x"][:4], [0.0, 10.0, 20.0, 30.0])
        np.testing.assert_array_equal(probes["12345"]["y"][:4], [0.0, 15.0, 30.0, 45.0])

    def test_missing_file(self, tmp_path):
        result = _parse_openephys_electrode_positions(tmp_path / "nope.xml")
        assert result == {}

    def test_no_probe(self, tmp_path):
        xml_path = tmp_path / "settings.xml"
        root = ET.Element("SETTINGS")
        ET.ElementTree(root).write(str(xml_path))
        assert _parse_openephys_electrode_positions(xml_path) == {}

    def test_probe_without_electrode_pos(self, tmp_path):
        """Probe element exists but no ELECTRODE_XPOS/YPOS."""
        xml_path = tmp_path / "settings.xml"
        root = ET.Element("SETTINGS")
        ET.SubElement(root, "NP_PROBE", probe_serial_number="99")
        ET.ElementTree(root).write(str(xml_path))
        assert _parse_openephys_electrode_positions(xml_path) == {}


# ===========================================================================
# Tests: conversion functions (Neo -> pynapple)
# ===========================================================================


# class TestMakeIntervalSetFromEpoch:
#     def test_basic(self):
#         epoch = neo.Epoch(
#             times=[1.0, 2.0] * pq.s,
#             durations=[0.5, 0.3] * pq.s,
#             labels=["a", "b"],
#         )
#         iset = _make_intervalset_from_epoch(epoch)
#         assert isinstance(iset, nap.IntervalSet)
#         assert len(iset) == 2
#         np.testing.assert_allclose(iset["start"], [1.0, 2.0])
#         np.testing.assert_allclose(iset["end"], [1.5, 2.3])
#
#     def test_labels_in_metadata(self):
#         epoch = neo.Epoch(
#             times=[1.0] * pq.s,
#             durations=[0.5] * pq.s,
#             labels=["trial1"],
#         )
#         iset = _make_intervalset_from_epoch(epoch)
#         assert "label" in iset.metadata.keys()


# class TestMakeIntervalSetFromEpochProxy:
#     def test_proxy_epoch(self):
#         epoch = neo.Epoch(
#             times=[1.0, 2.0] * pq.s,
#             durations=[0.5, 0.3] * pq.s,
#             labels=["a", "b"],
#         )
#         proxy = _FakeProxy(epoch)
#         iset = _make_intervalset_from_epoch(proxy)
#         assert isinstance(iset, nap.IntervalSet)
#         assert len(iset) == 2


class TestMakeIntervalSetFromEpochMultiseg:
    def test_two_segments(self, block):
        block2 = _make_block(n_segments=2)
        iset = _make_intervalset_from_epoch_multiseg(block2, ep_idx=0)
        assert isinstance(iset, nap.IntervalSet)
        assert len(iset) == 4  # 2 epochs per segment * 2 segments

    def test_empty(self):
        block = neo.Block()
        seg = neo.Segment()
        block.segments.append(seg)
        iset = _make_intervalset_from_epoch_multiseg(block, ep_idx=0)
        assert len(iset) == 0


class TestMakeTsFromEvent:
    def test_basic(self):
        event = neo.Event(times=[0.1, 0.2, 0.3] * pq.s, labels=["a", "b", "c"])
        ts = _make_ts_from_event(event)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 3
        np.testing.assert_allclose(ts.times(), [0.1, 0.2, 0.3])

    def test_proxy_event(self):
        event = neo.Event(times=[0.1, 0.2] * pq.s, labels=["a", "b"])
        proxy = _FakeProxy(event)
        ts = _make_ts_from_event(proxy)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 2


class TestMakeTsFromEventMultiseg:
    def test_two_segments(self):
        block = _make_block(n_segments=2, with_epoch=False, with_event=True)
        ts = _make_ts_from_event_multiseg(block, ev_idx=0)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 4  # 2 events per segment

    def test_missing_event_index(self):
        """ev_idx beyond available events should skip those segments."""
        block = _make_block(n_segments=1, with_event=True)
        ts = _make_ts_from_event_multiseg(block, ev_idx=99)
        assert len(ts) == 0

    def test_proxy_events(self):
        """Test with proxy event objects."""
        block = _make_block(n_segments=1, with_event=True)
        # Wrap events in proxies
        seg = block.segments[0]
        seg.events = _FakeProxy(seg.events[0])
        ts = _make_ts_from_event_multiseg(block, ev_idx=0)
        assert isinstance(ts, nap.Ts)


class TestMakeTsFromSpikeTrain:
    def test_basic(self):
        st = neo.SpikeTrain([0.1, 0.2, 0.3], units="s", t_stop=0.5)
        ts = _make_ts_from_spiketrain(st)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 3

    def test_proxy_spiketrain(self):
        st = neo.SpikeTrain([0.1, 0.2], units="s", t_stop=0.5)
        proxy = _FakeProxy(st)
        ts = _make_ts_from_spiketrain(proxy)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 2


class TestMakeTsFromSpikeTrainMultiseg:
    def test_basic(self, block):
        ts = _make_ts_from_spiketrain_multiseg(block, unit_idx=0)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 20  # 20 spikes per unit

    def test_proxy_spiketrains(self):
        block = _make_block(n_segments=1, n_units=2)
        seg = block.segments[0]
        seg.spiketrains = _FakeProxy(seg.spiketrains)
        ts = _make_ts_from_spiketrain_multiseg(block, unit_idx=0)
        assert isinstance(ts, nap.Ts)


class TestMakeTsGroupFromSpikeTrains:
    def test_basic(self, block):
        tsgroup = _make_tsgroup_from_spiketrains(block.segments[0].spiketrains)
        assert isinstance(tsgroup, nap.TsGroup)
        assert len(tsgroup) == 3  # 3 units

    def test_proxy_spiketrains(self):
        block = _make_block(n_units=2)
        proxies = [_FakeProxy(st) for st in block.segments[0].spiketrains]
        tsgroup = _make_tsgroup_from_spiketrains(proxies)
        assert isinstance(tsgroup, nap.TsGroup)
        assert len(tsgroup) == 2


class TestMakeTsGroupFromSpikeTrainsMultiseg:
    def test_two_segments(self):
        block = _make_block(n_segments=2, n_units=3)
        all_st = [s.spiketrains for s in block.segments]
        tsgroup = _make_tsgroup_from_spiketrains_multiseg(all_st)
        assert isinstance(tsgroup, nap.TsGroup)
        assert len(tsgroup) == 3
        # Each unit should have spikes from both segments
        for i in tsgroup.keys():
            assert len(tsgroup[i]) == 40  # 20 per segment * 2

    def test_proxy_spiketrains(self):
        block = _make_block(n_segments=2, n_units=2)
        all_st = [[_FakeProxy(st) for st in s.spiketrains] for s in block.segments]
        tsgroup = _make_tsgroup_from_spiketrains_multiseg(all_st)
        assert isinstance(tsgroup, nap.TsGroup)
        assert len(tsgroup) == 2

    def test_empty(self):
        time_support = nap.IntervalSet(start=0, end=1)
        tsgroup = _make_tsgroup_from_spiketrains_multiseg([], time_support=time_support)
        assert len(tsgroup) == 0


# ===========================================================================
# Tests: NeoSignalInterface
# ===========================================================================


class TestNeoSignalInterface:
    def test_shape_and_dtype(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        assert iface.shape == (100, 3)
        assert iface.dtype == np.float32

    def test_ndim(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        assert iface.ndim == 2

    def test_len(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        assert len(iface) == 100

    def test_times(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        assert len(iface.times) == 100
        assert iface.times[0] == pytest.approx(0.0)

    def test_integer_index(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        row = iface[0]
        assert row.shape == (3,)

    def test_slice_index(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        chunk = iface[0:10]
        assert chunk.shape == (10, 3)

    def test_tuple_index(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        col = iface[0:10, 1]
        assert col.shape == (10,)

    def test_multi_segment(self):
        block = _make_block(n_segments=2, n_channels=2)
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        assert iface.shape == (200, 2)  # 100 per segment
        assert len(iface.times) == 200

    def test_repr(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        assert "NeoSignalInterface" in repr(iface)

    def test_make_tsd_from_interface(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        result = _make_tsd_from_interface(iface)
        assert isinstance(result, nap.TsdFrame)

    def test_find_segment_negative_index(self, block):
        """_find_segment_for_index resolves negative indices."""
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        seg_idx, local_idx = iface._find_segment_for_index(-1)
        assert seg_idx == 0
        assert local_idx == 99

    def test_out_of_bounds_index(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        with pytest.raises(IndexError):
            iface[200]

    def test_negative_out_of_bounds(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        with pytest.raises(IndexError):
            iface[-200]

    def test_slice_with_step(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        chunk = iface[0:20:2]
        assert chunk.shape == (10, 3)

    def test_slice_negative_indices(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        chunk = iface[-10:]
        assert chunk.shape == (10, 3)

    def test_slice_defaults(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        # slice(None, None) == [:]
        chunk = iface[:]
        assert chunk.shape == (100, 3)

    def test_empty_range(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        chunk = iface[50:50]
        assert chunk.shape[0] == 0

    def test_array_index(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        result = iface[[0, 5, 10]]
        assert result.shape == (3, 3)

    def test_bool_index(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        mask = np.zeros(100, dtype=bool)
        mask[[0, 1, 2]] = True
        result = iface[mask]
        assert result.shape == (3, 3)

    def test_invalid_index_type(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        with pytest.raises(TypeError):
            iface["bad"]

    def test_iter(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        rows = list(iface)
        assert len(rows) == 100
        assert rows[0].shape == (3,)

    def test_irregular_signal(self):
        times = np.sort(np.random.default_rng(0).uniform(0, 1, size=50))
        data = np.random.default_rng(0).standard_normal((50, 2)).astype(np.float64)
        sig = neo.IrregularlySampledSignal(
            times,
            data,
            units="mV",
            time_units="s",
        )
        block = neo.Block()
        seg = neo.Segment()
        block.segments.append(seg)
        seg.irregularlysampledsignals.append(sig)
        iface = NeoSignalInterface(sig, block, sig_num=0)
        assert iface.is_analog is False
        assert iface.shape == (50, 2)
        assert iface.nap_type is nap.TsdFrame
        row = iface[0]
        assert row.shape == (2,)

    def test_unsupported_signal_type(self):
        block = neo.Block()
        with pytest.raises(TypeError, match="not recognized"):
            NeoSignalInterface("not_a_signal", block, sig_num=0)

    def test_multi_segment_cross_boundary(self):
        """Request data spanning two segments."""
        block = _make_block(n_segments=2, n_channels=2, n_samples=100)
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        # Total 200 samples: seg0=[0..99], seg1=[100..199]
        chunk = iface[90:110]
        assert chunk.shape == (20, 2)

    def test_find_segment_out_of_bounds(self, block):
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        with pytest.raises(IndexError, match="out of bounds"):
            iface._find_segment_for_index(500)

    def test_slice_negative_stop(self, block):
        """Negative stop index in slice."""
        sig = block.segments[0].analogsignals[0]
        iface = NeoSignalInterface(sig, block, sig_num=0)
        chunk = iface[0:-5]
        assert chunk.shape == (95, 3)

    def test_load_proxy_signal(self):
        """NeoSignalInterface with a signal that has a .load() method."""
        block = _make_block(n_channels=2, n_samples=50)
        sig = block.segments[0].analogsignals[0]
        # Wrap in proxy
        proxy_sig = _FakeProxy(sig)
        block.segments[0].analogsignals[0] = proxy_sig
        iface = NeoSignalInterface(sig, block, sig_num=0)
        chunk = iface[0:10]
        assert chunk.shape == (10, 2)


# ===========================================================================
# Tests: EphysReader with Neuroscope path
# ===========================================================================


class TestEphysReaderNeuroscope:
    def test_init(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        assert isinstance(reader, EphysReader)

    def test_has_dict_interface(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        assert hasattr(reader, "keys")
        assert hasattr(reader, "__getitem__")

    def test_keys_present(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        keys = reader.keys()
        assert any("dat" in k for k in keys)
        assert any("eeg" in k for k in keys)
        assert any("clu" in k for k in keys)

    def test_load_dat(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        dat_key = [k for k in reader.keys() if "dat" in k][0]
        tsd = reader[dat_key]
        assert isinstance(tsd, nap.TsdFrame)
        assert tsd.shape[1] == 4

    def test_load_eeg(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        eeg_key = [k for k in reader.keys() if "eeg" in k][0]
        tsd = reader[eeg_key]
        assert isinstance(tsd, nap.TsdFrame)

    def test_load_spikes(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        clu_key = [k for k in reader.keys() if "clu" in k][0]
        tsgroup = reader[clu_key]
        assert isinstance(tsgroup, nap.TsGroup)

    def test_caching(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        dat_key = [k for k in reader.keys() if "dat" in k][0]
        obj1 = reader[dat_key]
        obj2 = reader[dat_key]
        assert obj1 is obj2

    def test_str(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        s = str(reader)
        assert "sess" in s

    def test_repr(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        assert str(reader) == repr(reader)

    def test_invalid_key(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        with pytest.raises(KeyError):
            reader["nonexistent"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EphysReader(tmp_path / "no_such_path")

    def test_format_string_neuroscope(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir, format="neuroscope")
        assert any("dat" in k for k in reader.keys())

    def test_is_neuroscope_with_type(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir, format=neo.io.NeuroScopeIO)
        assert any("dat" in k for k in reader.keys())

    def test_is_neuroscope_binary_file(self, neuroscope_dir):
        """Passing a .dat file directly should trigger Neuroscope path."""
        dat_file = list(neuroscope_dir.glob("*.dat"))[0]
        reader = EphysReader(dat_file)
        assert any("dat" in k for k in reader.keys())

    def test_items(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        items = reader.items()
        assert len(items) > 0
        for k, v in items:
            assert isinstance(k, str)

    def test_values(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        vals = reader.values()
        assert len(vals) > 0

    def test_close(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        reader = EphysReader(fpath, format="RawBinarySignalIO")
        reader.close()  # Should not raise

    def test_close_neuroscope(self, neuroscope_dir):
        reader = EphysReader(neuroscope_dir)
        reader.close()
        # After close, _ns should be cleaned up
        assert not hasattr(reader, "_ns")

    def test_str_after_load(self, neuroscope_dir):
        """After loading, __str__ should handle non-dict values."""
        reader = EphysReader(neuroscope_dir)
        dat_key = [k for k in reader.keys() if "dat" in k][0]
        reader[dat_key]  # load it so data[key] is now a TsdFrame, not dict
        s = str(reader)
        assert "TsdFrame" in s


# ===========================================================================
# Tests: EphysReader with RawBinarySignalIO (mmap path)
# ===========================================================================


class TestEphysReaderMmap:
    def test_init(self, raw_binary_file):
        fpath, _, n_samples, n_channels = raw_binary_file
        reader = EphysReader(
            fpath,
            format="RawBinarySignalIO",
        )
        assert isinstance(reader, EphysReader)

    def test_keys_present(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        reader = EphysReader(fpath, format="RawBinarySignalIO")
        assert len(reader.keys()) >= 1

    def test_load_returns_tsdframe(self, raw_binary_file):
        fpath, data, n_samples, n_channels = raw_binary_file
        reader = EphysReader(fpath, format="RawBinarySignalIO")
        key = reader.keys()[0]
        tsd = reader[key]
        assert isinstance(tsd, nap.TsdFrame)

    def test_mmap_shape(self, raw_binary_file):
        fpath, data, n_samples, n_channels = raw_binary_file
        reader = EphysReader(fpath, format="RawBinarySignalIO")
        key = reader.keys()[0]
        tsd = reader[key]
        assert tsd.shape == (n_samples, n_channels)

    def test_mmap_data(self, raw_binary_file):
        fpath, data, n_samples, n_channels = raw_binary_file
        reader = EphysReader(fpath, format="RawBinarySignalIO")
        key = reader.keys()[0]
        tsd = reader[key]
        # Data is memory-mapped (or a slice of a memmap)
        assert isinstance(tsd.d, np.ndarray)

    def test_caching(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        reader = EphysReader(fpath, format="RawBinarySignalIO")
        key = reader.keys()[0]
        obj1 = reader[key]
        obj2 = reader[key]
        assert obj1 is obj2


# ===========================================================================
# Tests: EphysReader.__getitem__ string loader dispatch
# ===========================================================================


class TestEphysReaderStringLoaders:
    """Test the string loader branches in __getitem__ by constructing
    data entries directly (simulating what _collect_data produces)."""

    @pytest.fixture
    def reader_with_block(self, neuroscope_dir):
        """Create a reader and inject a synthetic block with string loaders."""
        reader = EphysReader(neuroscope_dir)
        block = _make_block(n_segments=2, n_units=3)
        time_support = nap.IntervalSet(start=0, end=0.2)

        # Inject string-loader entries matching _collect_data patterns
        reader.data["test_analog"] = {
            "type": "TsdFrame",
            "loader": "analogsignal",
            "block": block,
            "sig_num": 0,
            "time_support": time_support,
        }
        reader.data["test_spiketrain"] = {
            "type": "Ts",
            "loader": "spiketrain",
            "block": block,
            "unit_idx": 0,
            "time_support": time_support,
        }
        reader.data["test_tsgroup"] = {
            "type": "TsGroup",
            "loader": "tsgroup",
            "block": block,
            "time_support": time_support,
        }
        reader.data["test_epoch"] = {
            "type": "IntervalSet",
            "loader": "epoch",
            "block": block,
            "ep_idx": 0,
            "time_support": time_support,
        }
        reader.data["test_event"] = {
            "type": "Ts",
            "loader": "event",
            "block": block,
            "ev_idx": 0,
            "time_support": time_support,
        }
        reader.data["test_unknown"] = {
            "type": "???",
            "loader": "unknown_loader_type",
        }
        return reader

    def test_load_analogsignal(self, reader_with_block):
        result = reader_with_block["test_analog"]
        assert isinstance(result, nap.TsdFrame)

    def test_load_spiketrain(self, reader_with_block):
        result = reader_with_block["test_spiketrain"]
        assert isinstance(result, nap.Ts)

    def test_load_tsgroup(self, reader_with_block):
        result = reader_with_block["test_tsgroup"]
        assert isinstance(result, nap.TsGroup)
        assert len(result) == 3

    def test_load_epoch(self, reader_with_block):
        result = reader_with_block["test_epoch"]
        assert isinstance(result, nap.IntervalSet)

    def test_load_event(self, reader_with_block):
        result = reader_with_block["test_event"]
        assert isinstance(result, nap.Ts)

    def test_load_unknown_raises(self, reader_with_block):
        with pytest.raises(ValueError, match="Unknown loader type"):
            reader_with_block["test_unknown"]

    def test_load_irregular_signal(self, neuroscope_dir):
        """Test the irregularsignal loader path."""
        reader = EphysReader(neuroscope_dir)

        # Create a block with an irregularly sampled signal
        block = neo.Block()
        seg = neo.Segment()
        block.segments.append(seg)
        times = np.sort(np.random.default_rng(0).uniform(0, 1, size=50))
        data = np.random.default_rng(0).standard_normal((50, 2))
        isig = neo.IrregularlySampledSignal(
            times,
            data,
            units="mV",
            time_units="s",
        )
        seg.irregularlysampledsignals.append(isig)

        reader.data["test_irreg"] = {
            "type": "TsdFrame",
            "loader": "irregularsignal",
            "block": block,
            "sig_num": 0,
            "time_support": nap.IntervalSet(start=0, end=1),
        }
        result = reader["test_irreg"]
        assert isinstance(result, nap.TsdFrame)


# ===========================================================================
# Tests: EphysReader._collect_data non-mmap path
# ===========================================================================


class TestEphysReaderNonMmap:
    def test_analogsignal_non_mmap_path(self, raw_binary_file, monkeypatch):
        """When has_buffer_description_api returns False, _collect_data should
        create analogsignal string-loader entries instead of mmap entries."""
        fpath, _, _, _ = raw_binary_file
        # Monkeypatch to disable buffer API so we go through the non-mmap path
        import neo.rawio.rawbinarysignalrawio as rbmod

        monkeypatch.setattr(
            rbmod.RawBinarySignalRawIO,
            "has_buffer_description_api",
            staticmethod(lambda: False),
        )
        reader = EphysReader(fpath, format="RawBinarySignalIO")
        key = reader.keys()[0]
        # The data entry should be a dict with "loader" == "analogsignal"
        assert reader.data[key]["loader"] == "analogsignal"
        # Loading should produce a TsdFrame
        result = reader[key]
        assert isinstance(result, nap.TsdFrame)


class TestEphysReaderStrFallback:
    def test_str_without_tabulate(self, neuroscope_dir, monkeypatch):
        """Test __str__ when tabulate is not available."""
        import pynapple.io.interface_neo as mod

        monkeypatch.setattr(mod, "HAS_TABULATE", False)
        reader = EphysReader(neuroscope_dir)
        s = str(reader)
        assert "sess" in s
        assert "TsdFrame" in s


class TestEphysReaderFormatHandling:
    def test_format_string_case_insensitive(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        reader = EphysReader(fpath, format="rawbinarysignalio")
        assert len(reader.keys()) >= 1

    def test_format_string_without_io_suffix(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        reader = EphysReader(fpath, format="rawbinarysignal")
        assert len(reader.keys()) >= 1

    def test_format_class(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        reader = EphysReader(fpath, format=neo.io.RawBinarySignalIO)
        assert len(reader.keys()) >= 1

    def test_format_invalid_string(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        with pytest.raises(ValueError, match="not found"):
            EphysReader(fpath, format="NoSuchFormatIO")

    def test_format_invalid_type(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file
        with pytest.raises(TypeError, match="format must be"):
            EphysReader(fpath, format=12345)

    def test_format_class_not_in_iolist(self, raw_binary_file):
        fpath, _, _, _ = raw_binary_file

        class FakeIO:
            __name__ = "FakeIO"

        with pytest.raises(ValueError, match="not in neo.io.iolist"):
            EphysReader(fpath, format=FakeIO)
