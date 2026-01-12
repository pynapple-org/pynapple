# -*- coding: utf-8 -*-
"""Tests of Neo interface for `pynapple` package."""

import warnings

import numpy as np
import pytest

import pynapple as nap

# Check if neo is installed
try:
    import neo
    import quantities as pq

    HAS_NEO = True
except ImportError:
    HAS_NEO = False

pytestmark = pytest.mark.skipif(not HAS_NEO, reason="Neo is not installed")


# =============================================================================
# Helper functions to create mock Neo objects
# =============================================================================


def create_mock_analog_signal(n_samples=100, n_channels=3, sampling_rate=1000.0, t_start=0.0):
    """Create a mock Neo AnalogSignal."""
    data = np.random.randn(n_samples, n_channels)
    signal = neo.AnalogSignal(
        data,
        units="mV",
        sampling_rate=sampling_rate * pq.Hz,
        t_start=t_start * pq.s,
    )
    return signal


def create_mock_spiketrain(n_spikes=50, t_start=0.0, t_stop=10.0):
    """Create a mock Neo SpikeTrain."""
    spike_times = np.sort(np.random.uniform(t_start, t_stop, n_spikes))
    spiketrain = neo.SpikeTrain(
        spike_times,
        units="s",
        t_start=t_start * pq.s,
        t_stop=t_stop * pq.s,
    )
    return spiketrain


def create_mock_epoch(n_epochs=5, t_start=0.0, max_duration=2.0):
    """Create a mock Neo Epoch."""
    times = np.sort(np.random.uniform(t_start, 10.0, n_epochs))
    durations = np.random.uniform(0.1, max_duration, n_epochs)
    labels = np.array([f"epoch_{i}" for i in range(n_epochs)])
    epoch = neo.Epoch(
        times=times * pq.s,
        durations=durations * pq.s,
        labels=labels,
    )
    return epoch


def create_mock_event(n_events=10, t_start=0.0, t_stop=10.0):
    """Create a mock Neo Event."""
    times = np.sort(np.random.uniform(t_start, t_stop, n_events))
    labels = np.array([f"event_{i}" for i in range(n_events)])
    event = neo.Event(
        times=times * pq.s,
        labels=labels,
    )
    return event


def create_mock_irregular_signal(n_samples=50, n_channels=2, t_start=0.0, t_stop=10.0):
    """Create a mock Neo IrregularlySampledSignal."""
    times = np.sort(np.random.uniform(t_start, t_stop, n_samples))
    data = np.random.randn(n_samples, n_channels)
    signal = neo.IrregularlySampledSignal(
        times=times * pq.s,
        signal=data,
        units="mV",
        time_units="s",
    )
    return signal


def create_mock_block_with_segments(n_segments=2, n_analog=2, n_spiketrains=3):
    """Create a mock Neo Block with multiple segments."""
    block = neo.Block(name="test_block")

    for seg_idx in range(n_segments):
        seg = neo.Segment(name=f"segment_{seg_idx}")

        # Add analog signals
        for sig_idx in range(n_analog):
            signal = create_mock_analog_signal(
                n_samples=100,
                n_channels=3,
                sampling_rate=1000.0,
                t_start=seg_idx * 10.0,
            )
            signal.name = f"analog_{sig_idx}"
            seg.analogsignals.append(signal)

        # Add spike trains
        for st_idx in range(n_spiketrains):
            spiketrain = create_mock_spiketrain(
                n_spikes=20,
                t_start=seg_idx * 10.0,
                t_stop=(seg_idx + 1) * 10.0,
            )
            spiketrain.name = f"unit_{st_idx}"
            seg.spiketrains.append(spiketrain)

        # Add an epoch
        epoch = create_mock_epoch(n_epochs=3, t_start=seg_idx * 10.0)
        epoch.name = "behavioral_states"
        seg.epochs.append(epoch)

        # Add an event
        event = create_mock_event(
            n_events=5,
            t_start=seg_idx * 10.0,
            t_stop=(seg_idx + 1) * 10.0,
        )
        event.name = "stimuli"
        seg.events.append(event)

        # Set segment times
        seg.t_start = seg_idx * 10.0 * pq.s
        seg.t_stop = (seg_idx + 1) * 10.0 * pq.s

        block.segments.append(seg)

    return block


# =============================================================================
# Tests for conversion functions: Neo -> Pynapple
# =============================================================================


class TestNeoToPynapple:
    """Test Neo to Pynapple conversion functions."""

    def test_analog_to_tsd(self):
        """Test AnalogSignal to Tsd conversion."""
        from pynapple.io.interface_neo import _make_tsd_from_analog

        # Single channel -> Tsd
        signal = create_mock_analog_signal(n_samples=100, n_channels=1)
        tsd = _make_tsd_from_analog(signal)
        assert isinstance(tsd, nap.Tsd)
        assert len(tsd) == 100

    def test_analog_to_tsdframe(self):
        """Test AnalogSignal to TsdFrame conversion."""
        from pynapple.io.interface_neo import _make_tsd_from_analog

        # Multi-channel -> TsdFrame
        signal = create_mock_analog_signal(n_samples=100, n_channels=3)
        tsdframe = _make_tsd_from_analog(signal)
        assert isinstance(tsdframe, nap.TsdFrame)
        assert len(tsdframe) == 100
        assert tsdframe.shape[1] == 3

    def test_spiketrain_to_ts(self):
        """Test SpikeTrain to Ts conversion."""
        from pynapple.io.interface_neo import _make_ts_from_spiketrain

        spiketrain = create_mock_spiketrain(n_spikes=50)
        ts = _make_ts_from_spiketrain(spiketrain)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 50

    def test_spiketrains_to_tsgroup(self):
        """Test multiple SpikeTrains to TsGroup conversion."""
        from pynapple.io.interface_neo import _make_tsgroup_from_spiketrains

        spiketrains = [create_mock_spiketrain(n_spikes=30) for _ in range(5)]
        tsgroup = _make_tsgroup_from_spiketrains(spiketrains)
        assert isinstance(tsgroup, nap.TsGroup)
        assert len(tsgroup) == 5

    def test_epoch_to_intervalset(self):
        """Test Epoch to IntervalSet conversion."""
        from pynapple.io.interface_neo import _make_intervalset_from_epoch

        epoch = create_mock_epoch(n_epochs=5)
        iset = _make_intervalset_from_epoch(epoch)
        assert isinstance(iset, nap.IntervalSet)
        assert len(iset) == 5

    def test_event_to_ts(self):
        """Test Event to Ts conversion."""
        from pynapple.io.interface_neo import _make_ts_from_event

        event = create_mock_event(n_events=10)
        ts = _make_ts_from_event(event)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 10

    def test_irregular_signal_to_tsd(self):
        """Test IrregularlySampledSignal to Tsd conversion."""
        from pynapple.io.interface_neo import _make_tsd_from_irregular

        signal = create_mock_irregular_signal(n_samples=50, n_channels=1)
        tsd = _make_tsd_from_irregular(signal)
        assert isinstance(tsd, nap.Tsd)
        assert len(tsd) == 50

    def test_irregular_signal_to_tsdframe(self):
        """Test IrregularlySampledSignal to TsdFrame conversion."""
        from pynapple.io.interface_neo import _make_tsd_from_irregular

        signal = create_mock_irregular_signal(n_samples=50, n_channels=3)
        tsdframe = _make_tsd_from_irregular(signal)
        assert isinstance(tsdframe, nap.TsdFrame)
        assert len(tsdframe) == 50
        assert tsdframe.shape[1] == 3


# =============================================================================
# Tests for conversion functions: Pynapple -> Neo
# =============================================================================


class TestPynappleToNeo:
    """Test Pynapple to Neo conversion functions."""

    def test_tsd_to_analog(self):
        """Test Tsd to AnalogSignal conversion."""
        from pynapple.io.interface_neo import to_neo_analogsignal

        tsd = nap.Tsd(t=np.arange(100) / 1000.0, d=np.random.randn(100))
        signal = to_neo_analogsignal(tsd, units="mV")

        assert isinstance(signal, neo.AnalogSignal)
        assert signal.shape[0] == 100
        assert signal.shape[1] == 1  # Tsd is converted to 2D

    def test_tsdframe_to_analog(self):
        """Test TsdFrame to AnalogSignal conversion."""
        from pynapple.io.interface_neo import to_neo_analogsignal

        tsdframe = nap.TsdFrame(
            t=np.arange(100) / 1000.0, d=np.random.randn(100, 3)
        )
        signal = to_neo_analogsignal(tsdframe, units="uV")

        assert isinstance(signal, neo.AnalogSignal)
        assert signal.shape == (100, 3)

    def test_ts_to_spiketrain(self):
        """Test Ts to SpikeTrain conversion."""
        from pynapple.io.interface_neo import to_neo_spiketrain

        ts = nap.Ts(t=np.sort(np.random.uniform(0, 10, 50)))
        spiketrain = to_neo_spiketrain(ts, t_stop=10.0)

        assert isinstance(spiketrain, neo.SpikeTrain)
        assert len(spiketrain) == 50

    def test_intervalset_to_epoch(self):
        """Test IntervalSet to Epoch conversion."""
        from pynapple.io.interface_neo import to_neo_epoch

        iset = nap.IntervalSet(start=[0, 5, 10], end=[2, 7, 12])
        epoch = to_neo_epoch(iset)

        assert isinstance(epoch, neo.Epoch)
        assert len(epoch) == 3
        np.testing.assert_array_almost_equal(
            epoch.times.rescale("s").magnitude, np.array([0, 5, 10])
        )
        np.testing.assert_array_almost_equal(
            epoch.durations.rescale("s").magnitude, np.array([2, 2, 2])
        )

    def test_ts_to_event(self):
        """Test Ts to Event conversion."""
        from pynapple.io.interface_neo import to_neo_event

        ts = nap.Ts(t=np.array([1.0, 2.5, 5.0, 7.5]))
        event = to_neo_event(ts)

        assert isinstance(event, neo.Event)
        assert len(event) == 4
        np.testing.assert_array_almost_equal(
            event.times.rescale("s").magnitude, np.array([1.0, 2.5, 5.0, 7.5])
        )


# =============================================================================
# Tests for NeoSignalInterface
# =============================================================================


class TestNeoSignalInterface:
    """Test NeoSignalInterface class."""

    def test_analog_interface_init(self):
        """Test initialization with AnalogSignal."""
        from pynapple.io.interface_neo import NeoSignalInterface

        block = create_mock_block_with_segments(n_segments=1, n_analog=1, n_spiketrains=0)
        signal = block.segments[0].analogsignals[0]
        time_support = nap.IntervalSet(start=0, end=10)

        interface = NeoSignalInterface(signal, block, time_support, sig_num=0)

        assert interface.is_analog is True
        assert interface.nap_type == nap.TsdFrame
        assert interface._signal_type == "analog"

    def test_spiketrain_interface_init(self):
        """Test initialization with SpikeTrain."""
        from pynapple.io.interface_neo import NeoSignalInterface

        block = create_mock_block_with_segments(n_segments=1, n_analog=0, n_spiketrains=1)
        spiketrain = block.segments[0].spiketrains[0]
        time_support = nap.IntervalSet(start=0, end=10)

        interface = NeoSignalInterface(spiketrain, block, time_support, sig_num=0)

        assert interface.is_analog is False
        assert interface.nap_type == nap.Ts
        assert interface._signal_type == "spiketrain"

    def test_tsgroup_interface_init(self):
        """Test initialization with multiple SpikeTrains."""
        from pynapple.io.interface_neo import NeoSignalInterface

        block = create_mock_block_with_segments(n_segments=1, n_analog=0, n_spiketrains=3)
        spiketrains = block.segments[0].spiketrains
        time_support = nap.IntervalSet(start=0, end=10)

        interface = NeoSignalInterface(spiketrains, block, time_support)

        assert interface.is_analog is False
        assert interface.nap_type == nap.TsGroup
        assert interface._signal_type == "tsgroup"

    def test_interface_load(self):
        """Test loading data through interface."""
        from pynapple.io.interface_neo import NeoSignalInterface

        block = create_mock_block_with_segments(n_segments=1, n_analog=1, n_spiketrains=0)
        signal = block.segments[0].analogsignals[0]
        time_support = nap.IntervalSet(start=0, end=0.1)

        interface = NeoSignalInterface(signal, block, time_support, sig_num=0)
        loaded = interface.load()

        assert isinstance(loaded, nap.TsdFrame)

    def test_interface_get_time_range(self):
        """Test getting data for a time range."""
        from pynapple.io.interface_neo import NeoSignalInterface

        block = create_mock_block_with_segments(n_segments=1, n_analog=1, n_spiketrains=0)
        signal = block.segments[0].analogsignals[0]
        time_support = nap.IntervalSet(start=0, end=0.1)

        interface = NeoSignalInterface(signal, block, time_support, sig_num=0)
        data = interface.get(0.0, 0.05)

        assert isinstance(data, nap.TsdFrame)

    def test_interface_restrict(self):
        """Test restricting data to epochs."""
        from pynapple.io.interface_neo import NeoSignalInterface

        block = create_mock_block_with_segments(n_segments=1, n_analog=1, n_spiketrains=0)
        signal = block.segments[0].analogsignals[0]
        time_support = nap.IntervalSet(start=0, end=0.1)

        interface = NeoSignalInterface(signal, block, time_support, sig_num=0)
        epoch = nap.IntervalSet(start=[0.01], end=[0.03])
        data = interface.restrict(epoch)

        assert isinstance(data, nap.TsdFrame)


# =============================================================================
# Tests for legacy interface
# =============================================================================


class TestLegacyInterface:
    """Test legacy NEOExperimentInterface for backward compatibility."""

    def test_legacy_deprecation_warning(self):
        """Test that legacy interface raises deprecation warning."""
        from pynapple.io.interface_neo import NEOExperimentInterface

        # Create a simple mock reader
        class MockReader:
            def read(self, lazy=False):
                return [create_mock_block_with_segments(n_segments=1)]

        with pytest.warns(DeprecationWarning, match="NEOExperimentInterface is deprecated"):
            NEOExperimentInterface(MockReader(), lazy=False)


# =============================================================================
# Tests for helper functions
# =============================================================================


class TestHelperFunctions:
    """Test helper functions."""

    def test_rescale_to_seconds(self):
        """Test rescaling quantities to seconds."""
        from pynapple.io.interface_neo import _rescale_to_seconds

        # Test milliseconds
        value_ms = 1000 * pq.ms
        assert _rescale_to_seconds(value_ms) == 1.0

        # Test seconds
        value_s = 5.0 * pq.s
        assert _rescale_to_seconds(value_s) == 5.0

    def test_get_signal_type(self):
        """Test signal type detection."""
        from pynapple.io.interface_neo import _get_signal_type

        # 1D signal -> Tsd
        signal_1d = neo.AnalogSignal(
            np.random.randn(100, 1), units="mV", sampling_rate=1000 * pq.Hz
        )
        assert _get_signal_type(signal_1d) == nap.Tsd

        # 2D signal -> TsdFrame
        signal_2d = neo.AnalogSignal(
            np.random.randn(100, 3), units="mV", sampling_rate=1000 * pq.Hz
        )
        assert _get_signal_type(signal_2d) == nap.TsdFrame

    def test_extract_annotations(self):
        """Test annotation extraction."""
        from pynapple.io.interface_neo import _extract_annotations

        signal = neo.AnalogSignal(
            np.random.randn(100, 1),
            units="mV",
            sampling_rate=1000 * pq.Hz,
            name="test_signal",
            description="A test signal",
            custom_annotation="custom_value",
        )

        annotations = _extract_annotations(signal)

        assert "neo_name" in annotations
        assert annotations["neo_name"] == "test_signal"
        assert "neo_description" in annotations
        assert annotations["neo_description"] == "A test signal"
        assert "custom_annotation" in annotations
        assert annotations["custom_annotation"] == "custom_value"


# =============================================================================
# Tests for round-trip conversion
# =============================================================================


class TestRoundTrip:
    """Test round-trip conversion (pynapple -> neo -> pynapple)."""

    def test_tsd_roundtrip(self):
        """Test Tsd round-trip conversion."""
        from pynapple.io.interface_neo import to_neo_analogsignal, _make_tsd_from_analog

        original = nap.Tsd(t=np.arange(100) / 1000.0, d=np.random.randn(100))

        # Convert to Neo
        neo_signal = to_neo_analogsignal(original)

        # Convert back to pynapple
        recovered = _make_tsd_from_analog(neo_signal)

        assert isinstance(recovered, nap.Tsd)
        np.testing.assert_array_almost_equal(original.values, recovered.values.flatten())

    def test_intervalset_roundtrip(self):
        """Test IntervalSet round-trip conversion."""
        from pynapple.io.interface_neo import to_neo_epoch, _make_intervalset_from_epoch

        original = nap.IntervalSet(start=[1.0, 5.0, 10.0], end=[2.0, 7.0, 12.0])

        # Convert to Neo
        neo_epoch = to_neo_epoch(original)

        # Convert back to pynapple
        recovered = _make_intervalset_from_epoch(neo_epoch)

        assert isinstance(recovered, nap.IntervalSet)
        np.testing.assert_array_almost_equal(original.start, recovered.start)
        np.testing.assert_array_almost_equal(original.end, recovered.end)

    def test_ts_roundtrip(self):
        """Test Ts round-trip conversion via SpikeTrain."""
        from pynapple.io.interface_neo import to_neo_spiketrain, _make_ts_from_spiketrain

        spike_times = np.sort(np.random.uniform(0, 10, 50))
        original = nap.Ts(t=spike_times)

        # Convert to Neo
        neo_spiketrain = to_neo_spiketrain(original, t_stop=10.0)

        # Convert back to pynapple
        recovered = _make_ts_from_spiketrain(neo_spiketrain)

        assert isinstance(recovered, nap.Ts)
        np.testing.assert_array_almost_equal(original.times(), recovered.times())


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_block_with_all_data_types(self):
        """Test processing a block with all data types."""
        block = create_mock_block_with_segments(n_segments=2, n_analog=2, n_spiketrains=3)

        # Verify block structure
        assert len(block.segments) == 2

        for seg in block.segments:
            assert len(seg.analogsignals) == 2
            assert len(seg.spiketrains) == 3
            assert len(seg.epochs) == 1
            assert len(seg.events) == 1

    def test_multi_segment_time_support(self):
        """Test that time support correctly spans multiple segments."""
        from pynapple.io.interface_neo import NeoSignalInterface

        block = create_mock_block_with_segments(n_segments=3, n_analog=1, n_spiketrains=0)

        # Build time support from segments
        starts = np.array([seg.t_start.rescale("s").magnitude for seg in block.segments])
        ends = np.array([seg.t_stop.rescale("s").magnitude for seg in block.segments])
        time_support = nap.IntervalSet(starts, ends)

        assert len(time_support) == 3

        interface = NeoSignalInterface(
            block.segments[0].analogsignals[0], block, time_support, sig_num=0
        )

        # Load all data
        loaded = interface.load()
        assert isinstance(loaded, nap.TsdFrame)