"""Tests of perievent for `pynapple` package."""

import numpy as np
import pytest

import pynapple as nap

# =============================================================================
# compute_perievent — discrete data (Ts input)
# =============================================================================


class TestComputePerieventDiscrete:
    def test_align_ts(self):
        tsd = nap.Ts(t=np.arange(100))
        tref = nap.Ts(t=np.arange(10, 100, 10))
        peth = nap.compute_perievent(tsd, tref, window=(-10, 10))

        assert len(peth) == len(tref)
        assert isinstance(peth, nap.TsGroup)
        # Check a central event (t=50) that has a full window on both sides.
        central = peth[4]  # tref[4] = 50
        np.testing.assert_array_almost_equal(central.index, np.arange(-10, 11))

    def test_align_ts_symmetric_window(self):
        """Single number window should be treated as symmetric."""
        tsd = nap.Ts(t=np.arange(100))
        tref = nap.Ts(t=np.arange(10, 100, 10))
        peth = nap.compute_perievent(tsd, tref, window=10.0)

        assert isinstance(peth, nap.TsGroup)
        assert len(peth) == len(tref)
        central = peth[4]  # tref[4] = 50
        np.testing.assert_array_almost_equal(central.index, np.arange(-10, 11))

    def test_align_ts_asymmetric_window(self):
        """Asymmetric window (-0.5, 1.5) should correctly bound each aligned train."""
        spikes = nap.Ts(t=[0.5, 1.2, 2.1, 3.5])
        events = nap.Ts(t=[1.0, 3.0])
        result = nap.compute_perievent(spikes, events, window=(-0.5, 1.5))

        # First event at t=1.0 → captures spikes at 0.5 (-0.5) and 1.2 (+0.2) and 2.1 (+1.1)
        np.testing.assert_array_almost_equal(result[0].index, [-0.5, 0.2, 1.1])

    def test_align_ts_with_epochs(self):
        """Events outside epochs should be excluded."""
        spikes = nap.Ts(t=[0.5, 1.2, 2.1, 3.5, 4.8])
        events = nap.Ts(t=[1.0, 3.0, 5.0])
        epochs = nap.IntervalSet(start=[0, 2.5], end=[2.4, 4.5])
        result = nap.compute_perievent(spikes, events, window=1.0, epochs=epochs)

        # Only 2 events fall within the epochs (t=1.0 and t=3.0)
        assert len(result) == 2

    def test_align_tsgroup(self):
        """TsGroup input should return a dict of TsGroups."""
        tsgroup = nap.TsGroup(
            {0: nap.Ts(t=np.arange(0, 100)), 1: nap.Ts(t=np.arange(0, 200))}
        )
        tref = nap.Ts(t=np.arange(10, 100, 10))
        peth = nap.compute_perievent(tsgroup, tref, window=(-10, 10))

        assert isinstance(peth, dict)
        assert list(tsgroup.keys()) == list(peth.keys())
        for i in peth.keys():
            assert len(peth[i]) == len(tref)
            # Check a central event (t=50) that has a full window on both sides.
            central = peth[i][4]  # tref[4] = 50
            np.testing.assert_array_almost_equal(central.index, np.arange(-10, 11))

    def test_time_units_discrete(self):
        """Window specified in ms/us should produce the same result as in seconds."""
        tsd = nap.Ts(t=np.arange(100))
        tref = nap.Ts(t=np.arange(10, 100, 10))
        for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
            peth = nap.compute_perievent(
                tsd, tref, window=(-10 * fa, 10 * fa), time_unit=tu
            )
            # Check a central event (t=50) that has a full window on both sides.
            central = peth[4]  # tref[4] = 50
            np.testing.assert_array_almost_equal(central.index, np.arange(-10, 11))


# =============================================================================
# compute_perievent — continuous data (Tsd / TsdFrame / TsdTensor input)
# =============================================================================


class TestComputePerieventContinuous:
    def test_tsd_returns_tsdf_rame(self):
        """Regularly-sampled Tsd → TsdFrame with one column per event."""
        times = np.arange(0, 10, 0.5)
        values = np.sin(times)
        data = nap.Tsd(t=times, d=values)
        events = nap.Ts(t=[2.0, 5.0, 8.0])
        result = nap.compute_perievent(data, events, window=1.0)

        assert isinstance(result, nap.TsdFrame)
        assert result.shape[1] == len(events)

    def test_tsd_values_correct(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        tref = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, tref, window=window)

        assert isinstance(pe, nap.TsdFrame)
        assert pe.shape[1] == len(tref)
        np.testing.assert_array_almost_equal(
            pe.index.values, np.arange(window[0], window[1] + 1)
        )
        expected = np.array(
            [np.arange(t + window[0], t + window[1] + 1) for t in tref.t]
        ).T
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_tsd_symmetric_window(self):
        """Single-number window interpreted as symmetric."""
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        tref = nap.Ts(t=np.array([20, 60]))
        window = 5
        pe = nap.compute_perievent(tsd, tref, window=window)

        assert isinstance(pe, nap.TsdFrame)
        np.testing.assert_array_almost_equal(
            pe.index.values, np.arange(-window, window + 1)
        )
        expected = np.array([np.arange(t - window, t + window + 1) for t in tref.t]).T
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_tsdframe_returns_tsdtensor(self):
        """Regularly-sampled TsdFrame → TsdTensor."""
        tsd = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 3))
        tref = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, tref, window=window)

        assert isinstance(pe, nap.TsdTensor)
        assert pe.d.ndim == 3
        assert pe.shape[1:] == (len(tref), tsd.shape[1])
        np.testing.assert_array_almost_equal(
            pe.index.values, np.arange(window[0], window[1] + 1)
        )
        expected = np.zeros(pe.shape)
        for i, t in enumerate(tref.t):
            idx = np.where(tsd.t == t)[0][0]
            expected[:, i, :] = tsd.values[idx + window[0] : idx + window[1] + 1]
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_tsdtensor_returns_tsdtensor(self):
        """Regularly-sampled TsdTensor → TsdTensor with extra dimensions."""
        tsd = nap.TsdTensor(t=np.arange(100), d=np.random.randn(100, 3, 4))
        tref = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, tref, window=window)

        assert isinstance(pe, nap.TsdTensor)
        assert pe.d.ndim == 4
        assert pe.shape[1:] == (len(tref), *tsd.shape[1:])

    def test_continuous_time_units(self):
        """Window in ms/us should yield the same result as in seconds."""
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        tref = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        expected = np.array(
            [np.arange(t + window[0], t + window[1] + 1) for t in tref.t]
        ).T
        for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
            pe = nap.compute_perievent(
                tsd, tref, window=(window[0] * fa, window[1] * fa), time_unit=tu
            )
            np.testing.assert_array_almost_equal(
                pe.index.values, np.arange(window[0], window[1] + 1)
            )
            np.testing.assert_array_almost_equal(pe.values, expected)

    def test_continuous_with_epochs(self):
        """Events outside epochs should be excluded from the output columns."""
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        tref = nap.Ts(t=np.array([10, 50, 80]))
        window = (-5, 10)
        ep = nap.IntervalSet(start=[0, 60], end=[40, 99])
        pe = nap.compute_perievent(tsd, tref, window=window, epochs=ep)

        # t=50 falls outside the epochs, so only 2 events remain
        assert pe.shape[1] == len(tref) - 1
        expected = np.array(
            [np.arange(t + window[0], t + window[1] + 1) for t in tref.restrict(ep).t]
        ).T
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_continuous_epoch_at_boundary_start(self):
        """Events at the start of an epoch: indices before the boundary should be NaN."""
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        ep = nap.IntervalSet(start=[0, 60], end=[40, 99])
        tref = nap.Ts(ep.start)
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, tref, window=window, epochs=ep)

        expected = np.array(
            [np.arange(t, t + window[1] + 1) for t in tref.restrict(ep).t]
        ).T
        np.testing.assert_array_almost_equal(pe.values[abs(window[0]) :], expected)

    def test_continuous_epoch_at_boundary_end(self):
        """Events at the end of an epoch: indices after the boundary should be NaN."""
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        ep = nap.IntervalSet(start=[0, 60], end=[40, 99])
        tref = nap.Ts(ep.end)
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, tref, window=window, epochs=ep)

        expected = np.array(
            [np.arange(t + window[0], t + 1) for t in tref.restrict(ep).t]
        ).T
        np.testing.assert_array_almost_equal(pe.values[: -abs(window[1])], expected)

    def test_continuous_events_outside_data_all_nan(self):
        """If all events fall outside the data range, output should be all NaN."""
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        ep = nap.IntervalSet(start=[100], end=[200])
        tref = nap.Ts(t=np.array([120, 150, 180]))
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, tref, window=window, epochs=ep)

        assert np.all(np.isnan(pe.values))

    def test_irregular_sampling_raises(self):
        """Irregularly-sampled continuous data should raise RuntimeError."""
        irregular_times = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        tsd = nap.Tsd(t=irregular_times, d=np.arange(5))
        tref = nap.Ts(t=[0.3])
        with pytest.raises(RuntimeError, match="regularly sampled"):
            nap.compute_perievent(tsd, tref, window=0.2)


# =============================================================================
# compute_perievent — error handling
# =============================================================================


class TestComputePerieventErrors:

    # --- time_unit ---

    @pytest.mark.parametrize(
        "bad_unit",
        [
            "a",
            "S",
            "",
            1,
            1.0,
            None,
            ["s"],
            ("ms",),
        ],
    )
    def test_invalid_time_unit(self, bad_unit):
        tsd = nap.Ts(t=np.arange(100))
        tref = nap.Ts(t=np.arange(10, 100, 10))
        with pytest.raises((RuntimeError, TypeError)):
            nap.compute_perievent(tsd, tref, window=10, time_unit=bad_unit)

    # --- window ---

    @pytest.mark.parametrize(
        "bad_window",
        [
            (1, 2, 3),
            (1,),
            (1, "2"),
            ("1", 2),
            ("1", "2"),
            {0: 1},
            "10",
        ],
    )
    def test_invalid_window(self, bad_window):
        tsd = nap.Ts(t=np.arange(100))
        tref = nap.Ts(t=np.arange(10, 100, 10))
        with pytest.raises((RuntimeError, TypeError)):
            nap.compute_perievent(tsd, tref, window=bad_window)

    # --- data ---

    @pytest.mark.parametrize(
        "bad_data",
        [
            np.arange(10),
            {"t": np.arange(10)},
            42,
            "data",
            (0, 1, 2),
        ],
    )
    def test_invalid_data_type(self, bad_data):
        tref = nap.Ts(t=np.arange(10, 100, 10))
        with pytest.raises(TypeError, match="data should be a time series object"):
            nap.compute_perievent(bad_data, tref, window=10)

    # --- events ---

    @pytest.mark.parametrize(
        "bad_events",
        [
            np.array([1.0, 2.0, 3.0]),
            {"t": np.array([1.0, 2.0])},
            42,
            "events",
            (1.0, 2.0),
        ],
    )
    def test_invalid_events_type(self, bad_events):
        tsd = nap.Ts(t=np.arange(100))
        with pytest.raises(TypeError, match="events should be a time series object"):
            nap.compute_perievent(tsd, bad_events, window=10)


# =============================================================================
# compute_event_triggered_average
# =============================================================================


class TestComputeEventTriggeredAverage:
    def test_tsd_single_event_train(self):
        """ETA of a Tsd against a single Ts should return a TsdFrame with one column."""
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])
        eta = nap.compute_event_triggered_average(
            feature, events, binsize=0.1, window=0.5
        )

        assert isinstance(eta, nap.TsdFrame)
        assert eta.shape[1] == 1
        # Time axis should span [-0.5, +0.5] in binsize steps
        expected_times = np.round(np.arange(-0.5, 0.6, 0.1), 9)
        np.testing.assert_array_almost_equal(
            eta.index.values, expected_times, decimal=5
        )

    def test_tsd_tsgroup_events(self):
        """ETA with TsGroup events should return one column per unit."""
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        unit1 = nap.Ts(t=[1.0, 3.0, 5.0])
        unit2 = nap.Ts(t=[2.0, 4.0, 6.0])
        spikes = nap.TsGroup({0: unit1, 1: unit2})
        eta = nap.compute_event_triggered_average(
            feature, spikes, binsize=0.1, window=0.5
        )

        assert isinstance(eta, nap.TsdFrame)
        assert eta.shape[1] == 2
        # Columns should be labelled with unit indices
        assert list(eta.columns) == [0, 1]

    def test_tsdframe_returns_tsdtensor(self):
        """ETA of a TsdFrame against events should return a TsdTensor."""
        times = np.arange(0, 10, 0.1)
        values = np.column_stack([np.sin(times), np.cos(times)])
        features = nap.TsdFrame(t=times, d=values, columns=["sin", "cos"])
        events = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])
        eta = nap.compute_event_triggered_average(
            features, events, binsize=0.1, window=0.5
        )

        assert isinstance(eta, nap.TsdTensor)
        assert eta.shape == (11, 1, 2)

    def test_asymmetric_window(self):
        """Asymmetric window (-0.3, 0.5) should produce a correctly sized time axis."""
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[2.0, 5.0, 8.0])
        eta = nap.compute_event_triggered_average(
            feature, events, binsize=0.1, window=(-0.3, 0.5)
        )

        assert isinstance(eta, nap.TsdFrame)
        # Expect bins from -0.3 to +0.5 inclusive: -0.3, -0.2, -0.1, 0, 0.1, ..., 0.5 → 9 bins
        assert eta.shape[0] == 9

    def test_with_epochs(self):
        """Restricting computation to epochs should change the ETA values."""
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])
        epochs = nap.IntervalSet(start=[0, 5], end=[4, 9])

        eta_full = nap.compute_event_triggered_average(
            feature, events, binsize=0.1, window=0.5
        )
        eta_ep = nap.compute_event_triggered_average(
            feature, events, binsize=0.1, window=0.5, epochs=epochs
        )

        assert isinstance(eta_ep, nap.TsdFrame)
        # The values should differ because the epoch excludes some events
        assert not np.allclose(eta_full.values, eta_ep.values)

    def test_time_units(self):
        """Window and binsize in ms should give the same result as in seconds."""
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])

        eta_s = nap.compute_event_triggered_average(
            feature, events, binsize=0.1, window=0.5, time_unit="s"
        )
        eta_ms = nap.compute_event_triggered_average(
            feature, events, binsize=100.0, window=500.0, time_unit="ms"
        )

        np.testing.assert_array_almost_equal(eta_s.index.values, eta_ms.index.values)
        np.testing.assert_array_almost_equal(eta_s.values, eta_ms.values)

    def test_irregular_sampling_raises(self):
        """Irregularly-sampled data should raise RuntimeError."""
        irregular_times = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1])
        feature = nap.Tsd(t=irregular_times, d=np.arange(7))
        events = nap.Ts(t=[1.0])
        with pytest.raises(RuntimeError, match="regularly sampled"):
            nap.compute_event_triggered_average(
                feature, events, binsize=0.1, window=0.3
            )

    # --- time_unit ---

    @pytest.mark.parametrize(
        "bad_unit",
        [
            "x",
            "MS",
            "",
            1,
            0.5,
            None,
            ["s"],
            ("ms",),
        ],
    )
    def test_invalid_time_unit(self, bad_unit):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[1.0, 3.0])
        with pytest.raises((RuntimeError, TypeError)):
            nap.compute_event_triggered_average(
                feature, events, binsize=0.1, window=0.5, time_unit=bad_unit
            )

    # --- window ---

    @pytest.mark.parametrize(
        "bad_window",
        [
            (0.1, 0.2, 0.3),
            (0.1,),
            (0.1, "0.2"),
            ("0.1", 0.2),
            ("0.1", "0.2"),
            {0: 0.1},
            "0.5",
        ],
    )
    def test_invalid_window_format(self, bad_window):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[1.0, 3.0])
        with pytest.raises((RuntimeError, TypeError)):
            nap.compute_event_triggered_average(
                feature, events, binsize=0.1, window=bad_window
            )

    # --- data ---

    @pytest.mark.parametrize(
        "bad_data",
        [
            np.arange(10),
            {"t": np.arange(10)},
            42,
            "data",
            (0, 1, 2),
            nap.Ts(t=[1.0, 2.0, 3.0]),
        ],
    )
    def test_invalid_data_type(self, bad_data):
        events = nap.Ts(t=[1.0, 3.0])
        with pytest.raises(TypeError, match="data should be a continuous time series"):
            nap.compute_event_triggered_average(
                bad_data, events, binsize=0.1, window=0.5
            )

    # --- events ---

    @pytest.mark.parametrize(
        "bad_events",
        [
            np.array([1.0, 2.0, 3.0]),
            {"t": np.array([1.0, 2.0])},
            42,
            "events",
            (1.0, 2.0),
        ],
    )
    def test_invalid_events_type(self, bad_events):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        with pytest.raises(TypeError, match="events should be a time series object"):
            nap.compute_event_triggered_average(
                feature, bad_events, binsize=0.1, window=0.5
            )


# =============================================================================
# compute_spike_triggered_average (alias)
# =============================================================================


class TestComputeSpikeTriggeredAverage:
    def test_is_alias(self):
        """compute_spike_triggered_average should produce identical results to compute_event_triggered_average."""
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        spikes = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])

        eta = nap.compute_event_triggered_average(
            feature, spikes, binsize=0.1, window=0.5
        )
        sta = nap.compute_spike_triggered_average(
            feature, spikes, binsize=0.1, window=0.5
        )

        np.testing.assert_array_equal(eta.index.values, sta.index.values)
        np.testing.assert_array_equal(eta.values, sta.values)
