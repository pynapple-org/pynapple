"""Tests of perievent for `pynapple` package."""

import numpy as np
import pytest

import pynapple as nap
from pynapple.process.perievent import _align_sorted

# =============================================================================
# compute_perievent — discrete data (Ts input)
# =============================================================================


class TestComputePerieventDiscrete:
    def test_align_ts(self):
        tsd = nap.Ts(t=np.arange(100))
        events = nap.Ts(t=np.arange(10, 100, 10))
        peth = nap.compute_perievent(tsd, events, window=(-10, 10))

        assert len(peth) == len(events)
        assert isinstance(peth, nap.TsGroup)
        np.testing.assert_array_almost_equal(peth[4].index, np.arange(-10, 11))

    def test_align_ts_empty(self):
        tsd = nap.Ts(t=np.arange(100))
        events = nap.Ts(t=np.arange(200, 300, 10))
        peth = nap.compute_perievent(tsd, events, window=(-10, 10))

        assert len(peth) == 0
        assert isinstance(peth, nap.TsGroup)
        np.testing.assert_array_almost_equal(
            peth.time_support.values, nap.IntervalSet(-10, 10).values
        )

    def test_align_ts_symmetric_window(self):
        tsd = nap.Ts(t=np.arange(100))
        events = nap.Ts(t=np.arange(10, 100, 10))
        peth = nap.compute_perievent(tsd, events, window=10.0)

        assert isinstance(peth, nap.TsGroup)
        assert len(peth) == len(events)
        np.testing.assert_array_almost_equal(peth[4].index, np.arange(-10, 11))

    def test_align_ts_asymmetric_window(self):
        tsd = nap.Ts(t=[0.5, 1.2, 2.1, 3.5])
        events = nap.Ts(t=[1.0, 3.0])
        result = nap.compute_perievent(tsd, events, window=(-0.5, 1.5))

        np.testing.assert_array_almost_equal(result[0].index, [-0.5, 0.2, 1.1])

    def test_align_ts_with_epochs(self):
        tsd = nap.Ts(t=[0.5, 1.2, 2.1, 3.5, 4.8])
        events = nap.Ts(t=[1.0, 3.0, 5.0])
        epochs = nap.IntervalSet(start=[0, 2.5], end=[2.4, 4.5])
        result = nap.compute_perievent(tsd, events, window=1.0, epochs=epochs)

        assert len(result) == 2

    def test_align_ts_metadata(self):
        tsd = nap.Ts(t=[0.5, 1.2, 2.1, 3.5, 4.8])
        events = nap.Ts(t=[1.0, 3.0, 5.0])
        result = nap.compute_perievent(tsd, events, window=1.0)

        assert "events" in result.metadata.columns
        np.testing.assert_array_almost_equal(
            result.metadata["events"], events.restrict(tsd.time_support).t
        )

    def test_align_ts_metadata_respects_epochs(self):
        tsd = nap.Ts(t=[0.5, 1.2, 2.1, 3.5, 4.8])
        events = nap.Ts(t=[1.0, 3.0, 5.0])
        epochs = nap.IntervalSet(start=[0, 2.5], end=[2.4, 4.5])
        result = nap.compute_perievent(tsd, events, window=1.0, epochs=epochs)

        assert "events" in result.metadata.columns
        np.testing.assert_array_almost_equal(
            result.metadata["events"], events.restrict(epochs).t
        )

    def test_align_tsgroup(self):
        tsgroup = nap.TsGroup(
            {0: nap.Ts(t=np.arange(0, 100)), 1: nap.Ts(t=np.arange(0, 200))}
        )
        events = nap.Ts(t=np.arange(10, 100, 10))
        peth = nap.compute_perievent(tsgroup, events, window=(-10, 10))

        assert isinstance(peth, dict)
        assert list(tsgroup.keys()) == list(peth.keys())
        for i in peth.keys():
            assert len(peth[i]) == len(events)
            np.testing.assert_array_almost_equal(peth[i][4].index, np.arange(-10, 11))

    def test_align_tsgroup_empty(self):
        tsgroup = nap.TsGroup(
            {0: nap.Ts(t=np.arange(0, 100)), 1: nap.Ts(t=np.arange(0, 200))}
        )
        events = nap.Ts(t=np.arange(300, 400, 10))
        peth = nap.compute_perievent(tsgroup, events, window=(-10, 10))

        assert isinstance(peth, dict)
        assert list(tsgroup.keys()) == list(peth.keys())

        for i in peth.keys():
            assert len(peth[i]) == 0
            np.testing.assert_array_almost_equal(
                peth[i].time_support.values, nap.IntervalSet(-10, 10).values
            )

    def test_align_tsgroup_metadata(self):
        unit1 = nap.Ts(t=[0.5, 1.2, 2.1, 3.5])
        unit2 = nap.Ts(t=[0.8, 1.5, 3.2, 4.1])
        tsgroup = nap.TsGroup({0: unit1, 1: unit2})
        events = nap.Ts(t=[1.0, 3.0, 4.0])
        result = nap.compute_perievent(tsgroup, events, window=0.5)

        assert isinstance(result, dict)
        for unit_id, aligned in result.items():
            assert isinstance(aligned, nap.TsGroup)
            assert "events" in aligned.metadata.columns
            np.testing.assert_array_almost_equal(
                aligned.metadata["events"],
                events.restrict(tsgroup[unit_id].time_support).t,
            )

    def test_align_tsgroup_metadata_respects_epochs(self):
        tsgroup = nap.TsGroup(
            {0: nap.Ts(t=np.arange(0, 10)), 1: nap.Ts(t=np.arange(0, 10))}
        )
        events = nap.Ts(t=[1.0, 3.0, 7.0])
        epochs = nap.IntervalSet(start=[0], end=[5])
        result = nap.compute_perievent(tsgroup, events, window=0.5, epochs=epochs)

        assert isinstance(result, dict)
        for aligned in result.values():
            assert "events" in aligned.metadata.columns
            np.testing.assert_array_almost_equal(
                aligned.metadata["events"], events.restrict(epochs).t
            )

    def test_time_units(self):
        tsd = nap.Ts(t=np.arange(100))
        events = nap.Ts(t=np.arange(10, 100, 10))
        for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
            peth = nap.compute_perievent(
                tsd, events, window=(-10 * fa, 10 * fa), time_unit=tu
            )
            np.testing.assert_array_almost_equal(peth[4].index, np.arange(-10, 11))


# =============================================================================
# compute_perievent — discrete data, against the per-event loop
# =============================================================================


def align_by_loop(ts, events, window, time_unit="s", epochs=None):
    """The former implementation of ``compute_perievent`` for a ``Ts``: one
    mask over all the spikes for each event. It is the reference for the
    sorted-array kernel."""
    if not isinstance(window, tuple):
        window = (window, window)
    if epochs is None:
        epochs = ts.time_support
    window = np.abs(nap.TsIndex.format_timestamps(np.array(window), time_unit))
    event_times = events.restrict(epochs).times()
    support = nap.IntervalSet(start=-window[0], end=window[1])
    aligned = {}
    for i, event in enumerate(event_times):
        mask = (ts.t >= event - window[0]) & (ts.t <= event + window[1])
        aligned[i] = nap.Ts(ts.t[mask] - event, time_support=support)
    return nap.TsGroup(aligned, metadata={"events": event_times}, time_support=support)


def assert_same_peth(a, b):
    """Check that two outputs of ``compute_perievent`` for spikes are equal."""
    assert isinstance(a, nap.TsGroup) and isinstance(b, nap.TsGroup)
    assert list(a) == list(b)
    np.testing.assert_array_equal(a.time_support.values, b.time_support.values)
    np.testing.assert_array_equal(a["events"], b["events"])
    np.testing.assert_array_equal(a.rates, b.rates)
    for i in a:
        np.testing.assert_array_equal(a[i].t, b[i].t)


def random_case(seed):
    """A Ts, a TsGroup with sparse keys and an empty unit, and events. Some
    events are outside the data and some windows overlap."""
    rng = np.random.default_rng(seed)
    support = nap.IntervalSet(0, 50)
    ts = nap.Ts(np.sort(rng.uniform(0, 50, 500)), time_support=support)
    group = nap.TsGroup(
        {
            3: ts,
            10**9: nap.Ts(np.sort(rng.uniform(0, 50, 200))),
            12: nap.Ts(np.array([])),
        },
        time_support=support,
    )
    events = nap.Ts(np.sort(rng.uniform(-5, 55, 40)))
    return ts, group, events


WINDOWS = [
    ((0.5, 1.5), "s"),
    (2.0, "s"),
    ((500.0, 1500.0), "ms"),
    ((0.0, 1.0), "s"),
    (0.0, "s"),
]


class TestComputePerieventSorted:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("window, time_unit", WINDOWS)
    @pytest.mark.parametrize("epochs", [None, nap.IntervalSet([5, 30], [20, 45])])
    def test_same_as_loop(self, seed, window, time_unit, epochs):
        ts, group, events = random_case(seed)
        assert_same_peth(
            nap.compute_perievent(ts, events, window, time_unit, epochs),
            align_by_loop(ts, events, window, time_unit, epochs),
        )
        peth = nap.compute_perievent(group, events, window, time_unit, epochs)
        assert list(peth) == list(group)
        for k in group:
            assert_same_peth(
                peth[k], align_by_loop(group[k], events, window, time_unit, epochs)
            )

    def test_spikes_on_window_bounds(self):
        """``t - event`` can fall just outside the window (1.4 - 1.7 gives
        -0.30000000000000004). Such spikes are dropped, as by a Ts on the
        window."""
        ts = nap.Ts(np.round(np.arange(0, 10, 0.1), 10))
        events = nap.Ts(np.array([0.3, 1.7, 2.9, 5.0]))
        peth = nap.compute_perievent(ts, events, (0.3, 0.7))
        assert_same_peth(peth, align_by_loop(ts, events, (0.3, 0.7)))
        assert -0.3 not in peth[1].t  # 1.4 - 1.7
        group = nap.TsGroup({0: ts, 1: nap.Ts(ts.t[::3])})
        for k, aligned in nap.compute_perievent(group, events, (0.3, 0.7)).items():
            assert_same_peth(aligned, align_by_loop(group[k], events, (0.3, 0.7)))

    def test_overlapping_windows(self):
        """A spike in two windows is aligned to each event."""
        ts = nap.Ts(np.array([5.0]), time_support=nap.IntervalSet(0, 10))
        peth = nap.compute_perievent(ts, nap.Ts(np.array([4.0, 5.5])), 1.0)
        np.testing.assert_array_equal(peth[0].t, [1.0])
        np.testing.assert_array_equal(peth[1].t, [-0.5])

    def test_spikes_outside_epochs(self):
        """``epochs`` restricts the events, not the spikes: a spike outside
        ``epochs`` but in a window is aligned."""
        ts = nap.Ts(np.array([4.5, 5.5]), time_support=nap.IntervalSet(0, 10))
        events = nap.Ts(np.array([5.0, 8.0]))
        epochs = nap.IntervalSet(5, 6)
        for peth in [
            nap.compute_perievent(ts, events, 1.0, epochs=epochs),
            nap.compute_perievent(nap.TsGroup({0: ts}), events, 1.0, epochs=epochs)[0],
        ]:
            assert list(peth) == [0]
            np.testing.assert_array_equal(peth[0].t, [-0.5, 0.5])

    def test_no_event(self):
        """No event in the support: one empty group per unit, with an int64
        index, as with events."""
        ts = nap.Ts(np.arange(10.0))
        events = nap.Ts(np.array([50.0]))
        for peth in [
            nap.compute_perievent(ts, events, 1.0),
            nap.compute_perievent(nap.TsGroup({0: ts}), events, 1.0)[0],
        ]:
            assert len(peth) == 0
            assert peth.index.dtype == np.int64
            np.testing.assert_array_equal(peth.time_support.values, [[-1.0, 1.0]])

    def test_empty_group(self):
        group = nap.TsGroup({}, time_support=nap.IntervalSet(0, 10))
        assert nap.compute_perievent(group, nap.Ts(np.array([5.0])), 1.0) == {}

    def test_group_of_tsd(self):
        """Members with values align as continuous data, one unit at a time."""
        group = nap.TsGroup(
            {
                0: nap.Tsd(t=np.arange(100.0), d=np.arange(100.0)),
                1: nap.Tsd(t=np.arange(100.0), d=-np.arange(100.0)),
            }
        )
        events = nap.Ts(np.array([20.0, 60.0]))
        peth = nap.compute_perievent(group, events, 5.0)
        for k in group:
            assert isinstance(peth[k], nap.TsdFrame)
            np.testing.assert_array_equal(
                peth[k].values, nap.compute_perievent(group[k], events, 5.0).values
            )


class TestAlignSorted:
    def test_order_and_offsets(self):
        """Sorted by unit, then by relative time, then by event."""
        times = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        unit_pos = np.array([1, 0, 1, 1, 0])
        event_times = np.array([2.0, 3.0])
        relative_times, event_idx, offsets = _align_sorted(
            times, unit_pos, 2, event_times, np.array([1.0, 1.0])
        )
        np.testing.assert_array_equal(offsets, [0, 3, 8])
        # unit 0: spikes 2.0 and 4.0
        np.testing.assert_array_equal(relative_times[:3], [-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(event_idx[:3], [1, 0, 1])
        # unit 1: spikes 1.0, 2.0 and 3.0; the ties keep the event order
        np.testing.assert_array_equal(relative_times[3:], [-1.0, -1.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(event_idx[3:], [0, 1, 0, 1, 0])

    @pytest.mark.parametrize(
        "times, unit_pos, n_units, event_times",
        [
            ([1.0, 2.0], [0, 1], 2, []),  # no event
            ([], [], 0, [5.0]),  # no unit
            ([1.0, 2.0], [0, 1], 2, [50.0]),  # no spike in the windows
        ],
    )
    def test_empty(self, times, unit_pos, n_units, event_times):
        relative_times, event_idx, offsets = _align_sorted(
            np.array(times, dtype=np.float64),
            np.array(unit_pos, dtype=np.int64),
            n_units,
            np.array(event_times, dtype=np.float64),
            np.array([1.0, 1.0]),
        )
        assert len(relative_times) == len(event_idx) == 0
        np.testing.assert_array_equal(offsets, np.zeros(n_units + 1))


# =============================================================================
# compute_perievent — continuous data (Tsd / TsdFrame / TsdTensor input)
# =============================================================================


class TestComputePerieventContinuous:
    def test_tsd_returns_tsdframe(self):
        data = nap.Tsd(t=np.arange(0, 10, 0.5), d=np.sin(np.arange(0, 10, 0.5)))
        events = nap.Ts(t=[2.0, 5.0, 8.0])
        result = nap.compute_perievent(data, events, window=1.0)

        assert isinstance(result, nap.TsdFrame)
        assert result.shape[1] == len(events)

    def test_tsd_values_correct(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        events = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, events, window=window)

        assert isinstance(pe, nap.TsdFrame)
        assert pe.shape[1] == len(events)
        np.testing.assert_array_almost_equal(
            pe.index.values, np.arange(window[0], window[1] + 1)
        )
        expected = np.array(
            [np.arange(t + window[0], t + window[1] + 1) for t in events.t]
        ).T
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_tsd_symmetric_window(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        events = nap.Ts(t=np.array([20, 60]))
        window = 5
        pe = nap.compute_perievent(tsd, events, window=window)

        assert isinstance(pe, nap.TsdFrame)
        np.testing.assert_array_almost_equal(
            pe.index.values, np.arange(-window, window + 1)
        )
        expected = np.array([np.arange(t - window, t + window + 1) for t in events.t]).T
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_tsdframe_returns_tsdtensor(self):
        tsd = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 3))
        events = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, events, window=window)

        assert isinstance(pe, nap.TsdTensor)
        assert pe.d.ndim == 3
        assert pe.shape[1:] == (len(events), tsd.shape[1])
        np.testing.assert_array_almost_equal(
            pe.index.values, np.arange(window[0], window[1] + 1)
        )
        expected = np.zeros(pe.shape)
        for i, t in enumerate(events.t):
            idx = np.where(tsd.t == t)[0][0]
            expected[:, i, :] = tsd.values[idx + window[0] : idx + window[1] + 1]
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_tsdtensor_returns_tsdtensor(self):
        tsd = nap.TsdTensor(t=np.arange(100), d=np.random.randn(100, 3, 4))
        events = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, events, window=window)

        assert isinstance(pe, nap.TsdTensor)
        assert pe.d.ndim == 4
        assert pe.shape[1:] == (len(events), *tsd.shape[1:])

    def test_time_units(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        events = nap.Ts(t=np.array([20, 60]))
        window = (-5, 10)
        expected = np.array(
            [np.arange(t + window[0], t + window[1] + 1) for t in events.t]
        ).T
        for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
            pe = nap.compute_perievent(
                tsd, events, window=(window[0] * fa, window[1] * fa), time_unit=tu
            )
            np.testing.assert_array_almost_equal(
                pe.index.values, np.arange(window[0], window[1] + 1)
            )
            np.testing.assert_array_almost_equal(pe.values, expected)

    def test_with_epochs(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        events = nap.Ts(t=np.array([10, 50, 80]))
        window = (-5, 10)
        epochs = nap.IntervalSet(start=[0, 60], end=[40, 99])
        pe = nap.compute_perievent(tsd, events, window=window, epochs=epochs)

        assert pe.shape[1] == len(events) - 1
        expected = np.array(
            [
                np.arange(t + window[0], t + window[1] + 1)
                for t in events.restrict(epochs).t
            ]
        ).T
        np.testing.assert_array_almost_equal(pe.values, expected)

    def test_epoch_at_boundary_start(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        epochs = nap.IntervalSet(start=[0, 60], end=[40, 99])
        events = nap.Ts(epochs.start)
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, events, window=window, epochs=epochs)

        expected = np.array(
            [np.arange(t, t + window[1] + 1) for t in events.restrict(epochs).t]
        ).T
        np.testing.assert_array_almost_equal(pe.values[abs(window[0]) :], expected)

    def test_epoch_at_boundary_end(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        epochs = nap.IntervalSet(start=[0, 60], end=[40, 99])
        events = nap.Ts(epochs.end)
        window = (-5, 10)
        pe = nap.compute_perievent(tsd, events, window=window, epochs=epochs)

        expected = np.array(
            [np.arange(t + window[0], t + 1) for t in events.restrict(epochs).t]
        ).T
        np.testing.assert_array_almost_equal(pe.values[: -abs(window[1])], expected)

    def test_events_outside_data_raises(self):
        tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
        epochs = nap.IntervalSet(start=[100], end=[200])
        events = nap.Ts(t=np.array([120, 150, 180]))
        with pytest.raises(ValueError, match="No overlap"):
            nap.compute_perievent(tsd, events, window=(-5, 10), epochs=epochs)

    def test_irregular_sampling_raises(self):
        tsd = nap.Tsd(t=np.array([0.0, 0.1, 0.3, 0.6, 1.0]), d=np.arange(5))
        with pytest.raises(RuntimeError, match="regularly sampled"):
            nap.compute_perievent(tsd, nap.Ts(t=[0.3]), window=0.2)


# =============================================================================
# compute_perievent — error handling
# =============================================================================


class TestComputePerieventErrors:
    @pytest.mark.parametrize("bad_unit", ["a", "S", "", 1, 1.0, None, ["s"], ("ms",)])
    def test_invalid_time_unit(self, bad_unit):
        tsd = nap.Ts(t=np.arange(100))
        events = nap.Ts(t=np.arange(10, 100, 10))
        with pytest.raises((RuntimeError, TypeError)):
            nap.compute_perievent(tsd, events, window=10, time_unit=bad_unit)

    @pytest.mark.parametrize(
        "bad_window", [(1, 2, 3), (1,), (1, "2"), ("1", 2), ("1", "2"), {0: 1}, "10"]
    )
    def test_invalid_window(self, bad_window):
        tsd = nap.Ts(t=np.arange(100))
        events = nap.Ts(t=np.arange(10, 100, 10))
        with pytest.raises((RuntimeError, TypeError)):
            nap.compute_perievent(tsd, events, window=bad_window)

    @pytest.mark.parametrize(
        "bad_data", [np.arange(10), {"t": np.arange(10)}, 42, "data", (0, 1, 2)]
    )
    def test_invalid_data_type(self, bad_data):
        events = nap.Ts(t=np.arange(10, 100, 10))
        with pytest.raises(TypeError, match="data should be a time series object"):
            nap.compute_perievent(bad_data, events, window=10)

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
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])
        eta = nap.compute_event_triggered_average(
            feature, events, binsize=0.1, window=0.5
        )

        assert isinstance(eta, nap.TsdFrame)
        assert eta.shape[1] == 1
        np.testing.assert_array_almost_equal(
            eta.index.values, np.round(np.arange(-0.5, 0.6, 0.1), 9), decimal=5
        )

    def test_tsd_tsgroup_events(self):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        spikes = nap.TsGroup(
            {0: nap.Ts(t=[1.0, 3.0, 5.0]), 1: nap.Ts(t=[2.0, 4.0, 6.0])}
        )
        eta = nap.compute_event_triggered_average(
            feature, spikes, binsize=0.1, window=0.5
        )

        assert isinstance(eta, nap.TsdFrame)
        assert eta.shape[1] == 2
        assert list(eta.columns) == [0, 1]

    @pytest.mark.parametrize("time_units, scale", [("s", 1.0), ("ms", 1_000.0)])
    def test_intervalset_events_use_interval_starts(self, time_units, scale):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        intervals = nap.IntervalSet(
            start=np.array([1.0, 3.0, 5.0]) * scale,
            end=np.array([1.2, 3.8, 5.4]) * scale,
            time_units=time_units,
        )

        actual = nap.compute_event_triggered_average(
            feature, intervals, binsize=0.1, window=0.5
        )
        expected = nap.compute_event_triggered_average(
            feature, nap.Ts(intervals.start), binsize=0.1, window=0.5
        )

        np.testing.assert_array_equal(actual.index.values, expected.index.values)
        np.testing.assert_array_equal(actual.values, expected.values)

    def test_intervalset_events_with_epochs(self):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        intervals = nap.IntervalSet(
            start=[1.0, 3.0, 5.0, 7.0], end=[1.2, 3.4, 5.6, 7.8]
        )
        epochs = nap.IntervalSet(start=[0.0, 6.0], end=[2.0, 9.0])

        actual = nap.compute_event_triggered_average(
            feature, intervals, binsize=0.1, window=0.5, epochs=epochs
        )
        expected = nap.compute_event_triggered_average(
            feature,
            nap.Ts(intervals.start),
            binsize=0.1,
            window=0.5,
            epochs=epochs,
        )

        np.testing.assert_array_equal(actual.index.values, expected.index.values)
        np.testing.assert_array_equal(actual.values, expected.values)

    def test_tsdframe_returns_tsdtensor(self):
        times = np.arange(0, 10, 0.1)
        features = nap.TsdFrame(
            t=times,
            d=np.column_stack([np.sin(times), np.cos(times)]),
            columns=["sin", "cos"],
        )
        events = nap.Ts(t=[1.0, 3.0, 5.0, 7.0])
        eta = nap.compute_event_triggered_average(
            features, events, binsize=0.1, window=0.5
        )

        assert isinstance(eta, nap.TsdTensor)
        assert eta.shape == (11, 1, 2)

    def test_asymmetric_window(self):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[2.0, 5.0, 8.0])
        eta = nap.compute_event_triggered_average(
            feature, events, binsize=0.1, window=(-0.3, 0.5)
        )

        assert isinstance(eta, nap.TsdFrame)
        assert eta.shape[0] == 9

    def test_with_epochs(self):
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
        assert not np.allclose(eta_full.values, eta_ep.values)

    def test_time_units(self):
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
        feature = nap.Tsd(
            t=np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1]), d=np.arange(7)
        )
        with pytest.raises(RuntimeError, match="regularly sampled"):
            nap.compute_event_triggered_average(
                feature, nap.Ts(t=[1.0]), binsize=0.1, window=0.3
            )

    @pytest.mark.parametrize("bad_unit", ["x", "MS", "", 1, 0.5, None, ["s"], ("ms",)])
    def test_invalid_time_unit(self, bad_unit):
        times = np.arange(0, 10, 0.1)
        feature = nap.Tsd(t=times, d=np.sin(times))
        events = nap.Ts(t=[1.0, 3.0])
        with pytest.raises((RuntimeError, TypeError)):
            nap.compute_event_triggered_average(
                feature, events, binsize=0.1, window=0.5, time_unit=bad_unit
            )

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
        with pytest.raises(
            TypeError, match="events should be a time series object or an IntervalSet"
        ):
            nap.compute_event_triggered_average(
                feature, bad_events, binsize=0.1, window=0.5
            )


# =============================================================================
# compute_spike_triggered_average (alias)
# =============================================================================


class TestComputeSpikeTriggeredAverage:
    def test_is_alias(self):
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
