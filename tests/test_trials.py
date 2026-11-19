"""Tests for trial-aligned objects."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import signal

import pynapple as nap
from pynapple.core.trials import _count_timestamps, _format_axes

REDUCTIONS = [
    np.mean,
    np.nanmean,
    np.sum,
    np.nansum,
    np.std,
    np.nanstd,
    np.var,
    np.nanvar,
    np.min,
    np.nanmin,
    np.max,
    np.nanmax,
    np.median,
    np.nanmedian,
    np.prod,
    np.nanprod,
    np.all,
    np.any,
]


# Fixtures and helpers


def make_source(
    time: np.ndarray,
    values: np.ndarray,
) -> nap.Tsd | nap.TsdFrame | nap.TsdTensor:
    """Construct a time series matching the data dimensions."""
    if values.ndim == 1:
        return nap.Tsd(t=time, d=values)
    if values.ndim == 2:
        return nap.TsdFrame(t=time, d=values)
    return nap.TsdTensor(t=time, d=values)


def scalar_trials(trials: nap.TsdTrials) -> nap.TsdTrials:
    """Select one trailing coordinate while retaining trial and time."""
    if trials.ndim == 2:
        return trials
    if trials.ndim == 3:
        return trials[:, :, trials.columns[0]]
    return trials[(slice(None), slice(None)) + (0,) * (trials.ndim - 2)]


def assert_same_axes(
    actual: nap.TsdTrials,
    expected: nap.TsdTrials,
) -> None:
    """Check axes, validity, and metadata after value-only operations."""
    np.testing.assert_array_equal(actual.t, expected.t)
    np.testing.assert_array_equal(actual.valid, expected.valid)
    np.testing.assert_array_equal(actual.trial_ids, expected.trial_ids)
    np.testing.assert_array_equal(actual.durations, expected.durations)
    np.testing.assert_array_equal(actual.window_starts, expected.window_starts)
    np.testing.assert_array_equal(actual.window_ends, expected.window_ends)

    assert actual.alignment == expected.alignment
    assert actual.sample_period == expected.sample_period
    assert actual.bin_size == expected.bin_size
    assert actual.data_kind == expected.data_kind
    assert actual._windowed == expected._windowed

    np.testing.assert_array_equal(
        actual.trials.values,
        expected.trials.values,
    )
    for column in expected.trials.metadata_columns:
        np.testing.assert_array_equal(
            actual.trials[column],
            expected.trials[column],
        )

    if expected.feature_names is None:
        assert actual.feature_names is None
        assert actual.feature_metadata is None
    else:
        np.testing.assert_array_equal(actual.columns, expected.columns)
        np.testing.assert_array_equal(
            actual.feature_metadata.columns,
            expected.feature_metadata.columns,
        )
        for column in expected.feature_metadata.columns:
            np.testing.assert_array_equal(
                actual.feature_metadata[column],
                expected.feature_metadata[column],
            )


def assert_time_series(
    actual: nap.Tsd | nap.TsdFrame | nap.TsdTensor,
    time: np.ndarray,
    values: np.ndarray,
) -> None:
    """Check the output type, timestamps, and values."""
    expected_type = (
        nap.Tsd
        if values.ndim == 1
        else nap.TsdFrame
        if values.ndim == 2
        else nap.TsdTensor
    )
    assert isinstance(actual, expected_type)
    np.testing.assert_array_equal(actual.t, time)
    np.testing.assert_allclose(actual.d, values, equal_nan=True)


@pytest.fixture
def intervals() -> nap.IntervalSet:
    return nap.IntervalSet(
        start=[0.0, 5.0, 10.0],
        end=[2.0, 6.5, 13.0],
        metadata={
            "condition": ["a", "b", "a"],
            "cue": [0.5, 5.5, 11.0],
        },
    )


@pytest.fixture
def spike_source() -> nap.TsGroup:
    support = nap.IntervalSet(0.0, 14.0)
    return nap.TsGroup(
        {
            10: nap.Ts(
                t=[
                    0.0,
                    0.5,
                    1.0,
                    2.0,
                    4.0,
                    5.0,
                    5.5,
                    6.5,
                    10.0,
                    11.0,
                    12.5,
                    13.0,
                ],
                time_support=support,
            ),
            20: nap.Ts(
                t=[0.25, 1.25, 5.25, 6.25, 10.25, 11.25],
                time_support=support,
            ),
        },
        time_support=support,
        metadata={"region": ["ca1", "ca3"]},
    )


@pytest.fixture
def ts_trials(
    spike_source: nap.TsGroup,
    intervals: nap.IntervalSet,
) -> nap.TsTrials:
    return nap.TsTrials(spike_source, intervals)


@pytest.fixture(params=["scalar", "frame", "matrix", "tensor"])
def sampled_source(
    request: pytest.FixtureRequest,
) -> nap.Tsd | nap.TsdFrame | nap.TsdTensor:
    time = np.arange(57) / 4

    if request.param == "scalar":
        return nap.Tsd(t=time, d=time)

    if request.param == "frame":
        return nap.TsdFrame(
            t=time,
            d=np.column_stack([time, -time]),
            columns=["x", "y"],
            metadata={"region": ["ca1", "ca3"]},
        )

    trailing_shape = (2, 3) if request.param == "matrix" else (2, 1, 3)
    shape = (len(time), *trailing_shape)
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100
    return nap.TsdTensor(t=time, d=values)


@pytest.fixture
def tsd_trials(
    sampled_source: nap.Tsd | nap.TsdFrame | nap.TsdTensor,
    intervals: nap.IntervalSet,
) -> nap.TsdTrials:
    return nap.TsdTrials(
        sampled_source,
        intervals,
        sample_period=0.25,
    )


@pytest.fixture(params=["timestamps", "sampled", "counts"])
def trial_data(
    request: pytest.FixtureRequest,
    ts_trials: nap.TsTrials,
    tsd_trials: nap.TsdTrials,
) -> nap.TsTrials | nap.TsdTrials:
    if request.param == "timestamps":
        return ts_trials
    if request.param == "sampled":
        return tsd_trials
    return ts_trials.count(0.5)


# Timestamp trials


class TestTsTrials:
    def test_create(
        self,
        ts_trials: nap.TsTrials,
        spike_source: nap.TsGroup,
        intervals: nap.IntervalSet,
    ) -> None:
        assert ts_trials.shape == (3, 2)
        assert len(ts_trials) == 3
        assert ts_trials.alignment == "start"

        np.testing.assert_array_equal(ts_trials.trial_ids, intervals.index)
        np.testing.assert_array_equal(ts_trials.unit_keys, [10, 20])
        np.testing.assert_allclose(ts_trials.durations, [2.0, 1.5, 3.0])
        np.testing.assert_array_equal(ts_trials.window_starts, [0.0] * 3)
        np.testing.assert_allclose(ts_trials.window_ends, [2.0, 1.5, 3.0])

        for trial_id, start, end in zip(
            ts_trials.trial_ids,
            intervals.start,
            intervals.end,
        ):
            for unit_key in spike_source.index:
                absolute = spike_source[unit_key].t
                expected = absolute[(absolute >= start) & (absolute <= end)]
                selected = (ts_trials.trial_index == trial_id) & (
                    ts_trials.unit_index == unit_key
                )
                np.testing.assert_allclose(
                    np.sort(ts_trials.timestamps[selected]),
                    expected - start,
                )

    def test_metadata(self, ts_trials: nap.TsTrials) -> None:
        np.testing.assert_array_equal(
            ts_trials.unit_metadata["region"],
            ["ca1", "ca3"],
        )
        assert "rate" not in ts_trials.unit_metadata.columns

    @pytest.mark.parametrize(
        "start, end",
        [
            (-1.0, 1.0),
            (4.0, 8.0),
            (4.0, 11.0),
            (6.0, 7.0),
        ],
    )
    def test_trials_outside_source_support(
        self,
        start: float,
        end: float,
    ) -> None:
        support = nap.IntervalSet(
            start=[0.0, 10.0],
            end=[5.0, 15.0],
        )
        source = nap.TsGroup(
            {10: nap.Ts(t=[0.5, 4.5, 10.5], time_support=support)},
            time_support=support,
        )

        with pytest.raises(ValueError, match="source time-support interval"):
            nap.TsTrials(source, nap.IntervalSet(start, end))

    def test_supported_silence_is_valid(self) -> None:
        support = nap.IntervalSet(0.0, 5.0)
        source = nap.TsGroup(
            {10: nap.Ts(t=[0.5], time_support=support)},
            time_support=support,
        )
        trials = nap.TsTrials(source, nap.IntervalSet(4.0, 5.0))

        counts = trials.count(0.5)

        assert counts.valid.all()
        np.testing.assert_array_equal(counts.d, np.zeros((1, 2, 1)))

    @pytest.mark.parametrize("trial", [0, 2, np.int64(1)])
    @pytest.mark.parametrize("unit", [10, np.int64(20)])
    def test_scalar_selection(
        self,
        ts_trials: nap.TsTrials,
        trial: int,
        unit: int,
    ) -> None:
        result = ts_trials[trial, unit]
        position = np.flatnonzero(ts_trials.trial_ids == trial)[0]
        selected = (ts_trials.trial_index == trial) & (ts_trials.unit_index == unit)

        assert isinstance(result, nap.Ts)
        np.testing.assert_allclose(
            result.t,
            np.sort(ts_trials.timestamps[selected]),
        )
        np.testing.assert_allclose(
            result.time_support.start,
            [ts_trials.window_starts[position]],
        )
        np.testing.assert_allclose(
            result.time_support.end,
            [ts_trials.window_ends[position]],
        )

    def test_trial_group(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials[0]

        assert isinstance(result, nap.TsGroup)
        np.testing.assert_array_equal(result.index, [10, 20])
        np.testing.assert_array_equal(result["region"], ["ca1", "ca3"])
        np.testing.assert_allclose(result[10].t, [0.0, 0.5, 1.0, 2.0])
        np.testing.assert_allclose(result[20].t, [0.25, 1.25])

    @pytest.mark.parametrize(
        "key, positions",
        [
            (slice(1, None), [1, 2]),
            (slice(None, None, -1), [0, 1, 2]),
            (slice(None, None, -2), [0, 2]),
            ([0, 2], [0, 2]),
            ([2, 0], [0, 2]),
            ([False, True, True], [1, 2]),
            (Ellipsis, [0, 1, 2]),
        ],
    )
    def test_trial_subset(
        self,
        ts_trials: nap.TsTrials,
        key: Any,
        positions: list[int],
    ) -> None:
        result = ts_trials[key]

        assert isinstance(result, nap.TsTrials)
        assert result.shape == (len(positions), 2)
        np.testing.assert_array_equal(
            result.trial_ids,
            ts_trials.trial_ids[positions],
        )
        np.testing.assert_array_equal(
            result.trials.values,
            ts_trials.trials.values[positions],
        )
        np.testing.assert_array_equal(
            result.durations,
            ts_trials.durations[positions],
        )

        for column in ts_trials.trials.metadata_columns:
            np.testing.assert_array_equal(
                result.trials[column],
                np.asarray(ts_trials.trials[column])[positions],
            )

        for trial_id in result.trial_ids:
            for unit_key in result.unit_keys:
                np.testing.assert_allclose(
                    result[trial_id, unit_key].t,
                    ts_trials[trial_id, unit_key].t,
                )

    def test_unit_subset(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials[:, [20]]

        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.unit_keys, [20])
        np.testing.assert_array_equal(
            result.unit_metadata["region"],
            ["ca3"],
        )
        np.testing.assert_array_equal(
            result.unit_index,
            np.full(len(result.timestamps), 20),
        )

    @pytest.mark.parametrize(
        "key, error",
        [
            (3, KeyError),
            (-1, KeyError),
            (-4, KeyError),
            ((slice(None), 0), KeyError),
            ((slice(None), 2), KeyError),
            ((slice(None), -1), KeyError),
            ((0, 10, 0), IndexError),
            (np.array([[0]]), IndexError),
        ],
    )
    def test_invalid_index(
        self,
        ts_trials: nap.TsTrials,
        key: Any,
        error: type[Exception],
    ) -> None:
        with pytest.raises(error):
            ts_trials[key]

    def test_realign(self, ts_trials: nap.TsTrials) -> None:
        original = ts_trials.timestamps.copy()
        result = ts_trials.realign_to("cue")
        shifts = np.array([0.5, 0.5, 1.0])

        assert result.alignment == "cue"
        np.testing.assert_allclose(result.window_starts, -shifts)
        np.testing.assert_allclose(
            result.window_ends,
            ts_trials.window_ends - shifts,
        )
        np.testing.assert_array_equal(result.durations, ts_trials.durations)
        np.testing.assert_array_equal(ts_trials.timestamps, original)

        for position, trial_id in enumerate(ts_trials.trial_ids):
            for unit_key in ts_trials.unit_keys:
                np.testing.assert_allclose(
                    result[trial_id, unit_key].t,
                    ts_trials[trial_id, unit_key].t - shifts[position],
                )

    @pytest.mark.parametrize("windowed", [False, True])
    def test_realign_to_start(
        self,
        ts_trials: nap.TsTrials,
        windowed: bool,
    ) -> None:
        original = ts_trials[[2, 0]]
        if windowed:
            original = original.get(0.25, 1.5)

        restored = original.realign_to("cue").realign_to("start")

        assert restored.alignment == "start"
        assert restored._windowed == original._windowed
        np.testing.assert_allclose(restored.t, original.t)
        np.testing.assert_array_equal(restored.d, original.d)
        np.testing.assert_array_equal(restored.trial_ids, original.trial_ids)
        np.testing.assert_allclose(
            restored.window_starts,
            original.window_starts,
        )
        np.testing.assert_allclose(
            restored.window_ends,
            original.window_ends,
        )
        np.testing.assert_array_equal(restored.durations, original.durations)

    @pytest.mark.parametrize(
        "values, error, message",
        [
            (["a", "b", "c"], TypeError, "must be numeric"),
            ([True, False, True], TypeError, "must be numeric"),
            ([np.nan, 5.5, 11.0], ValueError, "non-finite"),
            ([np.inf, 5.5, 11.0], ValueError, "non-finite"),
            ([-1.0, 5.5, 11.0], ValueError, "within their trials"),
            ([3.0, 5.5, 11.0], ValueError, "within their trials"),
        ],
    )
    def test_invalid_alignment(
        self,
        spike_source: nap.TsGroup,
        values: list[Any],
        error: type[Exception],
        message: str,
    ) -> None:
        intervals = nap.IntervalSet(
            start=[0.0, 5.0, 10.0],
            end=[2.0, 6.5, 13.0],
            metadata={"event": values},
        )
        trials = nap.TsTrials(spike_source, intervals)

        with pytest.raises(error, match=message):
            trials.realign_to("event")

    def test_get(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials.get(0.5, 1.5)

        np.testing.assert_allclose(result.window_starts, [0.5] * 3)
        np.testing.assert_allclose(result.window_ends, [1.5] * 3)
        np.testing.assert_allclose(result.durations, [1.0] * 3)
        np.testing.assert_allclose(result[0, 10].t, [0.5, 1.0])
        np.testing.assert_allclose(result[1, 10].t, [0.5, 1.5])
        assert np.all(result.timestamps >= 0.5)
        assert np.all(result.timestamps <= 1.5)

    @pytest.mark.parametrize(
        "start, end, expected",
        [
            (0.0, 1.0, [0.2, 1.0]),
            (0.0, 2.0, [0.2, 1.0]),
            (0.2, 1.0, [0.2, 1.0]),
            (0.0, 0.2, [0.2]),
            (1.0, 2.0, [1.0]),
            (2.0, 3.0, []),
            (-2.0, -1.0, []),
        ],
    )
    def test_get_inclusive_endpoint(
        self,
        start: float,
        end: float,
        expected: list[float],
    ) -> None:
        support = nap.IntervalSet(0.0, 5.0)
        source = nap.TsGroup(
            {0: nap.Ts(t=[0.2, 1.0], time_support=support)},
            time_support=support,
        )
        trials = nap.TsTrials(source, nap.IntervalSet(0.0, 1.0))

        result = trials.get(start, end)

        np.testing.assert_array_equal(result.timestamps, expected)
        assert result.shape == trials.shape
        np.testing.assert_array_equal(trials.timestamps, [0.2, 1.0])

    def test_get_no_overlap(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials.get(20.0, 21.0)

        assert result.shape == ts_trials.shape
        assert len(result.timestamps) == 0
        np.testing.assert_array_equal(result.durations, [0.0] * 3)

    def test_to_tsdframe(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials.to_tsdframe()
        order = np.argsort(ts_trials.timestamps, kind="stable")

        assert isinstance(result, nap.TsdFrame)
        assert list(result.columns) == ["trial", "unit"]
        np.testing.assert_allclose(result.t, ts_trials.timestamps[order])
        np.testing.assert_array_equal(
            result.d[:, 0],
            ts_trials.trial_index[order],
        )
        np.testing.assert_array_equal(
            result.d[:, 1],
            ts_trials.unit_index[order],
        )

    def test_event_array_properties(self, ts_trials: nap.TsTrials) -> None:
        subset = ts_trials[[2, 0], [20]]

        assert subset.t is subset.timestamps
        assert subset.d.shape == (len(subset.t), 2)
        np.testing.assert_array_equal(subset.trial_ids, [0, 2])
        np.testing.assert_array_equal(
            np.unique(subset.trial_index),
            [0, 2],
        )
        np.testing.assert_array_equal(
            subset.unit_index,
            np.full(len(subset.t), 20),
        )
        np.testing.assert_array_equal(subset.d[:, 0], subset.trial_index)
        np.testing.assert_array_equal(subset.d[:, 1], subset.unit_index)

        converted = subset.to_tsdframe()
        order = np.argsort(subset.t, kind="stable")
        np.testing.assert_array_equal(converted.t, subset.t[order])
        np.testing.assert_array_equal(converted.d, subset.d[order])

    def test_ids_survive_event_operations(
        self,
        ts_trials: nap.TsTrials,
    ) -> None:
        subset = ts_trials[[2, 0], [20]]
        aligned = subset.realign_to("cue")
        windowed = aligned.get(-0.25, 0.75)
        counts = windowed.count(0.5)

        for result in (aligned, windowed, counts):
            np.testing.assert_array_equal(result.trial_ids, [0, 2])

        for result in (aligned, windowed):
            np.testing.assert_array_equal(result.unit_keys, [20])
            np.testing.assert_array_equal(
                result.unit_index,
                np.full(len(result.t), 20),
            )

        np.testing.assert_allclose(aligned[2, 20].t, [-0.75, 0.25])
        np.testing.assert_allclose(windowed[2, 20].t, [0.25])
        np.testing.assert_array_equal(counts.columns, [20])

        empty = subset.get(20.0, 21.0)
        np.testing.assert_array_equal(empty.trial_ids, [0, 2])
        assert empty.t.shape == (0,)
        assert empty.d.shape == (0, 2)
        assert ts_trials[:0].count(0.5).trial_ids.size == 0

    def test_unit_labels_and_slices(self, ts_trials: nap.TsTrials) -> None:
        selected = ts_trials[:, [20]]
        sliced = ts_trials[:, 1:]

        np.testing.assert_array_equal(selected.unit_keys, sliced.unit_keys)
        np.testing.assert_array_equal(selected.d, sliced.d)
        np.testing.assert_array_equal(selected.t, sliced.t)

        reordered = ts_trials[:, [20, 10]]
        np.testing.assert_array_equal(reordered.unit_keys, [20, 10])
        np.testing.assert_array_equal(
            reordered.unit_metadata["region"],
            ["ca3", "ca1"],
        )

        for trial_id in reordered.trial_ids:
            for unit_key in reordered.unit_keys:
                np.testing.assert_array_equal(
                    reordered[trial_id, unit_key].t,
                    ts_trials[trial_id, unit_key].t,
                )

        with pytest.raises(ValueError, match="Repeated unit labels"):
            ts_trials[:, [10, 10]]


# Counting


class TestTrialCounts:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        "bin_size, time_units",
        [(1.0, "s"), (1000.0, "ms"), (1_000_000.0, "us")],
    )
    def test_count(
        self,
        ts_trials: nap.TsTrials,
        dtype: type,
        bin_size: float,
        time_units: str,
    ) -> None:
        result = ts_trials.count(
            bin_size,
            time_units=time_units,
            dtype=dtype,
        )
        expected = np.array(
            [
                [[2, 1], [1, 1], [np.nan, np.nan]],
                [[2, 1], [np.nan, np.nan], [np.nan, np.nan]],
                [[1, 1], [1, 1], [1, 0]],
            ],
            dtype=dtype,
        )

        assert isinstance(result, nap.TsdTrials)
        assert result.data_kind == "count"
        assert result.bin_size == 1.0
        assert result.sample_period == 1.0
        assert result.d.dtype == dtype
        np.testing.assert_allclose(result.t, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(result.d, expected, equal_nan=True)
        np.testing.assert_array_equal(
            result.valid,
            [
                [True, True, False],
                [True, False, False],
                [True, True, True],
            ],
        )
        np.testing.assert_array_equal(result.columns, [10, 20])
        np.testing.assert_array_equal(
            result.feature_metadata["region"],
            ["ca1", "ca3"],
        )

    def test_negative_times(self, ts_trials: nap.TsTrials) -> None:
        aligned = ts_trials.realign_to("cue")
        result = aligned.count(0.5)

        assert np.any(result.t < 0)
        assert result.alignment == "cue"

        for trial in range(len(aligned)):
            for time_index, center in enumerate(result.t):
                if not result.valid[trial, time_index]:
                    assert np.isnan(result.d[trial, time_index]).all()
                    continue

                left = center - 0.25
                right = center + 0.25
                for unit, unit_key in enumerate(aligned.unit_keys):
                    spikes = aligned[trial, unit_key].t
                    expected = np.sum((spikes >= left) & (spikes < right))
                    assert result.d[trial, time_index, unit] == expected

    @pytest.mark.parametrize(
        "bin_size, error, message",
        [
            ("1", TypeError, "int or float"),
            (0.0, ValueError, "positive"),
            (-1.0, ValueError, "positive"),
            (1e-12, ValueError, "at least one nanosecond"),
        ],
    )
    def test_invalid_bin_size(
        self,
        ts_trials: nap.TsTrials,
        bin_size: Any,
        error: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error, match=message):
            ts_trials.count(bin_size)

    @pytest.mark.parametrize("dtype", [int, np.int32, bool])
    def test_invalid_dtype(
        self,
        ts_trials: nap.TsTrials,
        dtype: type,
    ) -> None:
        with pytest.raises(TypeError, match="floating-point"):
            ts_trials.count(1.0, dtype=dtype)

    @pytest.mark.parametrize("empty", [False, True])
    def test_does_not_interpolate(
        self,
        ts_trials: nap.TsTrials,
        monkeypatch: pytest.MonkeyPatch,
        empty: bool,
    ) -> None:
        def unexpected_interpolation(**kwargs: Any) -> None:
            pytest.fail("Counting must not interpolate.")

        monkeypatch.setattr(
            nap.TsdTrials,
            "_align_source",
            staticmethod(unexpected_interpolation),
        )
        trials = ts_trials[:0] if empty else ts_trials

        result = trials.count(0.5)

        assert result.data_kind == "count"
        assert result.sample_period == 0.5
        assert result.bin_size == 0.5
        assert len(result) == len(trials)
        np.testing.assert_allclose(np.diff(result.t), 0.5)

    def test_get_requires_complete_bins(
        self,
        ts_trials: nap.TsTrials,
    ) -> None:
        counts = ts_trials.count(0.5)
        result = counts.get(0.2, 1.3)

        np.testing.assert_allclose(result.t, [0.75])
        np.testing.assert_array_equal(result.d, counts.d[:, 1:2])

    def test_unit_label_selection(self, ts_trials: nap.TsTrials) -> None:
        counts = ts_trials.count(0.5)

        np.testing.assert_array_equal(
            counts[:, :, 20].d,
            counts.d[:, :, 1],
        )
        np.testing.assert_array_equal(counts[:, :, [20]].columns, [20])

        with pytest.raises(KeyError):
            counts[:, :, 1]

    def test_restore_count_kind(self, ts_trials: nap.TsTrials) -> None:
        counts = ts_trials.count(0.5)
        continuous = counts.replace_data(counts.d, data_kind="continuous")
        restored = continuous.replace_data(
            continuous.d,
            data_kind="count",
        )

        assert_same_axes(restored, counts)
        np.testing.assert_array_equal(
            restored.get(0.2, 1.3).d,
            counts.get(0.2, 1.3).d,
        )

    def test_replace_data(self, ts_trials: nap.TsTrials) -> None:
        counts = ts_trials.realign_to("cue").count(0.5).get(-0.5, 1.5)
        original = counts.d.copy()

        result = counts.replace_data(np.full(counts.shape, 7))

        np.testing.assert_array_equal(result.d[result.valid], 7)
        assert np.isnan(result.d[~result.valid]).all()
        np.testing.assert_array_equal(counts.d, original)
        assert_same_axes(result, counts)


@pytest.mark.parametrize("compiled", [False, True], ids=["python", "numba"])
def test_count_timestamps(compiled: bool) -> None:
    count_timestamps = _count_timestamps if compiled else _count_timestamps.py_func
    valid = np.array([[True, False, True], [True, True, False]])

    result = count_timestamps(
        times_ns=np.array(
            [-11, -10, -1, 0, 10, 19, 20, -10, 0, 9, 10],
            dtype=np.int64,
        ),
        trial_index=np.array([0] * 7 + [1] * 4, dtype=np.int64),
        unit_index=np.array(
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
            dtype=np.int64,
        ),
        first_edge_ns=-10,
        bin_size_ns=10,
        valid=valid,
        n_units=2,
    )

    expected = np.array(
        [
            [[2, 0], [0, 0], [0, 2]],
            [[0, 1], [1, 1], [0, 0]],
        ],
        dtype=np.int64,
    )

    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("compiled", [False, True], ids=["python", "numba"])
@pytest.mark.parametrize("shape", [(2, 3), (0, 3), (2, 0)])
def test_count_timestamps_without_events(
    compiled: bool,
    shape: tuple[int, int],
) -> None:
    count_timestamps = _count_timestamps if compiled else _count_timestamps.py_func
    empty = np.array([], dtype=np.int64)

    result = count_timestamps(
        times_ns=empty,
        trial_index=empty,
        unit_index=empty,
        first_edge_ns=0,
        bin_size_ns=10,
        valid=np.ones(shape, dtype=bool),
        n_units=2,
    )

    np.testing.assert_array_equal(
        result,
        np.zeros((*shape, 2), dtype=np.int64),
    )


# Interpolation


class TestTsdTrialsInterpolation:
    def test_sampling_interval_is_required(self) -> None:
        source = nap.Tsd(t=[0.0, 1.0], d=[0.0, 1.0])

        with pytest.raises(TypeError, match="sample_period"):
            nap.TsdTrials(source, nap.IntervalSet(0.0, 1.0))

    @pytest.mark.parametrize("sample_period", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_sampling_interval(
        self,
        sample_period: float,
    ) -> None:
        source = nap.Tsd(t=[0.0, 1.0], d=[0.0, 1.0])

        with pytest.raises(ValueError, match="finite and positive"):
            nap.TsdTrials(
                source,
                nap.IntervalSet(0.0, 1.0),
                sample_period=sample_period,
            )

    def test_source_requires_two_samples(self) -> None:
        source = nap.Tsd(
            t=[0.0],
            d=[0.0],
            time_support=nap.IntervalSet(-2.0, 5.0),
        )

        with pytest.raises(ValueError, match="at least two samples"):
            nap.TsdTrials(
                source,
                nap.IntervalSet(0.0, 1.0),
                sample_period=0.5,
            )

    @pytest.mark.parametrize(
        "timestamps, start, end, expected",
        [
            (
                [0.0, 1.0, 2.5],
                0.0,
                1.0,
                [0.0, 0.5, 1.0],
            ),
            (
                [0.0, 1.0, 2.0],
                0.5,
                1.5,
                [0.5, 1.0, 1.5],
            ),
            (
                [0.0, 1.0, 2.0],
                -1.0,
                1.0,
                [np.nan, np.nan, 0.0, 0.5, 1.0],
            ),
            (
                [0.0, 1.0, 2.0],
                0.0,
                4.0,
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            ),
        ],
    )
    def test_nonuniform_or_partial_source(
        self,
        timestamps: list[float],
        start: float,
        end: float,
        expected: list[float],
    ) -> None:
        source = nap.Tsd(
            t=timestamps,
            d=np.arange(len(timestamps), dtype=float),
            time_support=nap.IntervalSet(-2.0, 5.0),
        )

        result = nap.TsdTrials(
            source,
            nap.IntervalSet(start, end),
            sample_period=0.5,
        )

        np.testing.assert_allclose(result.d[0], expected, equal_nan=True)
        np.testing.assert_array_equal(
            result.valid[0],
            np.isfinite(expected),
        )
        np.testing.assert_allclose(
            result.t,
            np.arange(len(expected)) * 0.5,
        )
        assert result.sample_period == 0.5

    @pytest.mark.parametrize("trailing_shape", [(), (2,), (2, 3), (2, 1, 3)])
    def test_linear_interpolation(
        self,
        trailing_shape: tuple[int, ...],
    ) -> None:
        time = np.array([0.0, 0.75, 1.5, 2.5])
        n_features = int(np.prod(trailing_shape))
        offsets = np.arange(n_features).reshape(trailing_shape)
        values = 2 * time.reshape((-1,) + (1,) * len(trailing_shape)) + offsets
        source = make_source(time, values)
        original = source.d.copy()
        intervals = nap.IntervalSet(
            start=[0.125, 1.25],
            end=[0.875, 2.25],
            metadata={"condition": ["a", "b"]},
        )

        result = nap.TsdTrials(
            source,
            intervals,
            sample_period=0.25,
        )

        assert result.shape == (2, 5, *trailing_shape)
        np.testing.assert_array_equal(result.t, np.arange(5) / 4)
        np.testing.assert_array_equal(result.valid.sum(axis=1), [4, 5])
        np.testing.assert_array_equal(
            result.trials["condition"],
            ["a", "b"],
        )

        for trial, start in enumerate(intervals.start):
            query = start + result.t[result.valid[trial]]
            expected = 2 * query.reshape((-1,) + (1,) * len(trailing_shape)) + offsets
            np.testing.assert_allclose(
                result.d[trial, result.valid[trial]],
                expected,
            )

        assert np.isnan(result.d[~result.valid]).all()
        np.testing.assert_array_equal(source.d, original)

    @pytest.mark.parametrize("trailing_shape", [(), (2,), (2, 3), (2, 1, 3)])
    def test_preserves_source_nan(
        self,
        trailing_shape: tuple[int, ...],
    ) -> None:
        time = np.arange(4, dtype=float)
        n_features = int(np.prod(trailing_shape))
        values = np.broadcast_to(time[:, None], (4, n_features)).copy()
        values[1, 0] = np.nan
        values = values.reshape(4, *trailing_shape)
        source = make_source(time, values)
        original = source.d.copy()

        result = nap.TsdTrials(
            source,
            nap.IntervalSet(0.0, 3.0),
            sample_period=0.5,
        )
        actual = result.d[0].reshape(7, n_features)
        expected = np.broadcast_to(
            result.t[:, None],
            (7, n_features),
        ).copy()
        expected[1:4, 0] = np.nan

        assert result.valid.all()
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        np.testing.assert_array_equal(source.d, original)

    def test_does_not_bridge_support_gaps(self) -> None:
        source = nap.Tsd(
            t=[0.0, 1.0, 3.0, 4.0],
            d=[0.0, 1.0, 3.0, 4.0],
            time_support=nap.IntervalSet(
                start=[0.0, 3.0],
                end=[1.0, 4.0],
            ),
        )

        result = nap.TsdTrials(
            source,
            nap.IntervalSet(0.0, 4.0),
            sample_period=0.5,
        )

        valid = (result.t <= 1.0) | (result.t >= 3.0)
        np.testing.assert_array_equal(result.valid[0], valid)
        np.testing.assert_allclose(result.d[0, valid], result.t[valid])
        assert np.isnan(result.d[0, ~valid]).all()

    def test_omits_off_grid_endpoint(self) -> None:
        source = nap.Tsd(t=[0.0, 1.0], d=[0.0, 1.0])

        result = nap.TsdTrials(
            source,
            nap.IntervalSet(0.0, 0.9),
            sample_period=0.25,
        )

        np.testing.assert_array_equal(result.t, [0.0, 0.25, 0.5, 0.75])
        np.testing.assert_array_equal(result.d[0], result.t)
        np.testing.assert_allclose(result.window_ends, [0.9])
        assert result.valid.all()

    def test_inclusive_endpoints(self) -> None:
        time = np.arange(6000) / 100
        source = nap.Tsd(t=time, d=np.arange(6000, dtype=float))
        trials = nap.TsdTrials(
            source,
            nap.IntervalSet(0.0, 59.0),
            sample_period=0.01,
        )

        assert trials.shape == (1, 5901)
        assert trials.d[0, -1] == pytest.approx(
            5900.0,
            rel=0,
            abs=1e-6,
        )
        assert trials.valid.all()

        result = trials.get(10.0, 11.0)
        expected = source.get(10.0, 11.0)

        assert result.shape == (1, 101)
        np.testing.assert_allclose(
            result.t,
            expected.t,
            rtol=0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            result.d[0],
            expected.d,
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.d[0, [0, -1]],
            [1000.0, 1100.0],
            rtol=0,
            atol=1e-6,
        )
        assert result.valid.all()

    def test_get_long_recording_endpoints(self) -> None:
        fs = 1000
        duration = 2000
        time = np.arange(fs * duration) / fs
        source = nap.Tsd(t=time, d=np.arange(len(time), dtype=float))
        trials = nap.TsdTrials(
            source,
            nap.IntervalSet(0.0, duration - 1.0),
            sample_period=1 / fs,
        )

        for start in (43.0, 1500.0, 1998.0):
            expected = source.get(start, start + 1.0)
            result = trials.get(start, start + 1.0)

            assert result.shape == (1, fs + 1)
            np.testing.assert_allclose(
                result.d[0],
                expected.d,
                rtol=0,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                result.t,
                expected.t,
                rtol=0,
                atol=1e-9,
            )


# Sampled trials: scalar, frame, and tensor


class TestTsdTrials:
    def test_create(
        self,
        tsd_trials: nap.TsdTrials,
        sampled_source: nap.Tsd | nap.TsdFrame | nap.TsdTensor,
        intervals: nap.IntervalSet,
    ) -> None:
        assert len(tsd_trials) == 3
        assert tsd_trials.shape == (3, 13, *sampled_source.shape[1:])
        assert tsd_trials.sample_period == pytest.approx(0.25)
        assert tsd_trials.bin_size is None
        assert tsd_trials.data_kind == "continuous"
        assert tsd_trials.alignment == "start"

        np.testing.assert_allclose(tsd_trials.t, np.arange(13) / 4)
        np.testing.assert_allclose(tsd_trials.durations, [2.0, 1.5, 3.0])
        np.testing.assert_array_equal(
            tsd_trials.valid.sum(axis=1),
            [9, 7, 13],
        )

        for trial, (start, end) in enumerate(zip(intervals.start, intervals.end)):
            selected = (sampled_source.t >= start) & (sampled_source.t <= end)
            np.testing.assert_allclose(
                tsd_trials.d[trial, tsd_trials.valid[trial]],
                sampled_source.d[selected],
            )

        assert np.isnan(tsd_trials.d[~tsd_trials.valid]).all()

    def test_single_trial(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials[1]

        assert_time_series(
            result,
            tsd_trials.t[:7],
            tsd_trials.d[1, :7],
        )
        np.testing.assert_allclose(result.time_support.start, [0.0])
        np.testing.assert_allclose(result.time_support.end, [1.5])

        if isinstance(result, nap.TsdFrame):
            np.testing.assert_array_equal(result.columns, ["x", "y"])
            np.testing.assert_array_equal(
                result["region"],
                ["ca1", "ca3"],
            )

    def test_columns(self, tsd_trials: nap.TsdTrials) -> None:
        if tsd_trials.ndim != 3:
            assert tsd_trials.feature_names is None
            assert tsd_trials.feature_metadata is None
            with pytest.raises(AttributeError, match="has columns"):
                _ = tsd_trials.columns
        else:
            np.testing.assert_array_equal(tsd_trials.columns, ["x", "y"])
            np.testing.assert_array_equal(
                tsd_trials.feature_metadata["region"],
                ["ca1", "ca3"],
            )

    def test_feature_selection(self, tsd_trials: nap.TsdTrials) -> None:
        if tsd_trials.ndim != 3:
            pytest.skip("Requires frame-like trial data.")

        scalar = tsd_trials[:, :, "y"]
        frame = tsd_trials[:, :, ["y"]]

        assert scalar.ndim == 2
        assert scalar.feature_names is None
        assert scalar.feature_metadata is None
        np.testing.assert_array_equal(scalar.d, tsd_trials.d[:, :, 1])

        assert frame.shape == (3, 13, 1)
        np.testing.assert_array_equal(frame.columns, ["y"])
        np.testing.assert_array_equal(
            frame.feature_metadata["region"],
            ["ca3"],
        )
        assert isinstance(tsd_trials[0, :, "y"], nap.Tsd)

    def test_feature_labels_and_slices(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        if tsd_trials.ndim != 3:
            pytest.skip("Requires frame-like trial data.")

        selected = tsd_trials[:, :, ["y"]]
        sliced = tsd_trials[:, :, 1:]
        np.testing.assert_array_equal(selected.d, sliced.d)
        assert_same_axes(selected, sliced)

        reordered = tsd_trials[:, :, ["y", "x"]]
        np.testing.assert_array_equal(reordered.columns, ["y", "x"])
        np.testing.assert_array_equal(
            reordered.d,
            tsd_trials.d[:, :, [1, 0]],
        )
        np.testing.assert_array_equal(
            reordered.feature_metadata["region"],
            ["ca3", "ca1"],
        )

        with pytest.raises(KeyError):
            tsd_trials[:, :, 0]

        with pytest.raises(ValueError, match="Repeated feature labels"):
            tsd_trials[:, :, ["x", "x"]]

    def test_get(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials.get(0.5, 1.5)

        np.testing.assert_allclose(result.t, [0.5, 0.75, 1.0, 1.25, 1.5])
        np.testing.assert_array_equal(result.d, tsd_trials.d[:, 2:7])
        np.testing.assert_array_equal(
            result.valid,
            tsd_trials.valid[:, 2:7],
        )
        np.testing.assert_allclose(result.window_starts, [0.5] * 3)
        np.testing.assert_allclose(result.window_ends, [1.5] * 3)
        np.testing.assert_allclose(result.durations, [1.0] * 3)

    def test_get_partial_windows(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials.get(0.5, 2.5)

        np.testing.assert_array_equal(result.t, tsd_trials.t[2:11])
        np.testing.assert_array_equal(result.d, tsd_trials.d[:, 2:11])
        np.testing.assert_array_equal(
            result.valid,
            tsd_trials.valid[:, 2:11],
        )
        np.testing.assert_allclose(result.window_starts, [0.5] * 3)
        np.testing.assert_allclose(result.window_ends, [2.0, 1.5, 2.5])
        np.testing.assert_allclose(result.durations, [1.5, 1.0, 2.0])
        assert np.isnan(result.d[~result.valid]).all()

    def test_ellipsis_last_axis(self, tsd_trials: nap.TsdTrials) -> None:
        key = "x" if tsd_trials.ndim == 3 else 0
        result = tsd_trials[..., key]
        actual = result.d if isinstance(result, nap.TsdTrials) else result

        np.testing.assert_array_equal(actual, tsd_trials.d[..., 0])

        if tsd_trials.ndim == 3:
            assert_same_axes(result, tsd_trials[:, :, "x"])

    def test_multiple_ellipses(self, tsd_trials: nap.TsdTrials) -> None:
        with pytest.raises(IndexError, match="one ellipsis"):
            tsd_trials[..., ...]

    def test_no_new_axis(self, tsd_trials: nap.TsdTrials) -> None:
        with pytest.raises(IndexError, match="None"):
            tsd_trials[None]

    def test_orthogonal_selection(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        trial_positions = [0, 2]
        time_positions = [0, 2, 4]
        result = tsd_trials[trial_positions, time_positions]
        expected = np.take(
            np.take(tsd_trials.d, trial_positions, axis=0),
            time_positions,
            axis=1,
        )

        assert isinstance(result, nap.TsdTrials)
        np.testing.assert_array_equal(result.d, expected)
        np.testing.assert_allclose(result.t, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(
            result.trials.values,
            tsd_trials.trials.values[trial_positions],
        )
        np.testing.assert_array_equal(
            result.valid,
            tsd_trials.valid[np.ix_(trial_positions, time_positions)],
        )

        for column in tsd_trials.trials.metadata_columns:
            np.testing.assert_array_equal(
                result.trials[column],
                np.asarray(tsd_trials.trials[column])[trial_positions],
            )

    @pytest.mark.parametrize("kind", ["sampled", "counts"])
    @pytest.mark.parametrize(
        "time_key",
        [
            slice(None, None, -1),
            slice(None, None, -2),
            [3, 1],
            [-1, 1],
        ],
    )
    def test_time_selection_order(
        self,
        tsd_trials: nap.TsdTrials,
        ts_trials: nap.TsTrials,
        kind: str,
        time_key: Any,
    ) -> None:
        trials = tsd_trials if kind == "sampled" else ts_trials.count(0.5)
        positions = np.sort(np.arange(trials.shape[1])[time_key])
        original = trials.d.copy()

        result = trials[:, time_key]

        assert isinstance(result, nap.TsdTrials)
        np.testing.assert_array_equal(result.t, trials.t[positions])
        np.testing.assert_array_equal(result.d, trials.d[:, positions])
        np.testing.assert_array_equal(
            result.valid,
            trials.valid[:, positions],
        )
        np.testing.assert_array_equal(result.trial_ids, trials.trial_ids)

        for trial_position, trial_id in enumerate(trials.trial_ids):
            single = trials[trial_id, time_key]
            valid = trials.valid[trial_position, positions]

            assert_time_series(
                single,
                trials.t[positions][valid],
                trials.d[trial_position, positions][valid],
            )
            if isinstance(single, nap.TsdFrame):
                np.testing.assert_array_equal(
                    single.columns,
                    trials.columns,
                )

        np.testing.assert_array_equal(trials.d, original)

    def test_boolean_selection(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials[[True, False, True]]
        np.testing.assert_array_equal(result.d, tsd_trials.d[[0, 2]])

    def test_scalar_time(self, tsd_trials: nap.TsdTrials) -> None:
        np.testing.assert_array_equal(tsd_trials[:, 0], tsd_trials.d[:, 0])
        np.testing.assert_array_equal(tsd_trials[0, 0], tsd_trials.d[0, 0])

        trailing = (
            (tsd_trials.columns[0],)
            if tsd_trials.ndim == 3
            else (0,) * (tsd_trials.ndim - 2)
        )
        key = (0, 0, *trailing)
        array_key = (0, 0) + (0,) * (tsd_trials.ndim - 2)

        assert np.isscalar(tsd_trials[key])
        assert tsd_trials[key] == tsd_trials.d[array_key]

    def test_array_conversion(self, tsd_trials: nap.TsdTrials) -> None:
        np.testing.assert_array_equal(np.asarray(tsd_trials), tsd_trials.d)

        copied = np.array(tsd_trials, dtype=np.float32, copy=True)
        assert copied.dtype == np.float32
        assert not np.shares_memory(copied, tsd_trials.d)

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda x: x + 2, id="add"),
            pytest.param(lambda x: 2 - x, id="reverse-subtract"),
            pytest.param(lambda x: x * 2, id="multiply"),
            pytest.param(lambda x: x / 2, id="divide"),
            pytest.param(lambda x: x > 2, id="compare"),
            pytest.param(np.negative, id="negative"),
            pytest.param(np.isfinite, id="isfinite"),
        ],
    )
    def test_operations(
        self,
        tsd_trials: nap.TsdTrials,
        operation: Any,
    ) -> None:
        original = tsd_trials.d.copy()
        valid = tsd_trials.valid.reshape(
            tsd_trials.valid.shape + (1,) * (tsd_trials.ndim - 2)
        )
        expected = np.where(valid, operation(original), np.nan)

        result = operation(tsd_trials)

        assert isinstance(result, nap.TsdTrials)
        np.testing.assert_array_equal(result.d, expected)
        np.testing.assert_array_equal(tsd_trials.d, original)
        assert_same_axes(result, tsd_trials)

    @pytest.mark.parametrize(
        "operation",
        [
            pytest.param(lambda x: x >= 0, id="greater-equal"),
            pytest.param(lambda x: x != -1, id="not-equal"),
            pytest.param(np.isfinite, id="isfinite"),
        ],
    )
    @pytest.mark.parametrize("kind", ["sampled", "counts"])
    def test_comparison_preserves_padding(
        self,
        tsd_trials: nap.TsdTrials,
        ts_trials: nap.TsTrials,
        operation: Any,
        kind: str,
    ) -> None:
        trials = tsd_trials if kind == "sampled" else ts_trials.count(0.5)
        trials = trials.replace_data(np.ones(trials.shape))

        result = operation(trials)

        assert np.isnan(result.d[~result.valid]).all()
        np.testing.assert_array_equal(result.d[result.valid], 1.0)
        assert_same_axes(result, trials)

        proportion = np.nanmean(result, axis=0)
        np.testing.assert_array_equal(
            proportion.d,
            np.ones_like(proportion.d),
        )

    def test_binary_trials(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials + tsd_trials

        np.testing.assert_allclose(
            result.d,
            tsd_trials.d * 2,
            equal_nan=True,
        )
        assert_same_axes(result, tsd_trials)

    def test_binary_trials_matching_axes(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        right = tsd_trials.replace_data(tsd_trials.d * 2)
        result = tsd_trials + right

        np.testing.assert_allclose(
            result.d,
            tsd_trials.d * 3,
            equal_nan=True,
        )
        assert_same_axes(result, tsd_trials)

    def test_binary_trials_different_intervals(self) -> None:
        time = np.arange(20) / 10
        source = nap.Tsd(t=time, d=np.arange(20.0))
        left = nap.TsdTrials(
            source,
            nap.IntervalSet(0.0, 0.5),
            sample_period=0.1,
        )
        right = nap.TsdTrials(
            source,
            nap.IntervalSet(1.0, 1.5),
            sample_period=0.1,
        )

        np.testing.assert_array_equal(left.trial_ids, right.trial_ids)
        np.testing.assert_array_equal(left.t, right.t)
        np.testing.assert_array_equal(left.valid, right.valid)

        with pytest.raises(TypeError):
            left + right

    @pytest.mark.parametrize("columns", [["c", "d"], ["b", "a"]])
    def test_binary_trials_different_columns(
        self,
        columns: list[str],
    ) -> None:
        time = np.arange(10) / 10
        values = np.zeros((len(time), 2))
        intervals = nap.IntervalSet(0.0, 0.5)
        left = nap.TsdTrials(
            nap.TsdFrame(t=time, d=values, columns=["a", "b"]),
            intervals,
            sample_period=0.1,
        )
        right = nap.TsdTrials(
            nap.TsdFrame(t=time, d=values, columns=columns),
            intervals,
            sample_period=0.1,
        )

        with pytest.raises(TypeError):
            left + right

    def test_ids_survive_value_operations(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        subset = tsd_trials[[2, 0]]

        for result in (
            subset + 1,
            subset.replace_data(subset.d),
            subset.normalize(),
            subset.smooth(0.25, windowsize=1.0),
            subset.convolve(np.ones(3) / 3),
        ):
            assert_same_axes(result, subset)

        np.testing.assert_array_equal(
            subset.get(0.5, 1.0).trial_ids,
            [0, 2],
        )

    def test_empty_axes(self, tsd_trials: nap.TsdTrials) -> None:
        selections = [tsd_trials[:0], tsd_trials[:, :0]]
        if tsd_trials.ndim > 2:
            selections.append(tsd_trials[..., :0])

        for selected in selections:
            assert selected.d.size == 0
            repr(selected)

            for result in (
                selected + 1,
                selected.convolve(np.ones(3)),
                selected.smooth(0.25, windowsize=1.0),
            ):
                assert result.shape == selected.shape
                assert_same_axes(result, selected)


# Tensor-specific indexing


class TestTsdTrialsTensor:
    def test_dimension_transitions(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        if tsd_trials.ndim < 4:
            pytest.skip("Requires tensor trial data.")

        key = (slice(None), slice(None)) + (0,) * (tsd_trials.ndim - 3) + (slice(None),)
        frame = tsd_trials[key]

        assert frame.shape == (3, 13, 3)
        np.testing.assert_array_equal(frame.d, tsd_trials.d[key])
        pd.testing.assert_index_equal(frame.columns, pd.RangeIndex(3))
        pd.testing.assert_frame_equal(
            frame.feature_metadata,
            pd.DataFrame(index=frame.columns),
        )

        single = tsd_trials[(1, *key[1:])]
        assert_time_series(single, frame.t[:7], frame.d[1, :7])
        np.testing.assert_array_equal(single.columns, frame.columns)

        scalar_key = key[:-1] + (0,)
        scalar = tsd_trials[scalar_key]

        assert scalar.shape == (3, 13)
        assert scalar.feature_names is None
        assert scalar.feature_metadata is None
        np.testing.assert_array_equal(
            scalar.d,
            tsd_trials.d[scalar_key],
        )

        single = tsd_trials[(1, *scalar_key[1:])]
        assert_time_series(single, scalar.t[:7], scalar.d[1, :7])

    def test_orthogonal_tensor_axes(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        if tsd_trials.ndim < 4:
            pytest.skip("Requires tensor trial data.")

        result = tsd_trials[[2, 0], [4, 0], [1, 0], ..., [2, 0]]
        expected = tsd_trials.d

        for axis, positions in (
            (0, [0, 2]),
            (1, [0, 4]),
            (2, [1, 0]),
            (tsd_trials.ndim - 1, [2, 0]),
        ):
            expected = np.take(expected, positions, axis=axis)

        np.testing.assert_array_equal(result.d, expected)
        np.testing.assert_array_equal(result.trial_ids, [0, 2])
        np.testing.assert_array_equal(result.t, [0.0, 1.0])
        np.testing.assert_array_equal(
            result.valid,
            tsd_trials.valid[np.ix_([0, 2], [0, 4])],
        )

    def test_positional_tensor_axes(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        if tsd_trials.ndim < 4:
            pytest.skip("Requires tensor trial data.")

        for key in (
            (..., -1),
            (..., slice(None, None, -1)),
            (..., [True, False, True]),
        ):
            result = tsd_trials[key]
            np.testing.assert_array_equal(result.d, tsd_trials.d[key])

        with pytest.raises(IndexError):
            tsd_trials[..., tsd_trials.shape[-1]]


# Normalization, convolution, and smoothing


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTsdTrialsProcessing:
    @pytest.mark.parametrize("axes", [None, 0, 1, (0, 1), -1])
    def test_normalize(
        self,
        tsd_trials: nap.TsdTrials,
        axes: int | tuple[int, ...] | None,
    ) -> None:
        original = tsd_trials.d.copy()
        mean = np.nanmean(original, axis=axes, keepdims=True)
        std = np.nanstd(original, axis=axes, keepdims=True)
        expected = (original - mean) / np.where(std == 0, 1.0, std)

        result = tsd_trials.normalize(axes=axes)

        np.testing.assert_allclose(result.d, expected, equal_nan=True)
        np.testing.assert_array_equal(tsd_trials.d, original)
        assert np.isnan(result.d[~result.valid]).all()
        assert_same_axes(result, tsd_trials)

    def test_normalize_constant(self, tsd_trials: nap.TsdTrials) -> None:
        result = (tsd_trials * 0 + 7).normalize()

        np.testing.assert_array_equal(result.d[result.valid], 0.0)
        assert np.isnan(result.d[~result.valid]).all()

    @pytest.mark.parametrize("trim", ["both", "left", "right"])
    def test_convolve(
        self,
        tsd_trials: nap.TsdTrials,
        trim: str,
    ) -> None:
        kernel = np.array([1.0, 2.0, 3.0, 4.0]) / 10
        original = tsd_trials.d.copy()
        expected = np.full_like(original, np.nan)
        start = {"both": 1, "left": 3, "right": 0}[trim]

        for trial in range(len(tsd_trials)):
            valid = tsd_trials.valid[trial]
            for coordinate in np.ndindex(tsd_trials.shape[2:]):
                key = (trial, slice(None), *coordinate)
                padded = np.where(valid, original[key], 0.0)
                convolved = np.convolve(padded, kernel, mode="full")
                trimmed = convolved[start : start + tsd_trials.shape[1]]
                expected[key] = np.where(valid, trimmed, np.nan)

        result = tsd_trials.convolve(kernel, trim=trim)

        np.testing.assert_allclose(result.d, expected, equal_nan=True)
        np.testing.assert_array_equal(tsd_trials.d, original)
        assert_same_axes(result, tsd_trials)

    @pytest.mark.parametrize("norm", [True, False])
    @pytest.mark.parametrize(
        "std, windowsize, time_units",
        [
            (0.25, None, "s"),
            (0.25, 1.0, "s"),
            (250.0, 1000.0, "ms"),
            (250_000.0, 1_000_000.0, "us"),
        ],
    )
    def test_smooth(
        self,
        tsd_trials: nap.TsdTrials,
        norm: bool,
        std: float,
        windowsize: float | None,
        time_units: str,
    ) -> None:
        original = tsd_trials.d.copy()
        result = tsd_trials.smooth(
            std,
            windowsize=windowsize,
            time_units=time_units,
            size_factor=4,
            norm=norm,
        )

        kernel = signal.windows.gaussian(5, std=1.0)
        if norm:
            kernel /= kernel.sum()

        expected = np.full_like(original, np.nan)
        for trial in range(len(tsd_trials)):
            valid = tsd_trials.valid[trial]
            for coordinate in np.ndindex(tsd_trials.shape[2:]):
                key = (trial, slice(None), *coordinate)
                padded = np.where(valid, original[key], 0.0)
                convolved = np.convolve(padded, kernel, mode="same")
                expected[key] = np.where(valid, convolved, np.nan)

        np.testing.assert_allclose(result.d, expected, equal_nan=True)
        np.testing.assert_array_equal(tsd_trials.d, original)
        assert_same_axes(result, tsd_trials)

    @pytest.mark.parametrize("method", ["smooth", "convolve"])
    def test_irregular_time(
        self,
        tsd_trials: nap.TsdTrials,
        method: str,
    ) -> None:
        irregular = tsd_trials[:, [0, 1, 3]]
        argument = 0.25 if method == "smooth" else np.ones(3)

        with pytest.raises(ValueError, match="uniformly sampled"):
            getattr(irregular, method)(argument)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"std": 0}, "std must be finite and positive"),
            ({"std": -1}, "std must be finite and positive"),
            ({"std": np.nan}, "std must be finite and positive"),
            ({"std": np.inf}, "std must be finite and positive"),
            ({"std": 1, "windowsize": 0}, "windowsize must be finite"),
            ({"std": 1, "size_factor": 0}, "positive integer"),
            ({"std": 1, "size_factor": 1.5}, "positive integer"),
        ],
    )
    def test_invalid_smoothing(
        self,
        tsd_trials: nap.TsdTrials,
        kwargs: dict[str, Any],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            tsd_trials.smooth(**kwargs)


# Replacement data


class TestTsdTrialsReplaceData:
    @pytest.mark.parametrize("as_list", [False, True])
    def test_replace_data(
        self,
        tsd_trials: nap.TsdTrials,
        as_list: bool,
    ) -> None:
        trials = tsd_trials.get(0.25, 2.5)
        original = trials.d.copy()
        values = np.arange(np.prod(trials.shape)).reshape(trials.shape)
        replacement = values.tolist() if as_list else values.copy()

        result = trials.replace_data(replacement)

        assert isinstance(result, nap.TsdTrials)
        assert result is not trials
        assert result.shape == trials.shape
        np.testing.assert_array_equal(
            result.d[result.valid],
            values[result.valid],
        )
        assert np.isnan(result.d[~result.valid]).all()
        np.testing.assert_array_equal(trials.d, original)
        np.testing.assert_array_equal(replacement, values)
        assert_same_axes(result, trials)

    def test_preserve_valid_nan(self, tsd_trials: nap.TsdTrials) -> None:
        values = np.ones(tsd_trials.shape)
        values[0, 0] = np.nan

        result = tsd_trials.replace_data(values)

        assert np.isnan(result.d[0, 0]).all()
        assert np.isnan(result.d[~result.valid]).all()
        assert_same_axes(result, tsd_trials)

    @pytest.mark.parametrize(
        "trailing_shape",
        [(), (1,), (2,), (3,), (2, 3), (2, 1, 3)],
    )
    def test_dimensions(
        self,
        tsd_trials: nap.TsdTrials,
        trailing_shape: tuple[int, ...],
    ) -> None:
        shape = (*tsd_trials.shape[:2], *trailing_shape)
        result = tsd_trials.replace_data(np.ones(shape))

        assert result.shape == shape
        np.testing.assert_array_equal(result.d[result.valid], 1.0)
        assert np.isnan(result.d[~result.valid]).all()
        assert_same_axes(scalar_trials(result), scalar_trials(tsd_trials))

        if result.ndim != 3:
            assert result.feature_names is None
            assert result.feature_metadata is None
        elif tsd_trials.feature_names is not None and result.shape[2] == len(
            tsd_trials.columns
        ):
            assert_same_axes(result, tsd_trials)
            assert result.feature_names is not tsd_trials.feature_names
            assert result.feature_metadata is not tsd_trials.feature_metadata
        else:
            pd.testing.assert_index_equal(
                result.columns,
                pd.RangeIndex(result.shape[2]),
            )
            pd.testing.assert_frame_equal(
                result.feature_metadata,
                pd.DataFrame(index=result.columns),
            )

    @pytest.mark.parametrize("as_index", [False, True])
    def test_explicit_columns(
        self,
        tsd_trials: nap.TsdTrials,
        as_index: bool,
    ) -> None:
        columns = ["first", "second"]
        if as_index:
            columns = pd.Index(columns, name="feature")

        original_names = tsd_trials.feature_names
        original_metadata = (
            None
            if tsd_trials.feature_metadata is None
            else tsd_trials.feature_metadata.copy()
        )

        result = tsd_trials.replace_data(
            np.ones((*tsd_trials.shape[:2], 2)),
            columns=columns,
        )

        pd.testing.assert_index_equal(result.columns, pd.Index(columns))
        pd.testing.assert_frame_equal(
            result.feature_metadata,
            pd.DataFrame(index=pd.Index(columns)),
        )
        assert tsd_trials.feature_names is original_names

        if original_metadata is not None:
            np.testing.assert_array_equal(
                tsd_trials.feature_metadata.columns,
                original_metadata.columns,
            )
            for column in original_metadata.columns:
                np.testing.assert_array_equal(
                    tsd_trials.feature_metadata[column],
                    original_metadata[column],
                )

    @pytest.mark.parametrize(
        "shape",
        [
            (3,),
            (3, 12, 1, 1),
            (2, 12),
            (3, 11),
            (2, 12, 2),
            (3, 11, 2),
        ],
    )
    def test_invalid_shape(
        self,
        tsd_trials: nap.TsdTrials,
        shape: tuple[int, ...],
    ) -> None:
        with pytest.raises(
            ValueError,
            match="matching trial and time dimensions",
        ):
            tsd_trials.replace_data(np.ones(shape))

    @pytest.mark.parametrize("trailing_shape", [(), (2, 3)])
    def test_columns_require_feature_axis(
        self,
        tsd_trials: nap.TsdTrials,
        trailing_shape: tuple[int, ...],
    ) -> None:
        values = np.ones((*tsd_trials.shape[:2], *trailing_shape))

        with pytest.raises(ValueError, match="three-dimensional values"):
            tsd_trials.replace_data(values, columns=["value"])

    @pytest.mark.parametrize(
        "columns",
        [[], ["one"], ["one", "two", "three"]],
    )
    def test_column_count_mismatch(
        self,
        tsd_trials: nap.TsdTrials,
        columns: list[str],
    ) -> None:
        values = np.ones((*tsd_trials.shape[:2], 2))

        with pytest.raises(ValueError, match="columns must match"):
            tsd_trials.replace_data(values, columns=columns)

    def test_data_kind_override(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials.replace_data(
            tsd_trials.d,
            data_kind="transformed",
        )

        assert result.data_kind == "transformed"
        assert tsd_trials.data_kind == "continuous"
        np.testing.assert_array_equal(result.d, tsd_trials.d)

    def test_count_kind_requires_bin_size(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        with pytest.raises(ValueError, match="requires an existing bin_size"):
            tsd_trials.replace_data(tsd_trials.d, data_kind="count")

    def test_frame_to_tensor_discards_feature_metadata(self) -> None:
        source = nap.TsdFrame(
            t=np.arange(5),
            d=np.ones((5, 2)),
            columns=["x", "y"],
            metadata={"region": ["ca1", "ca3"]},
        )
        trials = nap.TsdTrials(
            source,
            nap.IntervalSet(0.0, 4.0),
            sample_period=1.0,
        )

        result = trials.replace_data(np.ones((1, 5, 2, 3)))

        assert result.feature_names is None
        assert result.feature_metadata is None
        assert isinstance(result[0], nap.TsdTensor)
        np.testing.assert_array_equal(trials.columns, ["x", "y"])
        np.testing.assert_array_equal(
            trials.feature_metadata["region"],
            ["ca1", "ca3"],
        )


# NumPy reductions


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestTsdTrialsReductions:
    @pytest.mark.parametrize("reduction", REDUCTIONS, ids=lambda f: f.__name__)
    @pytest.mark.parametrize(
        "axis",
        [
            None,
            (),
            0,
            1,
            2,
            3,
            -1,
            -2,
            -3,
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 1, 2),
            (2, 0),
            "trailing",
            "trial-and-trailing",
        ],
    )
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduction(
        self,
        tsd_trials: nap.TsdTrials,
        reduction: Any,
        axis: int | tuple[int, ...] | str | None,
        keepdims: bool,
    ) -> None:
        if isinstance(axis, str):
            trailing = tuple(range(2, tsd_trials.ndim))
            axis = trailing if axis == "trailing" else (0, *trailing)

        axes = (
            tuple(range(tsd_trials.ndim))
            if axis is None
            else axis
            if isinstance(axis, tuple)
            else (axis,)
        )
        if any(not -tsd_trials.ndim <= value < tsd_trials.ndim for value in axes):
            pytest.skip("Axis is not present in this data.")

        axes = tuple(value % tsd_trials.ndim for value in axes)
        original = tsd_trials.d.copy()
        expected = reduction(original, axis=axis, keepdims=keepdims)
        result = reduction(tsd_trials, axis=axis, keepdims=keepdims)

        if axis is None or 1 in axes or (axes and keepdims):
            assert not isinstance(
                result,
                (nap.TsdTrials, nap.Tsd, nap.TsdFrame, nap.TsdTensor),
            )
            assert np.shape(result) == np.shape(expected)
            np.testing.assert_allclose(result, expected, equal_nan=True)

        elif 0 in axes:
            valid = tsd_trials.valid.any(axis=0)
            assert_time_series(
                result,
                tsd_trials.t[valid],
                expected[valid],
            )
            np.testing.assert_allclose(
                result.time_support.start,
                [tsd_trials.window_starts.min()],
            )
            np.testing.assert_allclose(
                result.time_support.end,
                [tsd_trials.window_ends.max()],
            )

            if isinstance(result, nap.TsdFrame):
                if tsd_trials.ndim == 3 and 2 not in axes:
                    np.testing.assert_array_equal(
                        result.columns,
                        tsd_trials.columns,
                    )
                    for column in tsd_trials.feature_metadata.columns:
                        np.testing.assert_array_equal(
                            result[column],
                            tsd_trials.feature_metadata[column],
                        )
                else:
                    np.testing.assert_array_equal(
                        result.columns,
                        np.arange(expected.shape[1]),
                    )
                    assert len(result.metadata_columns) == 0

        else:
            assert isinstance(result, nap.TsdTrials)
            np.testing.assert_allclose(
                result.d,
                expected,
                equal_nan=True,
            )
            assert_same_axes(
                result,
                tsd_trials.replace_data(expected),
            )

        np.testing.assert_array_equal(tsd_trials.d, original)

    @pytest.mark.parametrize("reduction", REDUCTIONS, ids=lambda f: f.__name__)
    def test_positional_axis(
        self,
        tsd_trials: nap.TsdTrials,
        reduction: Any,
    ) -> None:
        result = reduction(tsd_trials, 0)
        expected = reduction(tsd_trials.d, axis=0)
        valid = tsd_trials.valid.any(axis=0)

        assert_time_series(
            result,
            tsd_trials.t[valid],
            expected[valid],
        )

    @pytest.mark.parametrize("reduction", REDUCTIONS, ids=lambda f: f.__name__)
    def test_out(
        self,
        tsd_trials: nap.TsdTrials,
        reduction: Any,
    ) -> None:
        expected = reduction(tsd_trials.d, axis=0)
        output = np.empty_like(expected)

        result = reduction(tsd_trials, axis=0, out=output)

        assert result is output
        np.testing.assert_allclose(output, expected, equal_nan=True)

    @pytest.mark.parametrize(
        "reduction, kwargs",
        [
            (np.mean, {"dtype": np.float32}),
            (np.nanmean, {"dtype": np.float32}),
            (np.sum, {"initial": 5.0}),
            (np.nansum, {"initial": 5.0}),
            (np.std, {"ddof": 1}),
            (np.nanstd, {"ddof": 1}),
            (np.var, {"ddof": 1}),
            (np.nanvar, {"ddof": 1}),
            (np.min, {"initial": -100.0}),
            (np.max, {"initial": 100.0}),
        ],
    )
    def test_reduction_arguments(
        self,
        tsd_trials: nap.TsdTrials,
        reduction: Any,
        kwargs: dict[str, Any],
    ) -> None:
        expected = reduction(tsd_trials.d, axis=0, **kwargs)
        result = reduction(tsd_trials, axis=0, **kwargs)

        np.testing.assert_allclose(result.d, expected, equal_nan=True)
        assert result.d.dtype == expected.dtype

    def test_where(self, tsd_trials: nap.TsdTrials) -> None:
        valid = tsd_trials.valid.reshape(
            tsd_trials.valid.shape + (1,) * (tsd_trials.ndim - 2)
        )

        expected = np.mean(tsd_trials.d, axis=0, where=valid)
        result = np.mean(tsd_trials, axis=0, where=valid)

        np.testing.assert_allclose(result.d, expected, equal_nan=True)

    def test_mean_nan_behavior(self, tsd_trials: nap.TsdTrials) -> None:
        ordinary = np.mean(tsd_trials, axis=0)
        ignoring_nan = np.nanmean(tsd_trials, axis=0)
        partially_valid = tsd_trials.valid.any(axis=0) & (~tsd_trials.valid.all(axis=0))

        assert np.isnan(ordinary.d[partially_valid]).all()
        assert np.isfinite(ignoring_nan.d[partially_valid]).all()

    def test_omit_globally_invalid_times(
        self,
        tsd_trials: nap.TsdTrials,
    ) -> None:
        trials = tsd_trials[:2]
        valid = trials.valid.any(axis=0)
        result = np.nanmean(trials, axis=0)

        assert not valid.all()
        assert_time_series(
            result,
            trials.t[valid],
            np.nanmean(trials.d[:, valid], axis=0),
        )

    @pytest.mark.parametrize("axis", [100, (0, 0)])
    def test_invalid_axis(
        self,
        tsd_trials: nap.TsdTrials,
        axis: int | tuple[int, ...],
    ) -> None:
        with pytest.raises((IndexError, ValueError)):
            np.mean(tsd_trials, axis=axis)

    def test_unsupported_function(self, tsd_trials: nap.TsdTrials) -> None:
        with pytest.raises(TypeError):
            np.transpose(tsd_trials)


# Shared trial behavior


class TestTrialSelectionAndGrouping:
    @pytest.mark.parametrize(
        "key, expected_ids",
        [
            ([2, 0], [0, 2]),
            (slice(None, None, -1), [0, 1, 2]),
            (slice(None, None, -2), [0, 2]),
        ],
    )
    def test_trial_selection_order(
        self,
        trial_data: nap.TsTrials | nap.TsdTrials,
        key: Any,
        expected_ids: list[int],
    ) -> None:
        result = trial_data[key]

        np.testing.assert_array_equal(result.trial_ids, expected_ids)
        np.testing.assert_array_equal(
            result.trials.start,
            trial_data.trials.start[expected_ids],
        )
        np.testing.assert_array_equal(
            result.trials["condition"],
            np.asarray(trial_data.trials["condition"])[expected_ids],
        )

        if isinstance(result, nap.TsdTrials):
            np.testing.assert_array_equal(
                result.d,
                trial_data.d[expected_ids],
            )
        else:
            for trial_id in expected_ids:
                for unit_key in result.unit_keys:
                    np.testing.assert_array_equal(
                        result[trial_id, unit_key].t,
                        trial_data[trial_id, unit_key].t,
                    )

    def test_chained_trial_selection(
        self,
        trial_data: nap.TsTrials | nap.TsdTrials,
    ) -> None:
        subset = trial_data[[2, 0]]

        np.testing.assert_array_equal(subset.trial_ids, [0, 2])
        np.testing.assert_array_equal(subset[[2]].trial_ids, [2])
        np.testing.assert_array_equal(subset[1:].trial_ids, [2])
        np.testing.assert_array_equal(subset[-1:].trial_ids, [2])

        for missing in (1, -1):
            with pytest.raises(KeyError):
                subset[missing]

        if isinstance(subset, nap.TsTrials):
            np.testing.assert_array_equal(
                subset[2, 10].t,
                trial_data[2, 10].t,
            )
        else:
            np.testing.assert_array_equal(subset[2].d, trial_data[2].d)

    def test_repeated_trial_labels(
        self,
        trial_data: nap.TsTrials | nap.TsdTrials,
    ) -> None:
        with pytest.raises(ValueError, match="Repeated trial labels"):
            trial_data[[0, 0]]

    def test_groupby(
        self,
        trial_data: nap.TsTrials | nap.TsdTrials,
    ) -> None:
        groups = trial_data.groupby("condition")

        assert set(groups) == {"a", "b"}

        for condition, positions in {"a": [0, 2], "b": [1]}.items():
            group = groups[condition]

            assert len(group) == len(positions)
            np.testing.assert_array_equal(
                group.trial_ids,
                trial_data.trial_ids[positions],
            )
            np.testing.assert_array_equal(
                group.trials.values,
                trial_data.trials.values[positions],
            )
            np.testing.assert_array_equal(
                group.durations,
                trial_data.durations[positions],
            )

            for column in trial_data.trials.metadata_columns:
                np.testing.assert_array_equal(
                    group.trials[column],
                    np.asarray(trial_data.trials[column])[positions],
                )

            if isinstance(trial_data, nap.TsTrials):
                np.testing.assert_array_equal(
                    group.unit_keys,
                    trial_data.unit_keys,
                )
                for trial_id in group.trial_ids:
                    for unit_key in group.unit_keys:
                        np.testing.assert_allclose(
                            group[trial_id, unit_key].t,
                            trial_data[trial_id, unit_key].t,
                        )
            else:
                expected = trial_data[positions]
                np.testing.assert_array_equal(group.d, expected.d)
                assert_same_axes(group, expected)

        with pytest.raises(KeyError, match="Unknown trial metadata column"):
            trial_data.groupby("missing")

    def test_groupby_after_selection(
        self,
        trial_data: nap.TsTrials | nap.TsdTrials,
    ) -> None:
        subset = trial_data[[2, 1]]
        groups = subset.groupby("condition")

        np.testing.assert_array_equal(subset.trial_ids, [1, 2])
        np.testing.assert_array_equal(groups["a"].trial_ids, [2])
        np.testing.assert_array_equal(groups["b"].trial_ids, [1])

        nested = groups["a"].groupby("condition")
        np.testing.assert_array_equal(nested["a"].trial_ids, [2])

        assert subset.groupby_apply("condition", len) == {"a": 1, "b": 1}

    def test_groupby_apply(self, tsd_trials: nap.TsdTrials) -> None:
        groups = tsd_trials.groupby("condition")
        results = tsd_trials.groupby_apply(
            "condition",
            np.nansum,
            axis=0,
        )

        for condition, group in groups.items():
            valid = group.valid.any(axis=0)
            assert_time_series(
                results[condition],
                group.t[valid],
                np.nansum(group.d, axis=0)[valid],
            )

    @pytest.mark.parametrize(
        "start, end, message",
        [
            (1.0, 1.0, "start must be less than end"),
            (2.0, 1.0, "start must be less than end"),
            (np.nan, 1.0, "must be finite"),
            (0.0, np.inf, "must be finite"),
        ],
    )
    def test_invalid_window(
        self,
        trial_data: nap.TsTrials | nap.TsdTrials,
        start: float,
        end: float,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            trial_data.get(start, end)

    def test_empty_selection(
        self,
        trial_data: nap.TsTrials | nap.TsdTrials,
    ) -> None:
        result = trial_data[:0]

        assert result.shape == (0, *trial_data.shape[1:])
        assert len(result) == 0
        assert "No trials." in repr(result)

        if isinstance(result, nap.TsTrials):
            assert len(result.timestamps) == 0


def test_invalid_constructor_types(
    intervals: nap.IntervalSet,
    spike_source: nap.TsGroup,
    sampled_source: nap.Tsd | nap.TsdFrame | nap.TsdTensor,
) -> None:
    with pytest.raises(TypeError, match="source must be a TsGroup"):
        nap.TsTrials(sampled_source, intervals)

    with pytest.raises(TypeError, match="trials must be an IntervalSet"):
        nap.TsTrials(spike_source, [])

    with pytest.raises(
        TypeError,
        match="source must be a Tsd, TsdFrame, or TsdTensor",
    ):
        nap.TsdTrials(
            spike_source,
            intervals,
            sample_period=0.25,
        )

    with pytest.raises(TypeError, match="trials must be an IntervalSet"):
        nap.TsdTrials(
            sampled_source,
            [],
            sample_period=0.25,
        )


def test_empty_trials(
    spike_source: nap.TsGroup,
    sampled_source: nap.Tsd | nap.TsdFrame | nap.TsdTensor,
) -> None:
    intervals = nap.IntervalSet([], [])
    timestamps = nap.TsTrials(spike_source, intervals)
    sampled = nap.TsdTrials(
        sampled_source,
        intervals,
        sample_period=0.25,
    )
    counts = timestamps.count(0.5)

    assert timestamps.shape == (0, 2)
    assert len(timestamps.timestamps) == 0
    assert sampled.shape == (0, 0, *sampled_source.shape[1:])
    assert sampled.valid.shape == (0, 0)
    assert counts.shape == (0, 0, 2)
    assert counts.valid.shape == (0, 0)

    for trials in (timestamps, sampled, counts):
        assert len(trials) == 0
        assert "No trials." in repr(trials)


# Representation


def test_repr(trial_data: nap.TsTrials | nap.TsdTrials) -> None:
    representation = repr(trial_data)

    if isinstance(trial_data, nap.TsTrials):
        names = ["trial", "unit"]
    else:
        names = ["trial", "time"]
        if trial_data.ndim == 3:
            names.append("unit" if trial_data.data_kind == "count" else "feature")
        else:
            names.extend(f"axis_{axis}" for axis in range(2, trial_data.ndim))

        if trial_data.ndim > 3:
            assert "tensor (flattened)" in representation

    dimensions = ", ".join(
        f"{name}: {size}" for name, size in zip(names, trial_data.shape)
    )

    assert f"shape: ({dimensions})" in representation
    assert "align: 'start'" in representation
    assert "size:" in representation


@pytest.mark.parametrize(
    "left_columns, right_columns, left_names, right_names",
    [
        ([], [], "none", "none"),
        (["condition"], ["region"], "condition", "region"),
        ([10, 20], [30], "10, 20", "30"),
    ],
)
def test_format_axes(
    left_columns: list[str | int],
    right_columns: list[str | int],
    left_names: str,
    right_names: str,
) -> None:
    result = _format_axes(
        "Trials",
        3,
        left_columns,
        "Units",
        "2",
        right_columns,
        width=40,
    )

    assert result.splitlines() == [
        f"{'':10}{'Trials':14}  Units",
        f"{'':10}{'count: 3':14}  count: 2",
        f"metadata: {left_names:14}  {right_names}",
    ]


@pytest.mark.parametrize("wrap_left", [True, False])
def test_format_axes_wrapping(wrap_left: bool) -> None:
    columns = ["first_column", "second_column", "third_column"]
    left_columns = columns if wrap_left else ["region"]
    right_columns = ["region"] if wrap_left else columns

    result = _format_axes(
        "Trials",
        3,
        left_columns,
        "Units",
        2,
        right_columns,
        width=40,
    )

    expected_rows = [
        ("first_column,", "region"),
        ("second_column,", ""),
        ("third_column", ""),
    ]
    if not wrap_left:
        expected_rows = [(right, left) for left, right in expected_rows]

    expected = [
        f"{'':10}{'Trials':14}  Units",
        f"{'':10}{'count: 3':14}  count: 2",
    ]
    for index, (left, right) in enumerate(expected_rows):
        label = "metadata: " if index == 0 else " " * 10
        expected.append(f"{label}{left:14}  {right}".rstrip())

    assert result.splitlines() == expected
