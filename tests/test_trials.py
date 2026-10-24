"""Tests for trial-aligned timestamps and sampled data."""

from typing import Any

import numpy as np
import pytest
from scipy import signal

import pynapple as nap


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


@pytest.fixture(params=["scalar", "frame"])
def sampled_source(request: pytest.FixtureRequest) -> nap.Tsd | nap.TsdFrame:
    time = np.arange(57) / 4

    if request.param == "scalar":
        return nap.Tsd(t=time, d=time)

    return nap.TsdFrame(
        t=time,
        d=np.column_stack([time, -time]),
        columns=["x", "y"],
        metadata={"region": ["ca1", "ca3"]},
    )


@pytest.fixture
def tsd_trials(
    sampled_source: nap.Tsd | nap.TsdFrame,
    intervals: nap.IntervalSet,
) -> nap.TsdTrials:
    return nap.TsdTrials(sampled_source, intervals)


def assert_same_axes(actual: nap.TsdTrials, expected: nap.TsdTrials) -> None:
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
        for column in expected.feature_metadata.columns:
            np.testing.assert_array_equal(
                actual.feature_metadata[column],
                expected.feature_metadata[column],
            )


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
        np.testing.assert_array_equal(ts_trials.window_starts, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(ts_trials.window_ends, [2.0, 1.5, 3.0])

        for trial, (start, end) in enumerate(zip(intervals.start, intervals.end)):
            for unit, key in enumerate(spike_source.index):
                absolute = spike_source[key].t
                expected = absolute[(absolute >= start) & (absolute <= end)]
                selected = (ts_trials.trial_index == trial) & (
                    ts_trials.neuron_index == unit
                )
                np.testing.assert_allclose(
                    np.sort(ts_trials.spikes[selected]),
                    expected - start,
                )

    def test_metadata(self, ts_trials: nap.TsTrials) -> None:
        np.testing.assert_array_equal(
            ts_trials.unit_metadata["region"],
            ["ca1", "ca3"],
        )
        assert "rate" not in ts_trials.unit_metadata.columns

    @pytest.mark.parametrize("trial", [0, -1, np.int64(1)])
    @pytest.mark.parametrize("unit", [0, -1, np.int64(1)])
    def test_scalar_selection(
        self,
        ts_trials: nap.TsTrials,
        trial: int,
        unit: int,
    ) -> None:
        result = ts_trials[trial, unit]
        trial_position = trial % ts_trials.shape[0]
        unit_position = unit % ts_trials.shape[1]
        selected = (ts_trials.trial_index == trial_position) & (
            ts_trials.neuron_index == unit_position
        )

        assert isinstance(result, nap.Ts)
        np.testing.assert_allclose(
            result.t,
            np.sort(ts_trials.spikes[selected]),
        )
        np.testing.assert_allclose(
            result.time_support.start,
            [ts_trials.window_starts[trial_position]],
        )
        np.testing.assert_allclose(
            result.time_support.end,
            [ts_trials.window_ends[trial_position]],
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
            ([0, 2], [0, 2]),
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
            result.trials["condition"],
            ts_trials.trials["condition"][positions],
        )

        for new_trial, old_trial in enumerate(positions):
            for unit in range(2):
                np.testing.assert_allclose(
                    result[new_trial, unit].t,
                    ts_trials[old_trial, unit].t,
                )

    def test_unit_subset(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials[:, [1]]

        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.unit_keys, [20])
        np.testing.assert_array_equal(result.unit_metadata["region"], ["ca3"])
        np.testing.assert_array_equal(
            result.neuron_index,
            np.zeros(len(result.spikes), dtype=int),
        )

    @pytest.mark.parametrize(
        "key",
        [3, -4, (slice(None), 2), (0, 0, 0), np.array([[0]])],
    )
    def test_invalid_index(self, ts_trials: nap.TsTrials, key: Any) -> None:
        with pytest.raises(IndexError):
            ts_trials[key]

    def test_realign(self, ts_trials: nap.TsTrials) -> None:
        original = ts_trials.spikes.copy()
        result = ts_trials.realign_to("cue")
        shifts = np.array([0.5, 0.5, 1.0])

        assert result.alignment == "cue"
        np.testing.assert_allclose(result.window_starts, -shifts)
        np.testing.assert_allclose(
            result.window_ends,
            ts_trials.window_ends - shifts,
        )
        np.testing.assert_array_equal(result.durations, ts_trials.durations)
        np.testing.assert_array_equal(ts_trials.spikes, original)

        for trial in range(3):
            for unit in range(2):
                np.testing.assert_allclose(
                    result[trial, unit].t,
                    ts_trials[trial, unit].t - shifts[trial],
                )

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

        np.testing.assert_allclose(result.window_starts, [0.5, 0.5, 0.5])
        np.testing.assert_allclose(result.window_ends, [1.5, 1.5, 1.5])
        np.testing.assert_allclose(result.durations, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(result[0, 0].t, [0.5, 1.0])
        np.testing.assert_allclose(result[1, 0].t, [0.5])
        assert np.all(result.spikes >= 0.5)
        assert np.all(result.spikes < 1.5)

    def test_get_no_overlap(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials.get(20.0, 21.0)

        assert result.shape == ts_trials.shape
        assert len(result.spikes) == 0
        np.testing.assert_array_equal(result.durations, [0.0, 0.0, 0.0])

    def test_to_tsdframe(self, ts_trials: nap.TsTrials) -> None:
        result = ts_trials.to_tsdframe()
        order = np.argsort(ts_trials.spikes, kind="stable")

        assert isinstance(result, nap.TsdFrame)
        assert list(result.columns) == ["trial", "unit"]
        np.testing.assert_allclose(result.t, ts_trials.spikes[order])
        np.testing.assert_array_equal(
            result.d[:, 0],
            ts_trials.trial_ids[ts_trials.trial_index[order]],
        )
        np.testing.assert_array_equal(
            result.d[:, 1],
            ts_trials.unit_keys[ts_trials.neuron_index[order]],
        )

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
        result = ts_trials.count(bin_size, time_units=time_units, dtype=dtype)
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
            [[True, True, False], [True, False, False], [True, True, True]],
        )
        np.testing.assert_array_equal(result.columns, [10, 20])
        np.testing.assert_array_equal(
            result.feature_metadata["region"],
            ["ca1", "ca3"],
        )

    def test_count_negative_times(self, ts_trials: nap.TsTrials) -> None:
        aligned = ts_trials.realign_to("cue")
        result = aligned.count(0.5)

        assert np.any(result.t < 0)
        assert result.alignment == "cue"

        for trial in range(3):
            for time_index, center in enumerate(result.t):
                left = center - 0.25
                right = center + 0.25
                if not result.valid[trial, time_index]:
                    assert np.isnan(result.d[trial, time_index]).all()
                    continue

                for unit in range(2):
                    spikes = aligned[trial, unit].t
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
    def test_invalid_count_dtype(
        self,
        ts_trials: nap.TsTrials,
        dtype: type,
    ) -> None:
        with pytest.raises(TypeError, match="floating-point"):
            ts_trials.count(1.0, dtype=dtype)


class TestTsdTrials:
    def test_create(
        self,
        tsd_trials: nap.TsdTrials,
        sampled_source: nap.Tsd | nap.TsdFrame,
        intervals: nap.IntervalSet,
    ) -> None:
        assert len(tsd_trials) == 3
        assert tsd_trials.shape[:2] == (3, 12)
        assert tsd_trials.sample_period == pytest.approx(0.25)
        assert tsd_trials.bin_size is None
        assert tsd_trials.data_kind == "continuous"
        assert tsd_trials.alignment == "start"

        np.testing.assert_allclose(tsd_trials.t, np.arange(12) / 4)
        np.testing.assert_allclose(tsd_trials.durations, [2.0, 1.5, 3.0])
        np.testing.assert_array_equal(tsd_trials.valid.sum(axis=1), [8, 6, 12])

        for trial, (start, end) in enumerate(zip(intervals.start, intervals.end)):
            selected = (sampled_source.t >= start) & (sampled_source.t < end)
            np.testing.assert_allclose(
                tsd_trials.d[trial, tsd_trials.valid[trial]],
                sampled_source.d[selected],
            )

        assert np.isnan(tsd_trials.d[~tsd_trials.valid]).all()

    def test_columns(self, tsd_trials: nap.TsdTrials) -> None:
        if tsd_trials.ndim == 2:
            with pytest.raises(AttributeError, match="no columns"):
                _ = tsd_trials.columns
        else:
            np.testing.assert_array_equal(tsd_trials.columns, ["x", "y"])
            np.testing.assert_array_equal(
                tsd_trials.feature_metadata["region"],
                ["ca1", "ca3"],
            )

    def test_single_trial(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials[1]
        expected_type = nap.Tsd if tsd_trials.ndim == 2 else nap.TsdFrame

        assert isinstance(result, expected_type)
        np.testing.assert_allclose(result.t, tsd_trials.t[:6])
        np.testing.assert_allclose(result.d, tsd_trials.d[1, :6])
        np.testing.assert_allclose(result.time_support.start, [0.0])
        np.testing.assert_allclose(result.time_support.end, [1.5])

        if isinstance(result, nap.TsdFrame):
            np.testing.assert_array_equal(result.columns, ["x", "y"])
            np.testing.assert_array_equal(result["region"], ["ca1", "ca3"])

    def test_orthogonal_selection(self, tsd_trials: nap.TsdTrials) -> None:
        trial_positions = [0, 2]
        time_positions = [0, 2, 4]
        result = tsd_trials[trial_positions, time_positions]
        expected = np.take(
            np.take(tsd_trials.d, trial_positions, axis=0),
            time_positions,
            axis=1,
        )

        assert isinstance(result, nap.TsdTrials)
        np.testing.assert_allclose(result.d, expected, equal_nan=True)
        np.testing.assert_allclose(result.t, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(
            result.trials.start,
            tsd_trials.trials.start[trial_positions],
        )
        np.testing.assert_array_equal(
            result.trials.end,
            tsd_trials.trials.end[trial_positions],
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

    def test_boolean_selection(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials[[True, False, True]]

        np.testing.assert_allclose(
            result.d,
            tsd_trials.d[[0, 2]],
            equal_nan=True,
        )

    def test_scalar_time(self, tsd_trials: nap.TsdTrials) -> None:
        np.testing.assert_array_equal(tsd_trials[:, 0], tsd_trials.d[:, 0])
        np.testing.assert_array_equal(tsd_trials[0, 0], tsd_trials.d[0, 0])

    def test_feature_selection(self, tsd_trials: nap.TsdTrials) -> None:
        if tsd_trials.ndim == 2:
            return

        scalar = tsd_trials[:, :, 1]
        frame = tsd_trials[:, :, [1]]

        assert scalar.ndim == 2
        assert scalar.feature_names is None
        np.testing.assert_allclose(
            scalar.d,
            tsd_trials.d[:, :, 1],
            equal_nan=True,
        )

        assert frame.shape == (3, 12, 1)
        np.testing.assert_array_equal(frame.columns, ["y"])
        np.testing.assert_array_equal(frame.feature_metadata["region"], ["ca3"])
        assert isinstance(tsd_trials[0, :, 1], nap.Tsd)

    def test_get(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials.get(0.5, 1.5)

        np.testing.assert_allclose(result.t, [0.5, 0.75, 1.0, 1.25])
        np.testing.assert_allclose(
            result.d,
            tsd_trials.d[:, 2:6],
            equal_nan=True,
        )
        np.testing.assert_allclose(result.window_starts, [0.5, 0.5, 0.5])
        np.testing.assert_allclose(result.window_ends, [1.5, 1.5, 1.5])
        np.testing.assert_allclose(result.durations, [1.0, 1.0, 1.0])

    def test_array_conversion(self, tsd_trials: nap.TsdTrials) -> None:
        np.testing.assert_allclose(
            np.asarray(tsd_trials),
            tsd_trials.d,
            equal_nan=True,
        )

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
            pytest.param(np.negative, id="ufunc"),
        ],
    )
    def test_operations(
        self,
        tsd_trials: nap.TsdTrials,
        operation: Any,
    ) -> None:
        original = tsd_trials.d.copy()
        result = operation(tsd_trials)

        assert isinstance(result, nap.TsdTrials)
        np.testing.assert_array_equal(result.d, operation(original))
        np.testing.assert_array_equal(tsd_trials.d, original)
        assert_same_axes(result, tsd_trials)

    def test_binary_trials(self, tsd_trials: nap.TsdTrials) -> None:
        result = tsd_trials + tsd_trials

        np.testing.assert_allclose(
            result.d,
            tsd_trials.d * 2,
            equal_nan=True,
        )
        assert_same_axes(result, tsd_trials)

    @pytest.mark.parametrize("axes", [None, 0, 1, (0, 1)])
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
        constant = tsd_trials * 0 + 7
        result = constant.normalize()

        np.testing.assert_array_equal(result.d[result.valid], 0.0)
        assert np.isnan(result.d[~result.valid]).all()

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

        # All parameter combinations produce a five-sample kernel.
        kernel = signal.windows.gaussian(5, std=1.0)
        if norm:
            kernel /= kernel.sum()

        expected = np.full_like(original, np.nan)
        for trial in range(len(tsd_trials)):
            valid = tsd_trials.valid[trial]
            values = original[trial]
            if values.ndim == 1:
                values = values[:, None]

            for feature in range(values.shape[1]):
                padded = np.where(valid, values[:, feature], 0.0)
                convolved = np.convolve(padded, kernel, mode="same")
                if original.ndim == 2:
                    expected[trial, valid] = convolved[valid]
                else:
                    expected[trial, valid, feature] = convolved[valid]

        np.testing.assert_allclose(result.d, expected, equal_nan=True)
        np.testing.assert_array_equal(tsd_trials.d, original)
        assert_same_axes(result, tsd_trials)

    def test_smooth_irregular_time(self, tsd_trials: nap.TsdTrials) -> None:
        irregular = tsd_trials[:, [0, 1, 3]]

        with pytest.raises(ValueError, match="uniformly sampled"):
            irregular.smooth(0.25)

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


@pytest.mark.parametrize("kind", ["timestamps", "sampled", "counts"])
def test_groupby(
    kind: str,
    ts_trials: nap.TsTrials,
    tsd_trials: nap.TsdTrials,
) -> None:
    if kind == "timestamps":
        trials = ts_trials
    elif kind == "sampled":
        trials = tsd_trials
    else:
        trials = ts_trials.count(0.5)

    groups = trials.groupby("condition")

    assert set(groups) == {"a", "b"}

    for condition, positions in {"a": [0, 2], "b": [1]}.items():
        group = groups[condition]

        assert len(group) == len(positions)
        np.testing.assert_array_equal(
            group.trials.start,
            trials.trials.start[positions],
        )
        np.testing.assert_array_equal(
            group.trials.end,
            trials.trials.end[positions],
        )
        np.testing.assert_array_equal(
            group.durations,
            trials.durations[positions],
        )

        for column in trials.trials.metadata_columns:
            np.testing.assert_array_equal(
                group.trials[column],
                np.asarray(trials.trials[column])[positions],
            )

        if isinstance(trials, nap.TsTrials):
            np.testing.assert_array_equal(group.unit_keys, trials.unit_keys)

            for new_trial, old_trial in enumerate(positions):
                for unit in range(trials.shape[1]):
                    np.testing.assert_allclose(
                        group[new_trial, unit].t,
                        trials[old_trial, unit].t,
                    )
        else:
            np.testing.assert_array_equal(group.t, trials.t)
            np.testing.assert_array_equal(
                group.valid,
                trials.valid[positions],
            )
            np.testing.assert_allclose(
                group.d,
                trials.d[positions],
                equal_nan=True,
            )

    with pytest.raises(KeyError, match="Unknown trial metadata column"):
        trials.groupby("missing")


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
    ts_trials: nap.TsTrials,
    tsd_trials: nap.TsdTrials,
    start: float,
    end: float,
    message: str,
) -> None:
    for trials in (ts_trials, tsd_trials):
        with pytest.raises(ValueError, match=message):
            trials.get(start, end)


def test_invalid_constructor_types(
    intervals: nap.IntervalSet,
    spike_source: nap.TsGroup,
    sampled_source: nap.Tsd | nap.TsdFrame,
) -> None:
    with pytest.raises(TypeError, match="source must be a TsGroup"):
        nap.TsTrials(sampled_source, intervals)

    with pytest.raises(TypeError, match="trials must be an IntervalSet"):
        nap.TsTrials(spike_source, [])

    with pytest.raises(TypeError, match="source must be a Tsd or TsdFrame"):
        nap.TsdTrials(spike_source, intervals)

    with pytest.raises(TypeError, match="trials must be an IntervalSet"):
        nap.TsdTrials(sampled_source, [])


@pytest.mark.parametrize(
    "timestamps, start, end, message",
    [
        ([0.0], 0.0, 1.0, "at least two samples"),
        ([0.0, 1.0, 2.5], 0.0, 1.0, "uniformly sampled"),
        ([0.0, 1.0, 2.0], 0.5, 1.5, "sampling grid"),
        ([0.0, 1.0, 2.0], -1.0, 1.0, "source time range"),
        ([0.0, 1.0, 2.0], 0.0, 4.0, "source time range"),
    ],
)
def test_invalid_sampled_source(
    timestamps: list[float],
    start: float,
    end: float,
    message: str,
) -> None:
    source = nap.Tsd(
        t=timestamps,
        d=np.arange(len(timestamps)),
        time_support=nap.IntervalSet(-2.0, 5.0),
    )

    with pytest.raises(ValueError, match=message):
        nap.TsdTrials(source, nap.IntervalSet(start, end))


def test_empty_trials(
    spike_source: nap.TsGroup,
    sampled_source: nap.Tsd | nap.TsdFrame,
) -> None:
    intervals = nap.IntervalSet([], [])
    timestamps = nap.TsTrials(spike_source, intervals)
    sampled = nap.TsdTrials(sampled_source, intervals)
    counts = timestamps.count(0.5)

    assert timestamps.shape == (0, 2)
    assert len(timestamps.spikes) == 0
    assert sampled.shape[:2] == (0, 0)
    assert counts.shape == (0, 0, 2)
    assert counts.valid.shape == (0, 0)

    for trials in (timestamps, sampled, counts):
        assert len(trials) == 0
        assert "No trials." in repr(trials)


def test_empty_selection(
    ts_trials: nap.TsTrials,
    tsd_trials: nap.TsdTrials,
) -> None:
    timestamps = ts_trials[:0]
    sampled = tsd_trials[:0]

    assert timestamps.shape == (0, 2)
    assert len(timestamps.spikes) == 0
    assert sampled.shape == (0, *tsd_trials.shape[1:])


def test_count_get_requires_complete_bins(ts_trials: nap.TsTrials) -> None:
    counts = ts_trials.count(0.5)
    result = counts.get(0.2, 1.3)

    np.testing.assert_allclose(result.t, [0.75])
    np.testing.assert_array_equal(result.d, counts.d[:, 1:2])


def test_repr(
    ts_trials: nap.TsTrials,
    tsd_trials: nap.TsdTrials,
) -> None:
    for trials in (ts_trials, tsd_trials, ts_trials.count(0.5)):
        representation = repr(trials)

        assert "trial" in representation
        assert "align: 'start'" in representation
        assert f"shape: {trials.shape}" in representation
        assert "size:" in representation
