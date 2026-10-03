"""Shared correlogram API for spike trains and continuous signals."""

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


@pytest.fixture
def signals():
    values = np.random.default_rng(31).normal(size=(20, 3))
    return nap.TsdFrame(t=np.arange(20), d=values, columns=["a", "b", "c"])


def test_shared_crosscorrelogram_matches_pearson(signals):
    result = nap.compute_crosscorrelogram(signals, windowsize=0)
    assert list(result.columns) == [("a", "b"), ("a", "c"), ("b", "c")]
    expected = np.corrcoef(signals.values.T)[np.triu_indices(3, 1)]
    np.testing.assert_allclose(result.loc[0], expected)


@pytest.mark.parametrize("container", [tuple, list])
def test_shared_crosscorrelogram_between_frames(signals, container):
    result = nap.compute_crosscorrelogram(
        container((signals[:, [0]], signals[:, [1, 2]])), windowsize=0
    )
    assert list(result.columns) == [("a", "b"), ("a", "c")]
    expected = np.corrcoef(signals.values.T)[0, 1:]
    np.testing.assert_allclose(result.loc[0], expected)


@pytest.mark.parametrize("one_column", [False, True])
def test_shared_autocorrelogram_uses_only_same_column_pairs(
    signals, one_column, monkeypatch
):
    if one_column:
        signals = signals[:, [0]]
    module = nap.process.correlograms
    kernel = module._lagged_crosscorrelation
    calls = []

    def check_pairs(data1, data2, counts, pairs1, pairs2, max_lag):
        np.testing.assert_array_equal(pairs1, np.arange(signals.shape[1]))
        np.testing.assert_array_equal(pairs1, pairs2)
        calls.append(len(pairs1))
        return kernel(data1, data2, counts, pairs1, pairs2, max_lag)

    monkeypatch.setattr(module, "_lagged_crosscorrelation", check_pairs)
    result = nap.compute_autocorrelogram(signals, windowsize=2)
    pd.testing.assert_index_equal(result.columns, signals.columns)
    np.testing.assert_allclose(result.loc[0], 1)
    np.testing.assert_allclose(result.loc[-1], result.loc[1])
    assert calls == [signals.shape[1]]


@pytest.mark.parametrize(
    "function", [nap.compute_autocorrelogram, nap.compute_crosscorrelogram]
)
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"binsize": 0.1}, "binsize must be None"),
        ({"norm": False}, "norm must be True"),
    ],
)
def test_continuous_parameters_are_not_silently_ignored(
    signals, function, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        function(signals, windowsize=1, **kwargs)


def test_continuous_reverse_changes_reference_and_lag_sign(signals):
    forward = nap.compute_crosscorrelogram(signals, windowsize=2)
    reverse = nap.compute_crosscorrelogram(signals, windowsize=2, reverse=True)
    assert list(reverse.columns) == [("b", "a"), ("c", "a"), ("c", "b")]
    np.testing.assert_allclose(reverse.values, forward.values[::-1])


def test_mixed_spike_and_continuous_inputs_are_rejected(signals):
    group = nap.TsGroup({0: nap.Ts(t=np.arange(5))})
    with pytest.raises(TypeError):
        nap.compute_crosscorrelogram((group, signals), windowsize=1)


@pytest.mark.parametrize(
    "function", [nap.compute_autocorrelogram, nap.compute_crosscorrelogram]
)
def test_shared_api_requires_window(signals, function):
    with pytest.raises(TypeError, match="windowsize"):
        function(signals)


def test_continuous_autocorrelation_pools_epochs_without_boundary_pairs():
    values = np.column_stack(([1, 3, 2, 99, np.nan, 99, 8, 6, 7], np.ones(9)))
    frame = nap.TsdFrame(t=np.arange(9), d=values, columns=["varying", "constant"])
    result = nap.compute_autocorrelogram(
        frame, windowsize=3, ep=nap.IntervalSet(start=[0, 6], end=[2, 8])
    )
    # At lag 1, the pooled pairs are [1, 3, 8, 6] and [3, 2, 6, 7].
    expected = [np.nan, 1, 18 / np.sqrt(493), 1, 18 / np.sqrt(493), 1, np.nan]
    np.testing.assert_allclose(result["varying"], expected, equal_nan=True)
    assert result["constant"].isna().all()


def test_continuous_autocorrelation_nan_only_affects_overlapping_samples():
    frame = nap.TsdFrame(t=np.arange(5), d=np.array([1, 2, np.nan, 4, 8])[:, None])
    result = nap.compute_autocorrelogram(frame, windowsize=4)
    np.testing.assert_allclose(
        result.iloc[:, 0],
        [np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan],
        equal_nan=True,
    )


def test_continuous_autocorrelation_empty_epoch_preserves_lags_and_columns(signals):
    result = nap.compute_autocorrelogram(
        signals, windowsize=2, ep=nap.IntervalSet(30, 40)
    )
    np.testing.assert_array_equal(result.index, [-2, -1, 0, 1, 2])
    pd.testing.assert_index_equal(result.columns, signals.columns)
    assert result.isna().all().all()


@pytest.mark.parametrize(
    "function", [nap.compute_autocorrelogram, nap.compute_crosscorrelogram]
)
def test_continuous_explicit_none_and_positional_arguments(signals, function):
    expected = function(signals, windowsize=2)
    pd.testing.assert_frame_equal(function(signals, None, 2, None, True, "s"), expected)
    milliseconds = function(signals, binsize=None, windowsize=2000, time_units="ms")
    pd.testing.assert_frame_equal(milliseconds, expected)


@pytest.mark.parametrize(
    "function", [nap.compute_autocorrelogram, nap.compute_crosscorrelogram]
)
@pytest.mark.parametrize("window", [-1, np.nan, np.inf])
def test_continuous_invalid_window(function, window, signals):
    with pytest.raises(ValueError):
        function(signals, windowsize=window)


@pytest.mark.parametrize(
    "function", [nap.compute_autocorrelogram, nap.compute_crosscorrelogram]
)
@pytest.mark.parametrize("kwargs", [{"norm": 1}, {"ep": [(0, 1)]}, {"time_units": 1}])
def test_continuous_invalid_parameter_types(function, kwargs, signals):
    with pytest.raises(TypeError):
        function(signals, windowsize=1, **kwargs)


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_shared_crosscorrelogram_requires_exactly_two_frames(
    signals, container, length
):
    with pytest.raises(TypeError):
        nap.compute_crosscorrelogram(container([signals] * length), windowsize=1)


def test_continuous_input_does_not_enable_eventcorrelogram(signals):
    with pytest.raises(TypeError):
        nap.compute_eventcorrelogram(signals, nap.Ts(t=[1, 2]), 1, 2)


@pytest.mark.parametrize(
    "function", [nap.compute_autocorrelogram, nap.compute_crosscorrelogram]
)
def test_spike_inputs_still_require_binsize(function):
    group = nap.TsGroup({0: nap.Ts(t=[0, 1, 3]), 1: nap.Ts(t=[0, 2, 3])})
    with pytest.raises(TypeError):
        function(group, windowsize=1)


@pytest.mark.parametrize("seed", range(5))
def test_continuous_autocorrelation_matches_numpy_with_fragmented_support(seed):
    values = np.random.default_rng(seed).normal(size=(15, 3))
    support = nap.IntervalSet(start=[0, 8], end=[4, 14])
    frame = nap.TsdFrame(t=np.arange(15), d=values, time_support=support)
    result = nap.compute_autocorrelogram(frame, windowsize=6)
    for lag in range(7):
        first, second = [], []
        for start, stop in ((0, 5), (8, 15)):
            if stop - start > lag:
                first.extend(values[start : stop - lag])
                second.extend(values[start + lag : stop])
        for column in range(3):
            expected = np.nan
            if len(first) >= 2:
                expected = np.corrcoef(
                    np.asarray(first)[:, column], np.asarray(second)[:, column]
                )[0, 1]
            np.testing.assert_allclose(result.loc[lag, column], expected, atol=1e-14)
            np.testing.assert_allclose(result.loc[-lag, column], expected, atol=1e-14)
