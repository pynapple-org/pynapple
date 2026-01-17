"""Tests of randomize for `pynapple` package."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pynapple as nap


@pytest.mark.parametrize(
    "data, min_shift, max_shift, expectation",
    [
        # data type
        (nap.Ts(t=[1, 2, 3]), 1.0, 1.0, does_not_raise()),
        (nap.Tsd(t=[1, 2, 3], d=np.ones(3)), 1.0, 1.0, does_not_raise()),
        (nap.TsdFrame(t=[1, 2, 3], d=np.ones((3, 2))), 1.0, 1.0, does_not_raise()),
        (nap.TsdTensor(t=[1, 2, 3], d=np.ones((3, 2, 2))), 1.0, 1.0, does_not_raise()),
        (
            [1, 2, 3],
            1.0,
            1.0,
            pytest.raises(
                TypeError,
                match="Invalid input, data should be a time series object.",
            ),
        ),
        (
            nap.IntervalSet(1, 2),
            1.0,
            1.0,
            pytest.raises(
                TypeError,
                match="Invalid input, data should be a time series object.",
            ),
        ),
        # min shift
        (nap.Ts(t=[1, 2, 3]), 1.0, None, does_not_raise()),
        (
            nap.Ts(t=[1, 2, 3]),
            None,
            None,
            pytest.raises(
                TypeError,
                match="min_shift should be a number.",
            ),
        ),
        (
            nap.Ts(t=[1, 2, 3]),
            "1.0",
            None,
            pytest.raises(
                TypeError,
                match="min_shift should be a number.",
            ),
        ),
        (
            nap.Ts(t=[1, 2, 3]),
            [1.0],
            None,
            pytest.raises(
                TypeError,
                match="min_shift should be a number.",
            ),
        ),
        # max shift
        (nap.Ts(t=[1, 2, 3]), 1.0, None, does_not_raise()),
        (nap.Ts(t=[1, 2, 3]), 1.0, 1.0, does_not_raise()),
        (
            nap.Ts(t=[1, 2, 3]),
            1.0,
            "1.0",
            pytest.raises(TypeError, match="max_shift should be a number."),
        ),
        (
            nap.Ts(t=[1, 2, 3]),
            1.0,
            [1.0],
            pytest.raises(TypeError, match="max_shift should be a number."),
        ),
    ],
)
def test_shift_timestamps_type_errors(data, min_shift, max_shift, expectation):
    with expectation:
        nap.shift_timestamps(data, min_shift, max_shift)


@pytest.mark.parametrize(
    "shift, times, time_support, expectation",
    [
        (1, [25, 27, 33.3, 34.5], None, [26, 26, 28, 34.3]),
        (1, [25, 27, 33.3, 34.5], nap.IntervalSet(0, 34.5), [1, 26, 28, 34.3]),
        (1, [25, 27, 33.3, 34.5], nap.IntervalSet(0, 40), [26, 28, 34.3, 35.5]),
        (2, [25, 27, 33.3, 34.5], None, [25.8, 27, 27, 29]),
        (2, [25, 27, 33.3, 34.5], nap.IntervalSet(0, 34.5), [0.8, 2, 27, 29]),
        (2, [25, 27, 33.3, 34.5], nap.IntervalSet(0, 40), [27, 29, 35.3, 36.5]),
    ],
)
@pytest.mark.parametrize(
    "data_type, values",
    [
        (nap.Ts, None),
        (nap.Tsd, np.ones(4)),
        (nap.TsdFrame, np.ones((4, 2))),
        (nap.TsdTensor, np.ones((4, 2, 2))),
    ],
)
def test_shift_timestamps(data_type, values, shift, times, time_support, expectation):
    if values is None:
        data = data_type(t=times, time_support=time_support)
    else:
        data = data_type(t=times, d=values, time_support=time_support)
    shifted = nap.randomize.shift_timestamps(data, min_shift=shift, max_shift=shift)
    assert isinstance(shifted, data_type)
    if time_support is not None:
        np.testing.assert_array_equal(shifted.time_support, time_support)
    if values is not None:
        np.testing.assert_array_equal(shifted.values, values)
    np.testing.assert_array_equal(shifted.times(), expectation)


@pytest.mark.parametrize(
    "shift, data, expectation",
    [
        (1, nap.TsGroup({1: nap.Ts([25, 27, 33.3, 34.5])}), [[26, 26, 28, 34.3]]),
        (
            1,
            nap.TsGroup(
                {1: nap.Ts([25, 27, 33.3, 34.5])}, time_support=nap.IntervalSet(0, 34.5)
            ),
            [[1, 26, 28, 34.3]],
        ),
        (
            1,
            nap.TsGroup(
                {1: nap.Ts([25, 27, 33.3, 34.5])}, time_support=nap.IntervalSet(0, 40)
            ),
            [[26, 28, 34.3, 35.5]],
        ),
        (
            1,
            nap.TsGroup(
                {1: nap.Ts([25, 27, 33.3, 34.5]), 2: nap.Ts([24, 26, 32.3, 33.5])},
            ),
            [[25, 26, 28, 34.3], [24, 25, 27, 33.3]],
        ),
        (
            1,
            nap.TsGroup(
                {1: nap.Ts([25, 27, 33.3, 34.5]), 2: nap.Ts([24, 26, 32.3, 33.5])},
                time_support=nap.IntervalSet(0, 34.5),
            ),
            [[1, 26, 28, 34.3], [0, 25, 27, 33.3]],
        ),
        (
            1,
            nap.TsGroup(
                {1: nap.Ts([25, 27, 33.3, 34.5]), 2: nap.Ts([24, 26, 32.3, 33.5])},
                time_support=nap.IntervalSet(0, 40),
            ),
            [[26, 28, 34.3, 35.5], [25, 27, 33.3, 34.5]],
        ),
        (2, nap.TsGroup({1: nap.Ts([25, 27, 33.3, 34.5])}), [[25.8, 27, 27, 29]]),
        (
            2,
            nap.TsGroup(
                {1: nap.Ts([25, 27, 33.3, 34.5])}, time_support=nap.IntervalSet(0, 34.5)
            ),
            [[0.8, 2, 27, 29]],
        ),
        (
            2,
            nap.TsGroup(
                {1: nap.Ts([25, 27, 33.3, 34.5])}, time_support=nap.IntervalSet(0, 40)
            ),
            [[27, 29, 35.3, 36.5]],
        ),
    ],
)
def test_shift_timestamps_tsgroup(data, shift, expectation):
    shifted = nap.randomize.shift_timestamps(data, min_shift=shift, max_shift=shift)
    assert len(shifted) == len(data)
    for i, (true_key, new_key) in enumerate(zip(data.keys(), shifted.keys())):
        assert true_key == new_key
        np.testing.assert_array_equal(shifted[new_key].times(), expectation[i])


def test_shuffle_intervals_ts():
    ts = nap.Ts(t=np.arange(0, 100))
    shuff_ts = nap.randomize.shuffle_ts_intervals(ts)

    assert len(ts) == len(shuff_ts)
    assert isinstance(shuff_ts, nap.Ts)
    assert ts.time_support.values == pytest.approx(shuff_ts.time_support.values)
    assert np.diff(ts.times()) == pytest.approx(np.diff(shuff_ts.times()))


def test_resample_tsgroup():
    tsgroup = nap.TsGroup(
        {0: nap.Ts(t=np.arange(0, 100)), 1: nap.Ts(t=np.arange(0, 200))}
    )
    shuff_tsgroup = nap.randomize.shuffle_ts_intervals(tsgroup)

    assert isinstance(shuff_tsgroup, nap.TsGroup)
    assert len(tsgroup) == len(shuff_tsgroup)
    assert tsgroup.time_support.values == pytest.approx(
        shuff_tsgroup.time_support.values
    )

    for j, k in zip(tsgroup.keys(), shuff_tsgroup.keys()):
        assert j == k
        assert len(tsgroup[j]) == len(shuff_tsgroup[k])
        assert np.diff(tsgroup[j].times()) == pytest.approx(
            np.diff(shuff_tsgroup[k].times())
        )


def test_jitter_ts():
    ts = nap.Ts(t=np.arange(0, 100))
    jitter_ts = nap.randomize.jitter_timestamps(ts, max_jitter=0.1)

    assert isinstance(jitter_ts, nap.Ts)
    assert len(ts) == len(jitter_ts)


def test_jitter_tsgroup():
    tsgroup = nap.TsGroup(
        {0: nap.Ts(t=np.arange(0, 100)), 1: nap.Ts(t=np.arange(0, 200))}
    )
    jitter_tsgroup = nap.randomize.jitter_timestamps(tsgroup, max_jitter=0.1)

    assert isinstance(jitter_tsgroup, nap.TsGroup)
    assert len(tsgroup) == len(jitter_tsgroup)

    for j, k in zip(tsgroup.keys(), jitter_tsgroup.keys()):
        assert j == k
        assert len(tsgroup[j]) == len(jitter_tsgroup[k])


def test_resample_ts():
    ts = nap.Ts(t=np.arange(0, 100))
    resampled_ts = nap.randomize.resample_timestamps(ts)

    assert len(ts) == len(resampled_ts)
    assert isinstance(resampled_ts, nap.Ts)
    assert (ts.time_support.values == resampled_ts.time_support.values).all()


def test_resample_tsgroup():
    tsgroup = nap.TsGroup(
        {0: nap.Ts(t=np.arange(0, 100)), 1: nap.Ts(t=np.arange(0, 200))}
    )
    resampled_tsgroup = nap.randomize.resample_timestamps(tsgroup)

    assert isinstance(resampled_tsgroup, nap.TsGroup)
    assert len(tsgroup) == len(resampled_tsgroup)
    assert (tsgroup.time_support.values == resampled_tsgroup.time_support.values).all()

    for j, k in zip(tsgroup.keys(), resampled_tsgroup.keys()):
        assert j == k
        assert len(tsgroup[j]) == len(resampled_tsgroup[k])
