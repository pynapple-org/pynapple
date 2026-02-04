"""Tests of randomize for `pynapple` package."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pynapple as nap


@pytest.mark.parametrize(
    "data, min_shift, max_shift, mode, expectation",
    [
        # data type
        (nap.Ts(t=[1, 2, 3]), 1.0, 1.0, "drop", does_not_raise()),
        (nap.TsGroup({1: nap.Ts([1, 2, 3])}), 1.0, 1.0, "drop", does_not_raise()),
        (
            [1, 2, 3],
            1.0,
            1.0,
            "drop",
            pytest.raises(
                TypeError,
                match="Invalid input, data should be a Ts or TsGroup.",
            ),
        ),
        (
            nap.TsdFrame(t=[1, 2, 3], d=np.ones((3, 2))),
            1.0,
            1.0,
            "drop",
            pytest.raises(
                TypeError,
                match="Invalid input, data should be a Ts or TsGroup.",
            ),
        ),
        (
            nap.Tsd(t=[1, 2, 3], d=np.ones(3)),
            1.0,
            1.0,
            "drop",
            pytest.raises(
                TypeError,
                match="Invalid input, data should be a Ts or TsGroup.",
            ),
        ),
        (
            nap.IntervalSet(1, 2),
            1.0,
            1.0,
            "drop",
            pytest.raises(
                TypeError,
                match="Invalid input, data should be a Ts or TsGroup.",
            ),
        ),
        (
            nap.TsGroup({1: nap.Tsd(t=[1, 2, 3], d=[1, 1, 1])}),
            1.0,
            1.0,
            "drop",
            pytest.warns(
                UserWarning,
                match="TsGroup entry 1 was not a Ts, but treating it as one!",
            ),
        ),
        (
            nap.TsGroup({1: nap.Ts([1, 2, 3]), 2: nap.Tsd(t=[1, 2, 3], d=[1, 1, 1])}),
            1.0,
            1.0,
            "drop",
            pytest.warns(
                UserWarning,
                match="TsGroup entry 2 was not a Ts, but treating it as one!",
            ),
        ),
        # min shift
        (nap.Ts(t=[1, 2, 3]), 1.0, None, "drop", does_not_raise()),
        (
            nap.Ts(t=[1, 2, 3]),
            None,
            None,
            "drop",
            pytest.raises(
                TypeError,
                match="min_shift should be a number.",
            ),
        ),
        (
            nap.Ts(t=[1, 2, 3]),
            "1.0",
            None,
            "drop",
            pytest.raises(
                TypeError,
                match="min_shift should be a number.",
            ),
        ),
        (
            nap.Ts(t=[1, 2, 3]),
            [1.0],
            None,
            "drop",
            pytest.raises(
                TypeError,
                match="min_shift should be a number.",
            ),
        ),
        # max shift
        (nap.Ts(t=[1, 2, 3]), 1.0, None, "drop", does_not_raise()),
        (nap.Ts(t=[1, 2, 3]), 1.0, 1.0, "drop", does_not_raise()),
        (
            nap.Ts(t=[1, 2, 3]),
            1.0,
            "1.0",
            "drop",
            pytest.raises(TypeError, match="max_shift should be a number."),
        ),
        (
            nap.Ts(t=[1, 2, 3]),
            1.0,
            [1.0],
            "drop",
            pytest.raises(TypeError, match="max_shift should be a number."),
        ),
        # mode
        (nap.Ts(t=[1, 2, 3]), 1.0, None, "drop", does_not_raise()),
        (nap.Ts(t=[1, 2, 3]), 1.0, 1.0, "wrap", does_not_raise()),
        (
            nap.Ts(t=[1, 2, 3]),
            1.0,
            1.0,
            "roll",
            pytest.raises(ValueError, match="mode must be either 'drop' or 'wrap'."),
        ),
    ],
)
def test_shift_timestamps_type_errors(data, min_shift, max_shift, mode, expectation):
    with expectation:
        nap.shift_timestamps(data, min_shift, max_shift, mode)


@pytest.mark.parametrize(
    "mode, shift, times, time_support, expected",
    [
        ("drop", 1, [25, 27, 33.3, 34.5], None, [26, 28, 34.3]),
        ("wrap", 1, [25, 27, 33.3, 34.5], None, [26, 28, 34.3, 26]),
        ("drop", 2, [25, 27, 33.3, 34.5], None, [27, 29]),
        ("wrap", 2, [25, 27, 33.3, 34.5], None, [27, 29, 25.8, 27]),
        # -------------------------------
        # Bigger time support
        # -------------------------------
        ("drop", 1, [25, 27, 33.3, 34.5], nap.IntervalSet(0, 34.5), [26, 28, 34.3]),
        (
            "wrap",
            1,
            [25, 27, 33.3, 34.5],
            nap.IntervalSet(0, 34.5),
            [26, 28, 34.3, 1.0],
        ),
        ("drop", 2, [25, 27, 33.3, 34.5], nap.IntervalSet(0, 34.5), [27, 29]),
        ("wrap", 2, [25, 27, 33.3, 34.5], nap.IntervalSet(0, 34.5), [27, 29, 0.8, 2.0]),
        # -------------------------------
        # Multi-epoch time support
        # -------------------------------
        (
            "drop",
            1,
            [25, 27, 33.3, 34.5],
            nap.IntervalSet(start=[25, 30], end=[27, 34.5]),
            [26, 34.3],
        ),
        (
            "wrap",
            1,
            [25, 27, 33.3, 34.5],
            nap.IntervalSet(start=[25, 30], end=[27, 34.5]),
            [26, 34.3, 26],
        ),  # wrap ignores epoch boundaries
    ],
)
def test_shift_timestamps(mode, shift, times, time_support, expected):
    data = nap.Ts(t=times, time_support=time_support)
    shifted = nap.shift_timestamps(data, min_shift=shift, max_shift=shift, mode=mode)

    assert isinstance(shifted, nap.Ts)
    if time_support is not None:
        np.testing.assert_array_equal(shifted.time_support, time_support)

    np.testing.assert_allclose(shifted.times(), sorted(expected))


def test_shift_timestamps_group_drop():
    ts1 = nap.Ts([45, 48, 50])
    ts2 = nap.Ts([35, 38, 40])
    group = nap.TsGroup({1: ts1, 2: ts2}, time_support=nap.IntervalSet(0, 50))

    shifted_group = nap.shift_timestamps(group, min_shift=5, max_shift=5, mode="drop")
    assert isinstance(shifted_group, nap.TsGroup)

    np.testing.assert_allclose(shifted_group[1].times(), [50])
    np.testing.assert_allclose(shifted_group[2].times(), [40, 43, 45])


def test_shift_timestamps_group_wrap():
    ts1 = nap.Ts([45, 48, 50])
    ts2 = nap.Ts([35, 38, 40])
    group = nap.TsGroup({1: ts1, 2: ts2}, time_support=nap.IntervalSet(0, 50))

    shifted_group = nap.shift_timestamps(group, min_shift=5, max_shift=5, mode="wrap")
    assert isinstance(shifted_group, nap.TsGroup)

    np.testing.assert_allclose(sorted(shifted_group[1].times()), [0, 3, 5])
    np.testing.assert_allclose(sorted(shifted_group[2].times()), [40, 43, 45])


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
