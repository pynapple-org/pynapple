"""Tests of correlograms for `pynapple` package."""

from contextlib import nullcontext as does_not_raise
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


def test_cross_correlogram():
    t1 = np.array([0])
    t2 = np.array([1])
    cc, bincenter = nap.process.correlograms._cross_correlogram(t1, t2, 1, 100)
    np.testing.assert_approx_equal(cc[101], 1.0)

    cc, bincenter = nap.process.correlograms._cross_correlogram(t2, t1, 1, 100)
    np.testing.assert_approx_equal(cc[99], 1.0)

    t1 = np.array([0])
    t2 = np.array([100])
    cc, bincenter = nap.process.correlograms._cross_correlogram(t1, t2, 1, 100)
    np.testing.assert_approx_equal(cc[200], 1.0)

    t1 = np.array([0, 10])
    cc, bincenter = nap.process.correlograms._cross_correlogram(t1, t1, 1, 100)
    np.testing.assert_approx_equal(cc[100], 1.0)
    np.testing.assert_approx_equal(cc[90], 0.5)
    np.testing.assert_approx_equal(cc[110], 0.5)

    np.testing.assert_array_almost_equal(bincenter, np.arange(-100, 101))

    for t in [100, 200, 1000]:
        np.testing.assert_array_almost_equal(
            nap.process.correlograms._cross_correlogram(
                np.arange(0, t), np.arange(0, t), 1, t
            )[0],
            np.hstack(
                (np.arange(0, 1, 1 / t), np.ones(1), np.arange(0, 1, 1 / t)[::-1])
            ),
        )


#############################
# Type Error
#############################
def get_group():
    return nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100)),
            # 1: nap.Ts(t=np.arange(0, 100)),
            # 2: nap.Ts(t=np.array([0, 10])),
            # 3: nap.Ts(t=np.arange(0, 200)),
        },
        time_support=nap.IntervalSet(0, 100),
    )


def get_ep():
    return nap.IntervalSet(start=0, end=100)


def get_event():
    return nap.Ts(t=np.arange(0, 100), time_support=nap.IntervalSet(0, 100))


@pytest.mark.parametrize(
    "func",
    [
        # nap.compute_autocorrelogram,
        # nap.compute_crosscorrelogram,
        nap.compute_eventcorrelogram
    ],
)
@pytest.mark.parametrize(
    "group, binsize, windowsize, ep, norm, time_units, msg",
    [
        (
            get_group(),
            "a",
            10,
            get_ep(),
            True,
            "s",
            "Invalid type. Parameter binsize must be of type <class 'numbers.Number'>.",
        ),
        (
            get_group(),
            1,
            "a",
            get_ep(),
            True,
            "s",
            "Invalid type. Parameter windowsize must be of type <class 'numbers.Number'>.",
        ),
        (
            get_group(),
            1,
            10,
            "a",
            True,
            "s",
            "Invalid type. Parameter ep must be of type <class 'pynapple.core.interval_set.IntervalSet'>.",
        ),
        (
            get_group(),
            1,
            10,
            get_ep(),
            "a",
            "s",
            "Invalid type. Parameter norm must be of type <class 'bool'>.",
        ),
        (
            get_group(),
            1,
            10,
            get_ep(),
            True,
            1,
            "Invalid type. Parameter time_units must be of type <class 'str'>.",
        ),
    ],
)
def test_correlograms_type_errors(
    func, group, binsize, windowsize, ep, norm, time_units, msg
):
    with pytest.raises(TypeError, match=msg):
        func(
            group=group,
            binsize=binsize,
            windowsize=windowsize,
            ep=ep,
            norm=norm,
            time_units=time_units,
        )


@pytest.mark.parametrize(
    "func, args, msg",
    [
        (
            nap.compute_autocorrelogram,
            ([1, 2, 3], 1, 1),
            "Invalid type. Parameter group must be of type TsGroup",
        ),
        (
            nap.compute_crosscorrelogram,
            ([1, 2, 3], 1, 1),
            r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\).",
        ),
        (
            nap.compute_crosscorrelogram,
            (([1, 2, 3]), 1, 1),
            r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\).",
        ),
        (
            nap.compute_crosscorrelogram,
            ((get_group(), [1, 2, 3]), 1, 1),
            r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\).",
        ),
        (
            nap.compute_crosscorrelogram,
            ((get_group(), get_group(), get_group()), 1, 1),
            r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\).",
        ),
        (
            nap.compute_eventcorrelogram,
            ([1, 2, 3], 1, 1),
            "Invalid type. Parameter group must be of type TsGroup",
        ),
    ],
)
def test_correlograms_type_errors_group(func, args, msg):
    with pytest.raises(TypeError, match=msg):
        func(*args)


@pytest.mark.parametrize(
    "func, args, msg",
    [
        (
            nap.compute_eventcorrelogram,
            (get_group(), [1, 2, 3], 1, 1),
            r"Invalid type. Parameter event must be of type \(<class 'pynapple.core.time_series.Ts'>, <class 'pynapple.core.time_series.Tsd'>\).",
        ),
    ],
)
def test_correlograms_type_errors_event(func, args, msg):
    with pytest.raises(TypeError, match=msg):
        func(*args)


def _reference_lagged_correlation(data1, data2, counts, lag, pairs):
    values1 = []
    values2 = []
    start = 0
    for count in counts:
        end = start + count
        if lag >= 0:
            values1.append(data1[start : end - lag])
            values2.append(data2[start + lag : end])
        else:
            values1.append(data1[start - lag : end])
            values2.append(data2[start : end + lag])
        start = end

    values1 = np.concatenate(values1)
    values2 = np.concatenate(values2)
    return np.array([np.corrcoef(values1[:, i], values2[:, j])[0, 1] for i, j in pairs])


def test_lagged_crosscorrelation_matches_numpy_and_lag_sign():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=100)
    delayed = np.empty_like(reference)
    delayed[:3] = rng.normal(size=3)
    delayed[3:] = reference[:-3]
    timestamps = np.arange(100) / 10
    frame = nap.TsdFrame(
        t=timestamps,
        d=np.column_stack((reference, delayed, rng.normal(size=100))),
        columns=["reference", "delayed", "noise"],
        time_support=nap.IntervalSet(0, 10),
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=0.5)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [
        ("reference", "delayed"),
        ("reference", "noise"),
        ("delayed", "noise"),
    ]
    np.testing.assert_allclose(result.index, np.arange(-5, 6) / 10)
    assert result[("reference", "delayed")].idxmax() == pytest.approx(0.3)
    for lag_index, lag in enumerate(range(-5, 6)):
        expected = _reference_lagged_correlation(
            frame.values,
            frame.values,
            np.array([len(frame)]),
            lag,
            [(0, 1), (0, 2), (1, 2)],
        )
        np.testing.assert_allclose(result.iloc[lag_index], expected)


@pytest.mark.parametrize("container", [tuple, list])
def test_lagged_crosscorrelation_two_frames(container):
    rng = np.random.default_rng(1)
    timestamps = np.arange(20) / 2
    support = nap.IntervalSet(0, 10)
    frame1 = nap.TsdFrame(
        t=timestamps,
        d=rng.normal(size=(20, 2)),
        columns=["a", "b"],
        time_support=support,
    )
    frame2 = nap.TsdFrame(
        t=timestamps,
        d=rng.normal(size=(20, 2)),
        columns=["c", "d"],
        time_support=support,
    )

    result = nap.compute_lagged_crosscorrelation(container((frame1, frame2)), 1)

    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert list(result.columns) == [("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")]
    for lag_index, lag in enumerate(range(-2, 3)):
        expected = _reference_lagged_correlation(
            frame1.values,
            frame2.values,
            np.array([20]),
            lag,
            pairs,
        )
        np.testing.assert_allclose(result.iloc[lag_index], expected)


def test_lagged_crosscorrelation_pools_epochs_without_crossing_boundaries():
    rng = np.random.default_rng(2)
    timestamps = np.r_[np.arange(5), np.arange(10, 15)]
    support = nap.IntervalSet(start=[0, 10], end=[5, 15])
    values = rng.normal(size=(10, 2))
    frame = nap.TsdFrame(
        t=timestamps,
        d=values,
        columns=["a", "b"],
        time_support=support,
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=2)

    for lag_index, lag in enumerate(range(-2, 3)):
        expected = _reference_lagged_correlation(
            values, values, np.array([5, 5]), lag, [(0, 1)]
        )
        np.testing.assert_allclose(result.iloc[lag_index], expected)

    pooled_across_boundary = np.corrcoef(values[:-1, 0], values[1:, 1])[0, 1]
    assert result.loc[1.0, ("a", "b")] != pytest.approx(pooled_across_boundary)


def test_lagged_crosscorrelation_epochs_are_intersected_with_time_support():
    rng = np.random.default_rng(3)
    timestamps = np.r_[np.arange(5), np.arange(10, 15)]
    support = nap.IntervalSet(start=[0, 10], end=[5, 15])
    frame = nap.TsdFrame(
        t=timestamps,
        d=rng.normal(size=(10, 2)),
        time_support=support,
    )

    result = nap.compute_lagged_crosscorrelation(
        frame, windowsize=1, epochs=nap.IntervalSet(0, 15)
    )
    expected = _reference_lagged_correlation(
        frame.values, frame.values, np.array([5, 5]), 1, [(0, 1)]
    )

    np.testing.assert_allclose(result.loc[1.0], expected)


def test_lagged_crosscorrelation_nan_and_constant_values_propagate():
    timestamps = np.arange(5)
    frame = nap.TsdFrame(
        t=timestamps,
        d=np.column_stack((np.arange(5), [0.0, 1.0, np.nan, 3.0, 4.0], np.ones(5))),
        columns=["a", "nan", "constant"],
        time_support=nap.IntervalSet(0, 5),
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=0)

    assert np.isnan(result.loc[0.0, ("a", "nan")])
    assert np.isnan(result.loc[0.0, ("a", "constant")])


def test_lagged_crosscorrelation_is_stable_for_large_offsets():
    rng = np.random.default_rng(4)
    values = 1e12 + rng.normal(size=(1000, 2))
    frame = nap.TsdFrame(
        t=np.arange(1000),
        d=values,
        time_support=nap.IntervalSet(0, 1000),
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=0)
    expected = np.corrcoef(values[:, 0], values[:, 1])[0, 1]

    np.testing.assert_allclose(result.iloc[0, 0], expected, atol=1e-8)


def test_lagged_crosscorrelation_is_symmetric_when_inputs_are_swapped():
    rng = np.random.default_rng(5)
    timestamps = np.arange(50) / 5
    support = nap.IntervalSet(0, 10)
    frame1 = nap.TsdFrame(
        t=timestamps, d=rng.normal(size=(50, 1)), columns=["a"], time_support=support
    )
    frame2 = nap.TsdFrame(
        t=timestamps, d=rng.normal(size=(50, 1)), columns=["b"], time_support=support
    )

    forward = nap.compute_lagged_crosscorrelation((frame1, frame2), 1)
    backward = nap.compute_lagged_crosscorrelation((frame2, frame1), 1)

    np.testing.assert_allclose(
        forward[("a", "b")].values, backward[("b", "a")].values[::-1]
    )


def test_lagged_crosscorrelation_time_units():
    timestamps = np.arange(10) / 1000
    frame = nap.TsdFrame(
        t=timestamps,
        d=np.column_stack((np.arange(10), np.arange(10))),
        time_support=nap.IntervalSet(0, 0.01),
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=2, time_units="ms")

    np.testing.assert_allclose(result.index, np.arange(-2, 3) / 1000)


def test_lagged_crosscorrelation_uses_timestamp_spacing():
    timestamps = np.arange(10) / 10
    frame = nap.TsdFrame(
        t=timestamps,
        d=np.column_stack((np.arange(10), np.arange(10))),
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=0.2)

    np.testing.assert_allclose(result.index, np.arange(-2, 3) / 10)


def test_lagged_crosscorrelation_window_smaller_than_sample_interval():
    frame = nap.TsdFrame(
        t=np.arange(10) / 10,
        d=np.column_stack((np.arange(10), np.arange(10))),
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=0.05)

    np.testing.assert_array_equal(result.index, np.array([0.0]))
    assert result.iloc[0, 0] == pytest.approx(1.0)


def test_lagged_crosscorrelation_lags_without_observations_are_nan():
    frame = nap.TsdFrame(
        t=np.arange(3),
        d=np.column_stack((np.arange(3), np.arange(3))),
        time_support=nap.IntervalSet(0, 3),
    )

    result = nap.compute_lagged_crosscorrelation(frame, windowsize=3)

    assert np.isnan(result.loc[-3.0, (0, 1)])
    assert np.isnan(result.loc[3.0, (0, 1)])


def test_lagged_crosscorrelation_empty_epochs_are_nan():
    frame = nap.TsdFrame(
        t=np.arange(5),
        d=np.column_stack((np.arange(5), np.arange(5))),
        time_support=nap.IntervalSet(0, 5),
    )

    result = nap.compute_lagged_crosscorrelation(
        frame, windowsize=1, epochs=nap.IntervalSet(10, 11)
    )

    assert result.isna().all().all()


@pytest.mark.parametrize(
    "data, windowsize, epochs, time_units, error, message",
    [
        (np.arange(3), 1, None, "s", TypeError, "data must be a TsdFrame"),
        ([], 1, None, "s", TypeError, "data must be a TsdFrame"),
        (None, "one", None, "s", TypeError, "data must be a TsdFrame"),
    ],
)
def test_lagged_crosscorrelation_invalid_data(
    data, windowsize, epochs, time_units, error, message
):
    with pytest.raises(error, match=message):
        nap.compute_lagged_crosscorrelation(data, windowsize, epochs, time_units)


def test_lagged_crosscorrelation_invalid_parameters():
    frame = nap.TsdFrame(
        t=np.arange(5),
        d=np.column_stack((np.arange(5), np.arange(5))),
        time_support=nap.IntervalSet(0, 5),
    )

    with pytest.raises(TypeError, match="windowsize must be a number"):
        nap.compute_lagged_crosscorrelation(frame, "one")
    with pytest.raises(ValueError, match="finite and non-negative"):
        nap.compute_lagged_crosscorrelation(frame, -1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        nap.compute_lagged_crosscorrelation(frame, np.inf)
    with pytest.raises(TypeError, match="epochs must be an IntervalSet"):
        nap.compute_lagged_crosscorrelation(frame, 1, epochs=[(0, 1)])
    with pytest.raises(TypeError, match="time_units must be a string"):
        nap.compute_lagged_crosscorrelation(frame, 1, time_units=1)
    with pytest.raises(ValueError, match="unrecognized time units type"):
        nap.compute_lagged_crosscorrelation(frame, 1, time_units="minutes")


def test_lagged_crosscorrelation_requires_matching_timestamps():
    frame1 = nap.TsdFrame(t=np.arange(5), d=np.arange(5)[:, None])
    frame2 = nap.TsdFrame(t=np.arange(5) + 0.1, d=np.arange(5)[:, None])

    with pytest.raises(ValueError, match="identical timestamps"):
        nap.compute_lagged_crosscorrelation((frame1, frame2), 1)


def test_lagged_crosscorrelation_requires_two_columns_for_one_frame():
    frame = nap.TsdFrame(t=np.arange(5), d=np.arange(5)[:, None])

    with pytest.raises(ValueError, match="at least two columns"):
        nap.compute_lagged_crosscorrelation(frame, 1)


def test_lagged_crosscorrelation_requires_columns_in_both_frames():
    timestamps = np.arange(5)
    empty = nap.TsdFrame(t=timestamps, d=np.empty((5, 0)))
    frame = nap.TsdFrame(t=timestamps, d=np.arange(5)[:, None])

    with pytest.raises(ValueError, match="at least one column"):
        nap.compute_lagged_crosscorrelation((empty, frame), 1)


def test_lagged_crosscorrelation_requires_regular_sampling():
    frame = nap.TsdFrame(
        t=np.array([0.0, 0.1, 0.3, 0.6]),
        d=np.arange(8).reshape(4, 2),
    )

    with pytest.raises(RuntimeError, match="regularly sampled"):
        nap.compute_lagged_crosscorrelation(frame, 1)


def test_lagged_crosscorrelation_requires_a_sampling_interval():
    frame = nap.TsdFrame(
        t=np.array([0.0]),
        d=np.array([[1.0, 2.0]]),
        time_support=nap.IntervalSet(0, 1),
    )

    with pytest.raises(RuntimeError, match="sampling interval could not be determined"):
        nap.compute_lagged_crosscorrelation(frame, 0)


def test_lagged_crosscorrelation_rejects_complex_data():
    frame = nap.TsdFrame(
        t=np.arange(5),
        d=np.column_stack((np.arange(5), np.arange(5))) * (1 + 1j),
    )

    with pytest.raises(TypeError, match="real-valued data"):
        nap.compute_lagged_crosscorrelation(frame, 1)


#################################################
# Normal tests
#################################################


@pytest.mark.parametrize(
    "group, binsize, windowsize, kwargs, expected",
    [
        (
            get_group(),
            1,
            100,
            {},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            1,
            100,
            {"norm": False},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            nap.TsGroup({1: nap.Ts(t=np.array([0, 10]))}),
            1,
            100,
            {"norm": False},
            np.hstack(
                (
                    np.zeros(90),
                    np.array([0.5]),
                    np.zeros((19)),
                    np.array([0.5]),
                    np.zeros((90)),
                )
            )[:, np.newaxis],
        ),
        (
            get_group(),
            1,
            100,
            {"ep": get_ep()},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            1,
            100,
            {"time_units": "s"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            1 * 1e3,
            100 * 1e3,
            {"time_units": "ms"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            1 * 1e6,
            100 * 1e6,
            {"time_units": "us"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
    ],
)
def test_autocorrelogram(group, binsize, windowsize, kwargs, expected):
    cc = nap.compute_autocorrelogram(group, binsize, windowsize, **kwargs)
    assert isinstance(cc, pd.DataFrame)
    assert list(cc.keys()) == list(group.keys())
    if "time_units" in kwargs:
        if kwargs["time_units"] == "ms":
            np.testing.assert_array_almost_equal(
                cc.index.values * 1e3,
                np.arange(-windowsize, windowsize + binsize, binsize),
            )
        if kwargs["time_units"] == "us":
            np.testing.assert_array_almost_equal(
                cc.index.values * 1e6,
                np.arange(-windowsize, windowsize + binsize, binsize),
            )
        if kwargs["time_units"] == "s":
            np.testing.assert_array_almost_equal(
                cc.index.values, np.arange(-windowsize, windowsize + binsize, binsize)
            )
    else:
        np.testing.assert_array_almost_equal(
            cc.index.values, np.arange(-windowsize, windowsize + binsize, binsize)
        )
    np.testing.assert_array_almost_equal(cc.values, expected)


@pytest.mark.parametrize(
    "group, event, binsize, windowsize, kwargs, expected",
    [
        (
            get_group(),
            get_event(),
            1,
            100,
            {},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            get_event(),
            1,
            100,
            {"norm": False},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            get_event(),
            1,
            100,
            {"ep": get_ep()},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            get_event(),
            1,
            100,
            {"time_units": "s"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            get_event(),
            1 * 1e3,
            100 * 1e3,
            {"time_units": "ms"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group(),
            get_event(),
            1 * 1e6,
            100 * 1e6,
            {"time_units": "us"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
    ],
)
def test_eventcorrelogram(group, event, binsize, windowsize, kwargs, expected):
    cc = nap.compute_eventcorrelogram(group, event, binsize, windowsize, **kwargs)
    assert isinstance(cc, pd.DataFrame)
    assert list(cc.keys()) == list(group.keys())
    if "time_units" in kwargs:
        if kwargs["time_units"] == "ms":
            np.testing.assert_array_almost_equal(
                cc.index.values * 1e3,
                np.arange(-windowsize, windowsize + binsize, binsize),
            )
        if kwargs["time_units"] == "us":
            np.testing.assert_array_almost_equal(
                cc.index.values * 1e6,
                np.arange(-windowsize, windowsize + binsize, binsize),
            )
        if kwargs["time_units"] == "s":
            np.testing.assert_array_almost_equal(
                cc.index.values, np.arange(-windowsize, windowsize + binsize, binsize)
            )
    else:
        np.testing.assert_array_almost_equal(
            cc.index.values, np.arange(-windowsize, windowsize + binsize, binsize)
        )
    np.testing.assert_array_almost_equal(cc.values, expected)


def get_group2():
    return nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 100)),
            1: nap.Ts(t=np.arange(0, 100)),
            # 2: nap.Ts(t=np.array([0, 10])),
            # 3: nap.Ts(t=np.arange(0, 200)),
        },
        time_support=nap.IntervalSet(0, 100),
    )


@pytest.mark.parametrize(
    "group, binsize, windowsize, kwargs, expected",
    [
        (
            get_group2(),
            1,
            100,
            {},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group2(),
            1,
            100,
            {"norm": False},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            (get_group(), get_group()),
            1,
            100,
            {},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group2(),
            1,
            100,
            {"ep": get_ep()},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            (get_group(), get_group()),
            1,
            100,
            {"ep": get_ep()},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            (get_group(), get_group()),
            1,
            100,
            {"norm": False},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group2(),
            1,
            100,
            {"time_units": "s"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group2(),
            1 * 1e3,
            100 * 1e3,
            {"time_units": "ms"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
        (
            get_group2(),
            1 * 1e6,
            100 * 1e6,
            {"time_units": "us"},
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            )[:, np.newaxis],
        ),
    ],
)
def test_crosscorrelogram(group, binsize, windowsize, kwargs, expected):
    cc = nap.compute_crosscorrelogram(group, binsize, windowsize, **kwargs)
    assert isinstance(cc, pd.DataFrame)
    if isinstance(group, nap.TsGroup):
        assert list(cc.keys()) == list(combinations(group.keys(), 2))
    else:
        assert list(cc.keys()) == [(0, 0)]
    if "time_units" in kwargs:
        if kwargs["time_units"] == "ms":
            np.testing.assert_array_almost_equal(
                cc.index.values * 1e3,
                np.arange(-windowsize, windowsize + binsize, binsize),
            )
        if kwargs["time_units"] == "us":
            np.testing.assert_array_almost_equal(
                cc.index.values * 1e6,
                np.arange(-windowsize, windowsize + binsize, binsize),
            )
        if kwargs["time_units"] == "s":
            np.testing.assert_array_almost_equal(
                cc.index.values, np.arange(-windowsize, windowsize + binsize, binsize)
            )
    else:
        np.testing.assert_array_almost_equal(
            cc.index.values, np.arange(-windowsize, windowsize + binsize, binsize)
        )
    np.testing.assert_array_almost_equal(cc.values, expected)


def test_crosscorrelogram_reverse():
    cc = nap.compute_crosscorrelogram(get_group2(), 1, 100, reverse=True)
    assert isinstance(cc, pd.DataFrame)
    assert list(cc.keys()) == [(1, 0)]


@pytest.mark.parametrize(
    "args, expectation",
    [
        # data
        (
            ([],),
            pytest.raises(
                TypeError,
                match="data should be a Ts, TsGroup, Tsd, TsdFrame, TsdTensor.",
            ),
        ),
        (
            (nap.Ts([1, 2]),),
            does_not_raise(),
        ),
        (
            (get_group(),),
            does_not_raise(),
        ),
        (
            (nap.Tsd(t=[1, 2], d=[1, 1]),),
            does_not_raise(),
        ),
        (
            (nap.TsdFrame(t=[1, 2, 3], d=np.ones((3, 2))),),
            does_not_raise(),
        ),
        (
            (nap.TsdTensor(t=[1, 2, 3], d=np.ones((3, 2, 2))),),
            does_not_raise(),
        ),
        # bins
        (
            (get_group(), 2.0),
            pytest.raises(
                TypeError, match="bins should be either int, list or np.ndarray."
            ),
        ),
        (
            (get_group(), "2.0"),
            pytest.raises(
                TypeError, match="bins should be either int, list or np.ndarray."
            ),
        ),
        ((get_group(), 2), does_not_raise()),
        ((get_group(), [1, 2, 3]), does_not_raise()),
        ((get_group(), np.array((10,))), does_not_raise()),
        # log_scale
        (
            (get_group(), 2, []),
            pytest.raises(TypeError, match="log_scale should be of type bool."),
        ),
        ((get_group(), 2, True), does_not_raise()),
        # epochs
        (
            (get_group(), 2, True, [0, 100]),
            pytest.raises(
                TypeError, match="epochs should be an object of type IntervalSet"
            ),
        ),
        ((get_group(), 2, True, nap.IntervalSet([0, 100])), does_not_raise()),
    ],
)
def test_compute_isi_distribution_type_errors(args, expectation):
    with expectation:
        nap.compute_isi_distribution(*args)


@pytest.mark.parametrize(
    "args, expectation",
    [
        (
            (get_group(), -1),
            pytest.raises(ValueError, match="`bins` must be positive, when an integer"),
        ),
        (
            (get_group(), [1, 2, 3, 2, 4]),
            pytest.raises(
                ValueError, match="`bins` must increase monotonically, when an array"
            ),
        ),
        (
            (get_group(), np.ones((10, 2))),
            pytest.raises(ValueError, match="`bins` must be 1d, when an array"),
        ),
    ],
)
def test_compute_isi_distribution_value_errors(args, expectation):
    with expectation:
        nap.compute_isi_distribution(*args)


@pytest.mark.parametrize(
    "data",
    [
        nap.Ts(t=np.sort(np.random.uniform(0, 1000, 2000))),
        nap.TsGroup(
            {
                0: nap.Ts(t=np.sort(np.random.uniform(0, 1000, 2000))),
                1: nap.Ts(t=np.sort(np.random.uniform(0, 1000, 1000))),
            }
        ),
        nap.Tsd(t=np.sort(np.random.uniform(0, 1000, 1000)), d=np.ones(1000)),
        nap.TsdFrame(t=np.sort(np.random.uniform(0, 1000, 1000)), d=np.ones((1000, 2))),
        nap.TsdTensor(
            t=np.sort(np.random.uniform(0, 1000, 1000)), d=np.ones((1000, 2, 2))
        ),
    ],
)
@pytest.mark.parametrize(
    "bins",
    [
        1,
        10,
        list(range(0, 10)),
        np.linspace(0, 10, 10),
        np.linspace(0, 2000, 100),
        np.linspace(0, 2000, 1000),
        np.geomspace(1, 100, 100),
    ],
)
@pytest.mark.parametrize(
    "epochs",
    [
        None,
        nap.IntervalSet([0, 10]),
        nap.IntervalSet([0, 100]),
        nap.IntervalSet([0, 2000]),
        nap.IntervalSet([0, 4000]),
        nap.IntervalSet([0, 11, 21], [10, 20, 60]),
    ],
)
def test_compute_isi_distribution(data, bins, epochs):
    actual = nap.compute_isi_distribution(data, bins=bins, epochs=epochs)
    assert isinstance(actual, pd.DataFrame)

    time_diff = data.time_diff(epochs=epochs)
    if not isinstance(time_diff, dict):
        time_diff = {0: time_diff}
    if isinstance(data, nap.TsGroup) and isinstance(bins, int):
        min_isi = min([isi for isis in time_diff.values() for isi in isis])
        max_isi = max([isi for isis in time_diff.values() for isi in isis])
        bins = np.linspace(min_isi, max_isi, bins + 1)

    for i in time_diff:
        expected_values, expected_edges = np.histogram(time_diff[i].values, bins=bins)
        expected_index = expected_edges[:-1] + np.diff(expected_edges) / 2
        np.testing.assert_array_almost_equal(actual[i].to_numpy(), expected_values)
        np.testing.assert_array_almost_equal(actual.index, expected_index)

    np.testing.assert_array_almost_equal(
        actual.columns, list(data.keys()) if isinstance(data, nap.TsGroup) else [0]
    )
