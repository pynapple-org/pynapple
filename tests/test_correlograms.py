"""Tests of correlograms for `pynapple` package."""

from contextlib import nullcontext as does_not_raise
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


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


@pytest.mark.parametrize("spacing", [0.005, 0.0025])
def test_autocorrelogram_lags_on_bin_edges(spacing):
    # Spikes every `spacing` s with 0.01 s bins: half the lags fall exactly on a
    # bin edge, and each belongs to the bin on its right. Bin [-0.095, -0.085)
    # then holds the lags -0.095 + k * spacing, minus those reaching before the
    # first spike.
    t = np.arange(0, 10, spacing)
    group = nap.TsGroup({0: nap.Ts(t=t)})
    cc = nap.compute_autocorrelogram(group, 0.01, 0.1, norm=False)
    lags = -0.095 + np.arange(int(round(0.01 / spacing))) * spacing
    count = sum(np.sum(t >= -lag - 1e-12) for lag in lags)
    np.testing.assert_allclose(cc.loc[-0.09].values, count / (len(t) * 0.01))


@pytest.mark.parametrize("spacing", [0.005, 0.0025])
def test_crosscorrelogram_lags_on_bin_edges(spacing):
    # A unit cross-correlated with an identical copy of itself must match its
    # autocorrelogram outside lag 0, including lags exactly on bin edges.
    t = np.arange(0, 10, spacing)
    group = nap.TsGroup({0: nap.Ts(t=t), 1: nap.Ts(t=t)})
    cc = nap.compute_crosscorrelogram(group, 0.01, 0.1, norm=False)
    ac = nap.compute_autocorrelogram(group, 0.01, 0.1, norm=False)
    nonzero = ac.index != 0
    np.testing.assert_array_equal(cc[(0, 1)].values[nonzero], ac[0].values[nonzero])
    cc2 = nap.compute_crosscorrelogram((group[[0]], group[[1]]), 0.01, 0.1, norm=False)
    np.testing.assert_array_equal(cc2.values, cc.values)


@pytest.mark.parametrize("spacing", [0.005, 0.0025])
def test_eventcorrelogram_lags_on_bin_edges(spacing):
    # Events correlated with a unit must match the cross-correlogram that takes
    # the events as reference, including lags exactly on bin edges.
    t = np.arange(0, 10, spacing)
    event = nap.Ts(t=t)
    group = nap.TsGroup({0: nap.Ts(t=t), 1: nap.Ts(t=t[::3])})
    ec = nap.compute_eventcorrelogram(group, event, 0.01, 0.1, norm=False)
    cc = nap.compute_crosscorrelogram(
        (nap.TsGroup({0: event}), group), 0.01, 0.1, norm=False
    )
    np.testing.assert_array_equal(ec.values, cc.values)


def test_crosscorrelogram_binsize_below_precision():
    group = nap.TsGroup({0: nap.Ts(t=np.array([0.0, 1.0])), 1: nap.Ts(t=[0.5, 2.0])})
    with pytest.raises(ValueError, match="binsize must be at least 1e-9 s"):
        nap.compute_crosscorrelogram(group, 1e-10, 1e-9)


@pytest.mark.parametrize(
    "binsize, time_units, expectation",
    [
        (1e-9, "s", does_not_raise()),
        (1e-6, "ms", does_not_raise()),
        (1e-3, "us", does_not_raise()),
        (0.5e-6, "ms", pytest.raises(ValueError, match="binsize must be at least")),
    ],
)
def test_binsize_precision_limit(binsize, time_units, expectation):
    """The smallest bin size is the time precision of pynapple (1 ns), in any
    time unit."""
    group = nap.TsGroup({0: nap.Ts(t=np.array([0.0, 1.0]))})
    with expectation:
        nap.compute_autocorrelogram(group, binsize, 2 * binsize, time_units=time_units)


def test_autocorrelogram_binsize_below_precision():
    group = nap.TsGroup({0: nap.Ts(t=np.array([0.0, 1.0]))})
    with pytest.raises(ValueError, match="binsize must be at least 1e-9 s"):
        nap.compute_autocorrelogram(group, 1e-10, 1e-9)


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
        _, bins = np.histogram(
            np.concatenate([isis.values for isis in time_diff.values()]), bins=bins
        )

    for i in time_diff:
        expected_values, expected_edges = np.histogram(time_diff[i].values, bins=bins)
        expected_index = expected_edges[:-1] + np.diff(expected_edges) / 2
        np.testing.assert_array_almost_equal(actual[i].to_numpy(), expected_values)
        np.testing.assert_array_almost_equal(actual.index, expected_index)

    np.testing.assert_array_almost_equal(
        actual.columns, list(data.keys()) if isinstance(data, nap.TsGroup) else [0]
    )


@pytest.mark.parametrize("data_type", ["Ts", "Tsd", "TsdFrame", "TsdTensor", "TsGroup"])
@pytest.mark.parametrize("n_events", [2, 4])
@pytest.mark.parametrize("bins", [1, 10])
@pytest.mark.parametrize("log_scale", [False, True])
def test_compute_isi_distribution_constant_intervals(
    data_type, n_events, bins, log_scale
):
    times = 1.0 + 2.0 * np.arange(n_events)
    epochs = nap.IntervalSet(0, 8)
    if data_type == "TsGroup":
        data = nap.TsGroup({0: nap.Ts(times), 1: nap.Ts(times)}, time_support=epochs)
    elif data_type == "Ts":
        data = nap.Ts(times, time_support=epochs)
    else:
        shape = {
            "Tsd": (n_events,),
            "TsdFrame": (n_events, 2),
            "TsdTensor": (n_events, 2, 2),
        }
        data = getattr(nap, data_type)(
            times, np.ones(shape[data_type]), time_support=epochs
        )

    intervals = np.diff(times)
    if log_scale:
        intervals = np.log(intervals)
    expected, edges = np.histogram(intervals, bins=bins)
    actual = nap.compute_isi_distribution(
        data, bins=bins, log_scale=log_scale, epochs=epochs
    )

    for column in actual:
        np.testing.assert_array_equal(actual[column], expected)
    np.testing.assert_allclose(actual.index, edges[:-1] + np.diff(edges) / 2)
    assert actual.index.is_unique


@pytest.mark.parametrize("n_units", [2, 5, 12])
@pytest.mark.parametrize("binsize, windowsize", [(0.005, 0.1), (0.01, 0.05)])
def test_cross_correlograms_matches_pairs(n_units, binsize, windowsize):
    """The parallel integer function gives the same values as a count of all
    the lags, pair by pair. Some spikes have equal times in two units, and one
    unit is empty."""
    from pynapple.process.correlograms import _cross_correlograms

    rng = np.random.default_rng(n_units)
    shared = np.array([2.0, 5.0, 5.0])
    units = {
        k: nap.Ts(np.sort(np.r_[rng.random(150) * 10, shared[: 1 + k % 3]]))
        for k in range(n_units)
    }
    units[n_units] = nap.Ts(np.array([]))
    group = nap.TsGroup(units, time_support=nap.IntervalSet(0, 10))
    order, offsets = group._ragged_index
    precision = 10.0**nap.nap_config.time_index_precision
    times = np.round(group._times[order] * precision).astype(np.int64)
    pairs = [(i, j) for i in range(len(group)) for j in range(len(group)) if i != j]
    ref = np.array([i for i, _ in pairs])
    target = np.array([j for _, j in pairs])
    rates, lags = _cross_correlograms(
        times, offsets, times, offsets, ref, target, binsize, windowsize, precision
    )

    # the expected bins, in integer time units
    nbins = int((windowsize * 2) // binsize)
    nbins = nbins + 1 if nbins % 2 == 0 else nbins
    binsize_int = int(np.round(binsize * precision))
    expected_lags = (np.arange(nbins) - nbins // 2) * binsize_int / precision
    np.testing.assert_array_equal(lags, expected_lags)

    # The lags are doubled, so that the bin edges are integers.
    edges = -nbins * binsize_int + np.arange(nbins + 1) * 2 * binsize_int
    for p, (i, j) in enumerate(pairs):
        t1 = times[offsets[i] : offsets[i + 1]]
        t2 = times[offsets[j] : offsets[j + 1]]
        d = 2 * (t2[None, :] - t1[:, None]).ravel()
        d = d[(d >= edges[0]) & (d < edges[-1])]
        counts = np.bincount(
            np.searchsorted(edges, d, side="right") - 1, minlength=nbins
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = counts / (len(t1) * binsize_int / precision)
        np.testing.assert_array_equal(rates[p], expected)
