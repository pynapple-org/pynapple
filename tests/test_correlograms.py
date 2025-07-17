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
