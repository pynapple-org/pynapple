
"""Tests of correlograms for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
from itertools import combinations


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
            nap.process.correlograms._cross_correlogram(np.arange(0, t), np.arange(0, t), 1, t)[0],
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
            time_support = nap.IntervalSet(0, 100)
        )

def get_ep():
    return nap.IntervalSet(start=0, end=100)


@pytest.mark.parametrize("func", [
    nap.compute_autocorrelogram,
    nap.compute_crosscorrelogram,
    nap.compute_eventcorrelogram])
@pytest.mark.parametrize("group, binsize, windowsize, ep, norm, time_units, msg", [
    (get_group(), "a", 10, get_ep(), True, "s", "Invalid type. Parameter binsize must be of type <class 'numbers.Number'>."),
    (get_group(), 1, "a", get_ep(), True, "s", "Invalid type. Parameter windowsize must be of type <class 'numbers.Number'>."),
    (get_group(), 1, 10, "a", True, "s", "Invalid type. Parameter ep must be of type <class 'pynapple.core.interval_set.IntervalSet'>."),
    (get_group(), 1, 10, get_ep(), "a", "s", "Invalid type. Parameter norm must be of type <class 'bool'>."),
    (get_group(), 1, 10, get_ep(), True, 1, "Invalid type. Parameter time_units must be of type <class 'str'>."),
])
def test_correlograms_type_errors(func, group, binsize, windowsize, ep, norm, time_units, msg):
    with pytest.raises(TypeError, match=msg):
        func(group=group, binsize=binsize, windowsize=windowsize, ep=ep, norm=norm, time_units=time_units)

@pytest.mark.parametrize("func, args, msg", [
    (nap.compute_autocorrelogram, ([1, 2, 3], 1, 1), "Invalid type. Parameter group must be of type TsGroup"),
    (nap.compute_crosscorrelogram, ([1, 2, 3], 1, 1), r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\)."),
    (nap.compute_crosscorrelogram, (([1, 2, 3]), 1, 1), r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\)."),
    (nap.compute_crosscorrelogram, ((get_group(), [1,2,3]), 1, 1), r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\)."),
    (nap.compute_crosscorrelogram, ((get_group(), get_group(), get_group()), 1, 1), r"Invalid type. Parameter group must be of type TsGroup or a tuple\/list of \(TsGroup, TsGroup\)."),
    (nap.compute_eventcorrelogram, ([1, 2, 3], 1, 1), "Invalid type. Parameter group must be of type TsGroup"),
])
def test_correlograms_type_errors_group(func, args, msg):
    with pytest.raises(TypeError, match=msg):
        func(*args)

#################################################
# Normal tests
#################################################

@pytest.mark.parametrize("group, binsize, windowsize, kwargs, expected", [
    (get_group(), 1, 100, {}, np.hstack((np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1]))[:,np.newaxis] ),
    (get_group(), 1, 100, {"norm":False}, np.hstack((np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1]))[:,np.newaxis] ),
    (nap.TsGroup({1:nap.Ts(t=np.array([0, 10]))}), 1, 100, {"norm":False}, np.hstack((np.zeros(90),np.array([0.5]),np.zeros((19)),np.array([0.5]),np.zeros((90))))[:,np.newaxis]),
    (get_group(), 1, 100, {"ep":get_ep()}, np.hstack((np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1]))[:,np.newaxis] ),
    (get_group(), 1, 100, {"time_units":"s"}, np.hstack((np.arange(0, 1, 1 / 100), np.zeros(1), np.arange(0, 1, 1 / 100)[::-1]))[:,np.newaxis] ),
])
def test_autocorrelogram(group, binsize, windowsize, kwargs, expected):
    cc = nap.compute_autocorrelogram(group, binsize, windowsize, **kwargs)
    assert isinstance(cc, pd.DataFrame)
    assert list(cc.keys()) == list(group.keys())
    np.testing.assert_array_almost_equal(cc.index.values, np.arange(-windowsize, windowsize+binsize, binsize))
    np.testing.assert_array_almost_equal(cc.values, expected)





@pytest.mark.parametrize(
    "group",[get_group()],
)
class Test_Correlograms:

    def test_autocorrelogram_time_units(self, group):
        cc = nap.compute_autocorrelogram(group, 1, 100, time_units="s")
        cc2 = nap.compute_autocorrelogram(group, 1 * 1e3, 100 * 1e3, time_units="ms")
        cc3 = nap.compute_autocorrelogram(group, 1 * 1e6, 100 * 1e6, time_units="us")
        pd.testing.assert_frame_equal(cc, cc2)
        pd.testing.assert_frame_equal(cc, cc3)

    def test_crosscorrelogram(self, group):
        cc = nap.compute_crosscorrelogram(group, 1, 100, norm=False)
        assert isinstance(cc, pd.DataFrame)
        assert list(cc.keys()) == list(combinations(group.keys(), 2))
        np.testing.assert_array_almost_equal(cc.index.values, np.arange(-100, 101, 1))
        np.testing.assert_array_almost_equal(
            cc[(0, 1)].values,
            np.hstack(
                (np.arange(0, 1, 1 / 100), np.ones(1), np.arange(0, 1, 1 / 100)[::-1])
            ),
        )

    def test_crosscorrelogram_reverse(self, group):
        cc = nap.compute_crosscorrelogram(group, 1, 100, reverse=True)
        assert isinstance(cc, pd.DataFrame)

        from itertools import combinations

        pairs = list(combinations(group.index, 2))
        pairs = list(map(lambda n: (n[1], n[0]), pairs))

        assert pairs == list(cc.keys())

    def test_crosscorrelogram_with_ep(self, group):
        ep = get_ep()
        cc = nap.compute_crosscorrelogram(group, 1, 100, ep=ep, norm=False)
        np.testing.assert_array_almost_equal(cc[(0, 1)].values, cc[(0, 3)].values)

    def test_crosscorrelogram_with_norm(self, group):
        cc = nap.compute_crosscorrelogram(group, 1, 100, norm=False)
        cc2 = nap.compute_crosscorrelogram(group, 1, 100, norm=True)
        tmp = group._metadata["rate"].values.astype("float")
        tmp = tmp[[t[1] for t in cc.columns]]
        np.testing.assert_array_almost_equal(cc / tmp, cc2)

    def test_crosscorrelogram_time_units(self, group):
        cc = nap.compute_crosscorrelogram(group, 1, 100, time_units="s")
        cc2 = nap.compute_crosscorrelogram(group, 1 * 1e3, 100 * 1e3, time_units="ms")
        cc3 = nap.compute_crosscorrelogram(group, 1 * 1e6, 100 * 1e6, time_units="us")
        pd.testing.assert_frame_equal(cc, cc2)
        pd.testing.assert_frame_equal(cc, cc3)


    def test_crosscorrelogram_with_tuple(self, group):
        from itertools import product
        groups = (group[[0,1]], group[[2,3]])
        cc = nap.compute_crosscorrelogram(groups, 1, 100, norm=False)

        assert isinstance(cc, pd.DataFrame)
        assert list(cc.keys()) == list(product(groups[0].keys(), groups[1].keys()))
        np.testing.assert_array_almost_equal(cc.index.values, np.arange(-100, 101, 1))

        cc2 = nap.compute_crosscorrelogram(group[[0,2]], 1, 100, norm=False)
        np.testing.assert_array_almost_equal(
            cc[(0, 2)].values,
            cc2[(0,2)].values
            )

    def test_eventcorrelogram(self, group):
        cc = nap.compute_eventcorrelogram(group, group[0], 1, 100, norm=False)
        cc2 = nap.compute_crosscorrelogram(group, 1, 100, norm=False)
        assert isinstance(cc, pd.DataFrame)
        assert list(cc.keys()) == list(group.keys())
        np.testing.assert_array_almost_equal(cc[1].values, cc2[(0, 1)].values)

    def test_eventcorrelogram_with_ep(self, group):
        ep = nap.IntervalSet(start=0, end=99)
        cc = nap.compute_eventcorrelogram(group, group[0], 1, 100, ep=ep, norm=False)
        cc2 = nap.compute_crosscorrelogram(group, 1, 100, ep=ep, norm=False)
        assert isinstance(cc, pd.DataFrame)
        assert list(cc.keys()) == list(group.keys())
        np.testing.assert_array_almost_equal(cc[1].values, cc2[(0, 1)].values)

    def test_eventcorrelogram_with_norm(self, group):
        cc = nap.compute_eventcorrelogram(group, group[0], 1, 100, norm=False)
        cc2 = nap.compute_eventcorrelogram(group, group[0], 1, 100, norm=True)
        # tmp = group._metadata["rate"].values.astype("float")
        tmp = group.get_info("rate").values
        np.testing.assert_array_almost_equal(cc / tmp, cc2)

    def test_eventcorrelogram_time_units(self, group):
        cc = nap.compute_eventcorrelogram(group, group[0], 1, 100, time_units="s")
        cc2 = nap.compute_eventcorrelogram(
            group, group[0], 1 * 1e3, 100 * 1e3, time_units="ms"
        )
        cc3 = nap.compute_eventcorrelogram(
            group, group[0], 1 * 1e6, 100 * 1e6, time_units="us"
        )
        pd.testing.assert_frame_equal(cc, cc2)
        pd.testing.assert_frame_equal(cc, cc3)


