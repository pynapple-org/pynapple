# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-01 09:57:55
# @Last Modified by:   gviejo
# @Last Modified time: 2022-11-29 23:12:34
#!/usr/bin/env python

"""Tests of time series for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest


def test_create_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    assert isinstance(tsd, pd.Series)

def test_create_empty_tsd():
    tsd = nap.Tsd(t=np.array([]), d=np.array([]))
    assert len(tsd) == 0

def test_create_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 4))
    assert isinstance(tsdframe, pd.DataFrame)

    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 4), columns=['a', 'b', 'c', 'd'])
    assert isinstance(tsdframe, pd.DataFrame)
    assert np.all(tsdframe.columns == np.array(['a', 'b', 'c', 'd']))

def test_create_empty_tsdframe():
    tsdframe = nap.TsdFrame(t=np.array([]), d=np.array([]))
    assert len(tsdframe) == 0
    tsdframe = nap.TsdFrame(t=np.arange(100))
    assert len(tsdframe) == 100
    assert isinstance(tsdframe, nap.TsdFrame)

    tsdframe = nap.TsdFrame(t=np.array([]), d = np.empty(()))
    assert isinstance(tsdframe, nap.TsdFrame)
    assert len(tsdframe) == 0

    ep = nap.IntervalSet(start = 0, end = 9)
    tsdframe = nap.TsdFrame(t=np.arange(100), d = None, time_support=ep)
    assert len(tsdframe) == 10
    assert isinstance(tsdframe, nap.TsdFrame)    

def test_create_1d_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100))
    assert isinstance(tsdframe, pd.DataFrame)

def test_create_ts():
    ts = nap.Ts(t=np.arange(100))
    assert isinstance(ts, pd.Series)


def test_create_ts_from_us():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="us")
    np.testing.assert_array_almost_equal(ts.index.values, a / 1000 / 1000)


def test_create_ts_from_ms():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="ms")
    np.testing.assert_array_almost_equal(ts.index.values, a / 1000)


def test_create_ts_from_s():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="s")
    np.testing.assert_array_almost_equal(ts.index.values, a)


def test_create_tsdframe_from_us():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="us")
    np.testing.assert_array_almost_equal(tsdframe.index.values, a / 1000 / 1000)


def test_create_tsdframe_from_ms():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="ms")
    np.testing.assert_array_almost_equal(tsdframe.index.values, a / 1000)


def test_create_tsdframe_from_s():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="s")
    np.testing.assert_array_almost_equal(tsdframe.index.values, a)


def test_create_ts_wrong_units():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    with pytest.raises(ValueError):
        nap.Ts(t=a, time_units="min")


def test_create_tsdframe_wrong_units():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    with pytest.raises(ValueError):
        nap.TsdFrame(t=a, d=d, time_units="min")


@pytest.mark.filterwarnings("ignore")
def test_create_ts_from_non_sorted():
    a = np.random.randint(0, 1000, 100)
    ts = nap.Ts(t=a, time_units="s")
    np.testing.assert_array_almost_equal(ts.index.values, np.sort(a))


@pytest.mark.filterwarnings("ignore")
def test_create_ts_from_non_sorted():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="s")
    np.testing.assert_array_almost_equal(tsdframe.index.values, np.sort(a))


def test_create_ts_with_time_support():
    ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
    ts = nap.Ts(t=np.arange(100), time_units="s", time_support=ep)
    assert len(ts) == 22
    np.testing.assert_array_almost_equal(ts.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(ts.time_support.end, ep.end)


def test_create_tsd_with_time_support():
    ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
    tsd = nap.Tsd(
        t=np.arange(100), d=np.random.rand(100), time_units="s", time_support=ep
    )
    assert len(tsd) == 22
    np.testing.assert_array_almost_equal(tsd.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(tsd.time_support.end, ep.end)


def test_create_tsdframe_with_time_support():
    ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
    tsdframe = nap.TsdFrame(
        t=np.arange(100), d=np.random.rand(100, 3), time_units="s", time_support=ep
    )
    assert len(tsdframe) == 22
    np.testing.assert_array_almost_equal(tsdframe.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(tsdframe.time_support.end, ep.end)


####################################################
# General test for time series
####################################################
@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.arange(100), time_units="s"),
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 5), time_units="s"),
        nap.Ts(t=np.arange(100), time_units="s"),
    ],
)
class Test_Time_Series_1:
    def test_as_units(self, tsd):
        tmp2 = tsd.index.values
        np.testing.assert_array_almost_equal(tsd.as_units("s").index.values, tmp2)
        np.testing.assert_array_almost_equal(
            tsd.as_units("ms").index.values, tmp2 * 1e3
        )
        np.testing.assert_array_almost_equal(
            tsd.as_units("us").index.values, tmp2 * 1e6
        )
        # np.testing.assert_array_almost_equal(tsd.as_units(units="a").index.values, tmp2)

    def test_rate(self, tsd):
        rate = len(tsd) / tsd.time_support.tot_length("s")
        np.testing.assert_approx_equal(tsd.rate, rate)

    def test_times(self, tsd):
        tmp = tsd.index.values
        np.testing.assert_array_almost_equal(tsd.times("s"), tmp)
        np.testing.assert_array_almost_equal(tsd.times("ms"), tmp * 1e3)
        np.testing.assert_array_almost_equal(tsd.times("us"), tmp * 1e6)

    def test_start_end(self, tsd):
        assert tsd.start_time() == tsd.index.values[0]
        assert tsd.end_time() == tsd.index.values[-1]

    def test_time_support_interval_set(self, tsd):
        assert isinstance(tsd.time_support, nap.IntervalSet)

    def test_time_support_start_end(self, tsd):
        np.testing.assert_approx_equal(tsd.time_support.start[0], 0)
        np.testing.assert_approx_equal(tsd.time_support.end[0], 99.0)

    def test_time_support_include_tsd(self, tsd):
        np.testing.assert_approx_equal(tsd.time_support.start[0], tsd.index.values[0])
        np.testing.assert_approx_equal(tsd.time_support.end[0], tsd.index.values[-1])

    def test_value_from(self, tsd):
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2)
        assert len(tsd) == len(tsd3)
        np.testing.assert_array_almost_equal(tsd2.values[::10], tsd3.values)

    def test_value_from_with_restrict(self, tsd):
        ep = nap.IntervalSet(start=0, end=50, time_units="s")
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2, ep)
        assert len(tsd.restrict(ep)) == len(tsd3)
        np.testing.assert_array_almost_equal(
            tsd2.restrict(ep).values[::10], tsd3.values
        )

    def test_restrict(self, tsd):
        ep = nap.IntervalSet(start=0, end=50)
        assert len(tsd.restrict(ep)) == 51

    def test_restrict_multiple_epochs(self, tsd):
        ep = nap.IntervalSet(start=[0, 20], end=[10, 30])
        assert len(tsd.restrict(ep)) == 22

    def test_restrict_inherit_time_support(self, tsd):
        ep = nap.IntervalSet(start=0, end=50)
        tsd2 = tsd.restrict(ep)
        np.testing.assert_approx_equal(tsd2.time_support.start[0], ep.start[0])
        np.testing.assert_approx_equal(tsd2.time_support.end[0], ep.end[0])


####################################################
# Test for tsd
####################################################
@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s"),
    ],
)
class Test_Time_Series_2:
    def test_as_series(self, tsd):
        assert isinstance(tsd.as_series(), pd.Series)

    def test_SingleBlockManager(self, tsd):
        a = tsd.loc[0:10]
        b = nap.Tsd(t=tsd.index.values[0:11], d=tsd.values[0:11])
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(a.index.values, b.index.values)
        np.testing.assert_array_almost_equal(a.values, b.values)
        pd.testing.assert_frame_equal(
            a.time_support, b.time_support
            )

    def test_count(self, tsd):
        count = tsd.count(1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index.values, np.arange(0.5, 99, 1))

    def test_count_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=100)
        count = tsd.count(1, ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

    def test_bin_average(self, tsd):
        meantsd = tsd.bin_average(10)
        assert len(meantsd) == 10
        np.testing.assert_array_almost_equal(meantsd.index.values, np.arange(5, 100, 10))
        bins = np.arange(tsd.time_support.start[0], tsd.time_support.end[0]+1, 10)
        tmp = tsd.groupby(np.digitize(tsd.index.values, bins)).mean()
        np.testing.assert_array_almost_equal(meantsd.values, tmp.values)

    def test_bin_average_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=40)
        meantsd = tsd.bin_average(10, ep)
        assert len(meantsd) == 4
        np.testing.assert_array_almost_equal(meantsd.index.values, np.arange(5, 40, 10))
        bins = np.arange(ep.start[0], ep.end[0]+10, 10)
        tsd = tsd.restrict(ep)
        tmp = tsd.groupby(np.digitize(tsd.index.values, bins)).mean()
        np.testing.assert_array_almost_equal(meantsd.values, tmp.loc[np.arange(1,5)].values)

    def test_count_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=100)
        count = tsd.count(1, ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

    def test_threshold(self, tsd):
        thrs = tsd.threshold(0.5, "above")
        assert len(thrs) == np.sum(tsd.values > 0.5)
        thrs = tsd.threshold(0.5, "below")
        assert len(thrs) == np.sum(tsd.values < 0.5)

    def test_threshold_time_support(self, tsd):
        thrs = tsd.threshold(0.5, "above")
        time_support = thrs.time_support
        thrs2 = tsd.restrict(time_support)
        assert len(thrs2) == np.sum(tsd.values > 0.5)

        with pytest.raises(ValueError) as e_info:
            tsd.threshold(0.5, "bla")
        assert str(e_info.value) == "Method {} for thresholding is not accepted.".format("bla")


    def test_data(self, tsd):
        np.testing.assert_array_almost_equal(tsd.values, tsd.data())

    def test_operators(self, tsd):
        v = tsd.values

        a = tsd + 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v + 0.5))

        a = tsd - 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v - 0.5))

        a = tsd * 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v * 0.5))

        a = tsd / 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v / 0.5))

        a = tsd // 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v // 0.5))        

        a = tsd % 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v % 0.5))

        a = tsd ** 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v ** 0.5))

        a = tsd > 0.5
        assert isinstance(a, pd.Series)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v > 0.5))

        a = tsd >= 0.5
        assert isinstance(a, pd.Series)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v >= 0.5))

        a = tsd < 0.5
        assert isinstance(a, pd.Series)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v < 0.5))

        a = tsd <= 0.5
        assert isinstance(a, pd.Series)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v <= 0.5))        

        tsd = nap.Tsd(t=np.arange(10), d=np.arange(10))
        v = tsd.values
        a = tsd == 5
        assert isinstance(a, pd.Series)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v == 5))

        a = tsd != 5
        assert isinstance(a, pd.Series)
        np.testing.assert_array_almost_equal(tsd.index.values, a.index.values)
        assert np.all(a.values == (v != 5))

    def test_repr_(self, tsd):
        assert pd.Series(tsd).__repr__() == tsd.__repr__()

    def test_repr_(self, tsd):
        assert pd.Series(tsd).__str__() == tsd.__str__()


####################################################
# Test for tsdframe
####################################################
@pytest.mark.parametrize(
    "tsdframe",
    [
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), time_units="s"),
    ],
)
class Test_Time_Series_3:
    def test_as_dataframe(self, tsdframe):
        assert isinstance(tsdframe.as_dataframe(), pd.DataFrame)

    def test_horizontal_slicing(self, tsdframe):
        assert isinstance(tsdframe[0], nap.Tsd)
        np.testing.assert_array_almost_equal(tsdframe.values[:,0], tsdframe[0].values)
        assert isinstance(tsdframe[0].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdframe.time_support, tsdframe[0].time_support)
        
        assert isinstance(tsdframe[[0,2]], nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.values[:,[0,2]], tsdframe[[0,2]].values)
        assert isinstance(tsdframe[[0,2]].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdframe.time_support, tsdframe[[0,2]].time_support)

    def test_vertical_slicing(self, tsdframe):
        assert isinstance(tsdframe.loc[0:10], nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.values[0:11], tsdframe.loc[0:10].values)
        assert isinstance(tsdframe.loc[0:10].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdframe.loc[0:10].time_support, nap.IntervalSet(0, 10))

    def test_operators(self, tsdframe):
        v = tsdframe.values

        a = tsdframe + 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v + 0.5))

        a = tsdframe - 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v - 0.5))

        a = tsdframe * 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v * 0.5))

        a = tsdframe / 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v / 0.5))

        a = tsdframe // 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v // 0.5))        

        a = tsdframe % 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v % 0.5))

        a = tsdframe ** 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v ** 0.5))

        a = tsdframe > 0.5
        assert isinstance(a, pd.DataFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v > 0.5))

        a = tsdframe >= 0.5
        assert isinstance(a, pd.DataFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v >= 0.5))

        a = tsdframe < 0.5
        assert isinstance(a, pd.DataFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v < 0.5))

        a = tsdframe <= 0.5
        assert isinstance(a, pd.DataFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v <= 0.5))        

        tsdframe = nap.TsdFrame(t=np.arange(10), d=np.tile(np.arange(10)[:,np.newaxis], 3))
        v = tsdframe.values
        a = tsdframe == 5
        assert isinstance(a, pd.DataFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v == 5))

        a = tsdframe != 5
        assert isinstance(a, pd.DataFrame)
        np.testing.assert_array_almost_equal(tsdframe.index.values, a.index.values)
        assert np.all(a.values == (v != 5))

    def test_repr_(self, tsdframe):
        assert pd.DataFrame(tsdframe).__repr__() == tsdframe.__repr__()

    def test_repr_(self, tsdframe):
        assert pd.DataFrame(tsdframe).__str__() == tsdframe.__str__()

    def test_data(self, tsdframe):
        np.testing.assert_array_almost_equal(tsdframe.values, tsdframe.data())

    def test_bin_average(self, tsdframe):
        meantsd = tsdframe.bin_average(10)
        assert len(meantsd) == 10
        np.testing.assert_array_almost_equal(meantsd.index.values, np.arange(5, 100, 10))
        bins = np.arange(tsdframe.time_support.start[0], tsdframe.time_support.end[0]+1, 10)
        tmp = tsdframe.groupby(np.digitize(tsdframe.index.values, bins)).mean()
        np.testing.assert_array_almost_equal(meantsd.values, tmp.values)

    def test_bin_average_with_ep(self, tsdframe):
        ep = nap.IntervalSet(start=0, end=40)
        meantsd = tsdframe.bin_average(10, ep)
        assert len(meantsd) == 4
        np.testing.assert_array_almost_equal(meantsd.index.values, np.arange(5, 40, 10))
        bins = np.arange(ep.start[0], ep.end[0]+10, 10)
        tsdframe = tsdframe.restrict(ep)
        tmp = tsdframe.groupby(np.digitize(tsdframe.index.values, bins)).mean()
        np.testing.assert_array_almost_equal(meantsd.values, tmp.loc[np.arange(1,5)].values)


####################################################
# Test for ts
####################################################
@pytest.mark.parametrize(
    "ts",
    [
        nap.Ts(t=np.arange(100), time_units="s"),
    ],
)
class Test_Time_Series_4:

    def test_repr_(self, ts):
        assert pd.Series(ts).fillna("").__repr__() == ts.__repr__()

    def test_repr_(self, ts):
        assert pd.Series(ts).fillna("").__str__() == ts.__str__()
