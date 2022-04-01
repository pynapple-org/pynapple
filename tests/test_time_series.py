# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-01 09:57:55
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-01 09:58:45
#!/usr/bin/env python

"""Tests of time series for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest


def test_create_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    assert isinstance(tsd, pd.Series)

def test_create_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100),d=np.random.rand(100,4))
    assert isinstance(tsdframe, pd.DataFrame)

def test_create_ts():
    ts = nap.Ts(t=np.arange(100))
    assert isinstance(ts, pd.Series)

def test_create_ts_from_us():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units = 'us')
    np.testing.assert_array_almost_equal(ts.index.values, a/1000/1000)

def test_create_ts_from_ms():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units = 'ms')
    np.testing.assert_array_almost_equal(ts.index.values, a/1000)

def test_create_ts_from_s():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units = 's')
    np.testing.assert_array_almost_equal(ts.index.values, a)

def test_create_tsdframe_from_us():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units = 'us')
    np.testing.assert_array_almost_equal(tsdframe.index.values, a/1000/1000)

def test_create_tsdframe_from_ms():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units = 'ms')
    np.testing.assert_array_almost_equal(tsdframe.index.values, a/1000)

def test_create_tsdframe_from_s():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units = 's')
    np.testing.assert_array_almost_equal(tsdframe.index.values, a)

def test_create_ts_wrong_units():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    with pytest.raises(ValueError):        
        nap.Ts(t=a, time_units = 'min')

def test_create_tsdframe_wrong_units():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    with pytest.raises(ValueError):        
        nap.TsdFrame(t=a, d=d,time_units = 'min')

@pytest.mark.filterwarnings("ignore")
def test_create_ts_from_non_sorted():
    a = np.random.randint(0, 1000, 100)
    ts = nap.Ts(t=a, time_units='s')
    np.testing.assert_array_almost_equal(ts.index.values, np.sort(a))

@pytest.mark.filterwarnings("ignore")
def test_create_ts_from_non_sorted():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    tsdframe = nap.TsdFrame(t=a, d=d, time_units='s')
    np.testing.assert_array_almost_equal(tsdframe.index.values, np.sort(a))

def test_create_ts_with_time_support():
    ep = nap.IntervalSet(start=[0,20], end=[10,30])
    ts = nap.Ts(t=np.arange(100), time_units='s', time_support=ep)
    assert len(ts) == 22
    np.testing.assert_array_almost_equal(ts.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(ts.time_support.end, ep.end)

def test_create_tsd_with_time_support():
    ep = nap.IntervalSet(start=[0,20], end=[10,30])
    tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units='s', time_support=ep)
    assert len(tsd) == 22
    np.testing.assert_array_almost_equal(tsd.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(tsd.time_support.end, ep.end)

def test_create_tsdframe_with_time_support():
    ep = nap.IntervalSet(start=[0,20], end=[10,30])
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100,3), time_units='s', time_support=ep)
    assert len(tsdframe) == 22
    np.testing.assert_array_almost_equal(tsdframe.time_support.start, ep.start)
    np.testing.assert_array_almost_equal(tsdframe.time_support.end, ep.end)

####################################################
# General test for time series
####################################################
@pytest.mark.parametrize("tsd", [
    nap.Tsd(t=np.arange(100), d=np.arange(100), time_units = 's'),
    nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 5), time_units = 's'),
    nap.Ts(t=np.arange(100), time_units = 's')
    ])
class Test_Time_Series_1:

    def test_as_units(self, tsd):
        tmp2 = tsd.index.values
        np.testing.assert_array_almost_equal(tsd.as_units('s').index.values, tmp2)
        np.testing.assert_array_almost_equal(tsd.as_units('ms').index.values, tmp2*1e3)
        np.testing.assert_array_almost_equal(tsd.as_units('us').index.values, tmp2*1e6)

    def test_rate(self, tsd):
        rate = len(tsd)/tsd.time_support.tot_length('s')
        np.testing.assert_approx_equal(tsd.rate, rate)

    def test_times(self, tsd):
        tmp = tsd.index.values
        np.testing.assert_array_almost_equal(tsd.times('s'), tmp)
        np.testing.assert_array_almost_equal(tsd.times('ms'), tmp*1e3)
        np.testing.assert_array_almost_equal(tsd.times('us'), tmp*1e6)

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
        ep = nap.IntervalSet(start=0, end=50, time_units = 's')
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2, ep)
        assert len(tsd.restrict(ep)) == len(tsd3)
        np.testing.assert_array_almost_equal(tsd2.restrict(ep).values[::10], tsd3.values)

    def test_restrict(self, tsd):
        ep = nap.IntervalSet(start=0, end=50)
        assert len(tsd.restrict(ep)) == 51

    def test_restrict_multiple_epochs(self, tsd):
        ep = nap.IntervalSet(start=[0,20], end=[10,30])
        assert len(tsd.restrict(ep)) == 22

    def test_restrict_inherit_time_support(self, tsd):
        ep = nap.IntervalSet(start=0, end=50)
        tsd2 = tsd.restrict(ep)
        np.testing.assert_approx_equal(tsd2.time_support.start[0], ep.start[0])
        np.testing.assert_approx_equal(tsd2.time_support.end[0], ep.end[0])

####################################################
# Test for tsd
####################################################
@pytest.mark.parametrize("tsd", [
    nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units = 's'),
    ])
class Test_Time_Series_2:

    def test_as_series(self, tsd):
        assert isinstance(tsd.as_series(), pd.Series)

    def test_count(self, tsd):
        count = tsd.count(1)
        assert len(count)==99
        np.testing.assert_array_almost_equal(
            count.index.values,
            np.arange(0.5, 99, 1)
            )

    def test_count_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=100)
        count = tsd.count(1, ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(
            count.values,
            np.ones(100)
            )

    def test_threshold(self, tsd):
        thrs = tsd.threshold(0.5, 'above')
        assert len(thrs) == np.sum(tsd.values>0.5)
        thrs = tsd.threshold(0.5, 'below')
        assert len(thrs) == np.sum(tsd.values<0.5)

    def test_threshold_time_support(self, tsd):        
        thrs = tsd.threshold(0.5, 'above')
        time_support = thrs.time_support
        thrs2 = tsd.restrict(time_support)
        assert len(thrs2) == np.sum(tsd.values>0.5)

####################################################
# Test for tsdframe
####################################################
@pytest.mark.parametrize("tsdframe", [
    nap.TsdFrame(t=np.arange(100), d=np.random.rand(100,3), time_units = 's'),
    ])
class Test_Time_Series_3:

    def test_as_dataframe(self, tsdframe):
        assert isinstance(tsdframe.as_dataframe(), pd.DataFrame)

    def test_slicing(self, tsdframe):
        assert isinstance(tsdframe[0], nap.Tsd)