# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-01 09:57:55
# @Last Modified by:   gviejo
# @Last Modified time: 2023-10-18 11:18:07
#!/usr/bin/env python

"""Tests of time series for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest


def test_create_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    assert isinstance(tsd, nap.Tsd)

def test_create_empty_tsd():
    tsd = nap.Tsd(t=np.array([]), d=np.array([]))
    assert len(tsd) == 0

def test_create_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 4))
    assert isinstance(tsdframe, nap.TsdFrame)

    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 4), columns=['a', 'b', 'c', 'd'])
    assert isinstance(tsdframe, nap.TsdFrame)
    assert np.all(tsdframe.columns == np.array(['a', 'b', 'c', 'd']))

def test_create_empty_tsdframe():
    tsdframe = nap.TsdFrame(t=np.array([]), d=np.array([]))
    assert len(tsdframe) == 0
    assert isinstance(tsdframe, nap.TsdFrame)

    with pytest.raises(RuntimeError):
        tsdframe = nap.TsdFrame(t=np.arange(100))

def test_create_1d_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100))
    assert isinstance(tsdframe, nap.TsdFrame)

def test_create_ts():
    ts = nap.Ts(t=np.arange(100))
    assert isinstance(ts, nap.Ts)


def test_create_ts_from_us():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="us")
    np.testing.assert_array_almost_equal(ts.index, a / 1000 / 1000)


def test_create_ts_from_ms():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="ms")
    np.testing.assert_array_almost_equal(ts.index, a / 1000)


def test_create_ts_from_s():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    ts = nap.Ts(t=a, time_units="s")
    np.testing.assert_array_almost_equal(ts.index, a)


def test_create_tsdframe_from_us():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="us")
    np.testing.assert_array_almost_equal(tsdframe.index, a / 1000 / 1000)


def test_create_tsdframe_from_ms():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="ms")
    np.testing.assert_array_almost_equal(tsdframe.index, a / 1000)


def test_create_tsdframe_from_s():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    a.sort()
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="s")
    np.testing.assert_array_almost_equal(tsdframe.index, a)


def test_create_ts_wrong_units():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    with pytest.raises(ValueError):
        nap.Ts(t=a, time_units="min")

def test_create_tsd_wrong_units():
    a = np.random.randint(0, 1000, 100)
    a.sort()
    with pytest.raises(ValueError):
        nap.Tsd(t=a, d=a, time_units="min")

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
    np.testing.assert_array_almost_equal(ts.index, np.sort(a))


@pytest.mark.filterwarnings("ignore")
def test_create_tsdframe_from_non_sorted():
    a = np.random.randint(0, 1000, 100)
    d = np.random.rand(100, 3)
    tsdframe = nap.TsdFrame(t=a, d=d, time_units="s")
    np.testing.assert_array_almost_equal(tsdframe.index, np.sort(a))


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

def test_concatenate_tsd():

    func = nap.core.time_series._concatenate_tsd

    tsd1 = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 5), columns = ['a', 'b', 'c', 'd', 'e'])
    tsd2 = nap.TsdFrame(t=np.arange(100)+100, d=np.random.rand(100, 5))
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))

    with pytest.raises(TypeError):
        func(np.concatenate, 1)

    with pytest.raises(AssertionError, match=r"Inputs should be Tsd, TsdFrame or TsdTensor"):
        func(np.concatenate, [0, tsd1, 2])

    with pytest.raises(AssertionError, match=r"Objects should all be the same."):
        func(np.concatenate, [tsd, tsd1])

    with pytest.raises(RuntimeError, match=r"The order of the Tsd index should be strictly increasing and non overlapping."):
        func(np.concatenate, [tsd2, tsd1])

    new_tsd = func(np.concatenate, (tsd1, tsd2))

    np.testing.assert_array_almost_equal(new_tsd.values, np.concatenate((tsd1.values, tsd2.values)))

    assert np.all(new_tsd.columns == tsd1.columns)

    new_tsd = func(np.concatenate, (tsd,))

    np.testing.assert_array_equal(new_tsd.values, tsd.values)

def test_create_tsdtensor():
    tsd = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 3, 2))
    assert isinstance(tsd, nap.TsdTensor)

    tsd = nap.TsdTensor(t=list(np.arange(100)), d=list(np.random.rand(100, 3, 2)))
    assert isinstance(tsd, nap.TsdTensor)    

def test_create_empty_tsd():
    tsd = nap.TsdTensor(t=np.array([]), d=np.array([[[]]]).T)
    assert len(tsd) == 0

def test_raise_error_tsdtensor_init():
    with pytest.raises(RuntimeError, match=r"Missing argument d when initializing TsdTensor"):
        nap.TsdTensor(t=np.arange(100), d=None)

    with pytest.raises(AssertionError, match=r"Data should have more than 2 dimensions. If ndim < 3, use TsdFrame or Tsd object"):
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 10))

    with pytest.raises(ValueError):#, match=r"Length of values (10) does not match length of index (100)"):
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(10, 10,3))

def test_index_error():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    with pytest.raises(IndexError):
        tsd[1000] = 0

    ts = nap.Ts(t=np.arange(100))
    with pytest.raises(IndexError):
        ts[1000]

def test_find_support():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    ep = tsd.find_support(1.0)
    assert ep.loc[0, 'start'] == 0
    assert ep.loc[0, 'end'] == 99.0 + 1e-6

    t = np.hstack((np.arange(10), np.arange(20, 30)))
    tsd = nap.Tsd(t=t, d=np.arange(20))
    ep = tsd.find_support(1.0)
    np.testing.assert_array_equal(ep.start.values, np.array([0.0, 20.0]))
    np.testing.assert_array_equal(ep.end.values, np.array([9.0+1e-6, 29+1e-6]))

def test_properties():
    t = np.arange(100)
    d = np.random.rand(100)
    tsd = nap.Tsd(t=t, d = d)

    assert hasattr(tsd, "t")
    assert hasattr(tsd, "d")
    assert hasattr(tsd, "start")
    assert hasattr(tsd, "end")
    assert hasattr(tsd, "shape")
    assert hasattr(tsd, "ndim")
    assert hasattr(tsd, "size")

    np.testing.assert_array_equal(tsd.t, t)
    np.testing.assert_array_equal(tsd.d, d)
    assert tsd.start == 0.0
    assert tsd.end == 99.0
    assert tsd.shape == (100,)
    assert tsd.ndim == 1
    assert tsd.size == 100

####################################################
# General test for time series
####################################################
@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.arange(100), time_units="s"),
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 5), time_units="s"),
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 2), time_units="s"),
        nap.Ts(t=np.arange(100), time_units="s"),
    ],
)
class Test_Time_Series_1:
    def test_as_units(self, tsd):
        if hasattr(tsd, "as_units"):
            tmp2 = tsd.index
            np.testing.assert_array_almost_equal(tsd.as_units("s").index, tmp2)
            np.testing.assert_array_almost_equal(
                tsd.as_units("ms").index, tmp2 * 1e3
            )
            np.testing.assert_array_almost_equal(
                tsd.as_units("us").index, tmp2 * 1e6
            )
            # np.testing.assert_array_almost_equal(tsd.as_units(units="a").index.values, tmp2)

    def test_rate(self, tsd):
        rate = len(tsd) / tsd.time_support.tot_length("s")
        np.testing.assert_approx_equal(tsd.rate, rate)

    def test_times(self, tsd):
        tmp = tsd.index
        np.testing.assert_array_almost_equal(tsd.times("s"), tmp)
        np.testing.assert_array_almost_equal(tsd.times("ms"), tmp * 1e3)
        np.testing.assert_array_almost_equal(tsd.times("us"), tmp * 1e6)

    def test_start_end(self, tsd):
        assert tsd.start_time() == tsd.index[0]
        assert tsd.end_time() == tsd.index[-1]
        assert tsd.start == tsd.index[0]
        assert tsd.end == tsd.index[-1]

    def test_time_support_interval_set(self, tsd):
        assert isinstance(tsd.time_support, nap.IntervalSet)

    def test_time_support_start_end(self, tsd):
        np.testing.assert_approx_equal(tsd.time_support.start[0], 0)
        np.testing.assert_approx_equal(tsd.time_support.end[0], 99.0)

    def test_time_support_include_tsd(self, tsd):
        np.testing.assert_approx_equal(tsd.time_support.start[0], tsd.index[0])
        np.testing.assert_approx_equal(tsd.time_support.end[0], tsd.index[-1])

    def test_value_from_tsd(self, tsd):
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2)
        assert len(tsd) == len(tsd3)
        np.testing.assert_array_almost_equal(tsd2.values[::10], tsd3.values)

    def test_value_from_tsdframe(self, tsd):
        tsdframe = nap.TsdFrame(t=np.arange(0, 100, 0.1), d=np.random.rand(1000,3))
        tsdframe2 = tsd.value_from(tsdframe)
        assert len(tsd) == len(tsdframe2)
        np.testing.assert_array_almost_equal(tsdframe.values[::10], tsdframe2.values)

    def test_value_from_value_error(self, tsd):
        with pytest.raises(RuntimeError):
            tsd.value_from(np.arange(10))
        
    def test_value_from_with_restrict(self, tsd):
        ep = nap.IntervalSet(start=0, end=50, time_units="s")
        tsd2 = nap.Tsd(t=np.arange(0, 100, 0.1), d=np.random.rand(1000))
        tsd3 = tsd.value_from(tsd2, ep)
        assert len(tsd.restrict(ep)) == len(tsd3)
        np.testing.assert_array_almost_equal(
            tsd2.restrict(ep).values[::10], tsd3.values
        )

        tsdframe = nap.TsdFrame(t=np.arange(0, 100, 0.1), d=np.random.rand(1000,2))
        tsdframe2 = tsd.value_from(tsdframe, ep)
        assert len(tsd.restrict(ep)) == len(tsdframe2)
        np.testing.assert_array_almost_equal(
            tsdframe.restrict(ep).values[::10], tsdframe2.values
        )

    def test_restrict(self, tsd):
        ep = nap.IntervalSet(start=0, end=50)
        assert len(tsd.restrict(ep)) == 51

    def test_restrict_error(self, tsd):        
        with pytest.raises(AssertionError, match=r"Argument should be IntervalSet"):
            tsd.restrict([0, 1])

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

    def test__getitems__(self, tsd):
        a = tsd[0:10]
        b = nap.Tsd(t=tsd.index[0:10], d=tsd.values[0:10])
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(a.index, b.index)
        np.testing.assert_array_almost_equal(a.values, b.values)
        pd.testing.assert_frame_equal(
            a.time_support, tsd.time_support
            )

    # def test_loc(self, tsd):
    #     a = tsd.loc[0:10] # should be 11 elements similar to pandas Series
    #     b = nap.Tsd(t=tsd.index[0:11], d=tsd.values[0:11])
    #     assert isinstance(a, nap.Tsd)
    #     np.testing.assert_array_almost_equal(a.index, b.index)
    #     np.testing.assert_array_almost_equal(a.values, b.values)
    #     pd.testing.assert_frame_equal(
    #         a.time_support, b.time_support
    #         )

    def test_count(self, tsd):
        count = tsd.count(1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))
        
        count = tsd.count(bin_size=1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

    def test_count_time_units(self, tsd):
        for b, tu in zip([1, 1e3, 1e6],['s', 'ms', 'us']):
            count = tsd.count(b, time_units = tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))
            
            count = tsd.count(b, tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))                

    def test_count_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=100)
        count = tsd.count(1, ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

        count = tsd.count(1, ep=ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

    def test_count_with_ep_only(self, tsd):
        ep = nap.IntervalSet(start=0, end=100)
        count = tsd.count(ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = tsd.count(ep=ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = tsd.count()
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))


    def test_count_errors(self, tsd):        
        with pytest.raises(ValueError):
            tsd.count(bin_size = {})

        with pytest.raises(ValueError):
            tsd.count(ep = {})

        with pytest.raises(ValueError):
            tsd.count(time_units = {})

    def test_bin_average(self, tsd):
        meantsd = tsd.bin_average(10)
        assert len(meantsd) == 10
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 100, 10))
        bins = np.arange(tsd.time_support.start[0], tsd.time_support.end[0]+1, 10)
        idx = np.digitize(tsd.index, bins)
        tmp = np.array([np.mean(tsd.values[idx==i]) for i in np.unique(idx)])        
        np.testing.assert_array_almost_equal(meantsd.values, tmp)

    def test_bin_average_with_ep(self, tsd):
        ep = nap.IntervalSet(start=0, end=40)
        meantsd = tsd.bin_average(10, ep)
        assert len(meantsd) == 4
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 40, 10))
        bins = np.arange(ep.start[0], ep.end[0]+10, 10)
        # tsd = tsd.restrict(ep)
        idx = np.digitize(tsd.index, bins)
        tmp = np.array([np.mean(tsd.values[idx==i]) for i in np.unique(idx)])                        
        np.testing.assert_array_almost_equal(meantsd.values, tmp[0:-1])

    def test_threshold(self, tsd):
        thrs = tsd.threshold(0.5, "above")
        assert len(thrs) == np.sum(tsd.values > 0.5)
        thrs = tsd.threshold(0.5, "below")
        assert len(thrs) == np.sum(tsd.values < 0.5)
        thrs = tsd.threshold(0.5, "aboveequal")
        assert len(thrs) == np.sum(tsd.values >= 0.5)
        thrs = tsd.threshold(0.5, "belowequal")
        assert len(thrs) == np.sum(tsd.values <= 0.5)

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

    def test_repr_(self, tsd):
        # assert pd.Series(tsd).__repr__() == tsd.__repr__()
        assert isinstance(tsd.__repr__(), str)

    def test_str_(self, tsd):
        # assert pd.Series(tsd).__str__() == tsd.__str__()
        assert isinstance(tsd.__str__(), str)

    def test_to_tsgroup(self, tsd):
        t = []
        d = []
        group = {}
        for i in range(3):
            t.append(np.sort(np.random.rand(10)*100))
            d.append(np.ones(10)*i)
            group[i] = nap.Ts(t=t[-1])

        times = np.array(t).flatten()
        data = np.array(d).flatten()
        idx = np.argsort(times)
        times = times[idx]
        data = data[idx]

        tsd = nap.Tsd(t=times, d=data)

        tsgroup = tsd.to_tsgroup()

        assert len(tsgroup) == 3
        np.testing.assert_array_almost_equal(np.arange(3), tsgroup.index)
        for i in range(3):
            np.testing.assert_array_almost_equal(tsgroup[i].index, t[i])
        
    def test_save_npz(self, tsd):
        import os

        with pytest.raises(RuntimeError) as e:
            tsd.save(dict)
        assert str(e.value) == "Invalid type; please provide filename as string"

        with pytest.raises(RuntimeError) as e:
            tsd.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format("./")

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsd.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(fake_path)

        tsd.save("tsd.npz")
        os.listdir('.')
        assert "tsd.npz" in os.listdir(".")

        tsd.save("tsd2")
        os.listdir('.')
        assert "tsd2.npz" in os.listdir(".")

        file = np.load("tsd.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'd' in keys
        assert 'start' in keys
        assert 'end' in keys

        np.testing.assert_array_almost_equal(file['t'], tsd.index)
        np.testing.assert_array_almost_equal(file['d'], tsd.values)
        np.testing.assert_array_almost_equal(file['start'], tsd.time_support.start.values)
        np.testing.assert_array_almost_equal(file['end'], tsd.time_support.end.values)

        os.remove("tsd.npz")
        os.remove("tsd2.npz")

    def test_interpolate(self, tsd):
        
        y = np.arange(0, 1001)

        tsd = nap.Tsd(t=np.arange(0, 101), d=y[0::10])

        # Ts
        ts = nap.Ts(t=y/10)
        tsd2 = tsd.interpolate(ts)
        np.testing.assert_array_almost_equal(tsd2.values, y)

        # Tsd
        ts = nap.Tsd(t=y/10, d=np.zeros_like(y))
        tsd2 = tsd.interpolate(ts)
        np.testing.assert_array_almost_equal(tsd2.values, y)

        # TsdFrame
        ts = nap.TsdFrame(t=y/10, d=np.zeros((len(y), 2)))
        tsd2 = tsd.interpolate(ts)
        np.testing.assert_array_almost_equal(tsd2.values, y)
        
        with pytest.raises(RuntimeError) as e:
            tsd.interpolate([0, 1, 2])
        assert str(e.value) == "First argument should be an instance of Ts, Tsd or TsdFrame"

        # Right left
        ep = nap.IntervalSet(start=0, end=5)
        tsd = nap.Tsd(t=np.arange(1,4), d=np.arange(3), time_support=ep)
        ts = nap.Ts(t=np.arange(0, 5))
        tsd2 = tsd.interpolate(ts, left=1234)
        assert float(tsd2.values[0]) == 1234.0
        tsd2 = tsd.interpolate(ts, right=4321)
        assert float(tsd2.values[-1]) == 4321.0

    def test_interpolate_with_ep(self, tsd):        
        y = np.arange(0, 1001)
        tsd = nap.Tsd(t=np.arange(0, 101), d=y[0::10])        
        ts = nap.Ts(t=y/10)
        ep = nap.IntervalSet(start=np.arange(0, 100, 20), end=np.arange(10, 110, 20))
        tsd2 = tsd.interpolate(ts, ep)
        tmp = ts.restrict(ep).index*10
        np.testing.assert_array_almost_equal(tmp, tsd2.values)

        # Empty ep
        ep = nap.IntervalSet(start=200, end=300)
        tsd2 = tsd.interpolate(ts, ep)
        assert len(tsd2) == 0

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
        assert isinstance(tsdframe[:,0], nap.Tsd)
        np.testing.assert_array_almost_equal(tsdframe[:,0].values, tsdframe.values[:,0])
        assert isinstance(tsdframe[:,0].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdframe.time_support, tsdframe[:,0].time_support)
        
        assert isinstance(tsdframe[:,[0,2]], nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.values[:,[0,2]], tsdframe[:,[0,2]].values)
        assert isinstance(tsdframe[:,[0,2]].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdframe.time_support, tsdframe[:,[0,2]].time_support)

    def test_vertical_slicing(self, tsdframe):
        assert isinstance(tsdframe[0:10], nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.values[0:10], tsdframe[0:10].values)
        assert isinstance(tsdframe[0:10].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdframe[0:10].time_support, tsdframe.time_support)

    def test_str_indexing(self, tsdframe):
        tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), time_units="s", columns=['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(tsdframe.values[:,0], tsdframe['a'])
        np.testing.assert_array_almost_equal(tsdframe.values[:,[0,2]], tsdframe[['a', 'c']])

    def test_operators(self, tsdframe):
        v = tsdframe.values

        a = tsdframe + 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v + 0.5))

        a = tsdframe - 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v - 0.5))

        a = tsdframe * 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v * 0.5))

        a = tsdframe / 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v / 0.5))

        a = tsdframe // 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v // 0.5))        

        a = tsdframe % 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v % 0.5))

        a = tsdframe ** 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        np.testing.assert_array_almost_equal(a.values, v ** 0.5)

        a = tsdframe > 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v > 0.5))

        a = tsdframe >= 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v >= 0.5))

        a = tsdframe < 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v < 0.5))

        a = tsdframe <= 0.5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v <= 0.5))        

        tsdframe = nap.TsdFrame(t=np.arange(10), d=np.tile(np.arange(10)[:,np.newaxis], 3))
        v = tsdframe.values
        a = tsdframe == 5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v == 5))

        a = tsdframe != 5
        assert isinstance(a, nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.index, a.index)
        assert np.all(a.values == (v != 5))

    def test_repr_(self, tsdframe):
        # assert pd.DataFrame(tsdframe).__repr__() == tsdframe.__repr__()
        assert isinstance(tsdframe.__repr__(), str)

    def test_str_(self, tsdframe):
        # assert pd.DataFrame(tsdframe).__str__() == tsdframe.__str__()
        assert isinstance(tsdframe.__str__(), str)

    def test_data(self, tsdframe):
        np.testing.assert_array_almost_equal(tsdframe.values, tsdframe.data())

    def test_bin_average(self, tsdframe):
        meantsd = tsdframe.bin_average(10)
        assert len(meantsd) == 10
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 100, 10))
        bins = np.arange(tsdframe.time_support.start[0], tsdframe.time_support.end[0]+1, 10)
        tmp = tsdframe.as_dataframe().groupby(np.digitize(tsdframe.index, bins)).mean()
        np.testing.assert_array_almost_equal(meantsd.values, tmp.values)

    def test_bin_average_with_ep(self, tsdframe):
        ep = nap.IntervalSet(start=0, end=40)
        meantsd = tsdframe.bin_average(10, ep)
        assert len(meantsd) == 4
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 40, 10))
        bins = np.arange(ep.start[0], ep.end[0]+10, 10)
        tsdframe = tsdframe.restrict(ep)
        tmp = tsdframe.as_dataframe().groupby(np.digitize(tsdframe.index, bins)).mean()
        np.testing.assert_array_almost_equal(meantsd.values, tmp.loc[np.arange(1,5)].values)

    def test_save_npz(self, tsdframe):
        import os

        with pytest.raises(RuntimeError) as e:
            tsdframe.save(dict)
        assert str(e.value) == "Invalid type; please provide filename as string"

        with pytest.raises(RuntimeError) as e:
            tsdframe.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format("./")

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsdframe.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(fake_path)

        tsdframe.save("tsdframe.npz")
        os.listdir('.')
        assert "tsdframe.npz" in os.listdir(".")

        tsdframe.save("tsdframe2")
        os.listdir('.')
        assert "tsdframe2.npz" in os.listdir(".")

        file = np.load("tsdframe.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'd' in keys
        assert 'start' in keys
        assert 'end' in keys
        assert 'columns' in keys

        np.testing.assert_array_almost_equal(file['t'], tsdframe.index)
        np.testing.assert_array_almost_equal(file['d'], tsdframe.values)
        np.testing.assert_array_almost_equal(file['start'], tsdframe.time_support.start.values)
        np.testing.assert_array_almost_equal(file['end'], tsdframe.time_support.end.values)
        np.testing.assert_array_almost_equal(file['columns'], tsdframe.columns)

        os.remove("tsdframe.npz")
        os.remove("tsdframe2.npz")

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
        # assert pd.Series(ts).fillna("").__repr__() == ts.__repr__()
        assert isinstance(ts.__repr__(), str)        

    def test_str_(self, ts):
        # assert pd.Series(ts).fillna("").__str__() == ts.__str__()
        assert isinstance(ts.__str__(), str)

    def test_save_npz(self, ts):
        import os

        with pytest.raises(RuntimeError) as e:
            ts.save(dict)
        assert str(e.value) == "Invalid type; please provide filename as string"

        with pytest.raises(RuntimeError) as e:
            ts.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format("./")

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            ts.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(fake_path)

        ts.save("ts.npz")
        os.listdir('.')
        assert "ts.npz" in os.listdir(".")

        ts.save("ts2")
        os.listdir('.')
        assert "ts2.npz" in os.listdir(".")

        file = np.load("ts.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'start' in keys
        assert 'end' in keys

        np.testing.assert_array_almost_equal(file['t'], ts.index)
        np.testing.assert_array_almost_equal(file['start'], ts.time_support.start.values)
        np.testing.assert_array_almost_equal(file['end'], ts.time_support.end.values)

        os.remove("ts.npz")
        os.remove("ts2.npz")

    def test_fillna(self, ts):
        with pytest.raises(AssertionError):
            ts.fillna([1])

        tsd = ts.fillna(0)

        assert isinstance(tsd, nap.Tsd)

        np.testing.assert_array_equal(np.zeros(len(ts)), tsd.values)
        np.testing.assert_array_equal(ts.index.values, tsd.index.values)

    def test_set(self, ts):
        ts.__setitem__(0, 1)

    def test_get(self, ts):
        assert isinstance(ts[0], nap.Ts)
        assert len(ts[0]) == 1
        assert len(ts[0:10]) == 10
        assert len(ts[[0,2,5]]) == 3

        np.testing.assert_array_equal(ts[[0,2,5]].index, np.array([0, 2, 5]))

####################################################
# Test for tsdtensor
####################################################
@pytest.mark.parametrize(
    "tsdtensor",
    [
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 3,2), time_units="s"),
    ],
)
class Test_Time_Series_5:

    def test_return_ndarray(self, tsdtensor):
        np.testing.assert_array_equal(tsdtensor[0], tsdtensor.values[0])

    def test_horizontal_slicing(self, tsdtensor):
        assert isinstance(tsdtensor[:,0], nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdtensor[:,0].values, tsdtensor.values[:,0])
        assert isinstance(tsdtensor[:,0].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdtensor.time_support, tsdtensor[:,0].time_support)
        
        assert isinstance(tsdtensor[:,[0,2]], nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.values[:,[0,2]], tsdtensor[:,[0,2]].values)
        assert isinstance(tsdtensor[:,[0,2]].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdtensor.time_support, tsdtensor[:,[0,2]].time_support)

    def test_vertical_slicing(self, tsdtensor):
        assert isinstance(tsdtensor[0:10], nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.values[0:10], tsdtensor[0:10].values)
        assert isinstance(tsdtensor[0:10].time_support, nap.IntervalSet)
        pd.testing.assert_frame_equal(tsdtensor[0:10].time_support, tsdtensor.time_support)

    def test_operators(self, tsdtensor):
        v = tsdtensor.values

        a = tsdtensor + 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v + 0.5))

        a = tsdtensor - 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v - 0.5))

        a = tsdtensor * 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v * 0.5))

        a = tsdtensor / 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v / 0.5))

        a = tsdtensor // 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v // 0.5))        

        a = tsdtensor % 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v % 0.5))

        a = tsdtensor ** 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        np.testing.assert_array_almost_equal(a.values, v ** 0.5)

        a = tsdtensor > 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v > 0.5))

        a = tsdtensor >= 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v >= 0.5))

        a = tsdtensor < 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v < 0.5))

        a = tsdtensor <= 0.5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v <= 0.5))        

        tsdtensor = nap.TsdTensor(t=np.arange(10), d=np.atleast_3d(np.tile(np.arange(10)[:,np.newaxis], 3)))
        v = tsdtensor.values
        a = tsdtensor == 5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v == 5))

        a = tsdtensor != 5
        assert isinstance(a, nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.index, a.index)
        assert np.all(a.values == (v != 5))

    def test_repr_(self, tsdtensor):
        # assert pd.DataFrame(tsdtensor).__repr__() == tsdtensor.__repr__()
        assert isinstance(tsdtensor.__repr__(), str)

    def test_str_(self, tsdtensor):
        # assert pd.DataFrame(tsdtensor).__str__() == tsdtensor.__str__()
        assert isinstance(tsdtensor.__str__(), str)

    def test_data(self, tsdtensor):
        np.testing.assert_array_almost_equal(tsdtensor.values, tsdtensor.data())

    def test_bin_average(self, tsdtensor):
        meantsd = tsdtensor.bin_average(10)
        assert len(meantsd) == 10
        np.testing.assert_array_almost_equal(meantsd.index, np.arange(5, 100, 10))
        bins = np.arange(tsdtensor.time_support.start[0], tsdtensor.time_support.end[0]+1, 10)
        idx = np.digitize(tsdtensor.index, bins)
        tmp = []
        for i in np.unique(idx):
            tmp.append(np.mean(tsdtensor.values[idx==i],0))
        tmp = np.array(tmp)
        np.testing.assert_array_almost_equal(meantsd.values, tmp)

    def test_save_npz(self, tsdtensor):
        import os

        with pytest.raises(RuntimeError) as e:
            tsdtensor.save(dict)
        assert str(e.value) == "Invalid type; please provide filename as string"

        with pytest.raises(RuntimeError) as e:
            tsdtensor.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format("./")

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsdtensor.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(fake_path)

        tsdtensor.save("tsdtensor.npz")
        os.listdir('.')
        assert "tsdtensor.npz" in os.listdir(".")

        tsdtensor.save("tsdtensor2")
        os.listdir('.')
        assert "tsdtensor2.npz" in os.listdir(".")

        file = np.load("tsdtensor.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'd' in keys
        assert 'start' in keys
        assert 'end' in keys        

        np.testing.assert_array_almost_equal(file['t'], tsdtensor.index)
        np.testing.assert_array_almost_equal(file['d'], tsdtensor.values)
        np.testing.assert_array_almost_equal(file['start'], tsdtensor.time_support.start.values)
        np.testing.assert_array_almost_equal(file['end'], tsdtensor.time_support.end.values)

        os.remove("tsdtensor.npz")
        os.remove("tsdtensor2.npz")
