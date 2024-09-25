"""Tests of time series for `pynapple` package."""

import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from contextlib import nullcontext as does_not_raise

import pynapple as nap

# tsd1 = nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s")
# tsd2 = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), columns = ['a', 'b', 'c'])
# tsd3 = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 4), time_units="s")
# tsd4 = nap.Ts(t=np.arange(100), time_units="s")


def test_create_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    assert isinstance(tsd, nap.Tsd)

def test_create_empty_tsd():
    tsd = nap.Tsd(t=np.array([]), d=np.array([]))
    assert len(tsd) == 0

@pytest.mark.filterwarnings("ignore")
def test_create_tsd_from_number():
    tsd = nap.Tsd(t=1, d=2)

def test_create_tsdframe():
    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 4))
    assert isinstance(tsdframe, nap.TsdFrame)

    tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 4), columns=['a', 'b', 'c', 'd'])
    assert isinstance(tsdframe, nap.TsdFrame)
    assert np.all(tsdframe.columns == np.array(['a', 'b', 'c', 'd']))

@pytest.mark.filterwarnings("ignore")
def test_create_empty_tsdframe():
    tsdframe = nap.TsdFrame(t=np.array([]), d=np.empty(shape=(0, 2)))
    assert len(tsdframe) == 0
    assert isinstance(tsdframe, nap.TsdFrame)

    with pytest.raises(AssertionError):
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

@pytest.mark.filterwarnings("ignore")
def test_create_tsdtensor():
    tsd = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 3, 2))
    assert isinstance(tsd, nap.TsdTensor)

    tsd = nap.TsdTensor(t=list(np.arange(100)), d=list(np.random.rand(100, 3, 2)))
    assert isinstance(tsd, nap.TsdTensor)    

def test_create_empty_tsd():
    tsd = nap.TsdTensor(t=np.array([]), d=np.empty(shape=(0,2,3)))
    assert len(tsd) == 0

@pytest.mark.filterwarnings("ignore")
def test_raise_error_tsdtensor_init():
    with pytest.raises(RuntimeError, match=r"Unknown format for d. Accepted formats are numpy.ndarray, list, tuple or any array-like objects."):
        nap.TsdTensor(t=np.arange(100), d=None)

    # with pytest.raises(AssertionError, match=r"Data should have more than 2 dimensions. If ndim < 3, use TsdFrame or Tsd object"):
    #     nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 10))

    with pytest.raises(AssertionError):#, match=r"Length of values (10) does not match length of index (100)"):
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(10, 10,3))

def test_index_error():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    ts = nap.Ts(t=np.arange(100))

    if isinstance(tsd.d, np.ndarray):
        with pytest.raises(IndexError):
            tsd[1000] = 0

        with pytest.raises(IndexError):
            ts[1000]

def test_find_support():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    ep = tsd.find_support(1.0)
    assert ep[0, 0] == 0
    assert ep[0, 1] == 99.0 + 1e-6

    t = np.hstack((np.arange(10), np.arange(20, 30)))
    tsd = nap.Tsd(t=t, d=np.arange(20))
    ep = tsd.find_support(1.0)
    np.testing.assert_array_equal(ep.start, np.array([0.0, 20.0]))
    np.testing.assert_array_equal(ep.end, np.array([9.0+1e-6, 29+1e-6]))

def test_properties():
    t = np.arange(100)
    d = np.random.rand(100).astype(np.float32) # to match pynajax
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

    with pytest.raises(RuntimeError):
        tsd.rate = 0

def test_base_tsd_class():
    class DummyTsd(nap.core.time_series.BaseTsd):
        def __init__(self, t, d):
            super().__init__(t, d)

        def __getitem__(self, key):
            return self.values.__getitem__(key)

    tsd = DummyTsd([0, 1], [1, 2])
    assert isinstance(tsd.rate, float)
    assert isinstance(tsd.index, nap.TsIndex)
    try:
        assert isinstance(tsd.values, np.ndarray)
    except AssertionError:
        assert nap.core.utils.is_array_like(tsd.values) # for pynajax

    assert isinstance(tsd.__repr__(), str)

    with pytest.raises((IndexError, TypeError)):
        tsd['a']


####################################################
# General test for time series
####################################################
@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s"),
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
        with pytest.raises(AssertionError, match=r"First argument should be an instance of Tsd, TsdFrame or TsdTensor"):
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

    def test_get_interval(self, tsd):
        tsd2 = tsd.get(10, 20)
        assert len(tsd2) == 11
        np.testing.assert_array_equal(tsd2.index.values, tsd.index.values[10:21])
        if not isinstance(tsd, nap.Ts):
            np.testing.assert_array_equal(tsd2.values, tsd.values[10:21])

        with pytest.raises(Exception):
            tsd.get(20, 10)

        with pytest.raises(Exception):
            tsd.get(10, [20])

        with pytest.raises(Exception):
            tsd.get([10], 20)

    def test_get_timepoint(self, tsd):
        if not isinstance(tsd, nap.Ts):
            np.testing.assert_array_equal(tsd.get(-1), tsd[0])
            np.testing.assert_array_equal(tsd.get(0), tsd[0])
            np.testing.assert_array_equal(tsd.get(0.1), tsd[0])
            np.testing.assert_array_equal(tsd.get(0.5), tsd[1])
            np.testing.assert_array_equal(tsd.get(0.6), tsd[1])
            np.testing.assert_array_equal(tsd.get(1), tsd[1])
            np.testing.assert_array_equal(tsd.get(1000), tsd[-1])

    def test_dropna(self, tsd):
        if not isinstance(tsd, nap.Ts):

            new_tsd = tsd.dropna()
            np.testing.assert_array_equal(tsd.index.values, new_tsd.index.values)
            np.testing.assert_array_equal(tsd.values, new_tsd.values)

            tmp = np.random.rand(*tsd.shape)
            tmp[tmp>0.9] = np.nan
            tsd = tsd.__class__(t=tsd.t, d=tmp)
            
            new_tsd = tsd.dropna()
            assert not np.all(np.isnan(new_tsd.values))
            tokeep = np.array([~np.any(np.isnan(tsd[i])) for i in range(len(tsd))])            
            np.testing.assert_array_equal(tsd.index.values[tokeep], new_tsd.index.values)
            np.testing.assert_array_equal(tsd.values[tokeep], new_tsd.values)

            newtsd2 = tsd.restrict(new_tsd.time_support)
            np.testing.assert_array_equal(newtsd2.index.values, new_tsd.index.values)
            np.testing.assert_array_equal(newtsd2.values, new_tsd.values)

            new_tsd = tsd.dropna(update_time_support=False)
            np.testing.assert_array_equal(tsd.index.values[tokeep], new_tsd.index.values)
            np.testing.assert_array_equal(tsd.values[tokeep], new_tsd.values)
            np.testing.assert_array_equal(new_tsd.time_support, tsd.time_support)

            tsd = tsd.__class__(t=tsd.t, d=np.ones(tsd.shape)*np.nan)            
            new_tsd = tsd.dropna()
            assert len(new_tsd) == 0
            assert len(new_tsd.time_support) == 0

    def test_convolve_raise_errors(self, tsd):
        if not isinstance(tsd, nap.Ts):

            with pytest.raises(IOError) as e_info:
                tsd.convolve([1,2,3])
            assert str(e_info.value) == "Input should be a numpy array (or jax array if pynajax is installed)."

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.array([]))
            assert str(e_info.value) == "Input array is length 0"

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.ones(3), trim='a')
            assert str(e_info.value) == "Unknow argument. trim should be 'both', 'left' or 'right'."

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.ones((2,3,4)))
            assert str(e_info.value) == "Array should be 1 or 2 dimension."

            with pytest.raises(IOError) as e_info:
                tsd.convolve(np.ones(3), ep=[1,2,3,4])
            assert str(e_info.value) == "ep should be an object of type IntervalSet"


    def test_convolve_1d_kernel(self, tsd):
        array = np.random.randn(10)
        if not isinstance(tsd, nap.Ts):
            tsd2 = tsd.convolve(array)
            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            for i in range(tmp.shape[-1]):
                tmp2[:,i] = np.convolve(tmp[:,i], array, mode='full')[4:-5]
            np.testing.assert_array_almost_equal(
                tmp2, 
                tsd2.values.reshape(tsd2.shape[0], -1)
                )

            ep = nap.IntervalSet(start=[0, 60], end=[40,100])
            tsd3 = tsd.convolve(array, ep)
            
            tmp3 = []
            for i in range(len(ep)):
                tmp2 = tsd.restrict(ep[i]).values
                tmp2 = np.array(tmp2.reshape(tmp2.shape[0], -1)) # for pynajax
                for j in range(tmp2.shape[-1]):
                    tmp2[:,j] = np.convolve(tmp2[:,j], array, mode='full')[4:-5]                    
                tmp3.append(tmp2)
                np.testing.assert_array_almost_equal(
                    tmp2,
                    tsd3.restrict(ep[i]).values.reshape(tmp2.shape[0], -1)
                    )

            # Trim
            for trim, sl in zip(['left', 'both', 'right'], [slice(9,None),slice(4,-5),slice(None,-9)]):
                tsd2 = tsd.convolve(array, trim=trim)
                tmp = tsd.values.reshape(tsd.shape[0], -1)
                tmp2 = np.zeros_like(tmp)
                for i in range(tmp.shape[-1]):
                    tmp2[:,i] = np.convolve(tmp[:,i], array, mode='full')[sl]
                np.testing.assert_array_almost_equal(
                    tmp2, 
                    tsd2.values.reshape(tsd2.shape[0], -1)
                    )                        

    def test_convolve_2d_kernel(self, tsd):
        array = np.random.randn(10, 3)
        if not isinstance(tsd, nap.Ts):
            # no epochs
            tsd2 = tsd.convolve(array)
            tmp = tsd.values.reshape(tsd.shape[0], -1)
            
            output = []

            for i in range(tmp.shape[1]):
                for j in range(array.shape[1]):
                    output.append(
                        np.convolve(tmp[:,i], array[:,j], mode='full')[4:-5]
                        )

            output = np.array(output).T            
            np.testing.assert_array_almost_equal(output,tsd2.values.reshape(tsd.shape[0], -1))

            # epochs
            ep = nap.IntervalSet(start=[0, 60], end=[40,100])
            tsd2 = tsd.convolve(array, ep)
                        
            for k in range(len(ep)):
                tmp = tsd.restrict(ep[k])
                tmp2 = tmp.values.reshape(tmp.shape[0], -1)
                output = []
                for i in range(tmp2.shape[1]):
                    for j in range(array.shape[1]):
                        output.append(
                            np.convolve(tmp2[:,i], array[:,j], mode='full')[4:-5]
                            )
                output = np.array(output).T            
                np.testing.assert_array_almost_equal(
                    output,tsd2.restrict(ep[k]).values.reshape(tmp.shape[0], -1)
                    )

    def test_smooth(self, tsd):
        if not isinstance(tsd, nap.Ts):            
            from scipy import signal
            tsd2 = tsd.smooth(1, size_factor=10)

            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            std = int(tsd.rate * 1)
            M = std*11
            window = signal.windows.gaussian(M, std=std)
            window = window / window.sum()            
            for i in range(tmp.shape[-1]):
                tmp2[:,i] = np.convolve(tmp[:,i], window, mode='full')[M//2:1-M//2-1]
            np.testing.assert_array_almost_equal(
                tmp2, 
                tsd2.values.reshape(tsd2.shape[0], -1)
                )

            tsd2 = tsd.smooth(1000, time_units='ms')
            np.testing.assert_array_almost_equal(tmp2, 
                tsd2.values.reshape(tsd2.shape[0], -1))

            tsd2 = tsd.smooth(1000000, time_units='us')
            np.testing.assert_array_almost_equal(tmp2, 
                tsd2.values.reshape(tsd2.shape[0], -1))

            tsd2 = tsd.smooth(1, size_factor=200, norm=False)
            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            std = int(tsd.rate * 1)
            M = std*201
            window = signal.windows.gaussian(M, std=std)            
            for i in range(tmp.shape[-1]):
                tmp2[:,i] = np.convolve(tmp[:,i], window, mode='full')[M//2:1-M//2-1]
            np.testing.assert_array_almost_equal(
                tmp2, 
                tsd2.values.reshape(tsd2.shape[0], -1)
                )

            tsd2 = tsd.smooth(1, windowsize = 10, norm=False)
            tmp = tsd.values.reshape(tsd.shape[0], -1)
            tmp2 = np.zeros_like(tmp)
            std = int(tsd.rate * 1)
            M = int(tsd.rate * 11)
            window = signal.windows.gaussian(M, std=std)            
            for i in range(tmp.shape[-1]):
                tmp2[:,i] = np.convolve(tmp[:,i], window, mode='full')[M//2:1-M//2-1]
            np.testing.assert_array_almost_equal(
                tmp2, 
                tsd2.values.reshape(tsd2.shape[0], -1)
                )

    def test_smooth_raise_error(self, tsd):
        if not isinstance(tsd, nap.Ts):
            with pytest.raises(IOError) as e_info:
                tsd.smooth('a')
            assert str(e_info.value) == "std should be type int or float"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, size_factor='b')
            assert str(e_info.value) == "size_factor should be of type int"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, norm=1)
            assert str(e_info.value) == "norm should be of type boolean"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, time_units = 0)
            assert str(e_info.value) == "time_units should be of type str"

            with pytest.raises(IOError) as e_info:
                tsd.smooth(1, windowsize='a')
            assert str(e_info.value) == "windowsize should be type int or float"

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
        np.testing.assert_array_almost_equal(
            a.time_support, tsd.time_support
            )

    def test_count(self, tsd):
        count = tsd.count(1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))
        
        count = tsd.count(bin_size=1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

        count = tsd.count(bin_size=1, dtype=np.int16)
        assert len(count) == 99
        assert count.dtype == np.dtype(np.int16)


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
        assert len(thrs) == int(np.sum(tsd.values > 0.5))
        thrs = tsd.threshold(0.5, "below")
        assert len(thrs) == int(np.sum(tsd.values < 0.5))
        thrs = tsd.threshold(0.5, "aboveequal")
        assert len(thrs) == int(np.sum(tsd.values >= 0.5))
        thrs = tsd.threshold(0.5, "belowequal")
        assert len(thrs) == int(np.sum(tsd.values <= 0.5))

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
        with pytest.raises(TypeError) as e:
            tsd.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsd.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format(Path("./").resolve())

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsd.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(Path(fake_path).resolve())

        tsd.save("tsd.npz")
        assert "tsd.npz" in [f.name for f in Path('.').iterdir()]

        tsd.save("tsd2")
        assert "tsd2.npz" in [f.name for f in Path('.').iterdir()]

        file = np.load("tsd.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'd' in keys
        assert 'start' in keys
        assert 'end' in keys

        np.testing.assert_array_almost_equal(file['t'], tsd.index)
        np.testing.assert_array_almost_equal(file['d'], tsd.values)
        np.testing.assert_array_almost_equal(file['start'], tsd.time_support.start)
        np.testing.assert_array_almost_equal(file['end'], tsd.time_support.end)

        # Path("tsd.npz").unlink()
        # Path("tsd2.npz").unlink()

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
        
        with pytest.raises(IOError) as e:
            tsd.interpolate([0, 1, 2])
        assert str(e.value) == "First argument should be an instance of Ts, Tsd, TsdFrame or TsdTensor"

        with pytest.raises(IOError) as e:
            tsd.interpolate(ts, left='a')
        assert str(e.value) == "Argument left should be of type float or int"

        with pytest.raises(IOError) as e:
            tsd.interpolate(ts, right='a')
        assert str(e.value) == "Argument right should be of type float or int"

        with pytest.raises(IOError) as e:
            tsd.interpolate(ts, ep=[1,2,3,4])
        assert str(e.value) == "ep should be an object of type IntervalSet"

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
        np.testing.assert_array_almost_equal(tsdframe.time_support, tsdframe[:,0].time_support)
        
        assert isinstance(tsdframe[:,[0,2]], nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.values[:,[0,2]], tsdframe[:,[0,2]].values)
        assert isinstance(tsdframe[:,[0,2]].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(tsdframe.time_support, tsdframe[:,[0,2]].time_support)

    def test_vertical_slicing(self, tsdframe):
        assert isinstance(tsdframe[0:10], nap.TsdFrame)
        np.testing.assert_array_almost_equal(tsdframe.values[0:10], tsdframe[0:10].values)
        assert isinstance(tsdframe[0:10].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(tsdframe[0:10].time_support, tsdframe.time_support)

    def test_str_indexing(self, tsdframe):
        tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), time_units="s", columns=['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(tsdframe.values[:,0], tsdframe['a'].values)
        np.testing.assert_array_almost_equal(tsdframe.values[:,[0,2]], tsdframe[['a', 'c']].values)

        with pytest.raises(Exception):
            tsdframe['d']

        with pytest.raises(Exception):
            tsdframe[['d', 'e']]


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
        with pytest.raises(TypeError) as e:
            tsdframe.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsdframe.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format(Path("./").resolve())

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsdframe.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(Path(fake_path).resolve())

        tsdframe.save("tsdframe.npz")
        assert "tsdframe.npz" in [f.name for f in Path('.').iterdir()]

        tsdframe.save("tsdframe2")
        assert "tsdframe2.npz" in [f.name for f in Path('.').iterdir()]

        file = np.load("tsdframe.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'd' in keys
        assert 'start' in keys
        assert 'end' in keys
        assert 'columns' in keys

        np.testing.assert_array_almost_equal(file['t'], tsdframe.index)
        np.testing.assert_array_almost_equal(file['d'], tsdframe.values)
        np.testing.assert_array_almost_equal(file['start'], tsdframe.time_support.start)
        np.testing.assert_array_almost_equal(file['end'], tsdframe.time_support.end)
        np.testing.assert_array_almost_equal(file['columns'], tsdframe.columns)

        # Path("tsdframe.npz").unlink()
        # Path("tsdframe2.npz").unlink()

    def test_interpolate(self, tsdframe):
        
        y = np.arange(0, 1001)
        data_stack = np.stack([np.arange(0, 1001),]*4).T

        tsdframe = nap.TsdFrame(t=np.arange(0, 101), d=data_stack[0::10, :])

        # Ts
        ts = nap.Ts(t=y/10)
        tsdframe2 = tsdframe.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdframe2.values, data_stack)

        # Tsd
        ts = nap.Tsd(t=y/10, d=np.zeros_like(y))
        tsdframe2 = tsdframe.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdframe2.values, data_stack)

        # TsdFrame
        ts = nap.TsdFrame(t=y/10, d=np.zeros((len(y), 2)))
        tsdframe2 = tsdframe.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdframe2.values, data_stack)
        
        with pytest.raises(IOError) as e:
            tsdframe.interpolate([0, 1, 2])
        assert str(e.value) == "First argument should be an instance of Ts, Tsd, TsdFrame or TsdTensor"

        # Right left
        ep = nap.IntervalSet(start=0, end=5)
        tsdframe = nap.Tsd(t=np.arange(1,4), d=np.arange(3), time_support=ep)
        ts = nap.Ts(t=np.arange(0, 5))
        tsdframe2 = tsdframe.interpolate(ts, left=1234)
        assert float(tsdframe2.values[0]) == 1234.0
        tsdframe2 = tsdframe.interpolate(ts, right=4321)
        assert float(tsdframe2.values[-1]) == 4321.0

    def test_interpolate_with_ep(self, tsdframe):        
        y = np.arange(0, 1001)
        data_stack = np.stack([np.arange(0, 1001),]*4).T

        tsdframe = nap.TsdFrame(t=np.arange(0, 101), d=data_stack[0::10, :])        
        ts = nap.Ts(t=y/10)
        ep = nap.IntervalSet(start=np.arange(0, 100, 20), end=np.arange(10, 110, 20))
        tsdframe2 = tsdframe.interpolate(ts, ep)
        tmp = ts.restrict(ep).index * 10
        print(tmp, tsdframe2.values)
        print(tmp.shape, tsdframe2.values.shape)
        print(tmp.mean(0), tsdframe2.values.mean(0))
        np.testing.assert_array_almost_equal(tmp, tsdframe2.values[:, 0])

        # Empty ep
        ep = nap.IntervalSet(start=200, end=300)
        tsdframe2 = tsdframe.interpolate(ts, ep)
        assert len(tsdframe2) == 0

    def test_convolve_keep_columns(self, tsdframe):
        array = np.random.randn(10)
        tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 3), time_units="s", columns=['a', 'b', 'c'])        
        tsd2 = tsdframe.convolve(array)

        assert isinstance(tsd2, nap.TsdFrame)
        np.testing.assert_array_equal(tsd2.columns, tsdframe.columns)

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
        with pytest.raises(TypeError) as e:
            ts.save(dict)

        with pytest.raises(RuntimeError) as e:
            ts.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format(Path("./").resolve())

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            ts.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(Path(fake_path).resolve())

        ts.save("ts.npz")
        assert "ts.npz" in [f.name for f in Path('.').iterdir()]

        ts.save("ts2")
        assert "ts2.npz" in [f.name for f in Path('.').iterdir()]

        file = np.load("ts.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'start' in keys
        assert 'end' in keys

        np.testing.assert_array_almost_equal(file['t'], ts.index)
        np.testing.assert_array_almost_equal(file['start'], ts.time_support.start)
        np.testing.assert_array_almost_equal(file['end'], ts.time_support.end)

        # Path("ts.npz").unlink()
        # Path("ts2.npz").unlink()

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

        assert len(ts[0:10,0]) == 10

    def test_count(self, ts):
        count = ts.count(1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))
        
        count = ts.count(bin_size=1)
        assert len(count) == 99
        np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))

        count = ts.count(bin_size=1, dtype=np.int16)
        assert len(count) == 99
        assert count.dtype == np.dtype(np.int16)
        

    def test_count_time_units(self, ts):
        for b, tu in zip([1, 1e3, 1e6],['s', 'ms', 'us']):
            count = ts.count(b, time_units = tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))
            
            count = ts.count(b, tu)
            assert len(count) == 99
            np.testing.assert_array_almost_equal(count.index, np.arange(0.5, 99, 1))                

    def test_count_with_ep(self, ts):
        ep = nap.IntervalSet(start=0, end=100)
        count = ts.count(1, ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

        count = ts.count(1, ep=ep)
        assert len(count) == 100
        np.testing.assert_array_almost_equal(count.values, np.ones(100))

    def test_count_with_ep_only(self, ts):
        ep = nap.IntervalSet(start=0, end=100)
        count = ts.count(ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = ts.count(ep=ep)
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

        count = ts.count()
        assert len(count) == 1
        np.testing.assert_array_almost_equal(count.values, np.array([100]))

    def test_count_errors(self, ts):        
        with pytest.raises(ValueError):
            ts.count(bin_size = {})

        with pytest.raises(ValueError):
            ts.count(ep = {})

        with pytest.raises(ValueError):
            ts.count(time_units = {})

    @pytest.mark.parametrize(
        "dtype, expectation",
        [
            (None, does_not_raise()),
            (float, does_not_raise()),
            (int, does_not_raise()),
            (np.int32, does_not_raise()),
            (np.int64, does_not_raise()),
            (np.float32, does_not_raise()),
            (np.float64, does_not_raise()),
            (1, pytest.raises(ValueError, match=f"1 is not a valid numpy dtype")),
        ]
    )
    def test_count_dtype(self, dtype, expectation, ts):
        with expectation:
            count = ts.count(bin_size=0.1, dtype=dtype)
            if dtype:
                assert np.issubdtype(count.dtype, dtype)

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
        np.testing.assert_array_almost_equal(tsdtensor.time_support, tsdtensor[:,0].time_support)
        
        assert isinstance(tsdtensor[:,[0,2]], nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.values[:,[0,2]], tsdtensor[:,[0,2]].values)
        assert isinstance(tsdtensor[:,[0,2]].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(tsdtensor.time_support, tsdtensor[:,[0,2]].time_support)

    def test_vertical_slicing(self, tsdtensor):
        assert isinstance(tsdtensor[0:10], nap.TsdTensor)
        np.testing.assert_array_almost_equal(tsdtensor.values[0:10], tsdtensor[0:10].values)
        assert isinstance(tsdtensor[0:10].time_support, nap.IntervalSet)
        np.testing.assert_array_almost_equal(tsdtensor[0:10].time_support, tsdtensor.time_support)

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
        with pytest.raises(TypeError) as e:
            tsdtensor.save(dict)

        with pytest.raises(RuntimeError) as e:
            tsdtensor.save('./')
        assert str(e.value) == "Invalid filename input. {} is directory.".format(Path("./").resolve())

        fake_path = './fake/path'
        with pytest.raises(RuntimeError) as e:
            tsdtensor.save(fake_path+'/file.npz')
        assert str(e.value) == "Path {} does not exist.".format(Path(fake_path).resolve())

        tsdtensor.save("tsdtensor.npz")
        assert "tsdtensor.npz" in [f.name for f in Path('.').iterdir()]

        tsdtensor.save("tsdtensor2")
        assert "tsdtensor2.npz" in [f.name for f in Path('.').iterdir()]

        file = np.load("tsdtensor.npz")

        keys = list(file.keys())
        assert 't' in keys
        assert 'd' in keys
        assert 'start' in keys
        assert 'end' in keys        

        np.testing.assert_array_almost_equal(file['t'], tsdtensor.index)
        np.testing.assert_array_almost_equal(file['d'], tsdtensor.values)
        np.testing.assert_array_almost_equal(file['start'], tsdtensor.time_support.start)
        np.testing.assert_array_almost_equal(file['end'], tsdtensor.time_support.end)

        # Path("tsdtensor.npz").unlink()
        # Path("tsdtensor2.npz").unlink()

    def test_interpolate(self, tsdtensor):
        
        y = np.arange(0, 1001)
        data_stack = np.stack([np.stack([np.arange(0, 1001),] * 4)] * 3).T

        tsdtensor = nap.TsdTensor(t=np.arange(0, 101), d= data_stack[0::10, ...])

        # Ts
        ts = nap.Ts(t = y / 10)
        tsdtensor2 = tsdtensor.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdtensor2.values, data_stack)

        # Tsd
        ts = nap.Tsd(t=y/10, d=np.zeros_like(y))
        tsdtensor2 = tsdtensor.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdtensor2.values, data_stack)

        # TsdFrame
        ts = nap.TsdFrame(t=y/10, d=np.zeros((len(y), 2)))
        tsdtensor2 = tsdtensor.interpolate(ts)
        np.testing.assert_array_almost_equal(tsdtensor2.values, data_stack)
        
        with pytest.raises(IOError) as e:
            tsdtensor.interpolate([0, 1, 2])
        assert str(e.value) == "First argument should be an instance of Ts, Tsd, TsdFrame or TsdTensor"

        # Right left
        ep = nap.IntervalSet(start=0, end=5)
        tsdtensor = nap.Tsd(t=np.arange(1,4), d=np.arange(3), time_support=ep)
        ts = nap.Ts(t=np.arange(0, 5))
        tsdtensor2 = tsdtensor.interpolate(ts, left=1234)
        assert float(tsdtensor2.values[0]) == 1234.0
        tsdtensor2 = tsdtensor.interpolate(ts, right=4321)
        assert float(tsdtensor2.values[-1]) == 4321.0

    def test_interpolate_with_ep(self, tsdtensor):        
        y = np.arange(0, 1001)
        data_stack = np.stack([np.stack([np.arange(0, 1001),] * 4),] * 3).T

        tsdtensor = nap.TsdTensor(t=np.arange(0, 101), d=data_stack[::10, ...]) 
        
        ts = nap.Ts(t=y / 10)


        ep = nap.IntervalSet(start=np.arange(0, 100, 20), end=np.arange(10, 110, 20))
        tsdframe2 = tsdtensor.interpolate(ts, ep)  
        tmp = ts.restrict(ep).index * 10

        np.testing.assert_array_almost_equal(tmp, tsdframe2.values[:, 0, 0])

        # Empty ep
        ep = nap.IntervalSet(start=200, end=300)
        tsdframe2 = tsdtensor.interpolate(ts, ep)
        assert len(tsdframe2) == 0

@pytest.mark.parametrize("obj",
                         [
                             nap.Tsd(t=np.arange(10), d=np.random.rand(10), time_units="s"),
                             nap.TsdFrame(
                                 t=np.arange(10), d=np.random.rand(10, 3), time_units="s", columns=["a","b","c"]
                             ),
                             nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 3, 2), time_units="s"),
                         ])
def test_pickling(obj):
    """Test that pikling works as expected."""
    # pickle and unpickle ts_group
    pickled_obj = pickle.dumps(obj)
    unpickled_obj = pickle.loads(pickled_obj)

    # Ensure time is the same
    assert np.all(obj.t == unpickled_obj.t)

    # Ensure data is the same
    assert np.all(obj.d == unpickled_obj.d)

    # Ensure time support is the same
    assert np.all(obj.time_support == unpickled_obj.time_support)


####################################################
# Test for slicing
####################################################


@pytest.mark.parametrize(
    "start, end, mode, n_points, expectation",
    [
        (1, 3, "closest_t", None, does_not_raise()),
        (None, 3, "closest_t", None, pytest.raises(ValueError, match="'start' must be an int or a float")),
        (2, "a", "closest_t", None, pytest.raises(ValueError, match="'end' must be an int or a float. Type <class 'str'> provided instead!")),
        (1, 3, "closest_t", "a", pytest.raises(TypeError, match="'n_points' must be of type int or None. Type <class 'str'> provided instead!")),        
        (1, None, "closest_t", 1, pytest.raises(ValueError, match="'n_points' can be used only when 'end' is specified!")),        
        (1, 3, "banana", None, pytest.raises(ValueError, match="'mode' only accepts 'before_t', 'after_t', 'closest_t' or 'restrict'.")),                
        (3, 1, "closest_t", None, pytest.raises(ValueError, match="'start' should not precede 'end'")),
        (1, 3, "restrict", 1, pytest.raises(ValueError, match="Fixing the number of time points is incompatible with 'restrict' mode.")),
        (1., 3., "closest_t", None, does_not_raise()),
        (1., None, "closest_t", None, does_not_raise()),
    ]
)
def test_get_slice_raise_errors(start, end, mode, n_points, expectation):
    ts = nap.Ts(t=np.array([1, 2, 3, 4]))
    with expectation:
        ts._get_slice(start, end, mode, n_points)


@pytest.mark.parametrize(
    "start, end, mode, expected_slice, expected_array",
    [
        (1, 3, "after_t", slice(0, 2), np.array([1, 2])),
        (1, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (1, 3, "closest_t", slice(0, 2), np.array([1, 2])),
        (1, 2.7, "after_t", slice(0, 2), np.array([1, 2])),
        (1, 2.7, "before_t", slice(0, 1), np.array([1])),
        (1, 2.7, "closest_t", slice(0, 2), np.array([1, 2])),
        (1, 2.4, "after_t", slice(0, 2), np.array([1, 2])),
        (1, 2.4, "before_t", slice(0, 1), np.array([1])),
        (1, 2.4, "closest_t", slice(0, 1), np.array([1])),
        (1.1, 3, "after_t", slice(1, 2), np.array([2])),
        (1.1, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (1.1, 3, "closest_t", slice(0, 2), np.array([1, 2])),
        (1.6, 3, "after_t", slice(1, 2), np.array([2])),
        (1.6, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (1.6, 3, "closest_t", slice(1, 2), np.array([2])),
        (1.6, 1.8, "before_t", slice(0, 0), np.array([])),
        (1.6, 1.8, "after_t", slice(1, 1), np.array([])),
        (1.6, 1.8, "closest_t", slice(1, 1), np.array([])),
        (1.4, 1.6, "closest_t", slice(0, 1), np.array([1])),
        (3, 3, "after_t", slice(2, 2), np.array([])),
        (3, 3, "before_t", slice(2, 2), np.array([])),
        (3, 3, "closest_t", slice(2, 2), np.array([])),
        (0, 3, "after_t", slice(0, 2), np.array([1, 2])),
        (0, 3, "before_t", slice(0, 2), np.array([1, 2])),
        (0, 3, "closest_t", slice(0, 2), np.array([1, 2])),
        (0, 4, "after_t", slice(0, 3), np.array([1, 2, 3])),
        (0, 4, "before_t", slice(0, 3), np.array([1, 2, 3])),
        (0, 4, "closest_t", slice(0, 3), np.array([1, 2, 3])),
        (4, 4, "after_t", slice(3, 3), np.array([])),
        (4, 4, "before_t", slice(3, 3), np.array([])),
        (4, 4, "closest_t", slice(3, 3), np.array([])),
        (4, 5, "after_t", slice(3, 4), np.array([4])),
        (4, 5, "before_t", slice(3, 3), np.array([])),
        (4, 5, "closest_t", slice(3, 3), np.array([])),
        (0, 1, "after_t", slice(0, 0), np.array([])),
        (0, 1, "before_t", slice(0, 1), np.array([1])),
        (0, 1, "closest_t", slice(0, 0), np.array([])),
        (0, None, "after_t", slice(0, 1), np.array([1])),
        (0, None, "before_t", slice(0, 0), np.array([])),
        (0, None, "closest_t", slice(0, 1), np.array([1])),
        (1, None, "after_t", slice(0, 1), np.array([1])),
        (1, None, "before_t", slice(0, 1), np.array([1])),
        (1, None, "closest_t", slice(0, 1), np.array([1])),
        (5, None, "after_t", slice(3, 3), np.array([])),
        (5, None, "before_t", slice(3, 4), np.array([4])),
        (5, None, "closest_t", slice(3, 4), np.array([4])),
        (1, 3, "restrict", slice(0, 3), np.array([1, 2, 3])),
        (1, 2.7, "restrict", slice(0, 2), np.array([1, 2])),
        (1, 2.4, "restrict", slice(0, 2), np.array([1, 2])),
        (1.1, 3, "restrict", slice(1, 3), np.array([2, 3])),
        (1.6, 3, "restrict", slice(1, 3), np.array([2, 3])),
        (1.6, 1.8, "restrict", slice(1, 1), np.array([])),
        (1.4, 1.6, "restrict", slice(1, 1), np.array([])),
        (3, 3, "restrict", slice(2, 3), np.array([3])),
        (0, 3, "restrict", slice(0, 3), np.array([1, 2, 3])),
        (0, 4, "restrict", slice(0, 4), np.array([1, 2, 3, 4])),
        (4, 4, "restrict", slice(3, 4), np.array([4])),
        (4, 5, "restrict", slice(3, 4), np.array([4])),
        (0, 1, "restrict", slice(0, 1), np.array([1])),

    ]
)
@pytest.mark.parametrize("ts",
                         [
                             nap.Ts(t=np.array([1, 2, 3, 4])),
                             nap.Tsd(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])),
                             nap.TsdFrame(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None]),
                             nap.TsdTensor(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None, None])
                         ])
def test_get_slice_value(start, end, mode, expected_slice, expected_array, ts):
    out_slice = ts._get_slice(start, end=end, mode=mode)
    out_array = ts.t[out_slice]
    assert out_slice == expected_slice
    assert np.all(out_array == expected_array)
    if mode == "restrict":
        iset = nap.IntervalSet(start, end)
        out_restrict = ts.restrict(iset)
        assert np.all(out_restrict.t == out_array)


def test_get_slice_vs_get_random_val_start_end_value():
    np.random.seed(123)
    ts = nap.Ts(np.linspace(0.2, 0.8, 100))
    se_vec = np.random.uniform(0, 1, size=(10000, 2))
    starts = np.min(se_vec, axis=1)
    ends = np.max(se_vec, axis=1)
    for start, end in zip(starts, ends):
        out_slice = ts.get_slice(start=start, end=end)
        out_ts = ts[out_slice]
        out_get = ts.get(start, end)
        assert np.all(out_get.t == out_ts.t)


def test_get_slice_vs_get_random_val_start_value():
    np.random.seed(123)
    ts = nap.Ts(np.linspace(0.2, 0.8, 100))
    starts = np.random.uniform(0, 1, size=(10000, ))

    for start in starts:
        out_slice = ts.get_slice(start=start, end=None)
        out_ts = ts[out_slice]
        out_get = ts.get(start)
        assert np.all(out_get.t == out_ts.t)



@pytest.mark.parametrize(
    "end, n_points, expectation",
    [
        (1, 3, does_not_raise()),
        (None, 3, pytest.raises(ValueError, match="'n_points' can be used only when")),

    ]
)
@pytest.mark.parametrize("time_unit", ["s", "ms",  "us"])
@pytest.mark.parametrize("mode", ["closest_t", "before_t",  "after_t"])
def test_get_slice_n_points(end, n_points, expectation, time_unit, mode):
    ts = nap.Ts(t=np.array([1, 2, 3, 4]))
    with expectation:
        ts._get_slice(1, end, n_points=n_points, mode=mode)



@pytest.mark.parametrize(
    "start, end, n_points, mode, expected_slice, expected_array",
    [
        # smaller than n_points
        (1, 2, 2, "after_t", slice(0, 1), np.array([1])),
        (1, 2, 2, "before_t", slice(0, 1), np.array([1])),
        (1, 2, 2, "closest_t", slice(0, 1), np.array([1])),
        # larger than n_points
        (1, 5, 2, "after_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding down
        (1, 5.2, 2, "after_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.2, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.2, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding down
        (1, 6.2, 2, "after_t", slice(0, 6, 3), np.array([1, 4])),
        (1, 6.2, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 6.2, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding up
        (1, 5.6, 2, "after_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.6, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 5.6, 2, "closest_t", slice(0, 4, 2), np.array([1, 3])),
        # larger than n_points with rounding up
        (1, 6.6, 2, "after_t", slice(0, 6, 3), np.array([1, 4])),
        (1, 6.6, 2, "before_t", slice(0, 4, 2), np.array([1, 3])),
        (1, 6.6, 2, "closest_t", slice(0, 6, 3), np.array([1, 4])),
    ]
)
@pytest.mark.parametrize("ts",
                         [
                             nap.Ts(t=np.arange(1, 10)),
                             nap.Tsd(t=np.arange(1, 10), d=np.arange(1, 10)),
                             nap.TsdFrame(t=np.arange(1, 10), d=np.arange(1, 10)[:, None]),
                             nap.TsdTensor(t=np.arange(1, 10), d=np.arange(1, 10)[:, None, None])
                         ])
def test_get_slice_value_step(start, end, n_points, mode, expected_slice, expected_array, ts):
    out_slice = ts._get_slice(start, end=end, mode=mode, n_points=n_points)
    out_array = ts.t[out_slice]
    assert out_slice == expected_slice
    assert np.all(out_array == expected_array)

@pytest.mark.parametrize(
    "start, end, expected_slice, expected_array",
    [
        (1, 3, slice(0, 3), np.array([1, 2, 3])),
        (1, 2.7,  slice(0, 2), np.array([1, 2])),
        (1, 2.4,  slice(0, 2), np.array([1, 2])),
        (1.1, 3,  slice(1, 3), np.array([2, 3])),
        (1.6, 3,  slice(1, 3), np.array([2, 3])),
        (1.6, 1.8,  slice(1, 1), np.array([])),
        (1.4, 1.6,  slice(1, 1), np.array([])),
        (3, 3,  slice(2, 3), np.array([3])),
        (0, 3,  slice(0, 3), np.array([1, 2, 3])),
        (0, 4,  slice(0, 4), np.array([1, 2, 3, 4])),
        (4, 4,  slice(3, 4), np.array([4])),
        (4, 5,  slice(3, 4), np.array([4])),
        (0, 1,  slice(0, 1), np.array([1])),
        (0, None, slice(0, 1), np.array([1])),
        (1, None, slice(0, 1), np.array([1])),
        (4, None, slice(3, 4), np.array([4])),
        (5, None, slice(3, 4), np.array([4])),
        (-1, 0, slice(0, 0), np.array([])),
        (5, 6, slice(4, 4), np.array([])),
    ]
)
@pytest.mark.parametrize("ts",
                         [
                             nap.Ts(t=np.array([1, 2, 3, 4])),
                             nap.Tsd(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])),
                             nap.TsdFrame(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None]),
                             nap.TsdTensor(t=np.array([1, 2, 3, 4]), d=np.array([1, 2, 3, 4])[:, None, None])
                         ])
def test_get_slice_public(start, end, expected_slice, expected_array, ts):
    out_slice = ts.get_slice(start, end=end)
    out_array = ts.t[out_slice]
    assert out_slice == expected_slice
    assert np.all(out_array == expected_array)
