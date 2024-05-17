# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:16:53
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-05-07 15:22:24
#!/usr/bin/env python

"""Tests of perievent for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest


def test_align_tsd():
    tsd = nap.Ts(t=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    peth = nap.compute_perievent(tsd, tref, minmax=(-10, 10))

    assert len(peth) == len(tref)
    assert isinstance(peth, nap.TsGroup)
    for i, j in zip(peth.keys(), np.arange(0, 100, 10)):
        np.testing.assert_array_almost_equal(peth[i].index, np.arange(-10, 10))

def test_compute_perievent_with_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    peth = nap.compute_perievent(tsd, tref, minmax=(-10, 10))

    assert isinstance(peth, nap.TsGroup)
    assert len(peth) == len(tref)
    np.testing.assert_array_almost_equal(
        peth.get_info("ref_times").values, tref.index
    )
    for i, j in zip(peth.keys(), np.arange(0, 100, 10)):
        np.testing.assert_array_almost_equal(peth[i].index, np.arange(-10, 10))
        np.testing.assert_array_almost_equal(peth[i].values, np.arange(j, j + 20))

def test_compute_perievent_minmax():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    peth = nap.compute_perievent(tsd, tref, minmax=10.0)

    assert isinstance(peth, nap.TsGroup)
    assert len(peth) == len(tref)
    np.testing.assert_array_almost_equal(
        peth.get_info("ref_times").values, tref.index
    )
    for i, j in zip(peth.keys(), np.arange(0, 100, 10)):
        np.testing.assert_array_almost_equal(peth[i].index, np.arange(-10, 10))
        np.testing.assert_array_almost_equal(peth[i].values, np.arange(j, j + 20))

def test_compute_perievent_raise_error():
    tsd = nap.Ts(t=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent(tsd, [0,1,2], minmax=(-10, 10))
    assert str(e_info.value) == "tref should be a Ts or Tsd object."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent([0,1,2], tref, minmax=(-10, 10))
    assert str(e_info.value) == "data should be a Ts, Tsd or TsGroup."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent(tsd, tref, minmax={0:1})
    assert str(e_info.value) == "minmax should be a tuple or int or float."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent(tsd, tref, minmax=10, time_unit=1)
    assert str(e_info.value) == "time_unit should be a str."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent(tsd, tref, minmax=10, time_unit='a')
    assert str(e_info.value) == "time_unit should be 's', 'ms' or 'us'"

def test_compute_perievent_with_tsgroup():
    tsgroup = nap.TsGroup(
        {0: nap.Ts(t=np.arange(0, 100)), 1: nap.Ts(t=np.arange(0, 200))}
    )
    tref = nap.Ts(t=np.arange(10, 100, 10))
    peth = nap.compute_perievent(tsgroup, tref, minmax=(-10, 10))

    assert isinstance(peth, dict)
    assert list(tsgroup.keys()) == list(peth.keys())
    for i in peth.keys():
        assert len(peth[i]) == len(tref)
        np.testing.assert_array_almost_equal(
            peth[i].get_info("ref_times").values, tref.index
        )
        for j, k in zip(peth[i].keys(), np.arange(0, 100, 10)):
            np.testing.assert_array_almost_equal(
                peth[i][j].index, np.arange(-10, 10)
            )

def test_compute_perievent_time_units():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
        peth = nap.compute_perievent(
            tsd, tref, minmax=(-10 * fa, 10 * fa), time_unit=tu
        )
        for i, j in zip(peth.keys(), np.arange(0, 100, 10)):
            np.testing.assert_array_almost_equal(peth[i].index, np.arange(-10, 10))
            np.testing.assert_array_almost_equal(peth[i].values, np.arange(j, j + 20))


def test_compute_perievent_continuous():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.array([20, 60]))
    minmax=(-5, 10)
    
    # time_array = tsd.t
    # data_array = tsd.d
    # time_target_array = tref.t
    # starts = tsd.time_support.start
    # ends = tsd.time_support.end
    # window = np.abs(minmax)
    # binsize = time_array[1] - time_array[0]
    # idx1 = -np.arange(0, window[0] + binsize, binsize)[::-1][:-1]
    # idx2 = np.arange(0, window[1] + binsize, binsize)[1:]
    # time_idx = np.hstack((idx1, np.zeros(1), idx2))
    # windowsize = np.array([idx1.shape[0], idx2.shape[0]])    

    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax)    

    assert isinstance(pe, nap.TsdFrame)
    assert pe.shape[1] == len(tref)
    np.testing.assert_array_almost_equal(pe.index.values, np.arange(minmax[0], minmax[-1]+1))
    tmp = np.array([np.arange(t+minmax[0], t+minmax[1]+1) for t in tref.t]).T
    np.testing.assert_array_almost_equal(pe.values, tmp)

    minmax=(5, 10)
    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax)    
    np.testing.assert_array_almost_equal(pe.values, tmp)

    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.array([20, 60]))
    minmax=5
    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax)    
    assert isinstance(pe, nap.TsdFrame)
    assert pe.shape[1] == len(tref)
    np.testing.assert_array_almost_equal(pe.index.values, np.arange(-minmax, minmax+1))
    tmp = np.array([np.arange(t-minmax, t+minmax+1) for t in tref.t]).T
    np.testing.assert_array_almost_equal(pe.values, tmp)

    tsd = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 3))
    tref = nap.Ts(t=np.array([20, 60]))
    minmax=(-5, 10)
    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax)    
    assert isinstance(pe, nap.TsdTensor)
    assert pe.d.ndim == 3
    assert pe.shape[1:] == (len(tref), tsd.shape[1])
    np.testing.assert_array_almost_equal(pe.index.values, np.arange(minmax[0], minmax[-1]+1))
    tmp = np.zeros(pe.shape)
    for i,t in enumerate(tref.t):
        idx = np.where(tsd.t == t)[0][0]
        tmp[:,i,:] = tsd.values[idx+minmax[0]:idx+minmax[1]+1]    
    np.testing.assert_array_almost_equal(pe.values, tmp)

    tsd = nap.TsdTensor(t=np.arange(100), d=np.random.randn(100, 3, 4))
    tref = nap.Ts(t=np.array([20, 60]))
    minmax=(-5, 10)
    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax)    
    assert isinstance(pe, nap.TsdTensor)
    assert pe.d.ndim == 4
    assert pe.shape[1:] == (len(tref), *tsd.shape[1:])
    np.testing.assert_array_almost_equal(pe.index.values, np.arange(minmax[0], minmax[-1]+1))
    tmp = np.zeros(pe.shape)
    for i,t in enumerate(tref.t):
        idx = np.where(tsd.t == t)[0][0]
        tmp[:,i,:] = tsd.values[idx+minmax[0]:idx+minmax[1]+1]    
    np.testing.assert_array_almost_equal(pe.values, tmp)


def test_compute_perievent_continuous_time_units():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.array([20, 60]))    
    minmax = (-5, 10)
    for tu, fa in zip(["s", "ms", "us"], [1, 1e3, 1e6]):
        pe = nap.compute_perievent_continuous(tsd, tref, minmax=(minmax[0] * fa, minmax[1] * fa), time_unit=tu)
        np.testing.assert_array_almost_equal(pe.index.values, np.arange(minmax[0], minmax[1]+1))
        tmp = np.array([np.arange(t+minmax[0], t+minmax[1]+1) for t in tref.t]).T
        np.testing.assert_array_almost_equal(pe.values, tmp)


def test_compute_perievent_continuous_with_ep():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.array([10, 50, 80]))
    minmax=(-5, 10)
    ep = nap.IntervalSet(start = [0, 60], end = [40, 99])
    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax, ep=ep)
    
    assert pe.shape[1] == len(tref)-1
    tmp = np.array([np.arange(t+minmax[0], t+minmax[1]+1) for t in tref.restrict(ep).t]).T
    np.testing.assert_array_almost_equal(pe.values, tmp)
    
    tref = ep.starts

    # time_array = tsd.t
    # data_array = tsd.d
    # time_target_array = tref.t
    # starts = ep.start
    # ends = ep.end
    # window = np.abs(minmax)
    # binsize = time_array[1] - time_array[0]
    # idx1 = -np.arange(0, window[0] + binsize, binsize)[::-1][:-1]
    # idx2 = np.arange(0, window[1] + binsize, binsize)[1:]
    # time_idx = np.hstack((idx1, np.zeros(1), idx2))
    # windowsize = np.array([idx1.shape[0], idx2.shape[0]])    


    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax, ep=ep)
    tmp = np.array([np.arange(t, t+minmax[1]+1) for t in tref.restrict(ep).t]).T
    np.testing.assert_array_almost_equal(pe.values[abs(minmax[0]):], tmp)

    tref = ep.ends
    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax, ep=ep)
    tmp = np.array([np.arange(t+minmax[0], t+1) for t in tref.restrict(ep).t]).T
    np.testing.assert_array_almost_equal(pe.values[:-abs(minmax[1])], tmp)

    ep = nap.IntervalSet(start = [100], end = [200])
    tref = nap.Ts(t=np.array([120, 150, 180]))
    pe = nap.compute_perievent_continuous(tsd, tref, minmax=minmax, ep=ep)    
    assert np.all(np.isnan(pe.values))


def test_compute_perievent_continuous_raise_error():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent_continuous(tsd, [0,1,2], minmax=(-10, 10))
    assert str(e_info.value) == "tref should be a Ts or Tsd object."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent_continuous([0,1,2], tref, minmax=(-10, 10))
    assert str(e_info.value) == "data should be a Tsd, TsdFrame or TsdTensor."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent_continuous(tsd, tref, minmax={0:1})
    assert str(e_info.value) == "minmax should be a tuple or int or float."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent_continuous(tsd, tref, minmax=10, time_unit=1)
    assert str(e_info.value) == "time_unit should be a str."

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent_continuous(tsd, tref, minmax=10, time_unit='a')
    assert str(e_info.value) == "time_unit should be 's', 'ms' or 'us'"

    with pytest.raises(AssertionError) as e_info:
        nap.compute_perievent_continuous(tsd, tref, minmax=10, ep='a')
    assert str(e_info.value) == "ep should be an IntervalSet object."
