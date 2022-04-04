# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:16:30
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-04 16:09:22

"""Tests of tuning curves for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest

def test_compute_1d_tuning_curves():
    tsgroup = nap.TsGroup({0:nap.Ts(t=np.arange(0,100))})
    feature = nap.Tsd(t=np.arange(0, 100, 0.1),d=np.arange(0, 100, 0.1)%1.0)
    tc = nap.compute_1d_tuning_curves(tsgroup, feature, nb_bins=10)

    assert len(tc) == 10
    assert list(tc.columns) == list(tsgroup.keys())
    np.testing.assert_array_almost_equal(
        tc[0].values[1:], np.zeros(9))
    assert int(tc[0].values[0]) == 10

def test_compute_1d_tuning_curves_with_ep():
    tsgroup = nap.TsGroup({0:nap.Ts(t=np.arange(0,100))})
    feature = nap.Tsd(t=np.arange(0, 100, 0.1),d=np.arange(0, 100, 0.1)%1.0)
    tc1 = nap.compute_1d_tuning_curves(tsgroup, feature, nb_bins=10)
    ep = nap.IntervalSet(start=0, end=50)
    tc2 = nap.compute_1d_tuning_curves(tsgroup, feature, nb_bins=10, ep = ep)
    pd.testing.assert_frame_equal(tc1, tc2)

def test_compute_1d_tuning_curves_with_min_max():
    tsgroup = nap.TsGroup({0:nap.Ts(t=np.arange(0,100))})
    feature = nap.Tsd(t=np.arange(0, 100, 0.1),d=np.arange(0, 100, 0.1)%1.0)
    tc = nap.compute_1d_tuning_curves(tsgroup, feature, nb_bins=10, minmax=(0, 1))
    assert len(tc) == 10
    np.testing.assert_array_almost_equal(
        tc[0].values[1:], np.zeros(9))
    assert tc[0].values[0] >= 9

def test_compute_2d_tuning_curves():
    tsgroup = nap.TsGroup(
        {0:nap.Ts(t=np.arange(0,100, 10)),
        1:nap.Ts(t=np.array([50, 149]))}
        )
    tmp = np.vstack((np.repeat(np.arange(0,100),10),np.tile(np.arange(0,100),10))).T
    features = nap.TsdFrame(
        t=np.arange(0, 200, 0.1),
        d=np.vstack((tmp,tmp[::-1])))
    tc, xy = nap.compute_2d_tuning_curves(tsgroup, features, 10)

    assert isinstance(tc, dict)
    assert list(tc.keys()) == list(tsgroup.keys())
    for i in tc.keys():
        assert tc[i].shape == (10,10)
    np.testing.assert_array_almost_equal(
        tc[0][:,1:], np.zeros((10,9)))    
    assert tc[1][5,0] >= 1.0
    assert isinstance(xy, list)
    assert len(xy) == 2
    for i in range(2):
        assert np.min(xy)>0
        assert np.max(xy)<100

def test_compute_2d_tuning_curves_with_ep():
    tsgroup = nap.TsGroup(
        {0:nap.Ts(t=np.arange(0,100, 10)),
        1:nap.Ts(t=np.array([50, 149]))}
        )
    tmp = np.vstack((np.repeat(np.arange(0,100),10),np.tile(np.arange(0,100),10))).T
    features = nap.TsdFrame(
        t=np.arange(0, 300, 0.1),
        d=np.vstack((tmp,tmp[::-1],tmp)))
    ep = nap.IntervalSet(start=0, end=200)
    tc, xy = nap.compute_2d_tuning_curves(tsgroup, features, 10,ep=ep)
    for i in tc.keys():
        assert tc[i].shape == (10,10)
    np.testing.assert_array_almost_equal(
        tc[0][:,1:], np.zeros((10,9)))    
    assert tc[1][5,0] >= 1.0

@pytest.mark.filterwarnings("ignore")
def test_compute_2d_tuning_curves_with_minmax():
    tsgroup = nap.TsGroup(
        {0:nap.Ts(t=np.arange(0,100, 10)),
        1:nap.Ts(t=np.array([50, 149]))}
        )
    tmp = np.vstack((np.repeat(np.arange(0,100),10),np.tile(np.arange(0,100),10))).T
    features = nap.TsdFrame(
        t=np.arange(0, 200, 0.1),
        d=np.vstack((tmp,tmp[::-1])))
    minmax=(20,40,0,10)
    tc, xy = nap.compute_2d_tuning_curves(tsgroup, features, 10, minmax=minmax)

    assert len(xy) == 2
    xbins = np.linspace(minmax[0],minmax[1],11)
    np.testing.assert_array_almost_equal(
        xy[0],xbins[0:-1]+np.diff(xbins)/2)
    ybins = np.linspace(minmax[2],minmax[3],11)
    np.testing.assert_array_almost_equal(
        xy[1],ybins[0:-1]+np.diff(ybins)/2)

def test_compute_1d_mutual_info():
    tc = pd.DataFrame(index=np.arange(0, 2),data=np.array([0,10]))
    feature = nap.Tsd(t=np.arange(100), d=np.tile(np.arange(2), 50))
    si = nap.compute_1d_mutual_info(tc, feature)
    assert isinstance(si, pd.DataFrame)
    assert list(si.columns) == ['SI']
    assert list(si.index.values) == list(tc.columns)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 1.0)
    si = nap.compute_1d_mutual_info(tc, feature, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 5.0)

    ep = nap.IntervalSet(start=0, end = 49)
    si = nap.compute_1d_mutual_info(tc, feature, ep=ep)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 1.0)
    si = nap.compute_1d_mutual_info(tc, feature, ep=ep, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 5.0)

    minmax = (0, 1)
    si = nap.compute_1d_mutual_info(tc, feature, minmax=minmax)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 1.0)

def test_compute_2d_mutual_info():
    tc = {0:np.array([[0,1],[0,0]])}
    features = nap.TsdFrame(
        t=np.arange(100),
        d=np.tile(np.array([[0,0,1,1],[0,1,0,1]]),25).T)
    si = nap.compute_2d_mutual_info(tc, features)
    assert isinstance(si, pd.DataFrame)
    assert list(si.columns) == ['SI']
    assert list(si.index.values) == list(tc.keys())
    np.testing.assert_approx_equal(si.loc[0,'SI'], 2.0)
    si = nap.compute_2d_mutual_info(tc, features, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 0.5)


    ep = nap.IntervalSet(start=0, end = 7)
    si = nap.compute_2d_mutual_info(tc, features, ep=ep)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 2.0)
    si = nap.compute_2d_mutual_info(tc, features, ep=ep, bitssec=True)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 0.5)

    minmax = (0, 1, 0, 1)
    si = nap.compute_2d_mutual_info(tc, features, minmax=minmax)
    np.testing.assert_approx_equal(si.loc[0,'SI'], 2.0)

def test_compute_1d_tuning_curves_continuous():
    tsdframe = nap.TsdFrame(t = np.arange(0, 100),d=np.ones((100,1)))
    feature = nap.Tsd(t=np.arange(0, 100, 0.1),d=np.arange(0, 100, 0.1)%1.0)
    tc = nap.compute_1d_tuning_curves_continous(tsdframe, feature, nb_bins=10)

    assert len(tc) == 10
    assert list(tc.columns) == list(tsdframe.columns)
    np.testing.assert_array_almost_equal(
        tc[0].values[1:], np.zeros(9))
    assert int(tc[0].values[0]) == 1.0

def test_compute_1d_tuning_curves_continuous_with_ep():
    tsdframe = nap.TsdFrame(t = np.arange(0, 100),d=np.ones((100,1)))
    feature = nap.Tsd(t=np.arange(0, 100, 0.1),d=np.arange(0, 100, 0.1)%1.0)
    ep = nap.IntervalSet(start=0, end=50)
    tc1 = nap.compute_1d_tuning_curves_continous(tsdframe, feature, nb_bins=10)
    tc2 = nap.compute_1d_tuning_curves_continous(tsdframe, feature, nb_bins=10, ep=ep)
    pd.testing.assert_frame_equal(tc1, tc2)

def test_compute_1d_tuning_curves_continuous_with_min_max():
    tsdframe = nap.TsdFrame(t = np.arange(0, 100),d=np.ones((100,1)))
    feature = nap.Tsd(t=np.arange(0, 100, 0.1),d=np.arange(0, 100, 0.1)%1.0)
    tc = nap.compute_1d_tuning_curves_continous(tsdframe, feature, nb_bins=10, minmax=(0,1))
    assert len(tc) == 10
    np.testing.assert_array_almost_equal(
        tc[0].values[1:], np.zeros(9))
    assert tc[0].values[0] == 1.0

def test_compute_2d_tuning_curves_continuous():
    tsdframe = nap.TsdFrame(t = np.arange(0, 100),
        d=np.hstack((np.ones((100,1)), np.ones((100,1))*2)))
    features = nap.TsdFrame(
        t=np.arange(100),
        d=np.tile(np.array([[0,0,1,1],[0,1,0,1]]),25).T)
    tc, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, 2)

    assert isinstance(tc, dict)
    assert list(tc.keys()) == list(tsdframe.columns)
    for i in tc.keys():
        assert tc[i].shape == (2,2)
    tmp = np.zeros((2,2,2))
    tmp[:,0,0] = [1,2]
    for i in range(2):
        np.testing.assert_array_almost_equal(
            tc[i], tmp[i]) 
    assert isinstance(xy, list)
    assert len(xy) == 2
    for i in range(2):
        assert np.min(xy)>0
        assert np.max(xy)<1

def test_compute_2d_tuning_curves_continuous_with_ep():
    tsdframe = nap.TsdFrame(t = np.arange(0, 100),
        d=np.hstack((np.ones((100,1)), np.ones((100,1))*2)))
    features = nap.TsdFrame(
        t=np.arange(100),
        d=np.tile(np.array([[0,0,1,1],[0,1,0,1]]),25).T)
    ep = nap.IntervalSet(start=0, end=7)
    tc1, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, 2, ep=ep)
    tc2, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, 2)

    for i in tc1.keys():
        np.testing.assert_array_almost_equal(tc1[i], tc2[i])

@pytest.mark.filterwarnings("ignore")
def test_compute_2d_tuning_curves_with_minmax():
    tsdframe = nap.TsdFrame(t = np.arange(0, 100),
        d=np.hstack((np.ones((100,1)), np.ones((100,1))*2)))
    features = nap.TsdFrame(
        t=np.arange(100),
        d=np.tile(np.array([[0,0,1,1],[0,1,0,1]]),25).T)
    minmax=(0, 10, 0, 2)
    tc, xy = nap.compute_2d_tuning_curves_continuous(tsdframe, features, 2, minmax=minmax)

    assert len(xy) == 2
    xbins = np.linspace(minmax[0],minmax[1],3)
    np.testing.assert_array_almost_equal(
        xy[0],xbins[0:-1]+np.diff(xbins)/2)
    ybins = np.linspace(minmax[2],minmax[3],3)
    np.testing.assert_array_almost_equal(
        xy[1],ybins[0:-1]+np.diff(ybins)/2)

