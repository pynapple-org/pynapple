# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:16:53
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-03 21:32:43
#!/usr/bin/env python

"""Tests of perievent for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest

def test_align_to_event():
    tref = np.arange(10, 100, 10)
    times = np.arange(0, 110)
    data = np.ones(110)
    x, y = nap.align_to_event(times, data, tref, (10, 10))
    assert len(x) == len(tref)
    assert len(y) == len(tref)
    for i in range(len(tref)):
        np.testing.assert_array_almost_equal(x[i], np.arange(-9, 10))
        np.testing.assert_array_almost_equal(y[i], np.ones(len(y[i])))

def test_compute_perievent_with_tsd():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    peth = nap.compute_perievent(tsd, tref, minmax=(-10,10))

    assert isinstance(peth, nap.TsGroup)
    assert len(peth) == len(tref)   
    np.testing.assert_array_almost_equal(
        peth.get_info('ref_times').values,
        tref.index.values)
    for i,j in zip(peth.keys(),np.arange(1,100,10)):
        np.testing.assert_array_almost_equal(
            peth[i].index.values, np.arange(-9, 10))
        np.testing.assert_array_almost_equal(
            peth[i].values, np.arange(j,j+19))

def test_compute_perievent_raise_error():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = np.arange(10, 100, 10)
    with pytest.raises(Exception) as e_info:
        nap.compute_perievent(tsd, tref, minmax=(-10,10))    
    assert str(e_info.value) == "tref should be a Tsd object."
    tsd = t=np.arange(100)
    tref = nap.Ts(t=np.arange(10, 100, 10))
    with pytest.raises(Exception) as e_info:
        nap.compute_perievent(tsd, tref, minmax=(-10,10))    
    assert str(e_info.value) == "Unknown format for data"

def test_compute_perievent_with_tsgroup():
    tsgroup = nap.TsGroup({
        0:nap.Ts(t=np.arange(0,100)),
        1:nap.Ts(t=np.arange(0,200))
    })    
    tref = nap.Ts(t=np.arange(10, 100, 10))
    peth = nap.compute_perievent(tsgroup, tref, minmax=(-10,10))

    assert isinstance(peth, dict)
    assert list(tsgroup.keys()) == list(peth.keys())
    for i in peth.keys():
        assert len(peth[i]) == len(tref)
        np.testing.assert_array_almost_equal(
            peth[i].get_info('ref_times').values,
            tref.index.values)
        for j,k in zip(peth[i].keys(),np.arange(1,100,10)):
            np.testing.assert_array_almost_equal(
                peth[i][j].index.values, np.arange(-9, 10))

def test_compute_perievent_time_units():
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    tref = nap.Ts(t=np.arange(10, 100, 10))
    for tu, fa in zip(['s', 'ms', 'us'], [1, 1e3, 1e6]):
        peth = nap.compute_perievent(tsd, tref, minmax=(-10*fa,10*fa), time_unit=tu)
        for i,j in zip(peth.keys(),np.arange(1,100,10)):
            np.testing.assert_array_almost_equal(
                peth[i].index.values, np.arange(-9, 10))
            np.testing.assert_array_almost_equal(
                peth[i].values, np.arange(j,j+19))
