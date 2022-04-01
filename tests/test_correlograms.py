# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:16:22
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-01 17:16:42
#!/usr/bin/env python

"""Tests of correlograms for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
from itertools import combinations

def test_cross_correlogram():
	t1 = np.array([0])
	t2 = np.array([1])
	cc, bincenter = nap.cross_correlogram(t1, t2, 1, 100)
	np.testing.assert_approx_equal(cc[101], 1.0)

	cc, bincenter = nap.cross_correlogram(t2, t1, 1, 100)
	np.testing.assert_approx_equal(cc[99], 1.0)

	t1 = np.array([0])
	t2 = np.array([100])
	cc, bincenter = nap.cross_correlogram(t1, t2, 1, 100)
	np.testing.assert_approx_equal(cc[200], 1.0)	

	t1 = np.array([0, 10])	
	cc, bincenter = nap.cross_correlogram(t1, t1, 1, 100)
	np.testing.assert_approx_equal(cc[100], 1.0)
	np.testing.assert_approx_equal(cc[90], 0.5)
	np.testing.assert_approx_equal(cc[110], 0.5)

	np.testing.assert_array_almost_equal(
		bincenter,
		np.arange(-100, 101)
		)

	for t in [100, 200, 1000]:
		np.testing.assert_array_almost_equal(
			nap.cross_correlogram(
				np.arange(0, t), np.arange(0, t), 1, t
				)[0],
			np.hstack((np.arange(0, 1, 1/t),
				np.ones(1),
				np.arange(0, 1, 1/t)[::-1]
				))
			)

@pytest.mark.parametrize("group", [
    nap.TsGroup({
    	0:nap.Ts(t=np.arange(0,100)),
	    1:nap.Ts(t=np.arange(0,100)),
	    2:nap.Ts(t=np.array([0,10])),
	    3:nap.Ts(t=np.arange(0,200))
    	})
    ])
class Test_Correlograms:
		
	def test_autocorrelogram(self, group):
		cc = nap.compute_autocorrelogram(group, 1, 100, norm=False)
		assert isinstance(cc, pd.DataFrame)
		assert list(cc.keys()) == list(group.keys())
		np.testing.assert_array_almost_equal(
			cc.index.values, np.arange(-100, 101, 1)
			)
		np.testing.assert_array_almost_equal(
			cc[0].values,
			np.hstack((np.arange(0, 1, 1/100),
				np.zeros(1),
				np.arange(0, 1, 1/100)[::-1]
				))
			)
		np.testing.assert_array_almost_equal(
			cc[0].values,
			np.hstack((np.arange(0, 1, 1/100),
				np.zeros(1),
				np.arange(0, 1, 1/100)[::-1]
				))
			)
		tmp = np.zeros(len(cc))
		tmp[[90,110]] = 0.5
		np.testing.assert_array_almost_equal(
			tmp,
			cc[2]
			)

	def test_autocorrelogram_with_ep(self, group):
		ep = nap.IntervalSet(start=0, end=99)
		cc = nap.compute_autocorrelogram(group, 1, 100, ep=ep, norm=False)
		np.testing.assert_array_almost_equal(
			cc[0].values,
			cc[3].values
			)

	def test_autocorrelogram_with_norm(self, group):		
		cc = nap.compute_autocorrelogram(group, 1, 100, norm=False)
		cc2 = nap.compute_autocorrelogram(group, 1, 100, norm=True)
		tmp = group._metadata['freq'].values.astype('float')
		np.testing.assert_array_almost_equal(cc/tmp, cc2)

	def test_autocorrelogram_time_units(self, group):
		cc = nap.compute_autocorrelogram(group, 1, 100, time_units='s')
		cc2 = nap.compute_autocorrelogram(group, 1*1e3, 100*1e3, time_units='ms')
		cc3 = nap.compute_autocorrelogram(group, 1*1e6, 100*1e6, time_units='us')
		pd.testing.assert_frame_equal(cc, cc2)
		pd.testing.assert_frame_equal(cc, cc3)


	def test_crosscorrelogram(self, group):	
		cc = nap.compute_crosscorrelogram(group, 1, 100, norm=False)
		assert isinstance(cc, pd.DataFrame)
		assert list(cc.keys()) == list(combinations(group.keys(),2))
		np.testing.assert_array_almost_equal(
			cc.index.values, np.arange(-100, 101, 1)
			)
		np.testing.assert_array_almost_equal(
			cc[(0,1)].values,
			np.hstack((np.arange(0, 1, 1/100),
				np.ones(1),
				np.arange(0, 1, 1/100)[::-1]
				))
			)

	def test_crosscorrelogram_with_ep(self, group):
		ep = nap.IntervalSet(start=0, end=99)
		cc = nap.compute_crosscorrelogram(group, 1, 100, ep=ep, norm=False)
		np.testing.assert_array_almost_equal(
			cc[(0,1)].values,
			cc[(0,3)].values
			)

	def test_crosscorrelogram_with_norm(self, group):		
		cc = nap.compute_crosscorrelogram(group, 1, 100, norm=False)
		cc2 = nap.compute_crosscorrelogram(group, 1, 100, norm=True)
		tmp = group._metadata['freq'].values.astype('float')
		tmp = tmp[[t[1] for t in cc.columns]]
		np.testing.assert_array_almost_equal(cc/tmp, cc2)

	def test_crosscorrelogram_time_units(self, group):
		cc =  nap.compute_crosscorrelogram(group, 1, 100, time_units='s')
		cc2 = nap.compute_crosscorrelogram(group, 1*1e3, 100*1e3, time_units='ms')
		cc3 = nap.compute_crosscorrelogram(group, 1*1e6, 100*1e6, time_units='us')
		pd.testing.assert_frame_equal(cc, cc2)
		pd.testing.assert_frame_equal(cc, cc3)


	def test_eventcorrelogram(self, group):
		cc = nap.compute_eventcorrelogram(group, group[0], 1, 100, norm=False)
		cc2 = nap.compute_crosscorrelogram(group, 1, 100, norm=False)
		assert isinstance(cc, pd.DataFrame)
		assert list(cc.keys()) == list(group.keys())
		np.testing.assert_array_almost_equal(
			cc[1].values, cc2[(0,1)].values
			)

	def test_eventcorrelogram_with_ep(self, group):
		ep = nap.IntervalSet(start=0, end=99)
		cc = nap.compute_eventcorrelogram(group, group[0], 1, 100, ep=ep, norm=False)
		cc2 = nap.compute_crosscorrelogram(group, 1, 100, ep=ep, norm=False)
		assert isinstance(cc, pd.DataFrame)
		assert list(cc.keys()) == list(group.keys())
		np.testing.assert_array_almost_equal(
			cc[1].values, cc2[(0,1)].values
			)

	def test_eventcorrelogram_with_norm(self, group):
		cc = nap.compute_eventcorrelogram(group, group[0], 1, 100, norm=False)
		cc2 = nap.compute_eventcorrelogram(group, group[0], 1, 100, norm=True)
		tmp = group._metadata['freq'].values.astype('float')	
		np.testing.assert_array_almost_equal(cc/tmp, cc2)

	def test_eventcorrelogram_time_units(self, group):
		cc =  nap.compute_eventcorrelogram(group, group[0], 1, 100, time_units='s')
		cc2 = nap.compute_eventcorrelogram(group, group[0], 1*1e3, 100*1e3, time_units='ms')
		cc3 = nap.compute_eventcorrelogram(group, group[0], 1*1e6, 100*1e6, time_units='us')
		pd.testing.assert_frame_equal(cc, cc2)
		pd.testing.assert_frame_equal(cc, cc3)

