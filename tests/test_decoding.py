# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-03-30 11:16:39
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-01 17:57:41
#!/usr/bin/env python

"""Tests of decoding for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest

def get_testing_set_1d():
	feature = nap.Tsd(
		t=np.arange(0, 100, 0.1),
		d=np.repeat(np.arange(0, 100),10)
		)
	group = nap.TsGroup({
		i:nap.Ts(t=feature.index.values[feature.values==i]) for i in range(100)
		})
	tc = nap.compute_1d_tuning_curves(
		group=group,
		feature=feature,
		nb_bins=100,
		minmax=(0,100)
		)	
	ep = nap.IntervalSet(start=0, end=100)
	return feature, group, tc, ep

def test_decode_1d():
	feature, group, tc, ep = get_testing_set_1d()
	decoded, proba = nap.decode_1d(tc, group, ep, bin_size=0.1, feature = feature)


def test_decode_2d():
	pass