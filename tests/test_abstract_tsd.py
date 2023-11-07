# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-09-25 11:53:30
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-25 12:37:45

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
from pynapple.core.time_series import _AbstractTsd


class MyClass(_AbstractTsd):

	def __getitem__(self, key):
		return key

	def __setitem__(self, key, value):
		pass

	def __getitem__(self, key):
		return key

	def __str__(self):
		return "In str"

	def __repr__(self):
		return "In repr"


def test_create_atsd():
	a = MyClass()

	assert hasattr(a, "rate")
	assert hasattr(a, "index")
	assert hasattr(a, "values")
	assert hasattr(a, "time_support")

	assert a.rate is np.NaN
	assert isinstance(a.index, nap.TsIndex)
	assert isinstance(a.values, np.ndarray)
	assert isinstance(a.time_support, nap.IntervalSet)

	assert hasattr(a, "t")
	assert hasattr(a, "d")
	assert hasattr(a, "start")
	assert hasattr(a, "end")
	assert hasattr(a, "__array__")
	np.testing.assert_array_equal(a.values, np.empty(0))
	np.testing.assert_array_equal(a.__array__(), np.empty(0))

	assert len(a) == 0

	assert a.__repr__() == "In repr"
	assert a.__str__() == "In str"

	assert hasattr(a, "__getitem__")
	assert hasattr(a, "__setitem__")
	assert a[0] == 0
	

def test_methods():
	a = MyClass()

	np.testing.assert_array_equal(a.times(), np.empty(0))
	np.testing.assert_array_equal(a.as_array(), np.empty(0))
	np.testing.assert_array_equal(a.data(), np.empty(0))
	np.testing.assert_array_equal(a.to_numpy(), np.empty(0))

	assert a.start_time() is None
	assert a.end_time() is None

	assert hasattr(a, "value_from")
	assert hasattr(a, "count")
	assert hasattr(a, "restrict")


