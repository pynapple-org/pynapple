# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-09-25 11:53:30
# @Last Modified by:   gviejo
# @Last Modified time: 2024-02-26 13:06:55

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
from pynapple.core.time_series import BaseTsd
from pynapple.core.base_class import Base
from pynapple.core.time_index import TsIndex


class MyClass(BaseTsd):

	def __getitem__(self, key):
		return key

	def __setitem__(self, key, value):
		pass

	def __str__(self):
		return "In str"

	def __repr__(self):
		return "In repr"

class MyClass2(Base):

	def __getitem__(self, key):
		return key

	def __setitem__(self, key, value):
		pass

	def __str__(self):
		return "In str"

	def __repr__(self):
		return "In repr"

def test_create_atsd():
	a = MyClass(t=np.arange(10), d=np.arange(10))

	assert hasattr(a, "rate")
	assert hasattr(a, "index")
	assert hasattr(a, "values")
	assert hasattr(a, "time_support")

	assert np.isclose(a.rate, 10/9)
	assert isinstance(a.index, nap.TsIndex)
	assert isinstance(a.values, np.ndarray)
	assert isinstance(a.time_support, nap.IntervalSet)

	assert hasattr(a, "t")
	assert hasattr(a, "d")
	assert hasattr(a, "start")
	assert hasattr(a, "end")
	assert hasattr(a, "__array__")
	assert hasattr(a, "shape")
	assert hasattr(a, "ndim")
	assert hasattr(a, "size")

	np.testing.assert_array_equal(a.values, np.arange(10))
	np.testing.assert_array_equal(a.__array__(), np.arange(10))

	assert len(a) == 10

	assert a.__repr__() == "In repr"
	assert a.__str__() == "In str"

	assert hasattr(a, "__getitem__")
	assert hasattr(a, "__setitem__")
	assert a[0] == 0

	b = a.copy()
	np.testing.assert_array_equal(a.values, b.values)
	np.testing.assert_array_equal(a.index.values, b.index.values)

def test_create_ats():

	a = MyClass2(t=np.arange(10))

	assert hasattr(a, "rate")
	assert hasattr(a, "index")
	assert hasattr(a, "time_support")
	assert hasattr(a, "shape")

	assert np.isclose(a.rate, 10/9)
	assert isinstance(a.index, nap.TsIndex)	
	assert isinstance(a.time_support, nap.IntervalSet)
	assert a.shape == a.index.shape

	assert hasattr(a, "t")
	assert a[0] == 0

def test_create_ats_from_tsindex():

	a = MyClass2(t=TsIndex(np.arange(10)))

	assert hasattr(a, "rate")
	assert hasattr(a, "index")
	assert hasattr(a, "time_support")
	assert hasattr(a, "shape")

	assert np.isclose(a.rate, 10/9)
	assert isinstance(a.index, nap.TsIndex)	
	assert isinstance(a.time_support, nap.IntervalSet)
	assert a.shape == a.index.shape

	assert hasattr(a, "t")	

@pytest.mark.filterwarnings("ignore")
def test_create_ats_from_number():

	a = MyClass2(t=1)

	assert hasattr(a, "rate")
	assert hasattr(a, "index")
	assert hasattr(a, "time_support")
	assert hasattr(a, "shape")
	

def test_methods():
	a = MyClass(t=[], d=[])

	np.testing.assert_array_equal(a.times(), np.empty(0))
	np.testing.assert_array_equal(a.as_array(), np.empty(0))
	np.testing.assert_array_equal(a.data(), np.empty(0))
	np.testing.assert_array_equal(a.to_numpy(), np.empty(0))

	assert a.start_time() is None
	assert a.end_time() is None

	assert hasattr(a, "value_from")
	assert hasattr(a, "count")
	assert hasattr(a, "restrict")
	assert hasattr(a, "as_array")
	assert hasattr(a, "data")
	assert hasattr(a, "to_numpy")
	assert hasattr(a, "copy")
	assert hasattr(a, "bin_average")
	assert hasattr(a, "dropna")
	assert hasattr(a, "convolve")
	assert hasattr(a, "smooth")
	assert hasattr(a, "interpolate")



