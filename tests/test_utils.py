"""Tests of utils for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest

def test_get_backend():
	assert nap.core.utils.get_backend() in ["numba", "jax"]

def test_is_array_like():
	assert nap.core.utils.is_array_like(np.ones(3))
	assert nap.core.utils.is_array_like(np.array([]))
	assert not nap.core.utils.is_array_like([1,2,3])
	assert not nap.core.utils.is_array_like(1)
	assert not nap.core.utils.is_array_like('a')
	assert not nap.core.utils.is_array_like(True)
	assert not nap.core.utils.is_array_like((1,2,3))
	assert not nap.core.utils.is_array_like({0:1})
	assert not nap.core.utils.is_array_like(np.array(0))