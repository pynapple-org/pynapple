# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-11-30 09:29:21
# @Last Modified by:   gviejo
# @Last Modified time: 2022-11-30 18:38:52
"""Tests of time units for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
from pynapple.core.time_units import format_timestamps, return_timestamps, sort_timestamps
import warnings

def test_format_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(t, format_timestamps(t))
    np.testing.assert_array_almost_equal(t/1e3, format_timestamps(t, 'ms'))
    np.testing.assert_array_almost_equal(t/1e6, format_timestamps(t, 'us'))

    with pytest.raises(ValueError, match=r"unrecognized time units type"):
        format_timestamps(t, 'aaaa')

def test_return_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(t, return_timestamps(t))
    np.testing.assert_array_almost_equal(t/1e3, return_timestamps(t, 'ms'))
    np.testing.assert_array_almost_equal(t/1e6, return_timestamps(t, 'us'))

    with pytest.raises(ValueError, match=r"unrecognized time units type"):
        return_timestamps(t, 'aaaa')

def test_return_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(np.sort(t), sort_timestamps(t, False))

    with warnings.catch_warnings(record=True) as w:
        sort_timestamps(t, True)
    assert str(w[0].message) == "timestamps are not sorted"

