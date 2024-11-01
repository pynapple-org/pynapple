# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-11-30 09:29:21
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-21 15:59:54
"""Tests of time units for `pynapple` package."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pynapple as nap

# from pynapple.core.time_units import format_timestamps, return_timestamps, sort_timestamps
from pynapple.core.time_index import TsIndex


def test_format_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(t, TsIndex.format_timestamps(t))
    np.testing.assert_array_almost_equal(t / 1e3, TsIndex.format_timestamps(t, "ms"))
    np.testing.assert_array_almost_equal(t / 1e6, TsIndex.format_timestamps(t, "us"))

    with pytest.raises(ValueError, match=r"unrecognized time units type"):
        TsIndex.format_timestamps(t, "aaaa")


def test_return_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(t, TsIndex.return_timestamps(t))
    np.testing.assert_array_almost_equal(t / 1e3, TsIndex.return_timestamps(t, "ms"))
    np.testing.assert_array_almost_equal(t / 1e6, TsIndex.return_timestamps(t, "us"))

    with pytest.raises(ValueError, match=r"unrecognized time units type"):
        TsIndex.return_timestamps(t, "aaaa")


def test_return_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(np.sort(t), TsIndex.sort_timestamps(t, False))

    with warnings.catch_warnings(record=True) as w:
        TsIndex.sort_timestamps(t, True)
    assert str(w[0].message) == "timestamps are not sorted"
