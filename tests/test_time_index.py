# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-11-30 09:29:21
# @Last Modified by:   gviejo
# @Last Modified time: 2024-02-26 13:14:41
"""Tests of time units for `pynapple` package."""

# from pynapple.core.time_units import format_timestamps, return_timestamps, sort_timestamps
# from pynapple.core.time_index import TsIndex
import warnings

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


def test_format_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(t, nap.TsIndex.format_timestamps(t))
    np.testing.assert_array_almost_equal(
        t / 1e3, nap.TsIndex.format_timestamps(t, "ms")
    )
    np.testing.assert_array_almost_equal(
        t / 1e6, nap.TsIndex.format_timestamps(t, "us")
    )

    with pytest.raises(ValueError, match=r"unrecognized time units type"):
        nap.TsIndex.format_timestamps(t, "aaaa")


def test_return_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(t, nap.TsIndex.return_timestamps(t))
    np.testing.assert_array_almost_equal(
        t * 1e3, nap.TsIndex.return_timestamps(t, "ms")
    )
    np.testing.assert_array_almost_equal(
        t * 1e6, nap.TsIndex.return_timestamps(t, "us")
    )

    with pytest.raises(ValueError, match="unrecognized time units type"):
        nap.TsIndex.return_timestamps(t, units="aaaa")


def test_sort_timestamps():
    t = np.random.rand(100)

    np.testing.assert_array_almost_equal(
        np.sort(t), nap.TsIndex.sort_timestamps(t, False)
    )

    with warnings.catch_warnings(record=True) as w:
        nap.TsIndex.sort_timestamps(t, True)
    assert str(w[0].message) == "timestamps are not sorted"


def test_TsIndex():
    a = nap.TsIndex(np.arange(10))
    np.testing.assert_array_equal(a, np.arange(10))
    np.testing.assert_array_equal(a.values, np.arange(10))
    np.testing.assert_array_equal(a.to_numpy(), np.arange(10))

    np.testing.assert_array_equal(a.in_units("s"), np.arange(10))
    np.testing.assert_array_equal(a.in_units("ms"), np.arange(10) * 1e3)
    np.testing.assert_array_equal(a.in_units("us"), np.arange(10) * 1e6)

    with pytest.raises(RuntimeError, match=r"TsIndex object is not mutable."):
        a[0] = 1
