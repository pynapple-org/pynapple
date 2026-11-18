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


@pytest.mark.parametrize("dtype", [np.float64, object])
@pytest.mark.parametrize(
    "key",
    [
        [[0, 1], [2, 3]],
        np.array([[0], [2]]),
        np.empty((0, 2), dtype=int),
        [[[0]]],
        None,
        (slice(None), None),
        (Ellipsis, None),
        True,
        np.array(False),
    ],
    ids=[
        "nested-list",
        "integer-array",
        "empty-array",
        "three-dimensions",
        "leading-newaxis",
        "trailing-newaxis",
        "ellipsis-newaxis",
        "true",
        "false-array",
    ],
)
def test_TsIndex_getitem_rejects_multidimensional_result(key, dtype):
    index = nap.TsIndex(np.arange(4)).astype(dtype)

    with pytest.raises(ValueError, match="one-dimensional"):
        index[key]


@pytest.mark.parametrize("dtype", [np.float64, object])
@pytest.mark.parametrize("key, expected", [(0, 0.0), (-1, 3.0), (np.array(2), 2.0)])
def test_TsIndex_getitem_preserves_scalar(key, expected, dtype):
    index = nap.TsIndex(np.arange(4)).astype(dtype)

    result = index[key]

    assert np.isscalar(result)
    assert result == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        (slice(1, 3), [1.0, 2.0]),
        (slice(None, None, -1), [3.0, 2.0, 1.0, 0.0]),
        ([3, 1], [3.0, 1.0]),
        (np.array([True, False, True, False]), [0.0, 2.0]),
        ([], []),
        (np.array([], dtype=int), []),
        (Ellipsis, [0.0, 1.0, 2.0, 3.0]),
        ((), [0.0, 1.0, 2.0, 3.0]),
        ((None, 1), [1.0]),
        ((1, None), [1.0]),
    ],
)
def test_TsIndex_getitem_preserves_one_dimensional_result(key, expected):
    index = nap.TsIndex(np.arange(4))

    result = index[key]

    assert isinstance(result, nap.TsIndex)
    assert result.ndim == 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "key, error",
    [
        (4, IndexError),
        ([True, False], IndexError),
        (1.5, IndexError),
        ((Ellipsis, Ellipsis), IndexError),
        (slice(None, None, 0), ValueError),
    ],
)
def test_TsIndex_getitem_preserves_invalid_index_errors(key, error):
    index = nap.TsIndex(np.arange(4))

    with pytest.raises(error):
        index[key]
