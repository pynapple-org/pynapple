"""Tests for metadata in IntervalSet, TsdFrame, and TsGroup"""

import inspect
import pickle
import warnings
from contextlib import nullcontext as does_not_raise
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


#################
## IntervalSet ##
#################
@pytest.fixture
def iset_meta():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    metadata = {"label": ["a", "b", "c", "d"], "info": np.arange(4)}
    return nap.IntervalSet(start=start, end=end, metadata=metadata)


@pytest.mark.parametrize(
    "index1",
    [
        # 0,
        # -1,
        # [0, 2],
        # [0, -1],
        # [0, 1, 3],
        # [0, 1, 2, 3],
        # slice(None),
        # slice(0, None),
        # slice(None, 2),
        # slice(0, 2),
        # slice(None, -1),
        # slice(0, -1),
        # slice(0, 1),
        # slice(None, 1),
        # slice(1, 3),
        # slice(1, -1),
        [True, False, True, False],
        # pd.Series([True, False, True, False]),
    ],
)
@pytest.mark.parametrize(
    "index2, output_type, has_metadata",
    [
        (None, nap.IntervalSet, True),
        (slice(None), nap.IntervalSet, True),
        (slice(0, None), nap.IntervalSet, True),
        (slice(None, 3), nap.IntervalSet, True),
        (slice(0, 3), nap.IntervalSet, True),
        (slice(None, 10), nap.IntervalSet, True),
        (slice(0, 10), nap.IntervalSet, True),
        (slice(None, 1), (np.ndarray, np.float64), False),
        (slice(0, 1), (np.ndarray, np.float64), False),
        (slice(1, 2), (np.ndarray, np.float64), False),
        (slice(0, 2), nap.IntervalSet, False),
        (slice(None, 2), nap.IntervalSet, False),
        (slice(1, 3), (np.ndarray, np.float64), False),
        (slice(3, 10), (np.ndarray, np.float64), False),
        ([0, 1], nap.IntervalSet, False),
        ([0, -1], nap.IntervalSet, False),
        ([0, 0, 1, 1], (np.ndarray, np.float64), False),
        ([0, 1, 0, 1], (np.ndarray, np.float64), False),
        (0, (np.ndarray, np.float64), False),
        (-1, (np.ndarray, np.float64), False),
        ([True, False], (np.ndarray, np.float64), False),
        ([True, True], nap.IntervalSet, False),
        (pd.Series([0, 1]), nap.IntervalSet, False),
        (pd.Series([0, 1, 1]), (np.ndarray, np.float64), False),
        (pd.Series([True, True]), nap.IntervalSet, False),
        (pd.Series([True, False]), (np.ndarray, np.float64), False),
    ],
)
def test_numpy_slice_iset_with_metadata(
    iset_meta, index1, index2, output_type, has_metadata
):
    if isinstance(index1, (list, pd.Series, pd.Index)) and isinstance(
        index2, (list, pd.Series, pd.Index)
    ):
        if np.issubdtype(np.array(index1).dtype, bool):
            len1 = np.sum(index1)
        else:
            len1 = len(index1)
        if np.issubdtype(np.array(index2).dtype, bool):
            len2 = np.sum(index2)
        else:
            len2 = len(index2)
        if len1 != len2:
            pytest.skip("index1 and index2 must have the same length")

    if index2 is None:
        index = index1
    else:
        index = (index1, index2)

    print(iset_meta)
    assert isinstance(iset_meta[index], output_type)

    expected = iset_meta.values[index]

    # check that indexing iset is the same as indexing the np.array values
    if output_type is nap.IntervalSet:
        np.testing.assert_array_almost_equal(
            iset_meta[index].values.squeeze(), expected.squeeze()
        )
    else:
        np.testing.assert_array_almost_equal(iset_meta[index], expected)

    if has_metadata:
        for col in iset_meta.metadata_columns:
            assert np.all(
                iset_meta[index].get_info(col) == iset_meta.get_info(col)[index1]
            )


@pytest.mark.parametrize(
    "index1",
    [
        0,
        -1,
        [0, -1],
        [0, 1, 2, 3],
        slice(None),
        slice(0, None),
        slice(None, 2),
        slice(0, 2),
        slice(None, -1),
        slice(0, -1),
        slice(0, 1),
        slice(None, 1),
        slice(1, 3),
        slice(1, -1),
    ],
)
@pytest.mark.parametrize(
    "index2",
    [
        slice(0, -1),
        slice(None, -1),
    ],
)
def test_numpy_slice_iset_with_metadata_special(iset_meta, index1, index2):
    index = (index1, index2)

    # first slice gets rid of metadata
    res1 = iset_meta[index]
    assert isinstance(res1, nap.IntervalSet)
    np.testing.assert_array_almost_equal(
        iset_meta.values[index1].squeeze(), res1.values.squeeze()
    )
    assert len(res1.metadata_columns) == 0

    # second slice gets rid of last column
    res2 = res1[index]
    assert isinstance(res2, (np.ndarray, np.float64))
    np.testing.assert_array_almost_equal(res2, res1.values[index])


@pytest.mark.parametrize(
    "index, expected",
    [
        (
            (slice(None), 2),
            pytest.raises(
                IndexError,
                match="index 2 is out of bounds for axis 1 with size 2",
            ),
        ),
        (
            (slice(None), [0, 3]),
            pytest.raises(
                IndexError,
                match="index 3 is out of bounds",
            ),
        ),
        (
            (slice(None), "label"),
            pytest.raises(
                IndexError,
                match="only integers",
            ),
        ),
        (
            (slice(None), ["label", "info"]),
            pytest.raises(
                IndexError,
                match="only integers",
            ),
        ),
        (
            (slice(None), [True, True, False]),
            pytest.raises(
                IndexError,
                match="boolean index did not match",
            ),
        ),
    ],
)
def test_slice_iset_with_metadata_errors(iset_meta, index, expected):
    with expected:
        iset_meta[index]
