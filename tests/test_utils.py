"""Tests of utils for `pynapple` package."""

import numpy as np
import pandas as pd
import pytest

import pynapple as nap


def test_get_backend():
    assert nap.core.utils.get_backend() in ["numba", "jax"]


def test_is_array_like():
    assert nap.core.utils.is_array_like(np.ones(3))
    assert nap.core.utils.is_array_like(np.array([]))
    assert not nap.core.utils.is_array_like([1, 2, 3])
    assert not nap.core.utils.is_array_like(1)
    assert not nap.core.utils.is_array_like("a")
    assert not nap.core.utils.is_array_like(True)
    assert not nap.core.utils.is_array_like((1, 2, 3))
    assert not nap.core.utils.is_array_like({0: 1})
    assert not nap.core.utils.is_array_like(np.array(0))


@pytest.mark.parametrize(
    "data,expected",
    [
        # Regular
        (nap.Ts(t=np.arange(10, dtype=float)), True),
        (nap.Tsd(t=np.arange(10, dtype=float), d=np.arange(10)), True),
        (nap.TsdFrame(t=np.arange(10, dtype=float), d=np.random.randn(10, 3)), True),
        (
            nap.TsdTensor(t=np.arange(10, dtype=float), d=np.random.randn(10, 3, 4)),
            True,
        ),
        # Irregular
        (nap.Ts(t=[0.0, 0.1, 0.3, 0.6, 1.0]), False),
        (nap.Tsd(t=[0.0, 0.1, 0.3, 0.6, 1.0], d=np.arange(5)), False),
        (nap.TsdFrame(t=[0.0, 0.1, 0.3, 0.6, 1.0], d=np.random.randn(5, 3)), False),
        (nap.TsdTensor(t=[0.0, 0.1, 0.3, 0.6, 1.0], d=np.random.randn(5, 3, 4)), False),
        # Edge cases
        (nap.Tsd(t=[0.0], d=[1.0]), True),
        (nap.Tsd(t=[0.0, 1.0], d=[1.0, 2.0]), True),
        # Time support: regular within epochs, gap between should be ignored
        (
            nap.Tsd(
                t=np.concatenate([np.arange(0, 5), np.arange(10, 15)]),
                d=np.arange(10),
                time_support=nap.IntervalSet(start=[0, 10], end=[4, 14]),
            ),
            True,
        ),
        # Time support: irregular within an epoch
        (
            nap.Tsd(
                t=[0.0, 1.0, 1.5, 2.5, 3.5],
                d=np.arange(5),
                time_support=nap.IntervalSet(start=[0], end=[10]),
            ),
            False,
        ),
    ],
)
def test_is_regularly_sampled(data, expected):
    assert nap.core.utils._is_regularly_sampled(data) == expected
