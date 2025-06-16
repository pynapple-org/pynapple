"""Tests for repr of all objects"""

import inspect
import os
import re
import warnings
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

import pynapple as nap
from pynapple.core.metadata_class import _Metadata


@contextmanager
def mock_terminal_size(size):
    columns, rows = size
    fake_size = os.terminal_size((columns, rows))
    with patch("os.get_terminal_size", return_value=fake_size), patch(
        "shutil.get_terminal_size", return_value=fake_size
    ):
        yield


@pytest.mark.parametrize(
    "terminal_size",
    [
        (200, 200),  # Normal
        (2, 100),  # Cut rows
        (100, 2),  # Cut cols
        (2, 2),  # Cut rows & cols
    ],
)
class TestObjectRepr:

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t": np.arange(10), "d": np.arange(10)},
            {"t": np.array([]), "d": np.array([])},
        ],
    )
    def test_repr_tsd(self, terminal_size, kwargs):
        with mock_terminal_size(terminal_size):
            # Checking the actual terminal size
            assert terminal_size == nap.core.utils._get_terminal_size()

            # Making pynapple object
            obj = nap.Tsd(**kwargs)
            output = repr(obj)

            assert isinstance(output, str)
            assert isinstance(obj.__str__(), str)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t": np.arange(100), "d": np.random.rand(100, 1)},
            {"t": np.arange(10), "d": np.random.rand(10, 1)},
            {"t": np.arange(10), "d": np.random.rand(10, 3)},
            {"t": np.array([]), "d": np.ndarray(shape=[0, 3])},  # Empty
            {
                "t": np.array([]),
                "d": np.ndarray(shape=[0, 3]),
                "metadata": {"l1": np.ndarray(shape=3), "l2": np.ndarray(shape=3)},
            },
            {
                "t": np.arange(1),
                "d": np.random.rand(1, 3),
                "time_support": nap.IntervalSet(0, 2),
            },
            {"t": np.arange(100), "d": np.random.rand(100, 3)},
            {
                "t": np.arange(100),
                "d": np.random.rand(100, 3),
                "columns": ["a", "b", "c"],
            },
            {
                "t": np.arange(100),
                "d": np.random.rand(100, 3),
                "metadata": {"l1": np.arange(3), "l2": ["x", "x", "y"]},
            },
            {
                "t": np.arange(100),
                "d": np.random.rand(100, 3),
                "columns": ["a", "b", "c"],
                "metadata": {
                    "l1": np.arange(3),
                    "l2": ["x", "x", "y"],
                    "l3": np.random.randn(3),
                },
            },
            {
                "t": np.arange(100),
                "d": np.random.rand(100, 16),
                "columns": np.arange(16),
                "metadata": {f"x{i}": np.arange(16) for i in range(5)},
            },
        ],
    )
    def test_repr_tsdframe(self, terminal_size, kwargs):
        with mock_terminal_size(terminal_size):
            # Checking the actual terminal size
            assert terminal_size == nap.core.utils._get_terminal_size()

            # Making pynapple object
            obj = nap.TsdFrame(**kwargs)
            output = repr(obj)

            assert isinstance(output, str)
            assert isinstance(obj.__str__(), str)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t": np.arange(10)},
            {"t": np.array([])},
        ],
    )
    def test_repr_ts(self, terminal_size, kwargs):
        with mock_terminal_size(terminal_size):
            # Checking the actual terminal size
            assert terminal_size == nap.core.utils._get_terminal_size()

            # Making pynapple object
            obj = nap.Ts(**kwargs)
            output = repr(obj)

            assert isinstance(output, str)
            assert isinstance(obj.__str__(), str)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t": np.arange(10), "d": np.random.randn(10, 10, 20)},
            {"t": np.array([]), "d": np.ndarray(shape=(0, 10, 20))},
            {"t": np.array([]), "d": np.ndarray(shape=(0, 1, 2))},
        ],
    )
    def test_repr_tsdtensor(self, terminal_size, kwargs):
        with mock_terminal_size(terminal_size):
            # Checking the actual terminal size
            assert terminal_size == nap.core.utils._get_terminal_size()

            # Making pynapple object
            obj = nap.TsdTensor(**kwargs)
            output = repr(obj)

            assert isinstance(output, str)
            assert isinstance(obj.__str__(), str)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"start": np.array([0, 10, 16]), "end": np.array([5, 15, 20])},
            {"start": [], "end": []},
            {"start": np.sort(np.random.uniform(0, 100, 20))},
            {
                "start": np.array([0, 10, 16]),
                "end": np.array([5, 15, 20]),
                "metadata": {
                    "l1": np.arange(3),
                    "l2": ["x", "x", "y"],
                    "l3": np.random.randn(3),
                },
            },
            {
                "start": np.array([0, 10, 16]),
                "end": np.array([5, 15, 20]),
                "metadata": {
                    "l1": np.array(
                        [np.array([str(i)], dtype="object") for i in range(3)]
                    ),
                    "l2": np.array(
                        [
                            np.array([str(i), str(i + 1)], dtype="object")
                            for i in range(3)
                        ]
                    ),
                    "l3": np.random.randn(3),
                },
            },
        ],
    )
    def test_repr_intervalset(self, terminal_size, kwargs):
        with mock_terminal_size(terminal_size):
            # Checking the actual terminal size
            assert terminal_size == nap.core.utils._get_terminal_size()

            # Making pynapple object
            obj = nap.IntervalSet(**kwargs)
            output = repr(obj)

            assert isinstance(output, str)
            assert isinstance(obj.__str__(), str)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"data": {i: nap.Ts(np.arange(10)) for i in range(10)}},
            {
                "data": {i: nap.Ts(np.arange(10)) for i in range(10)},
                "metadata": {
                    "abc": ["a"] * 10,
                    "bbb": [1] * 10,
                    "ccc": [np.pi] * 10,
                },
            },
            {
                "data": {i: nap.Ts(np.arange(10)) for i in range(200)},
                "metadata": {
                    "abc": ["a"] * 200,
                    "bbb": [1] * 200,
                    "ccc": [np.pi] * 200,
                },
            },
            {
                "data": {i: nap.Ts(np.arange(10)) for i in range(20)},
                "metadata": {"a" * i: [1] * 20 for i in range(1, 21)},
            },
            {
                "data": {},
                "time_support": nap.IntervalSet(0, 10),
                "metadata": {"l1": []},
            },
        ],
    )
    def test_repr_tsgroup(self, terminal_size, kwargs):
        with mock_terminal_size(terminal_size):
            # Checking the actual terminal size
            assert terminal_size == nap.core.utils._get_terminal_size()

            # Making pynapple object
            obj = nap.TsGroup(**kwargs)
            output = repr(obj)

            assert isinstance(output, str)
            assert isinstance(obj.__str__(), str)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "index": np.arange(10),
                "data": {
                    "abc": ["a"] * 10,
                    "bbb": [1] * 10,
                    "ccc": [np.pi] * 10,
                },
            },
        ],
    )
    def test_repr_metadata(self, terminal_size, kwargs):
        with mock_terminal_size(terminal_size):
            # Checking the actual terminal size
            assert terminal_size == nap.core.utils._get_terminal_size()

            # Making pynapple object
            obj = _Metadata(**kwargs)
            output = repr(obj)

            assert isinstance(output, str)
            assert isinstance(obj.__str__(), str)
