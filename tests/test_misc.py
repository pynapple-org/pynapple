# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-10 12:26:20
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-07-31 11:17:59

"""Tests of IO misc functions"""

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pynapple as nap
from tests.helper_tests import skip_if_backend

# look for tests folder
path = Path(__file__).parent
if path.name == "pynapple":
    path = path / "tests"
path = path / "npzfilestest"

# Recursively remove the folder:
shutil.rmtree(path, ignore_errors=True)
path.mkdir(exist_ok=True, parents=True)

path2 = path.parent / "sub"
path2.mkdir(exist_ok=True, parents=True)


@pytest.mark.parametrize("path", [path])
def test_load_file(path):
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    file_path = path / "tsd.npz"
    tsd.save(file_path)
    tsd2 = nap.load_file(file_path)

    assert isinstance(tsd2, nap.Tsd)
    np.testing.assert_array_equal(tsd.index, tsd2.index)
    np.testing.assert_array_equal(tsd.values, tsd2.values)
    np.testing.assert_array_equal(tsd.time_support.values, tsd2.time_support.values)

    # file_path.unlink()


@pytest.mark.parametrize("path", [path])
def test_load_file_filenotfound(path):
    with pytest.raises(FileNotFoundError) as e:
        nap.load_file("themissingfile.npz")

    assert str(e.value) == "File themissingfile.npz does not exist"


@pytest.mark.parametrize("path", [path])
def test_load_wrong_format(path):
    file_path = path / "test.npy"
    np.save(file_path, np.random.rand(10))
    with pytest.raises(RuntimeError) as e:
        nap.load_file(file_path)

    assert str(e.value) == "File format not supported"
    # file_path.unlink()


@pytest.mark.parametrize("path", [path])
def test_load_folder(path):
    folder = nap.load_folder(path)
    assert isinstance(folder, nap.io.Folder)


def test_load_folder_foldernotfound():
    with pytest.raises(FileNotFoundError) as e:
        nap.load_folder("MissingFolder")

    assert str(e.value) == "Folder MissingFolder does not exist"


@skip_if_backend("jax")
@pytest.mark.parametrize("path", [path])
def test_load_eeg(path):
    filepath = path / "memmap.dat"
    tmp = np.random.randn(10, 3).astype("int16")
    data = np.memmap(filename=filepath, dtype="int16", mode="w+", shape=(10, 3))
    data[:] = tmp
    data.flush()

    # All channels
    eeg = nap.load_eeg(filepath, n_channels=3, frequency=100, precision="int16")

    assert isinstance(eeg, nap.TsdFrame)
    np.testing.assert_array_almost_equal(tmp, eeg.values)
    np.testing.assert_array_almost_equal(eeg.t, np.arange(0, 10) / 100)
    assert isinstance(eeg.values, np.memmap)

    # List of channels
    eeg = nap.load_eeg(
        filepath, channel=[0, 2], n_channels=3, frequency=100, precision="int16"
    )

    assert isinstance(eeg, nap.TsdFrame)
    np.testing.assert_array_almost_equal(tmp[:, [0, 2]], eeg.values)
    assert isinstance(eeg.values, np.ndarray)

    # Single channel
    eeg = nap.load_eeg(
        filepath, channel=0, n_channels=3, frequency=100, precision="int16"
    )

    assert isinstance(eeg, nap.Tsd)
    np.testing.assert_array_almost_equal(tmp[:, 0], eeg.values)
    assert isinstance(eeg.values, np.ndarray)

    filepath.unlink()
