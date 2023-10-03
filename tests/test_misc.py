# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-10 12:26:20
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-18 16:05:24

"""Tests of IO misc functions"""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings
import os

# look for tests folder
path = os.getcwd()
if os.path.basename(path) == 'pynapple':
    path = os.path.join(path, "tests")

path = os.path.join(path, "npzfilestest")
if not os.path.isdir(path):
    os.mkdir(path)
path2 = os.path.join(path, "sub")
if not os.path.isdir(path):
    os.mkdir(path2)

@pytest.mark.parametrize("path", [path])
def test_load_file(path):
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))   
    file_path = os.path.join(path, "tsd.npz")
    tsd.save(file_path)
    tsd2 = nap.load_file(file_path)

    assert isinstance(tsd2, nap.Tsd)
    np.testing.assert_array_equal(tsd.index, tsd2.index)
    np.testing.assert_array_equal(tsd.values, tsd2.values)
    np.testing.assert_array_equal(tsd.time_support.values, tsd2.time_support.values)

    os.remove(file_path)

@pytest.mark.parametrize("path", [path])
def test_load_file_filenotfound(path):
    with pytest.raises(FileNotFoundError) as e:
        nap.load_file("themissingfile.npz")

    assert str(e.value) == "File themissingfile.npz does not exist"

@pytest.mark.parametrize("path", [path])
def test_load_wrong_format(path):
    file_path = os.path.join(path, "test.npy")
    np.save(file_path, np.random.rand(10))
    with pytest.raises(RuntimeError) as e:
        nap.load_file(file_path)

    assert str(e.value) == "File format not supported"
    os.remove(file_path)

@pytest.mark.parametrize("path", [path])
def test_load_folder(path):
    folder = nap.load_folder(path)
    assert isinstance(folder, nap.io.Folder)

def test_load_folder_foldernotfound():
    with pytest.raises(RuntimeError) as e:
        nap.load_folder("MissingFolder")

    assert str(e.value) == "Folder MissingFolder does not exist"

