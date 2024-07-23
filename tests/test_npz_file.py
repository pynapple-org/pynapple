# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-10 17:08:55
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2024-04-11 13:13:37

"""Tests of NPZ file functions"""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings
from pathlib import Path
import shutil

# look for tests folder
path = Path(__file__).parent
if path.name == 'pynapple':
    path = path / "tests" 
path = path / "npzfilestest"

# Recursively remove the folder:
shutil.rmtree(path, ignore_errors=True)
path.mkdir(exist_ok=True, parents=True)

path2 = path.parent / "sub"
path2.mkdir(exist_ok=True, parents=True)


# Populate the folder
data = {
    "tsd":nap.Tsd(t=np.arange(100), d=np.arange(100)),
    "ts":nap.Ts(t=np.sort(np.random.rand(10)*100)),
    "tsdframe":nap.TsdFrame(t=np.arange(100), d=np.random.rand(100,10)),
    "tsgroup":nap.TsGroup({
            0: nap.Ts(t=np.arange(0, 200)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
        }),
    "iset":nap.IntervalSet(start=np.array([0.0, 5.0]), end=np.array([1.0, 6.0]))
    }
for k, d in data.items():
    d.save(path / (k+".npz"))

@pytest.mark.parametrize("path", [path])
def test_init(path):
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))   
    file_path = path / "tsd.npz"
    tsd.save(file_path)
    file = nap.NPZFile(file_path)
    assert isinstance(file, nap.NPZFile)
    assert isinstance(file.type, np.str_)
    assert file.type == "Tsd"

@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsd', 'ts', 'tsdframe', 'tsgroup', 'iset'])
def test_load(path, k):
    file_path = path / (k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))

@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsgroup'])
def test_load_tsgroup(path, k):
    file_path = path / (k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert tmp.keys() == data[k].keys()
    assert np.all(tmp._metadata == data[k]._metadata)
    assert np.all(tmp[neu] == data[k][neu] for neu in tmp.keys())
    np.testing.assert_array_almost_equal(tmp.time_support.values, data[k].time_support.values)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsd'])
def test_load_tsd(path, k):
    file_path = path / (k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert np.all(tmp.d == data[k].d)
    assert np.all(tmp.t == data[k].t)
    np.testing.assert_array_almost_equal(tmp.time_support.values, data[k].time_support.values)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['ts'])
def test_load_ts(path, k):
    file_path = path / (k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert np.all(tmp.t == data[k].t)
    np.testing.assert_array_almost_equal(tmp.time_support.values, data[k].time_support.values)



@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsdframe'])
def test_load_tsdframe(path, k):
    file_path = path / (k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert np.all(tmp.t == data[k].t)
    np.testing.assert_array_almost_equal(tmp.time_support.values, data[k].time_support.values)
    assert np.all(tmp.columns == data[k].columns)
    assert np.all(tmp.d == data[k].d)



@pytest.mark.parametrize("path", [path])
def test_load_non_npz(path):
    file_path = path / "random.npz"
    tmp = np.random.rand(100)
    np.savez(file_path, a=tmp)
    file = nap.NPZFile(file_path)    

    assert file.type == "npz"
    a = file.load()
    assert isinstance(a, np.lib.npyio.NpzFile)
    np.testing.assert_array_equal(tmp, a['a'])
