# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-10 17:08:55
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-10 18:14:49

"""Tests of NPZ file functions"""

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

# Cleaning
for root, dirs, files in os.walk(path):
    for f in files:        
        os.remove(os.path.join(root, f))

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
    d.save(os.path.join(path, k+".npz"))

@pytest.mark.parametrize("path", [path])
def test_init(path):
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))   
    file_path = os.path.join(path, "tsd.npz")
    tsd.save(file_path)
    file = nap.NPZFile(file_path)
    assert isinstance(file, nap.NPZFile)
    assert isinstance(file.type, np.str_)
    assert file.type == "Tsd"

@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsd', 'ts', 'tsdframe', 'tsgroup', 'iset'])
def test_load(path, k):
    file_path = os.path.join(path, k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert type(tmp) == type(data[k])

@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsgroup'])
def test_load_tsgroup(path, k):
    file_path = os.path.join(path, k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert type(tmp) == type(data[k])
    assert tmp.keys() == data[k].keys()
    assert np.all(tmp._metadata == data[k]._metadata)
    assert np.all(tmp[neu] == data[k][neu] for neu in tmp.keys())
    assert np.all(tmp.time_support == data[k].time_support)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsd'])
def test_load_tsd(path, k):
    file_path = os.path.join(path, k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert type(tmp) == type(data[k])
    assert np.all(tmp.d == data[k].d)
    assert np.all(tmp.t == data[k].t)
    assert np.all(tmp.time_support == data[k].time_support)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['ts'])
def test_load_ts(path, k):
    file_path = os.path.join(path, k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert type(tmp) == type(data[k])
    assert np.all(tmp.t == data[k].t)
    assert np.all(tmp.time_support == data[k].time_support)



@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ['tsdframe'])
def test_load_tsdframe(path, k):
    file_path = os.path.join(path, k+".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert type(tmp) == type(data[k])
    assert np.all(tmp.t == data[k].t)
    assert np.all(tmp.time_support == data[k].time_support)
    assert np.all(tmp.columns == data[k].columns)
    assert np.all(tmp.d == data[k].d)



@pytest.mark.parametrize("path", [path])
def test_load_non_npz(path):
    file_path = os.path.join(path, "random.npz")
    tmp = np.random.rand(100)
    np.savez(file_path, a = tmp)
    file = nap.NPZFile(file_path)    

    assert file.type == "npz"
    a = file.load()
    assert isinstance(a, np.lib.npyio.NpzFile)
    np.testing.assert_array_equal(tmp, a['a'])





