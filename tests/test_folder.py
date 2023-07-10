# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-10 14:38:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-10 17:03:39

"""Tests of IO folder functions"""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings
import os
import json

# look for npzfilestest folder
path = ""
for root, dirs, files in os.walk(".", topdown=False):
    if "npzfilestest" in dirs:
        path = os.path.join(os.path.abspath(root), "npzfilestest")
        break

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
# Cleaning
for root, dirs, files in os.walk(path, topdown=False):
    for f in files:
        os.remove(os.path.join(root, f))

for k, d in data.items():
    d.save(os.path.join(path, k+".npz"))
for k, d in data.items():
    d.save(os.path.join(path, "sub", k+".npz"))

@pytest.mark.parametrize("path", [path])
def test_load_folder(path):
    folder = nap.Folder(path)
    assert isinstance(folder, nap.Folder)

@pytest.mark.parametrize("path", [path])
def test_get_item(path):
    folder = nap.Folder(path)
    for k in data.keys():
        assert isinstance(folder.data[k], nap.NPZFile)
        d = folder[k]
        assert type(data[k]) == type(d)
        assert type(folder.data[k]) == type(d)

##################################################################
folder = nap.Folder(path)

@pytest.mark.parametrize("folder", [folder])
def test_expand(folder):
    assert folder.expand() == None
    assert folder.view == None

@pytest.mark.parametrize("folder", [folder])
def test_save(folder):
    tsd2 = nap.Tsd(t=np.arange(10), d=np.arange(10))
    folder.save("tsd2", tsd2, "Test description")

    assert isinstance(folder['tsd2'], nap.Tsd)

    files = os.listdir(folder.path)
    assert "tsd2.json" in files

    # check json
    metadata = json.load(open(os.path.join(path, "tsd2.json"), "r"))
    assert "time" in metadata.keys()
    assert "info" in metadata.keys()
    assert "Test description" == metadata["info"]

# @pytest.mark.parametrize("folder", [folder])
# def test_metadata(folder):
#     tsd2 = nap.Tsd(t=np.arange(10), d=np.arange(10))
#     folder.save("tsd2", tsd2, "Test description")
#     folder.metadata("tsd2")
#     folder.doc("tsd2")

@pytest.mark.parametrize("path", [path])
def test_load(path):
    folder = nap.Folder(path)
    for k in data.keys():
        assert isinstance(folder.data[k], nap.NPZFile)    
    folder.load()
    for k in data.keys():
        assert type(folder[k]) == type(data[k])















