# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-07-10 14:38:27
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-07-11 16:00:06

"""Tests of IO folder functions"""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pynapple as nap

# look for tests folder
path = Path(__file__).parent
if path.name == "pynapple":
    path = path / "tests"
path = path / "npzfilestest"

# Recursively remove the folder:
shutil.rmtree(path, ignore_errors=True)
path.mkdir(exist_ok=True, parents=True)

# Populate the folder
data = {
    "tsd": nap.Tsd(t=np.arange(100), d=np.arange(100)),
    "ts": nap.Ts(t=np.sort(np.random.rand(10) * 100)),
    "tsdframe": nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 10)),
    "tsgroup": nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 200)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
        }
    ),
    "iset": nap.IntervalSet(start=np.array([0.0, 5.0]), end=np.array([1.0, 6.0])),
}

for k, d in data.items():
    d.save(path / (k + ".npz"))


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

    assert isinstance(folder["tsd2"], nap.Tsd)

    files = [f.name for f in path.iterdir()]
    assert "tsd2.json" in files

    # check json
    metadata = json.load(open(path / "tsd2.json", "r"))
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
