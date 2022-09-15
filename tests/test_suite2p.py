# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-09-09 11:59:30
# @Last Modified by:   gviejo
# @Last Modified time: 2022-09-15 13:51:30

"""Tests of Suite2P loader for `pynappple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings
import os

path = "nwbfilestest/suite2p/suite2p/plane0"

if not os.path.exists(path):
    path = os.path.join("tests", path)

F = np.load(os.path.join(path, "F.npy"), allow_pickle=True)
Fneu = np.load(os.path.join(path, "Fneu.npy"), allow_pickle=True)
spks = np.load(os.path.join(path, "spks.npy"), allow_pickle=True)
stat = np.load(os.path.join(path, "stat.npy"), allow_pickle=True)
ops = np.load(os.path.join(path, "ops.npy"), allow_pickle=True).item()
iscell = np.load(os.path.join(path, "iscell.npy"), allow_pickle=True)

idx = np.where(iscell[:, 0])[0]
F = F[idx]
Fneu = Fneu[idx]
spks = spks[idx]


@pytest.mark.filterwarnings("ignore")
def test_suite2p():
    try:
        data = nap.load_session("nwbfilestest/suite2p", "suite2p")
    except:
        data = nap.load_session("tests/nwbfilestest/suite2p", "suite2p")

    np.testing.assert_array_almost_equal(F, data.F.values.T)
    np.testing.assert_array_almost_equal(Fneu, data.Fneu.values.T)
    np.testing.assert_array_almost_equal(spks, data.spks.values.T)
    np.testing.assert_array_almost_equal(iscell, data.iscell)

    for k in ["xpix", "ypix", "lam"]:
        for n in idx:
            np.testing.assert_array_almost_equal(stat[n][k], data.stats[0][n][k])

    assert data.ops["Ly"] == ops["Ly"]
    assert data.ops["Lx"] == ops["Lx"]
