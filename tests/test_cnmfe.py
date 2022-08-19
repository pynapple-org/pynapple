# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-04 22:40:51
# @Last Modified by:   gviejo
# @Last Modified time: 2022-08-19 09:09:23

"""Tests of CNMFE loaders for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings


@pytest.mark.filterwarnings("ignore")
def test_inscopix_cnmfe():
    try:
        data = nap.load_session("nwbfilestest/inscopix-cnmfe", "inscopix-cnmfe")
    except:
        data = nap.load_session("tests/nwbfilestest/inscopix-cnmfe", "inscopix-cnmfe")
    assert isinstance(data.C, nap.TsdFrame)
    assert len(data.C.columns) == 10
    assert len(data.C) > 0
    assert isinstance(data.A, np.ndarray)
    assert len(data.A) == len(data.C.columns)


@pytest.mark.filterwarnings("ignore")
def test_minian():
    try:
        data = nap.load_session("nwbfilestest/minian", "minian")
    except:
        data = nap.load_session("tests/nwbfilestest/minian", "minian")
    assert isinstance(data.C, nap.TsdFrame)
    assert len(data.C.columns) == 10
    assert len(data.C) > 0
    assert isinstance(data.A, np.ndarray)
    assert len(data.A) == len(data.C.columns)


@pytest.mark.filterwarnings("ignore")
def test_cnmfe_matlab():
    try:
        data = nap.load_session("nwbfilestest/matlab-cnmfe", "cnmfe-matlab")
    except:
        data = nap.load_session("tests/nwbfilestest/matlab-cnmfe", "cnmfe-matlab")
    assert isinstance(data.C, nap.TsdFrame)
    assert len(data.C.columns) == 10
    assert len(data.C) > 0
    assert isinstance(data.A, np.ndarray)
    assert len(data.A) == len(data.C.columns)
