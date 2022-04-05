# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-04 22:40:51
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-05 10:49:38

"""Tests of CNMFE loaders for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings

@pytest.mark.filterwarnings("ignore")
def test_inscopix_cnmfe():
    data = nap.load_session('nwbfilestest/inscopix-cnmfe', 'inscopix-cnmfe')
    assert isinstance(data.C, nap.TsdFrame)
    assert len(data.C.columns) == 10
    assert len(data.C) > 0
    assert isinstance(data.A, np.ndarray)
    assert len(data.A) == len(data.C.columns)

@pytest.mark.filterwarnings("ignore")
def test_minian():
    data = nap.load_session('nwbfilestest/minian', 'minian')
    assert isinstance(data.C, nap.TsdFrame)
    assert len(data.C.columns) == 10
    assert len(data.C) > 0
    assert isinstance(data.A, np.ndarray)
    assert len(data.A) == len(data.C.columns)

@pytest.mark.filterwarnings("ignore")
def test_cnmfe_matlab():
    data = nap.load_session('nwbfilestest/matlab-cnmfe', 'cnmfe-matlab')
    assert isinstance(data.C, nap.TsdFrame)
    assert len(data.C.columns) == 10
    assert len(data.C) > 0
    assert isinstance(data.A, np.ndarray)
    assert len(data.A) == len(data.C.columns)


