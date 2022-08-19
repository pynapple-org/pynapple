# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-04 22:35:44
# @Last Modified by:   gviejo
# @Last Modified time: 2022-08-19 09:03:33

"""Tests of neurosuite loader for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings


@pytest.mark.filterwarnings("ignore")
def test_load_session():
    try:
        data = nap.load_session("nwbfilestest/neurosuite", "neurosuite")
    except:
        data = nap.load_session("tests/nwbfilestest/neurosuite", "neurosuite")


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        data = nap.load_session("nwbfilestest/neurosuite", "neurosuite")
    except:
        data = nap.load_session("tests/nwbfilestest/neurosuite", "neurosuite")


@pytest.mark.parametrize("data", [data])
class Test_Neurosuite:
    def test_epochs(self, data):
        epochs = data.epochs
        assert isinstance(epochs, dict)
        assert "wake" in epochs.keys()
        assert "sleep" in epochs.keys()
        for k in epochs.keys():
            assert isinstance(epochs[k], nap.IntervalSet)

    def test_position(self, data):
        position = data.position
        assert isinstance(position, nap.TsdFrame)
        assert len(position.columns) == 6
        assert len(position) == 63527
        assert not np.all(np.isnan(position.values))

    def test_time_support(self, data):
        assert isinstance(data.time_support, nap.IntervalSet)

    def test_spikes(self, data):
        assert isinstance(data.spikes, nap.TsGroup)
        assert len(data.spikes) == 15
        for i in data.spikes.keys():
            assert len(data.spikes[i])
