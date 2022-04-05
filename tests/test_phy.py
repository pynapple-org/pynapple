# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-04 22:40:51
# @Last Modified by:   gviejo
# @Last Modified time: 2022-04-04 23:15:15

"""Tests of phy loader for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings


@pytest.mark.filterwarnings("ignore")
def test_load_session():    
    data = nap.load_session('nwbfilestest/phy', 'phy')
    
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data = nap.load_session('nwbfilestest/phy', 'phy')

@pytest.mark.parametrize("data", [data])
class Test_PHY:

    def test_epochs(self, data):
        epochs = data.epochs
        assert isinstance(epochs, dict)
        assert "wake" in epochs.keys()        
        for k in epochs.keys():
            assert isinstance(epochs[k], nap.IntervalSet)

    def test_spikes(self, data):
    	assert isinstance(data.spikes, nap.TsGroup)
    	assert len(data.spikes) == 3
    	for i in data.spikes.keys():
    		assert len(data.spikes[i])

    