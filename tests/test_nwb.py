# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-04 21:32:10
# @Last Modified by:   gviejo
# @Last Modified time: 2023-07-28 17:14:04

"""Tests of nwb reading for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings
import pynwb
from pynwb.testing.mock.file import mock_NWBFile



############################################################
# DEPRECATED PART ##########################################
############################################################

@pytest.mark.filterwarnings("ignore")
def test_load_session():
    try:
        data = nap.load_session("nwbfilestest/basic")
    except:
        data = nap.load_session("tests/nwbfilestest/basic")


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        data = nap.load_session("nwbfilestest/basic")
    except:
        data = nap.load_session("tests/nwbfilestest/basic")


@pytest.mark.parametrize("data", [data])
class Test_NWB:
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

    @pytest.mark.filterwarnings("ignore")
    def test_nwb_meta_info(self, data):
        from pynwb import NWBFile, NWBHDF5IO

        io = NWBHDF5IO(data.nwbfilepath, "r")
        nwbfile = io.read()
        assert nwbfile.experimenter == ("guillaume",)
        io.close()

############################################################
############################################################

def test_NWBFile():
    from collections import UserDict

    nwbfile = mock_NWBFile()
    nwb = nap.NWBFile(nwbfile)

    assert isinstance(nwb, UserDict)
    assert len(nwb) == 0
    try:
        nwb = nap.NWBFile("tests/nwbfilestest/basic/pynapplenwb/A2929-200711.nwb")
    except:
        nwb = nap.NWBFile("nwbfilestest/basic/pynapplenwb/A2929-200711.nwb")
    
    assert nwb.name == "A2929-200711"
    assert isinstance(nwb.io, pynwb.NWBHDF5IO)

def test_NWBFile_missing_file():
    with pytest.raises(FileNotFoundError) as e_info:
        nap.NWBFile("tests/file1.nwb")
    assert str(e_info.value) == "[Errno 2] No such file or directory: 'tests/file1.nwb'"

def test_NWBFile_wrong_input():
    with pytest.raises(RuntimeError):
        nap.NWBFile(1)    

def test_add_TimeSeries():
    nwbfile = mock_NWBFile()
    from pynwb.testing.mock.base import mock_TimeSeries
    time_series = mock_TimeSeries()
    nwbfile.add_acquisition(time_series)

    nwb = nap.NWBFile(nwbfile)

    assert len(nwb) == 1
    assert "TimeSeries" in nwb.keys()

