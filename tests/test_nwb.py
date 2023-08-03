# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-04 21:32:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-08-03 10:35:24

"""Tests of nwb reading for `pynapple` package."""

import pynapple as nap
import numpy as np
import pandas as pd
import pytest
import warnings
import pynwb
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing.mock.utils import name_generator_registry


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
    from pynwb.testing.mock.base import mock_TimeSeries

    nwbfile = mock_NWBFile()
    time_series = mock_TimeSeries()
    nwbfile.add_acquisition(time_series)

    nwb = nap.NWBFile(nwbfile)

    assert len(nwb) == 1
    assert "TimeSeries" in nwb.keys()
    assert isinstance(nwb["TimeSeries"], nap.Tsd)
    tsd = nwb["TimeSeries"]
    np.testing.assert_array_almost_equal(tsd.index.values, np.arange(0, 0.4, 0.1))
    np.testing.assert_array_almost_equal(tsd.values, np.arange(1, 5))


def test_add_SpatialSeries():
    from pynwb.testing.mock.behavior import (
        mock_SpatialSeries,
        mock_Position,
        mock_PupilTracking,
        mock_CompassDirection,
    )

    for name, Series in zip(
        ["SpatialSeries", "Position", "PupilTracking", "CompassDirection"],
        [mock_SpatialSeries, mock_Position, mock_PupilTracking, mock_CompassDirection],
    ):
        name_generator_registry.clear()
        nwbfile = mock_NWBFile()
        nwbfile.add_acquisition(Series(name))
        nwb = nap.NWBFile(nwbfile)
        assert len(nwb) == 1
        assert "SpatialSeries" in nwb.keys() or "TimeSeries" in nwb.keys()
        assert isinstance(nwb[list(nwb.keys())[0]], nap.Tsd)


def test_add_Device():
    from pynwb.testing.mock.device import mock_Device

    nwbfile = mock_NWBFile()
    nwbfile.add_device(mock_Device())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 0


def test_add_Ecephys():
    from pynwb.testing.mock.ecephys import (
        mock_ElectrodeGroup,
        mock_ElectricalSeries,
        mock_SpikeEventSeries,
    )

    nwbfile = mock_NWBFile()
    nwbfile.add_electrode_group(mock_ElectrodeGroup())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 0

    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_acquisition(mock_ElectricalSeries())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "ElectricalSeries" in nwb.keys()
    data = nwb["ElectricalSeries"]
    assert isinstance(data, nap.TsdFrame)
    obj = nwbfile.acquisition["ElectricalSeries"]
    np.testing.assert_array_almost_equal(data.values, obj.data[:])
    np.testing.assert_array_almost_equal(
        data.index.values, obj.starting_time + np.arange(obj.num_samples) / obj.rate
    )
    np.testing.assert_array_almost_equal(data.columns.values, obj.electrodes["id"][:])

    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_acquisition(mock_SpikeEventSeries())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "SpikeEventSeries" in nwb.keys()
    data = nwb["SpikeEventSeries"]
    assert isinstance(data, nap.TsdFrame)
    obj = nwbfile.acquisition["SpikeEventSeries"]
    np.testing.assert_array_almost_equal(data.values, obj.data[:])
    np.testing.assert_array_almost_equal(data.index.values, obj.timestamps[:])
    np.testing.assert_array_almost_equal(data.columns.values, obj.electrodes["id"][:])


def test_add_Icephys():
    from pynwb.testing.mock.icephys import (
        mock_IntracellularElectrode,
        mock_VoltageClampStimulusSeries,
        mock_VoltageClampSeries,
        mock_CurrentClampSeries,
        mock_CurrentClampStimulusSeries,
        mock_IZeroClampSeries,
    )

    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_icephys_electrode(mock_IntracellularElectrode())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 0

    for name, Series in zip(
        [
            "VoltageClampStimulusSeries",
            "VoltageClampSeries",
            "CurrentClampSeries",
            "CurrentClampStimulusSeries",
            "IZeroClampSeries",
        ],
        [
            mock_VoltageClampStimulusSeries,
            mock_VoltageClampSeries,
            mock_CurrentClampSeries,
            mock_CurrentClampStimulusSeries,
            mock_IZeroClampSeries,
        ],
    ):
        name_generator_registry.clear()
        nwbfile = mock_NWBFile()
        nwbfile.add_acquisition(Series())
        nwb = nap.NWBFile(nwbfile)
        assert len(nwb) == 1
        assert name in nwb.keys()
        data = nwb[name]
        assert isinstance(data, nap.Tsd)
        obj = nwbfile.acquisition[name]
        np.testing.assert_array_almost_equal(data.values, obj.data[:])
        np.testing.assert_array_almost_equal(
            data.index.values, obj.starting_time + np.arange(obj.num_samples) / obj.rate
        )


def test_add_Ogen():
    from pynwb.testing.mock.ogen import (
        mock_OptogeneticStimulusSite,
        mock_OptogeneticSeries,
    )

    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_ogen_site(mock_OptogeneticStimulusSite())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 0

    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_acquisition(mock_OptogeneticSeries())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "OptogeneticSeries" in nwb.keys()
    data = nwb["OptogeneticSeries"]
    assert isinstance(data, nap.Tsd)
    obj = nwbfile.acquisition["OptogeneticSeries"]
    np.testing.assert_array_almost_equal(data.values, obj.data[:])
    np.testing.assert_array_almost_equal(
        data.index.values, obj.starting_time + np.arange(obj.num_samples) / obj.rate
    )


def test_add_Ophys():
    from pynwb.testing.mock.ophys import (
        mock_ImagingPlane,
        mock_OnePhotonSeries,
        mock_TwoPhotonSeries,
        mock_PlaneSegmentation,
        mock_ImageSegmentation,
        mock_RoiResponseSeries,
        mock_DfOverF,
        mock_Fluorescence
        )

    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_imaging_plane(mock_ImagingPlane())
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 0

    for Series in [
            # mock_OnePhotonSeries,
            # mock_TwoPhotonSeries,
            mock_RoiResponseSeries,
            mock_DfOverF,
            mock_Fluorescence
            ]:    
        name_generator_registry.clear()
        nwbfile = mock_NWBFile()
        nwbfile.add_acquisition(Series())
        nwb = nap.NWBFile(nwbfile)
        assert len(nwb) == 1
        assert "RoiResponseSeries" in nwb.keys()
        data = nwb["RoiResponseSeries"]
        assert isinstance(data, nap.TsdFrame)
        if "DfOverF" in nwbfile.acquisition.keys():
            obj = nwbfile.acquisition["DfOverF"]["RoiResponseSeries"]
        elif "Fluorescence" in nwbfile.acquisition.keys():
            obj = nwbfile.acquisition["Fluorescence"]["RoiResponseSeries"]
        else:
            obj = nwbfile.acquisition[list(nwbfile.acquisition.keys())[0]]
        np.testing.assert_array_almost_equal(data.values, obj.data[:])
        np.testing.assert_array_almost_equal(
            data.index.values, obj.starting_time + np.arange(obj.num_samples) / obj.rate
        )
        np.testing.assert_array_almost_equal(data.columns.values, obj.rois["id"][:])

def test_add_TimeIntervals():    
    # 1 epochset
    nwbfile = mock_NWBFile()
    nwbfile.add_trial(start_time=1.0, stop_time=5.0)
    nwbfile.add_trial(start_time=6.0, stop_time=10.0)

    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "trials" in nwb.keys()
    obj = nwbfile.trials
    data = nwb["trials"]
    assert isinstance(data, nap.IntervalSet)
    np.testing.assert_array_almost_equal(data.values, np.array([[1., 5.], [6., 10.]]))

    # Dict of epochs
    nwbfile = mock_NWBFile()
    nwbfile.add_trial_column(
        name="correct",
        description="whether the trial was correct",
    )
    nwbfile.add_trial(start_time=1.0, stop_time=5.0, correct=True)
    nwbfile.add_trial(start_time=6.0, stop_time=10.0, correct=False)

    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "trials" in nwb.keys()
    obj = nwbfile.trials
    data = nwb["trials"]
    assert isinstance(data, dict)
    assert True in data.keys() and False in data.keys()
    np.testing.assert_array_almost_equal(data[True].values, np.array([[1., 5.]]))
    np.testing.assert_array_almost_equal(data[False].values, np.array([[6., 10.]]))

    # Dataframe
    nwbfile = mock_NWBFile()
    nwbfile.add_trial_column(
        name="correct",
        description="whether the trial was correct",
    )
    nwbfile.add_trial_column(
        name="label",
        description="Whatever",
    )    
    nwbfile.add_trial(start_time=1.0, stop_time=5.0, correct=True, label=1)
    nwbfile.add_trial(start_time=6.0, stop_time=10.0, correct=False, label = 1)

    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "trials" in nwb.keys()
    obj = nwbfile.trials    
    with pytest.warns(UserWarning) as record:
        data = nwb["trials"]
    assert record[0].message.args[0] == "Too many metadata. Returning pandas.DataFrame, not IntervalSet"
    assert isinstance(data, pd.DataFrame)

def test_add_Epochs():
    # 1 epoch
    nwbfile = mock_NWBFile()
    nwbfile.add_epoch(
        start_time=2.0,
        stop_time=4.0,
        tags=["first", "example"],
        # timeseries=[time_series_with_timestamps],
    )

    nwbfile.add_epoch(
        start_time=6.0,
        stop_time=8.0,
        tags=["second", "example"],
        # timeseries=[time_series_with_timestamps],
    )
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "epochs" in nwb.keys()
    obj = nwbfile.epochs
    data = nwb["epochs"]
    assert isinstance(data, dict)
    np.testing.assert_array_almost_equal(data["first-example"].values, np.array([[2., 4.]]))
    np.testing.assert_array_almost_equal(data["second-example"].values, np.array([[6., 8.]]))
