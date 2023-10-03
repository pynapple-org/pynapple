# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-04-04 21:32:10
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-18 10:28:42

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

def test_wrong_key():
    nwbfile = mock_NWBFile()
    nwb = nap.NWBFile(nwbfile)    
    with pytest.raises(KeyError):
        nwb["a"]


def test_failed_to_build():
    from pynwb.file import Subject
    nwbfile = mock_NWBFile(subject=Subject(subject_id="mouse1"))
    nwb = nap.NWBFile(nwbfile)    
    for oid, obj in nwbfile.objects.items():                                                                                                                                   
        nwb.key_to_id[obj.name] = oid 
        nwb[obj.name] = {"id": oid, "type": "Tsd"}

    with pytest.warns(UserWarning) as record:
        nwb["subject"]
    assert (
        record[0].message.args[0]
        == "Failed to build Tsd.\n Returning the NWB object for manual inspection"
    )
    
def test_add_TimeSeries():
    from pynwb.testing.mock.base import mock_TimeSeries

    # Tsd
    nwbfile = mock_NWBFile()
    time_series = mock_TimeSeries()
    nwbfile.add_acquisition(time_series)
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "TimeSeries" in nwb.keys()
    assert isinstance(nwb["TimeSeries"], nap.Tsd)
    tsd = nwb["TimeSeries"]
    np.testing.assert_array_almost_equal(tsd.index, np.arange(0, 0.4, 0.1))
    np.testing.assert_array_almost_equal(tsd.values, np.arange(1, 5))

    # Tsd with timestamps
    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    time_series = mock_TimeSeries(timestamps=np.arange(10), data=np.arange(10), rate=None)
    nwbfile.add_acquisition(time_series)
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "TimeSeries" in nwb.keys()
    assert isinstance(nwb["TimeSeries"], nap.Tsd)
    tsd = nwb["TimeSeries"]
    np.testing.assert_array_almost_equal(tsd.index, np.arange(10))
    np.testing.assert_array_almost_equal(tsd.values, np.arange(10))


    # TsdFrame
    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    time_series = mock_TimeSeries(data=np.zeros((10,3)))
    nwbfile.add_acquisition(time_series)
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "TimeSeries" in nwb.keys()
    assert isinstance(nwb["TimeSeries"], nap.TsdFrame)
    tsdframe = nwb["TimeSeries"]
    np.testing.assert_array_almost_equal(tsdframe.index, np.arange(0, 1.0, 0.1))
    np.testing.assert_array_almost_equal(tsdframe.values, np.zeros((10, 3)))


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

    # Test for 2d and 3d
    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_acquisition(mock_SpatialSeries(data=np.zeros((4,2))))
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "SpatialSeries" in nwb.keys() or "TimeSeries" in nwb.keys()
    data = nwb[list(nwb.keys())[0]]
    assert isinstance(data, nap.TsdFrame)
    np.testing.assert_array_equal(data.values,np.zeros((4, 2)))
    np.testing.assert_array_equal(data.columns, ['x', 'y'])

    name_generator_registry.clear()
    nwbfile = mock_NWBFile()
    nwbfile.add_acquisition(mock_SpatialSeries(data=np.zeros((4,3))))
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "SpatialSeries" in nwb.keys() or "TimeSeries" in nwb.keys()
    data = nwb[list(nwb.keys())[0]]
    assert isinstance(data, nap.TsdFrame)
    np.testing.assert_array_equal(data.values,np.zeros((4, 3)))
    np.testing.assert_array_equal(data.columns, ['x', 'y', 'z'])


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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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
            data.index, obj.starting_time + np.arange(obj.num_samples) / obj.rate
        )
        np.testing.assert_array_almost_equal(data.columns.values, obj.electrodes["id"][:])

        # Try ElectrialSeries without channel mapping
        name_generator_registry.clear()
        nwbfile = mock_NWBFile()
        nwbfile.add_acquisition(mock_ElectricalSeries(electrodes=None))
        nwb = nap.NWBFile(nwbfile)
        assert len(nwb) == 1
        assert "ElectricalSeries" in nwb.keys()
        data = nwb["ElectricalSeries"]
        assert isinstance(data, nap.TsdFrame)
        obj = nwbfile.acquisition["ElectricalSeries"]
        np.testing.assert_array_almost_equal(data.values, obj.data[:])
        np.testing.assert_array_almost_equal(
            data.index, obj.starting_time + np.arange(obj.num_samples) / obj.rate
        )
        np.testing.assert_array_almost_equal(data.columns.values, np.arange(obj.data.shape[1]))


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
        np.testing.assert_array_almost_equal(data.index, obj.timestamps[:])
        np.testing.assert_array_almost_equal(data.columns.values, obj.electrodes["id"][:])


def test_add_Icephys():
    try:
        from pynwb.testing.mock.icephys import (
            mock_IntracellularElectrode,
            mock_VoltageClampStimulusSeries,
            mock_VoltageClampSeries,
            mock_CurrentClampSeries,
            mock_CurrentClampStimulusSeries,
            mock_IZeroClampSeries,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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
                    data.index,
                    obj.starting_time + np.arange(obj.num_samples) / obj.rate,
                )
    except:
        # Doesn't work for all versions.
        pass


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
        data.index, obj.starting_time + np.arange(obj.num_samples) / obj.rate
    )


def test_add_Ophys():
    try:
        from pynwb.testing.mock.ophys import (
            mock_ImagingPlane,
            mock_OnePhotonSeries,
            mock_TwoPhotonSeries,
            mock_PlaneSegmentation,
            mock_ImageSegmentation,
            mock_RoiResponseSeries,
            mock_DfOverF,
            mock_Fluorescence,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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
                mock_Fluorescence,
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
                    data.index,
                    obj.starting_time + np.arange(obj.num_samples) / obj.rate,
                )
                np.testing.assert_array_almost_equal(data.columns.values, obj.rois["id"][:])

    except:
        pass  # some issues with pynwb version


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
    np.testing.assert_array_almost_equal(
        data.values, np.array([[1.0, 5.0], [6.0, 10.0]])
    )

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
    np.testing.assert_array_almost_equal(data[True].values, np.array([[1.0, 5.0]]))
    np.testing.assert_array_almost_equal(data[False].values, np.array([[6.0, 10.0]]))

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
    nwbfile.add_trial(start_time=6.0, stop_time=10.0, correct=False, label=1)

    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "trials" in nwb.keys()
    obj = nwbfile.trials
    with pytest.warns(UserWarning) as record:
        data = nwb["trials"]
    assert (
        record[0].message.args[0]
        == "Too many metadata. Returning pandas.DataFrame, not IntervalSet"
    )
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
    np.testing.assert_array_almost_equal(
        data["first-example"].values, np.array([[2.0, 4.0]])
    )
    np.testing.assert_array_almost_equal(
        data["second-example"].values, np.array([[6.0, 8.0]])
    )


def test_add_Units():
    nwbfile = mock_NWBFile()
    nwbfile.add_unit_column(name="quality", description="sorting quality")
    nwbfile.add_unit_column(name="alpha", description="sorting quality")

    firing_rate = 20
    n_units = 10
    res = 1000
    duration = 20
    spks = {}
    alpha = np.random.randint(0, 10, n_units)
    for n_units_per_shank in range(n_units):
        spike_times = np.where(np.random.rand((res * duration)) < (firing_rate / res))[0] / res
        nwbfile.add_unit(spike_times=spike_times, quality="good", alpha=alpha[n_units_per_shank])
        spks[n_units_per_shank] = spike_times

    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "units" in nwb.keys()
    
    data = nwb['units']    
    assert isinstance(data, nap.TsGroup)
    assert len(data) == n_units
    for n in data.keys():
        np.testing.assert_array_almost_equal(data[n].index, spks[n])

    np.testing.assert_array_equal(data._metadata["quality"].values, np.array(["good"]*n_units))
    np.testing.assert_array_equal(data._metadata["alpha"].values, alpha)


def test_add_Timestamps():
    from pynwb.misc import AnnotationSeries
    from pynwb.core import DynamicTable, VectorData

    nwbfile = mock_NWBFile()
    nwbfile.add_acquisition(AnnotationSeries("test_ts", data=np.array(["test"]*100), timestamps=np.arange(100)))
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "test_ts" in nwb.keys()
    data = nwb['test_ts']
    assert isinstance(data, nap.Ts)
    assert len(data) == 100
    np.testing.assert_array_almost_equal(data.index, np.arange(100))

    # One ts only 
    nwbfile = mock_NWBFile()    
    test_ts = DynamicTable(
        name="test_ts",
        description="Test Timestamps",
        colnames=[
            "ts_times",
            ],
        columns=[
            VectorData(
                name="ts_times",
                data=np.arange(10),
                description="Test",
                ),
        ],
        )
    nwbfile.add_acquisition(test_ts)
    
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "test_ts" in nwb.keys()
    data = nwb['test_ts']
    assert isinstance(data, nap.Ts)
    assert len(data) == 10
    np.testing.assert_array_almost_equal(data.index, np.arange(10))

    # Multiple ts
    nwbfile = mock_NWBFile()
    test_ts = DynamicTable(
        name="test_ts",
        description="Test Timestamps",
        colnames=[
            "ts_times",
            "ts2_times"
            ],
        columns=[
            VectorData(
                name="ts_times",
                data=np.arange(10),
                description="Test",
                ),
            VectorData(
                name="ts2_times",
                data=np.arange(10)+1,
                description="Test",
                ),
            ],
        )
    nwbfile.add_acquisition(test_ts)
    
    nwb = nap.NWBFile(nwbfile)
    assert len(nwb) == 1
    assert "test_ts" in nwb.keys()
    data = nwb['test_ts']
    assert isinstance(data, dict)
    assert len(data) == 2
    for i, k in enumerate(data.keys()):
        np.testing.assert_array_almost_equal(data[k].index, np.arange(10)+i)

