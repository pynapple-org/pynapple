# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 23:30:51
# @Last Modified by:   gviejo
# @Last Modified time: 2023-11-16 13:17:59

"""
BaseLoader is the general class for loading session with pynapple.

@author: Guillaume Viejo
"""
import os
import warnings

import pandas as pd
from pynwb import NWBHDF5IO, TimeSeries
from pynwb.epoch import TimeIntervals

from .. import core as nap

warnings.simplefilter("always", DeprecationWarning)


def get_error_text(path):
    try:
        width, _ = os.get_terminal_size()
    except Exception:
        width = 50
    border = "".join(["@"] * width)

    txt1 = """
    No NWB file were found in {}. 

    The Graphical User Interface (GUI) helper to create a NWB file has been removed from pynapple. The GUI can still be found in the nwbmatic package (https://github.com/pynapple-org/nwbmatic). To install it :

        $ pip install nwbmatic
    
    The equivalent code is (for example for data processed with "phy")

        >>> import nwbmatic as ntm
        >>> data = ntm.load_session("path/to/my/session", "phy")

    A more advanced project for creating NWB files is neuroconv:
    https://neuroconv.readthedocs.io/en/main/
    """.format(
        path
    )

    error_txt = "\n" + border + "\n" + txt1 + "\n" + border
    return error_txt


class BaseLoader(object):
    """
    General loader for epochs and tracking data
    """

    def __init__(self, path=None):
        self.path = path

        file_found = False
        # Check if a pynapplenwb folder exist
        if self.path is not None:
            nwb_path = os.path.join(self.path, "pynapplenwb")
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith(".nwb")]):
                    file_found = True
                    self.load_data(path)

        # Starting the GUI
        if not file_found:
            raise RuntimeError(get_error_text(path))

    def load_data(self, path):
        """
        Load NWB data saved with pynapple in the pynapplenwb folder

        Parameters
        ----------
        path : str
            Path to the session folder
        """
        self.nwb_path = os.path.join(path, "pynapplenwb")
        if not os.path.exists(self.nwb_path):
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if "nwb" in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, "r+")
        nwbfile = io.read()

        position = {}
        acq_keys = nwbfile.acquisition.keys()
        if "CompassDirection" in acq_keys:
            compass = nwbfile.acquisition["CompassDirection"]
            for k in compass.spatial_series.keys():
                position[k] = pd.Series(
                    index=compass.get_spatial_series(k).timestamps[:],
                    data=compass.get_spatial_series(k).data[:],
                )
        if "Position" in acq_keys:
            tracking = nwbfile.acquisition["Position"]
            for k in tracking.spatial_series.keys():
                position[k] = pd.Series(
                    index=tracking.get_spatial_series(k).timestamps[:],
                    data=tracking.get_spatial_series(k).data[:],
                )
        if len(position):
            position = pd.DataFrame.from_dict(position)

            # retrieveing time support position if in epochs
            if "position_time_support" in nwbfile.intervals.keys():
                epochs = nwbfile.intervals["position_time_support"].to_dataframe()
                time_support = nap.IntervalSet(
                    start=epochs["start_time"], end=epochs["stop_time"], time_units="s"
                )

            self.position = nap.TsdFrame(
                position, time_units="s", time_support=time_support
            )

        if nwbfile.epochs is not None:
            epochs = nwbfile.epochs.to_dataframe()
            # NWB is dumb and cannot take a single string for labels
            epochs["label"] = [epochs.loc[i, "tags"][0] for i in epochs.index]
            epochs = epochs.drop(labels="tags", axis=1)
            epochs = epochs.rename(columns={"start_time": "start", "stop_time": "end"})
            self.epochs = self._make_epochs(epochs)

            self.time_support = self._join_epochs(epochs, "s")

        io.close()

        return

    def _make_epochs(self, epochs, time_units="s"):
        """
        Split GUI epochs into dict of epochs
        """
        labels = epochs.groupby("label").groups
        isets = {}
        for lbs in labels.keys():
            tmp = epochs.loc[labels[lbs]]
            isets[lbs] = nap.IntervalSet(
                start=tmp["start"], end=tmp["end"], time_units=time_units
            )
        return isets

    def _join_epochs(self, epochs, time_units="s"):
        """
        To create the global time support of the data
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            isets = nap.IntervalSet(
                start=epochs["start"].sort_values(),
                end=epochs["end"].sort_values(),
                time_units=time_units,
            )
            iset = isets.merge_close_intervals(1, time_units="us")
        if len(iset):
            return iset
        else:
            return None

    def save_nwb_intervals(self, iset, name, description=""):
        """
        Add epochs to the NWB file (e.g. ripples epochs)
        See pynwb.epoch.TimeIntervals

        Parameters
        ----------
        iset : IntervalSet
            The intervalSet to save
        name : str
            The name in the nwb file
        """
        io = NWBHDF5IO(self.nwbfilepath, "r+")
        nwbfile = io.read()

        epochs = iset.as_units("s")
        time_intervals = TimeIntervals(name=name, description=description)
        for i in epochs.index:
            time_intervals.add_interval(
                start_time=epochs.loc[i, "start"],
                stop_time=epochs.loc[i, "end"],
                tags=str(i),
            )

        nwbfile.add_time_intervals(time_intervals)
        io.write(nwbfile)
        io.close()

        return

    def save_nwb_timeseries(self, tsd, name, description=""):
        """
        Save timestamps in the NWB file (e.g. ripples time) with the time support.
        See pynwb.base.TimeSeries


        Parameters
        ----------
        tsd : TsdFrame
            _
        name : str
            _
        description : str, optional
            _
        """
        io = NWBHDF5IO(self.nwbfilepath, "r+")
        nwbfile = io.read()

        ts = TimeSeries(
            name=name,
            unit="s",
            data=tsd.values,
            timestamps=tsd.as_units("s").index.values,
        )

        time_support = TimeIntervals(
            name=name + "_timesupport", description="The time support of the object"
        )

        epochs = tsd.time_support.as_units("s")
        for i in epochs.index:
            time_support.add_interval(
                start_time=epochs.loc[i, "start"],
                stop_time=epochs.loc[i, "end"],
                tags=str(i),
            )
        nwbfile.add_time_intervals(time_support)
        nwbfile.add_acquisition(ts)
        io.write(nwbfile)
        io.close()

        return

    def load_nwb_intervals(self, name):
        """
        Load epochs from the NWB file (e.g. 'ripples')

        Parameters
        ----------
        name : str
            The name in the nwb file
        """
        io = NWBHDF5IO(self.nwbfilepath, "r")
        nwbfile = io.read()

        if name in nwbfile.intervals.keys():
            epochs = nwbfile.intervals[name].to_dataframe()
            isets = nap.IntervalSet(
                start=epochs["start_time"], end=epochs["stop_time"], time_units="s"
            )
            io.close()
            return isets
        else:
            io.close()
        return

    def load_nwb_timeseries(self, name):
        """
        Load timestamps in the NWB file (e.g. ripples time)

        Parameters
        ----------
        name : str
            _

        Returns
        -------
        Tsd
            _
        """
        io = NWBHDF5IO(self.nwbfilepath, "r")
        nwbfile = io.read()

        ts = nwbfile.acquisition[name]

        time_support = self.load_nwb_intervals(name + "_timesupport")

        tsd = nap.Tsd(
            t=ts.timestamps[:], d=ts.data[:], time_units="s", time_support=time_support
        )

        io.close()

        return tsd
