# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 23:30:51
# @Last Modified by:   gviejo
# @Last Modified time: 2022-03-22 12:23:17

"""
BaseLoader is the general class for loading session with pynapple.

@author: Guillaume Viejo
"""
import os, sys
import numpy as np
from .. import core as nap
from .loader_gui import BaseLoaderGUI

from PyQt5.QtWidgets import QApplication

from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.behavior import Position, SpatialSeries, CompassDirection
from pynwb.file import Subject
from pynwb.epoch import TimeIntervals
import datetime
import warnings
import pandas as pd
import scipy.signal

def format_timestamp(t, time_unit):
    if time_unit == 's':
        return t
    elif time_unit == 'ms':
        return t*1000
    elif time_unit == 'us':
        return t*1000000
    else:
        raise ValueError('unrecognized time units type')

class BaseLoader(object):
    """
    General loader for epochs and tracking data
    """
    def __init__(self, path=None):
        self.path = path

        start_gui = True

        # Check if a pynapplenwb folder exist to bypass GUI
        if self.path is not None:
            nwb_path = os.path.join(self.path, 'pynapplenwb')
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith('.nwb')]):
                    start_gui = False
                    self.load_data(path)

        # Starting the GUI
        if start_gui:
            app = QApplication([])
            app.setQuitOnLastWindowClosed(True)
            window = BaseLoaderGUI(path=path)
            window.show()

            app.exec()

            # Extracting all the informations from gui loader
            if window.status:
                self.session_information = window.session_information
                self.subject_information = window.subject_information
                self.name = self.session_information['name']
                self.tracking_frequency = window.tracking_frequency
                self.position = self._make_position(
                    window.tracking_parameters,
                    window.tracking_method,
                    window.tracking_frequency,
                    window.epochs,
                    window.time_units_epochs,
                    window.tracking_alignement
                    ) 
                self.epochs = self._make_epochs(
                    window.epochs, 
                    window.time_units_epochs
                    )
                self.time_support = self._join_epochs(
                    window.epochs,
                    window.time_units_epochs
                    )
                # Save the data
                self.create_nwb_file(path)            
            app.quit()
            # print('\n'.join(repr(w) for w in app.allWidgets()))
            # del app, window

    def load_default_csv(self, csv_file):
        """
        Load tracking data. The default csv should have the time index in the first column in seconds.
        If no header is provided, the column names will be the column index.
        
        Parameters
        ----------
        csv_file : str
            path to the csv file
        
        Returns
        -------
        pandas.DataFrame
            _
        """
        position = pd.read_csv(csv_file, header = [0], index_col = 0)
        position = position[~position.index.duplicated(keep='first')]        
        return position

    def load_optitrack_csv(self, csv_file):
        """
        Load tracking data exported with Optitrack.
        By default, the function reads rows 4 and 5 to build the column names.
        
        Parameters
        ----------
        csv_file : str
            path to the csv file
        
        Raises
        ------
        RuntimeError
            If header names are unknown. Should be 'Position' and 'Rotation'
        
        Returns
        -------
        pandas.DataFrame
            _
        """
        position = pd.read_csv(csv_file, header = [4,5], index_col = 1)
        if 1 in position.columns:
            position = position.drop(labels = 1, axis = 1)
        position = position[~position.index.duplicated(keep='first')]
        order = []
        cols = []
        for n in position.columns:
            if n[0] == 'Rotation':
                order.append('r'+n[1].lower())
                cols.append(n)
            elif n[0] == 'Position':
                order.append(n[1].lower())
                cols.append(n)
        if len(order) == 0:
                raise RuntimeError('Unknow tracking format for csv file {}'.format(csv_file))
        position = position[cols]
        position.columns = order
        return position

    def load_dlc_csv(self, csv_file):
        """
        Load tracking data exported with DeepLabCut
        
        Parameters
        ----------
        csv_file : str
            path to the csv file
        
        Returns
        -------
        pandas.DataFrame
            _
        """
        position = pd.read_csv(csv_file, header = [1,2], index_col = 0)
        position = position[~position.index.duplicated(keep='first')]
        position.columns = list(map(lambda x:"_".join(x), position.columns.values))
        return position

    def load_ttl_pulse(self, ttl_file, tracking_frequency, n_channels=1, channel=0, bytes_size=2, fs=20000.0, threshold=0.3,):
        """
        Load TTLs from a binary file. Each TTLs is then used to reaassign the time index of tracking frames.
        
        Parameters
        ----------
        ttl_file : str
            File name
        n_channels : int, optional
            The number of channels in the binary file.
        channel : int, optional
            Which channel contains the TTL
        bytes_size : int, optional
            Bytes size of the binary file.
        fs : float, optional
            Sampling frequency of the binary file
        
        Returns
        -------
        pd.Series
            A series containing the time index of the TTL.
        """
        f = open(ttl_file, 'rb')
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
        f.close()
        with open(ttl_file, 'rb') as f:
            data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
        if n_channels == 1:
            data = data.flatten().astype(np.int32)
        else:
            data = data[:,channel].flatten().astype(np.int32)
        data = data/data.max()
        peaks,_ = scipy.signal.find_peaks(np.diff(data), height=threshold, distance=int(fs/(tracking_frequency*2)))
        timestep = np.arange(0, len(data))/fs
        analogin = pd.Series(index = timestep, data = data)
        peaks+=1
        ttl = pd.Series(index = timestep[peaks], data = data[peaks])
        return ttl

    def _make_position(self, parameters, method, frequency, epochs, time_units, alignement):
        """
        Make the position TSDFrame with the parameters extracted from the GUI.
        """
        if len(parameters.index) == 0:
            return None
        else:
            if len(epochs) == 0:
                epochs.loc[0, 'start'] = 0.0
            frames = []
            time_supports_starts = []
            time_support_ends = []

            for i, f in enumerate(parameters.index):

                if method.lower() == 'optitrack':
                    position = self.load_optitrack_csv(parameters.loc[f,'csv'])
                elif method.lower() == 'deeplabcut':
                    position = self.load_dlc_csv(parameters.loc[f,'csv'])
                elif method.lower() == 'default':
                    position = self.load_default_csv(parameters.loc[f,'csv'])

                if alignement.lower() == 'local':
                    start_epoch = nap.TimeUnits.format_timestamps(epochs.loc[parameters.loc[f,'epoch'],'start'], time_units)
                    timestamps = position.index.values + nap.TimeUnits.return_timestamps(start_epoch,'s')[0]
                    position.index = pd.Index(timestamps)

                if alignement.lower() == 'ttl':
                    ttl = self.load_ttl_pulse(
                        ttl_file=parameters.loc[f,'ttl'],
                        tracking_frequency=frequency,
                        n_channels=parameters.loc[f,'n_channels'],
                        channel=parameters.loc[f,'tracking_channel'],
                        bytes_size=parameters.loc[f,'bytes_size'],
                        fs=parameters.loc[f,'fs'],
                        threshold=parameters.loc[f,'threshold'],
                        )
                    
                    if len(ttl):
                        length = np.minimum(len(ttl), len(position))
                        ttl = ttl.iloc[0:length]
                        position = position.iloc[0:length]
                    else:
                        raise RuntimeError("No ttl detected for {}".format(f))

                    # Make sure start epochis in seconds
                    start_epoch = format_timestamp(epochs.loc[parameters.loc[f,'epoch'],'start'], time_units)
                    timestamps = start_epoch + ttl.index.values
                    position.index = pd.Index(timestamps)

                frames.append(position)
                time_supports_starts.append(position.index[0])
                time_support_ends.append(position.index[-1])

            position = pd.concat(frames)

            time_supports = nap.IntervalSet(start = time_supports_starts,
                                            end = time_support_ends,
                                            time_units = 's')

            # Specific to optitrACK
            if set(['rx', 'ry', 'rz']).issubset(position.columns):
                position[['ry', 'rx', 'rz']] *= (np.pi/180)
                position[['ry', 'rx', 'rz']] += 2*np.pi
                position[['ry', 'rx', 'rz']] %= 2*np.pi

            position = nap.TsdFrame(
                t = position.index.values, 
                d = position.values,
                columns = position.columns.values,
                time_support = time_supports,
                time_units = 's')

            return position

    def _make_epochs(self, epochs, time_units='s'):
        """
        Split GUI epochs into dict of epochs
        """
        labels = epochs.groupby("label").groups
        isets = {}
        for l in labels.keys():
            tmp = epochs.loc[labels[l]]
            isets[l] = nap.IntervalSet(start=tmp['start'],end=tmp['end'],time_units=time_units)
        return isets

    def _join_epochs(self, epochs, time_units='s'):
        """
        To create the global time support of the data
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            isets = nap.IntervalSet(start=epochs['start'].sort_values(), end=epochs['end'].sort_values(),time_units=time_units)
            iset = isets.merge_close_intervals(1, time_units = 'us')
        if len(iset):
            return iset
        else:
            return None

    def create_nwb_file(self, path):
        """
        Initialize the NWB file in the folder pynapplenwb within the data folder.        
        
        Parameters
        ----------
        path : str
            The path to save the data
        
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if not os.path.exists(self.nwb_path):
            os.makedirs(self.nwb_path)
        self.nwbfilepath = os.path.join(self.nwb_path, self.session_information['name']+'.nwb')
        
        self.subject_information['date_of_birth'] = None

        nwbfile = NWBFile(
            session_description=self.session_information['description'],
            identifier=self.session_information['name'],
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
            experimenter=self.session_information['experimenter'],
            lab=self.session_information['lab'],
            institution=self.session_information['institution'],
            subject = Subject(**self.subject_information)
        )

        # Tracking
        if self.position is not None:
            data = self.position.as_units('s')
            position = Position()        
            for c in ['x', 'y', 'z']:
                tmp = SpatialSeries(
                    name=c, 
                    data=data[c].values, 
                    timestamps=data.index.values, 
                    unit='',
                    reference_frame='')
                position.add_spatial_series(tmp)
            direction = CompassDirection()
            for c in ['rx', 'ry', 'rz']:
                tmp = SpatialSeries(
                    name=c, 
                    data=data[c].values, 
                    timestamps=data.index.values, 
                    unit='radian',
                    reference_frame='')
                direction.add_spatial_series(tmp)

            nwbfile.add_acquisition(position)
            nwbfile.add_acquisition(direction)

            # Adding time support of position as TimeIntervals
            epochs = self.position.time_support.as_units('s')
            position_time_support = TimeIntervals(
                name="position_time_support",
                description="The time support of the position i.e the real start and end of the tracking"
                )
            for i in self.position.time_support.index:
                position_time_support.add_interval(
                    start_time=epochs.loc[i,'start'],
                    stop_time=epochs.loc[i,'end'],
                    tags=str(i)
                    )

            nwbfile.add_time_intervals(position_time_support)


        
        # Epochs
        for ep in self.epochs.keys():
            epochs = self.epochs[ep].as_units('s')
            for i in self.epochs[ep].index:
                nwbfile.add_epoch(
                    start_time=epochs.loc[i,'start'],
                    stop_time=epochs.loc[i,'end'],
                    tags=[ep] # This is stupid nwb who tries to parse the string
                    )


        with NWBHDF5IO(self.nwbfilepath, 'w') as io:
            io.write(nwbfile)

        return

    def load_data(self, path):
        """
        Load NWB data save with pynapple in the pynapplenwb folder
        
        Parameters
        ----------
        path : str
            Path to the session folder
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            files = os.listdir(self.nwb_path)
        else:
            raise RuntimeError("Path {} does not exist.".format(self.nwb_path))
        self.nwbfilename = [f for f in os.listdir(self.nwb_path) if 'nwb' in f][0]
        self.nwbfilepath = os.path.join(self.nwb_path, self.nwbfilename)

        io = NWBHDF5IO(self.nwbfilepath, 'r+')
        nwbfile = io.read()

        position = {}
        acq_keys = nwbfile.acquisition.keys()
        if 'CompassDirection' in acq_keys:
            compass = nwbfile.acquisition['CompassDirection']            
            for k in compass.spatial_series.keys():
                position[k] = pd.Series(
                    index = compass.get_spatial_series(k).timestamps[:],
                    data = compass.get_spatial_series(k).data[:],
                    )
        if 'Position' in acq_keys:
            tracking = nwbfile.acquisition['Position']
            for k in tracking.spatial_series.keys():
                position[k] = pd.Series(
                    index = tracking.get_spatial_series(k).timestamps[:],
                    data = tracking.get_spatial_series(k).data[:],
                    )
        if len(position):
            position = pd.DataFrame.from_dict(position)

            # retrieveing time support position if in epochs
            if 'position_time_support' in nwbfile.intervals.keys():
                epochs = nwbfile.intervals['position_time_support'].to_dataframe()
                time_support = nap.IntervalSet(
                    start = epochs['start_time'],
                    end = epochs['stop_time'],
                    time_units = 's')

            self.position = nap.TsdFrame(position, time_units = 's', time_support = time_support)

        if nwbfile.epochs is not None:
            epochs = nwbfile.epochs.to_dataframe()
            # NWB is dumb and cannot take a single string for labels
            epochs['label'] = [epochs.loc[i,'tags'][0] for i in epochs.index]
            epochs = epochs.drop(labels='tags', axis=1)
            epochs = epochs.rename(columns={'start_time':'start','stop_time':'end'})
            self.epochs = self._make_epochs(epochs)

            self.time_support = self._join_epochs(epochs,'s')


        io.close()

        return

    def save_nwb_intervals(self, iset, name, description = ''):
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
        io = NWBHDF5IO(self.nwbfilepath, 'r+')
        nwbfile = io.read()

        epochs = iset.as_units('s')
        time_intervals = TimeIntervals(
            name=name,
            description=description
            )
        for i in epochs.index:
            time_intervals.add_interval(
                start_time=epochs.loc[i,'start'],
                stop_time=epochs.loc[i,'end'],
                tags=str(i)
                )

        nwbfile.add_time_intervals(time_intervals)
        io.write(nwbfile)
        io.close()

        return

    def save_nwb_timeseries(self, tsd, name, description = ''):
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
        io = NWBHDF5IO(self.nwbfilepath, 'r+')
        nwbfile = io.read()

        ts = TimeSeries(
            name=name, 
            unit='s',
            data = tsd.values,
            timestamps=tsd.as_units('s').index.values)

        time_support = TimeIntervals(
            name=name+'_timesupport',
            description="The time support of the object"
            )
        
        epochs = tsd.time_support.as_units('s')
        for i in epochs.index:
            time_support.add_interval(
                start_time=epochs.loc[i,'start'],
                stop_time=epochs.loc[i,'end'],
                tags=str(i)
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
        io = NWBHDF5IO(self.nwbfilepath, 'r')
        nwbfile = io.read()

        if name in nwbfile.intervals.keys():

            epochs = nwbfile.intervals[name].to_dataframe()
            isets = nap.IntervalSet(
                start = epochs['start_time'],
                end = epochs['stop_time'],
                time_units = 's')
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
        tsd : TsdFrame
            _
        name : str
            _
        description : str, optional
            _
        """
        io = NWBHDF5IO(self.nwbfilepath, 'r')
        nwbfile = io.read()

        ts = nwbfile.acquisition[name]

        time_support = self.load_nwb_intervals(name+'_timesupport')

        tsd = nap.Tsd(
            t = ts.timestamps[:],
            d = ts.data[:],
            time_units = 's',
            time_support = time_support
            )

        io.close()

        return tsd

