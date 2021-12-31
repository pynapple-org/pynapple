#!/usr/bin/env python

"""
Class and functions for loading data processed with the Neurosuite (Klusters, Neuroscope, NDmanager)

@author: Guillaume Viejo
"""
import os, sys
import numpy as np
from .. import core as nap
from .loader_gui import BaseLoaderGUI

from PyQt5.QtWidgets import QApplication

from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.behavior import Position, SpatialSeries, CompassDirection
import datetime

import pandas as pd
import scipy.signal


class BaseLoader(object):
    """
    General loader for epochs and tracking data
    """
    def __init__(self, path=None):
        self.data = None
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
            self.window = BaseLoaderGUI(path=path)
            app.exec()

            # Extracting all the informations from gui loader
            if self.window.status:
                self.session_information = self.window.session_information
                self.name = self.session_information['name']
                self.tracking_frequency = self.window.tracking_frequency
                self.position = self._make_position(
                    self.window.tracking_parameters,
                    self.window.tracking_method,
                    self.window.tracking_frequency,
                    self.window.epochs,
                    self.window.time_units_epochs
                    )                
                self.epochs = self._make_epochs(
                    self.window.epochs, 
                    self.window.time_units_epochs
                    )

                self.time_support = self._join_epochs(
                    self.window.epochs,
                    self.window.time_units_epochs
                    )

            # Save the data
            self.create_nwb_file(path)

    def load_optitrack_csv(self, csv_file):
        """Summary
        
        Parameters
        ----------
        csv_file : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Raises
        ------
        RuntimeError
            Description
        """
        position = pd.read_csv(csv_file, header = [4,5], index_col = 1)
        if 1 in position.columns:
            position = position.drop(labels = 1, axis = 1)
        position = position[~position.index.duplicated(keep='first')]
        order = []
        for n in position.columns:
            if n[0] == 'Rotation':
                order.append('r'+n[1].lower())
            elif n[0] == 'Position':
                order.append(n[1].lower())
            else:
                raise RuntimeError('Unknow tracking format for csv file {}'.format(csv_file))
        position.columns = order
        return position

    def load_dlc_csv(self, csv_file):
        print("TODO")
        return

    def load_ttl_pulse(self, ttl_file, n_channels=1, channel=0, bytes_size=2, fs=20000.0):
        """Summary
        
        Parameters
        ----------
        ttl_file : TYPE
            Description
        n_channels : int, optional
            Description
        channel : int, optional
            Description
        bytes_size : int, optional
            Description
        fs : float, optional
            Description
        
        Returns
        -------
        TYPE
            Description
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
        peaks,_ = scipy.signal.find_peaks(np.diff(data), height=0.5)
        timestep = np.arange(0, len(data))/fs
        # analogin = pd.Series(index = timestep, data = data)
        peaks+=1
        ttl = pd.Series(index = timestep[peaks], data = data[peaks])    
        return ttl

    def _make_position(self, parameters, method, frequency, epochs, time_units):
        """
        Make the position TSDFrame with the parameters extracted from the GUI.
        """
        frames = []

        for i, f in enumerate(parameters.index):

            if method.lower() == 'optitrack':
                position = self.load_optitrack_csv(parameters.loc[f,'csv'])
            elif method.lower() == 'deeplabcut':
                position = self.load_dlc_csv(parameters.loc[f,'csv'])

            ttl = self.load_ttl_pulse(
                ttl_file=parameters.loc[f,'ttl'],
                n_channels=parameters.loc[f,'n_channels'],
                channel=parameters.loc[f,'tracking_channel'],
                bytes_size=parameters.loc[f,'bytes_size'],
                fs=parameters.loc[f,'fs']
                )
            
            if len(ttl):
                length = np.minimum(len(ttl), len(position))
                ttl = ttl.iloc[0:length]
                position = position.iloc[0:length]
            else:
                raise RuntimeError("No ttl detected for {}".format(f))

            start_epoch = nap.TimeUnits.format_timestamps(epochs.loc[parameters.loc[f,'epoch'],'start'], time_units)
            time_offset = nap.TimeUnits.return_timestamps(start_epoch,'s')[0] + ttl.index[0]

            position.index += time_offset
            # wake_ep.iloc[i,0] = np.int64(np.maximum(wake_ep.as_units('s').iloc[i,0], position.index[0])*1e6)
            # wake_ep.iloc[i,1] = np.int64(np.minimum(wake_ep.as_units('s').iloc[i,1], position.index[-1])*1e6)

            if len(position.columns) > 6:
                frames.append(position.iloc[:,0:6])
                others.append(position.iloc[:,6:])
            else:
                frames.append(position)

        position = pd.concat(frames)
        position[['ry', 'rx', 'rz']] *= (np.pi/180)
        position[['ry', 'rx', 'rz']] += 2*np.pi
        position[['ry', 'rx', 'rz']] %= 2*np.pi

        position = nap.TsdFrame(
            t = position.index.values, 
            d = position.values,
            columns = position.columns.values,
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
        isets = nap.IntervalSet(start=epochs['start'], end=epochs['end'],time_units=time_units)
        iset = isets.merge_close_intervals(0.0) 
        return iset

    def create_nwb_file(self, path):
        """Summary
        
        Parameters
        ----------
        path : str
            The path to save the data
        
        Returns
        -------
        TYPE
            Description
        """
        self.nwb_path = os.path.join(path, 'pynapplenwb')
        if not os.path.exists(self.nwb_path):
            os.makedirs(self.nwb_path)
        self.nwbfilepath = os.path.join(self.nwb_path, self.session_information['name']+'.nwb')
        nwbfile = NWBFile(
            session_description=self.session_information['description'],
            identifier=self.session_information['name'],
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
            experimenter=self.session_information['experimenter'],
            lab=self.session_information['lab']
        )

        # Tracking
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
        """Summary
        
        Parameters
        ----------
        path : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
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
        position = pd.DataFrame.from_dict(position)
        self.position = nap.TsdFrame(position, time_units = 's')

        epochs = nwbfile.epochs.to_dataframe()
        # NWB is dumb and cannot take a single string for labels
        epochs['label'] = [epochs.loc[i,'tags'][0] for i in epochs.index]
        epochs = epochs.drop(labels='tags', axis=1)
        epochs = epochs.rename(columns={'start_time':'start','stop_time':'end'})
        self.epochs = self._make_epochs(epochs)

        self.time_support = self._join_epochs(
            epochs,
            's'
            )


        io.close()

        return
