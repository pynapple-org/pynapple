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
import datetime




class BaseLoader(object):
    """
    General loader for epochs and tracking data
    """
    def __init__(self, path=None):
        self.data = None
        self.path = path

        # Check if a pynapplenwb folder exist to bypass GUI
        self.nwb_path = os.path.join(self.path, 'pynapplenwb')
        if os.path.exists(self.nwb_path):
            self.load_data(self.path)

        else:
            # Starting the GUI
            app = QApplication([])
            self.window = BaseLoaderGUI(path=path)
            app.exec()

            # Extracting all the informations from gui loader
            if self.window.status:
                self.session_information = self.window.session_information
                self.position = self._make_position(
                    self.tracking_parameters,
                    self.tracking_method,
                    self.tracking_frequency,
                    self.window.epochs,
                    self.window.time_units_epochs
                    )                
                self.epochs = self._make_epochs(
                    self.window.epochs, 
                    self.window.time_units_epochs
                    )

            # Save the data
            # self.save_data(self.nwb_path)

    def _make_position(self, parameters, method, frequency, epochs, time_units):
        """
        Make the position TSDFrame with the parameters extracted from the GUI.
        """
        frames = []
        others = []

        for i, f in enumerate(parameters.index):
            print(i, f)
            csv_file = os.path.join(path, "".join(s for s in files if f+'.csv' in s))
            position = pd.read_csv(csv_file, header = [4,5], index_col = 1)
            if 1 in position.columns:
                position = position.drop(labels = 1, axis = 1)
            position = position[~position.index.duplicated(keep='first')]
            analogin_file = os.path.splitext(csv_file)[0]+'_analogin.dat'
            if not os.path.split(analogin_file)[1] in files:
                print("No analogin.dat file found.")
                print("Please provide it as "+os.path.split(analogin_file)[1])
                print("Exiting ...")
                sys.exit()
            else:
                ttl = loadTTLPulse(analogin_file, n_channels, trackchannel)
            
            if len(ttl):
                length = np.minimum(len(ttl), len(position))
                ttl = ttl.iloc[0:length]
                position = position.iloc[0:length]
                time_offset = wake_ep.as_units('s').iloc[i,0] + ttl.index[0]
            else:
                print("No ttl for ", i, f)
                time_offset = wake_ep.as_units('s').iloc[i,0]
            
            position.index += time_offset
            wake_ep.iloc[i,0] = np.int64(np.maximum(wake_ep.as_units('s').iloc[i,0], position.index[0])*1e6)
            wake_ep.iloc[i,1] = np.int64(np.minimum(wake_ep.as_units('s').iloc[i,1], position.index[-1])*1e6)

            if len(position.columns) > 6:
                frames.append(position.iloc[:,0:6])
                others.append(position.iloc[:,6:])
            else:
                frames.append(position)



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



    def save_data(self, path):
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
        if not os.path.exists(path):
            os.makedirs(path)
        self.nwbfilepath = os.path.join(path, self.session_information['name']+'.nwb')
        self.nwbfile = NWBFile(
            session_description='',
            identifier=self.session_information['name'],
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )

        with NWBHDF5IO(self.nwbfilepath, 'w') as io:
            io.write(self.nwbfile)


        return

    def load_data(self, path):
        return