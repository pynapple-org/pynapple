#!/usr/bin/env python

"""
Various io functions

@author: Guillaume Viejo
"""
import os
from .neurosuite import NeuroSuite
from .phy import Phy
from .loader import BaseLoader
from xml.dom import minidom
import numpy as np
from .. import core as nap

def load_session(path=None, session_type=None):
    """
    General Loader for Neurosuite, Phy or default session.

    Parameters
    ----------
    path : str, optional
        The path to load the data
    session_type : str, optional
        Can be 'neurosuite', 'phy' or None for default loader.

    Returns
    -------
    Session
        A class holding all the data from the session.

    """
    if path:
        if not os.path.isdir(path):
            raise RuntimeError("Path {} is not found.".format(path))

    if session_type == 'neurosuite':
        return NeuroSuite(path)
    elif session_type == 'phy':
      return Phy(path)
    else:
        return BaseLoader(path)

def load_eeg(filepath, channel=None, n_channels=None, frequency=None, precision='int16', bytes_size=2):
    """
    Standalone function to load eeg/lfp/dat file in binary format.
    
    Parameters
    ----------
    filepath : str
        The path to the eeg file
    channel : int or list of int, optional
        The channel(s) to load. If None return a memory map of the dat file to avoid memory error
    n_channels : int, optional
        Number of channels 
    frequency : float, optional
        Sampling rate of the file
    precision : str, optional
        The precision of the binary file
    bytes_size : int, optional
        Bytes size of the binary file
    
    Raises
    ------
    RuntimeError
        If can't find the lfp/eeg/dat file
    
    Returns
    -------
    Tsd or TsdFrame
        The lfp in a time series format
    
    Deleted Parameters
    ------------------
    extension : str, optional
        The file extenstion (.eeg, .dat, .lfp). Make sure the frequency match
    
    """
    # Need to check if a xml file exists
    path = os.path.dirname(filepath)
    basename = os.path.basename(filepath).split('.')[0]
    listdir = os.listdir(path)

    if frequency is None or n_channels is None:    
        if basename + '.xml' in listdir:
            xmlpath     = os.path.join(path, basename+'.xml')
            xmldoc      = minidom.parse(xmlpath)
        else:
            raise RuntimeError("Can't find xml file; please specify sampling frequency or number of channels")

        if frequency is None:
            if filepath.endswith('.dat'):
                fs_dat      = int(xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data)
                frequency  = fs_dat
            elif filepath.endswith(('.lfp', '.eeg')):
                fs_eeg      = int(xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data)
                frequency = fs_eeg

        if n_channels is None:
            n_channels   = int(xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data)

    f = open(filepath, 'rb')
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2      
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    duration = n_samples/frequency
    interval = 1/frequency
    f.close()
    fp = np.memmap(filepath, np.int16, 'r', shape = (n_samples, n_channels))        
    timestep = np.arange(0, n_samples)/frequency

    time_support = nap.IntervalSet(start = 0, end = duration, time_units = 's')

    if channel is None:
        return fp
    elif type(channel) is int:
        return nap.Tsd(
            t = timestep, 
            d=fp[:,channel], 
            time_units = 's',
            time_support = time_support)
    elif type(channel) is list:            
        return nap.TsdFrame(
            t = timestep,
            d=fp[:,channel], 
            time_units = 's',
            time_support = time_support,
            columns=channel)
