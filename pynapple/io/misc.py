#!/usr/bin/env python

"""
Various io functions

@author: Guillaume Viejo
"""
import os
from .neurosuite import NeuroSuite
from .phy import Phy
from .loader import BaseLoader
from .cnmfe import InscopixCNMFE, Minian, CNMF_E
from xml.dom import minidom
import numpy as np
from .. import core as nap
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, LFP

def load_session(path=None, session_type=None):
    """
    General Loader for
        
    - Neurosuite\n
    - Phy\n
    - Minian\n
    - Inscopix-cnmfe\n
    - Matlab-cnmfe\n
    - None for default session.

    Parameters
    ----------
    path : str, optional
        The path to load the data
    session_type : str, optional
        Can be 'neurosuite', 'phy',
        'minian', 'inscopix-cnmfe', 'cnmfe-matlab',
         or None for default loader.

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

    elif session_type == 'inscopix-cnmfe':
        return InscopixCNMFE(path)

    elif session_type == 'minian':
        return Minian(path)

    elif session_type == 'cnmfe-matlab':
        return CNMF_E(path)

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

def append_NWB_LFP(path, lfp, channel=None):
    """Standalone function for adding lfp/eeg to already existing nwb files. 
    
    Parameters
    ----------
    path : str
        The path to the data. The function will looks for a nwb file in path
        or in path/pynapplenwb.
    lfp : Tsd or TsdFrame
        Description
    channel : None, optional
        channel number in int ff lfp is a Tsd
            
    Raises
    ------
    RuntimeError
        If can't find the nwb file \n
        If no channel is specify when passing a Tsd
    
    """
    new_path = os.path.join(path, 'pynapplenwb')
    nwb_path = ''
    if os.path.exists(new_path):                
        nwbfilename = [f for f in os.listdir(new_path) if f.endswith('.nwb')]
        if len(nwbfilename):
            nwb_path = os.path.join(path, 'pynapplenwb', nwbfilename[0])
    else:
        nwbfilename = [f for f in os.listdir(path) if f.endswith('.nwb')]
        if len(nwbfilename):
            nwb_path = os.path.join(path, 'pynapplenwb', nwbfilename[0])

    if len(nwb_path) == 0:
        raise RuntimeError("Can't find nwb file in {}".format(path))

    if isinstance(lfp, nap.TsdFrame):
        channels = lfp.columns.values
    elif isinstance(lfp, nap.Tsd):
        if isinstance(channel, int):
            channels = [channel]
        else:
            raise RuntimeError("Please specify which channel it is.")

    io = NWBHDF5IO(nwb_path, 'r+')
    nwbfile = io.read()


    all_table_region = nwbfile.create_electrode_table_region(
        region=channels,
        description='',
        name='electrodes')

    lfp_electrical_series = ElectricalSeries(
        name='ElectricalSeries',
        data=lfp.values,
        timestamps=lfp.index.values,
        electrodes=all_table_region,        
        )

    lfp = LFP(electrical_series=lfp_electrical_series)

    ecephys_module = nwbfile.create_processing_module(
        name='ecephys',
        description='processed extracellular electrophysiology data'
    )
    ecephys_module.add(lfp)

    io.write(nwbfile)
    io.close()

    return
