#!/usr/bin/env python

"""
Various io functions

"""
import os
from xml.dom import minidom

import numpy as np
from pynwb import NWBHDF5IO
from pynwb.ecephys import LFP, ElectricalSeries

from .. import core as nap
from .cnmfe import CNMF_E, InscopixCNMFE, Minian
from .folder import Folder
from .interface_npz import NPZFile
from .interface_nwb import NWBFile
from .loader import BaseLoader
from .neurosuite import NeuroSuite
from .phy import Phy
from .suite2p import Suite2P


def load_file(path):
    """Load file. Current format supported is (npz,nwb,)

    .npz -> If the file is compatible with a pynapple format, the function will return a pynapple object.
    Otherwise, the function will return the output of numpy.load

    .nwb -> Return the pynapple.io.NWBFile class wrapping the NWBFile

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    (Tsd, TsdFrame, Ts, IntervalSet, TsGroup, pynapple.io.NWBFile)
        One of the 5 pynapple objects or pynapple.io.NWBFile

    Raises
    ------
    FileNotFoundError
        If file is missing
    """
    if os.path.isfile(path):
        if path.endswith(".npz"):
            return NPZFile(path).load()
        elif path.endswith(".nwb"):
            return NWBFile(path)
        else:
            raise RuntimeError("File format not supported")
    else:
        raise FileNotFoundError("File {} does not exist".format(path))


def load_folder(path):
    """Load folder containing files or other folder.
    Pynapple will walk throught the subfolders to detect compatible npz files
    or nwb files.

    Parameters
    ----------
    path : str
        Path to the folder

    Returns
    -------
    Folder
        A dictionnary-like class containing all the sub-folders and compatible files (i.e. npz, nwb)

    Raises
    ------
    RuntimeError
        If folder is missing
    """
    if os.path.isdir(path):
        return Folder(path)
    else:
        raise RuntimeError("Folder {} does not exist".format(path))


def load_session(path=None, session_type=None):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % WARNING : THIS FUNCTION IS DEPRECATED %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    General Loader for

    - Neurosuite\n
    - Phy\n
    - Minian\n
    - Inscopix-cnmfe\n
    - Matlab-cnmfe\n
    - Suite2p
    - None for default session.

    Parameters
    ----------
    path : str, optional
        The path to load the data
    session_type : str, optional
        Can be 'neurosuite', 'phy',
        'minian', 'inscopix-cnmfe', 'cnmfe-matlab',
        'suite2p' or None for default loader.

    Returns
    -------
    Session
        A class holding all the data from the session.

    """
    if path:
        if not os.path.isdir(path):
            raise RuntimeError("Path {} is not found.".format(path))

    if isinstance(session_type, str):
        session_type = session_type.lower()

    if session_type == "neurosuite":
        return NeuroSuite(path)

    elif session_type == "phy":
        return Phy(path)

    elif session_type == "inscopix-cnmfe":
        return InscopixCNMFE(path)

    elif session_type == "minian":
        return Minian(path)

    elif session_type == "cnmfe-matlab":
        return CNMF_E(path)

    elif session_type == "suite2p":
        return Suite2P(path)

    else:
        return BaseLoader(path)


def load_eeg(
    filepath,
    channel=None,
    n_channels=None,
    frequency=None,
    precision="int16",
    bytes_size=2,
):
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
    basename = os.path.basename(filepath).split(".")[0]
    listdir = os.listdir(path)

    if frequency is None or n_channels is None:
        if basename + ".xml" in listdir:
            xmlpath = os.path.join(path, basename + ".xml")
            xmldoc = minidom.parse(xmlpath)
        else:
            raise RuntimeError(
                "Can't find xml file; please specify sampling frequency or number of channels"
            )

        if frequency is None:
            if filepath.endswith(".dat"):
                fs_dat = int(
                    xmldoc.getElementsByTagName("acquisitionSystem")[0]
                    .getElementsByTagName("samplingRate")[0]
                    .firstChild.data
                )
                frequency = fs_dat
            elif filepath.endswith((".lfp", ".eeg")):
                fs_eeg = int(
                    xmldoc.getElementsByTagName("fieldPotentials")[0]
                    .getElementsByTagName("lfpSamplingRate")[0]
                    .firstChild.data
                )
                frequency = fs_eeg

        if n_channels is None:
            n_channels = int(
                xmldoc.getElementsByTagName("acquisitionSystem")[0]
                .getElementsByTagName("nChannels")[0]
                .firstChild.data
            )

    f = open(filepath, "rb")
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2
    n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
    duration = n_samples / frequency
    f.close()
    fp = np.memmap(filepath, np.int16, "r", shape=(n_samples, n_channels))
    timestep = np.arange(0, n_samples) / frequency

    time_support = nap.IntervalSet(start=0, end=duration, time_units="s")

    if channel is None:
        return fp
    elif type(channel) is int:
        return nap.Tsd(
            t=timestep, d=fp[:, channel], time_units="s", time_support=time_support
        )
    elif type(channel) is list:
        return nap.TsdFrame(
            t=timestep,
            d=fp[:, channel],
            time_units="s",
            time_support=time_support,
            columns=channel,
        )


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
    new_path = os.path.join(path, "pynapplenwb")
    nwb_path = ""
    if os.path.exists(new_path):
        nwbfilename = [f for f in os.listdir(new_path) if f.endswith(".nwb")]
        if len(nwbfilename):
            nwb_path = os.path.join(path, "pynapplenwb", nwbfilename[0])
    else:
        nwbfilename = [f for f in os.listdir(path) if f.endswith(".nwb")]
        if len(nwbfilename):
            nwb_path = os.path.join(path, "pynapplenwb", nwbfilename[0])

    if len(nwb_path) == 0:
        raise RuntimeError("Can't find nwb file in {}".format(path))

    if isinstance(lfp, nap.TsdFrame):
        channels = lfp.columns.values
    elif isinstance(lfp, nap.Tsd):
        if isinstance(channel, int):
            channels = [channel]
        else:
            raise RuntimeError("Please specify which channel it is.")

    io = NWBHDF5IO(nwb_path, "r+")
    nwbfile = io.read()

    all_table_region = nwbfile.create_electrode_table_region(
        region=channels, description="", name="electrodes"
    )

    lfp_electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=lfp.values,
        timestamps=lfp.index.values,
        electrodes=all_table_region,
    )

    lfp = LFP(electrical_series=lfp_electrical_series)

    ecephys_module = nwbfile.create_processing_module(
        name="ecephys", description="processed extracellular electrophysiology data"
    )
    ecephys_module.add(lfp)

    io.write(nwbfile)
    io.close()

    return
