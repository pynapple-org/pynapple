#!/usr/bin/env python

"""
Class and functions for loading data processed with the Neurosuite (Klusters, Neuroscope, NDmanager)

@author: Guillaume Viejo
"""
import os, sys
import numpy as np
from .. import core as nap
from .loader import BaseLoader
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO
from pynwb.device import Device
from xml.dom import minidom 
from .ephys_gui import EphysGUI
from PyQt5.QtWidgets import QApplication
import re

class NeuroSuite(BaseLoader):
    """
    Loader for kluster data
    """
    def __init__(self, path):
        """
        Instantiate the data class from a neurosuite folder.
        
        Parameters
        ----------
        path : str
            The path to the data.
        """     
        self.basename = os.path.basename(path)
        self.time_support = None
        
        super().__init__(path)

        # Need to check if nwb file exists and if data are there
        loading_neurosuite = True
        if self.path is not None:
            nwb_path = os.path.join(self.path, 'pynapplenwb')
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith('.nwb')]):                    
                    success = self.load_nwb_spikes(path)
                    if success: loading_neurosuite = False

        # Bypass if data have already been transfered to nwb
        if loading_neurosuite:
            self.load_neurosuite_xml(path)
            # print("XML loaded")
            # To label the electrodes groups
            app = QApplication([])
            self.window = EphysGUI(path=path, groups=self.group_to_channel)
            app.exec()
            # print("GUI DONE")     
            if self.window.status:
                self.ephys_information = self.window.ephys_information

                self.load_neurosuite_spikes(path, self.basename, self.time_support)
                
                self.save_data(path)
            app.quit()

    def load_neurosuite_spikes(self,path, basename, time_support=None, fs = 20000.0):
        """
        Read the clus and res files and convert to NWB.
        Instantiate automatically a TsGroup object.
        
        Parameters
        ----------
        path : str
            The path to the data
        basename : str
            Basename of the clu and res files.
        time_support : IntevalSet, optional
            The time support of the data
        fs : float, optional
            Sampling rate of the recording.
                
        Raises
        ------
        RuntimeError
            If number of clu and res are not equal.

        """
        files = os.listdir(path)
        clu_files     = np.sort([f for f in files if '.clu.' in f and f[0] != '.'])
        res_files     = np.sort([f for f in files if '.res.' in f and f[0] != '.'])
        clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
        clu2         = np.sort([int(f.split(".")[-1]) for f in res_files])
        if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
            raise RuntimeError("Not the same number of clu and res files in "+path+"; Exiting ...")

        count = 0
        spikes = {}
        group = pd.Series(dtype=np.int32)
        for i, s in zip(range(len(clu_files)),clu1):
            clu = np.genfromtxt(os.path.join(path,basename+'.clu.'+str(s)),dtype=np.int32)[1:]
            if np.max(clu)>1: # getting rid of mua and noise
                res = np.genfromtxt(os.path.join(path,basename+'.res.'+str(s)))
                tmp = np.unique(clu).astype(int)
                idx_clu = tmp[tmp>1]
                idx_out = np.arange(count, count+len(idx_clu))

                for j,k in zip(idx_clu, idx_out):
                    t = res[clu==j]/fs
                    spikes[k] = nap.Ts(t=t, time_units='s')
                    group.loc[k] = s

                count+=len(idx_clu)    

        group = group - 1 # better to start it a 0    

        self.spikes = nap.TsGroup(
            spikes, 
            time_support=time_support,
            time_units='s',
            group=group)

        # adding some information to help parse the neurons
        names = pd.Series(
            index = group.index,
            data = [self.ephys_information[group.loc[i]]['name'] for i in group.index]
            )
        if ~np.all(names.values==''):
            self.spikes.set_info(name=names)
        locations = pd.Series(
            index = group.index,
            data = [self.ephys_information[group.loc[i]]['location'] for i in group.index]
            )
        if ~np.all(locations.values==''):
            self.spikes.set_info(location=locations)

        return

    def load_neurosuite_xml(self, path):
        """
        path should be the folder session containing the XML file
        
        Function reads
        --------------
        1. the number of channels
        2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
            eeg file first if both are present or both are absent
        3. the mappings shanks to channels as a dict
        
        Parameters
        ----------
        path: str
            The path to the data
                
        Raises
        ------
        RuntimeError
            If path does not contain the xml file.
        """
        listdir = os.listdir(path)
        xmlfiles = [f for f in listdir if f.endswith('.xml')]
        if not len(xmlfiles):
            raise RuntimeError("Path {} contains no xml files;".format(path))
            sys.exit()
        new_path = os.path.join(path, xmlfiles[0])
        
        
        self.xmldoc      = minidom.parse(new_path)
        self.nChannels   = self.xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
        self.fs_dat      = self.xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
        self.fs_eeg      = self.xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data

        self.group_to_channel = {}
        groups      = self.xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
        for i in range(len(groups)):
            self.group_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
        
        return


    def save_data(self, path):
        """
        Save the data to NWB format.
        
        Parameters
        ----------
        path : str
            The path to save the data
        
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

        electrode_groups = {}

        for g in self.group_to_channel:

            device = nwbfile.create_device(
                name=self.ephys_information[g]['device']['name']+'-'+str(g),
                description=self.ephys_information[g]['device']['description'],
                manufacturer=self.ephys_information[g]['device']['manufacturer']
                )

            if len(self.ephys_information[g]['position']) and type(self.ephys_information[g]['position']) is str:
                self.ephys_information[g]['position'] = re.split(';|,| ', self.ephys_information[g]['position'])
            elif self.ephys_information[g]['position'] == '':
                self.ephys_information[g]['position'] = None

            electrode_groups[g] = nwbfile.create_electrode_group(
                name='group'+str(g)+'_'+self.ephys_information[g]['name'],
                description=self.ephys_information[g]['description'],
                position=self.ephys_information[g]['position'],
                location=self.ephys_information[g]['location'],
                device=device
                )

            for idx in self.group_to_channel[g]:
                nwbfile.add_electrode(id=idx,
                                      x=0.0, y=0.0, z=0.0,
                                      imp=0.0,
                                      location=self.ephys_information[g]['location'], 
                                      filtering='none',
                                      group=electrode_groups[g])

        # Adding units
        nwbfile.add_unit_column('location', 'the anatomical location of this unit')
        nwbfile.add_unit_column('group', 'the group of the unit')
        for u in self.spikes.keys():
            nwbfile.add_unit(
                id=u,
                spike_times=self.spikes[u].as_units('s').index.values,                
                electrode_group=electrode_groups[self.spikes.get_info('group').loc[u]],
                location=self.ephys_information[self.spikes.get_info('group').loc[u]]['location'],
                group=self.spikes.get_info('group').loc[u]
                )

        io.write(nwbfile)
        io.close()

        return

    def load_nwb_spikes(self, path):
        """
        Read the NWB spikes to extract the spike times.

        Parameters
        ----------
        path : str
            The path to the data
        
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

        io = NWBHDF5IO(self.nwbfilepath, 'r')
        nwbfile = io.read()

        if nwbfile.units is None:
            io.close()
            return False
        else:
            units = nwbfile.units.to_dataframe()
            spikes = {n:nap.Ts(t=units.loc[n,'spike_times'], time_units='s') for n in units.index}

            self.spikes = nap.TsGroup(
                spikes, 
                time_support=self.time_support,
                time_units='s',
                group=units['group']
                )

            if ~np.all(units['location']==''):
                self.spikes.set_info(location=units['location'])

            io.close()
            return True

