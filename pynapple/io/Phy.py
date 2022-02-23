"""
Class and functions for loading data processed with Phy2

@author: Sara Mahallati
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

class Phy(BaseLoader):
    """
    Loader for Phy data
    """
    def __init__(self, path):
        """
        Instantiate the data class from a Phy folder.
        
        Parameters
        ----------
        path : str
            The path to the data.
        """     
        self.basename = os.path.basename(path)
        self.time_support = None
        
        super().__init__(path)

        # Need to check if nwb file exists and if data are there
        loading_phy = True
        if self.path is not None:
            nwb_path = os.path.join(self.path, 'pynapplenwb')
            if os.path.exists(nwb_path):
                files = os.listdir(nwb_path)
                if len([f for f in files if f.endswith('.nwb')]):                    
                    success = self.load_nwb_spikes(path)
                    if success: loading_phy = False

    def load_phy_spikes(self,path, basename, time_support=None):
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
        os.chdir(path)
        cluster_info = pd.read_csv('cluster_info.tsv', sep='\t')
        cluster_id_good = np.array(cluster_info[cluster_info.group =='good'].cluster_id)
        spike_times = np.load('spike_times.npy')
        spike_ids = np.load('spike_clusters.npy')

        index_good = np.in1ind(spike_ids,cluster_id_good)
        
        spike_times_good = spike_times[index_good]
        spike_ids_good = spike_ids[index_good]
        channels = np.array(cluster_info[cluster_info.group =='good'].ch)
        shank = np.array(cluster_info[cluster_info.group =='good'].sh)
        
        self.spikes = nap.TsGroup(
            spike_times_good, 
            time_support=time_support,
            time_units='s',
            group=spike_ids_good)


        return

    def load_phy_params(self, path):
        """
        path should be the folder session containing the params.py file
        
        Function reads :        
        1. the number of channels
        2. the sampling frequency of the dat file 
        
        Parameters
        ----------
        path: str
            The path to the data
                
        Raises
        ------
        RuntimeError
            If path does not contain the params file.
        """
        os.chdir(path)
        from params import +
        
        self.fs_dat = sample_rate
        self.nChannels = n_channels_dat
       
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

    def load_lfp(self, filename=None, channel=None, extension='.eeg', frequency=1250.0, precision='int16', bytes_size=2):
        """
        Load the LFP.
        
        Parameters
        ----------
        filename : str, optional
            The filename of the lfp file.
            It can be useful it multiple dat files are present in the data directory
        channel : int or list of int, optional
            The channel(s) to load. If None return a memory map of the dat file to avoid memory error
        extension : str, optional
            The file extenstion (.eeg, .dat, .lfp). Make sure the frequency match
        frequency : float, optional
            Default 1250 Hz for the eeg file
        precision : str, optional
            The precision of the binary file
        bytes_size : int, optional
            Bytes size of the lfp file
        
        Raises
        ------
        RuntimeError
            If can't find the lfp/eeg/dat file
        
        Returns
        -------
        Tsd or TsdFrame
            The lfp in a time series format
        """
        if filename is not None:
            filepath = os.path.join(self.path, filename)
        else:
            listdir = os.listdir(self.path)
            eegfile = [f for f in listdir if f.endswith(extension)]
            if not len(eegfile):
                raise RuntimeError("Path {} contains no {} files;".format(self.path, extension))
                
            filepath = os.path.join(self.path, eegfile[0])

        self.load_neurosuite_xml(self.path)

        n_channels = int(self.nChannels)

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
            return nap.TsdFrame(
                t = timestep, 
                d=fp, 
                time_units = 's', 
                time_support = time_support)
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

    def write_neuroscope_intervals(self, extension, isets, name):
        """Write events to load with neuroscope (e.g. ripples start and ends)
        
        Parameters
        ----------
        extension : str
            The extension of the file (e.g. basename.evt.py.rip)
        isets : IntervalSet
            The IntervalSet to write
        name : str
            The name of the events (e.g. Ripples)
        """
        start = isets.as_units('ms')['start'].values        
        ends = isets.as_units('ms')['end'].values

        datatowrite = np.vstack((start,ends)).T.flatten()

        n = len(isets)

        texttowrite = np.vstack(((np.repeat(np.array([name + ' start']), n)),                         
                                (np.repeat(np.array([name + ' end']), n))
                                    )).T.flatten()

        evt_file = os.path.join(self.path, self.basename + extension)

        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()        

        return
    
    def load_MeanWaveforms(self, epoch = None, waveform_window = nap.IntervalSet(start = -0.5, end = 1, time_units = 'ms'), spike_count = 1000):
        """
        load the mean waveforms from a dat file.
        
        
        Parameters
        ----------
        epoch : IntervalSet
            default = None
            Restrict spikes to an epoch.
        waveform_window : IntervalSet
            default = start = -0.0005, end = 0.001, time_units = 'ms'
            Limit waveform extraction before and after spike time
        spike_count : int
            default = 1000
            Number of spikes used per neuron for the calculation of waveforms
            
        Returns
        ----------
        dictionary
            the waveforms for all neurons
        pandas.Series
            the channel with the maximum waveform for each neuron
        
        """
        spikes = self.spikes
        if not os.path.exists(self.path): #check if path exists
            print("The path "+self.path+" doesn't exist; Exiting ...")
            sys.exit()    

        # Load XML INFO
        self.load_neurosuite_xml(self.path)
        n_channels = self.nChannels
        fs = self.fs_dat
        group_to_channel = self.group_to_channel
        group = spikes.get_info('group')
                  
        #Check if there is an epoch, restrict spike times to epoch
        if epoch is not None:
            if type(epoch) is not nap.IntervalSet:
                print('Epoch must be an IntervalSet')
                sys.exit()
            else:
                print('Restricting spikes to epoch')
                spikes = spikes.restrict(epoch)
                epstart = int(epoch.as_units('s')['start'].values[0]*fs)
                epend = int(epoch.as_units('s')['end'].values[0]*fs)

        #Find dat file
        files = os.listdir(self.path)
        dat_files    = np.sort([f for f in files if 'dat' in f and f[0] != '.'])

        #Need n_samples collected in the entire recording from dat file to load
        file = os.path.join(self.path, dat_files[0])
        f = open(file, 'rb') #open file to get number of samples collected in the entire recording
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
        f.close()     
        #map to memory all samples for all channels, channels are numbered according to neuroscope number
        fp = np.memmap(file, np.int16, 'r', shape = (n_samples, n_channels))
          
        #convert spike times to spikes in sample number
        sample_spikes = {neuron:(spikes[neuron].as_units('s').index.values*fs).astype('int') for neuron in spikes}

        #prep for waveforms
        overlap = int(waveform_window.tot_length(time_units='s')) #one spike's worth of overlap between windows
        waveform_window = abs(np.array(waveform_window.as_units('s'))[0] * fs).astype(int) #convert time to sample number
        neuron_waveforms = {n: np.zeros([np.sum(waveform_window), len(group_to_channel[group[n]])]) for n in sample_spikes}

        #divide dat file into batches that slightly overlap for faster loading
        batch_size = 3000000
        windows = np.arange(0, int(endoffile/n_channels/bytes_size), batch_size)
        if epoch is not None:
            print('Restricting dat file to epoch')
            windows = windows[(windows>=epstart) & (windows<=epend)]
        batches = []
        for i in windows: #make overlapping batches from the beginning to end of recording
            if i == windows[-1]:
                batches.append([i-overlap, n_samples])
            elif i-30 >= 0 and i+30 <= n_samples:
                batches.append([i-overlap, i+batch_size+overlap])
            else:
                batches.append([i, i+batch_size+overlap])
        batches = [np.int32(batch) for batch in batches]
        
        sample_counted_spikes = {}
        for index, neuron in enumerate(sample_spikes):
            if len(sample_spikes[neuron]) >= spike_count:
                sample_counted_spikes[neuron] = np.array(np.random.choice(list(sample_spikes[neuron]), spike_count))
            elif len(sample_spikes[neuron]) < spike_count:
                print('Not enough spikes in neuron ' + str(index) + '... using all spikes')
                sample_counted_spikes[neuron] = sample_spikes[neuron]
                
        #Make one array containing all selected spike times of all neurons - will be used to check for spikes before loading dat file
        spike_check = np.array([int(spikes_neuron) for spikes_neuron in sample_counted_spikes[neuron] for neuron in sample_counted_spikes])

        for index, timestep in enumerate(batches):
            print('Extracting waveforms from dat file: window ' + str(index+1) + '/' + str(len(windows)))
 
            if len(spike_check[(timestep[0]<spike_check) & (timestep[1]>spike_check)]) == 0:
                continue #if there are no spikes for any neurons in this batch, skip and go to the next one
            
            #Load dat file for timestep
            tmp = pd.DataFrame(data = fp[timestep[0]:timestep[1],:], columns = np.arange(n_channels), index = range(timestep[0],timestep[1])) #load dat file

            #Check if any spikes are present
            for neuron in sample_counted_spikes:
                neurontmp = sample_counted_spikes[neuron]
                tmp2 = neurontmp[(timestep[0]<neurontmp) & (timestep[1]>neurontmp)]
                if len(neurontmp) == 0:
                    continue #skip neuron if it has no spikes in this batch
                tmpn = tmp[group_to_channel[group[neuron]]] #restrict dat file to the channel group of the neuron

                for time in tmp2: #add each spike waveform to neuron_waveform
                    spikewindow = tmpn.loc[time-waveform_window[0]:time+waveform_window[1]-1] #waveform for this spike time
                    if spikewindow.isnull().values.any() == False:
                        neuron_waveforms[neuron] += spikewindow.values
        meanwf = {n: pd.DataFrame(data = np.array(neuron_waveforms[n])/spike_count, 
                                  columns = np.arange(len(group_to_channel[group[n]])), 
                                  index = np.array(np.arange(-waveform_window[0], waveform_window[1]))/fs) for n in sample_counted_spikes}
        
        #find the max channel for each neuron
        maxch = pd.Series(data = [meanwf[n][meanwf[n].loc[0].idxmin()].name for n in meanwf], 
                          index = spikes.keys())
        
        return meanwf, maxch

