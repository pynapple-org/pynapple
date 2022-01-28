import numpy as np
import sys,os
sys.path.append('../')
import pandas as pd
import sys
import pynapple as nap      

def loadMeanWaveforms(path, spikes, epoch = None, waveform_window = [0.5, 1], spike_count = 1000):
    """
    load waveforms from dat file.
    
    
    Parameters
    ----------
    path : string
        path in computer where dat and xml files are located    
    spikes : pynapple.core.ts_group.TsGroup
        A TsGroup should contain the spike times of n neurons.
    epoch : ap.interval_set.Intervalset
        default = None
        Restrict spikes to an epoch.
    waveform_window : array
        default = [0.5, 1]
        Limit waveform extraction to ms before and ms after spike time
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
    if not os.path.exists(path): #check if path exists
        print("The path "+path+" doesn't exist; Exiting ...")
        sys.exit()    

    # Load XML INFO
    n_channels, fs, shank_to_channel     = nap.loadXML(path)

    #Check spikes - can be Tsd or TsGroup
    if type(spikes) is not nap.core.ts_group.TsGroup:
        print('spikes must be a TsGroup object')
        sys.exit()
    elif type(spikes) is nap.core.ts_group.TsGroup: #if TsGroup, check that TsGroup contains spike times
        for neuron in spikes:
            if type(spikes[neuron]) is not nap.core.time_series.Tsd:
                print('spike times contain non-Tsd Objects')
                sys.exit()
    group = spikes.get_info('group')
              
    #Check if there is an epoch, restrict spike times to epoch
    if epoch is not None:
        if type(epoch) is not nap.core.interval_set.IntervalSet:
            print('Epoch must be an IntervalSet')
            sys.exit()
        else:
            print('Restricting spikes to epoch')
            spikes = spikes.restrict(epoch)
            epstart = int(epoch.as_units('s')['start'].values[0]*fs)
            epend = int(epoch.as_units('s')['end'].values[0]*fs)

    #Find dat file
    files = os.listdir(path)
    dat_files    = np.sort([f for f in files if 'dat' in f and f[0] != '.'])

    #Need n_samples collected in the entire recording from dat file to load
    file = os.path.join(path, dat_files[0])
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
    waveform_window = (np.array(waveform_window) * 0.001 * fs).astype(int) #convert ms time to number of samples
    neuron_waveforms = {n: np.array([[0]*len(shank_to_channel[group[n]])]*np.sum(waveform_window)) for n in sample_spikes}
    count = {n: 0 for n in sample_spikes}

    #divide dat file into batches that slightly overlap for faster loading
    batch_size = 3000000
    overlap = int(np.sum(waveform_window)) #one spike's worth of overlap between windows
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
    
    import random
    sample_counted_spikes = {}
    for index, neuron in enumerate(sample_spikes):
        if len(sample_spikes[neuron]) >= spike_count:
            sample_counted_spikes[neuron] = np.array(random.sample(list(sample_spikes[neuron]), spike_count))
        elif len(sample_spikes[neuron]) < spike_count:
            print('Not enough spikes in neuron ' + str(index) + '... using all spikes')
            sample_counted_spikes[neuron] = sample_spikes[neuron]

    for index, timestep in enumerate(batches):
        print('Extracting waveforms from dat file: window ' + str(index+1) + '/' + str(len(windows)))
        #Load dat file for timestep
        tmp = pd.DataFrame(data = fp[timestep[0]:timestep[1],:], columns = np.arange(n_channels), index = range(timestep[0],timestep[1])) #load dat file

        #Check if any spikes are present
        for neuron in sample_counted_spikes:
            tmpn = tmp[shank_to_channel[group[neuron]]] #restrict dat file to the shank of the neuron
            if len(sample_counted_spikes[neuron]) == 0:
                print('No spikes for neuron ' + str(neuron) + ', skipping...')
                continue
            tmp2 = sample_counted_spikes[neuron][(timestep[0]<sample_counted_spikes[neuron]) & (timestep[1]>sample_counted_spikes[neuron])]
            
            if len(tmp2) > 0: #if spikes present in timestep
                for time in tmp2: #add each spike waveform to neuron_waveform
                    spikewindow = tmpn.loc[time-waveform_window[0]:time+waveform_window[1]-1] #waveform for this spike time
                    if spikewindow.isnull().values.any() == False:
                        try:
                            neuron_waveforms[neuron] += spikewindow.values
                            count[neuron] += 1
                        except:
                            pass #if full waveform is not present in this batch, it is ignored. It will be fully present in the next batch
        del tmp, tmpn, tmp2
    meanwf = {n: pd.DataFrame(data = np.array(neuron_waveforms[n])/count[n], columns = np.arange(len(shank_to_channel[group[n]])), index = np.array(np.arange(-waveform_window[0], waveform_window[1]))/fs) for n in sample_counted_spikes}
    nwb_waveform = [meanwf[i] for i in meanwf]
    #find the max channel for each neuron
    maxch = pd.Series(data = [meanwf[n][meanwf[n].loc[0].idxmin()].name for n in meanwf], index = spikes.keys())
    
    return meanwf, maxch