#!/usr/bin/env python

'''	
	Author: Guillaume Viejo
	Date created: 13/10/2017    
	Python Version: 3.5.2

This scripts will show you how to use the wrappers function to load MATLAB data
The data should be found in StarterPack/data_matlab/

It's a bit more complex than the raw data provided as we have the sleep scoring that has been done
So the idea here is to work with real IntervalSet


'''
import numpy as np
import pandas as pd
import neuroseries as nts
import os
from scipy.io import loadmat
from pylab import *

# first we define a string for the data directory
data_directory = '../data_matlab/Mouse12-120806'

# All files that have a .mat extension are data that have been saved with Matlab
# At some point, data files should be saved to be easily compatible for matlab and python
# But those are not
# Except for the file called GeneralInfo.mat, you should use the Wrappers provided in functions.py

# First we load general info about the session recording
# We need the scipy library which is another scientific library and the io.loadmat function
generalinfo = loadmat(data_directory+'/Analysis/GeneralInfo.mat')

# Type your variable in your terminal and observe the results
generalinfo
# The type of your variable is a dictionnary
print(type(generalinfo))
# We can check the keys of the dictionarray:
for k in generalinfo.keys():
	print(k)

# What is interesting here is the shankStructure
# The recording probe is made of shanks (usually 8)
# And shanks can go in different regions of the brain
# If you type :
generalinfo['shankStructure']
# you see that it return a matlab object with name of region i.e. hippocampus, thalamus, pfc ...

# To parse this object is complex, therefore you should use now the wrapper I provided in functions.py
from wrappers import loadShankStructure
# You call the function by giving the generalinfo dictionnary
shankStructure = loadShankStructure(generalinfo)
# You check your variable by typing :
shankStructure
# And now each region is associated with a list of number.
# Each number indicates the index of a shank
# For example, the thalamus shanks index are :
shankStructure['thalamus']
# Therefore we have here 8 shanks in the thalamus
# This will be useful to know which spikes were recorded in the thalamus


# Now we can load the spikes in Mouse12-120806_SpikeData.mat by using another wrapper
from wrappers import loadSpikeData
# and we want only the spikes from the thalamus
# So you need to pass the shankStructure of the thalamus as an argument of the function 
spikes,shank = loadSpikeData(data_directory, shankStructure['thalamus'])
# To acces one neuron:
spikes[0]
# It returns a Ts object with the time occurence of spikes on the left column and NaN in the right columns

# Which neurons of the thalamus are head-direction neurons?
# To know that, you need to load Mouse12-120806_HDCells.mat
# But first, you need to give the index of the thalamic neuron for which we are interested here
# The neuron index is given by the keys of the dictionnary spikes
# You can extract the keys and put them in another variable with :
my_thalamus_neuron_index = list(spikes.keys())
# Now you can call the function to load HD info 
from wrappers import loadHDCellInfo
hd_neuron_index = loadHDCellInfo(data_directory+'/Analysis/HDCells.mat', my_thalamus_neuron_index)
# You have now a new array of neuron index
# You can now separate the thalamic neurons and the head-direction thalamic neurons
# First you declare a new dictionnary
hd_spikes = {}
# Then you need to loop over the hd_neuron_index to put each neuron in the new dict
for neuron in hd_neuron_index:
	hd_spikes[neuron] = spikes[neuron]

# You can check that you have less neurons in the new dictionnary
# There are some neurons in the thalamus that are not head-direction neurons
hd_spikes.keys()


# Then we can load different eppoch here
# THe sleep scoring has been done already
from wrappers import loadEpoch
# You need to specify the type of epoch you want to load
# Possibles inputs are ['wake', 'sleep', 'sws', 'rem']
wake_ep 		= loadEpoch(data_directory, 'wake')
# The function will automaticaly search for the rigth file
# You can check your variables by typing them


# Next step is to load the angular value at each time steps
# We need to load Mouse12-120806_PosHD.txt which is a text file
# We can use the function genfromtxt of numpy that load a simple text file
data = np.genfromtxt('../data_matlab/Mouse12-120806/Mouse12-120806_PosHD.txt')
# Check your variable by typing it
# It's an array, we can check the dimension by typing :
data.shape
# It has 40858 lines and 4 columns
# The columns are respectively [times | x position in the arena | y position in the arena | angular value of the head]
# So we can use the TsdFrame object of neuroseries as seen in main3.py
mouse_position = nts.TsdFrame(d = data[:,[1,2,3]], t = data[:,0], time_units = 's', columns = ['x', 'y', 'ang'])

# It's good to always check the data by plotting them
# To see the position of the mouse in the arena during the session
# you can plot the x position versus the y position

figure()
plot(mouse_position['x'].values, mouse_position['y'].values)
xlabel("x position (cm)")
ylabel("y position (cm)")
show()

# Now we are going to compute the tuning curve for all neurons during exploration
# The process of making a tuning curve has been covered in main3_tuningcurves.py
# So here we are gonna define a function that will be looped over each HD neurons

def computeAngularTuningCurve(spike_ts, angle_tsd, nb_bins = 60, frequency = 39.065):
	angle_spike = angle_tsd.realign(spike_ts)
	bins = np.linspace(0, 2*np.pi, nb_bins)
	spike_count, bin_edges = np.histogram(angle_spike, bins)
	occupancy, _ = np.histogram(angle_tsd, bins)
	spike_count = spike_count/occupancy
	tuning_curve = spike_count*frequency
	tuning_curve = pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = tuning_curve)
	return tuning_curve

# Let's prepare a dataframe to receive our tuning curves
column_names = ['Mouse12-120806_'+str(k) for k in hd_spikes.keys()]
tuning_curves = pd.DataFrame(columns = column_names)

# let's do the loop
for k in hd_spikes.keys():
	spks = hd_spikes[k]
	angle = mouse_position['ang']
	# first thing is to restrict the data to the exploration period
	spks = spks.restrict(wake_ep)
	# BAD TO CHANGE
	angle = angle[~angle.index.duplicated(keep='first')]
	angle = angle.restrict(wake_ep)
	# second we can call the function
	tcurve = computeAngularTuningCurve(spks, angle)
	# third we can add the new tuning curve to the dataframe ready
	tuning_curves['Mouse12-120806_'+str(k)] = tcurve
	
# And let's plot all the tuning curves


figure()
plot(tuning_curves)
xlabel("Head-direction (rad)")
ylabel("Firing rate")
title("ADn neuron")
grid()
show()

# Even cooler we can do a polar plot
figure()
subplot(111, projection='polar')
plot(tuning_curves)
xlabel("Head-direction (rad)")
ylabel("Firing rate")
title("ADn neuron")
grid()
show()



