#!/usr/bin/env python

'''
	File name: main4_raw_data.py
	Author: Guillaume Viejo
	Date created: 13/10/2017    
	Python Version: 3.5.2

This scripts will show you how to use the wrappers function to load raw data
A typical preprocessing pipeline shall output 
	- Mouse-Session.clu.*
	- Mouse-Session.res.*
	- Mouse-Session.fet.*
	- Mouse-Session.spk.*
	- Mouse-Session.xml
	- Mouse-Session.eeg
	- Epoch_TS.csv
	- Mouse-Session_*.csv
	- Mouse-Session_*_analogin.dat



This script will show you how to load the various data you need

The function are already written in the file wrappers.py that should be in the same directory as this script

To speed up loading of the data, a folder called /Analysis will be created and some data will be saved here
So that next time, you load the script, the wrappers will search in /Analysis to load faster
'''

import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *

# The data should be found in pynapple/tutorials/data/KA28-190405
# First thing is to put some data there.
# The data were too big to live in github, please download the following zip:
url = "https://www.dropbox.com/sh/8bu167zgk6u9r1r/AAAdiK1zW6r_Pil78Ger6SlVa?dl=1"
# Extract the zip file in the folder pynapple/tutorial/data_raw/

# We define a string for the data directory (assuming we are in pynapple/tutorials)
data_directory = 'data/KA28-190405'

# To list the files in the directory, you use the os package (for Operating System) and the listdir function
import os
files = os.listdir(data_directory) 
# Check your variables by typing files in your terminal
files

# First, we load the spikes
# Here you can use the pynapple wrapper loadSpikeData
spikes = nap.loadSpikeData(data_directory)
# Type your variables in the terminal to see what it looks like

# Second, we need some information about the recording session like the geometry of the shanks and sampling frequency
# For this particular file format, the info are stored in a xml file (See http://neurosuite.sourceforge.net/information.html)
# You can use the loadXML wrapper
n_channels, fs, shank_to_channel = nap.loadXML(data_directory)
# Again type your variables

# Let's load the animal's position.
# In the data folder, there is a file called Epoch_TS.csv which contains the start and end of the different recording epochs
# This is somehow a Peyrache Lab specific thing, not too important.
# In this recording session, the animal first slept and then explored an environment.
# So you define the episode keys and the index of the events for wake

episodes = ['sleep', 'wake']
events = ['1']

# Now we can load the position and head angles contained into the file Tracking_data.csv
# The order is by default [rotation y, rotation x, rotation z, position x, position y, position z]
position = nap.loadPosition(data_directory, events, episodes)

# The loadPosition is doing many things in the background
# in particular it's making a BehavEpoch.h5 in the folder analysis
# It contains the start and end of all the epochs
# plus it's automatically realigning the start and end of the wake epoch to the start and end of the tracking

# We load the different epoch 
wake_ep 							= nap.loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= nap.loadEpoch(data_directory, 'sleep')					

# We can look at the position of the animal in 2d with a figure
figure()
plot(position['x'], position['z'])
show()


# Now we are going to compute the tuning curve for all neurons during exploration
# The process of making a tuning curve has been covered in main3_tuningcurves.py
# So here we are gonna use the function computeAngularTuningCurves from functions.py 
tuning_curves = nap.computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)

	
# And let's plot all the tuning curves in a polar plot
from pylab import *
figure()
for i, n in enumerate(tuning_curves.columns):
	subplot(5,5,i+1, projection = 'polar')
	plot(tuning_curves[n])	
show()


# It's a bit dirty. Let's smooth the tuning curves ...
tuning_curves = nap.smoothAngularTuningCurves(tuning_curves, 10, 2)

# and plot it again
figure()
for i, n in enumerate(tuning_curves.columns):
	subplot(5,5,i+1, projection = 'polar')
	plot(tuning_curves[n])	
show()
