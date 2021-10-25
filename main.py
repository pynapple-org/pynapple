'''
	Minimal working example

'''

import numpy as np
import pandas as pd
import pynapple as ap
from pylab import *
import sys

data_directory = '/home/guillaume/pynapple/data/A2929-200711'


episodes = ['sleep', 'wake']
events = ['1']

############################################################################################### 
# LOADING DATA
###############################################################################################

spikes, shank 						= ap.loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= ap.loadXML(data_directory)
position 							= ap.loadPosition(data_directory, events, episodes)
wake_ep 							= ap.loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= ap.loadEpoch(data_directory, 'sleep')					

############################################################################################### 
# LOADING DATA
###############################################################################################

tuning_curves 						= ap.computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves 						= ap.smoothAngularTuningCurves(tuning_curves, 10, 2)

		

############################################################################################### 
# PLOT
###############################################################################################

figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tuning_curves[i], label = str(shank[i]))
	legend()


show()
