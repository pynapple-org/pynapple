'''
	Minimal working example

'''

import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.pyplot import *
import sys

data_directory = '/home/guillaume/pynapple/data/A2929-200711'


episodes = ['sleep', 'wake']
events = ['1']

############################################################################################### 
# LOADING DATA
###############################################################################################

spikes	 							= nap.loadSpikeData(data_directory)
n_channels, fs, shank_to_channel 	= nap.loadXML(data_directory)
position 							= nap.loadPosition(data_directory, events, episodes)
wake_ep 							= nap.loadEpoch(data_directory, 'wake', episodes)
sleep_ep 							= nap.loadEpoch(data_directory, 'sleep')					

############################################################################################### 
# LOADING DATA
###############################################################################################

tuning_curves 						= nap.computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves 						= nap.smoothAngularTuningCurves(tuning_curves, 10, 2)

hd, stat 						= nap.findHDCells(tuning_curves)

spikes.set_info(hd = hd)
		

############################################################################################### 
# PLOT
###############################################################################################

figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tuning_curves[i])
	

show()
