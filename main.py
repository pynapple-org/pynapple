'''
	Minimal working example

'''

import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.pyplot import *
import sys

data_directory = '/home/guillaume/pynapple/data/A2929-200711'

################################################################
# LOADING DATA
################################################################
# data = nap.load_session(data_directory, 'neurosuite')
data = nap.load_session(data_directory, 'neurosuite')


spikes = data.spikes
position = data.position
wake_ep = data.epochs['wake']

################################################################
# COMPUTING TUNING CURVES
################################################################
tuning_curves = nap.computeAngularTuningCurves(spikes, position['ry'], wake_ep, 120)
tuning_curves = nap.smoothAngularTuningCurves(tuning_curves, 10, 2)

hd, stat = nap.findHDCells(tuning_curves)

#sys.exit()

spikes.set_info(hd = hd)
		

############################################################################################### 
# PLOT
###############################################################################################

figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tuning_curves[i])
	

show()
