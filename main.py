'''
	Minimal working example

'''

import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.pyplot import *
import sys

data_directory = 'your/path/to/A2929-200711'

################################################################
# LOADING DATA
################################################################
data = nap.load_session(data_directory, 'neurosuite')


spikes = data.spikes
position = data.position
wake_ep = data.epochs['wake']

################################################################
# COMPUTING TUNING CURVES
################################################################
tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], position['ry'].time_support, 120,  minmax=(0, 2*np.pi))

		
################################################################
# PLOT
################################################################

figure()
for i in spikes:
	subplot(6,7,i+1, projection = 'polar')
	plot(tuning_curves[i])
	

show()
