'''
    Minimal working example

'''

import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap

DATA_DIRECTORY = 'your/path/to/A2929-200711'

# LOADING DATA
data = nap.load_session(DATA_DIRECTORY, 'neurosuite')

spikes = data.spikes
position = data.position
wake_ep = data.epochs['wake']

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], position['ry'].time_support, 120,  minmax=(0, 2*np.pi))


# PLOT
plt.figure()
for i in spikes:
    plt.subplot(6, 7, i+1, projection='polar')
    plt.plot(tuning_curves[i])

plt.show()
