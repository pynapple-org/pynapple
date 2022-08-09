"""
    Minimal working example

"""

import matplotlib.pyplot as plt
import numpy as np

import pynapple as nap

DATA_DIRECTORY = "your/path/to/A2929-200711"

# LOADING DATA
data = nap.load_session(DATA_DIRECTORY, "neurosuite")

spikes = data.spikes
position = data.position
wake_ep = data.epochs["wake"]

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(
    spikes, position["ry"], 120, minmax=(0, 2 * np.pi)
)


# PLOT
plt.figure()
for i in spikes:
    plt.subplot(3, 5, i + 1, projection="polar")
    plt.plot(tuning_curves[i])
    plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])

plt.show()
