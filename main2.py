"""
    Minimal working example

"""

import matplotlib.pyplot as plt
import numpy as np

import pynapple as nap


# LOADING DATA
DATA_DIRECTORY = "your/path/to/MyProject/"
data = nap.load_folder(DATA_DIRECTORY)

session = data["sub-A2929"]["A2929-200711"]

spikes = session["derivatives"]["spikes"]
position = session["derivatives"]["position"]
wake_ep = session["derivatives"]["wake_ep"]
sleep_ep = session["derivatives"]["sleep_ep"]


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
