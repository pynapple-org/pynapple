"""
Minimal working example

"""

import matplotlib.pyplot as plt
import numpy as np

import pynapple as nap

# LOADING DATA FROM NWB
data = nap.load_file("A2929-200711.nwb")

spikes = data["units"]
head_direction = data["ry"]
wake_ep = data["position_time_support"]

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_tuning_curves(
    spikes, head_direction, 120, epochs=wake_ep, range=[(0, 2 * np.pi)]
)

# PLOT
g = tuning_curves.plot(
    row="unit", col_wrap=5, subplot_kws={"projection": "polar"}, sharey=False
)
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
g.set_titles("")
g.set_xlabels("")
plt.show()
