# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-26 21:18:16
# @Last Modified by:   gviejo
# @Last Modified time: 2022-01-26 21:27:38
#!/usr/bin/env python

# # Quick start
#
# The example data to replicate the figure in the jupyter notebook can be found here :
# https://www.dropbox.com/s/1kc0ulz7yudd9ru/A2929-200711.tar.gz?dl=1
#
# The data contain a short sample of a simultaneous recording during sleep and wake
# from the anterodorsal nucleus of the thalamus and the hippocampus.
# It contains both head-direction cells (i.e. cells that fire for a particular direction in the horizontal plane) and place cells (i.e. cells that fire for a particular position in the environment).
#
# Preprocessing of the data was made with Kilosort 2.0 and spike sorting was made with Klusters.
#
# Instructions for installing pynapple can be found here :
# https://peyrachelab.github.io/pynapple/#installation
#
#
#
# This tutorial is meant to provide an overview of pynapple by going through:
# 1. **Input output (IO)**. In this case, pynapple will load a session containing data.
# 2. **Core functions** that handle time series, interval sets and group of time series.
# 3. **Process functions**. A small collection of high-level functions widely used in system neuroscience.


import numpy as np
import pandas as pd

import pynapple as nap

# The first step is to give the path to the data folder.

data_directory = "../your/path/to/A2929-200711"


# The first step is to load the session with the function *load_session*.
# When loading a session for the first time, pynapple will show a GUI
# in order for the user to provide the information about the session, the subject, the tracking, the epochs and the neuronal data.
# When informations has been entered, a [NWB file](https://pynwb.readthedocs.io/en/stable/) is created.
# In this example dataset, the NWB file already exists.

data = nap.load_session(data_directory, "neurosuite")


# The object *data* contains the information about the session such as the spike times of all the neurons,
# the tracking data and the start and ends of the epochs. We can check each object.

spikes = data.spikes
spikes


# *spikes* is a TsGroup object.
# It allows to group together time series with different timestamps and associate metainformation about each neuron.
# Under the hood, it wraps a dictionnary.
# In this case, the location of where the neuron was recorded has been added when loading the session for the first time.
#
# In this case it holds 15 neurons and it is possible to access, similar to a dictionnary, the spike times of a single neuron:

neuron_0 = spikes[0]
neuron_0


# *neuron_0* is a Ts object containing the times of the spikes. Under the hood, it's wrapping a pandas series.
#
# The other information about the session is contained in *data.epochs*.
# In this case, the start and end of the sleep and wake epochs.

epochs = data.epochs
epochs


# Finally this dataset contains tracking of the animal in the environment.
# It can be accessed through *data.position*. *rx, ry, rz* represent respectively
# the roll, the yaw and the pitch of the head of the animal. *x* and *z* represent the position of the animal in the horizontal plane while *y* represent the elevation.

position = data.position
print(position)


# The core functions of pynapple provides many ways to manipulate time series.
# In this example, spike times are restricted to the wake epoch. Notice how the frequencies are changing.

spikes_wake = spikes.restrict(epochs["wake"])

print(spikes_wake)


# The same operation can be applied to position.
# But in this example, we want all the epochs for which position in *x* is above a certain threhsold.
# We can used the function *threshold*.

import matplotlib.pyplot as plt

posx = position["x"]

threshold = 0.08

posxpositive = posx.threshold(threshold)

plt.plot(posx.as_units("s"))
plt.plot(posxpositive.as_units("s"), ".")
plt.axhline(threshold)
plt.xlabel("Time (s)")
plt.ylabel("x")


# The epochs above the threshold can be accessed through the time support of the Tsd object.
# The time support is an important concept in the pynapple package.
# It helps the user to define the epochs for which the time serie should be defined.
# By default, Ts, Tsd and TsGroup objects possess a time support (defined as an IntervalSet).
# It is recommended to pass the time support when instantiating one of those objects.

epochs_above_thr = posxpositive.time_support
print(epochs_above_thr)

# # Tuning curves
# Let's do more advanced analysis.
# Neurons from ADn (group 0 in the *spikes* group object) are know for firing for a particular direction.
# Therefore, we can compute their tuning curves, i.e. their firing rates as a function of the head-direction
# of the animal in the horizontal plane (*ry*).
# We can use the function *compute_1d_tuning_curves*.
# In this case, the tuning curves are computed over 120 bins and between 0 and 2$\pi$.


tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes,
    feature=position["ry"],
    ep=position["ry"].time_support,
    nb_bins=121,
    minmax=(0, 2 * np.pi),
)

tuning_curves


# We can plot tuning curves in polar plots.


neuron_location = spikes.get_info("location")  # to know where the neuron was recorded
plt.figure(figsize=(12, 9))

for i, n in enumerate(tuning_curves.columns):
    plt.subplot(3, 5, i + 1, projection="polar")
    plt.plot(tuning_curves[n])
    plt.title(neuron_location[n] + "-" + str(n), fontsize=18)

plt.tight_layout()
plt.show()


# While ADN neurons show obvious modulation for head-direction, it is not obvious for all CA1 cells.
# Therefore we want to restrict the remaining of the analysis to only ADN neurons.
# We can split the *spikes* group with the function *getby_category*.


spikes_by_location = spikes.getby_category("location")

print(spikes_by_location["adn"])
print(spikes_by_location["ca1"])

spikes_adn = spikes_by_location["adn"]


# # Correlograms
#
# A classical question with head-direction cells is how pairs stay coordinated across brain states i.e. wake vs sleep (see Peyrache, A., Lacroix, M. M., Petersen, P. C., & Buzsáki, G. (2015). Internally organized mechanisms of the head direction sense. Nature neuroscience, 18(4), 569-575.)
# In this example, this coordination across brain states will be evaluated with cross-correlograms of pairs of neurons.
# We can call the function *compute_crosscorrelogram* on both sleep and wake epochs.

cc_wake = nap.compute_crosscorrelogram(
    group=spikes_adn,
    ep=epochs["wake"],
    binsize=20,  # ms
    windowsize=4000,  # ms
    norm=True,
)
cc_sleep = nap.compute_crosscorrelogram(
    group=spikes_adn,
    ep=epochs["sleep"],
    binsize=5,  # ms
    windowsize=400,  # ms
    norm=True,
)


# From the previous figure, we can see that neurons 0 and 1 fires for opposite direction during wake.
# Therefore we expect their cross-correlograms to show a through around 0 time lag meaning those two neurons do not fire spikes together.
# A similar through during sleep for the same pair will thus indicates a persistence of their coordination even if the animal is not moving its head.

xtwake = cc_wake.index.values
xtsleep = cc_sleep.index.values

plt.figure(figsize=(15, 5))
plt.subplot(131, projection="polar")
plt.plot(tuning_curves[[0, 1]])  # The tuning curves of the pair [0,1]
plt.subplot(132)
# plt.plot(cc_wake[(0,1)], color = 'red') # The wake cross-corr of pair (0,1)
plt.bar(
    xtwake, cc_wake[(0, 1)].values, 20, color="green"
)  # The wake cross-corr of pair (0,1)
plt.title("wake")
plt.xlabel("Time (ms)")
plt.ylabel("CC")
plt.subplot(133)
# plt.plot(cc_sleep[(0,1)], color = 'red')
plt.bar(
    xtsleep, cc_sleep[(0, 1)].values, 5, color="green"
)  # The wake cross-corr of pair (0,1)
plt.title("sleep")
plt.xlabel("Time (ms)")
plt.ylabel("CC")
plt.tight_layout()
plt.show()


# # Decoding
#
# This last analysis shows how to use the function decoding of pynapple, in this case with head-direction cells.
#
# The previous result indicates a persistent coordination of head-direction cells during sleep.
# Therefore it is possible to decode a virtual head-direction signal even if the animal is not moving its head.
# This example uses the function *decode_1d* which implements bayesian decoding (see : Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J. (1998). Interpreting neuronal population activity by reconstruction: unified framework with application to hippocampal place cells. Journal of neurophysiology, 79(2), 1017-1044.)
#
# First we can validate the decoding function with the real position of the head of the animal during wake.


tuning_curves_adn = nap.compute_1d_tuning_curves(
    spikes_adn,
    position["ry"],
    position["ry"].time_support,
    nb_bins=121,
    minmax=(0, 2 * np.pi),
)

decoded, proba_angle = nap.decode_1d(
    tuning_curves=tuning_curves_adn,
    group=spikes_adn,
    feature=position["ry"],
    ep=position["ry"].time_support,
    bin_size=0.3,  # second
)
print(decoded)


# We can plot the decoded head-direction along with the true head-direction.


plt.figure(figsize=(15, 5))
plt.plot(position["ry"].as_units("s"), label="True")
plt.plot(decoded.as_units("s"), label="Decoded")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Head-direction (rad)")
plt.show()


# Finally we can plot the decoded activity during sleep and overlay spiking activity of ADN neurons
# as a raster plot (in this case only during the first 10 seconds).


decoded_sleep, proba_angle_Sleep = nap.decode_1d(
    tuning_curves=tuning_curves_adn,
    group=spikes_adn,
    feature=position["ry"],
    ep=epochs["sleep"],
    bin_size=0.1,  # second
)

# Finding quickly max direction of tuning curves
peaks_adn = tuning_curves_adn.idxmax()

# Defining a sub epoch during sleep
subep = nap.IntervalSet(start=0, end=10, time_units="s")

plt.figure(figsize=(16, 5))
# create a raster plot
for n in spikes_adn.keys():
    plt.plot(spikes_adn[n].restrict(subep).as_units("s").fillna(peaks_adn[n]), "|")

plt.plot(decoded_sleep.restrict(subep).as_units("s"), label="Decoded sleep")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Head-direction (rad)")
plt.show()
