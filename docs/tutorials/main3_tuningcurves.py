#!/usr/bin/env python

"""
	File name: main2.py
	Author: Guillaume Viejo
	Date created: 12/10/2017    
	Python Version: 3.5.2

Here we go into more details about neuroseries
and learn how to use one of the most important function : realign
"""

import numpy as np
import pandas as pd
from pylab import *
from scipy.stats import norm

import pynapple as nap

# Let's make our first tuning curve.
# In this first example, we are going to create some fake head-direction neuron data
# We need spike times and angular position
# For now, it does not matter if the tuning curve is ugly

# let's imagine that the animal is moving her head clockwise at a constant speed
angle = np.arange(0, 100, 0.1)
# let's bring that between 0 and 2pi
angle = angle % (2 * np.pi)
# let's imagine the sampling rate is 100Hz for detecting the position of the animal
# So we have a dt of
dt = 1 / 100.0
# and a duration of
duration = dt * len(angle)
# let's put angle in a pynapple tsd
angle = nap.Tsd(t=np.arange(0, duration, dt), d=angle, time_units="s")

# now let's imagine we have some spikes
spikes = np.sort(np.random.uniform(0, duration, 100))
spikes = nap.Ts(spikes, time_units="s")

# We can plot both angle and spikes together
figure()
plot(angle)
plot(spikes.times(), np.zeros(len(spikes)), "|", markersize=10)
show()

# So the question is: What was the angular position when a given spike was detected
# To this end, you use the value_from function which basically looks for the closest angle from the spike time.
angle_spike = spikes.value_from(angle, angle.time_support)
# The order matters here! It is NOT equivalent to spikes.realign(angle).
# let's look at what it does
figure()
plot(angle)
plot(spikes.times(), np.zeros(len(spikes)), "|", markersize=10)
plot(angle_spike, "o", markersize=5)
show()
# Observe how the spike times are aligned with the dot

# Now it's easy to get a tuning curve
# You just do a histogram of angle_spike
# Which is basically counting the spikes
# First you define bins between 0 and 2pi for example 30 bins
bins = np.linspace(0, 2 * np.pi, 30)
spike_count, bin_edges = np.histogram(angle_spike, bins)

# The animal may have spent most of her time in one particular direction
# You thus need to correct for occupancy
# To this end, you do the histogram of the angle WITH THE SAME BINS
occupancy, _ = np.histogram(angle, bins)

# So correcting for occupancy give :
spike_count = spike_count / occupancy

# Still, it's just a spike count. You want a rate (in Hz).
# This is given by the sampling frequency of the camera that track the position
tuning_curve = spike_count / dt

# let's put that in a nice dataframe
# it's not a time series so no need to use pynapple here
tuning_curve = pd.DataFrame(index=bins[0:-1] + np.diff(bins) / 2, data=tuning_curve)

# Now let's plot that along with the spikes
figure()
subplot(211)
plot(angle)
plot(spikes.times(), np.zeros(len(spikes)), "|", markersize=10)
plot(angle_spike, "o", markersize=5)
subplot(212)
plot(tuning_curve)
show()

# Ok it's ugly but who cares?
# It's just random data

# Now, let's make a place field!
# This time, the spike times are realigned to a 2d position
# Let's imagine the animal is in a circular environment
xpos = np.cos(angle.values) + np.random.randn(len(angle)) * 0.05
ypos = np.sin(angle.values) + np.random.randn(len(angle)) * 0.05

# We can stack the x,y position in a TsdFrame
position = np.vstack((xpos, ypos)).T
position = nap.TsdFrame(t=angle.times(), d=position, columns=["x", "y"])

# and we can plot it
figure()
plot(position["x"], position["y"])
show()

# Now, same thing as before, except that the histogram is 2D

position_spike = position.realign(spikes)
xbins = np.linspace(xpos.min(), xpos.max() + 0.01, 10)
ybins = np.linspace(ypos.min(), ypos.max() + 0.01, 10)
spike_count2, _, _ = np.histogram2d(
    position_spike["y"], position_spike["x"], [ybins, xbins]
)
occupancy2, _, _ = np.histogram2d(position["y"], position["x"], [ybins, xbins])
spike_count2 = spike_count2 / (occupancy2 + 1)
place_field = spike_count2 / dt

# Let's put that in a nice dataframe
place_field = pd.DataFrame(
    index=ybins[0:-1] + np.diff(ybins) / 2,
    columns=xbins[0:-1] + np.diff(xbins) / 2,
    data=place_field,
)


# And plot everything
figure()
subplot(211)

plot(position["x"], position["y"], alpha=0.5)
scatter(
    position_spike["x"], position_spike["y"], color="red", alpha=0.5, edgecolor=None
)
for i in xbins:
    axvline(i)
for i in ybins:
    axhline(i)
xlim(xbins[0], xbins[-1])
ylim(ybins[0], ybins[-1])
subplot(212)
imshow(place_field, origin="lower", aspect="auto")

show()
