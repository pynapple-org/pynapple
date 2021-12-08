#!/usr/bin/env python

'''	
	Author: Guillaume Viejo
	Date created: 03/01/2019
	Python Version: 3.5.2

This scripts will show you how to compute the auto and cross-correlograms of neuronal spikte trains

The main function crossCorr is already written in pynapple

'''
import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *


# The data should be found in pynapple/tutorials/data/A2929-200711
# First thing is to put some data there.
# The data were too big to live in github, please download the following zip:
url = "https://www.dropbox.com/s/1kc0ulz7yudd9ru/A2929-200711.tar.gz?dl=1"
# Extract the zip file in the folder pynapple/tutorials/data/


# We define a string for the data directory (assuming we are in pynapple/tutorials)
data_directory = 'data/A2929-200711'

spikes, shank = nap.loadSpikeData(data_directory)

# Restrict the spikes to the wake episode
wake_ep = nap.loadEpoch(data_directory, 'wake')

# Let's make the autocorrelogram of the first neuron
neuron_0 = spikes[0]

# restrict for wake epoch
neuron_0 = neuron_0.restrict(wake_ep)

# converting the times in millisecond
neuron_0 = neuron_0.as_units('ms')

# and extracting the index to feed the function crossCorr
neuron_0_t = neuron_0.index.values

# Let's say you want to compute the autocorr with 5 ms bins
binsize = 5
# with 200 bins
nbins = 200

# Now we can call the function crossCorr
autocorr_0 = nap.crossCorr(neuron_0_t, neuron_0_t, binsize, nbins)

# The corresponding times can be computed as follow 
times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

# Let's make a time series
autocorr_0 = pd.Series(index = times, data = autocorr_0)

# We need to replace the value at 0
autocorr_0.loc[0] = 0.0

# Let's plot it
figure()
plot(autocorr_0)
show()

# The autocorr_0 is not normalized.
# To normalize, you need to divide by the mean firing rate
mean_fr_0 = len(neuron_0)/wake_ep.tot_length('s')
autocorr_0 = autocorr_0 / mean_fr_0

# Let's plot it again
figure()
plot(autocorr_0)
show()


# Now let's call the function compute_AutoCorrs from functions.py
# We can call the function
autocorrs, firing_rates = ap.compute_AutoCorrs(spikes, wake_ep)

# Let's plot all the autocorrs
figure()
plot(autocorrs)
xlabel("Time lag (ms)")
show()

