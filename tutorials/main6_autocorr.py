#!/usr/bin/env python

'''	
	Author: Guillaume Viejo
	Date created: 03/01/2019
	Python Version: 3.5.2

This scripts will show you how to compute the auto and cross-correlograms of neurons
The data should be found in StarterPack/data_raw/A1110-180621/

The main function crossCorr is already written in StarterPack/python/functions.py

'''
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *


# First let's get some spikes
data_directory = '../data_raw/KA28-190405'
from wrappers import loadSpikeData
spikes, shank = loadSpikeData(data_directory)

# Let's restrict the spikes to the wake episode
from wrappers import loadEpoch
wake_ep = loadEpoch(data_directory, 'wake')

# Let's make the autocorrelogram of the first neuron
neuron_0 = spikes[0]

# restricted for wake
neuron_0 = neuron_0.restrict(wake_ep)

# transforming the times in millisecond
neuron_0 = neuron_0.as_units('ms')

# and extracting the index to feed the function crossCorr
neuron_0_t = neuron_0.index.values

# Let's say you want to compute the autocorr with 5 ms bins
binsize = 5
# with 200 bins
nbins = 200

# Now we can call the function crossCorr
from functions import crossCorr
autocorr_0 = crossCorr(neuron_0_t, neuron_0_t, binsize, nbins)

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
from functions import compute_AutoCorrs
autocorrs, firing_rates = compute_AutoCorrs(spikes, wake_ep)

# Let's plot all the autocorrs
figure()
plot(autocorrs)
xlabel("Time lag (ms)")
show()

