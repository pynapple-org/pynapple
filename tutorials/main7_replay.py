#!/usr/bin/env python

'''	
	Author: Guillaume Viejo
	Date created: 03/01/2019
	Python Version: 3.5.2

This scripts shows how to do bayesian decoding during wake
It can then be applied during sleep to search for replay

Here I will use the data from A110-180621 

The data should be found in StarterPack/data_raw/KA28-190405

'''
import numpy as np
import pandas as pd
import neuroseries as nts
import os
import sys
# from pylab import *

data_directory = '../data_raw/KA28-190405'

# Let's get some spikes, epochs and positions
from wrappers import loadSpikeData, loadEpoch, loadPosition
spikes, shank = loadSpikeData(data_directory)
wake_ep = loadEpoch(data_directory, 'wake')
position = loadPosition(data_directory)


# The first is just computing the tuning curves
# Here I use the function and code defined in main4_raw_data.py
from functions import computeAngularTuningCurves
tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 61)

# To use only HD cells, we can use the function findHDCells from functions.py
from functions import findHDCells
hd_idx, stat = findHDCells(tuning_curves)
tuning_curves = tuning_curves[hd_idx]


# The second step is to compute the binned spike count over the whole wake period.
# The bin size here will determine the resolution of the decoded angle
bin_size = 200 # ms
# Let's make the bins starting and ending in wake
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1], bin_size)

# Let's prepare the dataframe that will receive the binned firing rate
spike_counts = pd.DataFrame(index = bins[0:-1], columns = hd_idx)

# Let's loop over spikes dictionary taking only hd_idx
for k in hd_idx:
	spks = spikes[k].restrict(wake_ep).as_units('ms').index.values
	spike_counts[k], _ = np.histogram(spks, bins)

# Now we can start the bayesian decoding
# The equation can be found in Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
# Equation 36
# It can be decomposed in two parts with the first part being constante for all time steps
part1 = np.exp(-(bin_size/1000)*tuning_curves.sum(1))
# part 2 which is the occupancy of each position. it gives P(x) with the position
part2 = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]

# Then we need to loop over each time steps and compute the probability
# It's best to prepare the pandas dataframe that will receive the proba
# Here notice that the columns are the angular position
proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values)

# Now let's loop
for t in spike_counts.index:
	part3 = np.prod(tuning_curves**spike_counts.loc[t], 1)
	break
	p = part1 * part2 * part3
	proba_angle.loc[t] = p/p.sum() # Normalization process here

# Small problem with pandas you need to convert proba_angle to float64 to use idxmax
proba_angle = proba_angle.astype('float')

# But then getting the angle is straightforward
decoded = proba_angle.idxmax(1)

# Let's put that in a neuroseries 
decoded = nts.Tsd(t = decoded.index.values, d = decoded.values, time_units = 'ms')

# And now we can compare the two 
from pylab import *
figure()
plot(position['ry'].restrict(wake_ep), label = 'true')
plot(decoded, label = 'decoded')
legend()
show()

# Now this is a bit slow.
# the function decodeHD in function.py is faster 
# You just give the tuning curves, spike times and bin_size
from functions import decodeHD
spikes_hd = {k:spikes[k] for k in hd_idx}
occupancy = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
decoded, proba_angle = decodeHD(tuning_curves, spikes_hd, wake_ep, occupancy, bin_size = 200)

# Let's check again
figure()
plot(position['ry'].restrict(wake_ep), label = 'true')
plot(decoded, label = 'decoded')
legend()
show()

# now you should decode during sleep to see
