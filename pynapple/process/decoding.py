# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 23:34:48
# @Last Modified by:   gviejo
# @Last Modified time: 2022-01-03 18:03:16

import numpy as np
from numba import jit
import pandas as pd
from .. import core as nap

def decode_1d(tuning_curves, group, ep, bin_size):
	"""
	Perform bayesian decoding over one dimension
	
	Parameters
	----------
	tuning_curves : TYPE
	    Description
	group : TYPE
	    Description
	ep : TYPE
	    Description
	bin_size : TYPE
	    Description
	
	Returns
	-------
	TYPE
	    Description
	"""


	return

def decodeHD(tuning_curves, spikes, ep, bin_size = 200, px = None):
	"""
		See : Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
		tuning_curves: pd.DataFrame with angular position as index and columns as neuron
		spikes : dictionnary of spike times
		ep : nts.IntervalSet, the epochs for decoding
		bin_size : in ms (default:200ms)
		px : Occupancy. If None, px is uniform
	"""		
	if len(ep) == 1:
		bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
	else:
		# ep2 = nts.IntervalSet(ep.copy().as_units('ms'))
		# ep2 = ep2.drop_short_intervals(bin_size*2)
		# bins = []
		# for i in ep2.index:
		# 	bins.append(np.arange())
		# bins = np.arange(ep2.start.iloc[0], ep.end.iloc[-1], bin_size)
		print("TODO")
		sys.exit()


	order = tuning_curves.columns.values
	# TODO CHECK MATCH

	# smoothing with a non-normalized gaussian
	w = scipy.signal.gaussian(51, 2)
	
	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = order)
	for n in spike_counts:		
		spks = spikes[n].restrict(ep).as_units('ms').index.values
		tmp = np.histogram(spks, bins)
		spike_counts[n] = np.convolve(tmp[0], w, mode = 'same')
		# spike_counts[k] = tmp[0]

	tcurves_array = tuning_curves.values
	spike_counts_array = spike_counts.values
	proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

	part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
	if px is not None:
		part2 = px
	else:
		part2 = np.ones(tuning_curves.shape[0])
	#part2 = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
	
	for i in range(len(proba_angle)):
		part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
		p = part1 * part2 * part3
		proba_angle[i] = p/p.sum() # Normalization process here

	proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)	
	proba_angle = proba_angle.astype('float')
	decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
	return decoded, proba_angle, spike_counts
