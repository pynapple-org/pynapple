# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2022-01-02 23:33:42
# @Last Modified by:   gviejo
# @Last Modified time: 2022-01-04 15:36:12

import numpy as np
import pandas as pd
from .. import core as nap


def compute_1d_tuning_curves(group, variable, ep, nb_bins, minmax=None):
	"""
	Compute 1 dimensional tuning curves from target variable.
	
	Parameters
	----------
	group: TsGroup or dict of Ts/Tsd objects
	    The group input
	variable: Tsd
	    The unidimensional target feature (e.g. head-direction)
	ep: IntervalSet
	    The epoch to perform the operation
	nb_bins: int
	    Number of bins in the tuning curve
	minmax: tuple or list, optional
	    The min and max boundaries of the tuning curves.
	    If None, the boundaries is inferred from the target feature
	
	Returns
	-------
	pandas.DataFrame
	    DataFrame to hold the tuning curves
	"""
	if type(group) is dict:
		group = nap.TsGroup(group, time_support = ep)

	group_value = group.value_from(variable, ep)

	if minmax is None:
		bins = np.linspace(np.min(variable), np.max(variable), nb_bins)
	else:
		bins = np.linspace(minmax[0], minmax[1], nb_bins)
	idx = bins[0:-1]+np.diff(bins)/2

	tuning_curves = pd.DataFrame(index = idx, columns = list(group.keys()))	

	occupancy, _ 	= np.histogram(variable.values, bins)

	for k in group_value:
		count, bin_edges = np.histogram(group_value[k].values, bins)
		tuning_curves[k] = count
		count = count/occupancy
		tuning_curves[k] = count*variable.rate

	return tuning_curves

def compute_2d_tuning_curves(group, variable, ep, nb_bins, minmax=None):
	"""
	Computes 2 dimensional tuning curves from a 2d feature
	Variable should be a 2 dimensional TsdFrame (e.g. position x and y).
	
	Parameters
	----------
	group: TsGroup or dict of Ts/Tsd objects
	    The group input
	variable: TsdFrame
	    The 2d target variable
	ep: IntervalSet
	    The epoch to perform the operation
	nb_bins: int
	    Number of bins in the tuning curve
	minmax: tuple or list, optional
	    The min and max boundaries of the tuning curves given as:
	    (minx, maxx, miny, maxy)
	    If None, the boundaries is inferred from the target variable
	
	Returns
	-------
	numpy.ndarray
	    Stacked array of the tuning curves with dimensions (n, nb_bins, nb_bins).
	    n is the number of object in the input group. 
	list
		bins center in the two dimensions

	"""
	if type(group) is dict:
		group = nap.TsGroup(group, time_support = ep)

	if variable.shape[1] != 2:
		raise RuntimeError("Variable is not 2 dimensional.")

	cols = list(variable.columns)

	groups_value = {}
	binsxy = {}
	for i, c in enumerate(cols):
		groups_value[c] = group.value_from(variable[c], ep)
		if minmax is None:
			bins = np.linspace(np.min(variable[c]), np.max(variable[c]), nb_bins)
		else:
			bins = np.linspace(minmax[i+i%2], minmax[i+1+i%2], nb_bins)
		binsxy[c] = bins

	occupancy, _, _ = np.histogram2d(
		variable[cols[0]].values, 
		variable[cols[1]].values, 
		[binsxy[cols[0]], binsxy[cols[1]]])

	tc = {}
	for n in group.keys():
		count,_,_ = np.histogram2d(
			groups_value[cols[0]][n].values,
			groups_value[cols[1]][n].values,
			[binsxy[cols[0]], binsxy[cols[1]]]
			)
		count = count / occupancy
		tc[n] = count * variable.rate

	xy = [binsxy[c][0:-1] + np.diff(binsxy[c])/2 for c in binsxy.keys()]
	
	return tc, xy