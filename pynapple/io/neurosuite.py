#!/usr/bin/env python

"""
Class and functions for loading data processed with the Neurosuite (Klusters, Neuroscope, NDmanager)

@author: Guillaume Viejo
"""
import os, sys
import numpy as np
from .. import core as nap
from .loader import BaseLoader
import pandas as pd

class NeuroSuite(BaseLoader):
	"""
	Loader for kluster data
	"""
	def __init__(self, path):
		"""Summary
		
		Parameters
		----------
		path : TYPE
		    Description
		"""		
		self.basename = os.path.basename(path)
		self.time_support = None
		
		super().__init__(path)		

		self.load_spikes(path, self.basename, self.time_support)

	def load_spikes(self,path, basename, time_support, fs = 20000.0):
		"""Summary
		
		Parameters
		----------
		path : TYPE
		    Description
		
		Returns
		-------
		TYPE
		    Description
		"""
		files = os.listdir(path)
		clu_files     = np.sort([f for f in files if '.clu.' in f and f[0] != '.'])
		res_files     = np.sort([f for f in files if '.res.' in f and f[0] != '.'])
		clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
		clu2         = np.sort([int(f.split(".")[-1]) for f in res_files])
		if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
			raise RuntimeError("Not the same number of clu and res files in "+path+"; Exiting ...")

		count = 0
		spikes = {}
		group = pd.Series(dtype=np.int32)
		for i, s in zip(range(len(clu_files)),clu1):
			clu = np.genfromtxt(os.path.join(path,basename+'.clu.'+str(s)),dtype=np.int32)[1:]
			if np.max(clu)>1: # getting rid of mua and noise
				res = np.genfromtxt(os.path.join(path,basename+'.res.'+str(s)))
				tmp = np.unique(clu).astype(int)
				idx_clu = tmp[tmp>1]
				idx_out = np.arange(count, count+len(idx_clu))

				for j,k in zip(idx_clu, idx_out):
					t = res[clu==j]/fs
					spikes[k] = nap.Ts(t=t, time_units='s')#, time_support=time_support)
					group.loc[k] = s

				count+=len(idx_clu)

		self.spikes = nap.TsGroup(
			spikes, 
			time_support=time_support,
			time_units='s',
			group=group)

		return

		


