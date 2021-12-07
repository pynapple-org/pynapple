import numpy as np
import pandas as pd
import sys
import warnings
from collections import UserDict
from tabulate import tabulate
from .time_series import Ts, Tsd, TsdFrame
from .interval_set import IntervalSet
from .time_units import TimeUnits

class TsGroup(UserDict):
	"""

	"""
	def __init__(self, data=None, *args, **kwargs):
		"""
		"""
		index = np.sort(list(data.keys()))
		self.rates = pd.Series(index=index, dtype=np.float64)
		self._metadata = pd.DataFrame(index=index)
		
		UserDict.__init__(self, data)
		
		# Trying to add argument as metainfo
		self.set_info(*args, **kwargs)
		
	"""
	Base functions
	"""
	def __setitem__(self, key, value):
		"""
		Can't overwrite key that already exists
		If true, raise an error
		"""
		if self.__contains__(key):
			raise KeyError("Key {} already in group index.".format(key))
		else:
			if isinstance(value, (Ts, Tsd)):
				self.rates.loc[int(key)] = value.rate
				self._metadata.loc[int(key)] = np.nan
				super().__setitem__(int(key), value)
			elif isinstance(value, (np.ndarray, list)):
				warnings.warn('Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object.')
				tmp = Ts(t = value, time_units = 's')
				self.rates.loc[int(key)] = tmp.rate
				self._metadata.loc[int(key)] = np.nan
				super().__setitem__(int(key), tmp)
			else:
				raise ValueError("Value with key {} is not an iterable.".format(key))

	def __getitem__(self, key):
		"""
		Can parse group with key or iterables
		"""	
		if key.__hash__:
			if self.__contains__(key):
				return self.data[key]
			else:
				raise KeyError("Can't find key {} in group index.".format(key))
		else:			
			metadata = self._metadata.loc[key]
			return TsGroup({k:self[k] for k in key}, metadata)
	
	def __repr__(self):
		headers = ['Index', 'Freq. (Hz)'] + [c for c in self._metadata.columns]		
		lines = []
		for i in self.data.keys():
			lines.append([str(i), '%.2f' % self.rates[i]] + [self._metadata.loc[i,c] for c in self._metadata.columns])
		return tabulate(lines, headers = headers)		
		
	def __str__(self):
		return self.__repr__()

	def keys(self): 
		return list(self.data.keys())

	def items(self): 
		return list(self.data.items())

	def values(self): 
		return list(self.data.values())

	"""
	Metadata 
	"""
	def set_info(self, *args, **kwargs):		
		if len(args):
			for arg in args:
				if isinstance(arg, pd.DataFrame):					
					self._metadata = self._metadata.join(arg)
				elif isinstance(arg, (pd.Series, np.ndarray)):
					raise("Columns needs to be labelled for metadata")
		if len(kwargs):
			for k, v in kwargs.items():	
				self._metadata[k] = v
		return

	def get_info(self, key):
		return self._metadata[key]

	def _union_time_span(self):		
		idx = list(self.data.keys())
		i_sets = [self.data[i].time_span for i in idx]
		time = np.hstack([i_set['start'] for i_set in i_sets] +
						 [i_set['end'] for i_set in i_sets])
		start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
							  -1 * np.ones(len(time)//2, dtype=np.int32)))
		df = pd.DataFrame({'time': time, 'start_end': start_end})
		df.sort_values(by='time', inplace=True)
		df.reset_index(inplace=True, drop=True)
		df['cumsum'] = df['start_end'].cumsum()
		ix_stop = (df['cumsum']==0).to_numpy().nonzero()[0]
		ix_start = np.hstack((0, ix_stop[:-1]+1))
		return IntervalSet(df['time'][ix_start], df['time'][ix_stop])

	"""
	Generic functions of Tsd objects
	"""
	def restrict(self, iset):
		"""
		"""
		newgr = {}
		for k in self.data:
			newgr[k] = self.data[k].restrict(iset)
		return TsGroup(newgr, self._metadata)

	def realign(self, tsd, align='closest'):
		"""
		"""
		newgr = {}
		for k in self.data:
			newgr[k] = tsd.realign(self.data[k], align)
		return TsGroup(newgr, self._metadata)		

	def count(self, bin_size, ep = None, time_units = 's'):
		"""
		Count occurences of events within bin size for each item
		bin_size should be seconds unless specified		
		If no epochs is passed, the data will be binned based on the largest merge of time span.
		"""		
		if not isinstance(ep, IntervalSet):
			ep = self._union_time_span()
			
		bin_size_us = TimeUnits.format_timestamps(np.array([bin_size]), time_units)[0]

		# bin for each epochs
		time_index = []
		count = []
		for i in ep.index:
			bins = np.arange(ep.start[i], ep.end[i] + bin_size_us, bin_size_us)
			tmp = np.array([np.histogram(self.data[n].index.values, bins)[0] for n in self.keys()])
			count.append(np.transpose(tmp))
			time_index.append(bins[0:-1] + np.diff(bins)//2)

		count = np.vstack(count)
		time_index = np.hstack(time_index)

		return TsdFrame(t = time_index, d = count, span = ep)
		
	"""
	Special slicing of metadata
	"""
	def getby_threshold(self, key, thr, op = '>'):
		"""
		Return TsGroup above/below a threshold
		"""
		if op == '>':
			ix = list(self._metadata.index[self._metadata[key] > thr])
			return self[ix]
		elif op == '<':
			ix = list(self._metadata.index[self._metadata[key] < thr])
			return self[ix]
		elif op == '>=':
			ix = list(self._metadata.index[self._metadata[key] >= thr])
			return self[ix]
		elif op == '<=':
			ix = list(self._metadata.index[self._metadata[key] <= thr])
			return self[ix]
		else:
			raise RuntimeError("Operation {} not recognized.".format(op))


	def getby_intervals(self, key, bins):
		"""
		Return list of TsGroup sliced by bins and center bins
		"""
		idx = np.digitize(self._metadata['alpha'], bins)-1
		groups = self._metadata.index.groupby(idx)
		ix = np.unique(list(groups.keys()))
		ix = ix[ix>=0]
		ix = ix[ix<len(bins)-1]
		xb = bins[0:-1] + np.diff(bins)/2
		sliced = [self[groups[i]] for i in ix]	
		return sliced, xb[ix]			

	def getby_category(self, key):
		"""
		Return a dictionnay of all catergories
		"""
		return self._metadata.groupby(key).groups

