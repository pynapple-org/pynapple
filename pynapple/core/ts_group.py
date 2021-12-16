import numpy as np
import pandas as pd
import sys
import warnings
from collections import UserDict
from tabulate import tabulate
from .time_series import Ts, Tsd, TsdFrame
from .interval_set import IntervalSet
from .time_units import TimeUnits

def intersect_intervals(i_sets):			
	n_sets = len(i_sets)
	time1 = [i_set['start'] for i_set in i_sets]
	time2 = [i_set['end'] for i_set in i_sets]
	time1.extend(time2)
	time = np.hstack(time1)
	start_end = np.hstack((np.ones(len(time)//2, dtype=np.int32),
						  -1 * np.ones(len(time)//2, dtype=np.int32)))

	df = pd.DataFrame({'time': time, 'start_end': start_end})
	df.sort_values(by='time', inplace=True)
	df.reset_index(inplace=True, drop=True)
	df['cumsum'] = df['start_end'].cumsum()
	ix = (df['cumsum']==n_sets).to_numpy().nonzero()[0]
	return IntervalSet(df['time'][ix], df['time'][ix+1])

def union_intervals(i_sets):
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


class TsGroup(UserDict):
	"""

	"""
	def __init__(self, data=None, time_support=None, time_units='s', **kwargs):
		"""
		"""
		self._initialized = False
		
		index = np.sort(list(data.keys()))

		self._metadata = pd.DataFrame(index=index, columns = ['freq'])
		
		# Transform elements to Ts/Tsd objects
		for k in index:
			if isinstance(data[k], (np.ndarray,list)):
				warnings.warn('Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object.', stacklevel=2)
				data[k] = Ts(t = data[k], time_support = time_support, time_units = time_units)

		# If time_support is passed, all elements of data are restricted prior to init
		if isinstance(time_support, IntervalSet):
			self.time_support = time_support
			data = {k:data[k].restrict(self.time_support) for k in index}
		else:
			# Otherwise do the intersection of all time supports			
			time_support = intersect_intervals([data[k].time_support for k in index])
			if len(time_support) == 0:
				raise RuntimeError("Intersection of time supports is empty. Consider passing a time support as argument.")
			self.time_support = time_support				
			data = {k:data[k].restrict(self.time_support) for k in index}

		UserDict.__init__(self, data)
		
		# Making the TsGroup non mutable
		self._initialized = True

		# Trying to add argument as metainfo
		self.set_info(**kwargs)
		
	"""
	Base functions
	"""
	def __setitem__(self, key, value):
		"""
		Can't overwrite key that already exists
		If true, raise an error
		Can't add an item if TsGroup has been already initialized
		"""
		if self._initialized:
			raise RuntimeError("TsGroup object is not mutable.")
		if self.__contains__(key):
			raise KeyError("Key {} already in group index.".format(key))
		else:
			if isinstance(value, (Ts, Tsd)):
				self._metadata.loc[int(key),'freq'] = value.rate
				super().__setitem__(int(key), value)
			elif isinstance(value, (np.ndarray, list)):
				warnings.warn('Elements should not be passed as numpy array. Default time units is seconds when creating the Ts object.', stacklevel=2)
				tmp = Ts(t = value, time_units = 's')
				self._metadata.loc[int(key),'freq'] = tmp.rate
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
			metadata = self._metadata.loc[key,self._metadata.columns.drop('freq')]
			return TsGroup({k:self[k] for k in key}, **metadata)
	
	def __repr__(self):
		cols = self._metadata.columns.drop('freq')
		headers = ['Index', 'Freq. (Hz)'] + [c for c in cols]		
		lines = []		
	
		for i in self.data.keys():
			lines.append([str(i), '%.2f' % self._metadata.loc[i,'freq']] + [self._metadata.loc[i,c] for c in cols])
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
					raise RuntimeError("Columns needs to be labelled for metadata")
		if len(kwargs):
			for k, v in kwargs.items():	
				self._metadata[k] = v
		return

	def get_info(self, key):
		return self._metadata[key]

	def _union_time_support(self):		
		idx = list(self.data.keys())
		i_sets = [self.data[i].time_support for i in idx]
		return union_intervals(i_sets)

	def _intersect_time_support(self):
		idx = list(self.data.keys())
		i_sets = [self.data[i].time_support for i in idx]
		return intersect_intervals(i_sets)


	"""
	Generic functions of Tsd objects
	"""
	def restrict(self, ep):
		"""
		"""		
		newgr = {}
		for k in self.data:
			newgr[k] = self.data[k].restrict(ep)
		cols = self._metadata.columns.drop('freq')

		return TsGroup(newgr, time_support = ep, **self._metadata[cols])

	def value_from(self, tsd, ep, align='closest'):
		"""
		Assign to each time points the closest value from tsd within ep
		"""
		tsd = tsd.restrict(ep)
		newgr = {}
		for k in self.data:
			newgr[k] = self.data[k].value_from(tsd, ep, align)

		cols = self._metadata.columns.drop('freq')
		return TsGroup(newgr, time_support = ep, **self._metadata[cols])

	def count(self, bin_size, ep = None, time_units = 's'):
		"""
		Count occurences of events within bin size for each item
		bin_size should be seconds unless specified		
		If no epochs is passed, the data will be binned based on the largest merge of time support.
		"""		
		if not isinstance(ep, IntervalSet):
			ep = self._union_time_support()
			
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

		return TsdFrame(t = time_index, d = count, support = ep)
		
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
		idx = np.digitize(self._metadata[key], bins)-1
		groups = self._metadata.index.groupby(idx)
		ix = np.unique(list(groups.keys()))
		ix = ix[ix>=0]
		ix = ix[ix<len(bins)-1]
		xb = bins[0:-1] + np.diff(bins)/2
		sliced = [self[groups[i]] for i in ix]	
		return sliced, xb[ix]			

	def getby_category(self, key):
		"""
		Return a dictionnay of all categories
		"""
		groups = self._metadata.groupby(key).groups
		sliced = {k:self[groups[k]] for k in groups.keys()}
		return sliced

