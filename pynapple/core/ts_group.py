import numpy as np
import pandas as pd
import sys
from collections import UserDict
from tabulate import tabulate
from .time_series import Ts, Tsd

class TsGroup(UserDict):
	"""

	"""
	def __init__(self, data=None):
		"""
		"""
		UserDict.__init__(self, data)
		self.index = list(data.keys())

		# Copying frequency of Ts object as a separate panda series
		self.rates = pd.Series(index = self.index, data = [self.data[k].rate for k in self.index])
		self.metadata = pd.DataFrame(index = self.index)

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
				print("TODO update rate and index")		
				super().__setitem__(key, value)
			else:
				raise ValueError("Value with key {} is not a Ts/Tsd object.".format(key))

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
			print("TODO pass metadata when slicing")
			return TsGroup({k:self[k] for k in key})
	
	def __repr__(self):
		headers = ['Index', 'Freq. (Hz)']
		lines = []
		for i in self.index:
			lines.append([str(i), '%.2f' % self.rates[i]])
		return tabulate(lines, headers = headers)		
		
	def __str__(self):
		return self.__repr__()

	def keys(self): 
		return self.index

	def items(self): 
		return list(self.data.items())

	def values(self): 
		return list(self.data.values())

	"""
	Metadata 
	"""
	def set_info(self, data):
		pass

	"""
	Generic functions of Tsd objects
	"""
	def restrict(self):
		pass

	def realign(self):
		pass 

	def bin(self):
		pass

	"""
	Special slicing 
	"""
	def getby_threshold(self, thr):
		pass

	def getby_intervals(self, bins):
		pass

	def getby_labels(self, key):
		pass



