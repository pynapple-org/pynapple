#!/usr/bin/env python

"""
Class and functions for loading data processed with the Neurosuite (Klusters, Neuroscope, NDmanager)

@author: Guillaume Viejo
"""
import os, sys
import numpy as np
from .. import core as nap
from .loader import BaseLoader


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
		super().__init__(path)

		


