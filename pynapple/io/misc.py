#!/usr/bin/env python

"""
Various io functions

@author: Guillaume Viejo
"""
import sys, os
from .neurosuite import NeuroSuite
from .loader import BaseLoader

def load_session(path=None, session_type=None):
	"""
	General Loader for Neurosuite, Phy or default

	Parameters
	----------
	path : None, optional
	    Description
	session_type : str, optional
	    Description
	
	Returns
	-------
	TYPE
	    Description
	
	"""
	if path:
		if not os.path.isdir(path):
			raise RuntimeError("Path {} is not found.".format(path))	

	if session_type == 'neurosuite':
		return NeuroSuite(path)
	# elif session_type == 'phy':
	# 	return NeuroSuite(path)
	else:
		return BaseLoader(path)