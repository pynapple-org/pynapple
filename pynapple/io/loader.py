#!/usr/bin/env python

"""
Class and functions for loading data processed with the Neurosuite (Klusters, Neuroscope, NDmanager)

@author: Guillaume Viejo
"""
import os, sys
import numpy as np
from .. import core as nap
from .loader_gui import BaseLoaderGUI

from PyQt5.QtWidgets import QApplication



class BaseLoader(object):
	"""
	General loader for epochs and tracking data
	"""
	def __init__(self, path=None):
		self.data = None
		self.path = path

		# Check if a pynapplenwb folder exist to bypass GUI
		self.nwb_path = os.path.join(self.path, 'pynapplenwb')
		if os.path.exists(self.nwb_path):
			self.load_data(self.path)

		else:
			# Starting the GUI
			app = QApplication([])
			self.window = BaseLoaderGUI(path=path)
			app.exec()

			# Extracting all the informations from gui loader
			if self.window.status:
				self.session_information = self.window.session_information
				self.epochs = self._make_interval_set(self.window.epochs, self.window.time_units_epochs)
				self.position = None

			# Save the data


	def _make_interval_set(self, epochs, time_units='s'):
		"""
		Split GUI epochs into dict of epochs
		"""
		labels = epochs.groupby("label").groups
		isets = {}
		for l in labels.keys():
			tmp = epochs.loc[labels[l]]
			isets[l] = nap.IntervalSet(start=tmp['start'],end=tmp['end'],time_units=time_units)
		return isets

	def save_data():
		return

	def load_data(path):
		return