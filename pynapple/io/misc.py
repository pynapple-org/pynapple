import numpy as np
import sys,os
import scipy.io
sys.path.append('..')
import core as nts
import pandas as pd


def loadXML(path):
	"""
	path should be the folder session containing the XML file
	Function returns :
		1. the number of channels
		2. the sampling frequency of the dat file or the eeg file depending of what is present in the folder
			eeg file first if both are present or both are absent
		3. the mappings shanks to channels as a dict
	Args:
		path : string

	Returns:
		int, int, dict
	"""
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	listdir = os.listdir(path)
	xmlfiles = [f for f in listdir if f.endswith('.xml')]
	if not len(xmlfiles):
		print("Folder contains no xml files; Exiting ...")
		sys.exit()
	new_path = os.path.join(path, xmlfiles[0])
	
	from xml.dom import minidom	
	xmldoc 		= minidom.parse(new_path)
	nChannels 	= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
	fs_dat 		= xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
	fs_eeg 		= xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[0].firstChild.data	
	if os.path.splitext(xmlfiles[0])[0] +'.dat' in listdir:
		fs = fs_dat
	elif os.path.splitext(xmlfiles[0])[0] +'.eeg' in listdir:
		fs = fs_eeg
	else:
		fs = fs_eeg
	shank_to_channel = {}
	groups 		= xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[0].getElementsByTagName('group')
	for i in range(len(groups)):
		shank_to_channel[i] = np.sort([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
	return int(nChannels), int(fs), shank_to_channel