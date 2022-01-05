import sys, os
import numpy as np
import scipy.io
import pandas as pd
import scipy.signal
from .. import core as nap



'''
Wrappers should be able to distinguish between raw data or matlab processed data

TODO:
    load/write NWB
    

'''



## THIS SHOULD BE DISCUSSED.
## SEEMS TO COMPLICATED
def loadSpikeData(path, index=None, fs = 20000):
	"""
	if the path contains a folder named /Analysis, 
	the script will look into it to load either
		- SpikeData.mat saved from matlab
		- SpikeData.h5 saved from this same script
	if not, the res and clu file will be loaded 
	and an /Analysis folder will be created to save the data
	Thus, the next loading of spike times will be faster
	Notes :
		If the frequency is not givne, it's assumed 20kH
	Args:
		path : string

	Returns:
		dict, array    
	"""    
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if os.path.exists(new_path):
		new_path    = os.path.join(path, 'Analysis/')
		files        = os.listdir(new_path)
		if 'SpikeData.mat' in files:
			spikedata     = scipy.io.loadmat(new_path+'SpikeData.mat')
			shank         = spikedata['shank'] - 1
			if index is None:
				shankIndex     = np.arange(len(shank))
			else:
				shankIndex     = np.where(shank == index)[0]
			spikes         = {}    
			for i in shankIndex:    
				spikes[i]     = nap.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')
			a             = spikes[0].as_units('s').index.values    
			if ((a[-1]-a[0])/60.)/60. > 20. : # VERY BAD        
				spikes         = {}    
				for i in shankIndex:
					spikes[i]     = nap.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2]*0.0001, time_units = 's')
			return spikes, shank
		elif 'SpikeData.h5' in files:            
			final_path = os.path.join(new_path, 'SpikeData.h5')            
			try:
				spikes = pd.read_hdf(final_path, mode='r')
				# Returning a dictionnary | can be changed to return a dataframe
				toreturn = {}
				for i,j in spikes:
					toreturn[j] = nap.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')
				shank = spikes.columns.get_level_values(0).values[:,np.newaxis]
				return toreturn, shank
			except:
				spikes = pd.HDFStore(final_path, 'r')
				shanks = spikes['/shanks']
				toreturn = {}
				for j in shanks.index:
					toreturn[j] = nap.Ts(spikes['/spikes/s'+str(j)])
				#shank = shanks.values
				spikes.close()
				del spikes
				return nap.TsGroup(toreturn, group=shanks)
			
		else:            
			print("Couldn't find any SpikeData file in "+new_path)
			print("If clu and res files are present in "+path+", a SpikeData.h5 is going to be created")

	# Creating /Analysis/ Folder here if not already present
	if not os.path.exists(new_path): os.makedirs(new_path)
	files = os.listdir(path)
	clu_files     = np.sort([f for f in files if '.clu.' in f and f[0] != '.'])
	res_files     = np.sort([f for f in files if '.res.' in f and f[0] != '.'])
	clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
	clu2         = np.sort([int(f.split(".")[-1]) for f in res_files])
	if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
		print("Not the same number of clu and res files in "+path+"; Exiting ...")
		sys.exit()
	count = 0
	spikes = []
	basename = clu_files[0].split(".")[0]
	for i, s in zip(range(len(clu_files)),clu1):
		clu = np.genfromtxt(os.path.join(path,basename+'.clu.'+str(s)),dtype=np.int32)[1:]
		if np.max(clu)>1:
			# print(i,s)
			res = np.genfromtxt(os.path.join(path,basename+'.res.'+str(s)))
			tmp = np.unique(clu).astype(int)
			idx_clu = tmp[tmp>1]
			idx_col = np.arange(count, count+len(idx_clu))	        
			tmp = pd.DataFrame(index = np.unique(res)/fs,
								columns = pd.MultiIndex.from_product([[s],idx_col]),
								data = 0, 
								dtype = np.uint16)
			for j, k in zip(idx_clu, idx_col):
				tmp.loc[res[clu==j]/fs,(s,k)] = np.uint16(k+1)
			spikes.append(tmp)
			count+=len(idx_clu)

			# tmp2 = pd.DataFrame(index=res[clu==j]/fs, data = k+1, ))
			# spikes = pd.concat([spikes, tmp2], axis = 1)


	# Returning a dictionnary
	toreturn =  {}
	shank = []
	for s in spikes:
		shank.append(s.columns.get_level_values(0).values)
		sh = np.unique(shank[-1])[0]
		for i,j in s:
			toreturn[j] = nap.Ts(t=s[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')

	del spikes
	shank = np.hstack(shank)
	shank = pd.Series(index = list(toreturn.keys()), data = shank)

	final_path = os.path.join(new_path, 'SpikeData.h5')
	store = pd.HDFStore(final_path)
	for s in toreturn.keys():
		store.put('spikes/s'+str(s), toreturn[s].as_series())
	store.put('shanks', shank)
	store.close()

	# OLD WAY
	# spikes = pd.concat(spikes, axis = 1)
	# spikes = spikes.fillna(0)
	# spikes = spikes.astype(np.uint16)

	# Saving SpikeData.h5
	# final_path = os.path.join(new_path, 'SpikeData.h5')
	# spikes.columns.set_names(['shank', 'neuron'], inplace=True)    
	# spikes.to_hdf(final_path, key='spikes', mode='w')

	# Returning a dictionnary
	# toreturn = {}
	# for i,j in spikes:
	# 	toreturn[j] = nap.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')

	# shank = spikes.columns.get_level_values(0).values[:,np.newaxis].flatten()

	return nap.TsGroup(toreturn, group = shank)

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

def makeEpochs(path, order, file = None, start=None, end = None, time_units = 's'):
	"""
	The pre-processing pipeline should spit out a csv file containing all the successive epoch of sleep/wake
	This function will load the csv and write neuroseries.IntervalSet of wake and sleep in /Analysis/BehavEpochs.h5
	If no csv exists, it's still possible to give by hand the start and end of the epochs
	Notes:
		The function assumes no header on the csv file
	Args:
		path: string
		order: list
		file: string
		start: list/array (optional)
		end: list/array (optional)
		time_units: string (optional)
	Return: 
		none
	"""		
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()	
	if file:
		listdir 	= os.listdir(path)	
		if file not in listdir:
			print("The file "+file+" cannot be found in the path "+path)
			sys.exit()			
		filepath 	= os.path.join(path, file)
		epochs 		= pd.read_csv(filepath, header = None)
	elif file is None and len(start) and len(end):
		epochs = pd.DataFrame(np.vstack((start, end)).T)
	elif file is None and start is None and end is None:
		print("You have to specify either a file or arrays of start and end; Exiting ...")
		sys.exit()
	
	# Creating /Analysis/ Folder here if not already present
	new_path	= os.path.join(path, 'Analysis/')
	if not os.path.exists(new_path): os.makedirs(new_path)
	# Writing to BehavEpochs.h5
	new_file 	= os.path.join(new_path, 'BehavEpochs.h5')
	store 		= pd.HDFStore(new_file, 'a')
	epoch 		= np.unique(order)
	for i, n in enumerate(epoch):
		idx = np.where(np.array(order) == n)[0]
		ep = nap.IntervalSet(start = epochs.loc[idx,0],
							end = epochs.loc[idx,1],
							time_units = time_units)
		store[n] = pd.DataFrame(ep)
	store.close()

	return None

def makePositions(path, file_order, episodes, n_channels=1, trackchannel=0, names = ['ry', 'rx', 'rz', 'x', 'y', 'z'], update_wake_epoch = True):
	"""
	Assuming that makeEpochs has been runned and a file BehavEpochs.h5 can be 
	found in /Analysis/, this function will look into path  for analogin file 
	containing the TTL pulses. The position time for all events will thus be
	updated and saved in Analysis/Position.h5.
	BehavEpochs.h5 will although be updated to match the time between optitrack
	and intan
	
	Notes:
		The function assumes headers on the csv file of the position in the following order:
			['ry', 'rx', 'rz', 'x', 'y', 'z']
	Args:
		path: string
		file_order: list
		names: list
	Return: 
		None
	""" 
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	files = os.listdir(path)
	for f in file_order:
		if not np.any([f+'.csv' in g for g in files]):
			print("Could not find "+f+'.csv; Exiting ...')
			sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if not os.path.exists(new_path): os.makedirs(new_path)                
	file_epoch = os.path.join(path, 'Analysis', 'BehavEpochs.h5')
	if os.path.exists(file_epoch):
		wake_ep = loadEpoch(path, 'wake')
	else:
		makeEpochs(path, episodes, file = 'Epoch_TS.csv')
		wake_ep = loadEpoch(path, 'wake')
	if len(wake_ep) != len(file_order):
		print("Number of wake episodes doesn't match; Exiting...")
		sys.exit()

	frames = []
	others = []

	for i, f in enumerate(file_order):
		print(i, f)
		csv_file = os.path.join(path, "".join(s for s in files if f+'.csv' in s))
		position = pd.read_csv(csv_file, header = [4,5], index_col = 1)
		if 1 in position.columns:
			position = position.drop(labels = 1, axis = 1)
		position = position[~position.index.duplicated(keep='first')]
		analogin_file = os.path.splitext(csv_file)[0]+'_analogin.dat'
		if not os.path.split(analogin_file)[1] in files:
			print("No analogin.dat file found.")
			print("Please provide it as "+os.path.split(analogin_file)[1])
			print("Exiting ...")
			sys.exit()
		else:
			ttl = loadTTLPulse(analogin_file, n_channels, trackchannel)
		
		if len(ttl):
			length = np.minimum(len(ttl), len(position))
			ttl = ttl.iloc[0:length]
			position = position.iloc[0:length]
			time_offset = wake_ep.as_units('s').iloc[i,0] + ttl.index[0]
		else:
			print("No ttl for ", i, f)
			time_offset = wake_ep.as_units('s').iloc[i,0]
		
		position.index += time_offset
		wake_ep.iloc[i,0] = np.int64(np.maximum(wake_ep.as_units('s').iloc[i,0], position.index[0])*1e6)
		wake_ep.iloc[i,1] = np.int64(np.minimum(wake_ep.as_units('s').iloc[i,1], position.index[-1])*1e6)

		if len(position.columns) > 6:
			frames.append(position.iloc[:,0:6])
			others.append(position.iloc[:,6:])
		else:
			frames.append(position)
	
	position = pd.concat(frames)
	#position = nap.TsdFrame(t = position.index.values, d = position.values, time_units = 's', columns = names)
	position.columns = names
	position[['ry', 'rx', 'rz']] *= (np.pi/180)
	position[['ry', 'rx', 'rz']] += 2*np.pi
	position[['ry', 'rx', 'rz']] %= 2*np.pi
	
	if len(others):
		others = pd.concat(others)
		others.columns = names
		others[['ry', 'rx', 'rz']] *= (np.pi/180)
		others[['ry', 'rx', 'rz']] += 2*np.pi
		others[['ry', 'rx', 'rz']] %= 2*np.pi

	if update_wake_epoch:
		store = pd.HDFStore(file_epoch, 'a')
		store['wake'] = pd.DataFrame(wake_ep)
		store.close()
	
	position_file = os.path.join(path, 'Analysis', 'Position.h5')
	store = pd.HDFStore(position_file, 'w')
	store['position'] = position
	store.close()
	
	if len(others):
		walls_file = os.path.join(path, 'Analysis', 'Walls.h5')
		store = pd.HDFStore(walls_file, 'w')
		store['position'] = others
		store.close()

	return

def loadEpoch(path, epoch, episodes = None):
	"""
	load the epoch contained in path	
	If the path contains a folder analysis, the function will load either the BehavEpochs.mat or the BehavEpochs.h5
	Run makeEpochs(data_directory, ['sleep', 'wake', 'sleep', 'wake'], file='Epoch_TS.csv') to create the BehavEpochs.h5

	Args:
		path: string
		epoch: string

	Returns:
		neuroseries.IntervalSet
	"""			
	if not os.path.exists(path): # Check for path
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	if epoch in ['sws', 'rem']: 		
		# loading the .epoch.evt file
		file = os.path.join(path,os.path.basename(path)+'.'+epoch+'.evt')
		if os.path.exists(file):
			tmp = np.genfromtxt(file)[:,0]
			tmp = tmp.reshape(len(tmp)//2,2)/1000
			ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
			# TO make sure it's only in sleep since using the TheStateEditor
			sleep_ep = loadEpoch(path, 'sleep')
			ep = sleep_ep.intersect(ep)
			return ep
		else:
			print("The file ", file, "does not exist; Exiting ...")
			sys.exit()
	elif epoch == 'wake.evt.theta':
		file = os.path.join(path,os.path.basename(path)+'.'+epoch)
		if os.path.exists(file):
			tmp = np.genfromtxt(file)[:,0]
			tmp = tmp.reshape(len(tmp)//2,2)/1000
			ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
			return ep
		else:
			print("The file ", file, "does not exist; Exiting ...")
	filepath 	= os.path.join(path, 'Analysis')
	if os.path.exists(filepath): # Check for path/Analysis/	
		listdir		= os.listdir(filepath)
		file 		= [f for f in listdir if 'BehavEpochs' in f]
	if len(file) == 0: # Running makeEpochs		
		makeEpochs(path, episodes, file = 'Epoch_TS.csv')
		listdir		= os.listdir(filepath)
		file 		= [f for f in listdir if 'BehavEpochs' in f]
	if file[0] == 'BehavEpochs.h5':
		new_file = os.path.join(filepath, 'BehavEpochs.h5')
		store 		= pd.HDFStore(new_file, 'r')
		if '/'+epoch in store.keys():
			ep = store[epoch]
			store.close()
			return nap.IntervalSet(ep)
		else:
			print("The file BehavEpochs.h5 does not contain the key "+epoch+"; Exiting ...")
			sys.exit()
	elif file[0] == 'BehavEpochs.mat':
		behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))
		if epoch == 'wake':
			wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
			return nap.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
		elif epoch == 'sleep':
			sleep_pre_ep, sleep_post_ep = [], []
			if 'sleepPreEp' in behepochs.keys():
				sleep_pre_ep = behepochs['sleepPreEp'][0][0]
				sleep_pre_ep = np.hstack([sleep_pre_ep[1],sleep_pre_ep[2]])
				sleep_pre_ep_index = behepochs['sleepPreEpIx'][0]
			if 'sleepPostEp' in behepochs.keys():
				sleep_post_ep = behepochs['sleepPostEp'][0][0]
				sleep_post_ep = np.hstack([sleep_post_ep[1],sleep_post_ep[2]])
				sleep_post_ep_index = behepochs['sleepPostEpIx'][0]
			if len(sleep_pre_ep) and len(sleep_post_ep):
				sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
			elif len(sleep_pre_ep):
				sleep_ep = sleep_pre_ep
			elif len(sleep_post_ep):
				sleep_ep = sleep_post_ep						
			return nap.IntervalSet(sleep_ep[:,0], sleep_ep[:,1], time_units = 's')
		###################################
		# WORKS ONLY FOR MATLAB FROM HERE #
		###################################		
		elif epoch == 'sws':
			sampling_freq = 1250
			new_listdir = os.listdir(path)
			for f in new_listdir:
				if 'sts.SWS' in f:
					sws = np.genfromtxt(os.path.join(path,f))/float(sampling_freq)
					return nap.IntervalSet.drop_short_intervals(nap.IntervalSet(sws[:,0], sws[:,1], time_units = 's'), 0.0)

				elif '-states.mat' in f:
					sws = scipy.io.loadmat(os.path.join(path,f))['states'][0]
					index = np.logical_or(sws == 2, sws == 3)*1.0
					index = index[1:] - index[0:-1]
					start = np.where(index == 1)[0]+1
					stop = np.where(index == -1)[0]
					return nap.IntervalSet.drop_short_intervals(nap.IntervalSet(start, stop, time_units = 's', expect_fix=True), 0.0)

		elif epoch == 'rem':
			sampling_freq = 1250
			new_listdir = os.listdir(path)
			for f in new_listdir:
				if 'sts.REM' in f:
					rem = np.genfromtxt(os.path.join(path,f))/float(sampling_freq)
					return nap.IntervalSet(rem[:,0], rem[:,1], time_units = 's').drop_short_intervals(0.0)

				elif '-states/m' in listdir:
					rem = scipy.io.loadmat(path+f)['states'][0]
					index = (rem == 5)*1.0
					index = index[1:] - index[0:-1]
					start = np.where(index == 1)[0]+1
					stop = np.where(index == -1)[0]
					return nap.IntervalSet(start, stop, time_units = 's', expect_fix=True).drop_short_intervals(0.0)

def loadPosition(path, events = None, episodes = None, n_channels=1,trackchannel=0):
	"""
	load the position contained in /Analysis/Position.h5

	Notes:
		The order of the columns is assumed to be
			['ry', 'rx', 'rz', 'x', 'y', 'z']
	Args:
		path: string
		
	Returns:
		neuroseries.TsdFrame
	"""        
	if not os.path.exists(path): # Checking for path
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	new_path = os.path.join(path, 'Analysis')
	if not os.path.exists(new_path): os.mkdir(new_path)
	file = os.path.join(path, 'Analysis', 'Position.h5')
	if not os.path.exists(file):
		makePositions(path, events, episodes, n_channels, trackchannel)
	if os.path.exists(file):
		store = pd.HDFStore(file, 'r')
		position = store['position']
		store.close()
		position = nap.TsdFrame(t = position.index.values, d = position.values, columns = position.columns, time_units = 's')
		return position
	else:
		print("Cannot find "+file+" for loading position")
		sys.exit()    	

def loadTTLPulse(file, n_channels = 1, channel = 0, fs = 20000):
	"""
		load ttl from analogin.dat
	"""
	f = open(file, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 2        
	n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
	f.close()
	with open(file, 'rb') as f:
		data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
	if n_channels == 1:
		data = data.flatten().astype(np.int32)
	else:
		data = data[:,channel].flatten().astype(np.int32)
	peaks,_ = scipy.signal.find_peaks(np.diff(data), height=30000)
	timestep = np.arange(0, len(data))/fs
	# analogin = pd.Series(index = timestep, data = data)
	peaks+=1
	ttl = pd.Series(index = timestep[peaks], data = data[peaks])    
	return ttl

def loadAuxiliary(path, n_probe = 1, fs = 20000):
	"""
	Extract the acceleration from the auxiliary.dat for each epochs
	Downsampled at 100 Hz
	Args:
		path: string
		epochs_ids: list        
	Return: 
		TsdArray
	""" 	
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	if 'Acceleration.h5' in os.listdir(os.path.join(path, 'Analysis')):
		accel_file = os.path.join(path, 'Analysis', 'Acceleration.h5')
		store = pd.HDFStore(accel_file, 'r')
		accel = store['acceleration'] 
		store.close()
		accel = nap.TsdFrame(t = accel.index.values*1e6, d = accel.values) 
		return accel
	else:
		aux_files = np.sort([f for f in os.listdir(path) if 'auxiliary' in f])
		if len(aux_files)==0:
			print("Could not find "+f+'_auxiliary.dat; Exiting ...')
			sys.exit()

		accel = []
		sample_size = []
		for i, f in enumerate(aux_files):
			new_path 	= os.path.join(path, f)
			f 			= open(new_path, 'rb')
			startoffile = f.seek(0, 0)
			endoffile 	= f.seek(0, 2)
			bytes_size 	= 2
			n_samples 	= int((endoffile-startoffile)/(3*n_probe)/bytes_size)
			duration 	= n_samples/fs		
			f.close()
			tmp 		= np.fromfile(open(new_path, 'rb'), np.uint16).reshape(n_samples,3*n_probe)
			accel.append(tmp)
			sample_size.append(n_samples)
			del tmp

		accel = np.concatenate(accel)	
		factor = 37.4e-6
		# timestep = np.arange(0, len(accel))/fs
		# accel = pd.DataFrame(index = timestep, data= accel*37.4e-6)
		tmp  = []
		for i in range(accel.shape[1]):
			tmp.append(scipy.signal.resample_poly(accel[:,i]*factor, 1, 100))
		tmp = np.vstack(tmp).T
		timestep = np.arange(0, len(tmp))/(fs/100)
		tmp = pd.DataFrame(index = timestep, data = tmp)
		accel_file = os.path.join(path, 'Analysis', 'Acceleration.h5')
		store = pd.HDFStore(accel_file, 'w')
		store['acceleration'] = tmp
		store.close()
		accel = nap.TsdFrame(t = tmp.index.values*1e6, d = tmp.values) 
		return accel

def downsampleDatFile(path, n_channels = 32, fs = 20000):
	"""
	downsample .dat file to .eeg 1/16 (20000 -> 1250 Hz)
	
	Since .dat file can be very big, the strategy is to load one channel at the time,
	downsample it, and free the memory.

	Args:
		path: string
		n_channel: int
		fs: int
	Return: 
		none
	"""	
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	listdir 	= os.listdir(path)
	datfile 	= os.path.basename(path) + '.dat'
	if datfile not in listdir:
		print("Folder contains no " + datfile + " file; Exiting ...")
		sys.exit()

	new_path = os.path.join(path, datfile)

	f 			= open(new_path, 'rb')
	startoffile = f.seek(0, 0)
	endoffile 	= f.seek(0, 2)
	bytes_size 	= 2
	n_samples 	= int((endoffile-startoffile)/n_channels/bytes_size)
	duration 	= n_samples/fs
	f.close()

	chunksize 	= 200000000
	eeg 		= np.zeros((int(n_samples/16),n_channels), dtype = np.int16)

	for n in range(n_channels):
		print("Ch ", n)
		# Loading		
		rawchannel = np.zeros(n_samples, np.int16)
		count = 0
		while count < n_samples:
			f 			= open(new_path, 'rb')
			seekstart 	= count*n_channels*bytes_size
			f.seek(seekstart)
			block 		= np.fromfile(f, np.int16, n_channels*np.minimum(chunksize, n_samples-count))
			f.close()
			block 		= block.reshape(np.minimum(chunksize, n_samples-count), n_channels)
			rawchannel[count:count+np.minimum(chunksize, n_samples-count)] = np.copy(block[:,n])
			count 		+= chunksize
		# Downsampling		
		eeg[:,n] 	= scipy.signal.resample_poly(rawchannel, 1, 16).astype(np.int16)
		del rawchannel
	
	# Saving
	eeg_path 	= os.path.join(path, os.path.splitext(datfile)[0]+'.eeg')
	with open(eeg_path, 'wb') as f:
		eeg.tofile(f)
		
	return

def loadUFOs(path):
	"""
	Name of the file should end with .evt.py.ufo
	"""
	import os
	name = path.split("/")[-1]
	files = os.listdir(path)
	filename = os.path.join(path, name+'.evt.py.ufo')
	if name+'.evt.py.ufo' in files:
		tmp = np.genfromtxt(path + '/' + name + '.evt.py.ufo')[:,0]
		ripples = tmp.reshape(len(tmp)//3,3)/1000
	else:
		print("No ufo in ", path)
		sys.exit()
	return (nap.IntervalSet(ripples[:,0], ripples[:,2], time_units = 's'), 
			nap.Ts(ripples[:,1], time_units = 's'))

def loadRipples(path):
	"""
	Name of the file should end with .evt.py.rip
	"""
	import os
	name = path.split("/")[-1]
	files = os.listdir(path)
	filename = os.path.join(path, name+'.evt.py.rip')
	if name+'.evt.py.rip' in files:
		tmp = np.genfromtxt(path + '/' + name + '.evt.py.rip')[:,0]
		ripples = tmp.reshape(len(tmp)//3,3)/1000
	else:
		print("No ripples in ", path)
		sys.exit()
	return (nap.IntervalSet(ripples[:,0], ripples[:,2], time_units = 's'), 
			nap.Ts(ripples[:,1], time_units = 's'))



def loadMeanWaveforms(path):
	"""
	load waveforms
	quick and dirty	
	"""
	import scipy.io
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if os.path.exists(new_path):
		new_path    = os.path.join(path, 'Analysis/')
		files        = os.listdir(new_path)
		if 'SpikeWaveF.mat' in files:			
			# data = scipy.io.loadmat(path+'/Analysis/SpikeWaveF.mat')
			# meanWaveF = data['meanWaveF'][0]
			# maxIx = data['maxIx'][0]
			# generalinfo 	= scipy.io.loadmat(path+'/Analysis/GeneralInfo.mat')
			# shankStructure 	= loadShankStructure(generalinfo)
			# spikes,shank = loadSpikeData(path+'/Analysis/SpikeData.mat', shankStructure['thalamus'])	
			# index_neurons = [path.split("/")[-1]+"_"+str(n) for n in spikes.keys()]
			# for i, n in zip(list(spikes.keys()), index_neurons):	
			# 	to_return[n] = meanWaveF[i][maxIx[i]-1]			
			print("to test matlab")
			return
		elif "MeanWaveForms.h5" in files and "MaxWaveForms.h5" in files:
			meanwavef = pd.read_hdf(os.path.join(new_path, 'MeanWaveForms.h5'))
			maxch = pd.read_hdf(os.path.join(new_path, 'MaxWaveForms.h5'))
			return meanwavef, maxch

	# Creating /Analysis/ Folder here if not already present
	if not os.path.exists(new_path): os.makedirs(new_path)
	files = os.listdir(path)
	clu_files     = np.sort([f for f in files if 'clu' in f and f[0] != '.'])	
	spk_files	  = np.sort([f for f in files if 'spk' in f and f[0] != '.'])
	clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
	clu2         = np.sort([int(f.split(".")[-1]) for f in spk_files])
	if len(clu_files) != len(spk_files) or not (clu1 == clu2).any():
		print("Not the same number of clu and res files in "+path+"; Exiting ...")
		sys.exit()	

	# XML INFO
	n_channels, fs, shank_to_channel 	= loadXML(path)
	from xml.dom import minidom	
	xmlfile = os.path.join(path, [f for f in files if f.endswith('.xml')][0])
	xmldoc 		= minidom.parse(xmlfile)
	nSamples 	= int(xmldoc.getElementsByTagName('nSamples')[0].firstChild.data) # assuming constant nSamples

	import xml.etree.ElementTree as ET
	root = ET.parse(xmlfile).getroot()


	count = 0
	meanwavef = []
	maxch = []
	for i, s in zip(range(len(clu_files)),clu1):
		clu = np.genfromtxt(os.path.join(path,clu_files[i]),dtype=np.int32)[1:]
		mwf = []
		mch = []		
		if np.max(clu)>1:
			# load waveforms
			file = os.path.join(path, spk_files[i])
			f = open(file, 'rb')
			startoffile = f.seek(0, 0)
			endoffile = f.seek(0, 2)
			bytes_size = 2
			n_samples = int((endoffile-startoffile)/bytes_size)
			f.close()			
			n_channel = len(root.findall('spikeDetection/channelGroups/group')[s-1].findall('channels')[0])

			data = np.memmap(file, np.int16, 'r', shape = (len(clu), nSamples, n_channel))

			#data = np.fromfile(open(file, 'rb'), np.int16)
			#data = data.reshape(len(clu),nSamples,n_channel)

			tmp = np.unique(clu).astype(int)
			idx_clu = tmp[tmp>1]
			idx_col = np.arange(count, count+len(idx_clu))	        
			for j,k in zip(idx_clu, idx_col):
				# take only a subsample of spike if too big				
				idx = np.sort(np.random.choice(np.where(clu==j)[0], 5000))
				meanw = data[idx,:,:].mean(0)
				ch = np.argmax(np.max(np.abs(meanw), 0))
				mwf.append(meanw.flatten())
				mch.append(ch)
			mwf = pd.DataFrame(np.array(mwf).T)
			mwf.columns = pd.Index(idx_col)
			mch = pd.Series(index = idx_col, data = mch)
			count += len(idx_clu)
			meanwavef.append(mwf)
			maxch.append(mch)

	meanwavef = pd.concat(meanwavef, 1)
	maxch = pd.concat(maxch)	
	meanwavef.to_hdf(os.path.join(new_path, 'MeanWaveForms.h5'), key='waveforms', mode='w')
	maxch.to_hdf(os.path.join(new_path, 'MaxWaveForms.h5'), key='channel', mode='w')
	return meanwavef, maxch

# def loadNeuronWaveform(path, index):
# 	"""
# 	load waveforms
# 	quick and dirty	
# 	"""
# 	import scipy.io
# 	if not os.path.exists(path):
# 		print("The path "+path+" doesn't exist; Exiting ...")
# 		sys.exit()    

# 	# Creating /Analysis/ Folder here if not already present	
# 	files = os.listdir(path)
# 	clu_files     = np.sort([f for f in files if 'clu' in f and f[0] != '.'])	
# 	spk_files	  = np.sort([f for f in files if 'spk' in f and f[0] != '.'])
# 	clu1         = np.sort([int(f.split(".")[-1]) for f in clu_files])
# 	clu2         = np.sort([int(f.split(".")[-1]) for f in spk_files])
# 	if len(clu_files) != len(spk_files) or not (clu1 == clu2).any():
# 		print("Not the same number of clu and res files in "+path+"; Exiting ...")
# 		sys.exit()	

# 	# XML INFO
# 	n_channels, fs, shank_to_channel 	= loadXML(path)
# 	from xml.dom import minidom	
# 	xmlfile = os.path.join(path, [f for f in files if f.endswith('.xml')][0])
# 	xmldoc 		= minidom.parse(xmlfile)
# 	nSamples 	= int(xmldoc.getElementsByTagName('nSamples')[0].firstChild.data) # assuming constant nSamples

# 	import xml.etree.ElementTree as ET
# 	root = ET.parse(xmlfile).getroot()


# 	count = 0
# 	meanwavef = []
# 	maxch = []
# 	for i, s in zip(range(len(clu_files)),clu1):
# 		clu = np.genfromtxt(os.path.join(path,clu_files[i]),dtype=np.int32)[1:]
# 		mwf = []
# 		mch = []
# 		if np.max(clu)>1:
# 			# load waveforms
# 			file = os.path.join(path, spk_files[i])
# 			f = open(file, 'rb')
# 			startoffile = f.seek(0, 0)
# 			endoffile = f.seek(0, 2)
# 			bytes_size = 2
# 			n_samples = int((endoffile-startoffile)/bytes_size)
# 			f.close()			
# 			n_channel = len(root.findall('spikeDetection/channelGroups/group')[s-1].findall('channels')[0])
# 			data = np.fromfile(open(file, 'rb'), np.int16)
# 			data = data.reshape(len(clu),nSamples,n_channel)
# 			tmp = np.unique(clu).astype(int)
# 			idx_clu = tmp[tmp>1]
# 			idx_col = np.arange(count, count+len(idx_clu))	        
# 			for j,k in zip(idx_clu, idx_col):
# 				meanw = data[clu==j].mean(0)
# 				ch = np.argmax(np.max(np.abs(meanw), 0))
# 				mwf.append(meanw.flatten())
# 				mch.append(ch)
# 			mwf = pd.DataFrame(np.array(mwf).T)
# 			mwf.columns = pd.Index(idx_col)
# 			mch = pd.Series(index = idx_col, data = mch)
# 			count += len(idx_clu)
# 			meanwavef.append(mwf)
# 			maxch.append(mch)

# 	meanwavef = pd.concat(meanwavef, 1)
# 	maxch = pd.concat(maxch)	
# 	meanwavef.to_hdf(os.path.join(new_path, 'MeanWaveForms.h5'), key='waveforms', mode='w')
# 	maxch.to_hdf(os.path.join(new_path, 'MaxWaveForms.h5'), key='channel', mode='w')
# 	return meanwavef, maxch


def loadOptoEp(path, epoch, n_channels = 2, channel = 0, fs = 20000):
	"""
		load ttl from analogin.dat
	"""
	files = os.listdir(os.path.join(path, 'Analysis'))
	if 'OptoEpochs.h5' in files:
		new_file = os.path.join(path, 'Analysis/OptoEpochs.h5')
		opto_ep = pd.read_hdf(new_file)
		return nap.IntervalSet(opto_ep)
	else:
		files = os.listdir(path)
		afile = os.path.join(path, [f for f in files if '_'+str(epoch)+'_' in f][0])
		f = open(afile, 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2        
		n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
		f.close()
		with open(afile, 'rb') as f:
			data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
		data = data[:,channel].flatten().astype(np.int32)

		start,_ = scipy.signal.find_peaks(np.diff(data), height=3000)
		end,_ = scipy.signal.find_peaks(np.diff(data)*-1, height=3000)
		start -= 1
		timestep = np.arange(0, len(data))/fs
		# aliging based on epoch_TS.csv
		epochs = pd.read_csv(os.path.join(path, 'Epoch_TS.csv'), header = None)
		timestep = timestep + epochs.loc[epoch,0]
		opto_ep = nap.IntervalSet(start = timestep[start], end = timestep[end], time_units = 's')
		#pd.DataFrame(opto_ep).to_hdf(os.path.join(path, 'Analysis/OptoEpochs.h5'), 'opto')
		return opto_ep	

##########################################################################################################
# TODO
##########################################################################################################

def loadShankStructure(generalinfo):
	"""
	load Shank Structure from dictionnary 
	Only useful for matlab now
	Note : 
		TODO for raw data. 

	Args:
		generalinfo : dict        

	Returns: dict		    
	"""
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]-1
		else :
			shankStructure[k[0]] = []
	
	return shankStructure	

def loadShankMapping(path):		
	spikedata = scipy.io.loadmat(path)
	shank = spikedata['shank']
	return shank



def loadHDCellInfo(path, index):
	"""
	load the session_id_HDCells.mat file that contains the index of the HD neurons
	Only useful for matlab now
	Note : 
		TODO for raw data. 

	Args:
		generalinfo : string, array

	Returns:
		array
	"""	
	# units shoud be the value to convert in s 	
	import scipy.io
	hd_info = scipy.io.loadmat(path)['hdCellStats'][:,-1]
	return np.where(hd_info[index])[0]



def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nap	
	f = open(path, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 2		
	n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
	duration = n_samples/frequency
	interval = 1/frequency
	f.close()
	fp = np.memmap(path, np.int16, 'r', shape = (n_samples, n_channels))		
	timestep = np.arange(0, n_samples)/frequency

	if type(channel) is not list:
		timestep = np.arange(0, n_samples)/frequency
		return nap.Tsd(timestep, fp[:,channel], time_units = 's')
	elif type(channel) is list:
		timestep = np.arange(0, n_samples)/frequency
		return nap.TsdFrame(timestep, fp[:,channel], time_units = 's')


def loadBunch_Of_LFP(path,  start, stop, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nap	
	bytes_size = 2		
	start_index = int(start*frequency*n_channels*bytes_size)
	stop_index = int(stop*frequency*n_channels*bytes_size)
	fp = np.memmap(path, np.int16, 'r', start_index, shape = (stop_index - start_index)//bytes_size)
	data = np.array(fp).reshape(len(fp)//n_channels, n_channels)

	if type(channel) is not list:
		timestep = np.arange(0, len(data))/frequency
		return nap.Tsd(timestep, data[:,channel], time_units = 's')
	elif type(channel) is list:
		timestep = np.arange(0, len(data))/frequency		
		return nap.TsdFrame(timestep, data[:,channel], time_units = 's')

def loadUpDown(path):
	import neuroseries as nap
	import os
	name = path.split("/")[-1]
	files = os.listdir(path)
	if name + '.evt.py.dow' in files:
		tmp = np.genfromtxt(path+'/'+name+'.evt.py.dow')[:,0]
		tmp = tmp.reshape(len(tmp)//2,2)/1000
		down_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
	if name + '.evt.py.upp' in files:
		tmp = np.genfromtxt(path+'/'+name+'.evt.py.upp')[:,0]
		tmp = tmp.reshape(len(tmp)//2,2)/1000
		up_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
	return (down_ep, up_ep)


def writeNeuroscopeEvents(path, ep, name):
	f = open(path, 'w')
	for i in range(len(ep)):
		f.writelines(str(ep.as_units('ms').iloc[i]['start']) + " "+name+" start "+ str(1)+"\n")
		f.writelines(str(ep.as_units('ms').iloc[i]['end']) + " "+name+" end "+ str(1)+"\n")
	f.close()		
	return

def loadShankStructure(generalinfo):
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]-1
		else :
			shankStructure[k[0]] = []
	
	return shankStructure	