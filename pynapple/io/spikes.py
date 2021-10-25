import numpy as np
import sys,os
import scipy.io
sys.path.append('..')
import core as nts
import pandas as pd


'''
Wrappers should be able to distinguish between raw data or matlab processed data
'''

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
				spikes[i]     = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')
			a             = spikes[0].as_units('s').index.values    
			if ((a[-1]-a[0])/60.)/60. > 20. : # VERY BAD        
				spikes         = {}    
				for i in shankIndex:
					spikes[i]     = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2]*0.0001, time_units = 's')
			return spikes, shank
		elif 'SpikeData.h5' in files:            
			final_path = os.path.join(new_path, 'SpikeData.h5')            
			try:
				spikes = pd.read_hdf(final_path, mode='r')
				# Returning a dictionnary | can be changed to return a dataframe
				toreturn = {}
				for i,j in spikes:
					toreturn[j] = nts.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')
				shank = spikes.columns.get_level_values(0).values[:,np.newaxis]
				return toreturn, shank
			except:
				spikes = pd.HDFStore(final_path, 'r')
				shanks = spikes['/shanks']
				toreturn = {}
				for j in shanks.index:
					toreturn[j] = nts.Ts(spikes['/spikes/s'+str(j)])
				shank = shanks.values
				spikes.close()
				del spikes
				return toreturn, shank
			
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
			toreturn[j] = nts.Ts(t=s[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')

	del spikes
	shank = np.hstack(shank)

	final_path = os.path.join(new_path, 'SpikeData.h5')
	store = pd.HDFStore(final_path)
	for s in toreturn.keys():
		store.put('spikes/s'+str(s), toreturn[s].as_series())
	store.put('shanks', pd.Series(index = list(toreturn.keys()), data = shank))
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
	# 	toreturn[j] = nts.Ts(t=spikes[(i,j)].replace(0,np.nan).dropna().index.values, time_units = 's')

	# shank = spikes.columns.get_level_values(0).values[:,np.newaxis].flatten()

	return toreturn, shank