import numpy as np
from numba import jit
import pandas as pd
import sys, os
import scipy
from scipy import signal
from itertools import combinations
#from pycircstat.descriptive import mean as circmean
#from pylab import *
from .. import core as nts



'''
Utilities functions
Feel free to add your own
'''


#########################################################
# CORRELATION
#########################################################
@jit(nopython=True)
def crossCorr(t1, t2, binsize, nbins):
	''' 
		Fast crossCorr 
	'''
	nt1 = len(t1)
	nt2 = len(t2)
	if np.floor(nbins/2)*2 == nbins:
		nbins = nbins+1

	m = -binsize*((nbins+1)/2)
	B = np.zeros(nbins)
	for j in range(nbins):
		B[j] = m+j*binsize

	w = ((nbins/2) * binsize)
	C = np.zeros(nbins)
	i2 = 1

	for i1 in range(nt1):
		lbound = t1[i1] - w
		while i2 < nt2 and t2[i2] < lbound:
			i2 = i2+1
		while i2 > 1 and t2[i2-1] > lbound:
			i2 = i2-1

		rbound = lbound
		l = i2
		for j in range(nbins):
			k = 0
			rbound = rbound+binsize
			while l < nt2 and t2[l] < rbound:
				l = l+1
				k = k+1

			C[j] += k

	# for j in range(nbins):
	# C[j] = C[j] / (nt1 * binsize)
	C = C/(nt1 * binsize/1000)

	return C

def crossCorr2(t1, t2, binsize, nbins):
	'''
		Slow crossCorr
	'''
	window = np.arange(-binsize*(nbins/2),binsize*(nbins/2)+2*binsize,binsize) - (binsize/2.)
	allcount = np.zeros(nbins+1)
	for e in t1:
		mwind = window + e
		# need to add a zero bin and an infinite bin in mwind
		mwind = np.array([-1.0] + list(mwind) + [np.max([t1.max(),t2.max()])+binsize])	
		index = np.digitize(t2, mwind)
		# index larger than 2 and lower than mwind.shape[0]-1
		# count each occurences 
		count = np.array([np.sum(index == i) for i in range(2,mwind.shape[0]-1)])
		allcount += np.array(count)
	allcount = allcount/(float(len(t1))*binsize / 1000)
	return allcount

def xcrossCorr_slow(t1, t2, binsize, nbins, nbiter, jitter, confInt):		
	times 			= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	H0 				= crossCorr(t1, t2, binsize, nbins)	
	H1 				= np.zeros((nbiter,nbins+1))
	t2j	 			= t2 + 2*jitter*(np.random.rand(nbiter, len(t2)) - 0.5)
	t2j 			= np.sort(t2j, 1)
	for i in range(nbiter):			
		H1[i] 		= crossCorr(t1, t2j[i], binsize, nbins)
	Hm 				= H1.mean(0)
	tmp 			= np.sort(H1, 0)
	HeI 			= tmp[int((1-confInt)/2*nbiter),:]
	HeS 			= tmp[int((confInt + (1-confInt)/2)*nbiter)]
	Hstd 			= np.std(tmp, 0)

	return (H0, Hm, HeI, HeS, Hstd, times)

def xcrossCorr_fast(t1, t2, binsize, nbins, nbiter, jitter, confInt):		
	times 			= np.arange(0, binsize*(nbins*2+1), binsize) - (nbins*2*binsize)/2
	# need to do a cross-corr of double size to convolve after and avoid boundary effect
	H0 				= crossCorr(t1, t2, binsize, nbins*2)	
	window_size 	= 2*jitter//binsize
	window 			= np.ones(window_size)*(1/window_size)
	Hm 				= np.convolve(H0, window, 'same')
	Hstd			= np.sqrt(np.var(Hm))	
	HeI 			= np.NaN
	HeS 			= np.NaN	
	return (H0, Hm, HeI, HeS, Hstd, times)	

def compute_AutoCorrs(spks, ep, binsize = 5, nbins = 200):
	# First let's prepare a pandas dataframe to receive the data
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2	
	autocorrs = pd.DataFrame(index = times, columns = list(spks.keys()))
	firing_rates = pd.Series(index = list(spks.keys()))

	# Now we can iterate over the dictionnary of spikes
	for i in spks:
		# First we extract the time of spikes in ms during wake
		spk_time = spks[i].restrict(ep).as_units('ms').index.values
		# Calling the crossCorr function
		autocorrs[i] = crossCorr(spk_time, spk_time, binsize, nbins)
		# Computing the mean firing rate
		firing_rates[i] = len(spk_time)/ep.tot_length('s')

	# We can divide the autocorrs by the firing_rates
	autocorrs = autocorrs / firing_rates

	# And don't forget to replace the 0 ms for 0
	autocorrs.loc[0] = 0.0
	return autocorrs, firing_rates

def compute_CrossCorrs(spks, ep, binsize=10, nbins = 2000, norm = False):
	"""
		
	"""	
	neurons = list(spks.keys())
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	cc = pd.DataFrame(index = times, columns = list(combinations(neurons, 2)))
		
	for i,j in cc.columns:		
		spk1 = spks[i].restrict(ep).as_units('ms').index.values
		spk2 = spks[j].restrict(ep).as_units('ms').index.values		
		tmp = crossCorr(spk1, spk2, binsize, nbins)		
		fr = len(spk2)/ep.tot_length('s')
		if norm:
			cc[(i,j)] = tmp/fr
		else:
			cc[(i,j)] = tmp
	return cc

def compute_PairCrossCorr(spks, ep, pair, binsize=10, nbins = 2000, norm = False):
	"""
		
	"""		
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2	
	spk1 = spks[pair[0]].restrict(ep).as_units('ms').index.values
	spk2 = spks[pair[1]].restrict(ep).as_units('ms').index.values		
	tmp = crossCorr(spk1, spk2, binsize, nbins)		
	fr = len(spk2)/ep.tot_length('s')
	tmp = pd.Series(index = times, data = tmp)
	if norm:
		tmp = tmp/fr
	else:
		tmp = tmp
	return tmp

def compute_EventCrossCorr(spks, evt, ep, binsize = 5, nbins = 1000, norm=False):
	"""
	"""
	neurons = list(spks.keys())
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	cc = pd.DataFrame(index = times, columns = neurons)
	tsd1 = evt.restrict(ep).as_units('ms').index.values
	for i in neurons:
		spk2 = spks[i].restrict(ep).as_units('ms').index.values
		tmp = crossCorr(tsd1, spk2, binsize, nbins)
		fr = len(spk2)/ep.tot_length('s')
		if norm:
			cc[i] = tmp/fr
		else:
			cc[i] = tmp
	return cc
		
def compute_ISI(spks, ep, maxisi, nbins, log_=False):
	"""
	"""
	neurons = list(spks.keys())
	if log_:
		bins = np.linspace(np.log10(1), np.log10(maxisi), nbins)
	else:
		bins = np.linspace(0, maxisi, nbins)
		
	isi = pd.DataFrame(index =  bins[0:-1] + np.diff(bins)/2, columns = neurons)
	for i in neurons:
		tmp = []
		for j in ep.index:
			tmp.append(np.diff(spks[i].restrict(ep.loc[[j]]).as_units('ms').index.values))
		tmp = np.hstack(tmp)
		if log_:
			isi[i], _ = np.histogram(np.log10(tmp), bins)
		else:
			isi[i], _ = np.histogram(tmp, bins)

	return isi

def compute_AllPairsCrossCorrs(spks, ep, binsize=10, nbins = 2000, norm = False):
	"""
		
	"""	
	neurons = list(spks.keys())
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	cc = pd.DataFrame(index = times, columns = list(combinations(neurons, 2)))
		
	for i,j in cc.columns:		
		spk1 = spks[i].restrict(ep).as_units('ms').index.values
		spk2 = spks[j].restrict(ep).as_units('ms').index.values		
		tmp = crossCorr(spk1, spk2, binsize, nbins)		
		fr = len(spk2)/ep.tot_length('s')
		if norm:
			cc[(i,j)] = tmp/fr
		else:
			cc[(i,j)] = tmp
	return cc

def compute_AsyncCrossCorrs(spks, ep, binsize=10, nbins = 2000, norm = False, edge = 20):
	"""
		
	"""	
	neurons = list(spks.keys())
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	cc = pd.DataFrame(index = times, columns = list(combinations(neurons, 2)))
		
	for i,j in cc.columns:		
		spk1 = spks[i].restrict(ep).as_units('ms').index.values
		spk2 = spks[j].restrict(ep).as_units('ms').index.values		

		spksync = []
		spkasync = []
		for t in spk2:			
			if np.sum(np.abs(t-spk1)<edge):
				spksync.append(t)
			else:
				spkasync.append(t)


		# tmp = crossCorr(spk1, spk2, binsize, nbins)		
		tmp = crossCorr(spk1, np.array(spkasync), binsize, nbins)
		fr = len(spkasync)/ep.tot_length('s')
		if norm:
			cc[(i,j)] = tmp/fr
		else:
			cc[(i,j)] = tmp
	return cc

def compute_RandomCrossCorrs(spks, ep,  binsize=10, nbins = 2000, norm = False, percent = 0.5):
	"""
		
	"""	
	neurons = list(spks.keys())
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	cc = pd.DataFrame(index = times, columns = list(combinations(neurons, 2)))
		
	for i,j in cc.columns:		
		spk1 = spks[i].restrict(ep).as_units('ms').index.values
		spk2 = spks[j].restrict(ep).as_units('ms').index.values		

		spk1_random = np.sort(np.random.choice(spk1, int(len(spk1)*percent), replace=False))
		spk2_random = np.sort(np.random.choice(spk2, int(len(spk2)*percent), replace=False))

		# tmp = crossCorr(spk1, spk2, binsize, nbins)		
		tmp = crossCorr(spk1_random, spk2_random, binsize, nbins)
		fr = len(spk2_random)/ep.tot_length('s')
		if norm:
			cc[(i,j)] = tmp/fr
		else:
			cc[(i,j)] = tmp
	return cc

#########################################################
# VARIOUS
#########################################################
def computeLMNAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0, bin_size = 100):
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))	
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)	
	bin_size 		= bin_size * 1000
	time_bins		= np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()
	tmp3.index 		= time_bins[np.unique(index)-1]+bin_size/2
	tmp3 			= nts.Tsd(tmp3)
	tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
	newangle 		= nts.Tsd(t = tmp3.index.values, d = tmp3.values%(2*np.pi))
	velocity 		= nts.Tsd(t=tmp3.index.values[1:], d = tmp4)
	velocity 		= velocity.restrict(ep)	
	velo_spikes 	= {}	
	for k in spikes: velo_spikes[k]	= velocity.realign(spikes[k].restrict(ep))
	# bins_velocity	= np.array([velocity.min(), -2*np.pi/3, -np.pi/6, np.pi/6, 2*np.pi/3, velocity.max()+0.001])
	bins_velocity	= np.array([velocity.min(), -np.pi/6, np.pi/6, velocity.max()+0.001])

	idx_velocity 	= {k:np.digitize(velo_spikes[k].values, bins_velocity)-1 for k in spikes}

	bins 			= np.linspace(0, 2*np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	tuning_curves 	= {i:pd.DataFrame(index = idx, columns = list(spikes.keys())) for i in range(3)}	

	# for i,j in zip(range(3),range(0,6,2)):
	for i,j in zip(range(3),range(3)):
		for k in spikes:
			spks 			= spikes[k].restrict(ep)			
			spks 			= spks[idx_velocity[k] == j]
			angle_spike 	= newangle.restrict(ep).realign(spks)
			spike_count, bin_edges = np.histogram(angle_spike, bins)
			tmp 			= newangle.loc[velocity.index[np.logical_and(velocity.values>bins_velocity[j], velocity.values<bins_velocity[j+1])]]
			occupancy, _ 	= np.histogram(tmp, bins)
			spike_count 	= spike_count/occupancy	
			tuning_curves[i][k] = spike_count*(1/(bin_size*1e-6))

	return tuning_curves, velocity, bins_velocity

def computeAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0):
	"""
	Compute angular tuning curves

	Parameters
	----------
	spikes: dict
		Dictionnary of Ts objects
	angles: Tsd object
		Angle between 0 and 2pi
	ep: IntervalSet object
		Epoch to compute the tuning curve
	nb_bins: int
		Number of bins of the tuning curves
	frequency: float
		Sampling frequency of the angle

	Returns
	-------
	tuning_curves: DataFrame object

	Examples
	--------
	>>> import pynapple as ap
	>>> tuning_curves = ap.computeAngularTuningCurves(spikes, angles, wake_ep, 60, 120)
	"""
	bins 			= np.linspace(0, 2*np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	tuning_curves 	= pd.DataFrame(index = idx, columns = list(spikes.keys()))	
	angle 			= angle.restrict(ep)
	# Smoothing the angle here
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	angle			= nts.Tsd(tmp2%(2*np.pi))
	for k in spikes:
		spks 			= spikes[k]
		# true_ep 		= nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))		
		spks 			= spks.restrict(ep)	
		angle_spike 	= spks.value_from(angle, ep)
		#angle_spike 	= angle.restrict(ep).realign(spks)
		spike_count, bin_edges = np.histogram(angle_spike, bins)
		occupancy, _ 	= np.histogram(angle, bins)
		spike_count 	= spike_count/occupancy		
		tuning_curves[k] = spike_count*frequency	

	return tuning_curves

def computeSpatialInfo(tc, angle, ep):
	nb_bins = tc.shape[0]+1
	bins 	= np.linspace(0, 2*np.pi, nb_bins)
	angle 			= angle.restrict(ep)
	# Smoothing the angle here
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	angle			= nts.Tsd(tmp2%(2*np.pi))
	pf = tc.values
	occupancy, _ 	= np.histogram(angle, bins)
	occ = np.atleast_2d(occupancy/occupancy.sum()).T
	f = np.sum(pf * occ, 0)
	pf = pf / f
	SI = np.sum(occ * pf * np.log2(pf), 0)
	SI = pd.DataFrame(index = tc.columns, columns = ['SI'], data = SI)
	return SI


def findHDCells(tuning_curves, z = 50, p = 0.0001 , m = 1):
	"""
		Peak firing rate larger than 1
		and Rayleigh test p<0.001 & z > 100
	"""
	cond1 = tuning_curves.max()>m
	from pycircstat.tests import rayleigh
	stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
	for k in tuning_curves:
		stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
	cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
	tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]
	hd = pd.Series(index = tuning_curves.columns, data = 0)
	hd.loc[tokeep] = 1
	return hd, stat

def decodeHD(tuning_curves, spikes, ep, bin_size = 200, px = None):
	"""
		See : Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
		tuning_curves: pd.DataFrame with angular position as index and columns as neuron
		spikes : dictionnary of spike times
		ep : nts.IntervalSet, the epochs for decoding
		bin_size : in ms (default:200ms)
		px : Occupancy. If None, px is uniform
	"""		
	if len(ep) == 1:
		bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
	else:
		# ep2 = nts.IntervalSet(ep.copy().as_units('ms'))
		# ep2 = ep2.drop_short_intervals(bin_size*2)
		# bins = []
		# for i in ep2.index:
		# 	bins.append(np.arange())
		# bins = np.arange(ep2.start.iloc[0], ep.end.iloc[-1], bin_size)
		print("TODO")
		sys.exit()


	order = tuning_curves.columns.values
	# TODO CHECK MATCH

	# smoothing with a non-normalized gaussian
	w = scipy.signal.gaussian(51, 2)
	
	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = order)
	for n in spike_counts:		
		spks = spikes[n].restrict(ep).as_units('ms').index.values
		tmp = np.histogram(spks, bins)
		spike_counts[n] = np.convolve(tmp[0], w, mode = 'same')
		# spike_counts[k] = tmp[0]

	tcurves_array = tuning_curves.values
	spike_counts_array = spike_counts.values
	proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

	part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))
	if px is not None:
		part2 = px
	else:
		part2 = np.ones(tuning_curves.shape[0])
	#part2 = np.histogram(position['ry'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(position['ry'])/float(len(position['ry'])))[0]
	
	for i in range(len(proba_angle)):
		part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
		p = part1 * part2 * part3
		proba_angle[i] = p/p.sum() # Normalization process here

	proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)	
	proba_angle = proba_angle.astype('float')
	decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
	return decoded, proba_angle, spike_counts

def computePlaceFields(spikes, position, ep, nb_bins = 200, frequency = 120.0):
	place_fields = {}
	position_tsd = position.restrict(ep)
	xpos = position_tsd.iloc[:,0]
	ypos = position_tsd.iloc[:,1]
	xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
	ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)
	for n in spikes:
		position_spike = position_tsd.realign(spikes[n].restrict(ep))
		spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
		occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
		mean_spike_count = spike_count/(occupancy+1)
		place_field = mean_spike_count*frequency    
		place_fields[n] = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
		
	extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB
	return place_fields, extent

def computeOccupancy(position_tsd, nb_bins = 100):
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]    
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    return occupancy

def computeAngularVelocityTuningCurves(spikes, angle, ep, nb_bins = 61, bin_size = 10000, norm=True):
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	time_bins		= np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size) # assuming microseconds
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()
	tmp3.index 		= time_bins[np.unique(index)-1]+bin_size/2
	tmp3 			= nts.Tsd(tmp3)
	tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
	tmp2 			= nts.Tsd(tmp2)
	tmp4			= np.diff(tmp2.values)/np.diff(tmp2.as_units('s').index.values)	
	velocity 		= nts.Tsd(t=tmp2.index.values[1:], d = tmp4)
	velocity 		= velocity.restrict(ep)	
	bins 			= np.linspace(-2*np.pi, 2*np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	velo_curves		= pd.DataFrame(index = idx, columns = list(spikes.keys()))

	for k in spikes:
		spks 		= spikes[k]
		spks 		= spks.restrict(ep)
		speed_spike = velocity.realign(spks)
		spike_count, bin_edges = np.histogram(speed_spike, bins)
		occupancy, _ = np.histogram(velocity.restrict(ep), bins)
		spike_count = spike_count/(occupancy+1)
		velo_curves[k] = spike_count*(1/(bin_size*1e-6))
		# normalizing by firing rate 
		if norm:
			velo_curves[k] = velo_curves[k]/(len(spikes[k].restrict(ep))/ep.tot_length('s'))

	return velo_curves

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
	new_tuning_curves = {}	
	for i in tuning_curves.columns:
		tcurves = tuning_curves[i]
		offset = np.mean(np.diff(tcurves.index.values))
		padded 	= pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi)-offset,
												tcurves.index.values,
												tcurves.index.values+(2*np.pi)+offset)),
							data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
		smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)		
		new_tuning_curves[i] = smoothed.loc[tcurves.index]

	new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

	return new_tuning_curves

def computeMeanFiringRate(spikes, epochs, name):
	mean_frate = pd.DataFrame(index = spikes.keys(), columns = name)
	for n, ep in zip(name, epochs):
		for k in spikes:
			mean_frate.loc[k,n] = len(spikes[k].restrict(ep))/ep.tot_length('s')
	return mean_frate

def computeSpeedTuningCurves(spikes, position, ep, bin_size = 0.1, nb_bins = 20, speed_max = 0.4):
	time_bins 	= np.arange(position.index[0], position.index[-1]+bin_size*1e6, bin_size*1e6)
	index 		= np.digitize(position.index.values, time_bins)
	tmp 		= position.groupby(index).mean()
	tmp.index 	= time_bins[np.unique(index)-1]+(bin_size*1e6)/2
	distance	= np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
	speed 		= nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/bin_size)
	speed 		= speed.restrict(ep)
	bins 		= np.linspace(0, speed_max, nb_bins)
	idx 		= bins[0:-1]+np.diff(bins)/2
	speed_curves = pd.DataFrame(index = idx,columns = np.arange(len(spikes)))
	for k in spikes:
		spks 	= spikes[k]
		spks 	= spks.restrict(ep)
		speed_spike = speed.realign(spks)
		spike_count, bin_edges = np.histogram(speed_spike, bins)
		occupancy, _ = np.histogram(speed, bins)
		spike_count = spike_count/(occupancy+1)
		speed_curves[k] = spike_count/bin_size

	return speed_curves

def computeAccelerationTuningCurves(spikes, position, ep, bin_size = 0.1, nb_bins = 40):
	time_bins 	= np.arange(position.index[0], position.index[-1]+bin_size*1e6, bin_size*1e6)
	index 		= np.digitize(position.index.values, time_bins)
	tmp 		= position.groupby(index).mean()
	tmp.index 	= time_bins[np.unique(index)-1]+(bin_size*1e6)/2
	distance	= np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
	speed 		= nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/bin_size)
	speed 		= speed.restrict(ep)
	speed 		= speed.as_series()
	speed2 		= speed.rolling(window=10, win_type='gaussian', center= True, min_periods=1).mean(std = 1.0)
	accel 		= nts.Tsd(t = speed2.index.values[0:-1] + np.diff(speed2.index.values)/2, d = np.diff(speed2.values))	
	bins 		= np.linspace(accel.min(), accel.max(), nb_bins)
	idx 		= bins[0:-1]+np.diff(bins)/2
	accel_curves = pd.DataFrame(index = idx,columns = np.arange(len(spikes)))
	for k in spikes:
		spks 	= spikes[k]
		spks 	= spks.restrict(ep)
		accel_spike = accel.realign(spks)
		spike_count, bin_edges = np.histogram(accel_spike, bins)
		occupancy, _ = np.histogram(accel, bins)
		spike_count = spike_count/(occupancy+1)
		accel_curves[k] = spike_count/bin_size

	return accel_curves

def refineSleepFromAccel(acceleration, sleep_ep):
	vl = acceleration[0].restrict(sleep_ep)
	vl = vl.as_series().diff().abs().dropna()	
	a, _ = scipy.signal.find_peaks(vl, 0.025)
	peaks = nts.Tsd(vl.iloc[a])
	duration = np.diff(peaks.as_units('s').index.values)
	interval = nts.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[1:])

	newsleep_ep = interval.iloc[duration>15.0]
	newsleep_ep = newsleep_ep.reset_index(drop=True)
	newsleep_ep = newsleep_ep.merge_close_intervals(100000, time_units ='us')

	newsleep_ep	= sleep_ep.intersect(newsleep_ep)

	return newsleep_ep

def splitWake(ep):
	if len(ep) != 1:
		print('Cant split wake in 2')
		sys.exit()
	tmp = np.zeros((2,2))
	tmp[0,0] = ep.values[0,0]
	tmp[1,1] = ep.values[0,1]
	tmp[0,1] = tmp[1,0] = ep.values[0,0] + np.diff(ep.values[0])/2
	return nts.IntervalSet(start = tmp[:,0], end = tmp[:,1])

def centerTuningCurves(tcurve):
	"""
	center tuning curves by peak
	"""
	peak 			= pd.Series(index=tcurve.columns,data = np.array([circmean(tcurve.index.values, tcurve[i].values) for i in tcurve.columns]))
	new_tcurve 		= []
	for p in tcurve.columns:	
		x = tcurve[p].index.values - tcurve[p].index[tcurve[p].index.get_loc(peak[p], method='nearest')]
		x[x<-np.pi] += 2*np.pi
		x[x>np.pi] -= 2*np.pi
		tmp = pd.Series(index = x, data = tcurve[p].values).sort_index()
		new_tcurve.append(tmp.values)
	new_tcurve = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(new_tcurve).T, columns = tcurve.columns)
	return new_tcurve

def offsetTuningCurves(tcurve, diffs):
	"""
	offseting tuning curves synced by diff
	"""	
	new_tcurve 		= []
	for p in tcurve.columns:	
		x = tcurve[p].index.values - tcurve[p].index[tcurve[p].index.get_loc(diffs[p], method='nearest')]
		x[x<-np.pi] += 2*np.pi
		x[x>np.pi] -= 2*np.pi
		tmp = pd.Series(index = x, data = tcurve[p].values).sort_index()
		new_tcurve.append(tmp.values)
	new_tcurve = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(new_tcurve).T, columns = tcurve.columns)
	return new_tcurve


#########################################################
# LFP FUNCTIONS
#########################################################
def butter_bandpass(lowcut, highcut, fs, order=5):
	from scipy.signal import butter
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	from scipy.signal import lfilter
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def downsample(tsd, up, down):
	import scipy.signal
	import neuroseries as nts
	dtsd = scipy.signal.resample_poly(tsd.values, up, down)
	dt = tsd.as_units('s').index.values[np.arange(0, tsd.shape[0], down)]
	if len(tsd.shape) == 1:		
		return nts.Tsd(dt, dtsd, time_units = 's')
	elif len(tsd.shape) == 2:
		return nts.TsdFrame(dt, dtsd, time_units = 's', columns = list(tsd.columns))

def getPeaksandTroughs(lfp, min_points):
	"""	 
		At 250Hz (1250/5), 2 troughs cannont be closer than 20 (min_points) points (if theta reaches 12Hz);		
	"""
	import neuroseries as nts
	import scipy.signal
	if isinstance(lfp, nts.time_series.Tsd):
		troughs 		= nts.Tsd(lfp.as_series().iloc[scipy.signal.argrelmin(lfp.values, order =min_points)[0]], time_units = 'us')
		peaks 			= nts.Tsd(lfp.as_series().iloc[scipy.signal.argrelmax(lfp.values, order =min_points)[0]], time_units = 'us')
		tmp 			= nts.Tsd(troughs.realign(peaks, align = 'next').as_series().drop_duplicates('first')) # eliminate double peaks
		peaks			= peaks[tmp.index]
		tmp 			= nts.Tsd(peaks.realign(troughs, align = 'prev').as_series().drop_duplicates('first')) # eliminate double troughs
		troughs 		= troughs[tmp.index]
		return (peaks, troughs)
	elif isinstance(lfp, nts.time_series.TsdFrame):
		peaks 			= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		troughs			= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		for i in lfp.keys():
			peaks[i], troughs[i] = getPeaksandTroughs(lfp[i], min_points)
		return (peaks, troughs)

def getPhase(lfp, fmin, fmax, nbins, fsamp, power = False):
	""" Continuous Wavelets Transform
		return phase of lfp in a Tsd array
	"""
	import neuroseries as nts
	from Wavelets import MyMorlet as Morlet
	if isinstance(lfp, nts.time_series.TsdFrame):
		allphase 		= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		allpwr 			= nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
		for i in lfp.keys():
			allphase[i], allpwr[i] = getPhase(lfp[i], fmin, fmax, nbins, fsamp, power = True)
		if power:
			return allphase, allpwr
		else:
			return allphase			

	elif isinstance(lfp, nts.time_series.Tsd):
		cw 				= Morlet(lfp.values, fmin, fmax, nbins, fsamp)
		cwt 			= cw.getdata()
		cwt 			= np.flip(cwt, axis = 0)
		wave 			= np.abs(cwt)**2.0
		phases 			= np.arctan2(np.imag(cwt), np.real(cwt)).transpose()	
		cwt 			= None
		index 			= np.argmax(wave, 0)
		# memory problem here, need to loop
		phase 			= np.zeros(len(index))	
		for i in range(len(index)) : phase[i] = phases[i,index[i]]
		phases 			= None
		if power: 
			pwrs 		= cw.getpower()		
			pwr 		= np.zeros(len(index))		
			for i in range(len(index)):
				pwr[i] = pwrs[index[i],i]	
			return nts.Tsd(lfp.index.values, phase), nts.Tsd(lfp.index.values, pwr)
		else:
			return nts.Tsd(lfp.index.values, phase)

#########################################################
# INTERPOLATION
#########################################################
def interpolate(z, x, y, inter, bbox = None):	
	import scipy.interpolate
	xnew = np.arange(x.min(), x.max()+inter, inter)
	ynew = np.arange(y.min(), y.max()+inter, inter)
	if bbox == None:
		f = scipy.interpolate.RectBivariateSpline(y, x, z)
	else:
		f = scipy.interpolate.RectBivariateSpline(y, x, z, bbox = bbox)
	znew = f(ynew, xnew)
	return (xnew, ynew, znew)

def filter_(z, n):
	from scipy.ndimage import gaussian_filter	
	return gaussian_filter(z, n)


#########################################################
# HELPERS
#########################################################
def writeNeuroscopeEvents(path, ep, name):
	f = open(path, 'w')
	for i in range(len(ep)):
		f.writelines(str(ep.as_units('ms').iloc[i]['start']) + " "+name+" start "+ str(1)+"\n")
		f.writelines(str(ep.as_units('ms').iloc[i]['end']) + " "+name+" end "+ str(1)+"\n")
	f.close()		
	return

def getAllInfos(data_directory, datasets):
	allm = np.unique(["/".join(s.split("/")[0:2]) for s in datasets])
	infos = {}
	for m in allm:
		path = os.path.join(data_directory, m)
		csv_file = list(filter(lambda x: '.csv' in x, os.listdir(path)))[0]
		infos[m.split('/')[1]] = pd.read_csv(os.path.join(path, csv_file), index_col = 0)
	return infos

def computeSpeed(position, ep, bin_size = 0.1):
	time_bins 	= np.arange(position.index[0], position.index[-1]+bin_size*1e6, bin_size*1e6)
	index 		= np.digitize(position.index.values, time_bins)
	tmp 		= position.groupby(index).mean()
	tmp.index 	= time_bins[np.unique(index)-1]+(bin_size*1e6)/2
	distance	= np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
	speed 		= nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/bin_size)
	speed 		= speed.restrict(ep)
	return speed

#######################################################
# SYNCHRONY HELPERS
#######################################################
@jit(nopython=True, parallel=True)
def getSyncAsync(spk1, spk2):
	spksync = []
	spkasync = []
	for t in spk2:
		d = np.abs(t-spk1)
		idx = d<4
		if np.sum(idx):
			spksync.append(t)
		else:
			spkasync.append(t)
	spksync = np.array(spksync)
	spkasync = np.array(spkasync)
	return spksync, spkasync


def getSpikesSyncAsync(spks, hd_neurons):
	"""
	"""		
	allsync = {}
	allasync = {}
	n = len(spks)
	for i in range(n):
		for j in range(n):
			if i != j:				
				spk1 = spks[i]
				spk2 = spks[j]
				spksync, spkasync = getSyncAsync(spk1, spk2)
				allsync[(hd_neurons[i],hd_neurons[j])] = nts.Ts(spksync, time_units = 'ms')
				allasync[(hd_neurons[i],hd_neurons[j])] = nts.Ts(spkasync, time_units = 'ms')

	return allsync, allasync

	
def sample_frates(frates_value, time_index, hd_neurons, n = 30):
	spikes_random = {}
	for i in range(len(hd_neurons)):
		l = frates_value[:,i]
		s = np.random.poisson(l, size=(int(n),int(len(frates_value))))
		tmp = {}
		for j in range(len(s)):
			tmp[j] = nts.Ts(time_index[s[j]>0])
		spikes_random[hd_neurons[i]] = tmp
	return spikes_random

def sampleSpikesFromAngularPosition(tcurve, angles, ep, bin_size = 1000):
	"""
	"""
	hd_neurons = tcurve.columns.values
	bin_size = 1000  # us
	angles = pd.Series(index = angles.index, data = np.unwrap(angles.values))
	bins = np.arange(ep.loc[0,'start'], ep.loc[0, 'end'], bin_size)
	tmp = angles.groupby(np.digitize(angles.index.values, bins)-1).mean()
	angle2 = pd.Series(index = np.arange(len(bins)), data = np.nan)
	angle2.loc[tmp.index] = tmp.values
	angle2 = angle2.interpolate()
	angle2.index = bins + np.max(np.diff(bins)/2)
	angle2 = angle2%(2*np.pi)

	idx = np.argsort(np.abs(np.vstack(angle2.values) - tcurve.index.values), axis = 1)[:,0]
	idx = tcurve.index[idx]
	frates = tcurve.loc[idx]
	frates.index = angle2.index

	time_index = frates.index.values
	frates_value = frates.values*(bin_size*1e-6)

	spikes_random = sample_frates(frates_value, time_index, hd_neurons)

	return spikes_random

def sampleSpikesFromAngularVelocity(ahvcurve, angles, ep, bin_size = 1000):
	"""
	"""
	hd_neurons = ahvcurve.columns.values
	bin_size = 1000  # us

	tmp 			= pd.Series(index = angles.index.values, data = np.unwrap(angles.values))
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	time_bins		= np.arange(tmp.index[0], tmp.index[-1]+10000, 10000) # assuming microseconds
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()
	tmp3.index 		= time_bins[np.unique(index)-1]+bin_size/2
	tmp3 			= nts.Tsd(tmp3)
	tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
	tmp2 			= nts.Tsd(tmp2)
	tmp4			= np.diff(tmp2.values)/np.diff(tmp2.as_units('s').index.values)	
	velocity 		= nts.Tsd(t=tmp2.index.values[1:], d = tmp4)
	velocity 		= velocity.restrict(ep)	
	
	bins = np.arange(ep.loc[0,'start'], ep.loc[0, 'end'], bin_size)
	tmp = velocity.groupby(np.digitize(velocity.index.values, bins)-1).mean()
	velocity2 = pd.Series(index = np.arange(len(bins)), data = np.nan)
	velocity2.loc[tmp.index] = tmp.values
	velocity2 = velocity2.interpolate()
	velocity2.index = bins + np.max(np.diff(bins)/2)
	
	idx = np.argsort(np.abs(np.vstack(velocity2.values) - ahvcurve.index.values), axis = 1)[:,0]
	idx = ahvcurve.index[idx]
	frates = ahvcurve.loc[idx]
	frates.index = velocity2.index

	time_index = frates.index.values
	frates_value = frates.values*(bin_size*1e-6)

	spikes_random = sample_frates(frates_value, time_index, hd_neurons)

	return spikes_random

def loadShankStructure(generalinfo):
	shankStructure = {}
	for k,i in zip(generalinfo['shankStructure'][0][0][0][0],range(len(generalinfo['shankStructure'][0][0][0][0]))):
		if len(generalinfo['shankStructure'][0][0][1][0][i]):
			shankStructure[k[0]] = generalinfo['shankStructure'][0][0][1][0][i][0]-1
		else :
			shankStructure[k[0]] = []
	
	return shankStructure	

def binSpikeTrain(spikes, epochs, bin_size, std=0):
	if epochs is None:
		start = np.inf
		end = 0
		for n in spikes:
			start = np.minimum(spikes[n].index.values[0], start)
			end = np.maximum(spikes[n].index.values[-1], end)
		epochs = nts.IntervalSet(start = start, end = end)

	bins = np.arange(epochs['start'].iloc[0], epochs['end'].iloc[-1] + bin_size*1000, bin_size*1000)
	rate = []
	for i,n in enumerate(spikes.keys()):
		count, _ = np.histogram(spikes[n].index.values, bins)
		rate.append(count)
	rate = np.array(rate)	
	rate = nts.TsdFrame(t = bins[0:-1]+bin_size//2, d = rate.T)
	if std:
		rate = rate.as_dataframe()
		rate = rate.rolling(window=std*20,win_type='gaussian',center=True,min_periods=1).mean(std=std)	
		rate = nts.TsdFrame(rate)
	rate = rate.restrict(epochs)
	rate.columns = list(spikes.keys())
	return rate

def shuffleByIntervalSpikes(spikes, epochs):
	shuffled = {}
	for n in spikes.keys():
		isi = []
		for i in epochs.index:
			spk = spikes[n].restrict(epochs.loc[[i]])
			tmp = np.diff(spk.index.values)
			np.random.shuffle(tmp)
			isi.append(tmp)
		shuffled[n] = nts.Ts(t = np.cumsum(np.hstack(isi)) + epochs.loc[0,'start'])
	return shuffled

# def plotTuningCurves(tcurves, tokeep = []):	
# 	figure()
# 	for i in tcurves.columns:
# 		subplot(int(np.ceil(np.sqrt(tcurves.shape[1]))),int(np.ceil(np.sqrt(tcurves.shape[1]))),i+1, projection='polar')
# 		plot(tcurves[i])
# 		if len(tokeep):
# 			if i in tokeep:
# 				plot(tcurves[i], linewidth = 4)
# 	return


######################################################################################
# OPTO STUFFS
######################################################################################
def computeRasterOpto(spikes, opto_ep, bin_size = 100):
	"""
	Bin size in ms
	edge in ms
	"""
	rasters = {}
	frates = {}

	# assuming all opto stim are the same for a session
	stim_duration = opto_ep.loc[0,'end'] - opto_ep.loc[0,'start']
	
	bins = np.arange(0, stim_duration + 2*stim_duration + bin_size*1000, bin_size*1000)

	for n in spikes.keys():
		rasters[n] = []
		r = []
		for e in opto_ep.index:
			ep = nts.IntervalSet(start = opto_ep.loc[e,'start'] - stim_duration,
								end = opto_ep.loc[e,'end'] + stim_duration)
			spk = spikes[n].restrict(ep)
			tmp = pd.Series(index = spk.index.values - ep.loc[0,'start'], data = e)
			rasters[n].append(tmp)
			count, _ = np.histogram(tmp.index.values, bins)
			r.append(count)
		r = np.array(r)
		frates[n] = pd.Series(index = bins[0:-1]/1000, data = r.mean(0))
		rasters[n] = pd.concat(rasters[n])		

	frates = pd.concat(frates, 1)
	frates = nts.TsdFrame(t = frates.index.values, d = frates.values, time_units = 'ms')
	return frates, rasters, bins, stim_duration
