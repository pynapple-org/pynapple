# coding: utf-8
"""
Quick start
===========

The examplar data to replicate the figure in the jupyter notebook can be found [here](https://www.dropbox.com/s/su4oaje57g3kit9/A2929-200711.zip?dl=1). 

The data contains a sample recordings taken simultaneously from the anterodorsal thalamus and the hippocampus and contains both a sleep and wake session. It contains both head-direction cells (i.e. cells that fire for a particular head direction in the horizontal plane) and place cells (i.e. cells that fire for a particular position in the environment).

Preprocessing of the data was made with [Kilosort 2.0](https://github.com/MouseLand/Kilosort) and spike sorting was made with [Klusters](http://neurosuite.sourceforge.net/).

Instructions for installing pynapple can be found [here](https://pynapple-org.github.io/pynapple/#installation).

***

This notebook is meant to provide an overview of pynapple by going through:

- **Input output (IO)**. In this case, pynapple will load a NWB file using the [NWBFile object](https://pynapple-org.github.io/pynapple/io.nwb/) within a project [Folder](https://pynapple-org.github.io/pynapple/io.folder/) that represent a dataset. 
- **Core functions** that handle time series, interval sets and groups of time series. See this [notebook](https://pynapple-org.github.io/pynapple/notebooks/pynapple-core-notebook/) for a detailled usage of the core functions.
- **Process functions**. A small collection of high-level functions widely used in system neuroscience. This [notebook](https://pynapple-org.github.io/pynapple/notebooks/pynapple-process-notebook/) details those functions.

"""

# %%
# !!! warning
#     This tutorial uses seaborn and matplotlib for displaying the figure.
#
#     You can install both with `pip install matplotlib seaborn`

import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette = "colorblind", font_scale=1.5, rc=custom_params)

# %%
# ***
# IO
# -----------------
# The first step is to give the path to the data folder.
DATA_DIRECTORY = "../../your/path/to/MyProject/"


# %%
# We can load the session with the function [load_folder](https://pynapple-org.github.io/pynapple/io/#pynapple.io.misc.load_folder). Pynapple will walks throught the folder and collects every subfolders.
# We can use the attribute `view` or the function `expand` to display a tree view of the dataset. The treeview shows all the compatible data format (i.e npz files or NWBs files) and their equivalent pynapple type.
data = nap.load_folder(DATA_DIRECTORY)
data.view

# %%
# The object `data` is a [`Folder`](https://pynapple-org.github.io/pynapple/io.folder/) object that allows easy navigation and interaction with a dataset. 
# In this case, we want to load the NWB file in the folder `/pynapplenwb`. Data are always lazy loaded. No time series is loaded until it's actually called.
# When calling the NWB file, the object `nwb` is an interface to the NWB file. All the data inside the NWB file that are compatible with one of the pynapple objects are shown with their corresponding keys.
nwb = data["sub-A2929"]["ses-A2929-200711"]["pynapplenwb"]["A2929-200711"]
print(nwb)


# %%
# We can individually call each object and they are actually loaded. 
#
# `units` is a [TsGroup](https://pynapple-org.github.io/pynapple/core.ts_group/) object. It allows to group together time series with different timestamps and couple metainformation to each neuron. In this case, the location of where the neuron was recorded has been added when loading the session for the first time.
# We load `units` as `spikes`
spikes = nwb['units']
print(spikes)

# %%
# In this case, the TsGroup holds 15 neurons and it is possible to access, similar to a dictionnary, the spike times of a single neuron: 
neuron_0 = spikes[0]
print(neuron_0)

# %%
# *neuron_0* is a [Ts](https://pynapple-org.github.io/pynapple/core.time_series/#pynapple.core.time_series.Ts) object containing the times of the spikes.

# %%
# The other information about the session is contained in `nwb["epochs"]`. In this case, the start and end of the sleep and wake epochs. If the NWB time intervals contains tags of the epochs, pynapple will try to group them together and return a dictionnary of IntervalSet instead of IntervalSet.
epochs = nwb["epochs"]
print(epochs)


# %%
# Finally this dataset contains tracking of the animal in the environment. `rx`, `ry`, `rz` represent respectively the roll, the yaw and the pitch of the head of the animal. `x` and `z` represent the position of the animal in the horizontal plane while `y` represents the elevation.
# Here we load only the head-direction as a Tsd object.
head_direction = nwb["ry"]
print(head_direction)


# %%
# ***
# Core
# -----------------
# The core functions of pynapple provides many ways to manipulate time series. In this example, spike times are restricted to the wake epoch. Notice how the frequencies change from the original object.

wake_ep = epochs["wake"]

spikes_wake = spikes.restrict(wake_ep)
print(spikes_wake)


# %%
# The same operation can be applied to all time series. 

# In this example, we want all the epochs for which position in `x` is above a certain threhsold. For this we use the function `threshold`.
posx = nwb['x']

threshold = 0.08

posxpositive = posx.threshold(threshold)

plt.figure()
plt.plot(posx)
plt.plot(posxpositive, '.')
plt.axhline(threshold)
plt.xlabel("Time (s)")
plt.ylabel("x")
plt.title("x > {}".format(threshold))
plt.tight_layout()
plt.show()

# %%
# The epochs above the threshold can be accessed through the **time support** of the Tsd object. The time support is an important concept in the pynapple package. It helps the user to define the epochs for which the time serie should be defined. By default, Ts, Tsd and TsGroup objects possess a time support (defined as an IntervalSet). It is recommended to pass the time support when instantiating one of those objects.
epochs_above_thr = posxpositive.time_support
print(epochs_above_thr)


# %%
# ***
# Tuning curves
# -------------
# Let's do a more advanced analysis. Neurons from ADn (group 0 in the `spikes` group object) are know to fire for a particular direction. Therefore, we can compute their tuning curves, i.e. their firing rates as a function of the head-direction of the animal in the horizontal plane (*ry*). To do this, we can use the function [`compute_1d_tuning_curves`](https://pynapple-org.github.io/pynapple/process.tuning_curves/#pynapple.process.tuning_curves.compute_1d_tuning_curves). In this case, the tuning curves are computed over 120 bins and between 0 and 2$\pi$.

tuning_curves = nap.compute_1d_tuning_curves(group=spikes, 
                                             feature=head_direction,                                             
                                             nb_bins=121, 
                                             minmax=(0, 2*np.pi))

print(tuning_curves)

# %%
# We can plot tuning curves in polar plots.

neuron_location = spikes.get_info('location') # to know where the neuron was recorded
plt.figure(figsize=(12,9))

for i,n in enumerate(tuning_curves.columns):
    plt.subplot(3,5,i+1, projection = 'polar')
    plt.plot(tuning_curves[n])
    plt.title(neuron_location[n] + '-' + str(n), fontsize = 18)
    
plt.tight_layout()
plt.show()

# %%
# While ADN neurons show obvious modulation for head-direction, it is not obvious for all CA1 cells. Therefore we want to restrict the remaining of the analyses to only ADN neurons. We can split the `spikes` group with the function [`getby_category`](https://pynapple-org.github.io/pynapple/core.ts_group/#pynapple.core.ts_group.TsGroup.getby_category).

spikes_by_location = spikes.getby_category('location')

print(spikes_by_location['adn'])
print(spikes_by_location['ca1'])

spikes_adn = spikes_by_location['adn']


# %%
# ***
# Correlograms
# ------------
# A classical question with head-direction cells is how pairs stay coordinated across brain states i.e. wake vs sleep (see Peyrache, A., Lacroix, M. M., Petersen, P. C., & Buzsáki, G. (2015). Internally organized mechanisms of the head direction sense. Nature neuroscience, 18(4), 569-575.)
# 
# In this example, this coordination across brain states will be evaluated with cross-correlograms of pairs of neurons. We can call the function [`compute_crosscorrelogram`](https://pynapple-org.github.io/pynapple/process.correlograms/#pynapple.process.correlograms.compute_crosscorrelogram) during both sleep and wake epochs.

cc_wake = nap.compute_crosscorrelogram(group=spikes_adn, 
                                       binsize=20, # ms
                                       windowsize=4000, # ms
                                       ep=epochs['wake'], 
                                       norm=True,
                                       time_units='ms')
                                      
cc_sleep = nap.compute_crosscorrelogram(group=spikes_adn,
                                       binsize=5, # ms
                                       windowsize=400, # ms
                                        ep=epochs['sleep'], 
                                       norm=True,
                                       time_units='ms')

# %%
# From the previous figure, we can see that neurons 0 and 1 fires for opposite directions during wake. Therefore we expect their cross-correlograms to show a trough around 0 time lag, meaning those two neurons do not fire spikes together. A similar trough during sleep for the same pair thus indicates a persistence of their coordination even if the animal is not moving its head.
# mkdocs_gallery_thumbnail_number = 3

xtwake = cc_wake.index.values
xtsleep = cc_sleep.index.values

plt.figure(figsize = (15, 5))
plt.subplot(131, projection = 'polar')
plt.plot(tuning_curves[[0,1]]) # The tuning curves of the pair [0,1]
plt.subplot(132)
plt.fill_between(xtwake, np.zeros_like(xtwake), cc_wake[(0,1)].values, color = 'darkgray')
plt.title('wake')
plt.xlabel("Time (ms)")
plt.ylabel("CC")
plt.subplot(133)
plt.fill_between(xtsleep, np.zeros_like(xtsleep), cc_sleep[(0,1)].values, color = 'lightgrey')
plt.title('sleep')
plt.xlabel("Time (ms)")
plt.ylabel("CC")
plt.tight_layout()
plt.show()

# %%
# ***
# Decoding
# --------
# This last analysis shows how to use the pynapple's decoding function.
# 
# The previous result indicates a persistent coordination of head-direction cells during sleep. Therefore it is possible to decode a virtual head-direction signal even if the animal is not moving its head. 
# This example uses the function [`decode_1d`](https://pynapple-org.github.io/pynapple/process.decoding/#pynapple.process.decoding.decode_1d) which implements bayesian decoding (see : Zhang, K., Ginzburg, I., McNaughton, B. L., & Sejnowski, T. J. (1998). Interpreting neuronal population activity by reconstruction: unified framework with application to hippocampal place cells. Journal of neurophysiology, 79(2), 1017-1044.)
# 
# First we can validate the decoding function with the real position of the head of the animal during wake.

tuning_curves_adn = nap.compute_1d_tuning_curves(spikes_adn,
                                                 head_direction,
                                                 nb_bins=61,
                                                 minmax=(0, 2*np.pi))

decoded, proba_angle = nap.decode_1d(tuning_curves=tuning_curves_adn, 
                                     group=spikes_adn, 
                                     ep=epochs['wake'],                                 
                                     bin_size=0.3, # second
                                     feature=head_direction, 
                                    )
print(decoded)

# %%
# We can plot the decoded head-direction along with the true head-direction.


plt.figure(figsize=(20,5))
plt.plot(head_direction.as_units('s'), label = 'True')
plt.plot(decoded.as_units('s'), label = 'Decoded')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Head-direction (rad)")
plt.show()

# %%
# ***
# Raster
# ------
# Finally we can decode activity during sleep and overlay spiking activity of ADN neurons as a raster plot (in this case only during the first 4 seconds). Pynapple return as well the probability of being in a particular state. We can display it next to the spike train.
# 
# First let's decode during sleep with a bin size of 40 ms.
decoded_sleep, proba_angle_Sleep = nap.decode_1d(tuning_curves=tuning_curves_adn,
                                                 group=spikes_adn, 
                                                 ep=epochs['sleep'],
                                                 bin_size=0.04, # second
                                                 feature=head_direction, 
                                                 )

# %%
# Here we are gonna chain the TsGroup function [`set_info`](https://pynapple-org.github.io/pynapple/core.ts_group/#pynapple.core.ts_group.TsGroup.set_info) and the function [`to_tsd`](https://pynapple-org.github.io/pynapple/core.ts_group/#pynapple.core.ts_group.TsGroup.to_tsd) to flatten the TsGroup and quickly assign to each spikes a corresponding value found in the metadata table. Any columns of the metadata table can be assigned to timestamps in a TsGroup.
# 
# Here the value assign to the spikes comes from the preferred firing direction of the neurons. The following line is a quick way to sort the neurons based on their preferred firing direction
order = np.argsort(np.argmax(tuning_curves_adn.values,0))
print(order)

# %%
# Assigning order as a metadata of TsGroup
spikes_adn.set_info(order=order)
print(spikes_adn)

# %%
# "Flattening" the TsGroup to a Tsd based on `order`.
# It's then very easy to call plot on `tsd_adn` to display the raster
tsd_adn = spikes_adn.to_tsd("order")
print(tsd_adn)

#%%
# Plotting everything
subep = nap.IntervalSet(start=0, end=10, time_units='s')

plt.figure(figsize=(19,10))
plt.subplot(211)
plt.plot(tsd_adn.restrict(subep), '|', markersize=20)
plt.xlim(subep.start[0], subep.end[0])
plt.ylabel("Order")
plt.title("Decoding during sleep")
plt.subplot(212)
p = proba_angle_Sleep.restrict(subep)
plt.imshow(p.values.T, aspect='auto', origin='lower', cmap='viridis')
plt.title("Probability")
plt.xticks([0, p.shape[0]-1], subep.values[0])
plt.yticks([0, p.shape[1]], ['0', '360'])
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Head-direction (deg)")
plt.legend()
plt.show()


