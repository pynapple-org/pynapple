# coding: utf-8
"""
Streaming data from Dandi
=========================

This script shows how to stream data from the [dandi archive](https://dandiarchive.org/) all the way to pynapple.

> Warning **This tutorial is still under construction.**

"""
# %%
# !!! warning
#     This tutorial uses seaborn and matplotlib for displaying the figure as well as the dandi package
#
#     You can install all with `pip install matplotlib seaborn dandi dandischema`

# %% 
# ***
# Prelude
# -------
#
# The data used in this tutorial are hosted by the [DANDI](https://dandiarchive.org/dandiset/000003?page=3&sortOption=0&sortDir=-1&showDrafts=true&showEmpty=false&pos=21) archive and were used in this publication:
# __Senzai, Yuta, and György Buzsáki. "Physiological properties and behavioral correlates of hippocampal granule cells and mossy cells." Neuron 93.3 (2017): 691-704.__
#


# %% 
# ***
# Dandi
# -----
# Dandi allows you to stream data without downloading all the files. In this case the data extracted from the NWB file are stored in the nwb-cache folder.

from pynwb import NWBHDF5IO

from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py


# ecephys, Buzsaki Lab (15.2 GB)
dandiset_id, filepath = "000003", "sub-YutaMouse42/sub-YutaMouse42_ses-YutaMouse42-151106_behavior+ecephys.nwb"

with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

# first, create a virtual filesystem based on the http protocol
fs=fsspec.filesystem("http")

# create a cache to save downloaded data to disk (optional)
fs = CachingFileSystem(
    fs=fs,
    cache_storage="nwb-cache",  # Local folder for the cache
    )

# next, open the file
file = h5py.File(fs.open(s3_url, "rb"))
io = NWBHDF5IO(file=file, load_namespaces=True)

print(io)

# %%
# Pynapple
# --------
# If opening the NWB works, you can start streaming data straight into pynapple with the `NWBFile` class.

import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette = "colorblind", font_scale=1.5, rc=custom_params)

nwb = nap.NWBFile(io.read())

print(nwb)


# %%
# We can load the spikes as a TsGroup for inspection.

units = nwb['units']

print(units)


# %%
# 
# Let's do some cross-corrs during wake and non-REM sleep
# We start by retrieveing the epochs
epochs = nwb["states"]

print(epochs)

# In this case, it's a dictionnary of IntervalSet
wake_ep = epochs["awake"]
nrem_ep = epochs["nrem"]


print(mossy)


# %%
# Let's compute their cross-correlogram
cc = nap.compute_crosscorrelogram(mossy, 0.1, 2, norm=True)


plt.figure()
plt.plot(cc)
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Norm.")
plt.title("Cross-correlogram of Mossy cells")
plt.show()


# %%


stim_ts = nwb['PulseStim_5V_500ms_LD9999']

print(stim_ts)


# %%


# peth = nap.compute_perievent(mossy[19], stim_ts, (-0.04, 0.1))

# plt.figure()
# plt.subplot(211)
# plt.plot(peth.count(0.01).as_dataframe().sum(1))
# plt.axvline(0)
# plt.subplot(212)
# plt.plot(peth.to_tsd(), '|')
# plt.axvline(0)
# plt.show()


# # %%


# mossy_cell = units.getby_category("cell_type")["mossy cell"]
# granule_cell = units.getby_category("cell_type")["granule cell"]

# cc_mossy = nap.compute_eventcorrelogram(mossy_cell, stim_ts, 0.01, 0.4, norm=True)
# cc_granule = nap.compute_eventcorrelogram(granule_cell, stim_ts, 0.01, 0.4, norm=True)



# plt.figure()
# plt.subplot(121)
# plt.plot(cc_mossy, alpha = 0.1)
# plt.plot(cc_mossy.mean(1))
# plt.xlabel("Time from Opto Stimulation (s)")
# plt.ylabel("Norm. firing rate")
# plt.subplot(122)
# plt.plot(cc_granule, alpha = 0.1)
# plt.plot(cc_granule.mean(1))
# plt.xlabel("Time from Opto Stimulation (s)")
# plt.ylabel("Norm. firing rate")
# plt.show()

