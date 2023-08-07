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
dandiset_id, filepath = "000003", "sub-YutaMouse56/sub-YutaMouse56_ses-YutaMouse56-160911_behavior+ecephys.nwb"


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
import numpy as np
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette = "colorblind", font_scale=1.5, rc=custom_params)

nwb = nap.NWBFile(io.read())

print(nwb)



# %%
# We can load the spikes as a TsGroup for inspection.

units = nwb['units']

print(units)
# %%
# Let's get all the neurons whose spikes sufficiently

units = units.getby_threshold("rate", 0.5)


# %%
# 
# Let's do some cross-corrs during wake and non-REM sleep
# We start by retrieveing the epochs
epochs = nwb["states"]

print(epochs)

# In this case, it's a dictionnary of IntervalSet
wake_ep = epochs["awake"]
nrem_ep = epochs["nrem"]


# %%
# Let's replicate one result of the study. The authors report cross-correlograms between mossy cells and granule cells as a function of brain states (wake_ep vs nrem_ep in this case).
# First let's retrieve both sub-population
mossy_cells = units.getby_category("cell_type")["mossy cell"]
granule_cells = units.getby_category("cell_type")["granule cell"]

print(mossy_cells)


# %%
# Let's compute their cross-correlogram during wake and nREM sleep.
# The order in the tuple matters. In this case, granule cell is the reference unit.
cc_wake = nap.compute_crosscorrelogram((granule_cells, mossy_cells), 0.002, 0.2, ep=wake_ep, norm=True)
cc_nrem = nap.compute_crosscorrelogram((granule_cells, mossy_cells), 0.002, 0.2, ep=nrem_ep, norm=True)

plt.figure(figsize=(16,10))
gs = plt.GridSpec(len(mossy_cells), len(granule_cells))
for i, n in enumerate(mossy_cells.keys()):
    for j, k in enumerate(granule_cells.keys()):
        p = (k,n)
        plt.subplot(gs[i,j])
        tidx = cc_wake[p].index.values
        plt.fill_between(tidx, np.zeros_like(cc_wake[p]), cc_wake[p].values, color = 'lightgrey')
        plt.plot()
        plt.grid()
        plt.title(p)
        if i == len(mossy_cells)-1: plt.xlabel("Time (s)")
        if j == 0: plt.ylabel("Norm.")

plt.tight_layout()
plt.show()

# %%
# TODO : Run the stats for the connection, find a good example, display NREM

