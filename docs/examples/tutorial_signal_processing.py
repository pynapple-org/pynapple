# -*- coding: utf-8 -*-
"""
Computing Wavelet Transform
============
This tutorial demonstrates how we can use the signal processing tools within Pynapple to aid with data analysis.
We will examine the dataset from [Grosmark & Buzsáki (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4919122/).

Specifically, we will examine Local Field Potential data from a period of active traversal of a linear track.

This tutorial was made by Kipp Freud.

"""


# %%
# !!! warning
#     This tutorial uses matplotlib for displaying the figure
#
#     You can install all with `pip install matplotlib requests tqdm seaborn`
#
# mkdocs_gallery_thumbnail_number = 7
#
# First, import the necessary libraries:

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn
import tqdm

seaborn.set_theme()

import pynapple as nap

# %%
# ***
# Downloading the data
# ------------------
# Let's download the data and save it locally

path = "Achilles_10252013_EEG.nwb"
if path not in os.listdir("."):
    r = requests.get(f"https://osf.io/2dfvp/download", stream=True)
    block_size = 1024 * 1024
    with open(path, "wb") as f:
        for data in tqdm.tqdm(
            r.iter_content(block_size),
            unit="MB",
            unit_scale=True,
            total=math.ceil(int(r.headers.get("content-length", 0)) // block_size),
        ):
            f.write(data)


# %%
# ***
# Loading the data
# ------------------
# Let's load and print the full dataset.

data = nap.load_file(path)

print(data)

# %%
# First we can extract the data from the NWB. The local field potential has been downsampled to 1250Hz. We will call it `eeg`.
#
# The `time_support` of the object `data['position']` contains the interval for which the rat was running along the linear track. We will call it `wake_ep`.
#

FS = 1250

eeg = data["eeg"]

wake_ep = data["position"].time_support

# %%
# ***
# Selecting example
# -----------------------------------
# We will consider a single run of the experiment - where the rodent completes a full traversal of the linear track,
# followed by 4 seconds of post-traversal activity.

forward_ep = data["forward_ep"]
RUN_interval = nap.IntervalSet(forward_ep.start[7], forward_ep.end[7] + 4.0)

eeg_example = eeg.restrict(RUN_interval)[:, 0]
pos_example = data["position"].restrict(RUN_interval)

# %%
# ***
# Plotting the LFP and Behavioural Activity
# -----------------------------------

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
axd = fig.subplot_mosaic(
    [["ephys"], ["pos"]],
    height_ratios=[1, 0.4],
)
axd["ephys"].plot(eeg_example, label="CA1")
axd["ephys"].set_title("EEG (1250 Hz)")
axd["ephys"].set_ylabel("LFP (a.u.)")
axd["ephys"].set_xlabel("time (s)")
axd["ephys"].margins(0)
axd["ephys"].legend()
axd["pos"].plot(pos_example, color="black")
axd["pos"].margins(0)
axd["pos"].set_xlabel("time (s)")
axd["pos"].set_ylabel("Linearized Position")
axd["pos"].set_xlim(RUN_interval[0, 0], RUN_interval[0, 1])


# %%
# ***
# Getting the LFP Spectrogram
# -----------------------------------
# Let's take the Fourier transform of our data to get an initial insight into the dominant frequencies during exploration (`wake_ep`).


power = nap.compute_power_spectral_density(eeg, fs=FS, ep=wake_ep)
print(power)


# %%
# ***
# The returned object is a pandas dataframe which uses frequencies as indexes and spectral power as values.
#
# Let's plot the power between 1 and 100 Hz.
#
# The red area outlines the theta rhythm (6-12 Hz) which is proeminent in hippocampal LFP.

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 4))
ax.semilogy(
    np.abs(power[(power.index > 1.0) & (power.index < 100)]),
    alpha=0.5,
    label="LFP Frequency Power",
)
ax.axvspan(6, 12, color="red", alpha=0.1)
ax.set_xlabel("Freq (Hz)")
ax.set_ylabel("Frequency Power")
ax.set_title("LFP Fourier Decomposition")
ax.legend()

# %%
# ***
# Getting the Wavelet Decomposition
# -----------------------------------
# It looks like the prominent frequencies in the data may vary over time. For example, it looks like the
# LFP characteristics may be different while the animal is running along the track, and when it is finished.
# Let's generate a wavelet decomposition to look more closely at the changing frequency powers over time.

# We must define the frequency set that we'd like to use for our decomposition
freqs = np.geomspace(3, 250, 100)
# Compute and print the wavelet transform on our LFP data
mwt_RUN = nap.compute_wavelet_transform(eeg_example, fs=FS, freqs=freqs)


# %%
# `mwt_RUN` is a TsdFrame with each column being the convolution with one wavelet at a particular frequency.
print(mwt_RUN)

# %%
# ***
# Now let's plot it.

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
gs = plt.GridSpec(3, 1, figure=fig, height_ratios=[1.0, 0.5, 0.1])

ax0 = plt.subplot(gs[0, 0])
pcmesh = ax0.pcolormesh(mwt_RUN.t, freqs, np.transpose(np.abs(mwt_RUN)))
ax0.grid(False)
ax0.set_yscale("log")
ax0.set_title("Wavelet Decomposition")
cbar = plt.colorbar(pcmesh, ax=ax0, orientation="vertical")
ax0.set_label("Amplitude")

ax1 = plt.subplot(gs[1, 0], sharex=ax0)
ax1.plot(eeg_example)
ax1.set_ylabel("LFP (v)")

ax1 = plt.subplot(gs[2, 0], sharex=ax0)
ax1.plot(pos_example, color="black")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Pos.")


# %%
# ***
# Visualizing Theta Band Power
# -----------------------------------
# There seems to be a strong theta frequency present in the data during the maze traversal.
# Let's plot the estimated 8Hz component of the wavelet decomposition on top of our data, and see how well
# they match up

# Find the index of the frequency closest to theta band
theta_freq_index = np.argmin(np.abs(8 - freqs))
# Extract its real component, as well as its power envelope
theta_band_reconstruction = mwt_RUN[:, theta_freq_index].values.real
theta_band_power_envelope = np.abs(mwt_RUN[:, theta_freq_index].values)


# %%
# ***
# Now let's visualise the theta band component of the signal over time.

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
axd = fig.subplot_mosaic(
    [["ephys"], ["pos"]],
    height_ratios=[1, 0.4],
)
axd["ephys"].plot(eeg_example, label="CA1")
axd["ephys"].plot(
    eeg_example.index.values,
    theta_band_reconstruction,
    label=f"{np.round(freqs[theta_freq_index], 2)}Hz oscillations",
)
axd["ephys"].plot(
    eeg_example.index.values,
    theta_band_power_envelope,
    label=f"{np.round(freqs[theta_freq_index], 2)}Hz power envelope",
)
axd["ephys"].set_title("EEG (1250 Hz)")
axd["ephys"].set_ylabel("LFP (a.u.)")
axd["ephys"].set_xlabel("time (s)")
axd["ephys"].margins(0)
axd["ephys"].legend()
axd["pos"].plot(pos_example, color="black")
axd["pos"].margins(0)
axd["pos"].set_xlabel("time (s)")
axd["pos"].set_ylabel("Linearized Position")
axd["pos"].set_xlim(RUN_interval[0, 0], RUN_interval[0, 1])


# %%
# ***
# Visualizing Sharp Wave Ripple Power
# -----------------------------------
# There also seem to be peaks in the 200Hz frequency power after traversal of thew maze is complete.
# Let's plot the LFP, along with the 200Hz frequency power, to see if we can isolate these peaks and
# see what's going on.

# Find the index of the frequency closest to sharp wave ripple oscillations
ripple_freq_idx = np.argmin(np.abs(200 - freqs))
# Extract its power envelope
ripple_power = np.abs(mwt_RUN[:, ripple_freq_idx].values)


# %%
# ***
# Now let's visualise the 200Hz component of the signal over time.

fig = plt.figure(constrained_layout=True, figsize=(10, 5))
axd = fig.subplot_mosaic(
    [
        ["lfp_run"],
        ["rip_pow"],
    ],
    height_ratios=[1, 0.4],
)
axd["lfp_run"].plot(eeg_example, label="LFP Data")
axd["rip_pow"].plot(eeg_example.index.values, ripple_power)
axd["lfp_run"].set_ylabel("LFP (v)")
axd["lfp_run"].set_xlabel("Time (s)")
axd["lfp_run"].margins(0)
axd["lfp_run"].set_title(f"EEG (1250 Hz)")
axd["rip_pow"].margins(0)
axd["rip_pow"].set_xlim(eeg_example.index.min(), eeg_example.index.max())
axd["rip_pow"].set_ylabel(f"{np.round(freqs[ripple_freq_idx], 2)}Hz Power")

# %%
# ***
# Isolating Ripple Times
# -----------------------------------
# We can see one significant peak in the 200Hz frequency power. Let's smooth the power curve and threshold
# to try to isolate this event.

# Define threshold
threshold = 6000
# Smooth wavelet power TsdFrame at the SWR frequency
smoother_swr_power = (
    mwt_RUN[:, ripple_freq_idx]
    .abs()
    .smooth(std=0.025, windowsize=0.2, time_units="s", norm=False)
)
# Threshold our TsdFrame
is_ripple = smoother_swr_power.threshold(threshold)


# %%
# ***
# Now let's plot the threshold ripple power over time.

fig = plt.figure(constrained_layout=True, figsize=(10, 5))
axd = fig.subplot_mosaic(
    [
        ["lfp_run"],
        ["rip_pow"],
    ],
    height_ratios=[1, 0.4],
)
axd["lfp_run"].plot(eeg_example, label="LFP Data")
axd["rip_pow"].plot(smoother_swr_power)
axd["rip_pow"].axvspan(
    is_ripple.index.min(), is_ripple.index.max(), color="red", alpha=0.3
)
axd["lfp_run"].set_ylabel("LFP (v)")
axd["lfp_run"].set_xlabel("Time (s)")
axd["lfp_run"].set_title(f"EEG (1250 Hz)")
axd["rip_pow"].axhline(threshold, linestyle="--", color="black", alpha=0.4)
[axd[k].margins(0) for k in ["lfp_run", "rip_pow"]]
axd["rip_pow"].set_xlim(eeg_example.index.min(), eeg_example.index.max())
axd["rip_pow"].set_ylabel(f"{np.round(freqs[ripple_freq_idx], 2)}Hz Power")

# %%
# ***
# Plotting a Sharp Wave Ripple
# -----------------------------------
# Let's zoom in on out detected ripples and have a closer look!

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 4))
buffer = 0.1
ax.plot(
    eeg_example.restrict(
        nap.IntervalSet(
            start=is_ripple.index.min() - buffer, end=is_ripple.index.max() + buffer
        )
    ),
    color="blue",
    label="Non-SWR LFP",
)
ax.axvspan(
    is_ripple.index.min(),
    is_ripple.index.max(),
    color="red",
    alpha=0.3,
    label="SWR LFP",
)
ax.margins(0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("LFP (v)")
ax.legend()
ax.set_title("Sharp Wave Ripple Visualization")
