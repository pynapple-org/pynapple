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
FS = 1250
print(data)


# %%
# ***
# Selecting slices
# -----------------------------------
# We will consider a single run of the experiment - where the rodent completes a full traversal of the linear track,
# followed by 4 seconds of post-traversal activity.

# Define the run to use for this Analysis
run_index = 7
# Define the IntervalSet for this run and instantiate both LFP and
# Position TsdFrame objects
RUN_interval = nap.IntervalSet(
    data["forward_ep"]["start"][run_index],
    data["forward_ep"]["end"][run_index] + 4.0,
)
RUN_Tsd = data["eeg"].restrict(RUN_interval)
RUN_pos = data["position"].restrict(RUN_interval)
print(RUN_Tsd)

# %%
# ***
# Plotting the LFP and Behavioural Activity
# -----------------------------------

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
axd = fig.subplot_mosaic(
    [["ephys"], ["pos"]],
    height_ratios=[1, 0.4],
)

axd["ephys"].plot(
    RUN_Tsd[:, 0].restrict(
        nap.IntervalSet(
            data["forward_ep"]["start"][run_index], data["forward_ep"]["end"][run_index]
        )
    ),
    label="Traversal LFP Data",
    color="green",
)
axd["ephys"].plot(
    RUN_Tsd[:, 0].restrict(
        nap.IntervalSet(
            data["forward_ep"]["end"][run_index],
            data["forward_ep"]["end"][run_index] + 5.0,
        )
    ),
    label="Post Traversal LFP Data",
    color="blue",
)
axd["ephys"].set_title("Traversal & Post Traversal LFP")
axd["ephys"].set_ylabel("LFP (v)")
axd["ephys"].set_xlabel("time (s)")
axd["ephys"].margins(0)
axd["ephys"].legend()
axd["pos"].plot(RUN_pos, color="black")
axd["pos"].margins(0)
axd["pos"].set_xlabel("time (s)")
axd["pos"].set_ylabel("Linearized Position")
axd["pos"].set_xlim(RUN_Tsd.index.min(), RUN_Tsd.index.max())


# %%
# ***
# Getting the LFP Spectrogram
# -----------------------------------
# Let's take the Fourier transform of our data to get an initial insight into the dominant frequencies.

fft = nap.compute_power_spectral_density(RUN_Tsd, fs=int(FS))
print(fft)

# %%
# ***
# The returned object is a pandas dataframe which uses frequencies as indexes and spectral power as values.
#
# Now let's plot it

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 4))
ax.plot(
    fft.index,
    np.abs(fft.iloc[:, 0]),
    alpha=0.5,
    label="LFP Frequency Power",
    c="blue",
    linewidth=2,
)
ax.set_xlabel("Freq (Hz)")
ax.set_ylabel("Frequency Power")
ax.set_title("LFP Fourier Decomposition")
ax.set_xlim(1, 30)
ax.axvline(9.36, c="orange", label="9.36Hz", alpha=0.5)
ax.axvline(18.74, c="green", label="18.74Hz", alpha=0.5)
ax.legend()
# ax.set_yscale('log')
# ax.set_xscale('log')

# %%
# ***
# Getting the Wavelet Decomposition
# -----------------------------------
# It looks like the prominent frequencies in the data may vary over time. For example, it looks like the
# LFP characteristics may be different while the animal is running along the track, and when it is finished.
# Let's generate a wavelet decomposition to look more closely at the changing frequency powers over time.

# We must define the frequency set that we'd like to use for our decomposition
freqs = np.geomspace(5, 250, 25)
# Compute and print the wavelet transform on our LFP data
mwt_RUN = nap.compute_wavelet_transform(RUN_Tsd[:, 0], fs=FS, freqs=freqs)

# %%
# ***
# Now let's plot it.


# Define wavelet decomposition plotting function
def plot_timefrequency(times, freqs, powers, x_ticks=5, ax=None, **kwargs):
    if np.iscomplexobj(powers):
        powers = abs(powers)
    ax.imshow(powers, aspect="auto", **kwargs)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if isinstance(x_ticks, int):
        x_tick_pos = np.linspace(0, times.size, x_ticks)
        x_ticks = np.round(np.linspace(times[0], times[-1], x_ticks), 2)
    else:
        x_tick_pos = [np.argmin(np.abs(times - val)) for val in x_ticks]
    ax.set(xticks=x_tick_pos, xticklabels=x_ticks)
    y_ticks = [np.round(f, 2) for f in freqs]
    y_ticks_pos = [np.argmin(np.abs(freqs - val)) for val in y_ticks]
    ax.set(yticks=y_ticks_pos, yticklabels=y_ticks)
    ax.grid(False)


# And plot
fig = plt.figure(constrained_layout=True, figsize=(10, 8))
axd = fig.subplot_mosaic(
    [
        ["wd_run"],
        ["lfp_run"],
        ["pos_run"],
    ],
    height_ratios=[1.2, 0.2, 0.6],
)
plot_timefrequency(
    RUN_Tsd.index.values[:],
    freqs[:],
    np.transpose(mwt_RUN[:, :].values),
    ax=axd["wd_run"],
)
axd["wd_run"].set_title(f"Wavelet Decomposition")
axd["lfp_run"].plot(RUN_Tsd)
axd["pos_run"].plot(RUN_pos)
axd["pos_run"].set_xlim(RUN_Tsd.index.min(), RUN_Tsd.index.max())
axd["pos_run"].set_ylabel("Lin. Position (cm)")
for k in ["lfp_run", "pos_run"]:
    axd[k].margins(0)
    if k != "pos_run":
        axd[k].set_ylabel("LFP (v)")
    axd[k].get_xaxis().set_visible(False)
    for spine in ["top", "right", "bottom", "left"]:
        axd[k].spines[spine].set_visible(False)

# %%
# ***
# Visualizing Theta Band Power
# -----------------------------------
# There seems to be a strong theta frequency present in the data during the maze traversal.
# Let's plot the estimated 8Hz component of the wavelet decomposition on top of our data, and see how well
# they match up

# Find the index of the frequency closest to theta band
theta_freq_index = np.argmin(np.abs(10 - freqs))
# Extract its real component, as well as its power envelope
theta_band_reconstruction = mwt_RUN[:, theta_freq_index].values.real
theta_band_power_envelope = np.abs(mwt_RUN[:, theta_freq_index].values)


# %%
# ***
# Now let's visualise the theta band component of the signal over time.

fig = plt.figure(constrained_layout=True, figsize=(10, 6))
axd = fig.subplot_mosaic(
    [
        ["lfp_run"],
        ["pos_run"],
    ],
    height_ratios=[1, 0.4],
)

axd["lfp_run"].plot(RUN_Tsd.index.values, RUN_Tsd[:, 0], alpha=0.5, label="LFP Data")
axd["lfp_run"].plot(
    RUN_Tsd.index.values,
    theta_band_reconstruction,
    label=f"{np.round(freqs[theta_freq_index], 2)}Hz oscillations",
)
axd["lfp_run"].plot(
    RUN_Tsd.index.values,
    theta_band_power_envelope,
    label=f"{np.round(freqs[theta_freq_index], 2)}Hz power envelope",
)

axd["lfp_run"].set_ylabel("LFP (v)")
axd["lfp_run"].set_xlabel("Time (s)")
axd["lfp_run"].set_title(f"{np.round(freqs[theta_freq_index], 2)}Hz oscillation power.")
axd["pos_run"].plot(RUN_pos)
[axd[k].margins(0) for k in ["lfp_run", "pos_run"]]
[
    axd["pos_run"].spines[sp].set_visible(False)
    for sp in ["top", "right", "bottom", "left"]
]
axd["pos_run"].get_xaxis().set_visible(False)
axd["pos_run"].set_xlim(RUN_Tsd.index.min(), RUN_Tsd.index.max())
axd["pos_run"].set_ylabel("Lin. Position (cm)")
axd["lfp_run"].legend()

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
axd["lfp_run"].plot(RUN_Tsd.index.values, RUN_Tsd[:, 0], label="LFP Data")
axd["lfp_run"].set_ylabel("LFP (v)")
axd["lfp_run"].set_xlabel("Time (s)")
axd["lfp_run"].margins(0)
axd["lfp_run"].set_title(f"Traversal & Post-Traversal LFP")
axd["rip_pow"].plot(RUN_Tsd.index.values, ripple_power)
axd["rip_pow"].margins(0)
axd["rip_pow"].get_xaxis().set_visible(False)
axd["rip_pow"].spines["top"].set_visible(False)
axd["rip_pow"].spines["right"].set_visible(False)
axd["rip_pow"].spines["bottom"].set_visible(False)
axd["rip_pow"].spines["left"].set_visible(False)
axd["rip_pow"].set_xlim(RUN_Tsd.index.min(), RUN_Tsd.index.max())
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
axd["lfp_run"].plot(RUN_Tsd[:, 0], label="LFP Data")
axd["rip_pow"].plot(smoother_swr_power)
axd["rip_pow"].plot(is_ripple, color="red", linewidth=2)
axd["lfp_run"].set_ylabel("LFP (v)")
axd["lfp_run"].set_xlabel("Time (s)")
axd["lfp_run"].set_title(f"{np.round(freqs[ripple_freq_idx], 2)}Hz oscillation power.")
axd["rip_pow"].axhline(threshold)
[axd[k].margins(0) for k in ["lfp_run", "rip_pow"]]
[axd["rip_pow"].spines[sp].set_visible(False) for sp in ["top", "left", "right"]]
axd["rip_pow"].get_xaxis().set_visible(False)
axd["rip_pow"].set_xlim(RUN_Tsd.index.min(), RUN_Tsd.index.max())
axd["rip_pow"].set_ylabel(f"{np.round(freqs[ripple_freq_idx], 2)}Hz Power")

# %%
# ***
# Plotting a Sharp Wave Ripple
# -----------------------------------
# Let's zoom in on out detected ripples and have a closer look!

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 4))
buffer = 0.1
ax.plot(
    RUN_Tsd.restrict(
        nap.IntervalSet(
            start=is_ripple.index.min() - buffer, end=is_ripple.index.max() + buffer
        )
    ),
    color="blue",
    label="Non-SWR LFP",
)
ax.plot(
    RUN_Tsd.restrict(
        nap.IntervalSet(start=is_ripple.index.min(), end=is_ripple.index.max())
    ),
    color="red",
    label="SWR",
    linewidth=2,
)
ax.margins(0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("LFP (v)")
ax.legend()
ax.set_title("Sharp Wave Ripple Visualization")
