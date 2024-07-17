# -*- coding: utf-8 -*-
"""
Grosmark & Buzsáki (2016) Tutorial 2
============

In the previous [Grosmark & Buzsáki (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4919122/) tutorial,
we learned how to use Pynapple's signal processing tools with Local Field Potential data. Specifically, we
used wavelet decompositions to isolate Theta band activity during active traversal of a linear track,
as well as to find Sharp Wave Ripples which occurred after traversal.

In this tutorial we will learn how to isolate phase information from our wavelet decomposition and combine it
with spiking data, to find phase preferences of spiking units.

Specifically, we will examine LFP and spiking data from a period of REM sleep, after traversal of a linear track.

This tutorial was made by Kipp Freud.
"""

# %%
# !!! warning
#     This tutorial uses matplotlib for displaying the figure
#
#     You can install all with `pip install matplotlib requests tqdm`
#
# First, import the necessary libraries:

import math
import os

# ..todo: remove
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy
import tqdm

import pynapple as nap

matplotlib.use("TkAgg")

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
FS = len(data["eeg"].index[:]) / (data["eeg"].index[-1] - data["eeg"].index[0])
print(data)


# %%
# ***
# Selecting slices
# -----------------------------------
# Let's consider a 10-second slice of data taken during REM sleep

# Define the IntervalSet for this run and instantiate both LFP and
# Position TsdFrame objects
REM_minute_interval = nap.IntervalSet(
    data["rem"]["start"][0] + 90.0,
    data["rem"]["start"][0] + 100.0,
)
REM_Tsd = nap.TsdFrame(
    t=data["eeg"].restrict(REM_minute_interval).index.values
    - data["eeg"].restrict(REM_minute_interval).index.values.min(),
    d=data["eeg"].restrict(REM_minute_interval).values,
)

# We will also extract spike times from all units in our dataset
# which occur during our specified interval
spikes = {}
for i in data["units"].index:
    spikes[i] = (
        data["units"][i].times()[
            (data["units"][i].times() > REM_minute_interval["start"][0])
            & (data["units"][i].times() < REM_minute_interval["end"][0])
        ]
        - data["eeg"].restrict(REM_minute_interval).index.values.min()
    )

# The given dataset has only one channel, so we set channel = 0 here
channel = 0

# %%
# ***
# Plotting the LFP Activity
# -----------------------------------
# We should first plot our REM Local Field Potential data.

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))

ax.plot(
    REM_Tsd[:, channel],
    label="REM LFP Data",
    color="green",
)
ax.set_title("REM Local Field Potential")
ax.set_ylabel("LFP (v)")
ax.set_xlabel("time (s)")
ax.margins(0)
ax.legend()
plt.show()

# %%
# ***
# Getting the Wavelet Decomposition
# -----------------------------------
# As we would expect, it looks like we have a very strong theta oscillation within our data
# - this is a common feature of REM sleep. Let's perform a wavelet decomposition,
# as we did in the last tutorial, to see get a more informative breakdown of the
# frequencies present in the data.

# We must define the frequency set that we'd like to use for our decomposition;
# these have been manually selected based on  the frequencies used in
# Frey et. al (2021), but could also be defined as a linspace or logspace
freqs = np.array(
    [
        2.59,
        3.66,
        5.18,
        8.0,
        10.36,
        20.72,
        29.3,
        41.44,
        58.59,
        82.88,
        117.19,
        152.35,
        192.19,
        200.0,
        234.38,
        270.00,
        331.5,
        390.00,
    ]
)
mwt_REM = nap.compute_wavelet_transform(REM_Tsd[:, channel], fs=None, freqs=freqs)


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
    y_ticks = freqs
    y_ticks_pos = [np.argmin(np.abs(freqs - val)) for val in y_ticks]
    ax.set(yticks=y_ticks_pos, yticklabels=y_ticks)


# And plot it
fig = plt.figure(constrained_layout=True, figsize=(10, 6))
axd = fig.subplot_mosaic(
    [
        ["wd_rem"],
        ["lfp_rem"],
    ],
    height_ratios=[1, 0.2],
)
plot_timefrequency(
    REM_Tsd.index.values[:],
    freqs[:],
    np.transpose(mwt_REM[:, :].values),
    ax=axd["wd_rem"],
)
axd["wd_rem"].set_title(f"Wavelet Decomposition")
axd["lfp_rem"].plot(REM_Tsd)
axd["lfp_rem"].margins(0)
axd["lfp_rem"].set_ylabel("LFP (v)")
axd["lfp_rem"].get_xaxis().set_visible(False)
for spine in ["top", "right", "bottom", "left"]:
    axd["lfp_rem"].spines[spine].set_visible(False)
plt.show()

# %%
# ***
# Visualizing Theta Band Power and Phase
# -----------------------------------
# There seems to be a strong theta frequency present in the data during the maze traversal.
# Let's plot the estimated 8Hz component of the wavelet decomposition on top of our data, and see how well
# they match up. We will also extract and plot the phase of the 8Hz wavelet from the decomposition.
theta_freq_index = 3
theta_band_reconstruction = mwt_REM[:, theta_freq_index].values.real
# calculating phase here
theta_band_phase = np.angle(mwt_REM[:, theta_freq_index].values)

fig = plt.figure(constrained_layout=True, figsize=(10, 5))
axd = fig.subplot_mosaic(
    [
        ["theta_pow"],
        ["phase"],
    ],
    height_ratios=[0.4, 0.2],
)

axd["theta_pow"].plot(
    REM_Tsd.index.values, REM_Tsd[:, channel], alpha=0.5, label="LFP Data - REM"
)
axd["theta_pow"].plot(
    REM_Tsd.index.values,
    theta_band_reconstruction,
    label=f"{freqs[theta_freq_index]}Hz oscillations",
)
axd["theta_pow"].set_ylabel("LFP (v)")
axd["theta_pow"].set_xlabel("Time (s)")
axd["theta_pow"].set_title(f"{freqs[theta_freq_index]}Hz oscillation power.")  #
axd["theta_pow"].legend()
axd["phase"].plot(theta_band_phase)
[axd[k].margins(0) for k in ["theta_pow", "phase"]]
axd["phase"].set_ylabel("Phase")
plt.show()


# %%
# ***
# Finding Phase of Spikes
# -----------------------------------
# Now that we have the phase of our theta wavelet, and our spike times, we can find the theta phase at which every
# spike occurs

# We will start by throwing away cells which do not have enough
# spikes during our interval
spikes = {k: v for k, v in spikes.items() if len(v) > 20}
# Get phase of each spike
phase = {}
for i in spikes.keys():
    phase_i = []
    for spike in spikes[i]:
        phase_i.append(
            np.angle(
                mwt_REM[
                    np.argmin(np.abs(REM_Tsd.index.values - spike)), theta_freq_index
                ]
            )
        )
    phase[i] = np.array(phase_i)

# Let's plot phase histograms for the first six units to see if there's
# any obvious preferences
fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 6))
for ri in range(2):
    for ci in range(3):
        ax[ri, ci].hist(
            phase[list(phase.keys())[ri * 3 + ci]],
            bins=np.linspace(-np.pi, np.pi, 10),
            density=True,
        )
        ax[ri, ci].set_xlabel("Phase (rad)")
        ax[ri, ci].set_ylabel("Density")
        ax[ri, ci].set_title(f"Unit {list(phase.keys())[ri*3 + ci]}")
fig.suptitle("Phase Preference Histograms of First 6 Units")
plt.show()

# %%
# ***
# Isolating Strong Phase Preferences
# -----------------------------------
# It looks like there could be some phase preferences happening here, but there's a lot of cells to go through.
# Now that we have our phases of firing for each unit, we can sort the units by the circular variance of the phase
# of their spikes, to isolate the cells with the strongest phase preferences without manual inspection.

variances = {
    key: scipy.stats.circvar(value, low=-np.pi, high=np.pi)
    for key, value in phase.items()
}
spikes = dict(sorted(spikes.items(), key=lambda item: variances[item[0]]))
phase = dict(sorted(phase.items(), key=lambda item: variances[item[0]]))

# Now let's plot phase histograms for the six units with the least
# varied phase of spikes.
fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 6))
for ri in range(2):
    for ci in range(3):
        ax[ri, ci].hist(
            phase[list(phase.keys())[ri * 3 + ci]],
            bins=np.linspace(-np.pi, np.pi, 10),
            density=True,
        )
        ax[ri, ci].set_xlabel("Phase (rad)")
        ax[ri, ci].set_ylabel("Density")
        ax[ri, ci].set_title(f"Unit {list(phase.keys())[ri*3 + ci]}")
fig.suptitle(
    "Phase Preference Histograms of 6 Units with " + "Highest Phase Preference"
)
plt.show()

# %%
# ***
# Visualizing Phase Preferences
# -----------------------------------
# There is definitely some strong phase preferences happening here. Let's visualize the firing preferences
# of the 6 cells we've isolated to get an impression of just how striking these preferences are.

fig = plt.figure(constrained_layout=True, figsize=(10, 12))
axd = fig.subplot_mosaic(
    [
        ["lfp_run"],
        ["phase_0"],
        ["phase_1"],
        ["phase_2"],
        ["phase_3"],
        ["phase_4"],
        ["phase_5"],
    ],
    height_ratios=[0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
)
[axd[k].margins(0) for k in ["lfp_run"] + [f"phase_{i}" for i in range(6)]]
axd["lfp_run"].plot(
    REM_Tsd.index.values, REM_Tsd[:, channel], alpha=0.5, label="LFP Data - REM"
)
axd["lfp_run"].plot(
    REM_Tsd.index.values,
    theta_band_reconstruction,
    label=f"{freqs[theta_freq_index]}Hz oscillations",
)
axd["lfp_run"].set_ylabel("LFP (v)")
axd["lfp_run"].set_xlabel("Time (s)")
axd["lfp_run"].set_title(f"{freqs[theta_freq_index]}Hz oscillation power.")
axd["lfp_run"].legend()
for i in range(6):
    axd[f"phase_{i}"].plot(REM_Tsd.index.values, theta_band_phase, alpha=0.2)
    axd[f"phase_{i}"].scatter(
        spikes[list(spikes.keys())[i]], phase[list(spikes.keys())[i]]
    )
    axd[f"phase_{i}"].set_ylabel("Phase")
    axd[f"phase_{i}"].set_title(f"Unit {list(spikes.keys())[i]}")
fig.suptitle("Phase Preference Visualizations")
plt.show()
