# -*- coding: utf-8 -*-
"""
Computing Phase Preferences
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
#     You can install all with `pip install matplotlib requests tqdm seaborn`
#
# First, import the necessary libraries:

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
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
FS = 1250  # We know from the methods of the paper
print(data)


# %%
# ***
# Selecting slices
# -----------------------------------
# Let's consider a 10-second slice of data taken during REM sleep

# Define the IntervalSet for this run and instantiate both LFP and
# Position TsdFrame objects
REM_minute_interval = nap.IntervalSet(
    data["rem"]["start"][0] + 95.0,
    data["rem"]["start"][0] + 100.0,
)
REM_Tsd = data["eeg"].restrict(REM_minute_interval)

# We will also extract spike times from all units in our dataset
# which occur during our specified interval
spikes = data["units"].restrict(REM_minute_interval)

# %%
# ***
# Plotting the LFP Activity
# -----------------------------------
# We should first plot our REM Local Field Potential data.

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 3))
ax.plot(
    REM_Tsd,
    label="REM LFP Data",
    color="blue",
)
ax.set_title("REM Local Field Potential")
ax.set_ylabel("LFP (v)")
ax.set_xlabel("time (s)")
ax.margins(0)
ax.legend()

# %%
# ***
# Getting the Wavelet Decomposition
# -----------------------------------
# As we would expect, it looks like we have a very strong theta oscillation within our data
# - this is a common feature of REM sleep. Let's perform a wavelet decomposition,
# as we did in the last tutorial, to see get a more informative breakdown of the
# frequencies present in the data.


# We must define the frequency set that we'd like to use for our decomposition
freqs = np.geomspace(5, 200, 25)
# Compute the wavelet transform on our LFP data
mwt_REM = nap.compute_wavelet_transform(REM_Tsd[:, 0], fs=FS, freqs=freqs)

# %%
# ***
# Now let's plot the calculated wavelet scalogram.


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

# %%
# ***
# Visualizing Theta Band Power and Phase
# -----------------------------------
# There seems to be a strong theta frequency present in the data during the maze traversal.
# Let's plot the estimated 7Hz component of the wavelet decomposition on top of our data, and see how well
# they match up. We will also extract and plot the phase of the 7Hz wavelet from the decomposition.
theta_freq_index = np.argmin(np.abs(7 - freqs))
theta_band_reconstruction = mwt_REM[:, theta_freq_index].values.real
# calculating phase here
theta_band_phase = nap.Tsd(
    t=mwt_REM.index, d=np.angle(mwt_REM[:, theta_freq_index].values)
)

# %%
# ***
# Now let's plot the theta power and phase, along with the LFP.

fig = plt.figure(constrained_layout=True, figsize=(10, 5))
axd = fig.subplot_mosaic(
    [
        ["theta_pow"],
        ["phase"],
    ],
    height_ratios=[0.4, 0.2],
)

axd["theta_pow"].plot(REM_Tsd, alpha=0.5, label="LFP Data - REM")
axd["theta_pow"].plot(
    REM_Tsd.index.values,
    theta_band_reconstruction,
    label=f"{np.round(freqs[theta_freq_index], 2)}Hz oscillations",
)
axd["theta_pow"].set_ylabel("LFP (v)")
axd["theta_pow"].set_xlabel("Time (s)")
axd["theta_pow"].set_title(
    f"{np.round(freqs[theta_freq_index],2)}Hz oscillation power."
)  #
axd["theta_pow"].legend()
axd["phase"].plot(REM_Tsd.index.values, theta_band_phase, alpha=0.5)
[axd[k].margins(0) for k in ["theta_pow", "phase"]]
axd["phase"].set_ylabel("Phase")
axd["phase"].get_xaxis().set_visible(False)


# %%
# ***
# Finding Phase of Spikes
# -----------------------------------
# Now that we have the phase of our theta wavelet, and our spike times, we can find the phase firing preferences
# of each of the units using the compute_1d_tuning_curves function.
#
# We will start by throwing away cells which do not have a high enough firing rate during our interval.

# Filter units based on firing rate
spikes = spikes[spikes.rate > 5.0]
# Calculate theta phase firing preferences
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=theta_band_phase, nb_bins=61, minmax=(-np.pi, np.pi)
)

# %%
# ***
# Now we will plot these preferences as smoothed angular histograms. We will select the first 6 units
# to plot.


def smoothAngularTuningCurves(tuning_curves, sigma=2):
    tmp = np.concatenate(
        (tuning_curves.values, tuning_curves.values, tuning_curves.values)
    )
    tmp = scipy.ndimage.gaussian_filter1d(tmp, sigma=sigma, axis=0)
    return pd.DataFrame(
        index=tuning_curves.index,
        data=tmp[tuning_curves.shape[0] : tuning_curves.shape[0] * 2],
        columns=tuning_curves.columns,
    )


smoothcurves = smoothAngularTuningCurves(tuning_curves, sigma=2)
fig, axd = plt.subplot_mosaic(
    [["phase_0", "phase_1", "phase_2"], ["phase_3", "phase_4", "phase_5"]],
    constrained_layout=True,
    figsize=(10, 6),
    subplot_kw={"projection": "polar"},
)
for pl_i, sc_i in enumerate(list(smoothcurves)[:6]):
    axd[f"phase_{pl_i}"].plot(
        list(smoothcurves[sc_i].index) + list([smoothcurves[sc_i].index[0]]),
        list(smoothcurves[sc_i].values) + list([smoothcurves[sc_i].values[0]]),
    )
    axd[f"phase_{pl_i}"].set_xlabel("Phase (rad)")  # Angle in radian, on the X-axis
    axd[f"phase_{pl_i}"].set_ylabel(
        "Firing Rate (Hz)"
    )  # Firing rate in Hz, on the Y-axis
    axd[f"phase_{pl_i}"].set_xticks([])
    axd[f"phase_{pl_i}"].set_title(f"Unit {sc_i}")
fig.suptitle("Phase Preference Histograms of First 6 Units")


# %%
# ***
# Isolating Strong Phase Preferences
# -----------------------------------
# It looks like there could be some phase preferences happening here, but there's a lot of cells to go through.
# Now that we have our phases of firing for each unit, we can sort the units by the circular variance of the phase
# of their spikes, to isolate the cells with the strongest phase preferences without manual inspection.

# Get phase of each spike
phase = {}
for i in spikes:
    phase_i = [
        theta_band_phase[np.argmin(np.abs(REM_Tsd.index.values - s.index))]
        for s in spikes[i]
    ]
    phase[i] = np.array(phase_i)
phase_var = {
    key: scipy.stats.circvar(value, low=-np.pi, high=np.pi)
    for key, value in phase.items()
}
phase_var = dict(sorted(phase_var.items(), key=lambda item: item[1]))

# %%
# ***
# And now we plot the phase preference histograms of the 6 units with the least variance in the phase of their
# spiking behaviour.

fig, axd = plt.subplot_mosaic(
    [["phase_0", "phase_1", "phase_2"], ["phase_3", "phase_4", "phase_5"]],
    constrained_layout=True,
    figsize=(10, 6),
    subplot_kw={"projection": "polar"},
)
for pl_i, sc_i in enumerate(list(phase_var.keys())[:6]):
    axd[f"phase_{pl_i}"].plot(
        list(smoothcurves[sc_i].index) + list([smoothcurves[sc_i].index[0]]),
        list(smoothcurves[sc_i].values) + list([smoothcurves[sc_i].values[0]]),
    )
    axd[f"phase_{pl_i}"].set_xlabel("Phase (rad)")  # Angle in radian, on the X-axis
    axd[f"phase_{pl_i}"].set_ylabel(
        "Firing Rate (Hz)"
    )  # Firing rate in Hz, on the Y-axis
    axd[f"phase_{pl_i}"].set_xticks([])
    axd[f"phase_{pl_i}"].set_title(f"Unit {sc_i}")
fig.suptitle("Phase Preference Histograms of 6 Units with Highest Phase Preference ")

# %%
# ***
# Visualizing Phase Preferences
# -----------------------------------
# There is definitely some strong phase preferences happening here. Let's visualize the firing preferences
# of the 6 cells we've isolated to get an impression of just how striking these preferences are.

fig = plt.figure(constrained_layout=True, figsize=(10, 8))
axd = fig.subplot_mosaic(
    [
        ["lfp_run"],
        ["phase_0"],
        ["phase_1"],
        ["phase_2"],
    ],
    height_ratios=[0.4, 0.2, 0.2, 0.2],
)
[axd[k].margins(0) for k in ["lfp_run"] + [f"phase_{i}" for i in range(3)]]
axd["lfp_run"].plot(
    REM_Tsd.index.values, REM_Tsd[:, 0], alpha=0.5, label="LFP Data - REM"
)
axd["lfp_run"].plot(
    REM_Tsd.index.values,
    theta_band_reconstruction,
    label=f"{np.round(freqs[theta_freq_index],2)}Hz oscillations",
)
axd["lfp_run"].set_ylabel("LFP (v)")
axd["lfp_run"].set_xlabel("Time (s)")
axd["lfp_run"].set_title(f"{np.round(freqs[theta_freq_index],2)}Hz oscillation power.")
axd["lfp_run"].legend()
for i in range(3):
    axd[f"phase_{i}"].plot(REM_Tsd.index.values, theta_band_phase, alpha=0.2)
    axd[f"phase_{i}"].scatter(
        spikes[list(phase_var.keys())[i]].index, phase[list(phase_var.keys())[i]]
    )
    axd[f"phase_{i}"].set_ylabel("Phase")
    axd[f"phase_{i}"].set_title(f"Unit {list(phase_var.keys())[i]}")
fig.suptitle("Phase Preference Visualizations")
