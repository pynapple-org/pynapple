# -*- coding: utf-8 -*-
"""
Wavelet API tutorial
============

Working with Wavelets!

See the [documentation](https://pynapple-org.github.io/pynapple/) of Pynapple for instructions on installing the package.

This tutorial was made by Kipp Freud.

"""

# %%
# !!! warning
#     This tutorial uses matplotlib for displaying the figure
#
#     You can install all with `pip install matplotlib requests tqdm seaborn`
#
# Now, import the necessary libraries:

import matplotlib.pyplot as plt
import numpy as np
import seaborn

seaborn.set_theme()

import pynapple as nap

# %%
# ***
# Generating a Dummy Signal
# ------------------
# Let's generate a dummy signal to analyse with wavelets!
#
# Our dummy dataset will contain two components, a low frequency 2Hz sinusoid combined
# with a sinusoid which increases frequency from 5 to 15 Hz throughout the signal.

Fs = 2000
t = np.linspace(0, 5, Fs * 5)
two_hz_phase = t * 2 * np.pi * 2
two_hz_component = np.sin(two_hz_phase)
increasing_freq_component = np.sin(t * (5 + t) * np.pi * 2)
sig = nap.Tsd(
    d=two_hz_component + increasing_freq_component + np.random.normal(0, 0.1, 10000),
    t=t,
)

# %%
# Lets plot it.
fig, ax = plt.subplots(3, constrained_layout=True, figsize=(10, 5))
ax[0].plot(t, two_hz_component)
ax[1].plot(t, increasing_freq_component)
ax[2].plot(sig)
ax[0].set_title("2Hz Component")
ax[1].set_title("Increasing Frequency Component")
ax[2].set_title("Dummy Signal")
[ax[i].margins(0) for i in range(3)]
[ax[i].set_ylim(-2.5, 2.5) for i in range(3)]
[ax[i].spines[sp].set_visible(False) for sp in ["right", "top"] for i in range(3)]
[ax[i].set_xlabel("Time (s)") for i in range(3)]
[ax[i].set_ylabel("Signal") for i in range(3)]
[ax[i].set_ylim(-2.5, 2.5) for i in range(3)]


# %%
# ***
# Getting our Morlet Wavelet Filter Bank
# ------------------
# We will be decomposing our dummy signal using wavelets of different frequencies. These wavelets
# can be examined using the `generate_morlet_filterbank` function. Here we will use the default parameters
# to define a Morlet filter bank with which we will later use to deconstruct the signal.

# Define the frequency of the wavelets in our filter bank
freqs = np.linspace(1, 25, num=25)
# Get the filter bank
filter_bank = nap.generate_morlet_filterbank(
    freqs, Fs, gaussian_width=1.5, window_length=1.0
)


# %%
# Lets plot it.
def plot_filterbank(filter_bank, freqs, title):
    fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 7))
    offset = 1.0
    for f_i in range(filter_bank.shape[1]):
        ax.plot(filter_bank[:, f_i].real() + offset * f_i)
        ax.text(
            -2.3, offset * f_i, f"{np.round(freqs[f_i], 2)}Hz", va="center", ha="left"
        )
    ax.margins(0)
    ax.yaxis.set_visible(False)
    [ax.spines[sp].set_visible(False) for sp in ["left", "right", "top"]]
    ax.set_xlim(-2, 2)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)


title = "Morlet Wavelet Filter Bank (Real Components): gaussian_width=1.5, window_length=1.0"
plot_filterbank(filter_bank, freqs, title)

# %%
# ***
# Decomposing the Dummy Signal
# ------------------
# Here we will use the `compute_wavelet_transform` function to decompose our signal using the filter bank shown
# above. Wavelet decomposition breaks down a signal into its constituent wavelets, capturing both time and
# frequency information for analysis. We will calculate this decomposition and plot it's corresponding
# scalogram.

# Compute the wavelet transform using the parameters above
mwt = nap.compute_wavelet_transform(
    sig, fs=Fs, freqs=freqs, gaussian_width=1.5, window_length=1.0
)


# %%
# Lets plot it.
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
    ax.grid(False)


fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
plot_timefrequency(
    mwt.index.values[:],
    freqs[:],
    np.transpose(mwt[:, :].values),
    ax=ax,
)
ax.set_title("Wavelet Decomposition Scalogram")

# %%
# ***
# Reconstructing the Slow Oscillation and Phase
# ------------------
# We can see that the decomposition has picked up on the 2Hz component of the signal, as well as the component with
# increasing frequency. In this section, we will extract just the 2Hz component from the wavelet decomposition,
# and see how it compares to the original section.

# Get the index of the 2Hz frequency
two_hz_freq_idx = np.where(freqs == 2.0)[0]
# The 2Hz component is the real component of the wavelet decomposition at this index
slow_oscillation = mwt[:, two_hz_freq_idx].values.real
# The 2Hz wavelet phase is the angle of the wavelet decomposition at this index
slow_oscillation_phase = np.angle(mwt[:, two_hz_freq_idx].values)

# %%
# Lets plot it.
fig = plt.figure(constrained_layout=True, figsize=(10, 6))
axd = fig.subplot_mosaic(
    [["signal"], ["phase"]],
    height_ratios=[1, 0.4],
)
axd["signal"].plot(sig, label="Raw Signal", alpha=0.5)
axd["signal"].plot(t, slow_oscillation, label="2Hz Reconstruction")
axd["signal"].legend()
axd["phase"].plot(t, slow_oscillation_phase, alpha=0.5)
axd["phase"].set_ylabel("Phase (rad)")
axd["signal"].set_ylabel("Signal")
axd["phase"].set_xlabel("Time (s)")
[
    axd[f].spines[sp].set_visible(False)
    for sp in ["right", "top"]
    for f in ["phase", "signal"]
]
axd["signal"].get_xaxis().set_visible(False)
axd["signal"].spines["bottom"].set_visible(False)
[axd[k].margins(0) for k in ["signal", "phase"]]
axd["signal"].set_ylim(-2.5, 2.5)
axd["phase"].set_ylim(-np.pi, np.pi)

# %%
# ***
# Adding in the 15Hz Oscillation
# ------------------
# Let's see what happens if we also add the 15 Hz component of the wavelet decomposition to the reconstruction. We
# will extract the 15 Hz components, and also the 15Hz wavelet power over time. The wavelet power tells us to what
# extent the 15 Hz frequency is present in our signal at different times.
#
# Finally, we will add this 15 Hz reconstruction to the one shown above, to see if it improves out reconstructed
# signal.

# Get the index of the 15 Hz frequency
fifteen_hz_freq_idx = np.where(freqs == 15.0)[0]
# The 15 Hz component is the real component of the wavelet decomposition at this index
fifteenHz_oscillation = mwt[:, fifteen_hz_freq_idx].values.real
# The 15 Hz poser is the absolute value of the wavelet decomposition at this index
fifteenHz_oscillation_power = np.abs(mwt[:, fifteen_hz_freq_idx].values)

# %%
# Lets plot it.
fig, ax = plt.subplots(2, constrained_layout=True, figsize=(10, 6))
ax[0].plot(t, fifteenHz_oscillation, label="15Hz Reconstruction")
ax[0].plot(t, fifteenHz_oscillation_power, label="15Hz Power")
ax[1].plot(sig, label="Raw Signal", alpha=0.5)
ax[1].plot(
    t, slow_oscillation + fifteenHz_oscillation, label="2Hz + 15Hz Reconstruction"
)
[ax[i].set_ylim(-2.5, 2.5) for i in range(2)]
[ax[i].margins(0) for i in range(2)]
[ax[i].legend() for i in range(2)]
[ax[i].spines[sp].set_visible(False) for sp in ["right", "top"] for i in range(2)]
ax[0].get_xaxis().set_visible(False)
ax[0].spines["bottom"].set_visible(False)
ax[1].set_xlabel("Time (s)")
[ax[i].set_ylabel("Signal") for i in range(2)]


# %%
# ***
# Adding ALL the Oscillations!
# ------------------
# Let's now add together the real components of all frequency bands to recreate a version of the original signal.

combined_oscillations = mwt.sum(axis=1).real()

# %%
# Lets plot it.
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 4))
ax.plot(sig, alpha=0.5, label="Signal")
ax.plot(t, combined_oscillations, label="Wavelet Reconstruction", alpha=0.5)
[ax.spines[sp].set_visible(False) for sp in ["right", "top"]]
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal")
ax.set_title("Wavelet Reconstruction of Signal")
ax.set_ylim(-6, 6)
ax.margins(0)
ax.legend()


# %%
# ***
# Parametrization
# ------------------
# Our reconstruction seems to get the amplitude modulations of our signal correct, but the amplitude is overestimated,
# in particular towards the end of the time period. Often, this is due to a suboptimal choice of parameters, which
# can lead to a low spatial or temporal resolution. Let's explore what changing our parameters does to the
# underlying wavelets.

freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
window_lengths = [1.0, 2.0, 3.0]
gaussian_width = [1.0, 2.0, 3.0]

fig, ax = plt.subplots(
    len(window_lengths), len(gaussian_width), constrained_layout=True, figsize=(10, 8)
)
for row_i, wl in enumerate(window_lengths):
    for col_i, gw in enumerate(gaussian_width):
        filter_bank = nap.generate_morlet_filterbank(
            freqs, 1000, gaussian_width=gw, window_length=wl, precision=12
        )
        ax[row_i, col_i].plot(filter_bank[:, 0].real())
        ax[row_i, col_i].set_xlabel("Time (s)")
        ax[row_i, col_i].set_yticks([])
        [
            ax[row_i, col_i].spines[sp].set_visible(False)
            for sp in ["top", "right", "left"]
        ]
        if col_i != 0:
            ax[row_i, col_i].get_yaxis().set_visible(False)
for col_i, gw in enumerate(gaussian_width):
    ax[0, col_i].set_title(f"gaussian_width={gw}", fontsize=10)
for row_i, wl in enumerate(window_lengths):
    ax[row_i, 0].set_ylabel(f"window_length={wl}", fontsize=10)
fig.suptitle("Parametrization Visualization")


# %%
# Increasing time_decay increases the number of wavelet cycles present in the oscillations (cycles) within the
# Gaussian window of the Morlet wavelet. It essentially controls the trade-off between time resolution
# and frequency resolution.
#
# The scale parameter determines the dilation or compression of the wavelet. It controls the size of the wavelet in
# time, affecting the overall shape of the wavelet.

# %%
# ***
# Effect of gaussian_width
# ------------------
# Let's increase time_decay to 7.5 and see the effect on the resultant filter bank.

freqs = np.linspace(1, 25, num=25)
filter_bank = nap.generate_morlet_filterbank(
    freqs, 1000, gaussian_width=7.5, window_length=1.0
)

plot_filterbank(
    filter_bank,
    freqs,
    "Morlet Wavelet Filter Bank (Real Components): gaussian_width=7.5, center_frequency=1.0",
)

# %%
# ***
# Let's see what effect this has on the Wavelet Scalogram which is generated...
mwt = nap.compute_wavelet_transform(
    sig, fs=Fs, freqs=freqs, gaussian_width=7.5, window_length=1.0
)

fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
plot_timefrequency(
    mwt.index.values[:],
    freqs[:],
    np.transpose(mwt[:, :].values),
    ax=ax,
)
ax.set_title("Wavelet Decomposition Scalogram")

# %%
# ***
# And let's see if that has an effect on the reconstructed version of the signal

combined_oscillations = mwt.sum(axis=1).real()

# %%
# Lets plot it.
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 4))
ax.plot(sig, alpha=0.5, label="Signal")
ax.plot(t, combined_oscillations, label="Wavelet Reconstruction", alpha=0.5)
[ax.spines[sp].set_visible(False) for sp in ["right", "top"]]
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal")
ax.set_title("Wavelet Reconstruction of Signal")
ax.set_ylim(-6, 6)
ax.margins(0)
ax.legend()

# %%
# There's a small improvement, but perhaps we can do better.


# %%
# ***
# Effect of window_length
# ------------------
# Let's increase window_length to 2.0 and see the effect on the resultant filter bank.

freqs = np.linspace(1, 25, num=25)
filter_bank = nap.generate_morlet_filterbank(
    freqs, 1000, gaussian_width=7.5, window_length=2.0
)

plot_filterbank(
    filter_bank,
    freqs,
    "Morlet Wavelet Filter Bank (Real Components): gaussian_width=7.5, center_frequency=2.0",
)

# %%
# ***
# Let's see what effect this has on the Wavelet Scalogram which is generated...
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
mwt = nap.compute_wavelet_transform(
    sig, fs=Fs, freqs=freqs, gaussian_width=7.5, window_length=2.0
)
plot_timefrequency(
    mwt.index.values[:],
    freqs[:],
    np.transpose(mwt[:, :].values),
    ax=ax,
)
ax.set_title("Wavelet Decomposition Scalogram")

# %%
# ***
# And let's see if that has an effect on the reconstructed version of the signal

combined_oscillations = mwt.sum(axis=1).real()

# %%
# Lets plot it.
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 4))
ax.plot(sig, alpha=0.5, label="Signal")
ax.plot(t, combined_oscillations, label="Wavelet Reconstruction", alpha=0.5)
[ax.spines[sp].set_visible(False) for sp in ["right", "top"]]
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal")
ax.set_title("Wavelet Reconstruction of Signal")
ax.margins(0)
ax.set_ylim(-6, 6)
ax.legend()
