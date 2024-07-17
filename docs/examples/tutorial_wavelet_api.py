# -*- coding: utf-8 -*-
"""
Wavelet API tutorial
============

Working with Wavelets.

See the [documentation](https://pynapple-org.github.io/pynapple/) of Pynapple for instructions on installing the package.

This tutorial was made by Kipp Freud.

"""

# %%
# !!! warning
#     This tutorial uses matplotlib for displaying the figure
#
#     You can install all with `pip install matplotlib requests tqdm`
#
# Now, import the necessary libraries:

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

import pynapple as nap

# %%
# ***
# Generating a dummy signal
# ------------------
# Let's generate a dummy signal to analyse with wavelets!

# Our dummy dataset will contain two components, a low frequency 3Hz sinusoid combined
# with a weaker 25Hz sinusoid.
t = np.linspace(0, 10, 10000)
sig = nap.Tsd(d=np.sin(t * (5 + t) * np.pi * 2) + np.sin(t * 3 * np.pi * 2), t=t)
# Plot it
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 3))
ax.plot(sig)
ax.margins(0)
plt.show()

# %%
# ***
# Getting our Morlet wavelet filter bank
# ------------------

freqs = np.linspace(1, 25, num=25)
filter_bank = nap.generate_morlet_filterbank(
    freqs, 1000, n_cycles=1.5, scaling=1.0, precision=10
)
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
offset = 0.2
for f_i in range(filter_bank.shape[0]):
    ax.plot(
        np.linspace(-8, 8, filter_bank.shape[1]), filter_bank[f_i, :] + offset * f_i
    )
    ax.text(-2.2, offset * f_i, f"{np.round(freqs[f_i], 2)}Hz", va="center", ha="left")
ax.margins(0)
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(-2, 2)
ax.set_xlabel("Time (s)")
ax.set_title("Morlet Wavelet Filter Bank")
plt.show()

# %%
# ***
# Effect of n_cycles
# ------------------

freqs = np.linspace(1, 25, num=25)
filter_bank = nap.generate_morlet_filterbank(
    freqs, 1000, n_cycles=7.5, scaling=1.0, precision=10
)
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
offset = 0.2
for f_i in range(filter_bank.shape[0]):
    ax.plot(
        np.linspace(-8, 8, filter_bank.shape[1]), filter_bank[f_i, :] + offset * f_i
    )
    ax.text(-2.2, offset * f_i, f"{np.round(freqs[f_i], 2)}Hz", va="center", ha="left")
ax.margins(0)
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(-2, 2)
ax.set_xlabel("Time (s)")
ax.set_title("Morlet Wavelet Filter Bank")
plt.show()

# %%
# ***
# Effect of scaling
# ------------------

freqs = np.linspace(1, 25, num=25)
filter_bank = nap.generate_morlet_filterbank(
    freqs, 1000, n_cycles=7.5, scaling=2.0, precision=10
)
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
offset = 0.2
for f_i in range(filter_bank.shape[0]):
    ax.plot(
        np.linspace(-8, 8, filter_bank.shape[1]), filter_bank[f_i, :] + offset * f_i
    )
    ax.text(-2.2, offset * f_i, f"{np.round(freqs[f_i], 2)}Hz", va="center", ha="left")
ax.margins(0)
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlim(-2, 2)
ax.set_xlabel("Time (s)")
ax.set_title("Morlet Wavelet Filter Bank")
plt.show()


# %%
# ***
# Decomposing the dummy signal
# ------------------

mwt = nap.compute_wavelet_transform(
    sig, fs=None, freqs=freqs, n_cycles=1.5, scaling=1.0, precision=15
)


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


fig, ax = plt.subplots(1)
plot_timefrequency(
    mwt.index.values[:],
    freqs[:],
    np.transpose(mwt[:, :].values),
    ax=ax,
)
plt.show()


# %%
# ***
# Increasing n_cycles increases resolution of decomposition
# ------------------

mwt = nap.compute_wavelet_transform(
    sig, fs=None, freqs=freqs, n_cycles=7.5, scaling=1.0, precision=10
)
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
plot_timefrequency(
    mwt.index.values[:],
    freqs[:],
    np.transpose(mwt[:, :].values),
    ax=ax,
)
plt.show()

# %%
# ***
# Increasing n_cycles increases resolution of decomposition
# ------------------

mwt = nap.compute_wavelet_transform(
    sig, fs=None, freqs=freqs, n_cycles=7.5, scaling=2.0, precision=10
)
fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 6))
plot_timefrequency(
    mwt.index.values[:],
    freqs[:],
    np.transpose(mwt[:, :].values),
    ax=ax,
)
plt.show()
