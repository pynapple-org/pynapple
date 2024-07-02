# -*- coding: utf-8 -*-
"""
Signal Processing Local Field Potentials
============

Working with Local Field Potential data.

See the [documentation](https://pynapple-org.github.io/pynapple/) of Pynapple for instructions on installing the package.

This tutorial was made by Kipp Freud.

"""
# %%
# !!! warning
#     This tutorial uses matplotlib for displaying the figure
#
#     You can install all with `pip install matplotlib requests`
#
# mkdocs_gallery_thumbnail_number = 1
#
# Now, import the necessary libraries:

import os
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import requests

import pynapple as nap

# %%
# ***
# Downloading the data
# ------------------
# First things first: Let's download and extract the data
path = "data/A2929-200711"
extract_to = "data"
if extract_to not in os.listdir("."):
    os.mkdir(extract_to)
if path not in os.listdir("."):
    # Download the file
    response = requests.get(
        "https://www.dropbox.com/s/su4oaje57g3kit9/A2929-200711.zip?dl=1"
    )
    zip_path = os.path.join(extract_to, "/downloaded_file.zip")
    # Write the zip file to disk
    with open(zip_path, "wb") as f:
        f.write(response.content)
    # Unzip the file
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


# %%
# ***
# Parsing the data
# ------------------
# Now that we have the data, we must append the 2kHz LFP recording to the .nwb file
# eeg_path = "data/A2929-200711/A2929-200711.dat"
# frequency = 20000  # Hz
# n_channels = 16
# f = open(eeg_path, "rb")
# startoffile = f.seek(0, 0)
# endoffile = f.seek(0, 2)
# f.close()
# bytes_size = 2
# n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
# duration = n_samples / frequency
# interval = 1 / frequency
# fp = np.memmap(eeg_path, np.int16, "r", shape=(n_samples, n_channels))
# timestep = np.arange(0, n_samples) / frequency
# eeg = nap.TsdFrame(t=timestep, d=fp)
# nap.append_NWB_LFP("data/A2929-200711/", eeg)


# %%
# Let's save the RoiResponseSeries as a variable called 'transients' and print it
FS = 1250
# data = nap.load_file("data/A2929-200711/pynapplenwb/A2929-200711.nwb")
data = nap.load_file("data/stable.nwb")
print(data["ElectricalSeries"])

# %%
# ***
# Selecting slices
# -----------------------------------
# Let's consider a two 1-second slices of data, one from the sleep epoch and one from wake
NES = nap.TsdFrame(
    t=data["ElectricalSeries"].index.values,
    d=data["ElectricalSeries"].values,
    time_support=data["ElectricalSeries"].time_support,
)
wake_minute = NES.restrict(nap.IntervalSet(900, 960))
sleep_minute = NES.restrict(nap.IntervalSet(0, 60))
channel = 1

# %%
# ***
# Plotting the LFP activity of one slices
# -----------------------------------
# Let's plot
fig, ax = plt.subplots(2)
for channel in range(sleep_minute.shape[1]):
    ax[0].plot(
        sleep_minute[:, channel],
        alpha=0.5,
        label="Sleep Data",
    )
ax[0].set_title("Sleep ephys")
for channel in range(wake_minute.shape[1]):
    ax[1].plot(
        wake_minute[:, channel],
        alpha=0.5,
        label="Wake Data"
    )
ax[1].set_title("Wake ephys")
plt.show()


# %%
# There is much shared information between channels, and wake and sleep don't seem visibly different.
# Let's take the Fourier transforms of one channel for both and see if differences are present
fig, ax = plt.subplots(1)
fft_sig, fft_freqs = nap.compute_spectrum(sleep_minute[:, channel], fs=int(FS))
ax.plot(fft_freqs, np.abs(fft_sig), alpha=0.5, label="Sleep Data")
ax.set_xlim((0, FS / 2))
fft_sig, fft_freqs = nap.compute_spectrum(wake_minute[:, channel], fs=int(FS))
ax.plot(fft_freqs, np.abs(fft_sig), alpha=0.5, label="Wake Data")
ax.set_title(f"Fourier Decomposition for channel {channel}")
ax.legend()
plt.show()


# %%
# There seems to be some differences presenting themselves - a bump in higher frequencies for waking data?
# Let's explore further with a wavelet decomposition


def plot_timefrequency(times, freqs, powers, x_ticks=5, y_ticks=5, ax=None, **kwargs):
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
    if isinstance(y_ticks, int):
        y_ticks_pos = np.linspace(0, freqs.size, y_ticks)
        y_ticks = np.round(np.linspace(freqs[0], freqs[-1], y_ticks), 2)
    else:
        y_ticks_pos = [np.argmin(np.abs(freqs - val)) for val in y_ticks]
    ax.set(yticks=y_ticks_pos, yticklabels=y_ticks)


fig, ax = plt.subplots(2)
freqs = np.array(
    [
        2.59,
        3.66,
        5.18,
        8.0,
        10.36,
        14.65,
        20.72,
        29.3,
        41.44,
        58.59,
        82.88,
        117.19,
        165.75,
        234.38,
        331.5,
        468.75,
        624.0,
    ]
)
mwt_sleep = nap.compute_wavelet_transform(
    sleep_minute[:, channel], fs=None, freqs=freqs
)
plot_timefrequency(
    sleep_minute.index.values[:],
    freqs[:],
    np.transpose(mwt_sleep[:, :].values),
    ax=ax[0],
)
ax[0].set_title(f"Sleep Data Wavelet Decomposition: Channel {channel}")
mwt_wake = nap.compute_wavelet_transform(wake_minute[:, channel], fs=None, freqs=freqs)
plot_timefrequency(
    wake_minute.index.values[:], freqs[:], np.transpose(mwt_wake[:, :].values), ax=ax[1]
)
ax[1].set_title(f"Wake Data Wavelet Decomposition: Channel {channel}")
plt.margins(0)
plt.show()

# %%
# Let's focus on the waking data. Let's see if we can isolate the theta oscillations from the data
freq = 3
interval = (937, 939)
wake_second = wake_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
mwt_wake_second = mwt_wake.restrict(nap.IntervalSet(interval[0], interval[1]))
fig, ax = plt.subplots(1)
ax.plot(wake_second.index.values, wake_second[:, channel], alpha=0.5, label="Wake Data")
ax.plot(
    wake_second.index.values,
    mwt_wake_second[
        :, freq
    ].values.real,
    label="Theta oscillations",
)
ax.set_title(f"{freqs[freq]}Hz oscillation power.")
plt.show()


# %%
# Let's focus on the sleeping data. Let's see if we can isolate the slow wave oscillations from the data
freq = 0
# interval = (10, 15)
interval = (20, 25)
sleep_second = sleep_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
mwt_sleep_second = mwt_sleep.restrict(nap.IntervalSet(interval[0], interval[1]))
_, ax = plt.subplots(1)
ax.plot(
    sleep_second[:, channel], alpha=0.5, label="Wake Data"
)
ax.plot(
    sleep_second.index.values,
    mwt_sleep_second[
        :, freq
    ].values.real,
    label="Slow Wave Oscillations",
)
ax.set_title(f"{freqs[freq]}Hz oscillation power")
plt.show()

# %%
# Let's plot spike phase, time scatter plots to see if spikes display phase characteristics during slow wave sleep

_, ax = plt.subplots(20, figsize=(10, 50))
mwt_sleep = np.transpose(
    mwt_sleep_second
)
ax[0].plot(sleep_second.index, sleep_second.values[:, 0])
plot_timefrequency(sleep_second.index, freqs, np.abs(mwt_sleep[:, :]), ax=ax[1])

ax[2].plot(sleep_second.index, sleep_second.values[:, 0])
ax[2].plot(sleep_second.index, mwt_sleep[freq, :].real)
ax[2].set_title(f"{freqs[freq]}Hz")

ax[3].plot(sleep_second.index, np.abs(mwt_sleep[freq, :]))
# ax[3].plot(lfp.index, lfp.values[:,0])
ax[4].plot(sleep_second.index, np.angle(mwt_sleep[freq, :]))

spikes = {}
for i in data["units"].index:
    spikes[i] = data["units"][i].times()[
        (data["units"][i].times() > interval[0])
        & (data["units"][i].times() < interval[1])
    ]

phase = {}
for i in spikes.keys():
    phase_i = []
    for spike in spikes[i]:
        phase_i.append(
            np.angle(
                mwt_sleep[freq, np.argmin(np.abs(sleep_second.index.values - spike))]
            )
        )
    phase[i] = np.array(phase_i)

for i in range(15):
    ax[5 + i].scatter(spikes[i], phase[i])
    ax[5 + i].set_xlim(interval[0], interval[1])
    ax[5 + i].set_ylim(-np.pi, np.pi)
    ax[5 + i].set_xlabel("time (s)")
    ax[5 + i].set_ylabel("phase")

plt.tight_layout()
plt.show()
