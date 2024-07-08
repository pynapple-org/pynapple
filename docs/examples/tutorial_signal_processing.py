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
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

import pynapple as nap
from examples_utils import data, plotting

# %%
# ***
# Downloading the data
# ------------------
# First things first: Let's download and extract the data - download section currently commented as correct NWB
# is not online

# path = data.download_data(
#     "Achilles_10252013.nwb", "https://osf.io/hu5ma/download", "../data"
# )
# data = nap.load_file(path)

data = nap.load_file("../data/Achillies_ephys.nwb")
FS = len(data["LFP"].index[:]) / (data["LFP"].index[-1] - data["LFP"].index[0])
print(data)

# %%
# ***
# Selecting slices
# -----------------------------------
# Let's consider two 60-second slices of data, one from the sleep epoch and one from wake

wake_minute_interval = nap.IntervalSet(
    data["epochs"]["MazeEpoch"]["start"] + 60.0,
    data["epochs"]["MazeEpoch"]["start"] + 120.0,
)
sleep_minute_interval = nap.IntervalSet(
    data["epochs"]["POSTEpoch"]["start"] + 60.0,
    data["epochs"]["POSTEpoch"]["start"] + 120.0,
)
wake_minute = nap.TsdFrame(
    t=data["LFP"].restrict(wake_minute_interval).index.values,
    d=data["LFP"].restrict(wake_minute_interval).values,
    time_support=data["LFP"].restrict(wake_minute_interval).time_support,
)
sleep_minute = nap.TsdFrame(
    t=data["LFP"].restrict(sleep_minute_interval).index.values,
    d=data["LFP"].restrict(sleep_minute_interval).values,
    time_support=data["LFP"].restrict(sleep_minute_interval).time_support,
)
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
    ax[1].plot(wake_minute[:, channel], alpha=0.5, label="Wake Data")
ax[1].set_title("Wake ephys")
plt.show()


# %%
# Let's take the Fourier transforms of one channel for both waking and sleeping and see if differences are present
channel = 1
fig, ax = plt.subplots(2)
fft = nap.compute_spectogram(sleep_minute, fs=int(FS))
ax[0].plot(
    fft.index, np.abs(fft.iloc[:, channel]), alpha=0.5, label="Sleep Data", c="blue"
)
ax[0].set_xlim((0, FS / 2))
ax[0].set_xlabel("Freq (Hz)")
ax[0].set_ylabel("Frequency Power")

ax[0].set_title("Sleep LFP Decomposition")
fft = nap.compute_spectogram(wake_minute, fs=int(FS))
ax[1].plot(
    fft.index, np.abs(fft.iloc[:, channel]), alpha=0.5, label="Wake Data", c="orange"
)
ax[1].set_xlim((0, FS / 2))
fig.suptitle(f"Fourier Decomposition for channel {channel}")
ax[1].set_title("Sleep LFP Decomposition")
ax[1].set_xlabel("Freq (Hz)")
ax[1].set_ylabel("Frequency Power")

# ax.legend()
plt.show()


# %%
# Let's now consider the Welch spectograms of waking and sleeping data...

fig, ax = plt.subplots(2)
fft = nap.compute_welch_spectogram(sleep_minute, fs=int(FS))
ax[0].plot(
    fft.index, np.abs(fft.iloc[:, channel]), alpha=0.5, label="Sleep Data", color="blue"
)
ax[0].set_xlim((0, FS / 2))
ax[0].set_title("Sleep LFP Decomposition")
ax[0].set_xlabel("Freq (Hz)")
ax[0].set_ylabel("Frequency Power")
welch = nap.compute_welch_spectogram(wake_minute, fs=int(FS))
ax[1].plot(
    welch.index,
    np.abs(welch.iloc[:, channel]),
    alpha=0.5,
    label="Wake Data",
    color="orange",
)
ax[1].set_xlim((0, FS / 2))
fig.suptitle(f"Welch Decomposition for channel {channel}")
ax[1].set_title("Sleep LFP Decomposition")
ax[1].set_xlabel("Freq (Hz)")
ax[1].set_ylabel("Frequency Power")
# ax.legend()
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
interval = (wake_minute_interval["start"], wake_minute_interval["start"] + 2)
wake_second = wake_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
mwt_wake_second = mwt_wake.restrict(nap.IntervalSet(interval[0], interval[1]))
fig, ax = plt.subplots(1)
ax.plot(wake_second.index.values, wake_second[:, channel], alpha=0.5, label="Wake Data")
ax.plot(
    wake_second.index.values,
    mwt_wake_second[:, freq].values.real,
    label="Theta oscillations",
)
ax.set_title(f"{freqs[freq]}Hz oscillation power.")
plt.show()


# %%
# Let's focus on the sleeping data. Let's see if we can isolate the slow wave oscillations from the data
freq = 0
# interval = (10, 15)
interval = (sleep_minute_interval["start"] + 30, sleep_minute_interval["start"] + 35)
sleep_second = sleep_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
mwt_sleep_second = mwt_sleep.restrict(nap.IntervalSet(interval[0], interval[1]))
_, ax = plt.subplots(1)
ax.plot(sleep_second[:, channel], alpha=0.5, label="Wake Data")
ax.plot(
    sleep_second.index.values,
    mwt_sleep_second[:, freq].values.real,
    label="Slow Wave Oscillations",
)
ax.set_title(f"{freqs[freq]}Hz oscillation power")
plt.show()

# %%
# Let's plot spike phase, time scatter plots to see if spikes display phase characteristics during slow wave sleep

_, ax = plt.subplots(20, figsize=(10, 50))
mwt_sleep = np.transpose(mwt_sleep_second)
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

spikes = {k: v for k, v in spikes.items() if len(v) > 0}
phase = {k: v for k, v in phase.items() if len(v) > 0}

for i in range(15):
    ax[5 + i].scatter(spikes[list(spikes.keys())[i]], phase[list(phase.keys())[i]])
    ax[5 + i].set_xlim(interval[0], interval[1])
    ax[5 + i].set_ylim(-np.pi, np.pi)
    ax[5 + i].set_xlabel("time (s)")
    ax[5 + i].set_ylabel("phase")

plt.tight_layout()
plt.show()
