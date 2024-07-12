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
#     You can install all with `pip install matplotlib requests tqdm`
#
# mkdocs_gallery_thumbnail_number = 1
#
# Now, import the necessary libraries:
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import tqdm
import math
import pynapple as nap
import scipy

# %%
# ***
# Downloading the data
# ------------------
# First things first: Let's download the data and save it locally

path = "Achilles_10252013_EEG.nwb"
if path not in os.listdir("."):
  r = requests.get(f"https://osf.io/2dfvp/download", stream=True)
  block_size = 1024*1024
  with open(path, 'wb') as f:
    for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
      total=math.ceil(int(r.headers.get('content-length', 0))//block_size)):
      f.write(data)


# %%
# ***
# Loading the data
# ------------------
# Loading the data, calculating the sampling frequency

data = nap.load_file(path)
FS = len(data["eeg"].index[:]) / (data["eeg"].index[-1] - data["eeg"].index[0])
print(data)


# %%
# ***
# Selecting slices
# -----------------------------------
# Let's consider two 60-second slices of data, one from the sleep epoch and one from wake

REM_minute_interval = nap.IntervalSet(
    data["rem"]["start"][0] + 60.0,
    data["rem"]["start"][0] + 120.0,
)

SWS_minute_interval = nap.IntervalSet(
    data["nrem"]["start"][0] + 10.0,
    data["nrem"]["start"][0] + 70.0,
)

RUN_minute_interval = nap.IntervalSet(
    data["forward_ep"]["start"][-18] + 0.,
    data["forward_ep"]["start"][-18] + 60.,
)

REM_minute = nap.TsdFrame(
    t=data["eeg"].restrict(REM_minute_interval).index.values,
    d=data["eeg"].restrict(REM_minute_interval).values,
    time_support=data["eeg"].restrict(REM_minute_interval).time_support,
)

SWS_minute = nap.TsdFrame(
    t=data["eeg"].restrict(SWS_minute_interval).index.values,
    d=data["eeg"].restrict(SWS_minute_interval).values,
    time_support=data["eeg"].restrict(SWS_minute_interval).time_support,
)

RUN_minute = nap.TsdFrame(
    t=data["eeg"].restrict(RUN_minute_interval).index.values,
    d=data["eeg"].restrict(RUN_minute_interval).values,
    time_support=data["eeg"].restrict(RUN_minute_interval).time_support,
)
# RUN_position = nap.TsdFrame(
#     t=data["position"].restrict(RUN_minute_interval).index.values[1:],
#     d=np.diff(data['position'].restrict(RUN_minute_interval)),
#     time_support=data["position"].restrict(RUN_minute_interval).time_support,
# )
RUN_position = nap.TsdFrame(
    t=data["position"].restrict(RUN_minute_interval).index.values[:],
    d=data['position'].restrict(RUN_minute_interval),
    time_support=data["position"].restrict(RUN_minute_interval).time_support,
)

channel = 0

# %%
# ***
# Plotting the LFP activity of one slices
# -----------------------------------
# Let's plot

fig, ax = plt.subplots(3)

for channel in range(SWS_minute.shape[1]):
    ax[0].plot(
        SWS_minute[:, channel],
        alpha=0.5,
        label="Sleep Data",
    )
ax[0].set_title("non-REM ephys")
ax[0].set_ylabel("LFP (v)")
ax[0].set_xlabel("time (s)")
ax[0].margins(0)
for channel in range(REM_minute.shape[1]):
    ax[1].plot(REM_minute[:, channel], alpha=0.5, label="Wake Data", color="orange")
ax[1].set_ylabel("LFP (v)")
ax[1].set_xlabel("time (s)")
ax[1].set_title("REM ephys")
ax[1].margins(0)
for channel in range(RUN_minute.shape[1]):
    ax[2].plot(RUN_minute[:, channel], alpha=0.5, label="Wake Data", color="green")
ax[2].set_ylabel("LFP (v)")
ax[2].set_xlabel("time (s)")
ax[2].set_title("Running ephys")
ax[2].margins(0)
plt.show()


# %%
# Let's take the Fourier transforms of one channel for both waking and sleeping and see if differences are present
channel = 0
fig, ax = plt.subplots(3)
fft = nap.compute_spectogram(SWS_minute, fs=int(FS))
ax[0].plot(
    fft.index, np.abs(fft.iloc[:, channel]), alpha=0.5, label="Sleep Data", c="blue"
)
ax[0].set_xlim((0, FS / 2))
ax[0].set_xlabel("Freq (Hz)")
ax[0].set_ylabel("Frequency Power")

ax[0].set_title("non-REM LFP Decomposition")
fft = nap.compute_spectogram(REM_minute, fs=int(FS))
ax[1].plot(
    fft.index, np.abs(fft.iloc[:, channel]), alpha=0.5, label="Wake Data", c="orange"
)
ax[1].set_xlim((0, FS / 2))
fig.suptitle(f"Fourier Decomposition for channel {channel}")
ax[1].set_title("REM LFP Decomposition")
ax[1].set_xlabel("Freq (Hz)")
ax[1].set_ylabel("Frequency Power")

fft = nap.compute_spectogram(RUN_minute, fs=int(FS))
ax[2].plot(
    fft.index, np.abs(fft.iloc[:, channel]), alpha=0.5, label="Running Data", c="green"
)
ax[2].set_xlim((0, FS / 2))
fig.suptitle(f"Fourier Decomposition for channel {channel}")
ax[2].set_title("Running LFP Decomposition")
ax[2].set_xlabel("Freq (Hz)")
ax[2].set_ylabel("Frequency Power")

# ax.legend()
plt.show()


# %%
# Let's now consider the Welch spectograms of waking and sleeping data...

fig, ax = plt.subplots(3)
welch = nap.compute_welch_spectogram(SWS_minute, fs=int(FS))
ax[0].plot(
    welch.index,
    np.abs(welch.iloc[:, channel]),
    alpha=0.5,
    label="non-REM Data",
    color="blue"
)
ax[0].set_xlim((0, FS / 2))
ax[0].set_title("non-REM LFP Decomposition")
ax[0].set_xlabel("Freq (Hz)")
ax[0].set_ylabel("Frequency Power")
welch = nap.compute_welch_spectogram(REM_minute, fs=int(FS))
ax[1].plot(
    welch.index,
    np.abs(welch.iloc[:, channel]),
    alpha=0.5,
    label="REM Data",
    color="orange",
)
ax[1].set_xlim((0, FS / 2))
fig.suptitle(f"Welch Decomposition for channel {channel}")
ax[1].set_title("REM LFP Decomposition")
ax[1].set_xlabel("Freq (Hz)")
ax[1].set_ylabel("Frequency Power")

welch = nap.compute_welch_spectogram(RUN_minute, fs=int(FS))
ax[2].plot(
    welch.index,
    np.abs(welch.iloc[:, channel]),
    alpha=0.5,
    label="Running Data",
    color="green",
)
ax[2].set_xlim((0, FS / 2))
fig.suptitle(f"Welch Decomposition for channel {channel}")
ax[2].set_title("Running LFP Decomposition")
ax[2].set_xlabel("Freq (Hz)")
ax[2].set_ylabel("Frequency Power")
# ax.legend()
plt.show()

# %%
# There seems to be some differences presenting themselves - a bump in higher frequencies for waking data?
# Let's explore further with a wavelet decomposition


def plot_timefrequency(times, freqs, powers, x_ticks=5, y_ticks=None, ax=None, **kwargs):
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
        y_ticks = freqs
        y_ticks_pos = [np.argmin(np.abs(freqs - val)) for val in y_ticks]
    ax.set(yticks=y_ticks_pos, yticklabels=y_ticks)

fig = plt.figure(constrained_layout=True, figsize=(10, 50))
num_cells = 10
axd = fig.subplot_mosaic(
    [
        ["wd_sws"],
        ["lfp_sws"],
        ["wd_rem"],
        ["lfp_rem"],
        ["wd_run"],
        ["lfp_run"],
        ["pos_run"]
    ],
    height_ratios=[1, .2, 1, .2, 1, .2, .2]
)
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
        150.00,
        190.00,
        234.38,
        270.00,
        331.5,
        390.00,
        # 468.75,
        # 520.00,
        # 570.00,
        # 624.0,
    ]
)
mwt_SWS = nap.compute_wavelet_transform(
    SWS_minute[:, channel], fs=None, freqs=freqs
)
plot_timefrequency(
    SWS_minute.index.values[:],
    freqs[:],
    np.transpose(mwt_SWS[:, :].values),
    ax=axd["wd_sws"],
)
axd["wd_sws"].set_title(f"non-REM Data Wavelet Decomposition: Channel {channel}")

mwt_REM = nap.compute_wavelet_transform(REM_minute[:, channel], fs=None, freqs=freqs)
plot_timefrequency(
    REM_minute.index.values[:], freqs[:], np.transpose(mwt_REM[:, :].values), ax=axd["wd_rem"]
)
axd["wd_rem"].set_title(f"REM Data Wavelet Decomposition: Channel {channel}")

mwt_RUN = nap.compute_wavelet_transform(RUN_minute[:, channel], fs=None, freqs=freqs)
plot_timefrequency(
    RUN_minute.index.values[:], freqs[:], np.transpose(mwt_RUN[:, :].values), ax=axd["wd_run"]
)
axd["wd_run"].set_title(f"Running Data Wavelet Decomposition: Channel {channel}")

axd["lfp_sws"].plot(SWS_minute)
axd["lfp_rem"].plot(REM_minute)
axd["lfp_run"].plot(RUN_minute)
axd["pos_run"].plot(RUN_position)
axd["pos_run"].margins(0)
for k in ["lfp_sws", "lfp_rem", "lfp_run"]:
    axd[k].margins(0)
    axd[k].set_ylabel("LFP (v)")
    axd[k].get_xaxis().set_visible(False)
    axd[k].spines['top'].set_visible(False)
    axd[k].spines['right'].set_visible(False)
    axd[k].spines['bottom'].set_visible(False)
    axd[k].spines['left'].set_visible(False)
plt.show()

# %%g
freq = 3
interval = (REM_minute_interval["start"] + 0, REM_minute_interval["start"] + 5)
REM_second = REM_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
mwt_REM_second = mwt_REM.restrict(nap.IntervalSet(interval[0], interval[1]))
fig, ax = plt.subplots(1)
ax.plot(REM_second.index.values, REM_second[:, channel], alpha=0.5, label="Wake Data")
ax.plot(
    REM_second.index.values,
    mwt_REM_second[:, freq].values.real,
    label="Theta oscillations",
)
ax.set_title(f"{freqs[freq]}Hz oscillation power.")
plt.show()

# %%
# Let's plot spike phase, time scatter plots to see if spikes display phase characteristics during wakeful theta oscillation

fig = plt.figure(constrained_layout=True, figsize=(10, 50))
num_cells = 10
axd = fig.subplot_mosaic(
    [
        ["raw_lfp"]*2,
        ["wavelet"]*2,
        ["fit_wavelet"]*2,
        ["wavelet_power"]*2,
        ["wavelet_phase"]*2
    ] + [[f"spikes_phasetime_{i}", f"spikephase_hist_{i}"] for i in range(num_cells)],
)


# _, ax = plt.subplots(25, figsize=(10, 50))
mwt_REM = np.transpose(mwt_REM_second)
axd["raw_lfp"].plot(REM_second.index, REM_second.values[:, 0])
axd["raw_lfp"].margins(0)
plot_timefrequency(REM_second.index, freqs, np.abs(mwt_REM[:, :]), ax=axd["wavelet"])

axd["fit_wavelet"].plot(REM_second.index, REM_second.values[:, 0])
axd["fit_wavelet"].plot(REM_second.index, mwt_REM[freq, :].real)
axd["fit_wavelet"].set_title(f"{freqs[freq]}Hz")
axd["fit_wavelet"].margins(0)

axd["wavelet_power"].plot(REM_second.index, np.abs(mwt_REM[freq, :]))
axd["wavelet_power"].margins(0)
# ax[3].plot(lfp.index, lfp.values[:,0])
axd["wavelet_phase"].plot(REM_second.index, np.angle(mwt_REM[freq, :]))
axd["wavelet_phase"].margins(0)

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
                mwt_REM[freq, np.argmin(np.abs(REM_second.index.values - spike))]
            )
        )
    phase[i] = np.array(phase_i)

spikes = {k: v for k, v in spikes.items() if len(v) > 20}
phase = {k: v for k, v in phase.items() if len(v) > 20}

variances = {key: scipy.stats.circvar(value, low=-np.pi, high=np.pi) for key, value in phase.items()}
spikes = dict(sorted(spikes.items(), key=lambda item: variances[item[0]]))
phase = dict(sorted(phase.items(), key=lambda item: variances[item[0]]))

for i in range(num_cells):
    axd[f"spikes_phasetime_{i}"].scatter(spikes[list(spikes.keys())[i]], phase[list(phase.keys())[i]])
    axd[f"spikes_phasetime_{i}"].set_xlim(interval[0], interval[1])
    axd[f"spikes_phasetime_{i}"].set_ylim(-np.pi, np.pi)
    axd[f"spikes_phasetime_{i}"].set_xlabel("time (s)")
    axd[f"spikes_phasetime_{i}"].set_ylabel("phase")

    axd[f"spikephase_hist_{i}"].hist(phase[list(phase.keys())[i]], bins=np.linspace(-np.pi, np.pi, 10))
    axd[f"spikephase_hist_{i}"].set_xlim(-np.pi, np.pi)

plt.tight_layout()
plt.show()


# %%
# Let's focus on the sleeping data. Let's see if we can isolate the slow wave oscillations from the data
freq = 0
# interval = (10, 15)
interval = (SWS_minute_interval["start"] + 30, SWS_minute_interval["start"] + 50)
SWS_second = SWS_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
mwt_SWS_second = mwt_SWS.restrict(nap.IntervalSet(interval[0], interval[1]))
_, ax = plt.subplots(1)
ax.plot(SWS_second[:, channel], alpha=0.5, label="Wake Data")
ax.plot(
    SWS_second.index.values,
    mwt_SWS_second[:, freq].values.real,
    label="Slow Wave Oscillations",
)
ax.set_title(f"{freqs[freq]}Hz oscillation power")
plt.show()

# %%
# Let's plot spike phase, time scatter plots to see if spikes display phase characteristics during wakeful theta oscillation

fig = plt.figure(constrained_layout=True, figsize=(10, 50))
num_cells = 5
axd = fig.subplot_mosaic(
    [
        ["raw_lfp"]*2,
        ["wavelet"]*2,
        ["fit_wavelet"]*2,
        ["wavelet_power"]*2,
        ["wavelet_phase"]*2
    ] + [[f"spikes_phasetime_{i}", f"spikephase_hist_{i}"] for i in range(num_cells)],
)


# _, ax = plt.subplots(25, figsize=(10, 50))
mwt_SWS = np.transpose(mwt_SWS_second)
axd["raw_lfp"].plot(SWS_second.index, SWS_second.values[:, 0])
axd["raw_lfp"].margins(0)

plot_timefrequency(SWS_second.index, freqs, np.abs(mwt_SWS[:, :]), ax=axd["wavelet"])

axd["fit_wavelet"].plot(SWS_second.index, SWS_second.values[:, 0])
axd["fit_wavelet"].plot(SWS_second.index, mwt_SWS[freq, :].real)
axd["fit_wavelet"].set_title(f"{freqs[freq]}Hz")
axd["fit_wavelet"].margins(0)

axd["wavelet_power"].plot(SWS_second.index, np.abs(mwt_SWS[freq, :]))
axd["wavelet_power"].margins(0)
axd["wavelet_phase"].plot(SWS_second.index, np.angle(mwt_SWS[freq, :]))
axd["wavelet_phase"].margins(0)

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
                mwt_SWS[freq, np.argmin(np.abs(SWS_second.index.values - spike))]
            )
        )
    phase[i] = np.array(phase_i)

spikes = {k: v for k, v in spikes.items() if len(v) > 0}
phase = {k: v for k, v in phase.items() if len(v) > 0}

for i in range(num_cells):
    axd[f"spikes_phasetime_{i}"].scatter(spikes[list(spikes.keys())[i]], phase[list(phase.keys())[i]])
    axd[f"spikes_phasetime_{i}"].set_xlim(interval[0], interval[1])
    axd[f"spikes_phasetime_{i}"].set_ylim(-np.pi, np.pi)
    axd[f"spikes_phasetime_{i}"].set_xlabel("time (s)")
    axd[f"spikes_phasetime_{i}"].set_ylabel("phase")

    axd[f"spikephase_hist_{i}"].hist(phase[list(phase.keys())[i]], bins=np.linspace(-np.pi, np.pi, 10))
    axd[f"spikephase_hist_{i}"].set_xlim(-np.pi, np.pi)

plt.tight_layout()
plt.show()

# %%
# Let's focus on the sleeping data. Let's see if we can isolate the slow wave oscillations from the data
# interval = (10, 15)

# for run in [-16, -15, -13, -20]:
#     interval = (
#         data["forward_ep"]["start"][run],
#         data["forward_ep"]["end"][run]+3.,
#     )
#     print(interval)
#     RUN_second_r = RUN_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
#     RUN_position_r = RUN_position.restrict(nap.IntervalSet(interval[0], interval[1]))
#     mwt_RUN_second_r = mwt_RUN.restrict(nap.IntervalSet(interval[0], interval[1]))
#     _, ax = plt.subplots(3)
#     plot_timefrequency(
#         RUN_second_r.index.values[:], freqs[:], np.transpose(mwt_RUN_second_r[:, :].values), ax=ax[0]
#     )
#     ax[1].plot(RUN_second_r[:, channel], alpha=0.5, label="Wake Data")
#     ax[1].margins(0)
#
#     ax[2].plot(RUN_position, alpha=0.5, label="Wake Data")
#     ax[2].set_xlim(RUN_second_r[:, channel].index.min(), RUN_second_r[:, channel].index.max())
#     ax[2].margins(0)
#     plt.show()


RUN_minute_interval = nap.IntervalSet(
    data["forward_ep"]["start"][0],
    data["forward_ep"]["end"][-1]
)

RUN_minute = nap.TsdFrame(
    t=data["eeg"].restrict(RUN_minute_interval).index.values,
    d=data["eeg"].restrict(RUN_minute_interval).values,
    time_support=data["eeg"].restrict(RUN_minute_interval).time_support,
)

RUN_position = nap.TsdFrame(
    t=data["position"].restrict(RUN_minute_interval).index.values[:],
    d=data['position'].restrict(RUN_minute_interval),
    time_support=data["position"].restrict(RUN_minute_interval).time_support,
)

mwt_RUN = nap.compute_wavelet_transform(RUN_minute[:, channel],
                                        freqs=freqs,
                                        fs=None,
                                        norm=None,
                                        n_cycles=3.5,
                                        scaling=1)

for run in range(len(data["forward_ep"]["start"])):
    interval = (
        data["forward_ep"]["start"][run],
        data["forward_ep"]["end"][run]+5.,
    )
    if interval[1] - interval[0] < 6:
        continue
    print(interval)
    RUN_second_r = RUN_minute.restrict(nap.IntervalSet(interval[0], interval[1]))
    RUN_position_r = RUN_position.restrict(nap.IntervalSet(interval[0], interval[1]))
    mwt_RUN_second_r = mwt_RUN.restrict(nap.IntervalSet(interval[0], interval[1]))
    _, ax = plt.subplots(3)
    plot_timefrequency(
        RUN_second_r.index.values[:], freqs[:], np.transpose(mwt_RUN_second_r[:, :].values), ax=ax[0]
    )
    ax[1].plot(RUN_second_r[:, channel], alpha=0.5, label="Wake Data")
    ax[1].margins(0)

    ax[2].plot(RUN_position, alpha=0.5, label="Wake Data")
    ax[2].set_xlim(RUN_second_r[:, channel].index.min(), RUN_second_r[:, channel].index.max())
    ax[2].margins(0)
    plt.show()
