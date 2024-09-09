# -*- coding: utf-8 -*-
"""
Filtering
=========

The filtering module holds the functions for frequency manipulation :

- `nap.compute_bandstop_filter`
- `nap.compute_lowpass_filter`
- `nap.compute_highpass_filter`
- `nap.compute_bandpass_filter`

The functions have similar calling signatures. For example, to filter a 1000 Hz signal between
10 and 20 Hz using a Butterworth filter:

```{python}
>>> new_tsd = nap.compute_bandpass_filter(tsd, (10, 20), fs=1000, mode='butter')
```

Currently, the filtering module provides two methods for frequency manipulation: `butter`
for a recursive Butterworth filter and `sinc` for a Windowed-sinc convolution. This notebook provides
a comparison of the two methods.
"""

# %%
# !!! warning
#     This tutorial uses matplotlib for displaying the figure
#
#     You can install all with `pip install matplotlib requests tqdm seaborn`
#
# mkdocs_gallery_thumbnail_number = 1
#
# Now, import the necessary libraries:

import matplotlib.pyplot as plt
import numpy as np
import seaborn

import pynapple as nap

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
seaborn.set_theme(context='notebook', style="ticks", rc=custom_params)

# %%
# ***
# Introduction
# ------------
#
# We start by generating a signal with multiple frequencies (2, 10 and 50 Hz).
fs = 1000 # sampling frequency
t = np.linspace(0, 2, fs * 2)
f2 = np.cos(t*2*np.pi*2)
f10 = np.cos(t*2*np.pi*10)
f50 = np.cos(t*2*np.pi*50)

sig = nap.Tsd(t=t,d=f2+f10+f50 + np.random.normal(0, 0.5, len(t)))

# %%
# Let's plot  it
fig = plt.figure(figsize = (15, 5))
plt.plot(sig)
plt.xlabel("Time (s)")


# %%
# We can compute the Fourier transform of `sig` to verify that all the frequencies are there.
psd = nap.compute_power_spectral_density(sig, fs, norm=True)

fig = plt.figure(figsize = (15, 5))
plt.plot(np.abs(psd))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 100)


# %%
# Let's say we would like to see only the 10 Hz component.
# We can use the function `compute_bandpass_filter` with mode `butter` for Butterworth.

sig_butter = nap.compute_bandpass_filter(sig, (8, 12), fs, mode='butter')

# %%
# Let's compare it to the `sinc` mode for Windowed-sinc.
sig_sinc = nap.compute_bandpass_filter(sig, (8, 12), fs, mode='sinc')

# %%
# Let's plot it
fig = plt.figure(figsize = (15, 5))
plt.subplot(211)
plt.plot(t, f10, '-', color = 'gray', label = "10 Hz component")
plt.xlim(0, 1)
plt.legend()
plt.subplot(212)
# plt.plot(sig, alpha=0.5)
plt.plot(sig_butter, label = "Butterworth")
plt.plot(sig_sinc, '--', label = "Windowed-sinc")
plt.legend()
plt.xlabel("Time (Hz)")
plt.xlim(0, 1)


# %%
# This gives similar results except at the edges.
#
# Another use of filtering is to remove some frequencies. Here we can try to remove
# the 50 Hz component in the signal.

sig_butter = nap.compute_bandstop_filter(sig, cutoff=(45, 55), fs=fs, mode='butter')
sig_sinc = nap.compute_bandstop_filter(sig, cutoff=(45, 55), fs=fs, mode='sinc')


# %%
# Let's plot it
fig = plt.figure(figsize = (15, 5))
plt.subplot(211)
plt.plot(t, sig, '-', color = 'gray', label = "Original signal")
plt.xlim(0, 1)
plt.legend()
plt.subplot(212)
# plt.plot(sig, alpha=0.5)
plt.plot(sig_butter, label = "Butterworth")
plt.plot(sig_sinc, '--', label = "Windowed-sinc")
plt.legend()
plt.xlabel("Time (Hz)")
plt.xlim(0, 1)


# %%
# Let's see what frequencies remain;

psd_butter = nap.compute_power_spectral_density(sig_butter, fs, norm=True)
psd_sinc = nap.compute_power_spectral_density(sig_sinc, fs, norm=True)

fig = plt.figure(figsize = (10, 5))
plt.plot(np.abs(psd_butter), label = "Butterworth filter")
plt.plot(np.abs(psd_sinc), label = "Windowed-sinc convolution")
plt.legend()
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 70)


# %%
# The remaining notebook compares the two modes.
#
# ***
# Frequency responses
# -------------------
#
# In order to check the validity of the filter, the function `get_filter_frequency_response` provides the frequency
# response of the filters. The calling signature is similar to the previous functions.
# The function returns a pandas Series with the frequencies as index.
#
# Let's get the frequency response for a Butterworth low pass filter with different order:
butter_freq = {
    order: nap.get_filter_frequency_response(250, fs, "lowpass", "butter", order=order)
    for order in [2, 4, 6]}

# %%
# ... and the frequency response for the Windowed-sinc equivalent with different transition bandwidth.
sinc_freq = {
    tb:nap.get_filter_frequency_response(250, fs,"lowpass", "sinc", transition_bandwidth=tb)
    for tb in [0.002, 0.02, 0.2]}

# %%
# Let's plot the frequency response of both.

from scipy import signal

fig = plt.figure(figsize = (20, 10))
gs = plt.GridSpec(2, 2)
for order in butter_freq.keys():
    plt.subplot(gs[0, 0])
    plt.plot(butter_freq[order], label = f"order={order}")
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Butterworth recursive")
    plt.subplot(gs[1, 0])
    plt.plot(20*np.log10(butter_freq[order]), label = f"order={order}")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.ylim(-200,20)
    plt.legend()

for tb in sinc_freq.keys():
    plt.subplot(gs[0, 1])
    plt.plot(sinc_freq[tb], label= f"width={tb}")
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Windowed-sinc conv.")
    plt.subplot(gs[1, 1])
    plt.plot(20*np.log10(sinc_freq[tb]), label= f"width={tb}")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.ylim(-200,20)
    plt.legend()

# %%
# In some cases, the transition bandwidth that is too high generates a kernel that is too short. The amplitude of the
# original signal will then be lower than expected.
# In this case, the solution is to decrease the transition bandwidth when using the windowed-sinc mode.
# Note that this increases the length of the kernel significantly.
# Let see it with the band pass filter.


sinc_freq = {
    tb:nap.get_filter_frequency_response((100, 200), fs, "bandpass", "sinc", transition_bandwidth=tb)
    for tb in [0.004, 0.5]}


fig = plt.figure(figsize = (20, 10))
for tb in sinc_freq.keys():
    plt.plot(sinc_freq[tb], label= f"width={tb}")
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title("Windowed-sinc conv.")
    plt.legend()



# %%
# ***
# Performances
# ------------
# Let's compare the performance of each when varying the number of time points and the number of dimensions.
from time import perf_counter

def get_mean_perf(tsd, mode, n=10):
    tmp = np.zeros(n)
    for i in range(n):
        t1 = perf_counter()
        _ = nap.compute_lowpass_filter(tsd, 0.25 * tsd.rate, mode=mode)
        t2 = perf_counter()
        tmp[i] = t2 - t1
    return [np.mean(tmp), np.std(tmp)]

def benchmark_time_points(mode):
    times = []
    for T in np.arange(1000, 100000, 40000):
        time_array = np.arange(T)/1000
        data_array = np.random.randn(len(time_array))
        startend = np.linspace(0, time_array[-1], T//100).reshape(T//200, 2)
        ep = nap.IntervalSet(start=startend[::2,0], end=startend[::2,1])
        tsd = nap.Tsd(t=time_array, d=data_array, time_support=ep)
        times.append([T]+get_mean_perf(tsd, mode))
    return np.array(times)

def benchmark_dimensions(mode):
    times = []
    for n in np.arange(1, 100, 30):
        time_array = np.arange(10000)/1000
        data_array = np.random.randn(len(time_array), n)
        startend = np.linspace(0, time_array[-1], 10000//100).reshape(10000//200, 2)
        ep = nap.IntervalSet(start=startend[::2,0], end=startend[::2,1])
        tsd = nap.TsdFrame(t=time_array, d=data_array, time_support=ep)
        times.append([n]+get_mean_perf(tsd, mode))
    return np.array(times)


times_sinc = benchmark_time_points(mode="sinc")
times_butter = benchmark_time_points(mode="butter")

dims_sinc = benchmark_dimensions(mode="sinc")
dims_butter = benchmark_dimensions(mode="butter")


plt.figure(figsize = (16, 5))
plt.subplot(121)
for arr, label in zip(
    [times_sinc, times_butter],
    ["Windowed-sinc", "Butter"],
    ):
    plt.plot(arr[:, 0], arr[:, 1], "o-", label=label)
    plt.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.2)
plt.legend()
plt.xlabel("Number of time points")
plt.ylabel("Time (s)")
plt.title("Low pass filtering benchmark")
plt.subplot(122)
for arr, label in zip(
    [dims_sinc, dims_butter],
    ["Windowed-sinc", "Butter"],
    ):
    plt.plot(arr[:, 0], arr[:, 1], "o-", label=label)
    plt.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2], alpha=0.2)
plt.legend()
plt.xlabel("Number of dimensions")
plt.ylabel("Time (s)")
plt.title("Low pass filtering benchmark")

plt.show()
