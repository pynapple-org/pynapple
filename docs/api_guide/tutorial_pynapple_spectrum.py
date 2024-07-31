# -*- coding: utf-8 -*-
"""
Power spectral density
======================

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

F = [2, 10]

Fs = 2000
t = np.arange(0, 100, 1/Fs)
sig = nap.Tsd(
    t=t,
    d=np.cos(t*2*np.pi*F[0])+np.cos(t*2*np.pi*F[1])+np.random.normal(0, 2, len(t)),
    time_support = nap.IntervalSet(0, 10)
    )

# %%
# Let's plot it
plt.figure()
plt.plot(sig.get(0, 1))
plt.title("Signal")
plt.show()


# %%
# Computing power spectral density (PSD)
# --------------------------------------
#
# To compute a PSD of a signal, you can use the function `nap.compute_power_spectral_density`

psd = nap.compute_power_spectral_density(sig)

# %%
# Pynapple returns a pandas DataFrame.

print(psd)

# %%
# It is then easy to plot it.

plt.figure()
plt.plot(psd)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()

# %%
# Note that the output of the FFT is truncated to positive frequencies. To get positive and negative frequencies, you can set `full_range=True`.
# By default, the function returns the frequencies up to the Nyquist frequency.
# Let's zoom on the first 20 Hz.

plt.figure()
plt.plot(psd)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 20)
plt.show()

# %%
# We find the two frequencies 2 and 10 Hz.
#
# By default, pynapple assumes a constant sampling rate and a single epoch. For example, computing the FFT over more than 1 epoch will raise an error.
double_ep = nap.IntervalSet([0, 50], [20, 100])

try:
    nap.compute_power_spectral_density(sig, ep=double_ep)
except ValueError as e:
    print(e)


# %%
# Computing mean PSD
# ------------------
#
# It is possible to compute an average PSD over multiple epochs with the function `nap.compute_mean_power_spectral_density`.
# 
# In this case, the argument `interval_size` determines the duration of each epochs upon which the fft is computed.
# If not epochs is passed, the function will split the `time_support`.



