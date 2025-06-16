---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

Power spectral density
======================


```{code-cell} ipython3
:tags: [hide-input]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```


***
Generating a signal
------------------
Let's generate a dummy signal with 2Hz and 10Hz sinusoide with white noise.



```{code-cell} ipython3
F = [2, 10]

Fs = 2000
t = np.arange(0, 200, 1/Fs)
sig = nap.Tsd(
    t=t,
    d=np.cos(t*2*np.pi*F[0])+np.cos(t*2*np.pi*F[1])+2*np.random.normal(0, 3, len(t)),
    time_support = nap.IntervalSet(0, 200)
    )
```

```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
plt.plot(sig.get(0, 0.4))
plt.title("Signal")
plt.xlabel("Time (s)")
```

Computing FFT
--------------------------------------

To compute the FFT of a signal, you can use the function [`nap.compute_fft`](pynapple.process.spectrum.compute_fft). With `norm=True`, the output of the FFT is divided by the length of the signal.


```{code-cell} ipython3
fft = nap.compute_fft(sig, norm=True)
```

Pynapple returns a pandas DataFrame.


```{code-cell} ipython3
print(fft)
```

It is then easy to plot it.


```{code-cell} ipython3
plt.figure()
plt.plot(np.abs(fft))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
```

Note that the output of the FFT is truncated to positive frequencies. To get positive and negative frequencies, you can set `full_range=True`.
By default, the function returns the frequencies up to the Nyquist frequency.
Let's zoom on the first 20 Hz.


```{code-cell} ipython3
plt.figure()
plt.plot(np.abs(fft))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 20)
```

We find the two frequencies 2 and 10 Hz.

By default, pynapple assumes a constant sampling rate and a single epoch. For example, computing the FFT over more than 1 epoch will raise an error.


```{code-cell} ipython3
double_ep = nap.IntervalSet([0, 50], [20, 100])

try:
    nap.compute_fft(sig, ep=double_ep)
except ValueError as e:
    print(e)
```

Computing power spectral density (PSD)
--------------------------------------

Power spectral density can be returned through the function [`compute_power_spectral_density`](pynapple.process.spectrum.compute_power_spectral_density). Contrary to `compute_fft`, the
output is real-valued.

```{code-cell} ipython3
psd = nap.compute_power_spectral_density(sig, fs=Fs)
```

The output is also a pandas DataFrame.

```{code-cell} ipython3
plt.figure()
plt.plot(psd)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency")
plt.xlim(0, 20)
```

Computing mean PSD
------------------

It is possible to compute an average PSD over multiple epochs with the function [`nap.compute_mean_power_spectral_density`](pynapple.process.spectrum.compute_mean_power_spectral_density).

In this case, the argument `interval_size` determines the duration of each epochs upon which the FFT is computed.
If not epochs is passed, the function will split the `time_support`.

In this case, the FFT will be computed over epochs of 20 seconds.


```{code-cell} ipython3
mean_psd = nap.compute_mean_power_spectral_density(sig, interval_size=20.0, fs=Fs)
```

Let's compare `mean_psd` to `psd`. In both cases, the output is normalized and converted to dB.


```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
plt.plot(10*np.log10(psd), label='PSD')
plt.plot(10*np.log10(mean_psd), label='Mean PSD (20s)')

plt.ylabel("Power/Frequency (dB/Hz)")
plt.legend()
plt.xlim(0, 20)
plt.xlabel("Frequency (Hz)")

```

As we can see, [`nap.compute_mean_power_spectral_density`](pynapple.process.spectrum.compute_mean_power_spectral_density) was able to smooth out the noise.
