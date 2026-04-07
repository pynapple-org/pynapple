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

# Phases and envelopes
In this tutorial, we will introduce Pynapple's functionality for computing signal phases
and envelopes.
Most of this functionality is part of the [`pynapple.process.signal`](pynapple.process.signal) module and is built on [the 
Hilbert transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html).

```{code-cell} ipython3
:tags: [hide-input]
# we'll import the packages we're going to use
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns

# some configuration, you can ignore this
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(
    style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params
)
```

## Extracting the analytic signal
Let us start by simulating a simple oscillatory signal.
As an example, we will simulate a theta oscillation at 8Hz, 
with nested-gamma oscillations at 40Hz [(a known phenomenon in the brain)](https://www.sciencedirect.com/science/article/pii/S0896627313002316):
```{code-cell} ipython3
sampling_rate_hz = 1000
times = np.arange(0, 5, 1 / sampling_rate_hz)

# Theta oscillation (8 Hz)
theta_freq_hz = 8
theta = np.cos(2 * np.pi * theta_freq_hz * times)

# Gamma oscillation (40 Hz)
gamma_freq_hz = 40
gamma = np.cos(2 * np.pi * gamma_freq_hz * times)

# Compute theta phase
theta_phase = np.angle(np.exp(1j * 2 * np.pi * theta_freq_hz * times))

# Create square burst envelopes near theta peak (phase ~ 0)
phase_window = np.abs(theta_phase) < (np.pi / 6)  # narrow window
burst_envelope = np.zeros_like(times)
burst_centers = np.where(np.diff(phase_window.astype(int)) == 1)[0]
burst_duration_samples = int(0.03 * sampling_rate_hz)  # 30 ms burst length

for c in burst_centers:
    start_idx = c
    end_idx = min(c + burst_duration_samples, len(times))
    burst_envelope[start_idx:end_idx] = 1.0  # full amplitude square pulse

# Final signal: theta + bursty gamma
signal = theta + burst_envelope * gamma

# Convert to Tsd
signal = nap.Tsd(t=times, d=signal)
```

Let's visualize that:
```{code-cell} ipython3
:tags: [hide-input]
segment = nap.IntervalSet(1, 2)
plt.figure(figsize=(10,3))
plt.plot(signal.restrict(segment))
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.tight_layout();
```

Now, imagine that we are interested in the theta frequency part of the signal.
We can start by applying a bandpass filter using [`apply_bandpass_filter`](pynapple.process.filtering.apply_bandpass_filter)
to keep only the relevant frequencies (5-10Hz):
```{code-cell} ipython3
filtered_signal = nap.apply_bandpass_filter(
    signal, (5, 10), fs=sampling_rate_hz, mode="butter"
)
filtered_signal
```

Let's visualize that together with the original signal:
```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(10,3))
plt.plot(signal.restrict(segment), label="signal")
plt.plot(filtered_signal.restrict(segment), label="filtered signal")
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout();
```

We can now use [`apply_hilbert_transform`](pynapple.process.signal.apply_hilbert_transform)
to extract the analytic signal:
```{code-cell} ipython3
analytic_signal = nap.apply_hilbert_transform(filtered_signal)
analytic_signal
```

If we visualize the analytic signal with the input signal,
you will notice that the analytic signal appears identical to the original signal:
```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(10,3))
plt.plot(filtered_signal.restrict(segment), label="filtered signal")
plt.plot(analytic_signal.restrict(segment), label="analytic signal")
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout();
```

This happens because the analytic signal is complex-valued.
The real part is exactly the input signal.
The imaginary part is the Hilbert transform (a 90° phase-shifted version).
When you pass a complex signal to matplotlib,
it automatically plots only the real part (see the warnings above).

To actually see what’s going on, you can plot the real and imaginary parts separately:
```{code-cell} ipython3
:tags: [hide-input]
plt.plot(np.real(analytic_signal).restrict(segment), label="real part")
plt.plot(
    np.imag(analytic_signal).restrict(segment),
    label="imaginary part",
)
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1));
```

## Computing the signal envelope
From the analytic signal, it is often the case that we will compute other things.
For one, we can extract the envelope of a signal, by taking the absolute value.
To make things easy, Pynapple provides [`compute_hilbert_envelope`](pynapple.process.signal.compute_hilbert_envelope)
to compute the envelope in one go:
```{code-cell} ipython3
envelope = nap.compute_hilbert_envelope(filtered_signal)
envelope
```

Visualizing the envelope over the signal:
```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(10,3))
plt.plot(filtered_signal.restrict(segment), label="filtered signal")
plt.plot(envelope.restrict(segment), label="envelope", color="red")
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout();
```

## Computing the signal phase
We can also estimate the signal's phase, by taking angle and wrapping.
To make things easy, Pynapple provides [`compute_hilbert_phase`](pynapple.process.signal.compute_hilbert_phase)
to compute the phase in one go:
```{code-cell} ipython3
phase = nap.compute_hilbert_phase(filtered_signal)
phase
```

Visualizing the phase with the signal:
```{code-cell} ipython3
:tags: [hide-input]
fig, (ax_sig, ax_phase) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax_sig.plot(filtered_signal.restrict(segment), label="filtered signal")
ax_sig.set_ylabel("amplitude (a.u.)")
ax_phase.plot(phase.restrict(segment), label="phase")
ax_phase.set_xlabel("time (s)")
ax_phase.set_ylabel("phase (rad)")
plt.tight_layout()
```

## Detecting oscillatory events
Having looked at the theta part of our signal, we might also be interested in the gamma part.
To start with, we might simply be interested in finding the epochs where the signal is oscillating
at gamma frequencies.
Pynapple provides the [`detect_oscillatory_events`](pynapple.process.signal.detect_oscillatory_events)
exactly for such a goal.

To get it to work nicely, you will have the tune the following detection parameters:
- frequency band: the band of frequencies you are interested in (35 to 45Hz for gamma).
- threshold band: minimum and maximum thresholds to apply to the z-scored envelope of the squared signal.
- minimum and maximum duration of the events
- minimum interval between events
```{code-cell} ipython3
# Define detection parameters
freq_band = (35, 45)   # Gamma band
thres_band = (0.25, 5) # Thresholds for normalized squared envelope
min_dur = 0.04         # Minimum event duration
max_dur = 0.08         # Max event duration
min_inter = 0.05       # Minimum interval between events
epoch = signal.time_support

# Detect oscillatory events
events = nap.detect_oscillatory_events(
    signal,
    epoch,
    freq_band,
    thres_band,
    (min_dur, max_dur),
    min_inter,
)
events
```

We can then visualize to found events on top of the original signal as validation:
```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(10, 3))
plt.plot(signal.restrict(segment), label="signal")
first = True
for s, e in events.intersect(segment).values:
    if first:
        plt.axvspan(s, e, color="orange", alpha=0.3, label="gamma\nevent")
        first = False
    else:
        plt.axvspan(s, e, color="orange", alpha=0.3)
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout();
```
