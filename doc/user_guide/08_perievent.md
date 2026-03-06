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

# Perievent / spike-trigerred averages

The perievent module allows for aligning timeseries and timestamps data around events, 
as well as computing event-triggered averages (e.g. spike-trigerred averages).

```{contents}
:local:
:depth: 3
```

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

## Peri-Event Time Histograms (PETH)

### Spike data

We will start with the common use-case of aligning the spiking activity of a unit to a set of events.

#### Single unit

Let's simulate some uniform stimuli and a unit that has a gaussian firing field after the stimulus:
```{code-cell} ipython3
stimuli_times = np.arange(0, 1000, 5)
stimuli = nap.Tsd(t=stimuli_times, d=np.random.rand(200), time_units="s")

def generate_spiking_unit(offset):
    baseline = np.random.uniform(0, 1000, 500)
    burst = np.concatenate([
        np.random.normal(st + offset, 0.05, 3) for st in stimuli_times
    ])
    return nap.Ts(t=np.sort(np.concatenate([baseline, burst])), time_units="s")

ts = generate_spiking_unit(offset=0.1)
```

The [`compute_perievent`](pynapple.process.perievent.compute_perievent) function
can align timestamps to a particular set of events:

```{code-cell} ipython3
window = (-0.1, 0.4)
peth = nap.compute_perievent(data=ts, events=stimuli, window=window)
peth
```

The returned object is a `TsGroup` containing the aligned timestamps per event.
The event times are stored in the `events` metadata column.

If you want to get aligned counts, you can now easily call:
```{code-cell} ipython3
bin_size = 0.01
peth_counts = peth.count(bin_size)
peth_counts
```

Visualizing the aligned timestamps can now be done by using the [`to_tsd`](pynapple.TsGroup.to_tsd) function.
This function flattens the `TsGroup` into one `Tsd` containing all the timestamps and storing the event ids as data.
We can also take the mean of the counts and divide by the bin size to show an estimate of the aligned firing rate:

```{code-cell} ipython3
def plot_peth(unit_peth, unit_peth_counts, ax_mean, ax_spikes):
    ax_mean.plot(np.mean(unit_peth_counts, axis=1) / bin_size)
    ax_mean.set_ylabel("spikes/s")
    ax_mean.axvline(0.0, color="red")
    ax_spikes.plot(unit_peth.to_tsd(), "|", markersize=5)
    ax_spikes.set_xlabel("time from event (s)")
    ax_spikes.set_ylabel("event")
    ax_spikes.axvline(0.0, color="red")

fig, (ax_mean, ax_spikes) = plt.subplots(
    2, 1, sharex=True, height_ratios=[0.3, 1], figsize=(6, 6)
)
plot_peth(peth, peth_counts, ax_mean, ax_spikes)
```

#### Multiple units

The same function can be applied to a group of units:
```{code-cell} ipython3
tsgroup = nap.TsGroup(
    {1: ts, 2: generate_spiking_unit(offset=0.2), 3: generate_spiking_unit(0.3)}
)
peth = nap.compute_perievent(data=tsgroup, events=stimuli, window=window)
peth
```
In this case, it returns a dict of `TsGroup`, containing the same object as before, but now per unit.
We can again visualize easily:

```{code-cell} ipython3
fig, axs = plt.subplots(
    2,
    len(tsgroup),
    sharey="row",
    sharex=True,
    height_ratios=[0.3, 1.0],
    figsize=(15, 8),
)

for unit, unit_axs in zip(tsgroup, axs.T):
    plot_peth(peth[unit], peth[unit].count(bin_size), *unit_axs)
```

### Continuous data

If you have continuous data, e.g. calcium imaging traces, you can use the exact same function!
[`compute_perievent`](pynapple.process.perievent.compute_perievent) is designed in such a way
that it recognizes the input format and decides the correct computation.
Hence, when given a `Tsd`, `TsdFrame` or even a `TsdTensor`, it will know what to do.


```{admonition} Note
[`compute_perievent`](pynapple.process.perievent.compute_perievent) only works with regularly 
sampled continuous data. If your data is irregularly sampled, try padding it with `nan` first.
```

#### Single unit
Let's again start by simulating some data, but this time continuous traces:

```{code-cell} ipython3
def generate_continuous_unit(burst_offset):
    t = np.arange(0, 1000, 0.02)
    d = np.random.uniform(0, 1, len(t))
    for st in stimuli_times:
        d += 4 * np.exp(-((t - (st + burst_offset)) ** 2) / (2 * 0.1**2))
    return nap.Tsd(t=t, d=np.clip(d, 0, 5), time_units="s")

tsd = generate_continuous_unit(burst_offset=0.1)
```

We can pass continuous units (as a `Tsd`) to the function in the exact same way:
```{code-cell} ipython3
peth = nap.compute_perievent(data=tsd, events=stimuli, window=window)
peth
```
This time, it will return a `TsdFrame` with a column per event.

We can again visualize, this time using a heatmap (i.e. using `imshow`):

```{code-cell} ipython3
def plot_peth_continuous(unit_peth, ax_mean, ax):
    ax_mean.plot(np.nanmean(unit_peth, axis=1))
    ax_mean.set_ylabel("dF/F [a.u.]")
    im = ax.imshow(
        unit_peth.values.T,
        extent=(unit_peth.times()[0], unit_peth.times()[-1], 0, unit_peth.shape[1]),
        interpolation="none",
        aspect="auto",
        cmap="Grays",
    )
    ax.axvline(0.0, color="red")
    ax.set_xlabel("time from event (s)")
    ax.set_ylabel("event")
    return im

fig, (ax_mean, ax) = plt.subplots(
    2, 1, sharex=True, height_ratios=[0.3, 1.0], figsize=(6, 7)
)
im = plot_peth_continuous(peth, ax_mean, ax)
fig.colorbar(im, cax=fig.add_axes([0.92, 0.14, 0.02, 0.3]), label="dF/F [a.u.]");
```

#### Multiple units
The same function can also handle multiple continuous units, passed as a `TsdFrame`:

```{code-cell} ipython3
tsdframe = np.stack(
    [tsd, generate_continuous_unit(0.2), generate_continuous_unit(0.3)], axis=1
)
peth = nap.compute_perievent(data=tsdframe, events=stimuli, window=window)
peth
```

In this case, it returns a `TsdTensor`, containing an added dimension for the events.
We can again visualize by reusing our plotting function:

```{code-cell} ipython3
fig, axs = plt.subplots(
    2, len(tsdframe.columns), sharex=True, height_ratios=[0.3, 1.0], figsize=(15, 8)
)

for unit, unit_axs in zip(tsdframe.columns, axs.T):
    im = plot_peth_continuous(peth[:, :, unit], *unit_axs)
fig.colorbar(im, cax=fig.add_axes([0.92, 0.14, 0.02, 0.3]), label="dF/F [a.u.]");
```

In the most complex case, you can even pass a `TsdTensor` to [`compute_perievent`](pynapple.process.perievent.compute_perievent).
It will return a `TsdTensor` with an added dimension for the events.
Visualizing such a tensor becomes a bit complex, but feel free to try!

## Event-Triggered Average (ETA)

The [`compute_event_triggered_average`](pynapple.process.perievent.compute_event_triggered_average) computes
the average of a continuous feature aligned to a set of events.
The classic use-case is to recover the stimulus feature that drives a neuron's spiking.

```{note}
In neuroscience, this is commonly known as a spike-triggered average (STA).
For convenience, [`compute_spike_triggered_average`](pynapple.process.perievent.compute_event_triggered_average)
is provided as an alias.
```

### Single unit
Let's simulate a slowly varying stimulus and a neuron that fires preferentially at its peaks:
```{code-cell} ipython3
t = np.arange(0, 1000, 0.02)
feature = nap.Tsd(t=t, d=np.sin(2 * np.pi * t / 10), time_units="s")

def generate_spiking_unit(phase):
    rate = np.clip(np.sin(2 * np.pi * t / 10 + phase) * 10 + 10, 0, None)
    return nap.Ts(t=np.sort(t[np.random.rand(len(t)) < rate * 0.02]), time_units="s")

ts = generate_spiking_unit(phase=0.0)
```

We can pass this to the function, passing a bin size and a window:
```{code-cell} ipython3
eta = nap.compute_event_triggered_average(feature, ts, binsize=0.02, window=(-5,5))
eta
```

The result is a `TsdFrame` with one column. If the neuron is driven by the stimulus,
the ETA should recover the stimulus waveform preceding each spike:
```{code-cell} ipython3
plt.plot(eta)
plt.axvline(0.0, color="red")
plt.xlabel("time from spike (s)")
plt.ylabel("stimulus [a.u.]");
```

### Multiple units

The same function can be used for a group of units.
When passing a `TsGroup`, the function returns one column per unit.
Here, we simulate units driven by the stimulus at different phases:
```{code-cell} ipython3
tsgroup = nap.TsGroup({
    1: ts,
    2: generate_spiking_unit(phase=np.pi / 2),
    3: generate_spiking_unit(phase=np.pi),
})
eta = nap.compute_event_triggered_average(feature, tsgroup, binsize=0.02, window=10.0)
eta
```

Each unit recovers a phase-shifted version of the stimulus:
```{code-cell} ipython3
fig, ax = plt.subplots()
for unit in range(len(eta.columns)):
    ax.plot(eta[:, unit], label=f"unit {unit}")
ax.axvline(0.0, color="red")
ax.set_xlabel("time from spike (s)")
ax.set_ylabel("stimulus [a.u.]")
ax.legend();
```

### Multiple features

You can also pass multiple features at once.
Here, we simulate two units each driven by a different frequency, and pass both
stimulus features together:
```{code-cell} ipython3
tsdframe = nap.TsdFrame(
    t=t,
    d=np.stack([
        np.sin(2 * np.pi * t / 10),
        np.sin(2 * np.pi * t / 5),
    ], axis=1),
    columns=["10s", "5s"],
)

def generate_spiking_unit_from_feature(feature):
    rate = np.clip(feature * 10 + 10, 0, None)
    return nap.Ts(t=np.sort(t[np.random.rand(len(t)) < rate * 0.02]), time_units="s")

tsgroup = nap.TsGroup({
    1: generate_spiking_unit_from_feature(tsdframe[:, 0].d),
    2: generate_spiking_unit_from_feature(tsdframe[:, 1].d),
})

eta = nap.compute_event_triggered_average(tsdframe, tsgroup, binsize=0.02, window=10.0)
eta
```

The result is a `TsdTensor`. Each unit recovers its own driving feature while the other averages to near zero:
```{code-cell} ipython3
fig, axs = plt.subplots(1, len(tsgroup), sharey=True, figsize=(10, 4))
for i, (unit, ax) in enumerate(zip(tsgroup, axs)):
    for feat in range(len(tsdframe.columns)):
        ax.plot(eta.t, eta[:, i, feat], label=tsdframe.columns[feat])
    ax.axvline(0.0, color="red")
    ax.set_title(f"unit {unit}")
    ax.set_xlabel("time from spike (s)")
axs[0].set_ylabel("stimulus [a.u.]")
axs[-1].legend();
```
