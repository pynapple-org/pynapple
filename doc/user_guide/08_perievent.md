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

# Perievent

The perievent module allows to re-center time series and timestamps data around a particular event as well as computing events (spikes) trigger average.

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

## Peri-Event Time Histogram (PETH)


```{code-cell} ipython3
stim = nap.Tsd(
    t=np.sort(np.random.uniform(0, 1000, 50)), 
    d=np.random.rand(50), time_units="s"
)
ts1 = nap.Ts(t=np.sort(np.random.uniform(0, 1000, 2000)), time_units="s")
```

The function [`compute_perievent`](pynapple.process.perievent.compute_perievent) align timestamps to a particular set of timestamps.

```{code-cell} ipython3
peth = nap.compute_perievent(
  timestamps=ts1, 
  tref=stim, 
  minmax=(-0.1, 0.2), 
  time_unit="s")

print(peth)
```

The returned object is a `TsGroup`. The column `ref_times` is a 
metadata column that indicates the center timestamps.

### Raster plot

It is then easy to create a raster plot around the times of the 
stimulation event by calling the [`to_tsd`](pynapple.TsGroup.to_tsd) function of pynapple to "flatten" the TsGroup `peth`.

```{code-cell} ipython3
plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.plot(np.mean(peth.count(0.01), 1) / 0.01, linewidth=3, color="red")
plt.xlim(-0.1, 0.2)
plt.ylabel("Rate (spikes/sec)")
plt.axvline(0.0)
plt.subplot(212)
plt.plot(peth.to_tsd(), "|", markersize=20, color="red", mew=4)
plt.xlabel("Time from stim (s)")
plt.ylabel("Stimulus")
plt.xlim(-0.1, 0.2)
plt.axvline(0.0)
```

The same function can be applied to a group of neurons. 
In this case, it returns a dict of `TsGroup`.

## Event trigger average

The function [`compute_event_trigger_average`](pynapple.process.perievent.compute_event_trigger_average) compute the average feature around a particular event time.

```{code-cell}
:tags: [hide-cell]

group = {
    0: nap.Ts(t=np.sort(np.random.uniform(0, 100, 10))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30))),
}

tsgroup = nap.TsGroup(group)
```
```{code-cell}
eta = nap.compute_event_trigger_average(
  group=tsgroup, 
  feature=stim, 
  binsize=0.1, 
  windowsize=(-1, 1))

print(eta)
```


## Peri-Event continuous time series

The function [`nap.compute_perievent_continuous`](pynapple.process.perievent.compute_perievent_continuous) align a time series of any dimensions around events.

```{code-cell}
:tags: [hide-cell]

features = nap.TsdFrame(t=np.arange(0, 100), d=np.random.randn(100,6))
events = nap.Ts(t=np.sort(np.random.uniform(0, 100, 5)))
```

```{code-cell}

perievent = nap.compute_perievent_continuous(
  timeseries=features, 
  tref=events, 
  minmax=(-1, 1))

print(perievent)
```

The object perievent is now of shape (number of bins, (dimensions of input), number of events):


```{code-cell}
print(perievent.shape)
```







