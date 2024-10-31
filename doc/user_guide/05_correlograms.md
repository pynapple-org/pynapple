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

# Correlograms of discrete events

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

Let's generate some data. Here we have two neurons recorded together. We can group them in a TsGroup.

```{code-cell} ipython3
ts1 = nap.Ts(t=np.sort(np.random.uniform(0, 1000, 2000)), time_units="s")
ts2 = nap.Ts(t=np.sort(np.random.uniform(0, 1000, 1000)), time_units="s")
epoch = nap.IntervalSet(start=0, end=1000, time_units="s")
ts_group = nap.TsGroup({0: ts1, 1: ts2}, time_support=epoch)
print(ts_group)
```

## Autocorrelograms

We can compute their autocorrelograms meaning the number of spikes of a neuron observed in a time windows centered around its own spikes.
For this we can use the function `compute_autocorrelogram`.
We need to specifiy the `binsize` and `windowsize` to bin the spike train.

```{code-cell} ipython3
autocorrs = nap.compute_autocorrelogram(
    group=ts_group, binsize=100, windowsize=1000, time_units="ms", ep=epoch  # ms
)
print(autocorrs)
```
The variable `autocorrs` is a pandas DataFrame with the center of the bins 
for the index and each column is an autocorrelogram of one unit in the `TsGroup`.

## Cross-correlograms

Cross-correlograms are computed between pairs of neurons.

```{code-cell} ipython3
crosscorrs = nap.compute_crosscorrelogram(
    group=ts_group, binsize=100, windowsize=1000, time_units="ms"  # ms
)
print(crosscorrs)
```

Column name `(0, 1)` is read as cross-correlogram of neuron 0 and 1 with neuron 0 being the reference time.

## Event-correlograms

Event-correlograms count the number of event in the `TsGroup` based on an `event` timestamps object. 

```{code-cell} ipython3
eventcorrs = nap.compute_eventcorrelogram(
    group=ts_group, event = nap.Ts(t=[0, 10, 20]), binsize=0.1, windowsize=1
    )
print(eventcorrs)
```
