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

Trial-based tensors & time warping
=================================

The warping module contains functions for constructing trial-based tensors.
If the input is a TsGroup containing the activity of a population of neurons:

- [`nap.build_tensor`](pynapple.process.warping.build_tensor) returns a tensor of shape (number of neurons, number of trials, number of time bins) with padded values if unequal trial intervals.
- [`nap.warp_tensor`](pynapple.process.warping.warp_tensor) returns a tensor of shape (number of neurons, number of trials, number of time bins) with linearly warped time bins.

Both functions works for all time series object (`Tsd`, `TsdFrame` and `TsdTensor`) and timestamp objects (`Ts` and `TsGroup`). See examples below.

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

`nap.build_tensor`
------------------

The function [`nap.build_tensor`](pynapple.process.warping.build_tensor) slices a time series object or timestamps object for each interval of an `IntervalSet` object and returns 
a numpy array. The intervals can be of unequal durations. 

```{code-cell} ipython3
tsgroup = nap.TsGroup({0:nap.Ts(t=np.arange(0, 100)+0.5)})
ep = nap.IntervalSet(
    start=np.arange(20, 80, 20), end=np.arange(20, 80, 20) + np.arange(2, 8, 2),
    metadata = {'trials':['trial1', 'trial2', 'trial3']} 
    )
print(ep)
```

To build a trial-based count tensor from a `TsGroup` object with 1 second bins: 

```{code-cell} ipython3
tensor = nap.build_tensor(tsgroup, ep, bin_size=1, padding_value=np.nan)

print(tensor)
```

We can check the operation by plotting the spike times and the edges of the bins for each epoch.

 ```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
plt.subplot(211)
plt.plot(tsgroup.to_tsd().get(18, 70), '|', color='orange', label = "Spike times neuron 0")
[plt.axvspan(s, e, alpha=0.5) for s, e in ep.values]
plt.legend()
plt.xlabel("Time (s)")
for s, e in ep.values:
    [plt.axvline(t) for t in np.arange(s, e+1)]
plt.subplot(212)
im = plt.pcolormesh(tensor[0], edgecolors='k', linewidth=2, cmap='RdYlBu', vmin=0, vmax=1)
plt.xlabel("Bin time (s)")
plt.ylabel("Trials")
plt.title("Tensor neuron 0")
plt.tight_layout()
plt.colorbar(im)
plt.yticks([0,1,2])

plt.show()

```


:::{note}
This function is also available at the object level.
```
>>> tensor = tsgroup.trial_count(ep, bin_size=1, padding_value=np.nan)
```
:::




It is also possible to create a trial-based tensor from a time series. In this case the argument `bin_size` is not used.

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(100), d=np.arange(200).reshape(2,100).T)
tensor = nap.build_tensor(tsdframe, ep)

print(tensor)
```

:::{note}
This function is also available at the object level.
```
>>> tensor = tsdframe.to_trial_tensor(ep, padding_value=np.nan)
```
:::


 ```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
plt.subplot(211)
tmp = tsdframe.get(18, 70)[:,0]
plt.plot(tmp, '-', label="tsdframe[:,0]", color='grey')
cmap = plt.get_cmap('RdYlBu')
for i in range(len(tmp)):
    plt.plot(tmp.t[i], tmp.d[i], 'o', color=cmap(i/len(tmp)))
[plt.axvspan(s, e, alpha=0.5) for s, e in ep.values]
plt.legend()
plt.xlabel("Time (s)")
[plt.axvline(t) for t in ep.values.flatten()]
plt.subplot(212)
im = plt.pcolormesh(tensor[0], edgecolors='k', linewidth=2, cmap='RdYlBu', vmin=np.min(tmp), vmax=np.max(tmp))
plt.xlabel("Bin time (s)")
plt.ylabel("Trials")
plt.title("Tensor neuron 0")
plt.tight_layout()
plt.colorbar(im)
plt.yticks([0,1,2])

plt.show()

```


`nap.warp_tensor`
-----------------

The function [`nap.warp_tensor`](pynapple.process.warping.warp_tensor) is similar to `build_tensor`, but time is stretched linearly for each interval depending on
the parameter `num_bins`. In other words, the number of bins between the start and end of an epoch is always `num_bins`, but
the duration of each bin can vary across epochs.

```{code-cell} ipython3
tensor = nap.warp_tensor(tsgroup, ep, num_bins=10)

print(tensor)
```

 ```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
plt.subplot(211)
plt.plot(tsgroup.to_tsd().get(38, 70), '|', color='orange', label = "Spike times neuron 0")
[plt.axvspan(s, e, alpha=0.5) for s, e in ep.values[1:]]
plt.legend()
plt.xlabel("Time (s)")
for s, e in ep.values[1:]:
    [plt.axvline(t) for t in np.linspace(s, e, 11)]
plt.subplot(212)
im = plt.pcolormesh(tensor[0][1:], edgecolors='k', linewidth=2, cmap='RdYlBu', vmin=0, vmax=1)
plt.xlabel("Bin time (s)")
plt.ylabel("Trials")
plt.title("Tensor neuron 0")
plt.tight_layout()
plt.colorbar(im)
plt.yticks([0,1])

plt.show()

```

It is also possible to warp a time series to create a trial-based tensor. Under the hood, the time series is either 
bin-averaged or interpolated depending on the number of bins.

```{code-cell} ipython3
tensor = nap.warp_tensor(tsdframe, ep, num_bins=10)

print(tensor)
```

 ```{code-cell} ipython3
:tags: [hide-input]

plt.figure()
plt.subplot(211)
tmp = tsdframe.get(18, 70)[:,0]
plt.plot(tmp, '-', label="tsdframe[:,0]", color='grey')
cmap = plt.get_cmap('RdYlBu')
for i in range(len(tmp)):
    plt.plot(tmp.t[i], tmp.d[i], 'o', color=cmap(i/len(tmp)))
[plt.axvspan(s, e, alpha=0.5) for s, e in ep.values]
plt.legend()
plt.xlabel("Time (s)")
[plt.axvline(t) for t in ep.values.flatten()]
plt.subplot(212)
im = plt.pcolormesh(tensor[0], edgecolors='k', linewidth=2, cmap='RdYlBu', vmin=np.min(tmp), vmax=np.max(tmp))
plt.xlabel("Bin time (s)")
plt.ylabel("Trials")
plt.title("Tensor neuron 0")
plt.tight_layout()
plt.colorbar(im)
plt.yticks([0,1,2])

plt.show()

```