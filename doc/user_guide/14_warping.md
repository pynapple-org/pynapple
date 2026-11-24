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

Time warping
============

The warping module contains functions for warping times.
If the input is a TsGroup containing the activity of a population of neurons:

- [`nap.warp_tensor`](pynapple.process.warping.warp_tensor) returns a tensor of shape (number of neurons, number of trials, number of time bins) with linearly warped time bins.

The function works for all time series object (`Tsd`, `TsdFrame` and `TsdTensor`)
and timestamp objects (`Ts` and `TsGroup`). See examples below.


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

## `nap.warp_tensor`
--------------------

The [`nap.warp_tensor`](pynapple.process.warping.warp_tensor) function stretches linearly
for each interval depending on the parameter `num_bins`.
In other words, the number of bins between the start and end of an epoch is always `num_bins`,
but the duration of each bin can vary across epochs.

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

It is also possible to warp a time series to create a trial-based tensor.
Under the hood, the time series is either bin-averaged or interpolated depending on the number of bins.

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
