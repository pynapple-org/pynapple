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

# Tuning curves

Pynapple can compute 1 dimension tuning curves 
(for example firing rate as a function of angular direction) 
and 2 dimension tuning curves ( for example firing rate as a function 
of position). It can also compute average firing rate for different 
epochs (for example firing rate for different epochs of stimulus presentation).

:::{important}
If you are using calcium imaging data with the activity of the cell as a continuous transient, the function to call ends with `_continuous` for continuous time series (e.g. `compute_1d_tuning_curves_continuous`).
:::


```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

```{code-cell}
:tags: [hide-cell]

group = {
    0: nap.Ts(t=np.sort(np.random.uniform(0, 100, 10))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30))),
}

tsgroup = nap.TsGroup(group)
```

## from epochs


The epochs should be stored in a dictionnary : 
```{code-cell} ipython3
dict_ep =  {
                "stim0": nap.IntervalSet(start=0, end=20),
                "stim1":nap.IntervalSet(start=30, end=70)
            }
```

`nap.compute_discrete_tuning_curves` takes a `TsGroup` for spiking activity and a dictionary of epochs. 
The output is a pandas DataFrame where each column is a unit in the `TsGroup` and each row is one `IntervalSet` type.
The value is the mean firing rate of the neuron during this set of intervals.

```{code-cell} ipython3
mean_fr = nap.compute_discrete_tuning_curves(tsgroup, dict_ep)

print(mean_fr)
```


## from timestamps activity
  
### 1-dimension tuning curves

### 2-dimension tuning curves

First we will create the 2D features:
```{code-cell} ipython3
:tags: [hide-cell]

dt = 0.1
epoch = nap.IntervalSet(start=0, end=1000, time_units="s")
features = np.vstack((np.cos(np.arange(0, 1000, dt)), np.sin(np.arange(0, 1000, dt)))).T
# features += np.random.randn(features.shape[0], features.shape[1])*0.05
features = nap.TsdFrame(
    t=np.arange(0, 1000, dt),
    d=features,
    time_units="s",
    time_support=epoch,
    columns=["a", "b"],
)

print(features)
```
The `group` argument must be a `TsGroup` object.
The `features` argument must be a 2-columns `TsdFrame` object.
`nb_bins` can be an int or a tuple of 2 ints.
 
```{code-cell} ipython3
tcurves2d, binsxy = nap.compute_2d_tuning_curves(
    group=tsgroup, features=features, nb_bins=10
)
print(tcurves2d)
```


To check the accuracy of the tuning curves, we will display the spikes aligned to the features with the function `value_from` which assign to each spikes the corresponding feature value for neuron 0.
```{code-cell} ipython3
ts_to_features = ts_group[1].value_from(features)

plt.figure()
plt.plot(ts_to_features["a"], ts_to_features["b"], "o", color="red", markersize=4)
extents = (
    np.min(features["b"]),
    np.max(features["b"]),
    np.min(features["a"]),
    np.max(features["a"]),
)
plt.imshow(tcurves2d[1].T, origin="lower", extent=extents, cmap="viridis")
plt.title("Tuning curve unit 0 2d")
plt.xlabel("feature a")
plt.ylabel("feature b")
plt.grid(False)
plt.show()
```


## from continuous activity

### 1-dimension tuning curves

### 2-dimension tuning curves





