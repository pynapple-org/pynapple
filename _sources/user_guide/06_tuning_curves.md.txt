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

Pynapple can compute 1-dimensional tuning curves 
(for example firing rate as a function of angular direction) 
and 2-dimensional tuning curves (for example firing rate as a function 
of position). It can also compute average firing rate for different 
epochs (for example firing rate for different epochs of stimulus presentation).

:::{important}
If you are using calcium imaging data with the activity of the cell as a continuous transient, the function to call ends with `_continuous` for continuous time series (e.g. [`compute_1d_tuning_curves_continuous`](pynapple.process.tuning_curves.compute_1d_tuning_curves_continuous)).
:::


```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
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

## From epochs


The epochs should be stored in a dictionnary: 
```{code-cell} ipython3
dict_ep =  {
                "stim0": nap.IntervalSet(start=0, end=20),
                "stim1":nap.IntervalSet(start=30, end=70)
            }
```

[`nap.compute_discrete_tuning_curves`](pynapple.process.tuning_curves.compute_discrete_tuning_curves) takes a `TsGroup` for spiking activity and a dictionary of epochs. 
The output is a pandas DataFrame where each column is a unit in the `TsGroup` and each row is one `IntervalSet` type.
The value is the mean firing rate of the neuron during this set of intervals.

```{code-cell} ipython3
mean_fr = nap.compute_discrete_tuning_curves(tsgroup, dict_ep)

pprint(mean_fr)
```


## From timestamps activity
  
### 1-dimension tuning curves


```{code-cell} ipython3
:tags: [hide-cell]

from scipy.ndimage import gaussian_filter1d

# Fake Tuning curves
N = 6 # Number of neurons
bins = np.linspace(0, 2*np.pi, 61)
x = np.linspace(-np.pi, np.pi, len(bins)-1)
tmp = np.roll(np.exp(-(1.5*x)**2), (len(bins)-1)//2)
generative_tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T

# Feature
T = 50000
dt = 0.002
timestep = np.arange(0, T)*dt
feature = nap.Tsd(
    t=timestep,
    d=gaussian_filter1d(np.cumsum(np.random.randn(T)*0.5), 20)%(2*np.pi)
    )
index = np.digitize(feature, bins)-1

# Spiking activity
count = np.random.poisson(generative_tc[index])>0
tsgroup = nap.TsGroup(
    {i:nap.Ts(timestep[count[:,i]]) for i in range(N)},
    time_support = nap.IntervalSet(0, 100)
    )
```

Mandatory arguments are `TsGroup`, `Tsd` (or `TsdFrame` with 1 column only) 
and `nb_bins` for number of bins of the tuning curves.

If an `IntervalSet` is passed with `ep`, everything is restricted to `ep`
otherwise the time support of the feature is used.

The min and max of the tuning curve is by default the min and max of the feature. This can be tweaked with the argument `minmax`.

The output is a pandas DataFrame. Each column is a unit from `TsGroup` argument. The index of the DataFrame carries the center of the bin in feature space.

```{code-cell} ipython3
tuning_curve = nap.compute_1d_tuning_curves(
    group=tsgroup,
    feature=feature,
    nb_bins=120, 
    minmax=(0, 2*np.pi)
    )

print(tuning_curve)
```

```{code-cell} ipython3
plt.figure()
plt.plot(tuning_curve)
plt.xlabel("Feature space")
plt.ylabel("Firing rate (Hz)")
plt.show()
```

Internally, the function is calling the method [`value_from`](pynapple.Tsd.value_from) which maps timestamps to their closest values in time from a `Tsd` object. It is then possible to validate the tuning curves by displaying the timestamps as well as their associated values. 

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.subplot(121)
plt.plot(tsgroup[3].value_from(feature), 'o')
plt.plot(feature, label="feature")
plt.ylabel("Feature")
plt.xlim(0, 2)
plt.xlabel("Time (s)")
plt.subplot(122)
plt.plot(tuning_curve[3].values, tuning_curve[3].index.values, label="Tuning curve (unit=3)")
plt.xlabel("Firing rate (Hz)")
plt.legend()
plt.show()
```

### 2-dimensional tuning curves

```{code-cell} ipython3
:tags: [hide-cell]
dt = 0.01
T = 10
epoch = nap.IntervalSet(start=0, end=T, time_units="s")
features = np.vstack((np.cos(np.arange(0, T, dt)), np.sin(np.arange(0, T, dt)))).T
features = nap.TsdFrame(
    t=np.arange(0, T, dt),
    d=features,
    time_units="s",
    time_support=epoch,
    columns=["a", "b"],
)
tsgroup = nap.TsGroup({
    0: nap.Ts(t=np.sort(np.random.uniform(0, T, 10))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, T, 15))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, T, 20))),
}, time_support=epoch)
```

The `group` argument must be a `TsGroup` object.
The `features` argument must be a 2-columns `TsdFrame` object.
`nb_bins` can be an int or a tuple of 2 ints.
 
```{code-cell} ipython3
tcurves2d, binsxy = nap.compute_2d_tuning_curves(
    group=tsgroup, 
    features=features, 
    nb_bins=(5,5),
    minmax=(-1, 1, -1, 1)
)
pprint(tcurves2d)
```

`tcurves2d` is a dictionnary with each key a unit in `TsGroup`. `binsxy` is a numpy array representing the centers of the bins and is useful for plotting tuning curves. Bins that have never been visited by the feature have been assigned a NaN value.

Checking the accuracy of the tuning curves can be bone by displaying the spikes aligned to the features with the function `value_from` which assign to each spikes the corresponding features value for unit 0.

```{code-cell} ipython3
ts_to_features = tsgroup[0].value_from(features)
print(ts_to_features)
```
`tsgroup[0]` which is a `Ts` object has been transformed to a `TsdFrame` object with each timestamps (spike times) being associated with a features value.

```{code-cell} ipython3
:tags: [hide-input]


plt.figure()
plt.subplot(121)
plt.plot(features["b"], features["a"], label="features")
plt.plot(ts_to_features["b"], ts_to_features["a"], "o", color="red", markersize=4, label="spikes")
plt.xlabel("feature b")
plt.ylabel("feature a")
[plt.axvline(b, linewidth=0.5, color='grey') for b in np.linspace(-1, 1, 6)]
[plt.axhline(b, linewidth=0.5, color='grey') for b in np.linspace(-1, 1, 6)]
plt.subplot(122)
extents = (
    np.min(features["a"]),
    np.max(features["a"]),
    np.min(features["b"]),
    np.max(features["b"]),
)
plt.imshow(tcurves2d[0], 
    origin="lower", extent=extents, cmap="viridis",
    aspect='auto'
    )
plt.title("Tuning curve unit 0")
plt.xlabel("feature b")
plt.ylabel("feature a")
plt.grid(False)
plt.colorbar()
plt.tight_layout()
plt.show()
```


## From continuous activity

Tuning curves compute with the following functions are usually made with 
data from calcium imaging activities.

### 1-dimensional tuning curves

```{code-cell} ipython3
:tags: [hide-cell]

from scipy.ndimage import gaussian_filter1d

# Fake Tuning curves
N = 3 # Number of neurons
bins = np.linspace(0, 2*np.pi, 61)
x = np.linspace(-np.pi, np.pi, len(bins)-1)
tmp = np.roll(np.exp(-(1.5*x)**2), (len(bins)-1)//2)
generative_tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T

# Feature
T = 50000
dt = 0.002
timestep = np.arange(0, T)*dt
feature = nap.Tsd(
    t=timestep,
    d=gaussian_filter1d(np.cumsum(np.random.randn(T)*0.5), 20)%(2*np.pi)
    )
index = np.digitize(feature, bins)-1
tmp = generative_tc[index]
tmp = tmp + np.random.randn(*tmp.shape)*1
# Calcium activity
tsdframe = nap.TsdFrame(
    t=timestep,
    d=tmp
    )
```

Arguments are `TsdFrame` (for example continuous calcium data), `Tsd` or `TsdFrame` for the 1-d feature and `nb_bins` for the number of bins.

```{code-cell} ipython3

tuning_curves = nap.compute_1d_tuning_curves_continuous(
    tsdframe=tsdframe,
    feature=feature,
    nb_bins=120,
    minmax=(0, 2*np.pi)
    )

print(tuning_curves)
```

```{code-cell} ipython3
plt.figure()
plt.plot(tuning_curves)
plt.xlabel("Feature space")
plt.ylabel("Mean activity")
plt.show()
```

### 2-dimensional tuning curves


```{code-cell} ipython3
:tags: [hide-cell]

dt = 0.01
T = 10
epoch = nap.IntervalSet(start=0, end=T, time_units="s")
features = np.vstack((np.cos(np.arange(0, T, dt)), np.sin(np.arange(0, T, dt)))).T
features = nap.TsdFrame(
    t=np.arange(0, T, dt),
    d=features,
    time_units="s",
    time_support=epoch,
    columns=["a", "b"],
)


# Calcium activity
tsdframe = nap.TsdFrame(
    t=timestep,
    d=np.random.randn(len(timestep), 2)
    )
```

Arguments are `TsdFrame` (for example continuous calcium data), `Tsd` or `TsdFrame` for the 1-d feature and `nb_bins` for the number of bins.

```{code-cell} ipython3

tuning_curves, xy = nap.compute_2d_tuning_curves_continuous(
    tsdframe=tsdframe,
    features=features,
    nb_bins=5,    
    )

print(tuning_curves)
```

```{code-cell} ipython3
plt.figure()
plt.subplot(121)
plt.plot(features["b"], features["a"], label="features")
plt.xlabel("feature b")
plt.ylabel("feature a")
[plt.axvline(b, linewidth=0.5, color='grey') for b in np.linspace(-1, 1, 6)]
[plt.axhline(b, linewidth=0.5, color='grey') for b in np.linspace(-1, 1, 6)]
plt.subplot(122)
extents = (
    np.min(features["a"]),
    np.max(features["a"]),
    np.min(features["b"]),
    np.max(features["b"]),
)
plt.imshow(tuning_curves[0], 
    origin="lower", extent=extents, cmap="viridis",
    aspect='auto'
    )
plt.title("Tuning curve unit 0")
plt.xlabel("feature b")
plt.ylabel("feature a")
plt.grid(False)
plt.colorbar()
plt.tight_layout()
plt.show()
```



