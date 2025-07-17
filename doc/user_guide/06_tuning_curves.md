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

With Pynapple you can easily compute n-dimensional tuning curves
(for example, firing rate as a function of 1D angular direction or firing rate as a function of 2D position).
It is also possible to compute average firing rate for different epochs 
(for example firing rate for different epochs of stimulus presentation).

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from pprint import pprint
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
xr.set_options(display_expand_attrs=False)
```

```{code-cell} ipython3
:tags: [hide-cell]
group = {
    0: nap.Ts(t=np.sort(np.random.uniform(0, 100, 10))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30))),
}
tsgroup = nap.TsGroup(group)
```

<!-- #region -->
## From epochs

When computing from epochs, you should store them in a dictionary:
<!-- #endregion -->

```{code-cell} ipython3
dict_ep =  {
                "stim0": nap.IntervalSet(start=0, end=20),
                "stim1":nap.IntervalSet(start=30, end=70)
}
```

[`nap.compute_discrete_tuning_curves`](pynapple.process.tuning_curves.compute_discrete_tuning_curves) takes a `TsGroup` for spiking activity and a dictionary of epochs. 
The output is a pandas DataFrame where each column is a unit in the `TsGroup` and each row is one `IntervalSet`.
The output will be the mean firing rate of the neuron during this set of intervals.

```{code-cell} ipython3
mean_fr = nap.compute_discrete_tuning_curves(tsgroup, dict_ep)
print(mean_fr)
```

## From timestamps
  
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

When computing from general time-series, mandatory arguments are:
* a `TsGroup`, `Tsd`, or `TsdFrame` containing the neural activity of one or more units.
* a `Tsd` or `TsdFrame` containing one or more features.

By default, 10 bins are used for all features, but you can specify the number of bins,
or the bin edges explicitly, using the `bins` argument.

The min and max of the tuning curves are by default the minima and maxima of the features. 
This can be tweaked with the `range` argument.

If an `IntervalSet` is passed with `epochs`, everything is restricted to `epochs`,
otherwise the time support of the features is used.

If you do not want the sampling rate of the features to be estimated from the timestamps,
you can pass it explicitly using the `fs` argument.

You can further also pass a list of strings to label each dimension via `feature_names` 
(by default the columns of the features are used).

The output is an `xarray.DataArray` in which the first dimension represents the units and further dimensions represent the features.
The occupancy and bin edges are stored as attributes.

If you explicitly want a `pd.DataFrame` as output (which is only possible when you have just the one feature), 
you can set `return_pandas=True`. Note that this will not return the occupancy and bin edges.

### 1D tuning curves from spikes

```{code-cell} ipython3
tuning_curves_1d = nap.compute_tuning_curves(
    group=tsgroup,
    features=feature,
    bins=120, 
    range=(0, 2*np.pi),
    feature_names=["feature"]
    )
tuning_curves_1d
```

The `xarray.DataArray` can be treated like a `numpy` array.

It has a shape:
```{code-cell} ipython3
tuning_curves_1d.shape
```
It can be sliced:
```{code-cell} ipython3
tuning_curves_1d[1, 2:8]
```
It can also be indexed using the coordinates:
```{code-cell} ipython3
tuning_curves_1d.sel(unit=1)
```

`xarray` further has `matplotlib` support, allowing for easy visualization:

```{code-cell} ipython3
tuning_curves_1d.plot.line(x="feature", add_legend=False)
plt.ylabel("Firing rate (Hz)")
plt.show()
```

You can either customize the plot labels yourself using `matplotlib`, or you can set them in the tuning curve object:
```{code-cell} ipython3
tuning_curves_1d.name = "Firing rate"
tuning_curves_1d.attrs["unit"] = "Hz"
tuning_curves_1d.coords["feature"].attrs["unit"] = "rad"
tuning_curves_1d.plot.line(x="feature", add_legend=False)
plt.show()
```

Internally, the `compute_tuning_curves` is calling the method [`value_from`](pynapple.Tsd.value_from) which maps timestamps to their closest values in time from a `Tsd` object.
It is then possible to validate the tuning curves by displaying the timestamps as well as their associated values.

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
plt.plot(tuning_curves_1d[3].values, tuning_curves_1d.coords["feature"], label="Tuning curve (unit=3)")
plt.xlabel("Firing rate (Hz)")
plt.legend()
plt.show()
```

### 2D tuning curves from spikes

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

If you pass more than 1 feature, a multi-dimensional tuning curve is computed:
```{code-cell} ipython3
tuning_curves_2d = nap.compute_tuning_curves(
    group=tsgroup, 
    features=features, 
    bins=(5,5),
    range=[(-1, 1), (-1, 1)],
    feature_names=["a", "b"]
)
tuning_curves_2d
```

`tuning_curve_2d` is a again an `xarray.DataArray` but now with three dimensions: 
one for the units of `TsGroup` and 2 for the features, the coordinates contain the centers of the bins.
Bins that have never been visited by the feature have been assigned a NaN value.

Two-dimensional tuning curves can also easily be visualized:

```{code-cell} ipython3
tuning_curves_2d.name="Firing rate"
tuning_curves_2d.attrs["unit"]="Hz"
tuning_curves_2d.plot(col="unit")
plt.show()
```

Verifying the accuracy of the tuning curves can once more be done by displaying the spikes aligned 
to the features with the function `value_from` which assign to each spikes the corresponding features value for unit 0.

```{code-cell} ipython3
ts_to_features = tsgroup[0].value_from(features)
print(ts_to_features)
```

`tsgroup[0]` which is a `Ts` object has been transformed to a `TsdFrame` object with each timestamps (spike times) being associated with a features value.

```{code-cell} ipython3
:tags: [hide-input]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4), sharey=True)
ax1.plot(features["b"], features["a"], label="features")
ax1.plot(ts_to_features["b"], ts_to_features["a"], "o", color="red", markersize=4, label="spikes")
ax1.set_xlabel("b")
ax1.set_ylabel("a")
[ax1.axvline(b, linewidth=0.5, color='grey') for b in np.linspace(-1, 1, 6)]
[ax1.axhline(b, linewidth=0.5, color='grey') for b in np.linspace(-1, 1, 6)]
extents = (
    np.min(features["a"]),
    np.max(features["a"]),
    np.min(features["b"]),
    np.max(features["b"]),
)
tuning_curves_2d[0].plot(ax=ax2)
ax2.set_ylabel("")
plt.tight_layout()
plt.show()
```

### 1D tuning curves from continuous activity

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

We do not always have spikes. Sometimes we are analysing continuous firing rates or calcium intensities.
In that case, we can simply pass a `Tsd` or `TsdFrame` as group:

```{code-cell} ipython3
tuning_curves_1d = nap.compute_tuning_curves(
    group=tsdframe,
    features=feature,
    bins=120,
    range=(0, 2*np.pi),
    feature_names=["feature"]
    )
tuning_curves_1d
```

```{code-cell} ipython3
tuning_curves_1d.name="ΔF/F"
tuning_curves_1d.attrs["unit"]="a.u."
tuning_curves_1d.plot.line(x="feature", add_legend=False)
plt.show()
```

### 2D tuning curves from continuous activity

This also works with more than one feature:

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

```{code-cell} ipython3
tuning_curves_2d = nap.compute_tuning_curves(
    group=tsdframe,
    features=features,
    bins=5,
    feature_names=["a", "b"]
    )
tuning_curves_2d
```

```{code-cell} ipython3
tuning_curves_2d.name="ΔF/F"
tuning_curves_2d.attrs["unit"]="a.u."
tuning_curves_2d.plot(col="unit")
plt.show()
```
