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
(for example, firing rate for different epochs of stimulus presentation).

```{contents}
:depth: 3
```

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
xr.set_options(display_expand_attrs=False);
```

## From timestamps or continuous activity
  
Computing tuning curves is done using [`compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves).

When computing from general time-series, mandatory arguments are:
* `data`: a `TsGroup` (or single `Ts`) or `TsdFrame` (or single `Tsd`) containing the neural activity of one or more units.
* `features`: a `Tsd` or `TsdFrame` containing one or more features.

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
We will start by simulating some spiking units modulated by a 1D circular variable:

```{code-cell} ipython3
# Feature
T = 500
dt_feature = 0.02
times_feature = np.arange(0, T, dt_feature)
feature = nap.Tsd(
    t=times_feature, d=np.pi + np.pi * np.cos(2 * np.pi * times_feature / 10)
)

# Spikes
N = 6
max_rate = 20
dt_spikes = 0.002
feature_interp = feature.interpolate(nap.Ts(np.arange(0, T, dt_spikes)))
centers = np.linspace(0, 2 * np.pi, N, endpoint=False)
rates = max_rate * np.exp(
    -10 * (np.sin((feature_interp.d[:, np.newaxis] - centers) / 2)) ** 2
)
tsgroup_1d = nap.TsGroup(
    {
        i + 1: nap.Ts(
            feature_interp.t[np.random.poisson(rates[:, i] * dt_spikes) > 0]
        )
        for i in range(N)
    },
)
```

We now have all the ingredients to compute tuning curves:

```{code-cell} ipython3
tuning_curves_1d = nap.compute_tuning_curves(
    data=tsgroup_1d,
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

[`xarray`](https://docs.xarray.dev/en/latest/index.html) further has [`matplotlib`](https://matplotlib.org/stable/index.html)
support, allowing for easy visualization:

```{code-cell} ipython3
tuning_curves_1d.plot.line(x="feature", add_legend=False)
plt.ylabel("firing rate [Hz]");
```

You can either customize the plot labels yourself using [`matplotlib`](https://matplotlib.org/stable/index.html), 
or you can set them in the tuning curve object:
```{code-cell} ipython3
tuning_curves_1d.name = "firing rate"
tuning_curves_1d.attrs["unit"] = "Hz"
tuning_curves_1d.coords["feature"].attrs["unit"] = "rad"
tuning_curves_1d.plot.line(x="feature", add_legend=False);
```

Internally, the `compute_tuning_curves` calls the [`value_from`](pynapple.Tsd.value_from) method which maps timestamps to their closest values in time from a `Tsd` object.
It is then possible to validate the tuning curves by displaying the timestamps as well as their associated values.

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.subplot(121)
plt.plot(tsgroup_1d[3].value_from(feature), 'o')
plt.plot(feature, label="feature")
plt.ylabel("feature")
plt.xlim(0, 20)
plt.xlabel("Time [s]")
plt.subplot(122)
plt.plot(
    tuning_curves_1d[3].values,
    tuning_curves_1d.coords["feature"],
    label="tuning curve (unit 3)",
)
plt.xlabel("firing rate [Hz]")
plt.legend();
```

It is also possible to just get the spike counts per bins. This can be done by setting the argument `return_counts=True`.
The output is also a `xarray.DataArray` with the same dimensions as the tuning curves.

```{code-cell} ipython3
spike_counts = nap.compute_tuning_curves(
    data=tsgroup_1d,
    features=feature,
    bins=30, 
    range=(0, 2*np.pi),
    feature_names=["feature"],
    return_counts=True
)
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.subplot(131)
plt.plot(tsgroup_1d[3].value_from(feature), 'o')
plt.plot(feature, label="feature")
plt.ylabel("feature")
plt.xlim(0, 20)
plt.xlabel("time [s]")
plt.subplot(132)
plt.plot(tuning_curves_1d[3].values, tuning_curves_1d.coords["feature"])
plt.xlabel("firing rate [Hz]")
plt.subplot(133)
plt.barh(
    spike_counts.coords["feature"],
    width=spike_counts[3].values,
    height=np.mean(np.diff(spike_counts.coords["feature"])),
)
plt.xlabel("spike count")
plt.tight_layout()
```

### 2D tuning curves from spikes
Now, let us simulate some spiking units modulated by a 2D circular variable:

```{code-cell} ipython3
features = nap.TsdFrame(
    t=times_feature,
    d=np.stack(
        [
            np.cos(2 * np.pi * times_feature / 10),
            np.sin(2 * np.pi * times_feature / 10),
        ],
        axis=1,
    ),
    columns=["a", "b"],
)
features_interp = features.interpolate(nap.Ts(np.arange(0, T, dt_spikes)))
alpha = (
    np.arctan2(features_interp["b"].values, features_interp["a"].values)
    / np.pi
)

N = 6
centers_2d = np.linspace(-1, 1, N)
rates_2d = (
    max_rate
    * np.exp(50.0 * np.cos(alpha[:, np.newaxis] - centers_2d))
    / np.exp(50.0)
)
tsgroup_2d = nap.TsGroup(
    {
        i + 1: nap.Ts(
            features_interp.t[
                np.random.poisson(rates_2d[:, i] * dt_spikes) > 0
            ]
        )
        for i in range(N)
    },
)
```

If you pass more than 1 feature, a multi-dimensional tuning curve is computed:
```{code-cell} ipython3
tuning_curves_2d = nap.compute_tuning_curves(
    data=tsgroup_2d, 
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
tuning_curves_2d.name="firing rate"
tuning_curves_2d.attrs["unit"]="Hz"
tuning_curves_2d.plot(col="unit", col_wrap=3);
```

Verifying the accuracy of the tuning curves can once more be done by displaying the spikes aligned 
to the features with the function `value_from` which assign to each spikes the corresponding features value for unit 0.

```{code-cell} ipython3
ts_to_features = tsgroup_2d[1].value_from(features)
ts_to_features
```

`tsgroup[0]` which is a `Ts` object has been transformed to a `TsdFrame` object with each timestamps (spike times) being associated with a features value.

```{code-cell} ipython3
:tags: [hide-input]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
ax1.plot(features["b"], features["a"], label="features")
ax1.plot(
    ts_to_features["b"],
    ts_to_features["a"],
    "o",
    color="red",
    markersize=4,
    label="spikes",
)
ax1.set_xlabel("b")
ax1.set_ylabel("a")
[ax1.axvline(b, linewidth=0.5, color="grey") for b in np.linspace(-1, 1, 6)]
[ax1.axhline(b, linewidth=0.5, color="grey") for b in np.linspace(-1, 1, 6)]
extents = (
    np.min(features["a"]),
    np.max(features["a"]),
    np.min(features["b"]),
    np.max(features["b"]),
)
tuning_curves_2d[0].plot(ax=ax2)
ax2.set_ylabel("")
plt.tight_layout()
```

### 1D tuning curves from continuous activity

We do not always have spikes. 
Sometimes we are analysing continuous firing rates or calcium intensities.
As an example, we will simulate noisy continuous activity for some units modulated by the 1D variable:
```{code-cell} ipython3
noise_level = 2.0
traces = np.exp(-10 * (np.sin((feature.d[:, np.newaxis] - centers) / 2)) ** 2)
traces = traces * (1 + noise_level * np.random.randn(*traces.shape))
tsdframe_1d = nap.TsdFrame(t=times_feature, d=traces)
```

The same function can take a `Tsd` or `TsdFrame` as data and compute tuning curves for
continuous data:
```{code-cell} ipython3
tuning_curves_1d_continuous = nap.compute_tuning_curves(
    data=tsdframe_1d,
    features=feature,
    bins=120,
    range=(0, 2*np.pi),
    feature_names=["feature"]
)
tuning_curves_1d_continuous
```

```{code-cell} ipython3
tuning_curves_1d_continuous.name="ΔF/F"
tuning_curves_1d_continuous.attrs["unit"]="a.u."
tuning_curves_1d_continuous.plot.line(x="feature", add_legend=False);
```

### 2D tuning curves from continuous activity

This also works with more than one feature.
Let us first simulate noisy continuous activity for some units modulated by the 2D variable:
```{code-cell} ipython3
alpha = np.arctan2(features["b"].values, features["a"].values) / np.pi
traces = np.exp(50.0 * np.cos(alpha[:, np.newaxis] - centers_2d)) / np.exp(50.0)
traces = traces * (1 + noise_level * np.random.randn(*traces.shape))
tsdframe_2d = nap.TsdFrame(t=times_feature, d=traces, columns=range(1, N+1, 1))
```

The same function again handles computing the tuning curves:

```{code-cell} ipython3
tuning_curves_2d_continuous = nap.compute_tuning_curves(
    data=tsdframe_2d,
    features=features,
    bins=5,
    feature_names=["a", "b"]
    )
tuning_curves_2d_continuous
```

```{code-cell} ipython3
tuning_curves_2d_continuous.name="ΔF/F"
tuning_curves_2d_continuous.attrs["unit"]="a.u."
tuning_curves_2d_continuous.plot(col="unit", col_wrap=3);
```

## From epochs

When computing from epochs, you should store them in a dictionary:

```{code-cell} ipython3
epochs_dict =  {
    "A": nap.IntervalSet(start=[0, 20, 40, 60, 80], end=[10, 29, 49, 69, 89]),
    "B":nap.IntervalSet(start=[10, 30, 40, 60, 90], end=[19, 39, 59, 79, 99])
}
```
You can then compute the tuning curves using [`nap.compute_response_per_epoch`](pynapple.process.tuning_curves.compute_response_per_epoch).
You can pass either a `TsGroup` for spikes, or a `TsdFrame` for rates/calcium activity.

The output is an `xarray.DataArray` with labeled dimensions:

```{code-cell} ipython3
epochs_tuning_curves = nap.compute_response_per_epoch(tsgroup_2d, epochs_dict)
epochs_tuning_curves
```

We can visualize using barplots:

```{code-cell} ipython3
fig, axs = plt.subplots(
    1, N, constrained_layout=True, sharey=True, figsize=(8, 3)
)
for unit, ax in zip(epochs_tuning_curves.coords["unit"], axs):
    ax.bar(
        epochs_tuning_curves.coords["epochs"],
        epochs_tuning_curves.sel(unit=unit),
    )
    ax.set_title(f"unit {unit.item()}")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("firing rate [hz]");
```

# Error bars
Often, you will want error bars on your tuning curves, to be able to quantify uncertainty.
Pynapple does not provide explicit functions for this, but in this section
we will show how you can easily compute error bars yourself, using the functions we introduced above.

## From timestamps or continuous activity
If you are computing tuning curves against features, you can split your session into `n_splits`,
compute a tuning curve per split, and then compute statistics over those.

We will start by creating splits:
```{code-cell} ipython3
n_splits = 4
full_session = feature.time_support
split_length = full_session.tot_length() / n_splits
splits = full_session.split(split_length)
splits
```

Then, we can compute the tuning curves like before, by looping over the splits:
```{code-cell} ipython3
tuning_curves_per_split = [
        nap.compute_tuning_curves(
            tsgroup_1d,
            epochs=split,
            features=feature,
            bins=120,
            range=(0, 2 * np.pi),
            feature_names=["feature"],
        )
        for split in splits
]
```

To make things easier down the line, we advise combining these into one big 
`xarray.DataArray` using [`xarray.concat`](https://docs.xarray.dev/en/stable/generated/xarray.concat.html) 
, adding a dimension for the splits:

```{code-cell} ipython3
tuning_curves_per_split = xr.concat(tuning_curves_per_split, dim="split")
tuning_curves_per_split
```

Computing the mean and standard deviation can then be done easily using:
```{code-cell} ipython3
means = tuning_curves_per_split.mean(dim="split")
stds = tuning_curves_per_split.std(dim="split")
```

Visualizing also becomes more simple:
```{code-cell} ipython3
tuning_curves_per_split.name = "firing rate"
tuning_curves_per_split.attrs["unit"] = "Hz"
tuning_curves_per_split.coords["feature"].attrs["unit"] = "rad"
lines = means.plot.line(x="feature", add_legend=False)

for line, unit in zip(lines, means.coords["unit"]):
    mean = means.sel(unit=unit)
    std = stds.sel(unit=unit)
    plt.fill_between(
        means["feature"],
        mean - std,
        mean + std,
        color=line.get_color(),
        alpha=0.2,
    )
```

To make things easier in the future, here is a function that does all of this:
```{code-cell} ipython3
def compute_tuning_curves_with_error_bars(
    data, features, bins, range, feature_names, n_splits
):
    # Get splits
    full_session = features.time_support
    split_length = full_session.tot_length() / n_splits
    splits = full_session.split(split_length)

    # Compute tuning curves per split
    tuning_curves_per_split = [
        nap.compute_tuning_curves(
            data,
            features=features,
            epochs=split,
            bins=bins,
            range=range,
            feature_names=feature_names,
        )
        for split in splits
    ]
    tuning_curves_per_split = xr.concat(tuning_curves_per_split, dim="split")

    # Return mean and standard deviation
    return (
        tuning_curves_per_split.mean(dim="split"), 
        tuning_curves_per_split.std(dim="split")
    )
```
Feel free to extend it to your needs!

## From epochs
If you want error bars for epochs, the typical use-case will be that you have multiple presentations of a stimulus,
and you want the mean response over those:
```{code-cell} ipython3
epochs_dict
```

So, we can use [`nap.compute_response_per_epoch`](pynapple.process.tuning_curves.compute_response_per_epoch) in a loop to compute that:
```{code-cell} ipython3
epochs_tuning_curves_per_presentation = [
    nap.compute_response_per_epoch(
        tsgroup_2d, {"A": stimulus_A, "B": stimulus_B}
    )
    for stimulus_A, stimulus_B in zip(epochs_dict["A"], epochs_dict["B"])
]
```

To make things easier down the line, we advise combining these into one big 
`xarray.DataArray` using [`xarray.concat`](https://docs.xarray.dev/en/stable/generated/xarray.concat.html) 
, adding a dimension for the presentations:

```{code-cell} ipython3
epochs_tuning_curves_per_presentation = xr.concat(
    epochs_tuning_curves_per_presentation, dim="presentation"
)
epochs_tuning_curves_per_presentation
```

We can then visualize again, but now with error bars:
```{code-cell} ipython3
means = epochs_tuning_curves_per_presentation.mean(dim="presentation")
stds = epochs_tuning_curves_per_presentation.std(dim="presentation")

fig, axs = plt.subplots(
    1, N, constrained_layout=True, sharey=True, figsize=(8, 3)
)
for unit, ax in zip(epochs_tuning_curves.coords["unit"], axs):
    ax.bar(
        means.coords["epochs"], means.sel(unit=unit), yerr=stds.sel(unit=unit)
    )
    ax.set_title(f"unit {unit.item()}")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("firing rate [Hz]");
```

To make things easier in the future, here is a function that does all of this:
```{code-cell} ipython3
def compute_response_per_epoch_with_error_bars(data, epochs_dict):
    epochs_dict_per_presentation = [
        dict(zip(epochs_dict.keys(), presentations))
        for presentations in zip(*epochs_dict.values())
    ]
    epochs_tuning_curves_per_presentation = [
        nap.compute_response_per_epoch(data, presentation_epochs_dict)
        for presentation_epochs_dict in epochs_dict_per_presentation
    ]
    epochs_tuning_curves_per_presentation = xr.concat(
        epochs_tuning_curves_per_presentation, dim="presentation"
    )
    return (
        epochs_tuning_curves_per_presentation.mean(dim="presentation"),
        epochs_tuning_curves_per_presentation.std(dim="presentation"),
    )
```
Feel free to extend it to your needs!

# Mutual information
Given a set of tuning curves, you can use [`compute_mutual_information`](pynapple.process.tuning_curves.compute_mutual_information) to compute the mutual information between the activity of the neurons and the features, no matter what dimension.
See the [Skaggs et al. (1992)](https://proceedings.neurips.cc/paper/1992/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) paper for more information on what mutual information computes.

```{code-cell} ipython3
MI = nap.compute_mutual_information(tuning_curves_1d)
MI
```

```{code-cell} ipython3
MI = nap.compute_mutual_information(tuning_curves_2d)
MI
```
Take a look at the tutorial on [head direction cells](../examples/tutorial_HD_dataset.md) for a realistic example.


