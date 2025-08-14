---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Calcium Imaging
============

Working with calcium data.

As example dataset, we will be working with a recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging).
The area recorded for this experiment is the postsubiculum - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in a specific direction.

The NWB file for the example is hosted on [OSF](https://osf.io/sbnaw). We show below how to stream it.

```{code-cell} ipython3
:tags: [hide-output]
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests 
import xarray as xr

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
xr.set_options(display_expand_attrs=False)
```

***
Downloading the data
------------------
First things first: let's find our file.

```{code-cell} ipython3
path = "A0670-221213.nwb"
if path not in os.listdir("."):
  r = requests.get(f"https://osf.io/sbnaw/download", stream=True)
  block_size = 1024*1024
  with open(path, 'wb') as f:
    for data in r.iter_content(block_size):
      f.write(data)
```

***
Parsing the data
------------------
Now that we have the file, let's load the data:

```{code-cell} ipython3
data = nap.load_file(path, lazy_loading=False)
data
```

Let's save the RoiResponseSeries as a variable called 'transients' and print it:

```{code-cell} ipython3
transients = data['RoiResponseSeries']
transients
```

***
Plotting the activity of one neuron
-----------------------------------
Our transients are saved as a (35757, 65) TsdFrame. Looking at the printed object, you can see that we have 35757 data points for each of our 65 regions of interest (ROIs). We want to see which of these are head-direction cells, so we need to plot a tuning curve of fluorescence vs head-direction of the animal.

```{code-cell} ipython3
plt.figure(figsize=(6, 2))
plt.plot(transients[0:2000,0], linewidth=5)
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.show()
```

Here, we extract the head-direction as a variable called angle.

```{code-cell} ipython3
angle = data['ry']
angle
```

As you can see, we have a longer recording for our tracking of the animal's head than we do for our calcium imaging - something to keep in mind.

```{code-cell} ipython3
transients.time_support
```

***
Calcium tuning curves
---------------------
Here, we compute the tuning curves of all the ROIs.

```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(transients, angle, bins=120)
tuning_curves
```

This yields an `xarray.DataFrame`, which we can beautify by setting feature names and units:

```{code-cell} ipython3
def set_metadata(tuning_curves):
    _tuning_curves=tuning_curves.rename({"0": "Angle", "unit": "ROI"})
    _tuning_curves.name="Fluorescence"
    _tuning_curves.attrs["units"]="a.u."
    _tuning_curves.coords["Angle"].attrs["units"]="rad"
    return _tuning_curves

annotated_tuning_curves = set_metadata(tuning_curves)
annotated_tuning_curves
```

Having set some metadata, we can easily plot one ROI:

```{code-cell} ipython3
annotated_tuning_curves[4].plot()
plt.show()
```

It looks like this could be a head-direction cell. One important property of head-directions cells however, is that their firing with respect to head-direction is stable. To check for their stability, we can split our recording in two and compute a tuning curve for each half of the recording.

We start by finding the midpoint of the recording, using the function [`get_intervals_center`](pynapple.IntervalSet.get_intervals_center). Using this, then create one new IntervalSet with two rows, one for each half of the recording.

```{code-cell} ipython3
center = transients.time_support.get_intervals_center()

halves = nap.IntervalSet(
    start = [transients.time_support.start[0], center.t[0]],
    end = [center.t[0], transients.time_support.end[0]]
)
```

Now, we can compute the tuning curves for each half of the recording and plot the tuning curves again.

```{code-cell} ipython3
tuning_curves_half1 = nap.compute_tuning_curves(transients, angle, bins = 120, epochs = halves.loc[[0]])
tuning_curves_half2 = nap.compute_tuning_curves(transients, angle, bins = 120, epochs = halves.loc[[1]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
set_metadata(tuning_curves_half1[4]).plot(ax=ax1)
ax1.set_title("First half")
set_metadata(tuning_curves_half2[4]).plot(ax=ax2)
ax2.set_title("Second half")
plt.show()
```

***
Calcium decoding
---------------------

Given some tuning curves, we can also try to decode head direction from the population activity.
For calcium imaging data, Pynapple has `decode_template`, which implements a template matching algorithm.

```{code-cell} ipython3
epochs = nap.IntervalSet([50, 150])
decoded, dist = nap.decode_template(
    tuning_curves=tuning_curves,
    data=transients,
    epochs=epochs,
    bin_size=0.1,
    metric="correlation",
)
```

```{code-cell} ipython3
:tags: [hide-input]
# normalize distance for better visualization
dist_norm = (dist - np.min(dist.values, axis=1, keepdims=True)) / np.ptp(
    dist.values, axis=1, keepdims=True
)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 8), nrows=3, ncols=1, sharex=True)
ax1.plot(angle.restrict(epochs), label="True")
ax1.scatter(decoded.times(), decoded.values, label="Decoded", c="orange")
ax1.legend(frameon=False, bbox_to_anchor=(1.0, 1.0))
ax1.set_ylabel("Angle [rad]")

im = ax2.imshow(
    dist.values.T, 
    aspect="auto", 
    origin="lower", 
    cmap="inferno_r", 
    extent=(epochs.start[0], epochs.end[0], 0.0, 2*np.pi)
)
ax2.set_ylabel("Angle [rad]")
cbar_ax2 = fig.add_axes([0.95, ax2.get_position().y0, 0.015, ax2.get_position().height])
fig.colorbar(im, cax=cbar_ax2, label="Distance")

im = ax3.imshow(
    dist_norm.values.T, 
    aspect="auto", 
    origin="lower", 
    cmap="inferno_r", 
    extent=(epochs.start[0], epochs.end[0], 0.0, 2*np.pi)
)
cbar_ax3 = fig.add_axes([0.95, ax3.get_position().y0, 0.015, ax3.get_position().height])
fig.colorbar(im, cax=cbar_ax3, label="Norm. distance")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Angle [rad]")
plt.show()
```

The distance metric you choose can influence how well we decode.
Internally, ``decode_template`` uses `scipy.spatial.distance.cdist` to compute the distance matrix; 
you can take a look at [its documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) 
to see which metrics are supported, here are a couple examples:

```{code-cell} ipython3
:tags: [hide-input]
metrics = [
    "chebyshev",
    "dice",
    "canberra",
    "sqeuclidean",
    "minkowski",
    "euclidean",
    "cityblock",
    "mahalanobis",
    "correlation",
    "cosine",
    "seuclidean",
    "braycurtis",
    "jensenshannon",
]

fig, axs = plt.subplots(5, 1, figsize=(8,12), sharex=True, sharey=True)
for metric, ax in zip(metrics[-5:], axs.flatten()):
    decoded, dist = nap.decode_template(
        tuning_curves=tuning_curves,
        data=transients,
        bin_size=0.1,
        metric=metric,
        epochs=epochs,
    )
    # normalize distance for better visualization
    dist_norm = (dist - np.min(dist.values, axis=1, keepdims=True)) / np.ptp(
        dist.values, axis=1, keepdims=True
    )
    ax.plot(angle.restrict(epochs), label="True")
    im = ax.imshow(
        dist_norm.values.T, 
        aspect="auto", 
        origin="lower", 
        cmap="inferno_r", 
        extent=(epochs.start[0], epochs.end[0], 0.0, 2*np.pi)
    )
    if metric != metrics[-1]:
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    ax.set_ylabel(metric)
cbar_ax = fig.add_axes([0.92, ax.get_position().y0, 0.015, ax.get_position().height])
cbar=fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Norm. distance")
ax.set_xlabel("Time (s)")
plt.show()
```

We recommend to try out a bunch and see which one works best for you.
In the case of head direction, we can quantify how well we decode using the absolute angular error.
To get a fair estimate of error, we will compute the tuning curves on the first half of the data 
and compute the error for predictions of the second half.

```{code-cell} ipython3
def absolute_angular_error(x, y):
    return np.abs(np.angle(np.exp(1j * (x - y))))

# Compute errors
errors = {}
for metric in metrics:
    decoded, dist = nap.decode_template(
        tuning_curves=tuning_curves_half1,
        data=transients,
        bin_size=0.1,
        metric=metric,
        epochs=halves.loc[[1]],
    )
    errors[metric] = absolute_angular_error(
        angle.interpolate(decoded).values, decoded.values
    )
```

```{code-cell} ipython3
:tags: [hide-input]
sorted_items = sorted(errors.items(), key=lambda item: np.median(item[1]))
sorted_labels, sorted_values = zip(*sorted_items)

fig, ax = plt.subplots(figsize=(8, 8))
bp = ax.boxplot(
    x=sorted_values,
    tick_labels=sorted_labels,
    vert=False,
    showfliers=False
)
ax.set_xlabel("Angular error [rad]")
plt.show()
```

In this case, `jensenshannon` yields the lowest angular error.

:::{card}
Authors
^^^
Sofia Skromne Carrasco

Wolf De Wulf

:::
