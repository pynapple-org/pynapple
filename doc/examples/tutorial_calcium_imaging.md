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
angle.time_support
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
half1 = nap.compute_tuning_curves(transients, angle, bins = 120, epochs = halves.loc[[0]])
half2 = nap.compute_tuning_curves(transients, angle, bins = 120, epochs = halves.loc[[1]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
set_metadata(half1[4]).plot(ax=ax1)
ax1.set_title("First half")
set_metadata(half2[4]).plot(ax=ax2)
ax2.set_title("Second half")
plt.show()
```

:::{card}
Authors
^^^
Sofia Skromne Carrasco

:::
