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

For the example dataset, we will be working with a recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging). The area recorded for this experiment is the postsubiculum - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in a specific direction.

The NWB file for the example is hosted on [OSF](https://osf.io/sbnaw). We show below how to stream it.


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
import numpy as pd
import pynapple as nap
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import requests, math

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

***
Downloading the data
------------------
First things first: Let's find our file


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
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
Now that we have the file, let's load the data


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
data = nap.load_file(path)
print(data)
```

Let's save the RoiResponseSeries as a variable called 'transients' and print it


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
transients = data['RoiResponseSeries']
print(transients)
```

***
Plotting the activity of one neuron
-----------------------------------
Our transients are saved as a (35757, 65) TsdFrame. Looking at the printed object, you can see that we have 35757 data points for each of our 65 regions of interest. We want to see which of these are head-direction cells, so we need to plot a tuning curve of fluorescence vs head-direction of the animal.



```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
plt.figure(figsize=(6, 2))
plt.plot(transients[0:2000,0], linewidth=5)
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.show()
```

Here we extract the head-direction as a variable called angle


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
angle = data['ry']
print(angle)
```

As you can see, we have a longer recording for our tracking of the animal's head than we do for our calcium imaging - something to keep in mind.


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
print(transients.time_support)
print(angle.time_support)
```

***
Calcium tuning curves
---------------------
Here we compute the tuning curves of all the neurons


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
tcurves = nap.compute_1d_tuning_curves_continuous(transients, angle, nb_bins = 120)

print(tcurves)
```

We now have a DataFrame, where our index is the angle of the animal's head in radians, and each column represents the tuning curve of each region of interest. We can plot one neuron.


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
plt.figure()
plt.plot(tcurves[4])
plt.xlabel("Angle")
plt.ylabel("Fluorescence")
plt.show()
```

It looks like this could be a head-direction cell. One important property of head-directions cells however, is that their firing with respect to head-direction is stable. To check for their stability, we can split our recording in two and compute a tuning curve for each half of the recording.

We start by finding the midpoint of the recording, using the function [`get_intervals_center`](pynapple.IntervalSet.get_intervals_center). Using this, then create one new IntervalSet with two rows, one for each half of the recording.


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
center = transients.time_support.get_intervals_center()

halves = nap.IntervalSet(
	start = [transients.time_support.start[0], center.t[0]],
    end = [center.t[0], transients.time_support.end[0]]
    )
```

Now we can compute the tuning curves for each half of the recording and plot the tuning curves for the fifth region of interest. 


```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
half1 = nap.compute_1d_tuning_curves_continuous(transients, angle, nb_bins = 120, ep = halves.loc[[0]])
half2 = nap.compute_1d_tuning_curves_continuous(transients, angle, nb_bins = 120, ep = halves.loc[[1]])

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(half1[4])
plt.title("First half")
plt.xlabel("Angle")
plt.ylabel("Fluorescence")
plt.subplot(1,2,2)
plt.plot(half2[4])
plt.title("Second half")
plt.xlabel("Angle")
plt.show()
```

:::{card}
Authors
^^^
Sofia Skromne Carrasco

:::
