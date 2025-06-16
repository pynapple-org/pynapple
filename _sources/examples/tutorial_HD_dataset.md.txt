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


Analysing head-direction cells
============

This tutorial demonstrates how we use Pynapple to generate Figure 4a in the [publication](https://elifesciences.org/reviewed-preprints/85786).
The NWB file for the example is hosted on [OSF](https://osf.io/jb2gd). We show below how to stream it.
The entire dataset can be downloaded [here](https://dandiarchive.org/dandiset/000056).


```{code-cell} ipython3
import numpy as np
import pandas as pd
import pynapple as nap
import scipy.ndimage
import matplotlib.pyplot as plt
import seaborn as sns
import requests, math, os


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

***
Downloading the data
------------------

It's a small NWB file.


```{code-cell} ipython3
path = "Mouse32-140822.nwb"
if path not in os.listdir("."):
    r = requests.get(f"https://osf.io/jb2gd/download", stream=True)
    block_size = 1024*1024
    with open(path, 'wb') as f:
        for data in r.iter_content(block_size):
            f.write(data)
```

***
Parsing the data
------------------

The first step is to load the data and other relevant variables of interest


```{code-cell} ipython3
data = nap.load_file(path)  # Load the NWB file for this dataset
```

What does this look like ?


```{code-cell} ipython3
print(data)
```

***
Head-Direction Tuning Curves
------------------

To plot Head-Direction Tuning curves, we need the spike timings and the orientation of the animal. These quantities are stored in the variables 'units' and 'ry'.


```{code-cell} ipython3
spikes = data["units"]  # Get spike timings
epochs = data["epochs"]  # Get the behavioural epochs (in this case, sleep and wakefulness)
angle = data["ry"]  # Get the tracked orientation of the animal
```

What does this look like ?


```{code-cell} ipython3 
print(spikes)
```

Here, rate is the mean firing rate of the unit. Location indicates the brain region the unit was recorded from, and group refers to the shank number on which the cell was located.

This dataset contains units recorded from the anterior thalamus. Head-direction (HD) cells are found in the anterodorsal nucleus of the thalamus (henceforth referred to as ADn). Units were also recorded from nearby thalamic nuclei in this animal. For the purposes of our tutorial, we are interested in the units recorded in ADn. We can restrict ourselves to analysis of these units rather easily, using Pynapple.


```{code-cell} ipython3
spikes_adn = spikes.getby_category("location")["adn"]  # Select only those units that are in ADn
```

What does this look like ?


```{code-cell} ipython3
print(spikes_adn)
```

Let's compute some head-direction tuning curves. To do this in Pynapple, all you need is a single line of code!

Plot firing rate of ADn units as a function of heading direction, i.e. a head-direction tuning curve


```{code-cell} ipython3
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes_adn, 
    feature=angle, 
    nb_bins=61, 
    ep=epochs[epochs.tags == "wake"],
    minmax=(0, 2 * np.pi)
    )
```

What does this look like ?


```{code-cell} ipython3
tuning_curves
```

Each row indicates an angular bin (in radians), and each column corresponds to a single unit. Let's compute the preferred angle quickly as follows:


```{code-cell} ipython3
pref_ang = tuning_curves.idxmax()
```

For easier visualization, we will colour our plots according to the preferred angle of the cell. To do so, we will normalize the range of angles we have, over a colourmap.


```{code-cell} ipython3
norm = plt.Normalize()  # Normalizes data into the range [0,1]
color = plt.cm.hsv(norm([i / (2 * np.pi) for i in pref_ang.values]))  # Assigns a colour in the HSV colourmap for each value of preferred angle
color = pd.DataFrame(index=pref_ang.index, data = color, columns = ['r', 'g', 'b', 'a'])
```

To make the tuning curves look nice, we will smooth them before plotting, using this custom function:


```{code-cell} ipython3
from scipy.ndimage import gaussian_filter1d
def smoothAngularTuningCurves(tuning_curves, sigma=2):

    tmp = np.concatenate((tuning_curves.values, tuning_curves.values, tuning_curves.values))
    tmp = gaussian_filter1d(tmp, sigma=sigma, axis=0)

    return pd.DataFrame(index = tuning_curves.index,
        data = tmp[tuning_curves.shape[0]:tuning_curves.shape[0]*2], 
        columns = tuning_curves.columns
        )
```

Therefore, we have:


```{code-cell} ipython3
smoothcurves = smoothAngularTuningCurves(tuning_curves, sigma=3)
```

What does this look like? Let's plot the tuning curves!


```{code-cell} ipython3
plt.figure(figsize=(12, 9))
for i, n in enumerate(pref_ang.sort_values().index.values):
    plt.subplot(8, 4, i + 1, projection='polar')  # Plot the curves in 8 rows and 4 columns
    plt.plot(
        smoothcurves[n], color=color.loc[n]
    )  # Colour of the curves determined by preferred angle    
    plt.xticks([])
plt.show()
```

Awesome!


+++

***
Decoding
------------------

Now that we have HD tuning curves, we can go one step further. Using only the population activity of ADn units, we can decode the direction the animal is looking in. We will then compare this to the real head direction of the animal, and discover that population activity in the ADn indeed codes for HD.

To decode the population activity, we will be using a Bayesian Decoder as implemented in Pynapple. Just a single line of code!


```{code-cell} ipython3
decoded, proba_feature = nap.decode_1d(
    tuning_curves=tuning_curves,
    group=spikes_adn,
    ep=epochs[epochs.tags == "wake"],
    bin_size=0.1,  # second
    feature=angle,
)
```

What does this look like ?


```{code-cell} ipython3
print(decoded)
```

The variable 'decoded' indicates the most probable angle in which the animal was looking. There is another variable, 'proba_feature' that denotes the probability of a given angular bin at a given time point. We can look at it below:


```{code-cell} ipython3
print(proba_feature)
```

Each row is a time bin, and each column is an angular bin. The sum of all values in a row add up to 1.

Now, let's plot the raster plot for a given period of time, and overlay the actual and decoded HD on the population activity.


```{code-cell} ipython3
ep = nap.IntervalSet(
    start=10717, end=10730
)  # Select an arbitrary interval for plotting

plt.figure()
plt.rc("font", size=12)
for i, n in enumerate(spikes_adn.keys()):
    plt.plot(
        spikes[n].restrict(ep).fillna(pref_ang[n]), "|", color=color.loc[n]
    )  # raster plot for each cell
plt.plot(
    decoded.restrict(ep), "--", color="grey", linewidth=2, label="decoded HD"
)  # decoded HD
plt.legend(loc="upper left")
plt.xlabel("Time (s)")
plt.ylabel("Neurons")
```

From this plot, we can see that the decoder is able to estimate the head-direction based on the population activity in ADn. Amazing!

What does the probability distribution in this example event look like?
Ideally, the bins with the highest probability will correspond to the bins having the most spikes. Let's plot the probability matrix to visualize this.


```{code-cell} ipython3
smoothed = scipy.ndimage.gaussian_filter(
    proba_feature, 1
)  # Smoothening the probability distribution

# Create a DataFrame with the smoothed distribution
p_feature = pd.DataFrame(
    index=proba_feature.index.values,
    columns=proba_feature.columns.values,
    data=smoothed,
)
p_feature = nap.TsdFrame(p_feature)  # Make it a Pynapple TsdFrame

plt.figure()
plt.plot(
    angle.restrict(ep), "w", linewidth=2, label="actual HD", zorder=1
)  # Actual HD, in white
plt.plot(
    decoded.restrict(ep), "--", color="grey", linewidth=2, label="decoded HD", zorder=1
)  # Decoded HD, in grey

# Plot the smoothed probability distribution
plt.imshow(
    np.transpose(p_feature.restrict(ep).values),
    aspect="auto",
    interpolation="bilinear",
    extent=[ep["start"][0], ep["end"][0], 0, 2 * np.pi],
    origin="lower",
    cmap="viridis",
)

plt.xlabel("Time (s)")  # X-axis is time in seconds
plt.ylabel("Angle (rad)")  # Y-axis is the angle in radian
plt.colorbar(label="probability")
```

From this probability distribution, we observe that the decoded HD very closely matches the actual HD. Therefore, the population activity in ADn is a reliable estimate of the heading direction of the animal.

I hope this tutorial was helpful. If you have any questions, comments or suggestions, please feel free to reach out to the Pynapple Team!


:::{card}
Authors
^^^
Dhruv Mehrotra

Guillaume Viejo

:::