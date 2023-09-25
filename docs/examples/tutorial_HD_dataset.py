#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peyrache et al (2015) Dataset Tutorial
============

This tutorial demonstrates how we use Pynapple on various publicly available datasets in systems neuroscience to streamline analysis. In this tutorial, we will examine the dataset from [Peyrache et al (2015)](https://www.nature.com/articles/nn.3968), which was used to generate Figure 4a in the [publication](https://elifesciences.org/reviewed-preprints/85786).

The NWB file for the example used here is provided in [this](https://github.com/PeyracheLab/pynacollada/tree/main/pynacollada/Pynapple%20Paper%20Figures/Peyrache%202015/Mouse32/Mouse32-140822/pynapplenwb) repository. The entire dataset can be downloaded [here](https://dandiarchive.org/dandiset/000056).

See the [documentation](https://pynapple-org.github.io/pynapple/) of Pynapple for instructions on installing the package.

This tutorial was made by Dhruv Mehrotra.

First, import the necessary libraries:
    
"""
# %%

import numpy as np
import pandas as pd
import pynapple as nap
import scipy.ndimage
import matplotlib.pyplot as plt

# %%
# ***
# Head-Direction Tuning Curves
# ------------------
#
# The first step is to load the data and other relevant variables of interest

data_directory = (
    "/media/DataDhruv/Recordings/Mouse32/Mouse32-140822"  # Path to your data
)

data = nap.load_session(
    data_directory, "neurosuite"
)  # Load the NWB file for this dataset
spikes = data.spikes  # Get spike timings
epochs = data.epochs  # Get the behavioural epochs (in this case, sleep and wakefulness)
position = data.position  # Get the tracked position of the animal
spikes_by_location = spikes.getby_category(
    "location"
)  # Tells you which cells come from which brain region

# %%
# What does this look like ?
print(spikes_by_location)

# %%
# Here, index refers to the cluster number, Freq. (Hz) is the mean firing rate of the unit. Group refers to the shank number on which the cell was located, and location indicates the brain region the unit was recorded from.
#
# This dataset contains units recorded from the anterior thalamus. Head-direction (HD) cells are found in the anterodorsal nucleus of the thalamus (henceforth referred to as ADn). Units were also recorded from nearby thalamic nuclei in this animal. For the purposes of our tutorial, we are interested in the units recorded in ADn. We can restrict ourselves to analysis of these units rather easily, using Pynapple.

spikes_adn = spikes_by_location["adn"]  # Select only those units that are in ADn

# %%
# What does this look like ?
print(spikes_adn)

# %%
# Let's compute some head-direction tuning curves. To do this in Pynapple, all you need is a single line of code!
#
# Plot firing rate of ADn units as a function of heading direction, i.e. a head-direction tuning curve

tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes_adn, feature=position.loc["ry"], nb_bins=31, minmax=(0, 2 * np.pi)
)

# %%
# What does this look like ?
print(tuning_curves)

# %%
# Each row indicates an angular bin (in radians), and each column corresponds to a single unit. Let's compute the preferred angle as follows:

pref_ang = []

# Preferred angle is the angular bin with the maximal firing rate

for i in tuning_curves.columns:
    pref_ang.append(tuning_curves.loc[:, i].idxmax())

# %%
# For easier visualization, we will colour our plots according to the preferred angle of the cell. To do so, we will normalize the range of angles we have, over a colourmap.

norm = plt.Normalize()  # Normalizes data into the range [0,1]
color = plt.cm.hsv(
    norm([i / (2 * np.pi) for i in pref_ang])
)  # Assigns a colour in the HSV colourmap for each value of preferred angle

# %%
# To make the tuning curves look nice, we will smoothen them before plotting, using this custom function:


def smoothAngularTuningCurves(tuning_curves, window=20, deviation=3.0):
    new_tuning_curves = {}
    for i in tuning_curves.columns:
        tcurves = tuning_curves[i]
        offset = np.mean(np.diff(tcurves.index.values))
        padded = pd.Series(
            index=np.hstack(
                (
                    tcurves.index.values - (2 * np.pi) - offset,
                    tcurves.index.values,
                    tcurves.index.values + (2 * np.pi) + offset,
                )
            ),
            data=np.hstack((tcurves.values, tcurves.values, tcurves.values)),
        )
        smoothed = padded.rolling(
            window=window, win_type="gaussian", center=True, min_periods=1
        ).mean(std=deviation)
        new_tuning_curves[i] = smoothed.loc[tcurves.index]

    new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

    return new_tuning_curves


# %%
# Therefore, we have:

smoothcurves = smoothAngularTuningCurves(tuning_curves, window=20, deviation=3)

# %%
# What does this look like? Let's plot the tuning curves!

plt.figure(figsize=(12, 9))
for i, n in enumerate(smoothcurves.columns):
    plt.subplot(8, 4, i + 1)  # Plot the curves in 8 rows and 4 columns
    plt.plot(
        smoothcurves[n], color=color[i]
    )  # Colour of the curves determined by preferred angle
    plt.tight_layout()
    plt.xlabel("Angle (rad)")  # Angle in radian, on the X-axis
    plt.ylabel("Firing Rate (Hz)")  # Firing rate in Hz, on the Y-axis

# %%
# Awesome!

# %%
# ***
# Decoding
# ------------------
#
# Now that we have HD tuning curves, we can go one step further. Using only the population activity of ADn units, we can decode the direction the animal is looking in. We will then compare this to the real heead direction of the animal, and discover that population activity in the ADn indeed codes for HD.
#
# To decode the population activity, we will be using a Bayesian Decoder as implemented in Pynapple. Just a single line of code!

decoded, proba_feature = nap.decode_1d(
    tuning_curves=tuning_curves,
    group=spikes_adn,
    ep=epochs["wake"],
    bin_size=0.1,  # second
    feature=position["ry"],
)

# %%
# What does this look like ?

print(decoded)

# %%
# The variable 'decoded' indicates the most probable angle in which the animal was looking. There is another variable, 'proba_feature' that denotes the probability of a given angular bin at a given time point. We can look at it below:

print(proba_feature)

# %%
# Each row of this pandas DataFrame is a time bin, and each column is an angular bin. The sum of all values in a row add up to 1.
#
# Now, let's plot the raster plot for a given period of time, and overlay the actual and decoded HD on the population activity.

ep = nap.IntervalSet(
    start=10717, end=10730
)  # Select an arbitrary interval for plotting

plt.figure()
plt.rc("font", size=12)
for i, n in enumerate(spikes_adn.keys()):
    plt.plot(
        spikes[n].restrict(ep).as_units("s").fillna(pref_ang[i]), "|", color=color[i]
    )  # raster plot for each cell
plt.plot(
    decoded.restrict(ep), "--", color="grey", linewidth=2, label="decoded HD"
)  # decoded HD
plt.legend(loc="upper left")

# %%
# From this plot, we can see that the decoder is able to estimate the head-direction based on the population activity in ADn. Amazing!
#
# What does the probability distribution in this example event look like?
# Ideally, the bins with the highest probability will correspond to the bins having the most spikes. Let's plot the probability matrix to visualize this.

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
    position["ry"].restrict(ep), "w", linewidth=2, label="actual HD", zorder=1
)  # Actual HD, in white
plt.plot(
    decoded.restrict(ep), "--", color="grey", linewidth=2, label="decoded HD", zorder=1
)  # Decoded HD, in grey

# Plot the smoothed probability distribution
plt.imshow(
    np.transpose(p_feature.restrict(ep).values),
    aspect="auto",
    interpolation="bilinear",
    extent=[ep["start"].values[0], ep["end"].values[0], 0, 2 * np.pi],
    origin="lower",
    cmap="viridis",
)

plt.xlabel("Time (s)")  # X-axis is time in seconds
plt.ylabel("Angle (rad)")  # Y-axis is the angle in radian
plt.colorbar(label="probability")

# %%
# From this probability distribution, we observe that the decoded HD very closely matches the actual HD. Therefore, the population activity in ADn is a reliable estimate of the heading direction of the animal.
#
# I hope this tutorial was helpful. If you have any questions, comments or suggestions, please feel free to reach out to the Pynapple Team!
