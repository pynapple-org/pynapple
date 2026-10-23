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

# Trial-aligned population manifolds

How does a population of neurons evolve during a movement? 
Examining each neuron separately can reveal changes in firing rate, but it can be difficult to see how those changes are coordinated across the population.

A complementary approach is to represent population activity as a trajectory through a high-dimensional space. 
Each neuron contributes one coordinate, and each time point corresponds to one population activity pattern. 
Dimensionality reduction lets us visualize some of the structure in these trajectories.

In this tutorial, we will show you how to use Pynapple to:

1. Load spike times and trial information from DANDI.
2. Align neural activity to a behavioral event.
3. Convert spike times into smoothed, binned activity.
4. Use principal component analysis (PCA) to visualize population trajectories.
5. Compare the average trajectories associated with different reach targets.

This analysis introduces the population-trajectory perspective used in studies such as [Churchland et al. (2012)](https://www.nature.com/articles/nature11129). 
We will use ordinary PCA for visualization; we will not reproduce the paper’s rotational-dynamics analysis.

```{code-cell} ipython3
:tags: [hide-input]
# we'll import the packages we're going to use
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pynapple as nap

# some configuration, you can ignore this
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params);
```

## Loading data

We will stream a recording from [dandiset 00688](https://dandiarchive.org/dandiset/000688?search=reaching+monkey&pos=4),
which contains neural recordings from macaque primary motor cortex while they perform a center-out reaching task. 
In this task, the animal reaches from a central position toward a peripheral target:

```{code-cell} ipython3
:tags: [remove-input]

angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
target_positions = np.column_stack((np.cos(angles), np.sin(angles)))
selected_target = target_positions[1]

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    *target_positions.T,
    s=450,
    facecolors="none",
    edgecolors="0.65",
    linewidths=2,
)
ax.scatter(0, 0, s=180, color="0.25")
ax.scatter(*selected_target, s=450, color="tab:blue")

ax.annotate(
    "",
    xy=selected_target * 0.82,
    xytext=selected_target * 0.15,
    arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 2.5},
)
ax.text(0, -0.18, "Start", ha="center", va="top")
ax.text(
    selected_target[0] + 0.18,
    selected_target[1],
    "Target",
    va="center",
)
ax.set_title("Center-out reaching")
ax.text(
    0.5,
    -0.05,
    "Hold at center → Go cue → Reach to target",
    transform=ax.transAxes,
    ha="center",
    fontsize=12,
)
ax.set(xlim=(-1.35, 1.55), ylim=(-1.3, 1.3), aspect="equal")
ax.axis("off")
plt.show()
```

The recording is stored in an NWB file, a format that organizes neural activity, behavioral measurements, and experimental metadata in a common structure.
Pynapple allows loading NWB files using [`nap.load_file`](pynapple.io.misc.load_file).

See also the [tutorial on streaming data from DANDI](/examples/tutorial_pynapple_dandi.md) for more detail on accessing remote NWB files.

```{code-cell} ipython3
# These modules are used for data loading
from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py
from pynwb import NWBHDF5IO
```

```{code-cell} ipython3
dandiset_id = "000688"
# This is a randomly chosen recording session.
asset_path = "sub-C/sub-C_ses-CO-20150319_behavior+ecephys.nwb"
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(asset_path)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage=str(Path("~/.caches/nwb-cache").expanduser()),
)
io = NWBHDF5IO(file=h5py.File(fs.open(s3_url, "rb")), load_namespaces=True)
nwb = nap.NWBFile(io.read())
```

The NWB file is now accessible through Pynapple. 
Data are loaded on demand, so we do not need to download the entire recording before starting the analysis.

```{code-cell} ipython3
nwb
```
We will only use the neural activity (`units`) and the trial information (`trials`).
Inspecting these objects before processing them helps us understand which neurons, trials, and behavioral events are available.

```{code-cell} ipython3
units = nwb['units']
trials = nwb['trials']
```
```{code-cell} ipython3
units
```
```{code-cell} ipython3
trials
```

## Trial alignment
Spike times are initially expressed on the recording's time axis. 
To compare repeated movements, we need to organize those spikes into trials.

A [`TsTrials`](pynapple.core.TsTrials) object combines the spike trains with the trial intervals and metadata. 
This gives us a trial-based view of the recording while keeping the behavioral information associated with each trial.

```{code-cell} ipython3
trial_data = nap.TsTrials(source=units, trials=trials)
trial_data
```

## Subselecting trials based on metadata
We want to align neural activity to the go cue, so every trial included in the analysis needs a valid `go_cue_time`.

The following selection removes trials for which that value is missing:
```{code-cell} ipython3
trial_data = trial_data[~np.isnan(trial_data.trials["go_cue_time"])]
trial_data
```
Filtering through the trial object keeps the neural activity and trial metadata synchronized: 
when we remove a trial, its associated activity is removed as well.

## Visualising aligned spikes
Before applying any transformations, let's inspect the spikes from one example neuron.

A raster plot shows individual spikes as short vertical marks:
```{code-cell} ipython3
example_unit = 10
plt.plot(
    trial_data[:, example_unit].to_tsdframe()["trial"],
    marker="|",
    linestyle="none",
    markersize=1,
)
plt.ylabel("trial")
plt.xlabel("time from start of trial (s)")
plt.title(f"unit {example_unit}")
plt.show()
```
The horizontal axis gives time relative to the alignment event.
The vertical axis identifies the trial.
Each mark represents one spike.
Patterns that repeat at a similar time across trials appear as regions of greater spike density.

## Aligning to the go cue

Trial start is not always the most informative reference point. 
If the delay between trial start and the go cue varies, activity related to the go cue may appear spread out across trials.

We can instead set each trial's go cue to time zero.
This changes the temporal reference frame, not the recorded spikes:

```{code-cell} ipython3
trial_data = trial_data.realign_to("go_cue_time")
trial_data
```

```{code-cell} ipython3
plt.plot(
    trial_data[:, example_unit].to_tsdframe()["trial"],
    marker="|",
    linestyle="none",
    markersize=1,
)
plt.axvline(0.0, color="red")
plt.ylabel("trial")
plt.xlabel("time from go cue (s)")
plt.title(f"unit {example_unit}")
plt.show()
```
Negative times occur before the go cue.
Positive times occur after the go cue.
The red line marks the go cue. 
Comparing the two rasters can reveal whether activity is more consistently timed relative to trial start or to the go cue.

## Restricting the window
Trials can have different durations. 
For our comparison, we will focus on a common window from 200 ms before to 500 ms after the go cue.

This gives us a view of population activity around the same behavioral event,
rather than including unrelated portions of long trials.

```{code-cell} ipython3
trial_data = trial_data.get(start=-0.2, end=0.5)
trial_data
```

```{code-cell} ipython3
plt.plot(
    trial_data[:, example_unit].to_tsdframe()["trial"],
    marker="|",
    linestyle="none",
    markersize=1,
)
plt.axvline(0.0, color="red")
plt.ylabel("trial")
plt.xlabel("time from go cue (s)")
plt.title(f"unit {example_unit}")
plt.show()
```

## Binning

So far, the data consist of discrete spike times. 
PCA requires a matrix of numerical observations, so we first count the spikes in consecutive time bins.

With a bin size of 0.02 seconds, each value is the number of spikes produced by one neuron during one 20 ms interval.

The resulting [`TsdTrials`](pynapple.core.TsdTrials) object has three dimensions:
```{code-cell} ipython3
bin_size = 0.02
binned = trial_data.count(bin_size)
binned
```
The bin size controls a trade-off:

- Smaller bins preserve finer timing but usually produce noisier activity estimates.
- Larger bins reduce variability but can obscure rapid changes.

*Note:* These values are spike counts, not firing rates. 
Dividing by the bin duration in seconds would convert them to an average rate in spikes per second.

## Inspecting population activity
```{code-cell} ipython3
example_trial = 100
plt.imshow(
    binned[example_trial].values.T,
    aspect="auto",
    interpolation="none",
    extent=(
        binned.t[0],
        binned.t[-1],
        0,
        binned.shape[-1],
    ),
)
plt.axvline(0, color='red')
plt.ylabel('unit')
plt.xlabel('time from go cue (s)');
plt.title(f"trial {example_trial}")
plt.show()
```
The heatmap shows the binned activity for one trial. 
Each row is a neuron, each column is a time bin, and color represents the spike count.
Unlike the raster, which shows one neuron across many trials, this view shows many neurons within one trial.

## Preparing activity for PCA
We will smooth and normalize the binned activity before applying PCA.
These operations serve different purposes.

### Smoothing

Spiking activity varies from bin to bin. 
Gaussian smoothing averages activity over nearby time bins, making broader temporal patterns easier to see.
Here, we use a Gaussian standard deviation of 60 ms. 
This is wider than a single 20 ms bin, so neighboring bins contribute to the smoothed value.

Smoothing reduces temporal detail as well as variability. 
A wider smoothing kernel produces smoother trajectories, but it can blur sharp changes. 

```{code-cell} ipython3
smooth_sigma = 0.06
smoothed = binned.smooth(smooth_sigma)
```

### Normalizing
PCA is sensitive to the scale of its input features. 
Without normalization, neurons with larger activity fluctuations can dominate the result.

We normalize the smoothed activity before fitting PCA. 
The intended effect is to compare population patterns without allowing differences in individual-neuron scale to dominate.

```{code-cell} ipython3
normalized = smoothed.normalize()
```

## Constructing the PCA input matrix
At each time bin, the population activity is a vector with one value per neuron. 
With `N` neurons, this vector is a point in an `N`-dimensional space.
PCA expects a two-dimensional matrix of observations by features.

For this analysis:
- An observation is one time bin from one trial.
- A feature is one neuron.

We therefore combine the trial and time dimensions while leaving the neuron dimension unchanged:

```{code-cell} ipython3
n_trials, n_time, n_units = normalized.shape
X = normalized.values.reshape(n_trials*n_time, n_units)
```
This reshaping does not average trials or change the activity values. 
It only arranges the data into the format expected by [scikit-learn's](https://scikit-learn.org/stable/) [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

## Finding shared population axes with PCA
PCA finds orthogonal directions that capture progressively smaller amounts of variation in the input data:

- PC 1 captures the most variance.
- PC 2 captures the most remaining variance while being orthogonal to PC 1.
- PC 3 captures the most remaining variance while being orthogonal to both.

We fit one PCA model using all trial–time observations. 
This gives every trial a shared coordinate system, allowing us to compare trajectories across reach targets.

```{code-cell} ipython3
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
trajectories = pca.fit_transform(X).reshape(n_trials, n_time, 3)
```
We restore the trial and time dimensions, replacing the neuron dimension with three principal components: `(n_trials, n_time, 3)`.
Each trial is now a trajectory through a three-dimensional space rather than through a space with one dimension per neuron.
The components are weighted combinations of neurons, not individual neurons. 

## Comparing trajectories across reach targets

Individual trials can be noisy.
To summarize the activity associated with each reach target, we average the PCA trajectories across trials with the same target.

Each colored curve shows the mean population trajectory for one target:
```{code-cell} ipython3
targets = np.asarray(normalized.trials["target_id"])
categories = np.unique(targets)
colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(categories)))

fig = plt.figure(figsize=(8, 9))
ax = fig.add_subplot(projection="3d")
for target, color in zip(categories, colors, strict=True):
    mean = np.nanmean(trajectories[targets == target], axis=0)
    ax.plot(*mean.T, color=color, label=str(int(target)))
    ax.scatter(*mean[0], color="black", s=30)
    ax.scatter(*mean[-1], color="gray", s=30)
ax.set_xlabel("PC 1", labelpad=15)
ax.set_ylabel("PC 2", labelpad=15)
ax.set_zlabel("PC 3", labelpad=15)
ax.legend(title="target", loc="upper left", bbox_to_anchor=(1.12, 0.7), frameon=False)
plt.show()
```
Position along a curve changes with time.
The black marker indicates the first time bin.
The gray marker indicates the last time bin.

Low-dimensional projections help us see how neurons act together as a population. 
Instead of examining each neuron separately, we can follow a trajectory that summarizes how their collective activity changes over time.
This connects to the idea of a neural manifold: a lower-dimensional structure within the space of possible population activity patterns.
For an accessible introduction to neural manifolds, see [Matthew Perich's article in The Transmitter](https://www.thetransmitter.org/neural-dynamics/neural-population-based-approaches-have-opened-new-windows-into-neural-computations-and-behavior/).

:::{card}
Authors
^^^
[Wolf De Wulf](https://wulfdewolf.github.io)
:::
