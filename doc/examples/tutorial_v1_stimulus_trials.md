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

# Stimulus-aligned responses of a V1 neuron

Trial objects are not limited to experiments with a trial structure.
This tutorial uses a recording with one long white noise stimulus and no trials.
We define our own trials around single stimulus frames, and we use them to measure how a neuron in the primary visual cortex (V1) responds.

In this tutorial, we will show you how to use Pynapple to:

1. Estimate a receptive field with the spike-triggered average.
2. Convert the white noise movie into one value for each frame.
3. Define trials around stimulus frames, with metadata for each trial.
4. Align spikes to these frames with a [`TsTrials`](pynapple.TsTrials) object.
5. Compare the response between groups of trials.
6. Align the stimulus itself with a [`TsdTrials`](pynapple.TsdTrials) object.

Sonica Saraf collected the data in the Movshon lab at NYU.
The NeMoS library uses the same recording in its [tutorial on V1 cells](https://nemos-neuro.org/en/latest/tutorials/plot_04_v1_cells.html), which fits a generalized linear model to this neuron.

```{code-cell} ipython3
:tags: [hide-input]
# we'll import the packages we're going to use
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

import pynapple as nap

# some configuration, you can ignore this
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params);
```

## Loading data

[OSF](https://osf.io/download/xesdm/) hosts the NWB file. The file size is approximately 500 MB.

```{code-cell} ipython3
path = "m691l1.nwb"
if path not in os.listdir("."):
    r = requests.get("https://osf.io/download/xesdm/", stream=True)
    block_size = 1024 * 1024
    with open(path, "wb") as f:
        for data in r.iter_content(block_size):
            f.write(data)
```

```{code-cell} ipython3
data = nap.load_file(path)
data
```

The file contains the spike times of 73 neurons and a white noise movie.
Each frame has 51 x 51 pixels, and the movie has a frame rate of approximately 40 Hz:

```{code-cell} ipython3
units = data["units"]
stimulus = data["whitenoise"]
stimulus
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(stimulus[0], cmap="Greys_r")
ax.set_title("frame 0")
ax.axis("off")
plt.show()
```

We analyze unit 34, the neuron of the NeMoS tutorial.
The firing rate of this neuron is not stable during the recording. It falls to almost zero near 1000 s:

```{code-cell} ipython3
example_unit = 34
rate = units[example_unit].restrict(stimulus.time_support).count(60) / 60

plt.figure(figsize=(8, 3))
plt.plot(rate)
plt.xlabel("time (s)")
plt.ylabel("rate (Hz)")
plt.title(f"unit {example_unit}")
plt.show()
```

The receptive field also changes between the two halves of the recording.
We use the second half only, where the receptive field is the clearest.

## Separating the training frames and the test frames

A receptive field matches the frames before the spikes that we used to estimate it, even when these spikes are only noise.
If we measure the response on the same spikes, the response is too large.

To prevent this, we divide the second half of the recording into blocks of 20 seconds.
The even blocks are for training, and the odd blocks are for testing:

```{code-cell} ipython3
half = stimulus.t[len(stimulus) // 2]
edges = np.arange(half, stimulus.t[-1], 20.0)
train = nap.IntervalSet(start=edges[0:-1:2], end=edges[1::2] - 0.001)
test = nap.IntervalSet(start=edges[1:-1:2], end=edges[2::2] - 0.001)
test
```

## Estimating the receptive field

The spike-triggered average is the mean stimulus before a spike.
We calculate it on the training frames with [`compute_event_triggered_average`](pynapple.process.perievent.compute_event_triggered_average):

```{code-cell} ipython3
sta = nap.compute_event_triggered_average(
    stimulus.restrict(train),
    units[[example_unit]],
    binsize=0.025,
    window=(-0.15, 0.0),
)

vmax = np.abs(sta.values).max()
fig, axes = plt.subplots(1, len(sta), figsize=(2.5 * len(sta), 2.5))
for i, t in enumerate(sta.t):
    axes[i].imshow(sta[i, 0], vmin=-vmax, vmax=vmax, cmap="Greys_r")
    axes[i].set_title(f"{t:.3f} s", fontsize=14)
    axes[i].axis("off")
plt.show()
```

The structure is clear from -0.1 s to -0.05 s.
We average these lags to get one spatial filter.

Most of the 2601 pixels contain only noise, and together they are larger than the signal.
For this reason, we keep only the pixels with a large value:

```{code-cell} ipython3
receptive_field = np.mean(sta.get(-0.1, -0.05), axis=0)[0]
receptive_field = np.where(
    np.abs(receptive_field) > 4 * receptive_field.std(), receptive_field, 0.0
)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(receptive_field, cmap="Greys_r")
ax.set_title("receptive field")
ax.axis("off")
plt.show()
```

## Calculating the drive of each frame

We project each frame on the receptive field.
The result is a [`Tsd`](pynapple.Tsd) with one value for each frame.
We call this value the drive.
A high value shows that the frame matches the receptive field.
A low value shows that the frame matches the inverse of the receptive field.

```{code-cell} ipython3
drive = np.einsum("thw,hw->t", stimulus, receptive_field)
drive
```

## Defining trials

A trial must have a start and an end, and trials must not overlap.
We select one test frame every 16 frames (approximately 0.4 s).
Each trial starts 4 frames before the selected frame, and ends 10 frames after it.

We store the time and the drive of the selected frame in the metadata of the trials.
We also store the drive level, which is the quintile of the drive:

```{code-cell} ipython3
test_frames = np.flatnonzero(~np.isnan(test.in_interval(drive)))[::16]

trials = nap.IntervalSet(
    start=drive.t[test_frames - 4],
    end=drive.t[test_frames + 10],
    metadata={
        "frame_time": drive.t[test_frames],
        "drive": drive.values[test_frames],
    },
)
trials["level"] = pd.qcut(trials["drive"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
trials
```

## Aligning spikes to the frames

A [`TsTrials`](pynapple.TsTrials) object combines the spike trains with the trials.
By default, the time of each spike is relative to the start of its trial:

```{code-cell} ipython3
trial_data = nap.TsTrials(source=units, trials=trials)
trial_data
```

We use [`realign_to`](pynapple.TsTrials.realign_to) to measure time from the selected frame:

```{code-cell} ipython3
trial_data = trial_data.realign_to("frame_time")
trial_data
```

## Comparing the response between groups of trials

[`groupby`](pynapple.TsTrials.groupby) returns one trial object for each value of a metadata column.
We use it to make a raster of the two extreme drive levels:

```{code-cell} ipython3
groups = trial_data.groupby("level")
position = list(units.keys()).index(example_unit)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for ax, level, title in zip(axes, [1, 5], ["lowest drive", "highest drive"]):
    ax.plot(
        groups[level][:, position].to_tsdframe()["trial"],
        marker="|",
        linestyle="none",
        markersize=2,
    )
    ax.axvline(0.0, color="red")
    ax.set_title(title)
    ax.set_xlabel("time from frame (s)")
axes[0].set_ylabel("trial")
plt.show()
```

The raster has many trials and a low firing rate, so an effect is difficult to see.
We use [`count`](pynapple.TsTrials.count) to bin the spikes.
The result is a [`TsdTrials`](pynapple.TsdTrials) object with the dimensions trial, time and unit.
The first bin and the last bin are not complete, so they contain only NaN.
We remove them:

```{code-cell} ipython3
bin_size = 0.025
binned = trial_data.count(bin_size)[:, 1:-1]
binned
```

[`groupby_apply`](pynapple.TsdTrials.groupby_apply) applies a function to each group.
Here, we use it to calculate the mean count of each time bin, and we convert the mean to a firing rate:

```{code-cell} ipython3
mean_counts = binned.groupby_apply("level", np.nanmean, axis=0)

colors = plt.cm.coolwarm(np.linspace(0, 1, 5))
plt.figure(figsize=(8, 4))
for level, color in zip([1, 2, 3, 4, 5], colors):
    plt.plot(
        mean_counts[level].t,
        mean_counts[level].values[:, position] / bin_size,
        color=color,
        label=f"level {level}",
    )
plt.axvline(0.0, color="grey", linestyle="--")
plt.xlabel("time from frame (s)")
plt.ylabel("rate (Hz)")
plt.title(f"unit {example_unit}")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
plt.show()
```

The response has its peak between 50 ms and 100 ms after the frame.
The highest drive level increases the firing rate, and the lowest level decreases it.
The trials are test frames, so the estimate of the receptive field did not use these frames.

## Measuring the input-output function

The receptive field uses the lags from -0.1 s to -0.05 s, so we measure the response from 0.05 s to 0.1 s.
We use [`get`](pynapple.TsTrials.get) to keep this window, and we count the spikes of each trial.
The mean rate for each drive level gives the input-output function of the neuron:

```{code-cell} ipython3
window = (0.05, 0.1)
response = trial_data.get(*window).count(window[1] - window[0])
rates = response.values[:, 0, position] / (window[1] - window[0])

summary = pd.DataFrame(
    {"level": trials["level"], "drive": trials["drive"], "rate": rates}
)
summary = summary.groupby("level").agg(
    drive=("drive", "mean"),
    rate=("rate", "mean"),
    sem=("rate", "sem"),
)
summary
```

```{code-cell} ipython3
plt.figure(figsize=(5, 4))
plt.errorbar(summary["drive"], summary["rate"], yerr=summary["sem"], marker="o")
plt.xlabel("drive")
plt.ylabel("rate (Hz)")
plt.title("response from 50 ms to 100 ms")
plt.show()
```

## Aligning the stimulus

A continuous signal can use the same trials.
A [`TsdTrials`](pynapple.TsdTrials) object holds the drive of each trial.
Time is relative to the start of the trial, so the selected frame is at sample 4:

```{code-cell} ipython3
drive_trials = nap.TsdTrials(source=drive, trials=trials)
drive_trials
```

This gives us a control for the analysis above.
White noise frames are independent, so the frames near the selected frame must not depend on the drive level:

```{code-cell} ipython3
mean_drive = drive_trials.groupby_apply("level", np.nanmean, axis=0)

plt.figure(figsize=(8, 4))
for level, color in zip([1, 2, 3, 4, 5], colors):
    plt.plot(
        mean_drive[level].t - mean_drive[level].t[4],
        mean_drive[level].values,
        color=color,
        label=f"level {level}",
    )
plt.xlabel("time from frame (s)")
plt.ylabel("drive")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
plt.show()
```

Only the selected frame has a mean drive that depends on the level.
The response above is therefore the response to this frame.

:::{card}
Authors
^^^

Guillaume Viejo
:::
