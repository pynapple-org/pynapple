---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python3
  name: python3
---

# Trials

Pynapple provides two objects for organizing time series into trials:

- [`TsTrials`](pynapple.TsTrials) for timestamps, such as spikes or events.
- [`TsdTrials`](pynapple.TsdTrials) for sampled values, such as calcium traces or EEG.

Both combine a source time series with an [`IntervalSet`](pynapple.IntervalSet)
defining the trials.
By default, time is expressed relative to each trial's start.
Trial metadata can be used to select trials or align them to another event.

```{contents}
:depth: 3
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
```

## Defining trials

We will use three one-second trials.
Their boundaries are expressed in seconds on the original recording's time axis.

Metadata describe each trial: `target` identifies the trial condition, and
`go_cue_time` gives the cue timestamp on the same recording time axis.

```{code-cell} ipython3
trials = nap.IntervalSet(
    start=[1.0, 3.0, 5.0],
    end=[2.0, 4.0, 6.0],
)
trials.set_info(
    target=["left", "right", "left"],
    go_cue_time=[1.4, 3.4, 5.4],
)
trials
```

## Timestamp data (spikes, events, ...)

### Creating a TsTrials object

First, create a [`TsGroup`](pynapple.TsGroup) containing two synthetic spike trains.
The spike times are expressed in recording time.

```{code-cell} ipython3
units = nap.TsGroup(
    {
        0: nap.Ts(t=[1.1, 1.4, 1.5, 3.2, 3.4, 3.7, 5.1, 5.4, 5.8]),
        1: nap.Ts(t=[1.2, 1.6, 1.8, 3.1, 3.5, 3.9, 5.2, 5.5, 5.7]),
    },
    time_support=nap.IntervalSet(0, 6)
)

spike_trials = nap.TsTrials(source=units, trials=trials)
spike_trials
```

`TsTrials` organizes these timestamps by trial and unit.
Each trial initially starts at time zero.

### Selecting trials and units

Select a trial by its original identifier:
```{code-cell} ipython3
first_trial = spike_trials[0]
first_trial
```
This returns a [`TsGroup`](pynapple.TsGroup) object, containing the spikes for all units for this trial.


Select trials by their metadata:
```{code-cell} ipython3
left_trials = spike_trials[spike_trials.trials["target"] == "left"]
left_trials
```
This returns a [`TsTrials`](pynapple.TsTrials) object, but with a subset of the trials.

Use the second index to select a unit:
```{code-cell} ipython3
unit_trials = spike_trials[:, 0]
unit_trials
```
This returns a [`TsTrials`](pynapple.TsTrials) object, but with a subset of the units.

Trial or unit selection keeps the associated timestamps and metadata together.

### Aligning to an event

Use `realign_to` to set an event from the trial metadata to time zero:

```{code-cell} ipython3
aligned_spikes = spike_trials.realign_to("go_cue_time")
aligned_spikes
```

Times before the cue are now negative; times after it are positive.
The underlying recording timestamps are unchanged.

Use `get` to keep a window relative to the alignment event:

```{code-cell} ipython3
windowed_spikes = aligned_spikes.get(start=-0.2, end=0.4)
windowed_spikes
```

### Plotting a raster

For one unit, `to_tsdframe` provides a tabular time-series representation with a `trial` column.
Plotting that column produces a raster:

```{code-cell} ipython3
raster = windowed_spikes[:, 0].to_tsdframe()

plt.plot(raster["trial"], "|", markersize=10)
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Time from go cue (s)")
plt.ylabel("Trial")
plt.title("Unit 0")
plt.show()
```

### Converting timestamps to sampled values

`count` counts events in consecutive time bins and returns a
[`TsdTrials`](pynapple.TsdTrials) object:

```{code-cell} ipython3
bin_size = 0.1
counts = windowed_spikes.count(bin_size)
counts
```

For this population, the dimensions are `(trial, time, unit)`.

The values are spike counts. Divide by the bin duration to convert counts into
rates in spikes per second.

## Continuous data (calcium traces, EEG, ...)

### Creating a TsdTrials object

`TsdTrials` can also organize an existing sampled time series into trials.

We create two signals sampled at 100 Hz. Each trial contains a broad response near 0.5 seconds,
fast fluctuations, and noise.
Trials differ in baseline and response amplitude, while the two signals have different overall scales.

```{code-cell} ipython3
rng = np.random.default_rng(42)
t = np.arange(700) / 100
values = np.zeros((len(t), 2))

for start, end, amplitude, offset in zip(
    trials.start,
    trials.end,
    [0.7, 1.0, 1.4],
    [-0.5, 0.0, 0.6],
):
    selected = (t >= start) & (t <= end)
    relative_time = t[selected] - start

    response = np.exp(-0.5 * ((relative_time - 0.5) / 0.08) ** 2)
    fast = 0.3 * np.sin(2 * np.pi * 20 * relative_time)
    noise = rng.normal(0, 0.15, size=(selected.sum(), 2))

    values[selected, 0] = (
        10 + 3 * (offset + amplitude * response + fast + noise[:, 0])
    )
    values[selected, 1] = (
        100 + 20 * (offset + amplitude * response + fast + noise[:, 1])
    )

signals = nap.TsdFrame(
    t=t,
    d=values,
    columns=["signal_1", "signal_2"],
)
signal_trials = nap.TsdTrials(source=signals, trials=trials)
signal_trials
```

For this two-column source, the dimensions are `(trial, time, feature)`.

### Selecting trials and signals

Select a trial by its original identifier:
```{code-cell} ipython3
first_trial = signal_trials[0]
first_trial
```
This returns a [`TsdFrame`](pynapple.TsdFrame) object, containing the signals for this trial.


Select trials using their metadata:
```{code-cell} ipython3
left_signals = signal_trials[signal_trials.trials["target"] == "left"]
left_signals
```
This returns a [`TsdTrials`](pynapple.TsdTrials) object, but with a subset of the trials.

Use the third index to select a signal by its column name.
The axes are `(trial, time, feature)`:

```{code-cell} ipython3
first_signal = signal_trials[:, :, "signal_1"]
first_signal
```
This returns a scalar [`TsdTrials`](pynapple.TsdTrials) object with shape
`(trial, time)`. Use `["signal_1"]` instead to retain the feature axis.

### Extracting a time window
Time is expressed relative to each trial's start.
Use `get` to get a window on this relative time axis:
```{code-cell} ipython3
windowed_signals = signal_trials.get(start=0.0, end=0.8)
windowed_signals
```

### Accessing times and values

The aligned time coordinates are available through `.t`, and the data through
`.values`:

```{code-cell} ipython3
windowed_signals.t
```

```{code-cell} ipython3
windowed_signals.values.shape
```

Selecting one trial gives a time series that can be plotted directly:

```{code-cell} ipython3
plt.plot(windowed_signals[0])
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Time from trial start (s)")
plt.ylabel("Signal value")
plt.legend(["signal_1", "signal_2"])
plt.show()
```


### Smoothing

Sampled trial data can be smoothed with a Gaussian kernel. The smoothing
standard deviation is expressed in seconds:

```{code-cell} ipython3
smoothed = windowed_signals.smooth(0.03)
smoothed
```

```{code-cell} ipython3
:tags: [hide-input]
feature = windowed_signals.columns[0]
before = windowed_signals[:, :, feature]
after = smoothed[:, :, feature]
vmin = min(np.nanmin(before.d), np.nanmin(after.d))
vmax = max(np.nanmax(before.d), np.nanmax(after.d))
half_step = windowed_signals.sample_period / 2
extent = (
    before.t[0] - half_step,
    before.t[-1] + half_step,
    len(before) - 0.5,
    -0.5,
)
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4), sharex=True, sharey=True,
    layout="constrained",
)
for ax, data, title in zip(
    axes, (before, after), ("Before smoothing", "After smoothing")
):
    image = ax.imshow(
        data.d,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set(title=title, xlabel="Aligned time (s)")
    ax.set_yticks(np.arange(len(data)), labels=data.trial_ids)
axes[0].set_ylabel("Trial ID")
fig.suptitle(f"Signal = {feature}")
fig.colorbar(image, ax=axes, label="Signal")
plt.show()
```

### Convolving

Use `convolve` to apply a custom one-dimensional kernel along time.
For example, a five-sample moving average:

```{code-cell} ipython3
kernel = np.ones(5) / 5
filtered = windowed_signals.convolve(kernel)
filtered
```

```{code-cell} ipython3
:tags: [hide-input]
feature = windowed_signals.columns[0]
before = windowed_signals[:, :, feature]
after = filtered[:, :, feature]
vmin = min(np.nanmin(before.d), np.nanmin(after.d))
vmax = max(np.nanmax(before.d), np.nanmax(after.d))
half_step = windowed_signals.sample_period / 2
extent = (
    before.t[0] - half_step,
    before.t[-1] + half_step,
    len(before) - 0.5,
    -0.5,
)
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4), sharex=True, sharey=True,
    layout="constrained",
)
for ax, data, title in zip(
    axes, (before, after), ("Before convolving", "After convolving")
):
    image = ax.imshow(
        data.d,
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set(title=title, xlabel="Aligned time (s)")
    ax.set_yticks(np.arange(len(data)), labels=data.trial_ids)
axes[0].set_ylabel("Trial ID")
fig.suptitle(f"Signal = {feature}")
fig.colorbar(image, ax=axes, label="Signal")
plt.show()
```

Each trial and signal is processed independently. The output keeps the
original time axis and metadata.

Both `convolve` and `smooth` zero-pad outside each trial's valid samples.
No edge correction is applied, so values near trial boundaries may be
attenuated. Invalid samples remain `NaN`.

### Normalizing
Use `normalize` to normalize each signal across trials and time.
The `axes` argument specifies which dimensions to normalize over: axis `0` is trials,
and axis `1` is time.

```{code-cell} ipython3
normalized = smoothed.normalize(axes=(0, 1))
normalized
```

```{code-cell} ipython3
:tags: [hide-input]
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4), sharex=True, layout="constrained"
)

for ax, data, title, ylabel in zip(
    axes,
    (smoothed, normalized),
    ("Before normalizing", "After normalizing"),
    ("Signal value", "Z-score"),
):
    trial_means = np.nanmean(data.d, axis=1)

    for feature_position, feature in enumerate(data.columns):
        ax.plot(
            data.trial_ids,
            trial_means[:, feature_position],
            "o-",
            label=feature,
        )

    ax.set(
        title=title,
        xlabel="Trial ID",
        ylabel=ylabel,
        xticks=data.trial_ids,
    )
    ax.legend()
plt.show()
```
Before normalization, the signals have different baselines and scales.
After normalization, their trial means are expressed in standard-deviation units.
Between-trial differences remain because each signal uses the same normalization parameters for every trial.

### Using NumPy

Elementwise operations preserve the trial structure and metadata:

```{code-cell} ipython3
scaled = signal_trials * 2
scaled
```

```{code-cell} ipython3
absolute = np.abs(signal_trials)
absolute
```

Supported NumPy reductions return an object appropriate for the remaining
dimensions. Averaging across trials leaves time and signals, so the result
is a [`TsdFrame`](pynapple.TsdFrame):

```{code-cell} ipython3
mean_signal = np.nanmean(signal_trials, axis=0)
mean_signal
```

Averaging across signals leaves trials and time, so the result is a
[`TsdTrials`](pynapple.TsdTrials):

```{code-cell} ipython3
population_signal = np.nanmean(signal_trials, axis=2)
population_signal
```

Reducing time returns a NumPy array because there is no longer a time axis:

```{code-cell} ipython3
trial_means = np.nanmean(signal_trials, axis=1)
trial_means
```

Other supported reductions include `sum`, `std`, `var`, `min`, `max`,
`median`, and `prod`, together with their NaN-aware variants, plus `all`
and `any`.

Trials with different durations contain `NaN` padding. Use NaN-aware
reductions, such as `np.nanmean`, to exclude that padding.
Ordinary reductions such as `np.mean` propagate NaNs.

To use a NumPy function without preserving the trial structure, convert
the object explicitly:

```{code-cell} ipython3
array = np.asarray(signal_trials)
array.shape
```

## Grouping trials

Both trial classes support grouping by a trial metadata column.
`groupby` returns a dictionary containing a trial object for each group:

```{code-cell} ipython3
groups = signal_trials.groupby("target")
groups["left"]
```

The selected trials retain their time coordinates, values, and metadata.
The same operation works for timestamps:

```{code-cell} ipython3
spike_groups = spike_trials.groupby("target")
spike_groups["left"]
```

### Applying a function to each group

Use `groupby_apply` to apply a function separately to each group.
For example, average the signals across trials with the same target:

```{code-cell} ipython3
target_means = signal_trials.groupby_apply(
    "target",
    np.nanmean,
    axis=0,
)
target_means["left"]
```

The result is a dictionary with one entry per target. Each entry is a
[`TsdFrame`](pynapple.TsdFrame), since averaging across trials leaves the
time and signal dimensions.
