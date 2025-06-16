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

# Core methods

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False, "figure.figsize": (8, 4)}
sns.set_context("paper") 
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.3, rc=custom_params)
```
## Time series methods

```{code-cell} ipython3
:tags: [hide-cell]
tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 3), columns=['a', 'b', 'c'])
group = {
    0: nap.Ts(t=np.sort(np.random.uniform(0, 100, 10))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30))),
}
tsgroup = nap.TsGroup(group, time_support = nap.IntervalSet(0, 100))
epochs = nap.IntervalSet([10, 65], [25, 80])
tsd = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1)))
```

### `restrict`

[`restrict`](pynapple.Tsd.restrict) is used to get time points within an `IntervalSet`. This method is available for `TsGroup`, `Tsd`, `TsdFrame`, `TsdTensor` and `Ts` objects.

```{code-cell} ipython3
tsdframe.restrict(epochs) 
```
```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(tsdframe.restrict(epochs))
[plt.axvspan(s, e, alpha=0.2) for s, e in epochs.values]
plt.xlabel("Time (s)")
plt.title("tsdframe.restrict(epochs)")
plt.xlim(0, 100)
plt.show()
```

This operation update the time support attribute accordingly. 

```{code-cell} ipython3
print(epochs)
print(tsdframe.restrict(epochs).time_support) 
```

### `count`

[`count`](pynapple.Tsd.count) returns the number of timestamps within bins or epochs of an `IntervalSet` object.
This method is available for `TsGroup`, `Tsd`, `TsdFrame`, `TsdTensor` and `Ts` objects.

With a defined bin size:
```{code-cell} ipython3
count = tsgroup.count(bin_size=1.0, time_units='s')
print(count) 
```
```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.step(count.t, count[:,2], where='mid', label="count[:,2]")
plt.title("tsgroup.count(bin_size=1.0)")
plt.plot(tsgroup[2].fillna(-0.5), '|', markeredgewidth=3, label="tsgroup[2]")
[plt.axvline(t, linewidth=0.5, alpha=0.5) for t in np.arange(0, 31)]
plt.xlabel("Time (s)")
plt.xlim(0, 30)
plt.legend()
plt.show()
```

With an `IntervalSet`:
```{code-cell} ipython3
count_ep = tsgroup.count(ep=epochs)

print(count_ep)
```

### `trial_count`

`TsGroup` and `Ts` objects each have the method `trial_count`, which builds a trial-based count tensor from an `IntervalSet` object. 
Similar to `count`, this function requires a `bin_size` parameter which determines the number of time bins within each trial. 
The resulting tensor has shape (number of group elements, number of trials, number of time bins) for `TsGroup` objects, 
or (number of trials, number of time bins) for `Ts` objects. 

```{code-cell} ipython3
ep = nap.IntervalSet([5, 17, 30, 50], metadata={'trials':[1, 2]})
tensor = tsgroup.trial_count(ep, bin_size=2)
print(tensor, "\n")
print("Tensor shape = ", tensor.shape)
```

```{code-cell} ipython3
:tags: [hide-input]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
from matplotlib.colors import LinearSegmentedColormap
plt.figure()
gs = plt.GridSpec(3,2)
plt.subplot(gs[:,0])
for i, n in enumerate(tsgroup.keys()):
    plt.plot(tsgroup[n].fillna(i+1), '|', markeredgewidth=3, color=color_cycle[i])
for i in range(len(ep)):
    plt.axvspan(ep[i,0], ep[i,1], alpha=0.2)
    [plt.axvline(t, linewidth=0.5, alpha=0.5) for t in np.arange(ep[i,0], ep[i,1], 2.0)]
plt.title("tsgroup")
plt.ylim(0, len(tsgroup)+1)
plt.xlim(0, 60)
plt.xlabel("Time (s)")

for i in range(3):
    plt.subplot(gs[2-i,1])
    cmap = LinearSegmentedColormap.from_list("fade", ["lightgrey", color_cycle[i]])
    plt.pcolormesh(np.arange(0, tensor.shape[-1]), [1, 2], tensor[i], cmap=cmap)
    if i == 1: plt.ylabel("Trials")    
    if i == 2: plt.title("tsgroup.trial_cout(ep, bin_size=2)")
    if i == 0: plt.xlabel("Trial time")
    plt.text(1, 0.5, f"tensor[{i}]", transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()
```


The array is padded with NaNs when the trials have uneven durations, 
The padding value can be controlled using the parameter `padding_value`. 
Additionally, the parameter `align` can change whether the count is aligned to the "start" or "end" of each trial.

```{code-cell} ipython3
tensor = tsgroup.trial_count(ep, bin_size=2, align="end", padding_value=-1)
print(tensor, "\n")
print("Tensor shape = ", tensor.shape)
```

### `bin_average`

[`bin_average`](pynapple.Tsd.bin_average) downsamples time series by averaging data point falling within a bin. This method is available for `Tsd`, `TsdFrame` and `TsdTensor`. While `bin_average` is good for downsampling with precise control of the resulting bins, it does not apply any antialiasing filter. The function [`decimate`](pynapple.Tsd.decimate) is also available for down-sampling without aliasing.

```{code-cell} ipython3
tsdframe.bin_average(3.5)
```
```{code-cell} ipython3
:tags: [hide-input]
bin_size = 3.5
plt.figure()
plt.plot(tsdframe[:,0], '.--', label="tsdframe[:,0]")
plt.plot(tsdframe[:,0].bin_average(bin_size), 'o-', label="new_tsdframe[:,0]")
plt.title(f"tsdframe.bin_average(bin_size={bin_size})")
[plt.axvline(t, linewidth=0.5, alpha=0.5) for t in np.arange(0, 21,bin_size)]
plt.xlabel("Time (s)")
plt.xlim(0, 20)
plt.legend(bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
plt.show()
```


### `decimate`

The [`decimate`](pynapple.Tsd.decimate) method downsamples the time series by an integer factor after an antialiasing filter.

```{code-cell} ipython3
:tags: [hide-input]
noisy_data = np.random.rand(100) + np.sin(np.linspace(0, 2 * np.pi, 100))
tsd = nap.Tsd(t=np.arange(100), d=noisy_data, time_support=nap.IntervalSet(0, 100))
```

```{code-cell} ipython3
new_tsd = tsd.decimate(down=4)
```

The original time series was sampled at 1Hz. The new time series has a rate of 0.25 Hz. 

```{code-cell} ipython3
print(f"Original rate : {tsd.rate}")
print(f"New rate : {new_tsd.rate}") 
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(tsd, label="original")
plt.plot(new_tsd, marker="o", label="decimate")
plt.plot(tsd[::4], marker="o", label="naive downsample")
plt.legend()
plt.show()
```


### `interpolate`

The [`interpolate`](pynapple.Tsd.interpolate) method of `Tsd`, `TsdFrame` and `TsdTensor` can be used to fill gaps in a time series. It is a wrapper of [`numpy.interp`](https://numpy.org/devdocs/reference/generated/numpy.interp.html).



```{code-cell} ipython3
:tags: [hide-cell]
tsd = nap.Tsd(t=np.arange(0, 25, 5), d=np.random.randn(5))
ts = nap.Ts(t=np.arange(0, 21, 1))
```

```{code-cell} ipython3
new_tsd = tsd.interpolate(ts)
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(new_tsd, '.-', label="new_tsd")
plt.plot(tsd, 'o', label="tsd")
plt.plot(ts.fillna(0), '+', label="ts")
plt.title("tsd.interpolate(ts)")
plt.xlabel("Time (s)")
plt.legend(bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
plt.show()
```



### `value_from`

By default, [`value_from`](pynapple.Tsd.value_from) assigns to timestamps the closest value in time from another time series. Let's define the time series we want to assign values from.

For every timestamps in `tsgroup`, we want to assign the closest value in time from `tsd`.

```{code-cell} ipython3
:tags: [hide-cell]
tsd = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1)))
```

```{code-cell} ipython3
tsgroup_from_tsd = tsgroup.value_from(tsd)
```

We can display the first element of `tsgroup` and `tsgroup_sin`.

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(tsgroup[0].fillna(0), "|", label="tsgroup[0]", markersize=20, mew=3)
plt.plot(tsd, linewidth=2, label="tsd")
plt.plot(tsgroup_from_tsd[0], "o", label = "tsgroup_from_tsd[0]", markersize=20)
plt.title("tsgroup.value_from(tsd)")
plt.xlabel("Time (s)")
plt.yticks([-1, 0, 1])
plt.legend(bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
plt.show()
```

The argument `mode` can control if the nearest target time is taken before or 
after the reference time.

```{code-cell} ipython3
:tags: [hide-cell]
tsd = nap.Tsd(t=np.arange(0, 10, 1), d=np.arange(0, 100, 10))
ts = nap.Ts(t=np.arange(0.5, 9, 1))
```

In this case, the variable `ts` receive data from the time point before.

```{code-cell} ipython3
new_ts_before = ts.value_from(tsd, mode="before")
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(ts.fillna(-1), "|", label="ts", markersize=20, mew=3)
plt.plot(tsd, "*-", linewidth=2, label="tsd")
plt.plot(new_ts_before, "o-", label = "new_ts_before", markersize=10)
plt.title("ts.value_from(tsd, mode='before')")
plt.xlabel("Time (s)")
plt.legend(bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
plt.show()
```
```{code-cell} ipython3
:tags: [hide-input]
new_ts_after = ts.value_from(tsd, mode="after")
plt.figure()
plt.plot(ts.fillna(-1), "|", label="ts", markersize=20, mew=3)
plt.plot(tsd, "*-", linewidth=2, label="tsd")
plt.plot(new_ts_after, "o-", label = "new_ts_after", markersize=10)
plt.title("ts.value_from(tsd, mode='after')")
plt.xlabel("Time (s)")
plt.legend(bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
plt.show()
```

If there is no time point found before or after or within the interval, the function assigns
Nans.

```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(1, 10, 1), d=np.arange(10, 100, 10))
ep = nap.IntervalSet(start=0, end = 10)
ts = nap.Ts(t=[0, 9])

# First ts is at 0s. First tsd is at 1s.
ts.value_from(tsd, ep=ep, mode="before")
```



### `threshold`

The method [`threshold`](pynapple.Tsd.threshold) of `Tsd` returns a new `Tsd` with all the data above or below a certain threshold. Default is `above`. The time support of the new `Tsd` object get updated accordingly.

```{code-cell} ipython3
:tags: [hide-cell]
tsd = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1)))
```

```{code-cell} ipython3
tsd_above = tsd.threshold(0.5, method='above')
```
This method can be used to isolate epochs for which a signal
is above/below a certain threshold.

```{code-cell} ipython3
epoch_above = tsd_above.time_support
```
```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(tsd, label="tsd")
plt.plot(tsd_above, 'o-', label="tsd_above")
[plt.axvspan(s, e, alpha=0.2) for s, e in epoch_above.values]
plt.axhline(0.5, linewidth=0.5, color='grey')
plt.legend()
plt.xlabel("Time (s)")
plt.title("tsd.threshold(0.5)")
plt.show()
```

### `derivative`

The [`derivative`](pynapple.Tsd.derivative) method of `Tsd`, `TsdFrame` and `TsdTensor` can be used to calculate the derivative of a time series with respect to time. It is a wrapper of [`numpy.gradient`](https://numpy.org/devdocs/reference/generated/numpy.gradient.html).


```{code-cell} ipython3
:tags: [hide-cell]
tsd = nap.Tsd(
    t=np.arange(0, 10, 0.1),
    d=np.sin(np.arange(0, 10, 0.1)),
)
ep = nap.IntervalSet(start=[0, 6], end=[4, 10])
```

```{code-cell} ipython3
derivative = tsd.derivative(ep=ep)
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(tsd, label="tsd")
plt.plot(derivative, 'o-', label="derivative")
[plt.axvspan(s, e, alpha=0.2) for s, e in derivative.time_support.values]
plt.axhline(0, linewidth=0.5, color='grey')
plt.legend(loc="lower right")
plt.xlabel("Time (s)")
plt.title("tsd.derivative()")
plt.show()
```


### `to_trial_tensor`

`Tsd`, `TsdFrame`, and `TsdTensor` all have the method [`to_trial_tensor`](pynapple.Tsd.to_trial_tensor), which creates a numpy array from an `IntervalSet` by slicing the time series. The resulting tensor has shape (shape of time series, number of trials, number of time points), where the first dimension(s) is dependent on the object. 

```{code-cell} ipython3
tsd = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1))) 
ep = nap.IntervalSet([0, 10, 30, 50, 70, 75], metadata={'trials':[1, 2, 3]})
print(ep)
```

The following example returns a tensor with shape (3, 21), for 3 trials and 21 time points, where the first dimension is dropped due to this being a `Tsd` object.

```{code-cell} ipython3
tensor = tsd.to_trial_tensor(ep)
print(tensor, "\n")
print("Tensor shape = ", tensor.shape)
```

```{code-cell} ipython3
:tags: [hide-input]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
plt.subplot(121)
plt.plot(tsd, '-', label="tsd")
for i in range(len(ep)):
    plt.plot(tsd.get(ep[i,0], ep[i,1]), 'o-', color=color_cycle[i]) 
    plt.axvspan(ep[i,0], ep[i,1], alpha=0.2)
plt.legend(loc="lower right")
plt.xlabel("Time (s)")
plt.subplot(122)
plt.plot(tensor.T, 'o-')
plt.title("tsd.to_trial_tensor(ep)")
plt.tight_layout()
plt.xlabel("Trial time")
plt.show()
```


Since trial 2 is twice as long as trial 1, the array is padded with NaNs. The padding value can be changed by setting the parameter `padding_value`.

```{code-cell} ipython3
tensor = tsd.to_trial_tensor(ep, padding_value=-1)
print(tensor, "\n")
print("Tensor shape = ", tensor.shape)
```

By default, time series are aligned to the start of each trial. To align the time series to the end of each trial, the optional parameter `align` can be set to "end".

```{code-cell} ipython3
tensor = tsd.to_trial_tensor(ep, align="end")
print(tensor, "\n")
print("Tensor shape = ", tensor.shape)
```

```{code-cell} ipython3
:tags: [hide-input]
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
plt.subplot(121)
plt.plot(tsd, '-', label="tsd")
for i in range(len(ep)):
    plt.plot(tsd.get(ep[i,0], ep[i,1]), 'o-', color=color_cycle[i]) 
    plt.axvspan(ep[i,0], ep[i,1], alpha=0.2)
plt.legend(loc="lower right")
plt.xlabel("Time (s)")
plt.subplot(122)
plt.plot(tensor.T, 'o-')
plt.title(r"tsd.to_trial_tensor(ep, align='end')")
plt.tight_layout()
plt.xlabel("Trial time")
plt.show()
```

### `time_diff`

`Ts`, `Tsd`, `TsdFrame`, `TsdTensor`, and `TsGroup` all have the method `time_diff`, which computes the time differences between subsequent timepoints. 
For example, if a `Ts` object contained a set of spike times, `time_diff` would compute the inter-spike interval (ISI).

This method returns a new `Tsd` object, with values being each time difference, and time indices being their reference time point. 
Passing `epochs` restricts the computation to the given epochs.
The reference time point can be adjusted by the optional `align` parameter, which can be set to  `"start"`, `"center"`, or `"end"` (the default being `"center"`).

```{code-cell} ipython3
:tags: [hide-input]
ts = nap.Ts(t=[1,5,6,12,16,18,19])
```

```{code-cell} ipython3
time_diffs = ts.time_diff(align="center")
print(time_diffs)
```
Setting `align="center"` sets the reference time point to the midpoint between the timestamps used to calculate the time difference.
Setting `align="start"` or `align="end"` sets the reference time point to the earlier or later timestamp, respectively.
```{code-cell} ipython3
:tags: [hide-input]
fig, axs = plt.subplots(3, 1, layout="constrained", figsize=(5,6))
for ax, align in zip(axs, ["center", "start", "end"]):
    time_diffs = ts.time_diff(align=align)
    ax.plot(ts.fillna(0), "|", label="ts", markersize=20, mew=3)
    ax.plot(time_diffs, "o-", label="new_tsd")
    ax.set_ylabel("Time diffs (s)")
    ax.set_title(f'ts.time_diff(align="{align}")')
    if align != "end":
        ax.set_xticks([])
    for center, time_diff in zip(time_diffs.times(), time_diffs.values):
        ax.plot([center, center], [-.25, time_diff], linestyle="--", c="black", zorder=-1)
ax.set_xlabel("Time (s)")
axs[0].legend(bbox_to_anchor=(1.05, 0.5, 0.5, 0.5))
plt.show()
```

### Mapping between `TsGroup` and `Tsd`

It's is possible to transform a `TsGroup` to `Tsd` with the method
[`to_tsd`](pynapple.TsGroup.to_tsd) and a `Tsd` to `TsGroup` with the method [`to_tsgroup`](pynapple.Tsd.to_tsgroup).

This is useful to flatten the activity of a population in a single array.

```{code-cell} ipython3
tsd = tsgroup.to_tsd()

print(tsd)
```
The object `tsd` contains all the timestamps of the `tsgroup` with
the associated value being the index of the unit in the `TsGroup`.

The method [`to_tsgroup`](pynapple.Tsd.to_tsgroup) converts the `Tsd` object back to the original `TsGroup`. 

```{code-cell} ipython3
back_to_tsgroup = tsd.to_tsgroup()

print(back_to_tsgroup)
```

#### Parameterizing a raster

The method `to_tsd` makes it easier to display a raster plot.
`TsGroup` object can be plotted with `plt.plot(tsgroup.to_tsd(), 'o')`.
Timestamps can be mapped to any values passed directly to the method
or by giving the name of a specific metadata name of the `TsGroup`.

```{code-cell} ipython3
tsgroup['label'] = np.arange(3)*np.pi

print(tsgroup)
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.subplot(2,2,1)
plt.plot(tsgroup.to_tsd(), '|')
plt.title("tsgroup.to_tsd()")
plt.xlabel("Time (s)")

plt.subplot(2,2,2)
plt.plot(tsgroup.to_tsd([10,20,30]), '|')
plt.title("togroup.to_tsd([10,20,30])")
plt.xlabel("Time (s)")

plt.subplot(2,2,3)
plt.plot(tsgroup.to_tsd("label"), '|')
plt.title("togroup.to_tsd('label')")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
```

### Special slicing: TsdFrame

For users that are familiar with pandas, [`TsdFrame`](pynapple.TsdFrame) is the closest object to a [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), but there are distinctive behavior when slicing the object. `TsdFrame` behaves primarily like a numpy array. This section lists all the possible ways of slicing `TsdFrame`.

#### 1. If not column labels are passed 

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,3))
print(tsdframe)
```

Slicing should be done like numpy array:

```{code-cell} ipython3
tsdframe[0]
```

```{code-cell} ipython3
tsdframe[:, 1]
```

```{code-cell} ipython3
tsdframe[:, [0, 2]]
```

#### 2. If column labels are passed as integers

The typical case is channel mapping. The order of the columns on disk are different from the order of the columns on the
recording device it corresponds to. 

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,4), columns = [3, 2, 0, 1])
print(tsdframe)
```
In this case, indexing like numpy still has priority which can led to confusing behavior:

```{code-cell} ipython3
tsdframe[:, [0, 2]]
```
Note how this corresponds to column labels 3 and 0.  

To slice using column labels only, the `TsdFrame` object has the [`loc`](pynapple.TsdFrame.loc) method similar to Pandas:

```{code-cell} ipython3
tsdframe.loc[[0, 2]]
```
In this case, this corresponds to columns labelled 0 and 2.

#### 3. If column labels are passed as strings

Similar to Pandas, it is possible to label columns using strings.

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,3), columns = ["kiwi", "banana", "tomato"])
print(tsdframe)
```

When the column labels are all strings, it is possible to use either direct bracket indexing or using the [`loc`](pynapple.TsdFrame.loc) method:

```{code-cell} ipython3
print(tsdframe['kiwi'])
print(tsdframe.loc['kiwi']) 
```

#### 4. If column labels are mixed type

It is possible to mix types in column names.

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,3), columns = ["kiwi", 0, np.pi])
print(tsdframe)
```

Direct bracket indexing only works if the column label is a string.

```{code-cell} ipython3
print(tsdframe['kiwi'])
```

To slice with mixed types, it is best to use the [`loc`](pynapple.TsdFrame.loc) method:

```{code-cell} ipython3
print(tsdframe.loc[['kiwi', np.pi]])
```

In general, it is probably a bad idea to mix types when labelling columns.


## Interval sets methods
 
### Interaction between epochs 

Intervals can be combined in different ways. 

```{code-cell} ipython3
epoch1 = nap.IntervalSet(start=[0, 40], end=[10, 50])  # no time units passed. Default is us.
epoch2 = nap.IntervalSet(start=[5, 30], end=[20, 45])
print(epoch1, "\n")
print(epoch2, "\n")
```

#### [`union`](pynapple.IntervalSet.union)

```{code-cell} ipython3
epoch = epoch1.union(epoch2)
print(epoch)
```

```{code-cell} ipython3
:tags: [hide-input]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
[plt.axvspan(s, e, ymin=0.8, ymax=1, color=colors[0]) for s, e in epoch1.values]
[plt.axvspan(s, e, ymin=0.4, ymax=0.6, color=colors[1]) for s, e in epoch2.values]
[plt.axvspan(s, e, ymin=0, ymax=0.2, color=colors[2]) for s, e in epoch.values]
plt.xlabel("Time (s)")
plt.ylim(0, 1)
plt.xlim(0, 50)
plt.gca().spines["left"].set_visible(False)
plt.yticks([0.1, 0.5, 0.9], ['epoch1.union(epoch2)', 'epoch2', 'epoch1'])
plt.title("Union")
plt.show()
```



#### [`intersect`](pynapple.IntervalSet.intersect)

```{code-cell} ipython3
epoch = epoch1.intersect(epoch2)
print(epoch)
```

```{code-cell} ipython3
:tags: [hide-input]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
[plt.axvspan(s, e, ymin=0.8, ymax=1, color=colors[0]) for s, e in epoch1.values]
[plt.axvspan(s, e, ymin=0.4, ymax=0.6, color=colors[1]) for s, e in epoch2.values]
[plt.axvspan(s, e, ymin=0, ymax=0.2, color=colors[2]) for s, e in epoch.values]
plt.xlabel("Time (s)")
plt.ylim(0, 1)
plt.xlim(0, 50)
plt.gca().spines["left"].set_visible(False)
plt.yticks([0.1, 0.5, 0.9], ['epoch1.intersect(epoch2)', 'epoch2', 'epoch1'])
plt.title("Intersection")
plt.show()
```

#### [`set_diff`](pynapple.IntervalSet.set_diff)

```{code-cell} ipython3
epoch = epoch1.set_diff(epoch2)
print(epoch)
```

```{code-cell} ipython3
:tags: [hide-input]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
[plt.axvspan(s, e, ymin=0.8, ymax=1, color=colors[0]) for s, e in epoch1.values]
[plt.axvspan(s, e, ymin=0.4, ymax=0.6, color=colors[1]) for s, e in epoch2.values]
[plt.axvspan(s, e, ymin=0, ymax=0.2, color=colors[2]) for s, e in epoch.values]
plt.xlabel("Time (s)")
plt.ylim(0, 1)
plt.xlim(0, 50)
plt.gca().spines["left"].set_visible(False)
plt.yticks([0.1, 0.5, 0.9], ['epoch1.set_diff(epoch2)', 'epoch2', 'epoch1'])
plt.title("Difference")
plt.show()
```

### [`split`](pynapple.IntervalSet.split)

Useful for chunking time series, the [`split`](pynapple.IntervalSet.split) method splits an `IntervalSet` in a new `IntervalSet` based on the `interval_size` argument.

```{code-cell} ipython3
epoch = nap.IntervalSet(start=0, end=100)

print(epoch.split(10, time_units="s"))
```

### Drop intervals

```{code-cell} ipython3
epoch = nap.IntervalSet(start=[5, 30], end=[6, 45])
print(epoch)
```

#### [`drop_short_intervals`](pynapple.IntervalSet.drop_short_intervals)

```{code-cell} ipython3
print(
    epoch.drop_short_intervals(threshold=5)
    )
```

#### [`drop_long_intervals`](pynapple.IntervalSet.drop_long_intervals)

```{code-cell} ipython3
print(
    epoch.drop_long_intervals(threshold=5)
    )
```

#### [`merge_close_intervals`](pynapple.IntervalSet.merge_close_intervals)

```{code-cell} ipython3
:tags: [hide-input]
epoch = nap.IntervalSet(start=[1, 7], end=[6, 45])
print(epoch)
```
If two intervals are closer than the `threshold` argument, they are merged.

```{code-cell} ipython3
print(
    epoch.merge_close_intervals(threshold=2.0)
    )
```


