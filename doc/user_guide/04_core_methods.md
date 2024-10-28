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
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

## Interval sets methods

### Interaction between epochs 

```{code-cell} ipython3
epoch1 = nap.IntervalSet(start=0, end=10)  # no time units passed. Default is us.
epoch2 = nap.IntervalSet(start=[5, 30], end=[20, 45])
print(epoch1, "\n")
print(epoch2, "\n")
```

#### `union`

```{code-cell} ipython3
epoch = epoch1.union(epoch2)
print(epoch)
```

#### `intersect`

```{code-cell} ipython3
epoch = epoch1.intersect(epoch2)
print(epoch)
```

#### `set_diff`

```{code-cell} ipython3
epoch = epoch1.set_diff(epoch2)
print(epoch)
```

### `split`

Useful for chunking time series, the `split` method splits an `IntervalSet` in a new
`IntervalSet` based on the `interval_size` argument.

```{code-cell} ipython3
epoch = nap.IntervalSet(start=0, end=100)

print(epoch.split(10, time_units="s"))
```

### Drop intervals

```{code-cell} ipython3
epoch = nap.IntervalSet(start=[5, 30], end=[6, 45])
print(epoch)
```

#### `drop_short_intervals`

```{code-cell} ipython3
print(
    epoch.drop_short_intervals(threshold=5)
    )
```

#### `drop_long_intervals`

```{code-cell} ipython3
print(
    epoch.drop_long_intervals(threshold=5)
    )
```

### `merge_close_intervals`

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


## Metadata

One advantage of grouping time series is that metainformation 
can be added directly on an element-wise basis. 
In this case, we add labels to each Ts object when instantiating the group and after. 
We can then use this label to split the group. 
See the TsGroup documentation for a complete methodology for splitting TsGroup objects.

```{code-cell} ipython3
group = {
    0: nap.Ts(t=np.sort(np.random.uniform(0, 100, 10))),
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30))),
}
time_support = nap.IntervalSet(0, 100)

tsgroup = nap.TsGroup(group, time_support=time_support)

print(tsgroup, "\n")

```

## Time series method

```{code-cell} ipython3
:tags: [hide-cell]
tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 3), columns=['a', 'b', 'c'])
epochs = nap.IntervalSet([10, 65], [25, 80])
tsd = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1)))
tsd += np.random.randn(len(tsd))
```




### `restrict`

`restrict` is used to get time points within an `IntervalSet`. This method is available 
for `TsGroup`, `Tsd`, `TsdFrame`, `TsdTensor` and `Ts` objects.

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

`count` the number of timestamps within bins or epochs of an `IntervalSet` object.
This method is available for `TsGroup`, `Tsd`, `TsdFrame`, `TsdTensor` and `Ts` objects.

With a defined bin size:
```{code-cell} ipython3
count1 = tsgroup.count(bin_size=1.0, time_units='s')
print(count1) 
```
```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(count1[:,2], 'o-')
plt.title("tsgroup.count(bin_size=1.0)")
plt.plot(tsgroup[2].fillna(-1), '|', markeredgewidth=2)
[plt.axvline(t, linewidth=0.5, alpha=0.5) for t in np.arange(0, 21)]
plt.xlabel("Time (s)")
plt.xlim(0, 20)
plt.show()
```

With an `IntervalSet`:
```{code-cell} ipython3
count_ep = tsgroup.count(ep=epochs)

print(count_ep)
```

### `bin_average`

`bin_average` downsample time series by averaging data point falling within a bin.
This method is available for `Tsd`, `TsdFrame` and `TsdTensor`.

```{code-cell} ipython3
tsdframe.bin_average(3.5)
```
```{code-cell} ipython3
:tags: [hide-input]
bin_size = 3.5
plt.figure()
plt.plot(tsdframe[:,0], '.--')
plt.plot(tsdframe[:,0].bin_average(bin_size), 'o-')
plt.title(f"tsdframe.bin_average(bin_size={bin_size})")
[plt.axvline(t, linewidth=0.5, alpha=0.5) for t in np.arange(0, 21,bin_size)]
plt.xlabel("Time (s)")
plt.xlim(0, 20)
plt.show()
```



### `interpolate`

### `value_from`

`value_from` assign to every timestamps the closed value in time from another time series. Let's define the time series we want to assign values from.


For every timestamps in `tsgroup`, we want to assign the closest value in time from `tsd_sin`.

```{code-cell} ipython3
tsgroup_sin = tsgroup.value_from(tsd_sin)
```

We can display the first element of `tsgroup` and `tsgroup_sin`.

```{code-cell} ipython3
plt.figure()
plt.plot(tsgroup[0].fillna(0), "|", label="tsgroup[0]", markersize=20, mew=3)
plt.plot(tsd_sin, linewidth=2, label="tsd_sin")
plt.plot(tsgroup_sin[0], "o", label = "tsgroup_sin[0]", markersize=20)
plt.title("tsgroup.value_from(tsd)")
plt.xlabel("Time (s)")
plt.yticks([-1, 0, 1])
plt.legend()
plt.show()
```

### `threshold`

The method `threshold` of `Tsd` returns a new `Tsd` with all the data above or 
below a certain threshold. Default is `above`. The time support
of the new `Tsd` object get updated accordingly.

```{code-cell} ipython3
tsd_above = tsd_sin.threshold(0.5, method='above')
```
This method can be used to isolate epochs for which a signal
is above/below a certain threshold.

```{code-cell} ipython3
epoch_above = tsd_above.time_support
```
```{code-cell} ipython3
:tags: [hide-input]
plt.figure()
plt.plot(tsd_sin, label="tsd_sin")
plt.plot(tsd_above, 'o-', label="tsd_above")
[plt.axvspan(s, e, alpha=0.2) for s, e in epoch_above.values]
plt.axhline(0.5, linewidth=0.5, color='grey')
plt.legend()
plt.xlabel("Time (s)")
plt.title("tsd_sin.threshold(0.5)")
plt.show()
```


## Mapping between `TsGroup` and `Tsd`

It's is possible to transform a `TsGroup` to `Tsd` with the method
`to_tsd` and a `Tsd` to `TsGroup` with the method `to_tsgroup`.

This is useful to flatten the activity of a population in a single array.

```{code-cell} ipython3
tsd = tsgroup.to_tsd()

print(tsd)
```
The object `tsd` contains all the timestamps of the `tsgroup` with
the associated value being the index of the unit in the `TsGroup`.

The method `to_tsgroup` converts the `Tsd` object back to the original `TsGroup`. 

```{code-cell} ipython3
back_to_tsgroup = tsd.to_tsgroup()

print(back_to_tsgroup)
```

### Parameterizing a raster

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
