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
## Time series method

```{code-cell} ipython3
:tags: [hide-cell]
tsdframe = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 3), columns=['a', 'b', 'c'])
epochs = nap.IntervalSet([10, 65], [25, 80])
tsd = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1)))
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
plt.plot(tsdframe[:,0], '.--', label="tsdframe[:,0]")
plt.plot(tsdframe[:,0].bin_average(bin_size), 'o-', label="new_tsdframe[:,0]")
plt.title(f"tsdframe.bin_average(bin_size={bin_size})")
[plt.axvline(t, linewidth=0.5, alpha=0.5) for t in np.arange(0, 21,bin_size)]
plt.xlabel("Time (s)")
plt.xlim(0, 20)
plt.legend(bbox_to_anchor=(1.0, 0.5, 0.5, 0.5))
plt.show()
```



### `interpolate`

The`interpolate` method of `Tsd`, `TsdFrame` and `TsdTensor` can be used to fill gaps in a time series. It is a wrapper of [`numpy.interp`](https://numpy.org/devdocs/reference/generated/numpy.interp.html).



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

`value_from` assign to every timestamps the closed value in time from another time series. Let's define the time series we want to assign values from.


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

### `threshold`

The method `threshold` of `Tsd` returns a new `Tsd` with all the data above or 
below a certain threshold. Default is `above`. The time support
of the new `Tsd` object get updated accordingly.

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


### Mapping between `TsGroup` and `Tsd`

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
Metadata can be added to `TsGroup`, `IntervalSet`, and `TsdFrame` objects at initialization or after an object has been created.
- `TsGroup` metadata is information associated with each Ts/Tsd object, such as brain region or unit type.
- `IntervalSet` metadata is information assocaited with each interval, such as a trial label or stimulus condition.
- `TsdFrame` metadata is information associated with each column, such as a channel or position.


### Adding metadata
At initialization, metadata can be passed via a dictionary or pandas DataFrame using the keyword argument `metadata`. The metadata name is taken from the dictionary key or DataFrame column, and it can be set to any string name with a couple class-specific exceptions. 

```{admonition} Class-specific exceptions
- If column names are supplied to `TsdFrame`, metadata cannot overlap with those names.
- The `rate` attribute for `TsGroup` is stored with the metadata and cannot be overwritten.
```

The length of the metadata must match the length of the object it describes (see class examples below for more detail). 

```{code-cell} ipython3
:tags: [hide-cell]
import numpy as np
import pandas as pd
import pynapple as nap

# input parameters for TsGroup
group = {
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 10))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 20))),
    3: nap.Ts(t=np.sort(np.random.uniform(0, 100, 30))),
}

# input parameters for IntervalSet
starts = [0,10,20]
ends = [5,15,25]

# input parameters for TsdFrame
t = np.arange(5)
d = np.ones((5,3))
columns = ["a", "b", "c"]
```

#### `TsGroup`
Metadata added to `TsGroup` must match the number of `Ts`/`Tsd` objects, or the length of its `index` property.
```{code-cell} ipython3
metadata = {"region": ["pfc", "ofc", "hpc"]}

tsgroup = nap.TsGroup(group, metadata=metadata)
print(tsgroup)
```

When initializing with a DataFrame, the index must align with the input dictionary keys (only when a dictionary is used to create the `TsGroup`).
```{code-cell} ipython3
metadata = pd.DataFrame(
    index=group.keys(),
    data=["pfc", "ofc", "hpc"],
    columns=["region"]
)

tsgroup = nap.TsGroup(group, metadata=metadata)
print(tsgroup)
```


#### `IntervalSet`
Metadata added to `IntervalSet` must match the number of intervals, or the length of its `index` property. 

```{code-cell} ipython3
metadata = {
    "reward": [1, 0, 1],
    "choice": ["left", "right", "left"],    
}

intervalset = nap.IntervalSet(starts, ends, metadata=metadata)
print(intervalset)
```

Metadata can be initialized as a DataFrame using the metadata argument, or it can be inferred when initializing an `IntervalSet` with a DataFrame.
```{code-cell} ipython3
df = pd.DataFrame(
    data=[[0, 5, 1, "left"], [10, 15, 0, "right"], [20, 25, 1, "left"]], 
    columns=["start", "end", "reward", "choice"]
    )

intervalset = nap.IntervalSet(df)
print(intervalset)
```

#### `TsdFrame`
Metadata added to `TsdFrame` must match the number of data columns, or the length of its `columns` property. 
```{code-cell} ipython3
metadata = {
    "color": ["red", "blue", "green"], 
    "position": [10,20,30]
    }

tsdframe = nap.TsdFrame(d=d, t=t, columns=["a", "b", "c"], metadata=metadata)
print(tsdframe)
```

When initializing with a DataFrame, the DataFrame index must match the `TsdFrame` columns.
```{code-cell} ipython3
metadata = pd.DataFrame(
    index=["a", "b", "c"],
    data=[["red", 10], ["blue", 20], ["green", 30]], 
    columns=["color", "position"],
)

tsdframe = nap.TsdFrame(d=d, t=t, columns=["a", "b", "c"], metadata=metadata)
print(tsdframe)
```

#### `set_info`
After creation, metadata can be added using the class method `set_info()`. Metadata can be passed as a dictionary or pandas DataFrame as the first positional argument, or metadata can be passed as name-value keyword arguments.

```{admonition} Note
The remaining metadata examples will be shown on a `TsGroup` object; however, all examples can be directly applied to `IntervalSet` and `TsdFrame` objects.
```

```{code-cell} ipython3
tsgroup.set_info(unit_type = ["multi", "single", "single"])
print(tsgroup)
```

### Accessing metadata
Metadata is stored as a pandas DataFrame, which can be previewed using the `metadata` attribute.

```{code-cell} ipython3
print(tsgroup.metadata)
```

Single metadata columns or lists of columns can be retrieved using the `get_info()` class method:
```{code-cell} ipython3
print(tsgroup.get_info("region"))
```

### Metadata properties
User-set metadata is mutable and can be overwritten.
```{code-cell} ipython3
tsgroup.set_info(region=["A","B","C"])
print(tsgroup.get_info("region"))
```

If the metadata name does not overlap with an existing class column, it can be set and accessed via key indexing (i.e. using square brackets).

```{admonition} Note
As mentioned previously, metadata names must be strings. Bracket-indexing with an integer will produce different behavior based on object type and will not return metadata.
```

```{code-cell} ipython3
tsgroup["depth"] = [0,1,2]
print(tsgroup["depth"])
```

Similarly, if the metadata name is unique from other class attributes and methods, and it is formatted properly (i.e. only alpha-numeric characters and underscores), it can be set and accessed as an attribute (i.e. using a `.` followed by the metadata name).
```{code-cell} ipython3
tsgroup.unit_type = ["MUA","good","good"]
print(tsgroup.unit_type)
```

As long as the length of the metadata container matches the length of the object (number of columns for `TsdFrame` and number of indices for `IntervalSet` and `TsGroup`), elements of the metadata can be any data type.
```{code-cell} ipython3
tsgroup.coords = [[1,0],[0,1],[1,1]]
print(tsgroup.coords)
```

### Filtering
Metadata can be used to filter or threshold objects based on metadata values.
```{code-cell} ipython3
print(tsgroup[tsgroup.unit_type == "good"])
```

