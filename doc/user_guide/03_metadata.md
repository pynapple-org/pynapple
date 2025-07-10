---
jupyter:
  jupytext:
    default_lexer: ipython3
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: pynapple
    language: python
    name: python3
---

<!-- #region -->
# Metadata
Metadata can be added to `TsGroup`, `IntervalSet`, and `TsdFrame` objects at initialization or after an object has been created.
- `TsGroup` metadata is information associated with each Ts/Tsd object, such as brain region or unit type.
- `IntervalSet` metadata is information associated with each interval, such as a trial label or stimulus condition.
- `TsdFrame` metadata is information associated with each column, such as a channel or position.


## Adding metadata at initialization
At initialization, metadata can be passed via a dictionary or pandas DataFrame using the keyword argument `metadata`. The metadata name is taken from the dictionary key or DataFrame column, and it can be set to any string name with a couple class-specific exceptions. 

```{admonition} Class-specific exceptions
- If column names are supplied to `TsdFrame`, metadata cannot overlap with those names.
- The `rate` attribute for `TsGroup` is stored with the metadata and cannot be overwritten.
```

The length of the metadata must match the length of the object it describes (see class examples below for more detail).
<!-- #endregion -->

```python tags=["hide-cell"]
import numpy as np
import pandas as pd
import pynapple as nap

# input parameters for TsGroup
group = {
    1: nap.Ts(t=np.sort(np.random.uniform(0, 100, 100))),
    2: nap.Ts(t=np.sort(np.random.uniform(0, 100, 200))),
    3: nap.Ts(t=np.sort(np.random.uniform(0, 100, 300))),
    4: nap.Ts(t=np.sort(np.random.uniform(0, 100, 400))),
}

# input parameters for IntervalSet
starts = [0,35,70]
ends = [30,65,100]

# input parameters for TsdFrame
t = np.arange(5)
d = np.tile([1,2,3], (5, 1))
columns = ["a", "b", "c"]
```

### `TsGroup`
Metadata added to `TsGroup` must match the number of `Ts`/`Tsd` objects, or the length of its `index` property.

```python
metadata = {"region": ["pfc", "pfc", "hpc", "hpc"]}

tsgroup = nap.TsGroup(group, metadata=metadata)
print(tsgroup)
```

When initializing with a DataFrame, the index must align with the input dictionary keys (only when a dictionary is used to create the `TsGroup`).

```python
metadata = pd.DataFrame(
    index=group.keys(),
    data=["pfc", "pfc", "hpc", "hpc"],
    columns=["region"]
)

tsgroup = nap.TsGroup(group, metadata=metadata)
print(tsgroup)
```

### `IntervalSet`
Metadata added to `IntervalSet` must match the number of intervals, or the length of its `index` property.

```python
metadata = {
    "reward": [1, 0, 1],
    "choice": ["left", "right", "left"],    
}
intervalset = nap.IntervalSet(starts, ends, metadata=metadata)
print(intervalset)
```

Metadata can be initialized as a DataFrame using the metadata argument, or it can be inferred when initializing an `IntervalSet` with a DataFrame.

```python
df = pd.DataFrame(
    data=[[0, 30, 1, "left"], [35, 65, 0, "right"], [70, 100, 1, "left"]], 
    columns=["start", "end", "reward", "choice"]
    )

intervalset = nap.IntervalSet(df)
print(intervalset)
```

### `TsdFrame`
Metadata added to `TsdFrame` must match the number of data columns, or the length of its `columns` property.

```python
metadata = {
    "color": ["red", "blue", "green"], 
    "position": [10,20,30],
    "label": ["x", "x", "y"]
    }

tsdframe = nap.TsdFrame(d=d, t=t, columns=["a", "b", "c"], metadata=metadata)
print(tsdframe)
```

When initializing with a DataFrame, the DataFrame index must match the `TsdFrame` columns.

```python
metadata = pd.DataFrame(
    index=["a", "b", "c"],
    data=[["red", 10, "x"], ["blue", 20, "x"], ["green", 30, "y"]], 
    columns=["color", "position", "label"],
)

tsdframe = nap.TsdFrame(d=d, t=t, columns=["a", "b", "c"], metadata=metadata)
print(tsdframe)
```

## Adding metadata after initialization
After creation, metadata can be added using the class method [`set_info()`](pynapple.TsdFrame.set_info). Additionally, single metadata fields can be added as a dictionary-like key or as an attribute, with a few noted exceptions outlined below.

```{admonition} Note
The remaining metadata examples will be shown on a `TsGroup` object; however, all examples can be directly applied to `IntervalSet` and `TsdFrame` objects.
```

### `set_info`
Metadata can be passed as a dictionary or pandas DataFrame as the first positional argument, or metadata can be passed as name-value keyword arguments.

```python
tsgroup.set_info(unit_type=["multi", "single", "single", "single"])
print(tsgroup)
```

### Using dictionary-like keys (square brackets)
Most metadata names can set as a dictionary-like key (i.e. using square brackets). The only exceptions are for `IntervalSet`, where the names "start" and "end" are reserved for class properties.

```python
tsgroup["depth"] = [0, 1, 2, 3]
print(tsgroup)
```

### Using attribute assignment
If the metadata name is unique from other class attributes and methods, and it is formatted properly (i.e. only alpha-numeric characters and underscores), it can be set as an attribute (i.e. using a `.` followed by the metadata name).

```python
tsgroup.label=["MUA", "good", "good", "good"]
print(tsgroup)
```

## Allowed data types
As long as the length of the metadata container matches the length of the object (number of columns for `TsdFrame` and number of indices for `IntervalSet` and `TsGroup`), elements of the metadata can be any data type.

```python
tsgroup.coords = [[1,0],[0,1],[1,1],[2,1]]
print(tsgroup)
```

## Accessing metadata
Metadata is stored as a pandas DataFrame, which can be previewed using the `metadata` attribute.

```python
print(tsgroup.metadata)
```

Single metadata columns (or lists of columns) can be retrieved using the [`get_info()`](pynapple.TsGroup.get_info) class method:

```python
print(tsgroup.get_info("region"))
```

Similarly, metadata can be accessed using key indexing (i.e. square brakets)

```python
print(tsgroup["region"])
```

```{admonition} Note
Metadata names must be strings. Key indexing with an integer will produce different behavior based on object type.
```

Finally, metadata that can be set as an attribute can also be accessed as an attribute.

```python
print(tsgroup.region)
```

## Overwriting metadata
User-set metadata is mutable and can be overwritten.

```python
print(tsgroup, "\n")
tsgroup.set_info(label=["A", "B", "C", "D"])
print(tsgroup)
```

## Dropping metadata
To drop metadata, use the [`drop_info()`](pynapple.TsGroup.drop_info) method. Multiple metadata columns can be dropped by passing a list of metadata names.

```python
print(tsgroup, "\n")
tsgroup.drop_info("coords")
print(tsgroup)
```

## Using metadata to slice objects
Metadata can be used to slice or filter objects based on metadata values.

```python
print(tsgroup[tsgroup.label == "A"])
```

## `groupby`: Using metadata to group objects
Similar to pandas, metadata can be used to group objects based on one or more metadata columns using the object method [`groupby`](pynapple.TsGroup.groupby), where the first argument is the metadata columns name(s) to group by. This function returns a dictionary with keys corresponding to unique groups and values corresponding to object indices belonging to each group.

```python
print(tsgroup,"\n")
print(tsgroup.groupby("region"))
```

Grouping by multiple metadata columns should be passed as a list.

```python
tsgroup.groupby(["region","unit_type"])
```

The optional argument `get_group` can be provided to return a new object corresponding to a specific group.

```python
tsgroup.groupby("region", get_group="hpc")
```

## `groupby_apply`: Applying functions to object groups
The `groupby_apply` object method allows a specific function to be applied to object groups. The first argument, same as `groupby`, is the metadata column(s) used to group the object. The second argument is the function to apply to each group. If only these two arguments are supplied, it is assumed that the grouped object is the first and only input to the applied function. This function returns a dictionary, where keys correspond to each unique group, and values correspond to the function output on each group.

```python
print(tsdframe,"\n")
print(tsdframe.groupby_apply("label", np.mean))
```

If the applied function requires additional inputs, these can be passed as additional keyword arguments into `groupby_apply`.

```python
feature = nap.Tsd(t=np.arange(100), d=np.repeat([0,1], 50))
tsgroup.groupby_apply(
    "region", 
    nap.compute_tuning_curves, 
    features=feature, 
    bins=2)
```

Alternatively, an anonymous function can be passed instead that defines additional arguments.

```python
func = lambda x: nap.compute_tuning_curves(x, features=feature, bins=2)
tsgroup.groupby_apply("region", func)
```

An anonymous function can also be used to apply a function where the grouped object is not the first input.

```python
func = lambda x: nap.compute_tuning_curves(
    group=tsgroup, 
    features=feature, 
    bins=2, 
    epochs=x)
intervalset.groupby_apply("choice", func)
```

Alternatively, the optional parameter `input_key` can be passed to specify which keyword argument the grouped object corresponds to. Other required arguments of the applied function need to be passed as keyword arguments.

```python
intervalset.groupby_apply(
    "choice", 
    nap.compute_tuning_curves, 
    input_key="epochs", 
    group=tsgroup, 
    features=feature, 
    bins=2)
```

```python

```
