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

## Interaction between epochs 

```{code-cell} ipython3
epoch1 = nap.IntervalSet(start=0, end=10)  # no time units passed. Default is us.
epoch2 = nap.IntervalSet(start=[5, 30], end=[20, 45])
print(epoch1, "\n")
print(epoch2, "\n")
```

### Union

```{code-cell} ipython3
epoch = epoch1.union(epoch2)
print(epoch)
```

### Intersection

```{code-cell} ipython3
epoch = epoch1.intersect(epoch2)
print(epoch)
```

### Difference

```{code-cell} ipython3
epoch = epoch1.set_diff(epoch2)
print(epoch)
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

tsgroup = nap.TsGroup(group)

print(tsgroup, "\n")

```

## Time series method

### `value_from`

We can use `value_from` which as it indicates assign to every timestamps 
the closed value in time from another time series. Let's define the time series we want to assign values from.

```{code-cell} ipython3
tsd_sin = nap.Tsd(t=np.arange(0, 100, 1), d=np.sin(np.arange(0, 10, 0.1)))
```

For every timestamps in `tsgroup`, we want to assign the closest value in time from `tsd_sin`.

```{code-cell} ipython3
tsgroup_sin = tsgroup.value_from(tsd_sin)
```

We can display the first element of `tsgroup` and `tsgroup_sin`.

```{code-cell} ipython3
plt.figure(figsize=(12, 6))
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

