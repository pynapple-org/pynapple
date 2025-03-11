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

Trial-based tensor & time warping
=================================

The warping module contains functions for constructing trial-based tensor.
If the input is a TsGroup containing the activity of a population of neurons :

- `nap.build_tensor` -> returns a tensor of shape (number of neurons, number of trials, number of time bins) with padded values if unequal trial intervals.
- `nap.warp_tensor` -> returns a tensor of shape (number of neurons, number of trials, number of time bins) with linearly warped time.

See examples below.

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

`nap.build_tensor`
------------------

The function `nap.build_tensor` slices a time series object or timestamps object for each interval of an `IntervalSet` object and returns 
a numpy array. The intervals can be of unequal durations. 

```{code-cell} ipython3
tsgroup = nap.TsGroup({0:nap.Ts(t=np.arange(0, 100))})
ep = nap.IntervalSet(
    start=np.arange(20, 100, 20), end=np.arange(20, 100, 20) + np.arange(2, 10, 2),
    metadata = {'trials':['trial1', 'trial2', 'trial3', 'trial4']} 
    )
print(ep)
```

To build a trial-based tensor from a `TsGroup` object with 1 second bins: 

```{code-cell} ipython3
tensor = nap.build_tensor(tsgroup, ep, binsize=1, padding_value=np.nan)

print(tensor)
```

:::{note}
This function is also available at the object level.
```
>>> tensor = tsgroup.to_trial_tensor(ep, binsize=1, padding_value=np.nan)
```
:::


It is also possible to create a trial-based tensor from a time series. In this case the argument `binsize` is not used.

```{code-cell} ipython3
tsdframe = nap.TsdFrame(t=np.arange(100), d=np.arange(200).reshape(2,100).T)
tensor = nap.build_tensor(tsdframe, ep)

print(tensor)
```

`nap.warp_tensor`
-----------------

The function `nap.warp_tensor` is similar to `build_tensor` but time is stretched linearly for each interval depending on
the parameter `num_bin`

```
>>> tensor = nap.warp_tensor(tsgroup, ep, num_bin=1)
```

Both functions works for all time series object (`Tsd`, `TsdFrame` and `TsdTensor`) and timestamp objects (`Ts` and `TsGroup`).


