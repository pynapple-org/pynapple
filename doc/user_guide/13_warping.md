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

The function `nap.build_tensor` slices a time series object or timestamps object for each interval of an `IntervalSet` object and returns 
a numpy array. The intervals can be of unequal durations. To build a trial-based tensor from a `TsGroup` object :

```
>>> tensor = nap.build_tensor(tsgroup, ep, binsize=1, padding_value=np.nan)
```

This function is also available at the object level.

```
>>> tensor = tsgroup.to_trial_tensor(ep, binsize=1, padding_value=np.nan)
```

The function `nap.warp_tensor` is similar to `build_tensor` but time is stretched linearly for each interval depending on
the parameter `num_bin`

```
>>> tensor = nap.warp_tensor(tsgroup, ep, num_bin=1)
```

Both functions works for all time series object (`Tsd`, `TsdFrame` and `TsdTensor`) and timestamp objects (`Ts` and `TsGroup`).

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

