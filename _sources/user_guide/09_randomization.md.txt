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

# Randomization

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

Pynapple provides some ready-to-use randomization methods to compute null distributions for statistical testing.
Different methods preserve or destroy different features of the data.

## Shift timestamps

[`shift_timestamps`](pynapple.process.randomize.shift_timestamps) shifts all the timestamps in a `Ts` object by the same random amount, wrapping the end of the time support to its beginning. This randomization preserves the temporal structure in the data but destroys the temporal relationships with other quantities (e.g. behavioural data).
When applied on a `TsGroup` object, each series in the group is shifted independently.


```{code-cell} ipython3
ts = nap.Ts(t=np.sort(np.random.uniform(0, 100, 10)), time_units="ms")
rand_ts = nap.shift_timestamps(ts, min_shift=1, max_shift=20)
```

## Shuffle timestamp intervals

[`shuffle_ts_intervals`](pynapple.process.randomize.shift_timestamps) computes the intervals between consecutive timestamps, permutes them, and generates a new set of timestamps with the permuted intervals.
This procedure preserve the distribution of intervals, but not their sequence.


```{code-cell} ipython3
ts = nap.Ts(t=np.sort(np.random.uniform(0, 100, 10)), time_units="s")
rand_ts = nap.shuffle_ts_intervals(ts)
```

## Jitter timestamps

[`jitter_timestamps`](pynapple.process.randomize.jitter_timestamps) shifts each timestamp in the data of an independent random amount. When applied with a small `max_jitter`, this procedure destroys the fine temporal structure of the data, while preserving structure on longer timescales.


```{code-cell} ipython3
ts = nap.Ts(t=np.sort(np.random.uniform(0, 100, 10)), time_units="s")
rand_ts = nap.jitter_timestamps(ts, max_jitter=1)
```

## Resample timestamps

[`resample_timestamps`](pynapple.process.randomize.resample_timestamps) uniformly re-draws the same number of timestamps in `ts`, in the same time support. This procedures preserve the total number of timestamps, but destroys any other feature of the original data.


```{code-cell} ipython3
ts = nap.Ts(t=np.sort(np.random.uniform(0, 100, 10)), time_units="s")
rand_ts = nap.resample_timestamps(ts)
```
