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

# Decoding

```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

Input to the decoding functions always includes:
 - `tuning_curves`, computed using [`nap.compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves).
 - `group`, a group of units as a `TsGroup` (spikes), `TsdFrame` (e.g. smoothed rates), or dict of `Ts`/`Tsd`.
 - `epochs`, to restrict decoding to certain intervals.
 - `bin_size`, for when you pass spikes.

## Bayesian decoding
Pynapple supports n-dimensional decoding from spikes in the form of Bayesian decoding with a Poisson assumption. 
In addition to the default arguments, users can set `uniform_prior=False` to use the occupancy as a prior over the feature distribution. 
By default `uniform_prior=True`, and a uniform prior is used.

:::{important}
Bayesian decoding should only be used with spike or rate data, as these can be assumed to follow a Poisson distribution!
:::


### 1-dimensional Bayesian decoding

```{code-cell} ipython3
:tags: [hide-cell]
from scipy.ndimage import gaussian_filter1d

# Fake Tuning curves
N = 6 # Number of neurons
bins = np.linspace(0, 2*np.pi, 61)
x = np.linspace(-np.pi, np.pi, len(bins)-1)
tmp = np.roll(np.exp(-(1.5*x)**2), (len(bins)-1)//2)
tc = np.array([np.roll(tmp, i*(len(bins)-1)//N) for i in range(N)]).T

tc_1d = pd.DataFrame(index=bins[0:-1], data=tc)

# Feature
T = 10000
dt = 0.01
timestep = np.arange(0, T)*dt
feature = nap.Tsd(
    t=timestep,
    d=gaussian_filter1d(np.cumsum(np.random.randn(T)*0.5), 20)%(2*np.pi)
    )
index = np.digitize(feature, bins)-1

# Spiking activity

count = np.random.poisson(tc[index])>0
tsgroup = nap.TsGroup({i:nap.Ts(timestep[count[:,i]]) for i in range(N)})
epochs = nap.IntervalSet(0, 10)
```

First, we compute the tuning curves:

```{code-cell} ipython3
tuning_curves_1d = nap.compute_tuning_curves(
    tsgroup, feature, bins=61, range=(0, 2 * np.pi), feature_names=["feature"]
)
```

```{code-cell} ipython3
:tags: [hide-input]
tuning_curves_1d.name = "Firing rate"
tuning_curves_1d.attrs["unit"] = "Hz"
tuning_curves_1d.plot.line(x="feature", add_legend=False)
plt.show()
```

We can then use `nap.decode_bayes` for Bayesian decoding:

```{code-cell} ipython3
decoded, proba_feature = nap.decode_bayes(
    tuning_curves=tuning_curves_1d,
    group=tsgroup,
    epochs=epochs,
    bin_size=0.06,
)
```

`decoded` is a `Tsd` object containing the decoded feature value.
`proba_feature` is a `TsdFrame` containing the probabilities of being in a particular feature bin over time.

```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(feature.restrict(epochs), label="True")
plt.plot(decoded, label="Decoded")
plt.legend()
plt.xlim(epochs[0,0], epochs[0,1])
plt.subplot(212)
plt.imshow(proba_feature.values.T, aspect="auto", origin="lower", cmap="viridis")
plt.xticks([0, len(decoded)], epochs.values[0])
plt.xlabel("Time (s)")
plt.show()
```

### 2-dimensional Bayesian decoding

```{code-cell} ipython3
:tags: [hide-cell]
dt = 0.1
epochs = nap.IntervalSet(start=0, end=1000, time_units="s")
features = np.vstack((np.cos(np.arange(0, 1000, dt)), np.sin(np.arange(0, 1000, dt)))).T
features = nap.TsdFrame(t=np.arange(0, 1000, dt),
    d=features,
    time_units="s",
    time_support=epochs,
    columns=["a", "b"],
)

times = features.as_units("us").index.values
ft = features.values
alpha = np.arctan2(ft[:, 1], ft[:, 0])
bins = np.repeat(np.linspace(-np.pi, np.pi, 13)[::, np.newaxis], 2, 1)
bins += np.array([-2 * np.pi / 24, 2 * np.pi / 24])
ts_group = {}
for i in range(12):
    ts = times[(alpha >= bins[i, 0]) & (alpha <= bins[i + 1, 1])]
    ts_group[i] = nap.Ts(ts, time_units="us")

ts_group = nap.TsGroup(ts_group, time_support=epochs)
```

Decoding also works with multiple dimensions.
First, we compute the tuning curves:

```{code-cell} ipython3
tuning_curves_2d = nap.compute_tuning_curves(
    group=ts_group,
    features=features, # containing 2 features
    bins=10,
    epochs=epochs,
    range=[(-1.0, 1.0), (-1.0, 1.0)], # range can be specified for each feature
)
```

```{code-cell} ipython3
:tags: [hide-input]
tuning_curves_2d.name = "Firing rate"
tuning_curves_2d.attrs["unit"] = "Hz"
tuning_curves_2d.plot(row="unit", col_wrap=6)
plt.show()
```

and then, `nap.decode_bayes` again performs bayesian decoding:

```{code-cell} ipython3
decoded, proba_feature = nap.decode_bayes(
    tuning_curves=tuning_curves_2d,
    group=ts_group,
    epochs=epochs,
    bin_size=0.1,
)
```

```{code-cell} ipython3
:tags: [hide-input]
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(features["a"].get(0,20), label="True")
plt.plot(decoded["a"].get(0,20), label="Decoded")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Feature a")
plt.subplot(132)
plt.plot(features["b"].get(0,20), label="True")
plt.plot(decoded["b"].get(0,20), label="Decoded")
plt.legend()
plt.xlabel("Time (s)")
plt.title("Feature b")
plt.subplot(133)
plt.plot(
    features["a"].get(0,20),
    features["b"].get(0,20),
    label="True",
)
plt.plot(
    decoded["a"].get(0,20),
    decoded["b"].get(0,20),
    label="Decoded",
)
plt.xlabel("Feature a")
plt.title("Feature b")
plt.legend()
plt.tight_layout()
plt.show()
```
