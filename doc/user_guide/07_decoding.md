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

# Decoding

```python tags=["hide-cell"]
import pynapple as nap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

<!-- #region -->
Pynapple supports 1 dimensional and 2 dimensional bayesian decoding. The function returns the decoded feature as well as the probabilities for each timestamps. 


:::{hint}
Input to the bayesian decoding functions always include the tuning curves computed using [`nap.compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves).
:::

## 1-dimensional decoding
<!-- #endregion -->

```python tags=["hide-cell"]
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
epoch = nap.IntervalSet(0, 10)
```

To decode, we need to compute tuning curves in 1D.

```python
tuning_curves_1d = nap.compute_tuning_curves(
    tsgroup, feature, bins=61, range=[(0, 2 * np.pi)]
)
```

We can display the tuning curves of each neurons

```python tags=["hide-input"]
tuning_curves_1d.plot.line(x="feature0", add_legend=False)
plt.xlabel("Feature position")
plt.ylabel("Rate (Hz)")
plt.show()
```

`nap.decode_1d` performs bayesian decoding:

```python
decoded, proba_feature = nap.decode_1d(
    tuning_curves=tcurves_1d.to_pandas().T, # 1D tuning curves
    group=tsgroup, # Spiking activity
    ep=epoch, # Small epoch
    bin_size=0.06,  # How to bin the spike trains
    feature=feature, # Features
)
```

`decoded` is `Tsd` object containing the decoded feature value. `proba_feature` is a `TsdFrame` containing the probabilities of being in a particular feature bin over time.

```python tags=["hide-input"]
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(feature.restrict(epoch), label="True")
plt.plot(decoded, label="Decoded")
plt.legend()
plt.xlim(epoch[0,0], epoch[0,1])
plt.subplot(212)
plt.imshow(proba_feature.values.T, aspect="auto", origin="lower", cmap="viridis")
plt.xticks([0, len(decoded)], epoch.values[0])
plt.xlabel("Time (s)")
plt.show()
```

## 2-dimensional decoding

```python tags=["hide-cell"]
dt = 0.1
epochs = nap.IntervalSet(start=0, end=1000, time_units="s")
features = np.vstack((np.cos(np.arange(0, 1000, dt)), np.sin(np.arange(0, 1000, dt)))).T
features = nap.TsdFrame(t=np.arange(0, 1000, dt),
    d=features,
    time_units="s",
    time_support=epoch,
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

ts_group = nap.TsGroup(ts_group, time_support=epoch)
```

To decode, we need to compute tuning curves in 2D.

```python
tuning_curves_2d = nap.compute_tuning_curves(
    group=ts_group, # Spiking activity of 12 neurons
    features=features, # 2-dimensional features
    bins=10,
    epochs=epochs,
    range=[(-1.0, 1.0), (-1.0, 1.0)], # Minmax of the features
)
```

We can display the tuning curves of each neuron

```python
tuning_curves_2d.plot(row="unit", col_wrap=6)
```

`nap.decode_2d` performs bayesian decoding:

```python
tcs = {c: tuning_curves_2d.sel(unit=c).values for c in tuning_curves_2d.coords["unit"].values}
bins = [tuning_curves_2d.coords[dim].values for dim in tuning_curves_2d.coords if dim != "unit"]

decoded, proba_feature = nap.decode_2d(
    tuning_curves=tcs, # 2D tuning curves
    group=ts_group, # Spiking activity
    ep=epoch, # Epoch
    bin_size=0.1,  # How to bin the spike trains
    xy=bins, # Features position
    features=features, # Features
)
```

```python tags=["hide-input"]
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

```python

```
