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

# Tuning curves

Pynapple can compute 1 dimension tuning curves 
(for example firing rate as a function of angular direction) 
and 2 dimension tuning curves ( for example firing rate as a function 
of position).

:::{important}
If you are using calcium imaging data with the activity of the cell as a continuous transient, the function to call ends with `_continuous` for continuous time series (e.g. `compute_1d_tuning_curves_continuous`).
:::


```{code-cell} ipython3
:tags: [hide-cell]
import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```

## ..from epochs

## ... from timestamps activity
  
### 1-dimension tuning curves

### 2-dimension tuning curves

## ... from continuous activity

### 1-dimension tuning curves

### 2-dimension tuning curves


First we will create the 2D features:


```{code-cell} ipython3
dt = 0.1
features = np.vstack((np.cos(np.arange(0, 1000, dt)), np.sin(np.arange(0, 1000, dt)))).T
# features += np.random.randn(features.shape[0], features.shape[1])*0.05
features = nap.TsdFrame(
    t=np.arange(0, 1000, dt),
    d=features,
    time_units="s",
    time_support=epoch,
    columns=["a", "b"],
)

print(features)

plt.figure(figsize=(15, 7))
plt.subplot(121)
plt.plot(features[0:100])
plt.title("Features")
plt.xlabel("Time(s)")
plt.subplot(122)
plt.title("Features")
plt.plot(features["a"][0:100], features["b"][0:100])
plt.xlabel("Feature a")
plt.ylabel("Feature b")
```

Here we call the function `compute_2d_tuning_curves`.
To check the accuracy of the tuning curves, we will display the spikes aligned to the features with the function `value_from` which assign to each spikes the corresponding feature value for neuron 0.


```{code-cell} ipython3
tcurves2d, binsxy = nap.compute_2d_tuning_curves(
    group=ts_group, features=features, nb_bins=10
)

ts_to_features = ts_group[1].value_from(features)

plt.figure()
plt.plot(ts_to_features["a"], ts_to_features["b"], "o", color="red", markersize=4)
extents = (
    np.min(features["b"]),
    np.max(features["b"]),
    np.min(features["a"]),
    np.max(features["a"]),
)
plt.imshow(tcurves2d[1].T, origin="lower", extent=extents, cmap="viridis")
plt.title("Tuning curve unit 0 2d")
plt.xlabel("feature a")
plt.ylabel("feature b")
plt.grid(False)
plt.show()
```
