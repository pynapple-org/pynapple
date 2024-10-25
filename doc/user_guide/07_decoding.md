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
import matplotlib.pyplot as plt
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
```
Pynapple supports 1 dimensional and 2 dimensional bayesian decoding. The function returns the decoded feature as well as the probabilities for each timestamps.

First we generate some artificial "place fields" in 2 dimensions based on the features.

This part is just to generate units with a relationship to the features (i.e. "place fields")


```{code-cell} ipython3
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
print(ts_group)
```

To decode we need to compute tuning curves in 2D.


```{code-cell} ipython3
import warnings

warnings.filterwarnings("ignore")

tcurves2d, binsxy = nap.compute_2d_tuning_curves(
    group=ts_group,
    features=features,
    nb_bins=10,
    ep=epoch,
    minmax=(-1.0, 1.0, -1.0, 1.0),
)
```

Then we plot the "place fields".


```{code-cell} ipython3
plt.figure(figsize=(20, 9))
for i in ts_group.keys():
    plt.subplot(2, 6, i + 1)
    plt.imshow(
        tcurves2d[i], extent=(binsxy[1][0], binsxy[1][-1], binsxy[0][0], binsxy[0][-1])
    )
    plt.xticks()
plt.show()
```

Then we call the actual decoding function in 2d.


```{code-cell} ipython3
decoded, proba_feature = nap.decode_2d(
    tuning_curves=tcurves2d,
    group=ts_group,
    ep=epoch,
    bin_size=0.1,  # second
    xy=binsxy,
    features=features,
)


plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(features["a"].as_units("s").loc[0:20], label="True")
plt.plot(decoded["a"].as_units("s").loc[0:20], label="Decoded")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Feature a")
plt.subplot(132)
plt.plot(features["b"].as_units("s").loc[0:20], label="True")
plt.plot(decoded["b"].as_units("s").loc[0:20], label="Decoded")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Feature b")
plt.subplot(133)
plt.plot(
    features["a"].as_units("s").loc[0:20],
    features["b"].as_units("s").loc[0:20],
    label="True",
)
plt.plot(
    decoded["a"].as_units("s").loc[0:20],
    decoded["b"].as_units("s").loc[0:20],
    label="Decoded",
)
plt.xlabel("Feature a")
plt.ylabel("Feature b")
plt.legend()
plt.tight_layout()
plt.show()
```