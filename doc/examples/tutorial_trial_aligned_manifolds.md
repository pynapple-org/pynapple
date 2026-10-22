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

# Trial-aligned population manifolds

This example works through a basic pipeline for computing trial-aligned manifolds, as in the CITE
It includes loading the data from DANDI, processing and trial-aligning the neural activity using Pynapple's [`TsTrials`](pynapple.core.TsTrials) object, 
and using the [scikit-learn library](https://scikit-learn.org/stable/) to apply principal component analsysis (PCA).

```{code-cell} ipython3
:tags: [hide-input]
# we'll import the packages we're going to use
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

import pynapple as nap

# some configuration, you can ignore this
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params);
```

## Loading data

See also the [tutorial on streaming data from DANDI](/examples/tutorial_pynapple_dandi.md),
which includes more detail on this part

```{code-cell} ipython3
# These modules are used for data loading
from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py
from pynwb import NWBHDF5IO
```

```{code-cell} ipython3
dandiset_id = "000688"
# This is a randomly chosen recording session.
asset_path = "sub-C/sub-C_ses-CO-20150319_behavior+ecephys.nwb"
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(asset_path)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage=str(Path("~/.caches/nwb-cache").expanduser()),
)
io = NWBHDF5IO(file=h5py.File(fs.open(s3_url, "rb")), load_namespaces=True)
nwb = nap.NWBFile(io.read())
```

This NWB file, loaded into a Pynapple object, contains neural activity, behavioral data, and trial information.
They're all lazily loaded, so only what we use will be downloaded.

```{code-cell} ipython3
nwb
```

The only fields we'll use here are the neural activity ('units') and the trial information ('trials'):

```{code-cell} ipython3
units = nwb['units']
trials = nwb['trials']
```

```{code-cell} ipython3
units
```

```{code-cell} ipython3
trials
```

## Trial alignment
We will wrap both in a `TsTrials` object, which facilitates working with trial structure:
```{code-cell} ipython3
trial_data = nap.TsTrials(source=units, trials=trials)
trial_data
```

## Subselecting trials based on metadata
We can now easily discard trials with missing data:
```{code-cell} ipython3
trial_data = trial_data[~np.isnan(trial_data.trials["go_cue_time"])]
trial_data
```

## Visualising aligned spikes

We can select an example unit and visualize its trial-aligned spikes:
```{code-cell} ipython3
example_unit = 22
plt.plot(
    trial_data[:, example_unit].to_tsdframe()["trial"],
    marker="|",
    linestyle="none",
    markersize=1,
)
plt.ylabel("trial")
plt.xlabel("time from start of trial (s)")
plt.title(f"unit {example_unit}")
plt.show()
```

Or, if we want to align to a different time point stored in the metadata:
```{code-cell} ipython3
trial_data = trial_data.realign_to("go_cue_time")
trial_data
```

```{code-cell} ipython3
plt.plot(
    trial_data[:, example_unit].to_tsdframe()["trial"],
    marker="|",
    linestyle="none",
    markersize=1,
)
plt.axvline(0.0, color="red")
plt.ylabel("trial")
plt.xlabel("time from stimulus (s)")
plt.title(f"unit {example_unit}")
plt.show()
```

## Restricting the window
Each trial might have a different length, with some being very long.
To get a better overview of what is happening, we might want to restrict to a
fixed window around the alignment point:

```{code-cell} ipython3
trial_data = trial_data.get(start=-0.2, end=0.5)
trial_data
```

```{code-cell} ipython3
plt.plot(
    trial_data[:, example_unit].to_tsdframe()["trial"],
    marker="|",
    linestyle="none",
    markersize=1,
)
plt.axvline(0.0, color="red")
plt.ylabel("trial")
plt.xlabel("time from stimulus (s)")
plt.title(f"unit {example_unit}")
plt.show()
```

## Binning
We can then bin and count the spikes as follows, this yields a `TsdTrials` object:
```{code-cell} ipython3
bin_size = 0.02
binned = trial_data.count(bin_size)
binned
```

Which we can again visualize, but now as a heatmap for all units and an example trial:
```{code-cell} ipython3
example_trial = 42
plt.imshow(
    binned[example_trial].values.T,
    aspect="auto",
    interpolation="none",
    extent=(
        binned.t[0],
        binned.t[-1],
        0,
        binned.shape[-1],
    ),
)
plt.axvline(0, color='red')
plt.ylabel('unit')
plt.xlabel('time from stimulus (s)');
plt.title(f"trial {example_trial}")
plt.show()
```

## Principal components manifold

```{code-cell} ipython3
n_trials, n_time, n_units = binned.shape
X = binned.values
X = gaussian_filter1d(
    X / bin_size,
    sigma=0.04 / bin_size,
    axis=1,
)
X = X.reshape(-1, n_units)
unit_mean = X.mean(axis=0)
unit_std = X.std(axis=0)
X = (X - unit_mean) / np.maximum(unit_std, 1e-6)
```

```{code-cell} ipython3
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
trajectories = pca.fit_transform(X).reshape(n_trials, n_time, 3)
```


:::{card}
Authors
^^^
[Wolf De Wulf](https://wulfdewolf.github.io)
:::
