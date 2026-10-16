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

# Trial-aligned decoding with International Brain Lab data

This example works through a basic pipeline for decoding the mouse's choice from spiking activity in the International Brain Lab's decision task, including loading the data from DANDI, processing and trial-aligning the neural activity, and fitting a logistic regression with cross-validation using scikit-learn.

The International Brain Lab's Brain Wide Map dataset is available at [Dandiset 00409](https://dandiarchive.org/dandiset/000409/). The International Brain Lab's [BWM website](https://www.internationalbrainlab.com/brainwide-map) includes links to [their preprint](https://www.biorxiv.org/content/10.1101/2023.07.04.547681) and additional documentation. The IBL also has an excellent decoding demonstration in the [COSYNE section of their events webpage](https://www.internationalbrainlab.com/events) under "Tutorial 2: Advanced analysis", along with many other relevant demos.

For a more detailed tutorial on data loading with DANDI, see the "Streaming data from DANDI" example!

**Caveats**: This example is meant to provide a simple starting point for working with trial-aligned data and data from the IBL, and so it does not faithfully replicate the IBL's quality control and filtering criteria; the decoding here is also simpler than the analyses carried out in those works.

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
# The IBL dandiset:
dandiset_id = "000409"
# This is a randomly chosen recording session.
asset_path = "sub-CSH-ZAD-026/sub-CSH-ZAD-026_ses-15763234-d21e-491f-a01b-1238eb96d389_desc-processed_behavior+ecephys.nwb"
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

This NWB file, loaded into a pynapple object, contains neural activity, 
behavioral data, and raw electrophysiological traces. 
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
We can now easily discard trials where a choice was not made (a more complete analysis may include more criteria): 
```{code-cell} ipython3
trial_data = trial_data[trial_data.trials["mouse_wheel_choice"] != "none"]
trial_data
```

## Visualising aligned spikes

We can select an example unit and visualize its trial-aligned spikes:
```{code-cell} ipython3
example_unit = 420
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
trial_data = trial_data.realign_to("gabor_stimulus_onset_time")
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
trial_data = trial_data.get(start=-0.1, end=0.5)
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
bin_size = 0.05
binned = trial_data.count(bin_size)
binned
```

Which we can again visualize, but now as a heatmap for all units and an example trial:
```{code-cell} ipython3
example_trial = 420
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

## Decoding choice
Say we now want to decode the trial outcome from these binned spike counts.
We can use the [scikit-learn library](https://scikit-learn.org/stable/) to fit a [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
We can use their [cross-validation tools](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to pick the regularization parameter

We will define the label:
```{code-cell} ipython3
y = (binned.trials["mouse_wheel_choice"] == "clockwise").astype(int)
y
```

And the design matrix:
```{code-cell} ipython3
X = binned.values
# standardize per neuron + trial
X = (X - X.mean(2, keepdims=True)) / X.std(2, keepdims=True).clip(min=1e-6)
# reshape to n_trials x n_features
X = X.reshape(len(X), -1)
X
```

Now we can fit the model:
```{code-cell} ipython3
# loguniform randomized search CV logistic regression
clf = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions={'C': scipy.stats.loguniform(1e-3, 1e3)},
    random_state=0,
)
clf.fit(X, y);
```

And we can visualize the resulting prediction accuracy:
```{code-cell} ipython3
cv_df = pd.DataFrame(clf.cv_results_)
best_row = cv_df[cv_df.rank_test_score == 1]
best_row = best_row.iloc[best_row.param_C.argmin()]
fig, ax = plt.subplots(figsize=(4, 4))
plt.boxplot(best_row[[c for c in best_row.keys() if c.startswith("split")]])
plt.axhline(0.5, color="gray", label="random chance")
plt.ylabel("accuracy in fold")
plt.xticks([])
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
plt.title("choice decoding accuracy\nin 5-fold randomized CV")
plt.show()
```

:::{card}
Authors
^^^

Charlie Windolf

[Wolf De Wulf](https://wulfdewolf.github.io)
:::


