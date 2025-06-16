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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import scipy.stats

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
```

## Loading data

See also the "Streaming data from DANDI", which includes more detail.

```{code-cell} ipython3
# These modules are used for data loading
from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py
from pynwb import NWBHDF5IO
```

```{code-cell} ipython3
# The BWM dandiset:
dandiset_id = "000409"
# This is a randomly chosen recording session.
asset_path = "sub-CSH-ZAD-026/sub-CSH-ZAD-026_ses-15763234-d21e-491f-a01b-1238eb96d389_behavior+ecephys+image.nwb"
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

This NWB file, loaded into a pynapple object, contains neural activity, behavioral data, and raw electrophysiological traces. They're lazily loaded, so only what we use will be downloaded.

```{code-cell} ipython3
nwb
```

The only fields we'll use here are the neural activity ('units') and the trial information ('trials').

```{code-cell} ipython3
spikes = nwb['units']
trials = nwb['trials']
```

```{code-cell} ipython3
spikes
```

```{code-cell} ipython3
trials
```

## Trial alignment and binning

Here, we start by discarding some trials where a choice wasn't made (a fuller analysis may include more criteria). Next, we create an `IntervalSet` aligned to the stimulus onset time within each trial, where the window around each stimulus extends back 0.5s and forward 2s.

```{code-cell} ipython3
valid_choice = trials.choice != 0
trials = trials[valid_choice]
stim_on_intervals = nap.IntervalSet(
    start=trials.stim_on_time - 0.5,
    end=trials.stim_on_time + 2.0,
)
```

Now, use [`build_tensor`](pynapple.process.warping.build_tensor) to align the neural data to these stimulus windows and bin it in 0.1s time bins.

```{code-cell} ipython3
trial_aligned_binned_spikes = nap.build_tensor(
    spikes, stim_on_intervals, bin_size=0.1
)
trial_aligned_binned_spikes.shape
```

The result is `n_neurons x n_trials x n_bins`. Let's visualize the neural activity in a single trial.

```{code-cell} ipython3
trial = 645
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(
    trial_aligned_binned_spikes[:, trial],
    aspect='auto',
    cmap=plt.cm.gist_yarg,
    interpolation='none',
)
ax.axvline(5.5, lw=1, color='r', label='stimulus time')
ax.set_ylabel('units')
ax.set_xlabel('time bin (0.1s)')
ax.legend(frameon=False, loc='lower right')
ax.set_title(f'binned spiking activity in trial {trial}')
```

## Decoding choice

Let's use scikit-learn to fit a logistic regression.
We can use their cross-validation tools to pick the regularization parameter

```{code-cell} ipython3
# process the choice from -1, 1 to 0,1
y = (trials.choice + 1) / 2
```

```{code-cell} ipython3
# transpose the data to lead with trial dimension
X = trial_aligned_binned_spikes.transpose(1, 0, 2)
# standardize per neuron + trial
X = (X - X.mean(2, keepdims=True)) / X.std(2, keepdims=True).clip(min=1e-6)
# reshape to n_trials x n_features
X = X.reshape(len(X), -1)
```

```{code-cell} ipython3
# loguniform randomized search CV logistic regression
clf = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions={'C': scipy.stats.loguniform(1e-3, 1e3)},
    random_state=0,
)
clf.fit(X, y)
```

```{code-cell} ipython3
cv_df = pd.DataFrame(clf.cv_results_)
best_row = cv_df[cv_df.rank_test_score == 1]
best_row = best_row.iloc[best_row.param_C.argmin()]
plt.figure(figsize=(2, 4))
plt.boxplot(best_row[[c for c in best_row.keys() if c.startswith('split')]])
plt.ylabel('accuracy in fold')
plt.xticks([])
plt.title("choice decoding accuracy\nin 5-fold randomized CV")
```

:::{card}
Authors
^^^

Charlie Windolf
:::


