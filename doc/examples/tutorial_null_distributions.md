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

Null distributions to test spatial firing
============

Null distributions are ubiquitous in neuroscience. 
The brain can be incredibly noisy, and thus we need to use adequate statistical quantification to separate meaningful observations from random fluctuations.
While statistics can be daunting, most often designing the right statistical test comes down to coming up with a good null distribution.
A null distribution should reflect the expected behavior of the system under the assumption of no real effect, without introducing unnecessary assumptions or simplifications.
We can then compare our observations to such a null distribution to decide whether there is an effect.

If you want to learn the in's and out's of statistical analysis of neural data, we'd advise picking up a [book](https://link.springer.com/content/pdf/10.1007/978-1-4614-9602-1.pdf) about it!

In this tutorial, we will use Pynapple's randomization module to generate various null distributions to test whether the activity of neurons is modulated by position.

```{code-cell} ipython3
:tags: [remove-output]
# we'll import the packages we're going to use
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pynapple as nap
import seaborn as sns
from scipy.signal import correlate2d
from scipy.ndimage import (
    maximum_filter,
    label,
    center_of_mass,
    rotate,
    gaussian_filter,
)
import xarray as xr
from pathlib import Path
from pynwb import NWBHDF5IO

from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py

# some configuration, you can ignore this
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
xr.set_options(display_expand_attrs=False)
```

***
Downloading data from DANDI
---------------------------
We will start by downloading some data off of DANDI.
To do so, we need a dandiset ID, and a file path.
In this tutorial we will use data from:  
*Sargolini, Francesca, et al. “Conjunctive representation of position, direction, and velocity in entorhinal cortex.” Science 312.5774 (2006): 758-762*

This dataset contains electrophysiology recordings from neurons in the entorhinal cortex, which are known to modulate their firing with position.

```{code-cell} ipython3
:tags: [remove-output]
dandiset = "000582"
session = "sub-10884/sub-10884_ses-03080402_behavior+ecephys.nwb"

with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset, "draft").get_asset_by_path(session)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage=str(Path("~/.caches/nwb-cache").expanduser()),
)
io = NWBHDF5IO(file=h5py.File(fs.open(s3_url, "rb")), load_namespaces=True)
nwb = nap.NWBFile(io.read(), lazy_loading=False)

units = nwb["units"]
position = nwb["SpatialSeriesLED1"]
```


***
Spatial tuning curves
---------------------
We can then use [`compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves) to compute the tuning curve for each neuron's spikes with respect to the position of the animal:
```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(
    data=units,
    features=position,
    feature_names=["X", "Y"],
    range=[(-50, 50), (-50, 50)],
    bins=40,
)

# optional smoothing of tuning curves
def gaussian_filter_nan(a, sigma):
    v = np.where(np.isnan(a), 0, a)
    w = gaussian_filter((~np.isnan(a)).astype(float), sigma)
    return gaussian_filter(v, sigma) / w
tuning_curves_smooth = tuning_curves.copy()
tuning_curves_smooth.values = gaussian_filter_nan(tuning_curves.values, sigma=(0, 2, 2))

# Setting some xarray.DataArray attributes, for beauty
tuning_curves_smooth.name = "firing rate"
tuning_curves_smooth.attrs["unit"] = "Hz"
tuning_curves_smooth.coords["X"].attrs["unit"] = "cm"
tuning_curves_smooth.coords["Y"].attrs["unit"] = "cm"
tuning_curves
```

We can further also compute our metric of interest.
In this case, we are interested in the modulation of the neurons' firing by position. 
A metric that is typically used to quantify this is the mutual information, often called [spatial information](https://proceedings.neurips.cc/paper/1992/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) for this case specifically:
```{code-cell} ipython3
spatial_information = nap.compute_mutual_information(tuning_curves)
spatial_information
```

We can visualize all that together:
```{code-cell} ipython3
g = tuning_curves_smooth.plot(col="unit")
for ax, (unit_idx, unit_id) in zip(g.axs.flat, enumerate(units)):
    ax.set_title(f"unit={unit_id}\nSI={spatial_information['bits/spike'][unit_idx]:.2f}")
plt.show()
```

***
Randomization
-------------
Next, we can start thinking about null distributions.
Our ultimate goal is to be able to say that a neuron's activity is significantly modulated by position.
To do that, we want to contrast it to the activity of a neuron that is not modulated by position at all.
But how do we pick that neuron? 
We do not want to stray away from the firing patterns of our neuron of interest too much, so we can't just pick something arbitrary.

In practice, we will often take the activity of our neuron of interest and shuffle it somehow, thereby breaking the relation with our variable of interest (position), but keeping its general firing statistics as much as possible.

Pynapple provides four methods for such shuffling:
- [`jitter_timestamps`](pynapple.process.randomize.jitter_timestamps): jitters timestamps independently by random amounts uniformly drawn from a given range.
- [`resample_timestamps`](pynapple.process.randomize.resample_timestamps): uniformly resamples all timestamps within the time support.
- [`shift_timestamps`](pynapple.process.randomize.shift_timestamps): shifts all timestamps by a random amount drawn from a given range, wrapping the end of the time support to the beginning.
- [`shuffle_ts_intervals`](pynapple.process.randomize.shuffle_ts_intervals): randomizes timestamps by shuffling the intervals between them.

Let's apply them to an example neuron to look at their effect on spatial firing:
```{code-cell} ipython3
jitter = nap.jitter_timestamps(units[3], max_jitter=1.0)
resample = nap.resample_timestamps(units[3])
shift = nap.shift_timestamps(units[3], min_shift=10.0)
interval_shuffle = nap.shuffle_ts_intervals(units[3])

fig, axs = plt.subplots(1, 5, figsize=(10, 2))
for ax, (randomization_type, data) in zip(
    axs,
    [
        ("true", units[3]),
        ("jitter", jitter),
        ("resample", resample),
        ("shift", shift),
        ("interval_shuffle", interval_shuffle),
    ],
):
    ax.set_title(randomization_type.replace("_", "\n"))
    ax.plot(position[:, 0], position[:, 1], color="black", linewidth=0.5, zorder=-1)
    positional_spikes = data.value_from(position)
    ax.scatter(
        positional_spikes[:, 0],
        positional_spikes[:, 1],
        color="red",
        s=7,
        edgecolor="none",
        zorder=1
    )
    ax.axis("off")
plt.show()
```

***
Null distributions
------------------
We will opt for random shifts, as those result in the least possible change in the spatial firing of the neurons, while still randomizing it.
For each neuron, we will generate `N` pseudo-neurons and compute the mutual/spatial information.
The resulting distributions of values are what we call the null distributions.
```{code-cell} ipython3
N = 500
shuffles = [nap.shift_timestamps(units, min_shift=20.0) for _ in range(N)]

null_distributions = np.stack(
    [
        nap.compute_mutual_information(
            nap.compute_tuning_curves(
                data=shuffle,
                features=position,
                range=[(-50, 50), (-50, 50)],
                bins=40,
            )
        )["bits/spike"].values
        for shuffle in shuffles
    ],
    axis=1,
)
null_distributions.shape
```

***
Testing
------------------
Ultimately, we will apply the test.
From a statistical view, we see the null distribution as an empirical distribution of spatial information values that would be expected if the neuron’s firing were unrelated to spatial position.
We want to determine how unlikely it is that the observed spatial information arose from our null distribution.

Given a significance level, α, we assess whether the observed value lies in the extreme tail of the null distribution.
If the probability (the p-value) of obtaining a spatial information value **as large as or larger than** the observed one is less than α, we reject the null hypothesis.
Since we have empirical distributions, we use the percentile the approximate that probability/p-value.

```{code-cell} ipython3
alpha = 0.01
thresholds = np.nanpercentile(null_distributions, (1-alpha)*100, axis=1)
spatial_units = tuning_curves.coords["unit"][spatial_information["bits/spike"].values > thresholds]

g = tuning_curves_smooth.plot(col="unit")
for ax, (unit_idx, unit_id) in zip(g.axs.flat, enumerate(units)):
    null = null_distributions[unit_idx]
    score = spatial_information['bits/spike'][unit_idx]
    pval = 1- np.sum(null <= score) / len(null)
    
    ax.set_title(f"unit={unit_id}\nSI={score:.2f}\np={pval:.4f}{'*' if score>thresholds[unit_idx] else ''}")

    ax_hist = inset_axes(ax, width="100%", height="40%", loc="lower center",
                         bbox_to_anchor=(0, -1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    
    ax_hist.hist(null, histtype="stepfilled", edgecolor="none", bins=30)
    ax_hist.axvline(thresholds[unit_idx], color="black")
    ax_hist.axvline(score, color="red", linestyle="--")
    ax_hist.yaxis.set_visible(False)
    ax_hist.spines['left'].set_visible(False)
    if unit_idx == 0:
        ax_hist.set_xlabel("spatial\ninformation\n(bits/spike)")
plt.show()
```

<!-- #region -->
:::{card}
Authors
^^^
Wolf De Wulf
:::
<!-- #endregion -->
