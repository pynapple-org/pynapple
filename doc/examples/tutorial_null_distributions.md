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

Null distributions in neuroscience
============

```{code-cell} ipython3
:tags: [remove-output]
import numpy as np
import matplotlib.pyplot as plt
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

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
xr.set_options(display_expand_attrs=False)
```

```{code-cell} ipython3
dandiset = "000053"
session = "sub-Barbara/sub-Barbara_ses-20190521_behavior.nwb"
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset, "draft").get_asset_by_path(session)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage=str(Path("~/.caches/nwb-cache").expanduser()),
)
io = NWBHDF5IO(file=h5py.File(fs.open(s3_url, "rb")), load_namespaces=True)
nwb = nap.NWBFile(io.read(), lazy_loading=False)
```

```{code-cell} ipython3
fig, (ax_position, ax_speed) = plt.subplots(2, 1, sharex=True)
ax_position.plot(nwb["position"], linewidth=0.5)
ax_position.set_ylabel("position [cm]")
ax_position.legend(labels=["X", "Y"], loc="upper right")
ax_speed.plot(nwb["body_speed"], linewidth=0.5)
ax_speed.set_ylabel("speed [cm/s]")
plt.xlabel("time [s]")
plt.show()
```

```{code-cell} ipython3
moving = nwb["body_speed"].threshold(3, method="above").time_support
tuning_curves = nap.compute_tuning_curves(
    data=nwb["units"],
    features=nwb["position"],
    feature_names=["X", "Y"],
    epochs=moving,
    bins=40,
)
tuning_curves
```

```{code-cell} ipython3
def gaussian_filter_nan(a, sigma):
    v = np.where(np.isnan(a), 0, a)
    w = gaussian_filter((~np.isnan(a)).astype(float), sigma)
    return gaussian_filter(v, sigma) / w

tuning_curves.values = gaussian_filter_nan(tuning_curves.values, sigma=(0, 2, 2))
```

```{code-cell} ipython3
tuning_curves.name = "firing rate"
tuning_curves.attrs["unit"] = "Hz"
tuning_curves.coords["X"].attrs["unit"] = "cm"
tuning_curves.coords["Y"].attrs["unit"] = "cm"
tuning_curves.plot(row="unit", col_wrap=5)
plt.show()
```

```{code-cell} ipython3
mutual_information = nap.compute_mutual_information(tuning_curves)
mutual_information
```

```{code-cell} ipython3
N = 50
percentile = 99
shuffles = [nap.shift_timestamps(nwb["units"], min_shift=20.0) for _ in range(N)]

null_distribution = np.stack(
   [
       nap.compute_mutual_information(
           nap.compute_tuning_curves(
               data=shuffle,
               features=nwb["position"],
               epochs=moving,
               bins=40,
           )
       )["bits/spike"].values
       for shuffle in shuffles
   ],
   axis=1,
)
thresholds = np.percentile(null_distribution, percentile, axis=1)
spatial_units = tuning_curves.coords["unit"][
   mutual_information["bits/spike"].values > thresholds
]
```

```{code-cell} ipython3
plt.hist(null_distribution[2])
plt.axvline(thresholds[2], color="black")
plt.axvline(mutual_information["bits/spike"][2], color="red")
plt.show()
```

```{code-cell} ipython3
tuning_curves.sel(unit=spatial_units).plot(row="unit", col_wrap=5)
plt.show()
```

```{code-cell} ipython3
def find_local_peaks(arr, min_distance=2, threshold_rel=0.1):
    arr = np.nan_to_num(arr)
    neighborhood = maximum_filter(arr, size=min_distance)
    mask = (arr == neighborhood) & (arr > threshold_rel * arr.max())
    labeled, _ = label(mask)
    return np.array(center_of_mass(mask, labeled, range(1, labeled.max() + 1)))


def compute_grid_score(tuning_curve):
    # autocorrelation, centered
    tc = np.nan_to_num(tuning_curve - np.nanmean(tuning_curve))
    autocorr = correlate2d(tc, tc, mode="full", boundary="fill", fillvalue=0)
    center = np.array(autocorr.shape) // 2

    # find peaks
    peaks = find_local_peaks(autocorr)
    if len(peaks) < 7:
        return np.nan

    # sort by distance from center
    peaks = np.array(peaks)
    distances = np.linalg.norm(peaks - center, axis=1)
    sorted_idx = np.argsort(distances)[1:7]  # skip central peak
    peaks = peaks[sorted_idx]
    mean_distance = np.mean(distances[sorted_idx])

    # define ring mask
    y, x = np.ogrid[: autocorr.shape[0], : autocorr.shape[1]]
    inner, outer = mean_distance * 0.5, mean_distance * 1.25
    mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2 >= inner**2) & (
        (x - center[1]) ** 2 + (y - center[0]) ** 2 <= outer**2
    )
    ring = np.where(mask, autocorr, np.nan)

    # correlation at rotation angles
    angles = [30, 60, 90, 120, 150]
    valid_mask = ~np.isnan(ring)
    ring_filled = np.nan_to_num(ring)
    angle_scores = {}

    for ang in angles:
        rot_ring = rotate(ring_filled, ang, reshape=False, mode="constant", cval=0.0)
        rot_mask = (
            rotate(
                valid_mask.astype(float), ang, reshape=False, mode="constant", cval=0.0
            )
            >= 0.5
        )
        combined = mask & rot_mask & valid_mask
        if np.sum(combined) < 10:
            angle_scores[ang] = np.nan
        else:
            angle_scores[ang] = np.corrcoef(ring[combined], rot_ring[combined])[0, 1]

    # grid score
    return np.nanmin([angle_scores[60], angle_scores[120]]) - np.nanmax(
        [angle_scores[30], angle_scores[90], angle_scores[150]]
    )
```

```{code-cell} ipython3
grid_scores = [compute_grid_score(tuning_curve) for tuning_curve in tuning_curves]
null_distribution = np.stack(
    [
        [
            compute_grid_score(tuning_curve)
            for tuning_curve in nap.compute_tuning_curves(
                data=shuffle,
                features=nwb["position"],
                epochs=moving,
                bins=40,
            )
        ]
        for shuffle in shuffles
    ],
    axis=1,
)
thresholds = np.percentile(null_distribution, percentile, axis=1)
grid_units = tuning_curves.coords["unit"][grid_scores > thresholds]
```

```{code-cell} ipython3
plt.hist(null_distribution[0])
plt.axvline(thresholds[0], color="black")
plt.axvline(grid_scores[0], color="red")
plt.show()
```

```{code-cell} ipython3
tuning_curves.sel(unit=grid_units).plot(row="unit", col_wrap=5)
plt.show()
```

<!-- #region -->
:::{card}
Authors
^^^
Wolf de Wulf
:::
<!-- #endregion -->
