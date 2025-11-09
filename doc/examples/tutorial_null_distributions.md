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

Null distributions to classify grid cells
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

```{code-cell} ipython3
def gaussian_filter_nan(a, sigma):
    v = np.where(np.isnan(a), 0, a)
    w = gaussian_filter((~np.isnan(a)).astype(float), sigma)
    return gaussian_filter(v, sigma) / w


tuning_curves = nap.compute_tuning_curves(
    data=units,
    features=position,
    feature_names=["X", "Y"],
    range=[(-50, 50), (-50, 50)],
    bins=40,
)
tuning_curves.name = "firing rate"
tuning_curves.attrs["unit"] = "Hz"
tuning_curves.coords["X"].attrs["unit"] = "cm"
tuning_curves.coords["Y"].attrs["unit"] = "cm"
tuning_curves.values = gaussian_filter_nan(tuning_curves.values, sigma=(0, 2, 2))
tuning_curves.plot(col="unit", figsize=(14,3))
plt.show()
```

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

```{code-cell} ipython3
N = 500
percentile = 99
shuffles = [nap.shift_timestamps(units, min_shift=20.0) for _ in range(N)]

def compute_grid_score(tuning_curve, min_distance=2, threshold_rel=0.05):
    # Autocorrelation
    tc = np.nan_to_num(tuning_curve - np.nanmean(tuning_curve))
    autocorr = correlate2d(tc, tc, mode="full", boundary="fill", fillvalue=0)
    center = np.array(autocorr.shape) // 2
    
    # Find local peaks
    neighborhood = maximum_filter(autocorr, size=min_distance)
    mask = (autocorr == neighborhood) & (autocorr > threshold_rel * autocorr.max())
    labeled, num = label(mask)
    if num < 7:
        return np.nan
    peaks = np.array(center_of_mass(mask, labeled, range(1, num + 1)))
    
    # Calculate ring from peak distances
    distances = np.linalg.norm(peaks - center, axis=1)
    mean_dist = np.mean(np.sort(distances)[1:7])
    
    # Ring mask
    y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
    dist_sq = (x - center[1])**2 + (y - center[0])**2
    ring_mask = (dist_sq >= (mean_dist * 0.5)**2) & (dist_sq <= (mean_dist * 1.25)**2)
    
    # Correlations at key angles on ring
    ring = autocorr[ring_mask]
    corrs = {}
    for ang in [30, 60, 90, 120, 150]:
        rot = rotate(autocorr, ang, reshape=False, cval=0)[ring_mask]
        corrs[ang] = np.corrcoef(ring, rot)[0, 1] if len(ring) >= 10 else np.nan
    
    # Grid score: hexagonal - control angles
    return np.nanmin([corrs[60], corrs[120]]) - np.nanmax([corrs[30], corrs[90], corrs[150]])

grid_scores = [compute_grid_score(tuning_curve) for tuning_curve in tuning_curves]
```

```{code-cell} ipython3
null_distribution = np.stack(
    [
        [
            compute_grid_score(tuning_curve)
            for tuning_curve in nap.compute_tuning_curves(
                data=shuffle,
                features=position,
                range=[(-50, 50), (-50, 50)],
                bins=40,
            )
        ]
        for shuffle in shuffles
    ],
    axis=1,
)
thresholds = np.nanpercentile(null_distribution, percentile, axis=1)
grid_units = tuning_curves.coords["unit"][grid_scores > thresholds]
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, len(units), figsize=(3 * len(units), 8), sharex="row", sharey="row")
for (ax_tc, ax_hist), (i, unit) in zip(axes.T, enumerate(units)):
    score = grid_scores[i]
    ax_tc.imshow(tuning_curves.sel(unit=unit), cmap="viridis", aspect="equal")
    ax_tc.set_title(
        f"unit = {unit}\nscore={score:.2f}{'*' if score > thresholds[i] else ''}"
    )
    ax_tc.axis("off")
    ax_hist.hist(null_distribution[i], histtype="stepfilled", edgecolor="none", bins=30)
    ax_hist.axvline(thresholds[i], color="black")
    ax_hist.axvline(score, color="red")
    ax_hist.yaxis.set_visible(False)
    ax_hist.spines['left'].set_visible(False)
axes[1, 0].set_xlabel("grid score")
axes[1, 0].set_xlim(-1, 1)
plt.show()
```

<!-- #region -->
:::{card}
Authors
^^^
Wolf De Wulf
:::
<!-- #endregion -->
