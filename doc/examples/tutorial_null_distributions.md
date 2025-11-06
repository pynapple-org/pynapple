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

This tutorial demonstrates how we use Pynapple to generate Figure 4a in the [publication](https://elifesciences.org/reviewed-preprints/85786).
The NWB file for the example is hosted on [OSF](https://osf.io/jb2gd). We show below how to stream it.
The entire dataset can be downloaded [here](https://dandiarchive.org/dandiset/000056).

```{code-cell} ipython3
:tags: [remove-output]
import scipy
from pathlib import Path
import pandas as pd
import numpy as np
import pynapple as nap
import matplotlib.pyplot as plt
import requests, os
import seaborn as sns
import xarray as xr 

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
xr.set_options(display_expand_attrs=False)
```

```{code-cell} ipython3
# These modules are used for data loading
from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
from pynwb import NWBHDF5IO
import h5py

dandiset_id, filepath = ( "000582", "sub-10073/sub-10073_ses-17010302_behavior+ecephys.nwb")
with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage=str(Path("~/.caches/nwb-cache").expanduser()),
)
io = NWBHDF5IO(file=h5py.File(fs.open(s3_url, "rb")), load_namespaces=True)
nwb = nap.NWBFile(io.read())
```

```{code-cell} ipython3
print(nwb)
```

<!-- #region -->
:::{card}
Authors
^^^
Wolf de Wulf
:::
<!-- #endregion -->
