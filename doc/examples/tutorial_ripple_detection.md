---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Detecting sharp-wave ripples
============================
This tutorial demonstrates how to use Pynapple to detect sharp-wave ripples.

```{code-cell} ipython3
:tags: [hide-input]
# we'll import the packages we're going to use
import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import pynapple as nap

# some configuration, you can ignore this
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params);
```

Downloading the data
--------------------
We will examine the dataset from [Grosmark & Buzsáki (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4919122/).
Let's download the data and save it locally:

```{code-cell} ipython3
path = "Achilles_10252013_EEG.nwb"
if path not in os.listdir("."):
    r = requests.get(f"https://osf.io/2dfvp/download", stream=True)
    block_size = 1024 * 1024
    with open(path, "wb") as f:
        for data in r.iter_content(block_size):
            f.write(data)
data = nap.load_file(path)
data
```
We only need to local field potential (LFP), which has been downsampled to 1250Hz:

```{code-cell} ipython3
lfp = data["eeg"]
lfp
```

```{code-cell} ipython3
lfp.rate
```

Frequency filtering
-------------------
To look for ripples we will filter out frequencies outside of the 150 to 250Hz:
```{code-cell} ipython3
filtered_lfp = nap.apply_bandpass_filter(lfp, cutoff=(150, 250), fs=lfp.rate)
filtered_lfp
```

```{code-cell} ipython3
:tags: [hide-input]
segment = nap.IntervalSet(18356.0, 18357.5)
fig = plt.figure(figsize=(10, 6))
plt.plot(lfp.restrict(segment), label="LFP")
plt.plot(filtered_lfp.restrict(segment), label="filtered LFP (150-250Hz)")
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.tight_layout()
plt.legend();
```

Hilbert transform: computing the envelope
-----------------------------------------
Now, we will apply the Hilbert transform to the filtered LFP to extract the amplitude envelope, 
which reflects ripple strength over time. 

```{code-cell} ipython3
from scipy.signal import hilbert
envelope = hilbert(filtered_lfp.values, axis=0)
envelope = nap.TsdFrame(t=lfp.times(), d=np.abs(envelope), columns=lfp.columns)
envelope
```

We will plot the envelope alongside the filtered signal for visual confirmation:
```{code-cell} ipython3
:tags: [hide-input]
fig = plt.figure(figsize=(10, 6))
plt.plot(filtered_lfp.restrict(segment), label="filtered LFP (150-250Hz)")
plt.plot(envelope.restrict(segment), label="envelope", color="red")
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.tight_layout()
plt.legend();
```

Smooth and Z-score
------------------
We will smooth and z-score the envelope:

```{code-cell} ipython3
filter = np.ones(7) / 7
smoothed = envelope.convolve(filter)
zscored = (smoothed - smoothed.mean()) / smoothed.std()
```

Let's visualize everything up until now as an overview:
```{code-cell} ipython3
:tags: [hide-input]
fig, axs = plt.subplots(3, 1, figsize=(10,6), sharex=True)
axs[0].plot(lfp.restrict(segment))
axs[0].set_title("LFP")
axs[1].plot(filtered_lfp.restrict(segment))
axs[1].plot(envelope.restrict(segment), color="red")
axs[1].set_title("filtered LFP + envelope")
axs[2].plot(zscored.restrict(segment))
axs[2].set_title("smoothed & z-scored")
axs[2].set_xlabel("time (s)")
plt.tight_layout();
```

Ripple detection
----------------

Ripple density
--------------

:::{card}
Authors
^^^
[Wolf De Wulf](wulfdewolf.github.io)

Guillaume Viejo
:::
