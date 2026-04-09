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
lfp = data["eeg"][:, 0]
lfp
```

```{code-cell} ipython3
lfp.rate
```

Frequency filtering
-------------------
To look for ripples we will only keep frequencies within 150 to 250Hz:
```{code-cell} ipython3
filtered_lfp = nap.apply_bandpass_filter(lfp, cutoff=(120, 250), fs=lfp.rate)
filtered_lfp
```

```{code-cell} ipython3
:tags: [hide-input]
segment = nap.IntervalSet(17233.4, 17234.2)
fig = plt.figure(figsize=(10, 5))
plt.plot(lfp.restrict(segment), label="LFP")
plt.plot(filtered_lfp.restrict(segment), label="filtered LFP")
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.legend()
plt.title("nap.apply_bandpass_filter(lfp, cutoff=(120, 150), fs=lfp.rate")
plt.tight_layout();
```

Hilbert transform: computing the envelope
-----------------------------------------
Now, we will apply the Hilbert transform to the filtered LFP and take its absolute value 
to extract the amplitude envelope, which reflects ripple strength over time. 
Pynapple provides [`compute_hilbert_envelope`](pynapple.process.signal.compute_hilbert_envelope]:

```{code-cell} ipython3
envelope = nap.compute_hilbert_envelope(filtered_lfp)
envelope
```

See [`scipy.signal.hilbert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html) for
more information on what the Hilbert transform does!

We will plot the envelope over the filtered signal for visual confirmation:
```{code-cell} ipython3
:tags: [hide-input]
fig = plt.figure(figsize=(10, 5))
plt.plot(filtered_lfp.restrict(segment), label="filtered LFP")
plt.plot(envelope.restrict(segment), label="envelope", color="red")
plt.xlabel("time (s)")
plt.ylabel("amplitude (a.u.)")
plt.legend()
plt.title("nap.compute_hilbert_envelope(filtered_lfp)")
plt.tight_layout();
```

Smooth and Z-score
------------------
We will then smooth and z-score the envelope:

```{code-cell} ipython3
smoothed = envelope.smooth(0.005)
zscored_smoothed = (smoothed - smoothed.mean()) / smoothed.std()
```

Let's visualize everything up until now as an overview:
```{code-cell} ipython3
:tags: [hide-input]
fig, axs = plt.subplots(3, 1, figsize=(10,9), sharex=True)
axs[0].plot(lfp.restrict(segment))
axs[0].set_title("LFP")
axs[1].plot(filtered_lfp.restrict(segment))
axs[1].plot(envelope.restrict(segment), color="red")
axs[1].set_title("filtered LFP + envelope")
axs[2].plot(zscored_smoothed.restrict(segment))
axs[2].set_title("smoothed & z-scored")
axs[2].set_xlabel("time (s)")
plt.tight_layout();
```

Ripple detection
----------------
We detect ripple events by thresholding the z-scored smoothed signal with a threshold of 2 standard deviations.
We further filter detected events to keep only those between 30 ms and 300 ms in duration, typical for hippocampal ripples. 
```{code-cell} ipython3
threshold = 3
ripple_events = zscored_smoothed.threshold(threshold, method="above")
ripple_epochs = ripple_events.time_support
ripple_epochs = ripple_epochs.drop_short_intervals(0.03, time_units="s")
ripple_epochs = ripple_epochs.drop_long_intervals(0.2, time_units="s")    
ripple_epochs.intersect(segment)
```

Finally, we can plot the detected ripple events on top of the filtered LFP signal for visual confirmation:
```{code-cell} ipython3
:tags: [hide-input]
fig = plt.figure(figsize=(10, 5))
plt.plot(zscored_smoothed.restrict(segment), label="z-scored & smoothed")
for start, end in ripple_epochs.intersect(segment).values:
    plt.axvspan(start, end, alpha=0.3, color="red")
plt.axhline(threshold, color="black", linestyle="--")
plt.ylabel("amplitude (a.u.)")
plt.xlabel("time (s)")
plt.tight_layout();
```

:::{card}
Authors
^^^
[Wolf De Wulf](https://wulfdewolf.github.io)

Guillaume Viejo
:::
