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
We will examine the dataset from [Grosmark & Buzsáki (2016)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4919122/).

Visualizing High Frequency Oscillation
-----------------------------------
There also seem to be peaks in the 200Hz frequency power after traversal of thew maze is complete. Here we use the interval (18356, 18357.5) seconds to zoom in.

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: false
---
import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import pynapple as nap

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)

```

:::{card}
Authors
^^^
[Wolf De Wulf](wulfdewolf.github.io)

Guillaume Viejo

:::
