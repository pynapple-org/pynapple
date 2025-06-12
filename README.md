

[![image](https://img.shields.io/pypi/v/pynapple.svg)](https://pypi.python.org/pypi/pynapple)
[![pynapple CI](https://github.com/pynapple-org/pynapple/actions/workflows/main.yml/badge.svg)](https://github.com/pynapple-org/pynapple/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/pynapple-org/pynapple/branch/main/graph/badge.svg?token=VN9BDBOEGZ)](https://codecov.io/gh/pynapple-org/pynapple)
[![GitHub issues](https://img.shields.io/github/issues/pynapple-org/pynapple)](https://github.com/pynapple-org/pynapple/issues)
![GitHub contributors](https://img.shields.io/github/contributors/pynapple-org/pynapple)
![Twitter Follow](https://img.shields.io/twitter/follow/thepynapple?style=social)

PYthon Neural Analysis Package.

pynapple is a light-weight python library for neurophysiological data analysis. The goal is to offer a versatile set of tools to study typical data in the field, i.e. time series (spike times, behavioral events, etc.) and time intervals (trials, brain states, etc.). It also provides users with generic functions for neuroscience such as tuning curves and cross-correlograms.

-   Free software: MIT License
-   __Documentation__: [<https://pynapple.org>](https://pynapple-org.github.io/pynapple/)

> **Note**
> :page_with_curl: If you are using pynapple, please cite the following [paper](https://elifesciences.org/reviewed-preprints/85786)

------------------------------------------------------------------------

New release :fire:
------------------

### pynapple >= 0.8.2

The objects `IntervalSet`, `TsdFrame` and `TsGroup` inherits a new metadata class. It is now possible to add labels for 
each interval of an `IntervalSet`, each column of a `TsdFrame` and each unit of a `TsGroup`.

See the [documentation](https://pynapple.org/user_guide/03_metadata.html) for more details

### pynapple >= 0.7

Pynapple now implements signal processing. For example, to filter a 1250 Hz sampled time series between 10 Hz and 20 Hz:

```python
nap.apply_bandpass_filter(signal, (10, 20), fs=1250)
```
New functions includes power spectral density and Morlet wavelet decomposition. See the [documentation](https://pynapple-org.github.io/pynapple/reference/process/) for more details.

### pynapple >= 0.6

Starting with 0.6, [`IntervalSet`](https://pynapple-org.github.io/pynapple/reference/core/interval_set/) objects are behaving as immutable numpy ndarray. Before 0.6, you could select an interval within an `IntervalSet` object with:

```python
new_intervalset = intervalset.loc[[0]] # Selecting first interval
```

With pynapple>=0.6, the slicing is similar to numpy and it returns an `IntervalSet`

```python
new_intervalset = intervalset[0]
```

### pynapple >= 0.4

Starting with 0.4, pynapple rely on the [numpy array container](https://numpy.org/doc/stable/user/basics.dispatch.html) approach instead of Pandas for the time series. Pynapple builtin functions will remain the same except for functions inherited from Pandas. 

This allows for a better handling of returned objects.

Additionaly, it is now possible to define time series objects with more than 2 dimensions with `TsdTensor`. You can also look at this [notebook](https://pynapple-org.github.io/pynapple/generated/gallery/tutorial_pynapple_numpy/) for a demonstration of numpy compatibilities.

Community
---------

To ask any questions or get support for using pynapple, please consider joining our slack. Please send an email to thepynapple[at]gmail[dot]com to receive an invitation link.

Getting Started
---------------

### Installation

The best way to install pynapple is with pip inside a new [conda](https://docs.conda.io/en/latest/) environment:
    
``` {.sourceCode .shell}
$ conda create --name pynapple pip python=3.8
$ conda activate pynapple
$ pip install pynapple
```

> **Note**
> The package uses a pyproject.toml file for installation and dependencies management.

Running `pip install pynapple` will install all the dependencies, including: 

-   pandas
-   numpy
-   scipy
-   numba
-   pynwb 2.0
-   tabulate
-   h5py

For development, see the [contributor guide](CONTRIBUTING.md) for steps to install from source code.

<!-- For spyder users, it is recommended to install spyder after installing pynapple with :

``` {.sourceCode .shell}
$ conda create --name pynapple pip python=3.8
$ conda activate pynapple
$ pip install pynapple
$ pip install spyder
$ spyder
``` -->


Basic Usage
-----------

After installation, you can now import the package: 

``` {.sourceCode .shell}
$ python
>>> import pynapple as nap
```

You'll find an example of the package below. Click [here](https://osf.io/fqht6) to download the example dataset. The folder includes a NWB file containing the data.

``` py
import matplotlib.pyplot as plt
import numpy as np

import pynapple as nap

# LOADING DATA FROM NWB
data = nap.load_file("A2929-200711.nwb")

spikes = data["units"]
head_direction = data["ry"]
wake_ep = data["position_time_support"]

# COMPUTING TUNING CURVES
tuning_curves = nap.compute_1d_tuning_curves(
    spikes, head_direction, 120, ep=wake_ep, minmax=(0, 2 * np.pi)
)


# PLOT
plt.figure()
for i in spikes:
    plt.subplot(3, 5, i + 1, projection="polar")
    plt.plot(tuning_curves[i])
    plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])

plt.show()
```
Shown below, the final figure from the example code displays the firing rate of 15 neurons as a function of the direction of the head of the animal in the horizontal plane.

<!-- ![pic1](readme_figure.png) -->
<p align="center">
  <img width="80%" src="doc/_static/readme_figure.png">
</p>


### Credits

Special thanks to Francesco P. Battaglia
(<https://github.com/fpbattaglia>) for the development of the original
*TSToolbox* (<https://github.com/PeyracheLab/TStoolbox>) and
*neuroseries* (<https://github.com/NeuroNetMem/neuroseries>) packages,
the latter constituting the core of *pynapple*.

This package was developped by Guillaume Viejo
(<https://github.com/gviejo>) and other members of the Peyrache Lab.

<!-- Logo: Sofia Skromne Carrasco, 2021. -->

## Contributing

We welcome contributions, including documentation improvements. For more information, see the [contributor guide](CONTRIBUTING.md).
