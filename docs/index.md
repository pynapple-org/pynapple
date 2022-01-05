![pic1](pynapple_logo.png)
==========================

[![image](https://img.shields.io/pypi/v/pynapple.svg)](https://pypi.python.org/pypi/pynapple) [![image](https://img.shields.io/travis/gviejo/pynapple.svg)](https://travis-ci.com/gviejo/pynapple)

PYthon Neural Analysis Package.

pynapple is a Python library for analysing neurophysiological data. It
allows to handle time series and epochs but also to use generic
functions for neuroscience such as tuning curves and cross-correlogram
of spikes. It is heavily based on neuroseries.

-   Free software: GNU General Public License v3
-   Documentation:
    <https://peyrachelab.github.io/pynapple/html/index.html>

------------------------------------------------------------------------

Getting Started
---------------

### Requirements

-   Python 3.6+
-   Pandas 1.0.3+
-   numpy 1.17+
-   scipy 1.3+
-   numba 0.46+

### Installation

pynapple can be installed with pip:

``` {.sourceCode .shell}
$ pip install pynapple
```

or directly from the source code:

``` {.sourceCode .shell}
$ # clone the repository
$ git clone https://github.com/PeyracheLab/pynapple.git
$ cd pynapple
$ # Install in editable mode with `-e` or, equivalently, `--editable`
$ pip install -e
```

Features
--------

-   Automatic handling of spike times and epochs
-   Tuning curves
-   Loading data coming from various pipelines
-   More and more coming!

Basic Usage
-----------

After installation, the package can imported:

``` {.sourceCode .shell}
$ python
>>> import pynapple as nap
```

An example of the package can be seen below. The exemple data can be
found
[here](https://www.dropbox.com/s/1kc0ulz7yudd9ru/A2929-200711.tar.gz?dl=1).

```python
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.pyplot import *

data_directory = 'data/A2929-200711'

################################################################
# LOADING DATA
################################################################
data = nap.load_session(data_directory, 'neurosuite')


spikes = data.spikes
position = data.position
wake_ep = data.epochs['wake']

################################################################
# COMPUTING TUNING CURVES
################################################################
tuning_curves = nap.compute_1d_tuning_curves(spikes, position['ry'], position['ry'].time_support, 120)

        
################################################################
# PLOT
################################################################

figure()
for i in spikes:
    subplot(6,7,i+1, projection = 'polar')
    plot(tuning_curves[i])
    

show()

```

### Credits

Special thanks to Francesco P. Battaglia
(<https://github.com/fpbattaglia>) for the development of the original
*TSToolbox* (<https://github.com/PeyracheLab/TStoolbox>) and
*neuroseries* (<https://github.com/NeuroNetMem/neuroseries>) packages,
the latter constituting the core of *pynapple*.

This package was developped by Guillaume Viejo
(<https://github.com/gviejo>) and other members of the Peyrache Lab.

Logo: Sofia Skromne Carrasco, 2021.
