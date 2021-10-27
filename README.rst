========
|pic1|
========

.. |pic1| image:: pynapple_logo.png
   :width: 80%


.. image:: https://img.shields.io/pypi/v/pynapple.svg
        :target: https://pypi.python.org/pypi/pynapple

.. image:: https://img.shields.io/travis/gviejo/pynapple.svg
        :target: https://travis-ci.com/gviejo/pynapple

.. image:: https://readthedocs.org/projects/pynapple/badge/?version=latest
        :target: https://pynapple.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




PYthon Neural Analysis Package Pour Laboratoires d’Excellence

pynapple is a Python library for analysing neurophysiological data. It allows to handle time series and epochs but also to use generic functions for neuroscience such as tuning curves and cross-correlogram of spikes. It is heavily based on neuroseries.



* Free software: GNU General Public License v3
* Documentation: https://pynapple.readthedocs.io.

----------------------------

Getting Started
===============

Requirements
------------

* Python 3.6+
* Pandas 1.0.3+
* numpy 1.17+
* scipy 1.3+
* numba 0.46+

Installation
------------

pynapple can be installed with pip:

.. code-block:: shell

    $ pip install starstruct

or directly from the source code:

.. code-block:: shell

    $ # clone the repository
    $ git clone https://github.com/PeyracheLab/pynapple.git
    $ cd pynapple
    $ # Install in editable mode with `-e` or, equivalently, `--editable`
    $ pip install -e


Features
========

* Automatic handling of spike times and epochs
* Tuning curves
* Loading data coming from various pipelines

Basic Usage
===========


After installation, the package can imported:

.. code-block:: shell

    $ python
    >>> import pynapple as ap

An example of the package can be seen below
    
.. code-block:: python

    import numpy as np
    import pandas as pd
    import pynapple as ap
    from pylab import *
    import sys
    
    data_directory = 'data/A2929-200711'
    
    
    episodes = ['sleep', 'wake']
    events = ['1']
    
    # Loading Data
    
    spikes, shank = ap.loadSpikeData(data_directory)
    position = ap.loadPosition(data_directory, events, episodes)
    wake_ep = ap.loadEpoch(data_directory, 'wake', episodes)
    sleep_ep = ap.loadEpoch(data_directory, 'sleep')					
    
    # Computing tuning curves
    
    tuning_curves = ap.computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
    tuning_curves = ap.smoothAngularTuningCurves(tuning_curves, 10, 2)





Credits
-------

