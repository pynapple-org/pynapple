========
|pic1|
========

.. |pic1| image:: pynapple_logo.png
   :width: 80%


.. image:: https://img.shields.io/pypi/v/pynapple.svg
        :target: https://pypi.python.org/pypi/pynapple

.. image:: https://img.shields.io/travis/gviejo/pynapple.svg
        :target: https://travis-ci.com/gviejo/pynapple


PYthon Neural Analysis Package

pynapple is a Python library for analysing neurophysiological data. It allows to handle time series and epochs but also to use generic functions for neuroscience such as tuning curves and cross-correlogram of spikes. It is heavily based on neuroseries.



* Free software: GNU General Public License v3
* Documentation: https://peyrachelab.github.io/pynapple/html/index.html

----------------------------

Getting Started
===============

Requirements
------------

* Pandas 
* numpy
* scipy
* matplotlib
* numba
* pytables
* tabulate
* pycircstat (pip only)
* nose

Installation
------------

pynapple can be installed with pip:

.. code-block:: shell

    $ pip install pynapple

or directly from the source code:

.. code-block:: shell

    $ # fork the repository at PeyracheLab to your github account
    $ # clone your github repository
    $ git clone https://github.com/YOUR USERNAME/pynapple.git
    $ cd pynapple
    $ # open git bash terminal (or Github Desktop or any other software you use) and run the following line
    $ git remote add upstream https://github.com/PeyracheLab/pynapple.git
    $ # Install in editable mode with `-e` or, equivalently, `--editable`
    $ pip install -e

* We highly recommend to create an environment before doing this. Using environments is a great way to avoid future conflicts with the requirements of other projects in which you are working on.

One way is to do it through Anaconda navigator. 
1. Go to the environments section.
2. Clic on create button (bottom left). 
3. Select a fancy name and the recommended python version (3.6+).
You can even manage the package versions with Anaconda navigator. 

The other way is to do it through the terminal. You can follow this documentation for that:  https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Features
========

* Automatic handling of spike times and epochs
* Tuning curves
* Loading data coming from various pipelines
* More and more coming!

Basic Usage
===========


After installation, the package can imported:

.. code-block:: shell

    $ python
    >>> import pynapple as ap

An example of the package can be seen below. The exemple data can be found `here <https://www.dropbox.com/s/1kc0ulz7yudd9ru/A2929-200711.tar.gz?dl=1>`_.
    
.. code-block:: python

    import numpy as np
    import pandas as pd
    import pynapple as nap
    import sys
    
    data_directory = 'data/A2929-200711'
    
    
    episodes = ['sleep', 'wake']
    events = ['1']
    
    # Loading Data
    spikes = nap.loadSpikeData(data_directory)   
    position = nap.loadPosition(data_directory, events, episodes)
    wake_ep = nap.loadEpoch(data_directory, 'wake', episodes)
   
    # Computing tuning curves
    tuning_curves = nap.computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
    tuning_curves = nap.smoothAngularTuningCurves(tuning_curves, 10, 2)






Credits
-------
Thanks Francesco Battaglia for neuroseries.
Thanks for Sofia for the logo
