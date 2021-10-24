.. 
  image:: https://badge.fury.io/py/tensorly.svg
    :target: https://badge.fury.io/py/tensorly

.. 
  image:: https://anaconda.org/tensorly/tensorly/badges/version.svg   
    :target: https://anaconda.org/tensorly/tensorly

.. 
  image:: https://github.com/tensorly/tensorly/workflows/Test%20TensorLy/badge.svg
    :target: https://github.com/tensorly/tensorly/actions?query=workflow%3A%22Test+TensorLy%22

.. 
  image:: https://codecov.io/gh/tensorly/tensorly/branch/master/graph/badge.svg?token=mnZ234sGSA
     :target: https://codecov.io/gh/tensorly/tensorly



========
pynapple 
========
PYthon Neural Analysis Package Pour Laboratoires d’Excellence

pynapple is a Python library for analysing neurophysiological data. It allows to handle time series and epochs but also to use generic functions for neuroscience such as tuning curves and cross-correlogram of spikes. It is heavily based on `neuroseries <https://pypi.org/project/neuroseries/>`_.

..
  - **Website:** http://tensorly.org
- **Source-code:**  https://github.com/PeyracheLab/pynapple
..
  - **Jupyter Notebooks:** https://github.com/JeanKossaifi/tensorly-notebooks

----------------------------

Installing pynapple
===================

The only pre-requisite is to have **Python 3** installed. 

+-------------------------------------------+---------------------------------------------------+
|              **With conda**    (TODO)     |                                                   |
+-------------------------------------------+---------------------------------------------------+
|                                           |                                                   |
| .. code::                                 |                                                   |
|                                           |                                                   |
|   conda install -c conda-forge pynapple   |                                                   |
|                                           |                                                   |
|                                           |                                                   |
+-------------------------------------------+---------------------------------------------------+
|                               **Development (from git)**                                      |
+-------------------------------------------+---------------------------------------------------+
|                                                                                               |
|          .. code::                                                                            |
|                                                                                               |
|             # clone the repository                                                            |
|             git clone git@github.com:PeyracheLab/pynapple.git                                 |
|             cd pynapple                                                                       |
|             python setup.py sdist bdist_wheel                                                 |
|                                                                                               |
+-----------------------------------------------------------------------------------------------+  
 
..
  For detailed instruction, please see the `documentation <http://tensorly.org/dev/installation.html>`_.

------------------

Quickstart
==========

The best way to learn how to use the package is to do the tutorials in the following order :

1. `main1_basics <https://github.com/PeyracheLab/StarterPack/blob/master/python/main1_basics.py>`_ - *Nice and gentle walktrough of python, numpy and matplotlib.*
2. `main2_neuroseries <https://github.com/PeyracheLab/StarterPack/blob/master/python/main2_neuroseries.py>`_ - *Introduction to neuroseries for handling spike times, time series and epochs.*
3. `main3_tuningcurves <https://github.com/PeyracheLab/StarterPack/blob/master/python/main3_tuningcurves.py>`_ - *How to make an angular tuning curve?*
4. `main4_raw_data <https://github.com/PeyracheLab/StarterPack/blob/master/python/main4_raw_data.py>`_ - *How to load data coming from the preprocessing pipeline (i.e. .res, .clu files)?*
5. `main5_matlab_data <https://github.com/PeyracheLab/StarterPack/blob/master/python/main5_matlab_data.py>`_ - *Too bad, Adrien asked you to analyse his old data saved in matlab...*
6. `main6_autocorr <https://github.com/PeyracheLab/StarterPack/blob/master/python/main6_autocorr.py>`_ - *How to make an auto-correlogram ?*
7. `main7_replay <https://github.com/PeyracheLab/StarterPack/blob/master/python/main7_replay.py>`_ - *How to do bayesian decoding?*
