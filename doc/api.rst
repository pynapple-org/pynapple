.. _api_ref:

API reference
=============

Core objects
------------

.. rubric:: Time Series

.. currentmodule:: pynapple.core.time_series

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    Tsd
    TsdFrame
    TsdTensor


.. rubric:: Intervals

.. currentmodule:: pynapple.core.interval_set

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    IntervalSet


.. rubric:: Timestamps

.. currentmodule:: pynapple.core.time_series

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    Ts

.. rubric:: Group of timestamps

.. currentmodule:: pynapple.core.ts_group

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    TsGroup


Input-Ouput
-----------

.. currentmodule:: pynapple.io.interface_nwb

.. rubric:: Neurodata Without Borders (NWB)

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    NWBFile


.. currentmodule:: pynapple.io.interface_npz

.. rubric:: Numpy files

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    NPZFile


.. currentmodule:: pynapple.io

.. rubric:: Miscellaneous

.. autosummary::
    :toctree: generated/
    :nosignatures:

    misc
    folder.Folder


Analysis modules
----------------

.. currentmodule:: pynapple.process

.. autosummary::
    :toctree: generated/
    :nosignatures:

    correlograms
    decoding
    filtering
    perievent
    randomize
    spectrum
    tuning_curves
    wavelets


