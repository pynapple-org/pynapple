:html_theme.sidebar_secondary.remove:


pynapple: python neural analysis package
========================================

.. toctree::
   :maxdepth: 1
   :hidden:

   Installing <installing>
   User guide <user_guide>
   Examples <examples>
   API <api>
   About us <about>
   Releases <releases>
   External projects <external>
   GPU acceleration <pynajax>
   Citing <citing>


|

.. grid:: 1 1 2 2

   .. grid-item::

      Pynapple is a light-weight python library for
      neurophysiological data analysis. 

      The goal is to offer a versatile set of tools to 
      study typical data in the field, i.e. time series 
      (spike times, behavioral events, etc.) and time intervals 
      (trials, brain states, etc.). 

      It also provides users with generic functions for 
      neuroscience such as tuning curves, cross-correlograms 
      and filtering.

      .. grid:: auto

         .. button-ref:: installing
            :color: primary
            :shadow:

            Installing

         .. button-ref:: user_guide/01_introduction_to_pynapple
            :color: primary
            :shadow:

            Getting started

         .. button-ref:: citing
            :color: primary
            :shadow:

            Citing


.. grid:: 1 1 6 6
    :gutter: 2

    .. grid-item-card:: Time Series
        :text-align: center
        :link: ./user_guide/01_introduction_to_pynapple.html#instantiating-pynapple-objects

        .. image:: _static/example_thumbs/tsd.svg
            :width: 100px
            :class: dark-light

    .. grid-item-card:: Intervals
        :text-align: center
        :link: ./user_guide/01_introduction_to_pynapple.html#nap-intervalset-intervals

        .. image:: _static/example_thumbs/interval.svg
            :width: 100px
            :class: dark-light

    .. grid-item-card:: Timestamps
        :text-align: center
        :link: ./user_guide/01_introduction_to_pynapple.html#nap-ts-timestamps

        .. image:: _static/example_thumbs/tsgroup.svg
            :width: 100px
            :class: dark-light

    .. grid-item-card:: Time alignment
        :text-align: center
        :link: ./user_guide/01_introduction_to_pynapple.html#restricting-time-series-to-epochs

        .. image:: _static/example_thumbs/interaction.svg
            :width: 100px
            :class: dark-light

.. grid:: 1 1 6 6
   :gutter: 1

   .. grid-item-card:: Decoding
      :text-align: center
      :link: ./user_guide/07_decoding.html

      .. image:: _static/example_thumbs/decoding.svg
         :width: 100px
         :class: dark-light

   .. grid-item-card:: Perievent
      :text-align: center
      :link: ./user_guide/08_perievent.html

      .. image:: _static/example_thumbs/perievent.svg
         :width: 100px
         :class: dark-light

   .. grid-item-card:: Correlation
      :text-align: center
      :link: ./user_guide/05_correlograms_isi.html

      .. image:: _static/example_thumbs/correlation.svg
         :width: 100px
         :class: dark-light

   .. grid-item-card:: Tuning curves
      :text-align: center
      :link: ./user_guide/06_tuning_curves.html

      .. image:: _static/example_thumbs/tuningcurves.svg
         :width: 100px
         :class: dark-light

   .. grid-item-card:: Spectrogram
      :text-align: center
      :link: ./user_guide/11_wavelets.html

      .. image:: _static/example_thumbs/wavelets.svg
         :width: 100px
         :class: dark-light

   .. grid-item-card:: Filtering
      :text-align: center
      :link: ./user_guide/12_filtering.html

      .. image:: _static/example_thumbs/filtering.svg
         :width: 100px
         :class: dark-light



Support
~~~~~~~

This package is supported by the Center for Computational Neuroscience, in the Flatiron Institute of the Simons Foundation

.. image:: _static/CCN-logo-wText.png
   :width: 200px
   :class: only-light
   :target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/

.. image:: _static/logo_flatiron_white.svg
   :width: 200px
   :class: only-dark
   :target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/

