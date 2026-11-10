---
html_theme.sidebar_secondary.remove: true
---

Learning material
=================

Pynapple has been featured in many workshops, summer schools, and more.
Underneath you can find links to all the teaching material we generated for these events.
If you are not satisfied by the examples on the website, go and take a look at these!

If you made your own teaching material and think it could help others learn to use Pynapple, feel free to make a pull request to the [Pynapple GitHub](https://github.com/pynapple-org/pynapple) page so we can include it here.

## Basic tutorials

The following tutorials are meant to get you started with Pynapple. They are designed to be accessible to beginners, and they cover the basics of data loading, manipulation, and analysis.

| Topic  | Type  | Authors |
| :----- | :--: | :------ |
| [Learning the fundamentals of Pynapple](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/full/live_coding/01_fundamentals_of_pynapple.html) | web page | Guillaume Viejo |
| [Pynapple & NeMoS Cheat Sheet](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/_downloads/06faf07e60fcc887150dde0291f9610e/Feb_2026_workshop_cheatsheet.pdf) | cheat sheet | Aramis Tanelus |
| [Signal processing](https://colab.research.google.com/drive/1hlszOxRzIuoGsn-aO8PcEJ6m8mjukt8d) | notebook | Sarah Jo Venditto |
| [An introduction to Pynapple](https://wulfdewolf.github.io/pynapple-intro/) | web page | Wolf De Wulf |

## Curated examples

The following examples are curated from the Pynapple documentation and other teaching material. They are meant to be a starting point for your own analysis.

```{toctree}
:hidden:
:maxdepth: 3
examples/tutorial_HD_dataset
examples/tutorial_pynapple_dandi
examples/tutorial_null_distributions
examples/tutorial_calcium_imaging
examples/tutorial_wavelet_decomposition
examples/tutorial_phase_preferences
examples/tutorial_ibl_choice_decoding
examples/tutorial_ripple_detection
```

::::::{card} Analysing head-direction cells
:link: examples/tutorial_HD_dataset
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Compute head-direction tuning curves from a freely-moving recording, streaming the NWB file from OSF.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_HD_dataset.png
:alt: Analysing head-direction cells
:class: example-thumb dark-light
:::
::::

:::::
::::::

::::::{card} Streaming data from DANDI
:link: examples/tutorial_pynapple_dandi
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Stream an NWB file directly from the DANDI Archive into pynapple, without downloading the whole dataset.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_pynapple_dandi.png
:alt: Streaming data from DANDI
:class: example-thumb dark-light
:::
::::

:::::
::::::

::::::{card} Null distributions to test spatial firing
:link: examples/tutorial_null_distributions
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Use the randomization module to build null distributions and test whether firing is modulated by position.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_null_distributions.png
:alt: Null distributions to test spatial firing
:class: example-thumb dark-light
:::
::::

:::::
::::::

::::::{card} Calcium Imaging
:link: examples/tutorial_calcium_imaging
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Extract head-direction tuning from a one-photon Miniscope recording of a freely-moving mouse.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_calcium_imaging.png
:alt: Calcium Imaging
:class: example-thumb dark-light
:::
::::

:::::
::::::

::::::{card} Wavelet Transform
:link: examples/tutorial_wavelet_decomposition
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Apply the wavelet transform to hippocampal LFP recorded during active traversal of a linear track.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_wavelet_decomposition.png
:alt: Wavelet Transform
:class: example-thumb dark-light
:::
::::

:::::
::::::

::::::{card} Spikes-phase coupling
:link: examples/tutorial_phase_preferences
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Band-pass filter LFP from REM sleep to extract phase, and find the phase preference of spiking units.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_phase_preferences.png
:alt: Spikes-phase coupling
:class: example-thumb dark-light
:::
::::

:::::
::::::

::::::{card} Trial-aligned decoding with International Brain Lab data
:link: examples/tutorial_ibl_choice_decoding
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Trial-align spiking activity from the IBL decision task and decode the animal's choice with logistic regression.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_ibl_choice_decoding.png
:alt: Trial-aligned decoding with International Brain Lab data
:class: example-thumb dark-light
:::
::::

:::::
::::::

::::::{card} Detecting sharp-wave ripples
:link: examples/tutorial_ripple_detection
:link-type: doc

:::::{grid} 1 1 2 2
:gutter: 2

::::{grid-item}
:columns: 12 12 6 6

Detect hippocampal sharp-wave ripples in the Grosmark & Buzsáki dataset using pynapple's filtering tools.
::::

::::{grid-item}
:columns: 12 12 6 6

:::{image} _static/example_thumbs/tutorial_ripple_detection.png
:alt: Detecting sharp-wave ripples
:class: example-thumb dark-light
:::
::::

:::::
::::::


| Topic  | Type  | Authors |
| :----- | :--: | :------ |
| [Analyzing head-direction cells with Pynapple](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/users/group_projects/01_head_direction-users.html) | web page | Guillaume Viejo |
| [Calcium imaging analysis of head-direction cells](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/users/group_projects/03_calcium_imaging_analysis-users.html) | web page | Guillaume Viejo |
| [Analyzing hippocampal place cells with Pynapple and NeMoS](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/users/group_projects/04_place_cells-users.html) | web page | Sarah Jo Venditto |
| [Exploring the Allen Institute’s Visual Coding dataset](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/users/group_projects/05_visual_coding-users.html) | web page | Guillaume Viejo |
| [Data wrangling, 1D neural tuning, and model fitting](https://colab.research.google.com/drive/1V0t7uE0nJels52r6BLYFhs5XRrG2UvsY) | notebook | Sarah Jo Venditto |
| [2D neural tuning and model fitting](https://colab.research.google.com/drive/1GcFV6rAU0xBWWYe6IsjBlRAFqwToAQ9J) | notebook | Sarah Jo Venditto |
| [Neural decoding](https://colab.research.google.com/drive/1T1ewqbuSuXB06BlETSMSVSQDukZwX093) | notebook | Sarah Jo Venditto |
| [Fitting injected currents using GLMs](https://nemos.readthedocs.io/en/latest/tutorials/plot_01_current_injection.html) | web page | Edoardo Balzani |
| [Fitting a head direction population using GLMs](https://nemos.readthedocs.io/en/latest/tutorials/plot_02_head_direction.html) | web page | Edoardo Balzani |
| [Fitting grid cells using GLMs](https://nemos.readthedocs.io/en/latest/tutorials/plot_03_grid_cells.html) | web page | Edoardo Balzani |
| [Fitting place cells using GLMs](https://nemos.readthedocs.io/en/latest/tutorials/plot_05_place_cells.html) | web page | Edoardo Balzani |
| [Fitting V1 cells using GLMs](https://nemos.readthedocs.io/en/latest/tutorials/plot_04_v1_cells.html) | web page | Edoardo Balzani |
| [Fitting calcium imaging using GLMs](https://nemos.readthedocs.io/en/latest/tutorials/plot_06_calcium_imaging.html) | web page | Edoardo Balzani |


## Workshop materials

The following materials were generated for workshops and summer schools.

| Title | Date |
| :---- | :--: |
| [CCN Software Workshop](https://flatironinstitute.github.io/neurorse-workshops/workshops/jan-2025/branch/main/index.html#) | Jan 2025 |
| [CCN Software Workshop @ SfN](https://flatironinstitute.github.io/neurorse-workshops/workshops/sfn-2025/branch/main/) | Nov 2025 |
| [CCN Software Workshop @ FENS](https://flatironinstitute.github.io/neurorse-workshops/workshops/fens-2026/branch/main/) | July 2026 |
| [CCN Software Workshop](https://flatironinstitute.github.io/neurorse-workshops/workshops/feb-2026/branch/main/) | Feb 2026 |