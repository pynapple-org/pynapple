# External Projects

Pynapple has been designed as a lightweight package for representing time series and epochs in system neuroscience.
As such, it can function as a foundational element for other analysis packages handling time series data. Here we keep track of external projects that uses pynapple.


## Pynaviz

<img src="https://media.githubusercontent.com/media/pynapple-org/pynaviz/main/docs/examples/example_lfp_short.gif" alt="Pynaviz overview" width="300" style="height:auto;" />

..

[Pynaviz](https://pynapple-org.github.io/pynaviz/) provides interactive, high-performance visualizations designed to work 
seamlessly with Pynapple time series and video data. It allows synchronized exploration of neural signals and behavioral recordings. 
It is build on top of pygfx, a modern GPU-based rendering engine.

The easiest way to get started with pynaviz is to use the `pynaviz` command line tool, which can be installed with pip:

```bash
$ pip install pynaviz[qt]
$ pynaviz
```

## NeMoS

![image](https://raw.githubusercontent.com/flatironinstitute/nemos/main/docs/assets/glm_features_scheme.svg)

[NeMoS](https://nemos.readthedocs.io/en/latest/index.html) is a statistical modeling framework optimized for systems neuroscience and powered by JAX. It streamlines the process of defining and selecting models, through a collection of easy-to-use methods for feature design.

The core of nemos includes GPU-accelerated, well-tested implementations of standard statistical models, currently focusing on the Generalized Linear Model (GLM). 

Check out this [page](https://nemos.readthedocs.io/en/latest/tutorials/README.html) for many examples of neural modelling using nemos and pynapple.

```{eval-rst}
.. Note::
	Nemos is build on top of [jax](https://jax.readthedocs.io/en/latest/index.html), a library for high-performance numerical computing.
	To ensure full compatibility with nemos, consider installing [pynajax](https://github.com/pynapple-org/pynajax), a pynapple backend for jax.
```

## SpikeInterface

<img src="https://github.com/SpikeInterface/spikeinterface/blob/758d9d5806aaf474e51f6d3e04dc7d37692e3ec2/doc/images/overview.png?raw=true" alt="SpikeInterface overview" width="600" style="height:auto;" />

[SpikeInterface](https://spikeinterface.readthedocs.io/en/latest/) is a Python library for spike sorting and electrophysiological data analysis. 

With a few lines of code, SpikeInterface enables you to load and pre-process the recording, run several state-of-the-art spike sorters, 
post-process and curate the output, compute quality metrics, and visualize the results.

SpikeInterface can export the output of spike sorting to pynapple, allowing you to seamlessly integrate spike sorting results into your pynapple-based analysis pipeline. 
See [here](https://spikeinterface.readthedocs.io/en/stable/modules/exporters.html#exporting-to-pynapple) for more details.

``` python
import spikeinterface as si
from spikeinterface.exporters import to_pynapple_tsgroup

# load in an analyzer
analyzer = si.load_sorting_analyzer("path/to/analyzer")

my_tsgroup = to_pynapple_tsgroup(
    sorting_analyzer=analyzer,
    attach_unit_metadata=True,
)

# Note: can add metadata using e.g.
# my_tsgroup.set_info({'brain_region': ['MEC', 'MEC', ...]})

my_tsgroup.save("my_tsgroup_output.npz")
```