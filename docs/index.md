---
hide:
  - navigation
  - toc
---

# <div style="text-align: center;"> <img src="images/Pynapple_logo_final.svg" width="50%" alt="Pynapple logo."> </div>


<div style="text-align: center;" markdown>
   
[![image](https://img.shields.io/pypi/v/pynapple.svg)](https://pypi.python.org/pypi/pynapple)
[![pynapple CI](https://github.com/pynapple-org/pynapple/actions/workflows/main.yml/badge.svg)](https://github.com/pynapple-org/pynapple/actions/workflows/main.yml)
[![Coverage Status](https://coveralls.io/repos/github/pynapple-org/pynapple/badge.svg?branch=main)](https://coveralls.io/github/pynapple-org/pynapple?branch=main)
[![GitHub issues](https://img.shields.io/github/issues/pynapple-org/pynapple)](https://github.com/pynapple-org/pynapple/issues)
![GitHub contributors](https://img.shields.io/github/contributors/pynapple-org/pynapple)
![Twitter Follow](https://img.shields.io/twitter/follow/thepynapple?style=social)

[:material-book-open-variant-outline: __Cite the paper__](https://elifesciences.org/reviewed-preprints/85786)

</div>



## __Overview__


pynapple is a light-weight python library for neurophysiological data analysis. The goal is to offer a versatile set of tools to study typical data in the field, i.e. time series (spike times, behavioral events, etc.) and time intervals (trials, brain states, etc.). It also provides users with generic functions for neuroscience such as tuning curves and cross-correlograms.


<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Getting Started__

      ---

      New to Pynapple? Checkout the quickstart.

      [:octicons-arrow-right-24: Quickstart](quickstart)

-   :material-lightbulb-on-10:{ .lg .middle } &nbsp; __How-To Guide__

    ---

    Learn the pynapple API with notebooks.

    [:octicons-arrow-right-24: API guide](generated/api_guide/)

-   :material-brain:{ .lg .middle} &nbsp;  __Neural Analysis__

Starting with 0.6, [`IntervalSet`](reference/core/interval_set/) objects are behaving as immutable numpy ndarray. Before 0.6, you could select an interval within an `IntervalSet` object with:

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

Additionaly, it is now possible to define time series objects with more than 2 dimensions with `TsdTensor`. You can also look at this [notebook](generated/api_guide/tutorial_pynapple_numpy/) for a demonstration of numpy compatibilities.

Getting Started
---------------

### Installation

The best way to install pynapple is with pip within a new [conda](https://docs.conda.io/en/latest/) environment :
=======
    ---
>>>>>>> dev

    Explore fully worked examples to learn how to analyze neural recordings using pynapple.
    
    [:octicons-arrow-right-24: Tutorials](https://pynapple.org/generated/examples/)

-   :material-cog:{ .lg .middle } &nbsp; __API__

    ---

    Access a detailed description of each module and function, including parameters and functionality. 

    [:octicons-arrow-right-24: Modules](reference/)

-   :material-hammer-wrench:{ .lg .middle } &nbsp; __Installation Instructions__ 

    ---
    
    Run the following `pip` command in your virtual environment.
    === "macOS/Linux"

        ```bash
        pip install pynapple
        ```

    === "Windows"
    
        ```
        python -m pip install pynapple
        ```
    
    *For more information see:*<br>
    [:octicons-arrow-right-24: Install](installation)

-   :material-frequently-asked-questions:{ .lg .middle } &nbsp; __Community__

    ---

    To ask any questions or get support for using pynapple, please consider joining our slack. 

    Please send an email to thepynapple[at]gmail[dot]com to receive an invitation link.

    *To open an issue see :*<br>
    [:octicons-arrow-right-24: Issues](https://github.com/pynapple-org/pynapple/issues)

</div>



## :material-scale-balance:{ .lg } License

Open source, [licensed under MIT](https://github.com/pynapple-org/pynapple/blob/main/LICENSE).
