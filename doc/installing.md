# Installing and getting started

```
pip install pynapple
```

The best way to install pynapple is within a new [conda](https://docs.conda.io/en/latest/) environment:

```
conda create --name pynapple pip python
conda activate pynapple
pip install pynapple
```

:::{admonition} numba and llvmlite support for newer python versions
:class: warning 

numba and llvmlite only support certain python versions, and thus you may receive a `RuntimeError` while installing pynapple with an error message similar to `Cannot install on Python version 3.11.11; only versions >=3.6,<3.10 are supported`, referencing either llvmlite or numba.

There are two possible solutions:
- Install a newer version of numba at the same time as pynapple: `pip install numba>=0.60 pynapple`. See [the numba documentation](https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information) for which numba versions support which python versions.
- Clear numba from the dependency manager's cache. The exact command will depend on how you are installing pynapple:
  - If you are using uv: `uv cache clean numba` 
  - If you are using pip: `pip cache remove numba`

:::


## Getting started


Once installed, you can import pynapple with 

```python
import pynapple as nap
```

To get started with pynapple, please read the [introduction](user_guide/01_introduction_to_pynapple) that introduces the minimal concepts.

## Dependencies


### Supported python versions
  
  - Python 3.8+

### Mandatory dependencies


  -   pandas
  -   numpy
  -   scipy
  -   numba
  -   pynwb 2.0
  -   tabulate
  -   h5py
  -   rich

## Contributing

For contributing or developing with pynapple, you can install directly from the source code:

```
# clone the repository
git clone https://github.com/pynapple-org/pynapple.git
cd pynapple

# Install in editable mode with `-e` or, equivalently, `--editable`
pip install -e ".[dev]"
```

See our full contributor guide on [GitHub](https://github.com/pynapple-org/pynapple) for more details.
