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
