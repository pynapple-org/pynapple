# Installing and getting started

The best way to install pynapple is with pip within a new [conda](https://docs.conda.io/en/latest/) environment :

    
```
conda create --name pynapple pip python
conda activate pynapple
pip install pynapple
```

or directly from the source code:

```
conda create --name pynapple pip python
conda activate pynapple

# clone the repository
git clone https://github.com/pynapple-org/pynapple.git
cd pynapple

# Install in editable mode with `-e` or, equivalently, `--editable`
pip install -e .
```

## Getting started


Once installed, you can import pynapple with 

```python
import pynapple as nap
```

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

