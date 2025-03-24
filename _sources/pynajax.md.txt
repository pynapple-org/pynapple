# Accelerating pynapple with jax

## Motivation

```{eval-rst}
.. Warning::

    New in `0.6.6`
```

Multiple python packages exist for high-performance computing. Internally, pynapple makes extensive use of [numba](https://numba.pydata.org/) for accelerating some functions. Numba is a stable package that provide speed gains with minimal installation issues when running on CPUs.

Another high-performance toolbox for numerical analysis is 
[jax](https://jax.readthedocs.io/en/latest/index.html). In addition to accelerating python code on CPUs, GPUs, and TPUs, it provides a special representation of arrays using the [jax Array object](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html). Unfortunately, jax Array is incompatible with Numba. To solve this issue, we developped [pynajax](https://github.com/pynapple-org/pynajax).

Pynajax is an accelerated backend for pynapple built on top on jax. It offers a fast acceleration for some pynapple functions using CPU or GPU. Here is a minimal example on how to use pynajax:

``` bash
pip install pynajax
```



``` python
import pynapple as nap
import numpy as np

# Changed the backend from 'numba' to 'jax'
nap.nap_config.set_backend("jax") 

# This will convert the numpy array to a jax Array.
tsd = nap.Tsd(t=np.arange(100), d=np.random.randn(100)) 

# This will run on GPU or CPU depending on the jax installation
tsd.convolve(np.ones(11)) 
```

This [documentation page](https://pynapple-org.github.io/pynajax/generated/gallery/) keeps tracks of the list of pynapple functions that can be jax-accelerated as well as their performances compared to pure numba.

## Installation issues

To get the best of the pynajax backend, jax needs to use the GPU. 

While installing pynajax will install all the dependencies necessary to use jax, it does not guarantee
the use of the GPU. 

To check if jax is using the GPU, you can run the following python commands :

- no GPU found : 

	```python
	import jax
	print(jax.devices())
	[CpuDevice(id=0)]
	```

- GPU found :

	```python
	import jax
	print(jax.devices())
	[cuda(id=0)]
	```

Support for installing `JAX` for GPU users can be found in the [jax documentation](https://jax.readthedocs.io/en/latest/installation.html)


## Typical use-case


In addition to providing high performance numerical computing, jax can be used as a the backbone for a large scale machine learning model. Thus, pynajax can offer full compatibility between pynapple's time series representation and computational neuroscience models constructed using jax.

An example of a python package using both pynapple and jax is [NeMOs](https://nemos.readthedocs.io/en/stable/).