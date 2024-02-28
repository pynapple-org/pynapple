import pynapple as nap
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from time import time
from pynajax.jax_core import _convolve_vec, _convolve_mat

t = np.arange(100000)
d = jnp.asarray(np.random.randn(100000, 10))

if d.ndim >2:
    NAPTYPE = nap.TsdTensor

else:
    NAPTYPE = nap.TsdFrame

CONV_JAX = _convolve_vec

###
# SET BACKEND JAX
###
nap.nap_config.set_backend("jax")

ep = nap.IntervalSet(start=np.arange(0, 1000, 100), end = np.arange(0, 1000, 100)+50)

tsd_jax = NAPTYPE(t=t, d=d, time_support=ep)

kernel = np.ones(10)

tsd2 = tsd_jax.convolve(kernel)
t1 = time()
tsd2 = tsd_jax.convolve(kernel)
print("pynajax convolve multi-epoch", time() - t1)



CONV_JAX(tsd_jax.values, jnp.asarray(kernel))
t4 = time()
out = _convolve_vec(tsd_jax.values, jnp.asarray(kernel))
print("convolve-vec", time() - t4)

###
# SET BACKEND NUMBA
###
nap.nap_config.set_backend("numba")

tsd_numpy_one_ep = NAPTYPE(t=t, d=np.asarray(d))#, time_support = ep)
tsd3 = tsd_numpy_one_ep.convolve(kernel)
t2 = time()
tsd3 = tsd_numpy_one_ep.convolve(kernel)
print("numba multi-epoch", time() - t2)
