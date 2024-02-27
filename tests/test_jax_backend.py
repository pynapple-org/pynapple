import pynapple as nap
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from time import time

t = np.arange(10000)
d = jnp.asarray(np.random.randn(10000, 10))

nap.nap_config.set_backend("jax")

# ep = nap.IntervalSet(start=np.arange(0, 1000, 100),end = np.arange(0, 1000, 100)+50)

tsd = nap.TsdFrame(t=t, d=d)#, time_support = ep)

kernel = np.ones(10)

tsd2 = tsd.convolve(kernel)
t1 = time()
tsd2 = tsd.convolve(kernel)
print("jax", time() - t1)

from pynajax.jax_core import _convolve_vec

_convolve_vec(tsd.values, jnp.asarray(kernel))
t4 = time()
out = _convolve_vec(tsd.values, jnp.asarray(kernel))
print("convolvevec", time() - t4)

nap.nap_config.set_backend("numba")

tsd = nap.TsdFrame(t=t, d=np.asarray(d))#, time_support = ep)
tsd3 = tsd.convolve(kernel)
t2 = time()
tsd3 = tsd.convolve(kernel)
print("numba", time() - t2)








# # from numba import jit



# def decoratorcasting(func):
	
# 	def wrapper(array):		
# 		return func(np.asarray(array))

# 	return wrapper

# @decoratorcasting
# @jit(nopython=True)
# def test_numba(array):
# 	return np.sum(array)