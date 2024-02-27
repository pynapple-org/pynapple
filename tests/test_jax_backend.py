import pynapple as nap
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from time import time

t = np.arange(1000)
d = jnp.asarray(np.random.randn(1000))

nap.nap_config.set_backend("jax")

tsd = nap.Tsd(t=t, d=d)#, time_support = nap.IntervalSet(start=0, end=5))

t1 = time()
tsd2 = tsd.convolve(np.ones(10))
print(time() - t1)

nap.nap_config.set_backend("numba")

tsd = nap.Tsd(t=t, d=tsd.values)
t2 = time()
tsd3 = tsd.convolve(np.ones(10))
print(time() - t2)







# # from numba import jit



# def decoratorcasting(func):
	
# 	def wrapper(array):		
# 		return func(np.asarray(array))

# 	return wrapper

# @decoratorcasting
# @jit(nopython=True)
# def test_numba(array):
# 	return np.sum(array)