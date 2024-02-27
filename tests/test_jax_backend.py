import pynapple as nap
import numpy as np
import pytest

import jax
import jax.numpy as jnp

t = np.arange(10)
d = jnp.arange(10)

nap.nap_config.set_backend("jax")

tsd = nap.Tsd(t=t, d=d, time_support = nap.IntervalSet(start=0, end=5))



# # from numba import jit



# def decoratorcasting(func):
	
# 	def wrapper(array):		
# 		return func(np.asarray(array))

# 	return wrapper

# @decoratorcasting
# @jit(nopython=True)
# def test_numba(array):
# 	return np.sum(array)