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

ep = nap.IntervalSet(start=np.arange(0, 1000, 100),end = np.arange(0, 1000, 100)+50)

tsd_jax = NAPTYPE(t=t, d=d, time_support = ep)

kernel = np.ones(10)

tsd2 = tsd_jax.convolve(kernel)
t1 = time()
tsd2 = tsd_jax.convolve(kernel)
print("jax", time() - t1)



CONV_JAX(tsd_jax.values, jnp.asarray(kernel))
t4 = time()
out = CONV_JAX(tsd_jax.values, jnp.asarray(kernel))
print("convolvevec", time() - t4)

###
# SET BACKEND NUMBA
###
nap.nap_config.set_backend("numba")

tsd_numpy_one_ep = NAPTYPE(t=t, d=np.asarray(d))#, time_support = ep)
tsd3 = tsd_numpy_one_ep.convolve(kernel)
t2 = time()
tsd3 = tsd_numpy_one_ep.convolve(kernel)
print("numba", time() - t2)

#
# print("... testing multi epoch")
# tsd_numpy_multi_ep = NAPTYPE(t=t, d=np.asarray(d), time_support = ep)
# tsd3 = tsd_numpy_multi_ep.convolve(kernel)
# t2 = time()
# tsd3 = tsd_numpy_multi_ep.convolve(kernel)
# print("numba mutlti-ep", time() - t2)
#
# ###
# # SET BACKEND JAX
# ###
# nap.nap_config.set_backend("jax")
#
# tsd_jax = NAPTYPE(t=t, d=d, time_support = ep)
#
# tree_of_jax = [jnp.asarray(tsd_jax.get(s, e).d) for s, e in tsd_jax.time_support.values]
#
#
# @jax.jit
# def map_par(tree, jax_kernel):
#     return jax.tree_map(lambda x: CONV_JAX(x, jax_kernel), tree)
#
# jkernel = jnp.asarray(kernel)
# tree_of_jax = [jnp.asarray(tsd_jax.get(s, e).d) for s, e in tsd_jax.time_support.values]
# map_par(tree_of_jax, jkernel)
#
# t2 = time()
# tsd6 = map_par(tree_of_jax, kernel)
# print("jax jit pytree", time() - t2)
#
# # # from numba import jit
#
#
#
# # def decoratorcasting(func):
#
# # 	def wrapper(array):
# # 		return func(np.asarray(array))
#
# # 	return wrapper
#
# # @decoratorcasting
# # @jit(nopython=True)
# # def test_numba(array):
# # 	return np.sum(array)