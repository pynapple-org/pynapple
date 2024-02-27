import pynapple as nap
import numpy as np
import pytest

import jax
import jax.numpy as jnp

t = np.arange(10)
d = jnp.arange(10)

nap.nap_config.set_backend("jax")

tsd = nap.Tsd(t=t, d=d)

