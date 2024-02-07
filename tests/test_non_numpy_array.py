from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import numpy as np
import pytest

import pynapple as nap

from .mock import MockArray


class TestTsArray:

    @pytest.mark.parametrize("time, expectation",
                             [
                                 (jnp.array([1, 2, 3]), does_not_raise()),
                                 (MockArray(np.array([1, 2, 3])), does_not_raise()),
                                 ("abc", pytest.raises(AttributeError,
                                                       match="'str' object has no attribute 'astype'"))
                             ])
    def test_ts_init(self, time, expectation):
        with expectation:
            nap.Ts(t=time)

    @pytest.mark.parametrize("time, expectation",
                             [
                                 (jnp.array([1, 2, 3]), does_not_raise()),
                                 (MockArray(np.array([1, 2, 3])), does_not_raise())
                             ])
    def test_ts_type(self, time, expectation):
        with expectation:
            ts = nap.Ts(t=time)
            assert isinstance(ts.t, np.ndarray)

    @pytest.mark.parametrize("time, expectation",
                             [
                                 (np.array([1, 2, 3]), does_not_raise()),
                                 (jnp.array([1, 2, 3]),
                                  pytest.warns(UserWarning, match="Converting 't' to numpy.array")),
                                 (MockArray(np.array([1, 2, 3])),
                                  pytest.warns(UserWarning, match="Converting 't' to numpy.array"))
                             ])
    def test_ts_type(self, time, expectation):
        with expectation:
            nap.Ts(t=time)

    @pytest.mark.parametrize("time, expectation",
                             [
                                 (np.array([1, 2, 3]), does_not_raise()),
                                 (jnp.array([1, 2, 3]),
                                  pytest.warns(UserWarning, match="Converting 't' to numpy.array")),
                                 (MockArray(np.array([1, 2, 3])),
                                  pytest.warns(UserWarning, match="Converting 't' to numpy.array"))
                             ])
    def test_ts_type(self, time, expectation):
        with expectation:
            nap.Ts(t=time)

