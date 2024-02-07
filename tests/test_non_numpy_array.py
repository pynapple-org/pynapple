from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import numpy as np
import pytest

import pynapple as nap

from .mock import MockArray


class TestTsArray:

    @pytest.mark.parametrize(
        "time, expectation",
        [
            (jnp.array([1, 2, 3]), does_not_raise()),
            (MockArray(np.array([1, 2, 3])), does_not_raise()),
            (
                "abc",
                pytest.raises(
                    AttributeError, match="'str' object has no attribute 'astype'"
                ),
            ),
        ],
    )
    def test_ts_init(self, time, expectation):
        with expectation:
            nap.Ts(t=time)

    @pytest.mark.parametrize(
        "time, expectation",
        [
            (jnp.array([1, 2, 3]), does_not_raise()),
            (MockArray(np.array([1, 2, 3])), does_not_raise()),
        ],
    )
    def test_ts_type(self, time, expectation):
        with expectation:
            ts = nap.Ts(t=time)
            assert isinstance(ts.t, np.ndarray)

    @pytest.mark.parametrize(
        "time, expectation",
        [
            (np.array([1, 2, 3]), does_not_raise()),
            (
                jnp.array([1, 2, 3]),
                pytest.warns(UserWarning, match="Converting 't' to numpy.array"),
            ),
            (
                MockArray(np.array([1, 2, 3])),
                pytest.warns(UserWarning, match="Converting 't' to numpy.array"),
            ),
        ],
    )
    def test_ts_warn(self, time, expectation):
        with expectation:
            nap.Ts(t=time)


class TestTsdArray:

    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (jnp.array([1, 2, 3]), jnp.array([1, 2, 3]), does_not_raise()),
            (jnp.array([1, 2, 3]), MockArray(np.array([1, 2, 3])), does_not_raise()),
            (
                jnp.array([1, 2, 3]),
                "abc",
                pytest.raises(
                    AttributeError, match="'str' object has no attribute 'ndim'"
                ),
            ),
        ],
    )
    def test_tsd_init(self, time, data, expectation):
        with expectation:
            nap.Tsd(t=time, d=data)

    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([1, 2, 3]), does_not_raise()),
            (np.array([1, 2, 3]), MockArray(np.array([1, 2, 3])), does_not_raise()),
        ],
    )
    def test_tsd_type(self, time, data, expectation):
        with expectation:
            ts = nap.Tsd(t=time, d=data)
            assert isinstance(ts.d, np.ndarray)

    @pytest.mark.parametrize(
        "data, expectation",
        [
            (np.array([1, 2, 3]), does_not_raise()),
            (
                jnp.array([1, 2, 3]),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
            (
                MockArray(np.array([1, 2, 3])),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
        ],
    )
    def test_tsd_warn(self, data, expectation):
        with expectation:
            nap.Tsd(t=np.array(data), d=data)


class TestTsdFrameArray:

    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (jnp.array([1, 2, 3]), jnp.array([1, 2, 3]), does_not_raise()),
            (jnp.array([1, 2, 3]), MockArray(np.array([1, 2, 3])), does_not_raise()),
            (
                jnp.array([1, 2, 3]),
                "abc",
                pytest.raises(
                    AttributeError, match="'str' object has no attribute 'ndim'"
                ),
            ),
        ],
    )
    def test_tsdframe_init(self, time, data, expectation):
        with expectation:
            nap.TsdFrame(t=time, d=data)

    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([1, 2, 3]), does_not_raise()),
            (np.array([1, 2, 3]), MockArray(np.array([1, 2, 3])), does_not_raise()),
        ],
    )
    def test_tsdframe_type(self, time, data, expectation):
        with expectation:
            ts = nap.TsdFrame(t=time, d=data)
            assert isinstance(ts.d, np.ndarray)

    @pytest.mark.parametrize(
        "data, expectation",
        [
            (np.array([1, 2, 3]), does_not_raise()),
            (
                jnp.array([1, 2, 3]),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
            (
                MockArray(np.array([1, 2, 3])),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
        ],
    )
    def test_tsdframe_warn(self, data, expectation):
        with expectation:
            nap.TsdFrame(t=np.array(data), d=data)


class TestTsdTensorArray:

    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (jnp.array([1, 2, 3]), jnp.array([[[1]], [[2]], [[3]]]), does_not_raise()),
            (
                jnp.array([1, 2, 3]),
                MockArray(np.array([[[1]], [[2]], [[3]]])),
                does_not_raise(),
            ),
            (
                jnp.array([1, 2, 3]),
                "abc",
                pytest.raises(AssertionError, match="Data should have more than"),
            ),
        ],
    )
    def test_tsdtensor_init(self, time, data, expectation):
        with expectation:
            nap.TsdTensor(t=time, d=data)

    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([[[1]], [[2]], [[3]]]), does_not_raise()),
            (
                np.array([1, 2, 3]),
                MockArray(np.array([[[1]], [[2]], [[3]]])),
                does_not_raise(),
            ),
        ],
    )
    def test_tsdtensor_type(self, time, data, expectation):
        with expectation:
            ts = nap.TsdTensor(t=time, d=data)
            assert isinstance(ts.d, np.ndarray)

    @pytest.mark.parametrize(
        "data, expectation",
        [
            (np.array([[[1]], [[2]], [[3]]]), does_not_raise()),
            (
                jnp.array([[[1]], [[2]], [[3]]]),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
            (
                MockArray(np.array([[[1]], [[2]], [[3]]])),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
        ],
    )
    def test_tsdtensor_warn(self, data, expectation):
        with expectation:
            nap.TsdTensor(t=np.array(data), d=data)

