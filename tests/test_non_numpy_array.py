from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

import pynapple as nap

from .helper_tests import MockArray


class TestTsArray:

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, expectation",
        [
            (MockArray(np.array([1, 2, 3])), does_not_raise()),
            (
                "abc",
                pytest.raises(
                    RuntimeError,
                    match="Unknown format for t. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.",
                ),
            ),
        ],
    )
    def test_ts_init(self, time, expectation):
        """Verify the expected behavior of the initialization for Ts objects."""
        with expectation:
            nap.Ts(t=time)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, expectation",
        [(MockArray(np.array([1, 2, 3])), does_not_raise())],
    )
    def test_ts_type(self, time, expectation):
        """Verify that the time attribute 't' of a Ts object is stored as a numpy.ndarray."""
        with expectation:
            ts = nap.Ts(t=time)
            assert isinstance(ts.t, np.ndarray)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, expectation",
        [
            (np.array([1, 2, 3]), does_not_raise()),
            (
                MockArray(np.array([1, 2, 3])),
                pytest.warns(UserWarning, match="Converting 't' to numpy.array"),
            ),
        ],
    )
    def test_ts_warn(self, time, expectation):
        """Check for warnings when the time attribute 't' is automatically converted to numpy.ndarray."""
        with expectation:
            nap.Ts(t=time)


class TestTsdArray:

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (MockArray([1, 2, 3]), MockArray([1, 2, 3]), does_not_raise()),
            (MockArray([1, 2, 3]), MockArray(np.array([1, 2, 3])), does_not_raise()),
            (
                MockArray([1, 2, 3]),
                "abc",
                pytest.raises(
                    RuntimeError,
                    match="Unknown format for d. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.",
                ),
            ),
            (
                "abc",
                MockArray([1, 2, 3]),
                pytest.raises(
                    RuntimeError,
                    match="Unknown format for t. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.",
                ),
            ),
        ],
    )
    def test_tsd_init(self, time, data, expectation):
        """Verify the expected behavior of the initialization for Tsd objects."""
        with expectation:
            nap.Tsd(t=time, d=data)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([1, 2, 3]), does_not_raise()),
            (np.array([1, 2, 3]), MockArray(np.array([1, 2, 3])), does_not_raise()),
        ],
    )
    def test_tsd_type_d(self, time, data, expectation):
        """Verify that the data attribute 'd' of a Tsd object is stored as a numpy.ndarray."""
        with expectation:
            ts = nap.Tsd(t=time, d=data)
            if nap.nap_config.backend == "numba":
                assert isinstance(ts.d, np.ndarray)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([1, 2, 3]), does_not_raise()),
            (
                MockArray(np.array([1, 2, 3])),
                np.array([1, 2, 3]),
                does_not_raise(),
            ),
        ],
    )
    def test_tsd_type_t(self, time, data, expectation):
        """Verify that the time attribute 't' of a TsdFrame object is stored as a numpy.ndarray."""
        with expectation:
            ts = nap.Tsd(t=time, d=data)
            assert isinstance(ts.t, np.ndarray)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "data, expectation",
        [
            (np.array([1, 2, 3]), does_not_raise()),
            (
                MockArray(np.array([1, 2, 3])),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
        ],
    )
    def test_tsd_warn(self, data, expectation):
        """Check for warnings when the data attribute 'd' is automatically converted to numpy.ndarray."""
        if nap.nap_config.backend == "numba":
            with expectation:
                nap.Tsd(t=np.array(data), d=data)


class TestTsdFrameArray:

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (MockArray([1, 2, 3]), MockArray([1, 2, 3]), does_not_raise()),
            (
                MockArray([1, 2, 3]),
                "abc",
                pytest.raises(
                    RuntimeError,
                    match="Unknown format for d. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.",
                ),
            ),
        ],
    )
    def test_tsdframe_init(self, time, data, expectation):
        """Verify the expected behavior of the initialization for TsdFrame objects."""
        with expectation:
            nap.TsdFrame(t=time, d=data)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([1, 2, 3]), does_not_raise()),
            (np.array([1, 2, 3]), MockArray(np.array([1, 2, 3])), does_not_raise()),
        ],
    )
    def test_tsdframe_type(self, time, data, expectation):
        """Verify that the data attribute 'd' of a TsdFrame object is stored as a numpy.ndarray."""
        with expectation:
            ts = nap.TsdFrame(t=time, d=data)
            if nap.nap_config.backend == "numba":
                assert isinstance(ts.d, np.ndarray)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([[1], [2], [3]]), does_not_raise()),
            (
                MockArray(np.array([1, 2, 3])),
                np.array([[1], [2], [3]]),
                does_not_raise(),
            ),
        ],
    )
    def test_tsdframe_type_t(self, time, data, expectation):
        """Verify that the time attribute 't' of a TsdFrame object is stored as a numpy.ndarray."""
        with expectation:
            ts = nap.TsdFrame(t=time, d=data)
            assert isinstance(ts.t, np.ndarray)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "data, expectation",
        [
            (np.array([1, 2, 3]), does_not_raise()),
            (
                MockArray(np.array([1, 2, 3])),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
        ],
    )
    def test_tsdframe_warn(self, data, expectation):
        """Check for warnings when the data attribute 'd' is automatically converted to numpy.ndarray."""
        if nap.nap_config.backend == "numba":
            with expectation:
                nap.TsdFrame(t=np.array(data), d=data)


class TestTsdTensorArray:

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (
                MockArray([1, 2, 3]),
                MockArray(np.array([[[1]], [[2]], [[3]]])),
                does_not_raise(),
            ),
            (
                MockArray([1, 2, 3]),
                "abc",
                pytest.raises(
                    RuntimeError,
                    match="Unknown format for d. Accepted formats are numpy.ndarray, list, tuple or any array-like objects.",
                ),
            ),
        ],
    )
    def test_tsdtensor_init(self, time, data, expectation):
        """Verify the expected behavior of the initialization for TsdTensor objects."""
        with expectation:
            nap.TsdTensor(t=time, d=data)

    @pytest.mark.filterwarnings("ignore")
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
    def test_tsdtensor_type_d(self, time, data, expectation):
        """Verify that the data attribute 'd' of a TsdTensor object is stored as a numpy.ndarray."""
        with expectation:
            ts = nap.TsdTensor(t=time, d=data)
            if nap.nap_config.backend == "numba":
                assert isinstance(ts.d, np.ndarray)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "time, data, expectation",
        [
            (np.array([1, 2, 3]), np.array([[[1]], [[2]], [[3]]]), does_not_raise()),
            (
                MockArray(np.array([1, 2, 3])),
                np.array([[[1]], [[2]], [[3]]]),
                does_not_raise(),
            ),
        ],
    )
    def test_tsdtensor_type_t(self, time, data, expectation):
        """Verify that the time attribute 't' of a TsdTensor object is stored as a numpy.ndarray."""
        with expectation:
            ts = nap.TsdTensor(t=time, d=data)
            assert isinstance(ts.t, np.ndarray)

    @pytest.mark.filterwarnings("ignore")
    @pytest.mark.parametrize(
        "data, expectation",
        [
            (np.array([[[1]], [[2]], [[3]]]), does_not_raise()),
            (
                MockArray(np.array([[[1]], [[2]], [[3]]])),
                pytest.warns(UserWarning, match="Converting 'd' to numpy.array"),
            ),
        ],
    )
    def test_tsdtensor_warn(self, data, expectation):
        """Check for warnings when the data attribute 'd' is automatically converted to numpy.ndarray."""
        if nap.nap_config.backend == "numba":
            with expectation:
                nap.TsdTensor(t=np.ravel(np.array(data)), d=data)
