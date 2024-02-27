from contextlib import nullcontext as does_not_raise

import pytest

import pynapple as nap

import numpy as np

class MockArray:
    """
    A mock array class designed for testing purposes. It mimics the behavior of array-like objects
    by providing necessary attributes and supporting indexing and iteration, but it is not a direct
    instance of numpy.ndarray.
    """

    def __init__(self, data):
        """
        Initializes the MockArray with data.

        Parameters
        ----------
        data : Union[numpy.ndarray, List]
            A list of data elements that the MockArray will contain.
        """
        self.data = np.asarray(data)
        self.shape = self.data.shape # Simplified shape attribute
        self.dtype = 'float64'  # Simplified dtype; in real scenarios, this should be more dynamic
        self.ndim = self.data.ndim  # Simplified ndim for a 1-dimensional array

    def __getitem__(self, index):
        """
        Supports indexing into the mock array.

        Parameters
        ----------
        index : int or slice
            The index or slice of the data to access.

        Returns
        -------
        The element(s) at the specified index.
        """
        return self.data[index]

    def __iter__(self):
        """
        Supports iteration over the mock array.
        """
        return iter(self.data)

    def __len__(self):
        """
        Returns the length of the mock array.
        """
        return len(self.data)


@pytest.mark.parametrize("param, expectation",
                         [
                             (True, does_not_raise()),
                             (False, does_not_raise()),
                             (1, pytest.raises(ValueError,
                                               match="suppress_conversion_warnings must be a boolean value"))
                         ])
def test_config_setter_input_validity(param, expectation):
    """Test setting suppress_conversion_warnings with various inputs to validate type checking."""
    with expectation:
        nap.config.nap_config.suppress_conversion_warnings = param


def test_config_setter_output():
    """Test if suppress_conversion_warnings property correctly retains a True value after being set."""
    nap.config.nap_config.suppress_conversion_warnings = True
    assert nap.config.nap_config.suppress_conversion_warnings


def test_config_restore_default():
    """Test if the restore_defaults method correctly resets suppress_conversion_warnings to its default."""
    nap.config.nap_config.suppress_conversion_warnings = True
    nap.config.nap_config.restore_defaults()
    assert not nap.config.nap_config.suppress_conversion_warnings


@pytest.mark.parametrize("cls, t, d, conf, expectation",
                         [
                            (nap.Ts, [0, 1], None, True, does_not_raise()),
                            (nap.Ts, [0, 1], None, False,
                             pytest.warns(UserWarning, match=f"Converting 't' to numpy.array.")),
                            (nap.Tsd, [0, 1], [0, 1], True, does_not_raise()),
                            (nap.Tsd, [0, 1], [0, 1], False,
                             pytest.warns(UserWarning, match=f"Converting 't' to numpy.array.")),
                            (nap.TsdFrame, [0, 1], [[0], [1]], True, does_not_raise()),
                            (nap.TsdFrame, [0, 1], [[0], [1]], False,
                             pytest.warns(UserWarning, match=f"Converting 't' to numpy.array.")),
                            (nap.TsdTensor, [0, 1], [[[0]], [[1]]], True, does_not_raise()),
                            (nap.TsdTensor, [0, 1], [[[0]], [[1]]], False,
                             pytest.warns(UserWarning, match=f"Converting 't' to numpy.array.")),

                         ])
def test_config_supress_warning_t(cls, t, d, conf, expectation):
    """Test if the restore_defaults method correctly resets suppress_conversion_warnings to its default."""
    nap.config.nap_config.suppress_conversion_warnings = conf
    try:
        with expectation:
            if d is None:
                cls(t=MockArray(t))
            else:
                cls(t=MockArray(t), d=d)
    finally:
        nap.config.nap_config.restore_defaults()

@pytest.mark.parametrize("cls, t, d, conf, expectation",
                         [
                            (nap.Tsd, [0, 1], [0, 1], True, does_not_raise()),
                            (nap.Tsd, [0, 1], [0, 1], False,
                             pytest.warns(UserWarning, match=f"Converting 'd' to numpy.array.")),
                            (nap.TsdFrame, [0, 1], [[0], [1]], True, does_not_raise()),
                            (nap.TsdFrame, [0, 1], [[0], [1]], False,
                             pytest.warns(UserWarning, match=f"Converting 'd' to numpy.array.")),
                            (nap.TsdTensor, [0, 1], [[[0]], [[1]]], True, does_not_raise()),
                            (nap.TsdTensor, [0, 1], [[[0]], [[1]]], False,
                             pytest.warns(UserWarning, match=f"Converting 'd' to numpy.array.")),

                         ])
def test_config_supress_warning_d(cls, t, d, conf, expectation):
    """Test if the restore_defaults method correctly resets suppress_conversion_warnings to its default."""
    nap.config.nap_config.suppress_conversion_warnings = conf
    try:
        with expectation:
            cls(t=t, d=MockArray(d))
    finally:
        nap.config.nap_config.restore_defaults()


def test_get_time_index_precision():
    assert nap.config.nap_config.time_index_precision == 9