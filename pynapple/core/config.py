"""This module controls the pynapple configurations.

## Backend configuration

By default, pynapple core functions are compiled with [Numba](https://numba.pydata.org/).
It is possible to change the backend to [Jax](https://jax.readthedocs.io/en/latest/index.html)
through the [pynajax package](https://github.com/pynapple-org/pynajax).

While numba core functions runs on CPU, the `jax` backend allows pynapple to use GPU accelerated core functions.
For some core functions, the `jax` backend offers speed gains (provided that Jax runs on the GPU).

See the example below to update the backend. Don't forget to install [pynajax](https://github.com/pynapple-org/pynajax).


import pynapple as nap
import numpy as np
nap.nap_config.set_backend("jax") # Default option is 'numba'.

You can view the current backend with

>>> print(nap.nap_config.backend)
'jax'

## Warnings configuration

pynapple gives warnings that can be helpful to debug. For example when passing time indexes that are not sorted:


>>> import pynapple as nap
>>> t = [0, 2, 1]
>>> nap.Ts(t)
UserWarning: timestamps are not sorted
  warn("timestamps are not sorted", UserWarning)
Time (s)
0.0
1.0
2.0
shape: 3

pynapple's warnings can be suppressed :

>>> nap.nap_config.suppress_time_index_sorting_warnings = True
>>> nap.Ts(t=t)
Time (s)
0.0
1.0
2.0
shape: 3
"""

import importlib.util
import warnings


class PynappleConfig:
    """
    A class to hold configuration settings for pynapple.

    This class includes all configuration settings that control the behavior of
    pynapple. It offers a structured way to access and modify settings.

    Attributes
    ----------
    backend : str
        Current pynapple backend. Options are ('numba' [default], 'jax')
    suppress_conversion_warnings : boolean
        Determines whether to suppress warnings when automatically converting non-NumPy
        array-like objects to NumPy arrays. This is useful for users who frequently work with array-like objects from other
        libraries (e.g., JAX, TensorFlow) and prefer not to receive warnings for automatic
        conversions. Defaults to False, which means warnings will be shown.
    suppress_time_index_sorting_warnings : boolean
        Control the warning raised when passing a non-sorted array for time index.
        It can be useful to catch data where timestamps are not properly sorted before using pynapple.
    time_index_precision : int
        Number of decimal places to round time index. Pynapple's precision is set by default to 9.
    """

    def __init__(self):
        self.suppress_conversion_warnings = False
        self.suppress_time_index_sorting_warnings = False
        self.backend = "numba"

    @property
    def backend(self):
        """
        Pynapple backend. Can be "jax" or "numpy".
        """
        return self._backend

    @backend.setter
    def backend(self, backend):
        self.set_backend(backend)

    def set_backend(self, backend):
        assert backend in ["numba", "jax"], "Options for backend are 'jax' or 'numba'"

        # Try to import pynajax
        if backend == "jax":
            spec = importlib.util.find_spec("pynajax")
            if spec is None:
                warnings.warn(
                    "Package pynajax is not found. Falling back to numba backend. To use the jax backend for pynapple, please install pynajax",
                    stacklevel=2,
                )
                self._backend = "numba"
            else:
                self._backend = "jax"
        else:
            self._backend = "numba"

    @property
    def time_index_precision(self):
        """Precision for the time index

        Returns
        -------
        Int
            Parameter for the `numpy.around` function when rounding time index
        """
        return 9

    @property
    def suppress_conversion_warnings(self):
        """
        Gets or sets the suppression state for conversion warnings. When set to True,
        warnings for automatic conversions of non-NumPy array-like objects or pynapple objects to NumPy arrays
        are suppressed. Ensures that only boolean values are assigned.
        """
        return self._suppress_conversion_warnings

    @suppress_conversion_warnings.setter
    def suppress_conversion_warnings(self, value):
        if not isinstance(value, bool):
            raise ValueError("suppress_conversion_warnings must be a boolean value.")
        self._suppress_conversion_warnings = value

    @property
    def suppress_time_index_sorting_warnings(self):
        """
        Gets or sets the suppression state for sorting time index. When set to True,
        warnings for sorting are suppressed. Ensures that only boolean values are assigned.
        """
        return self._suppress_time_index_sorting_warnings

    @suppress_time_index_sorting_warnings.setter
    def suppress_time_index_sorting_warnings(self, value):
        if not isinstance(value, bool):
            raise ValueError(
                "suppress_time_index_sorting_warnings must be a boolean value."
            )
        self._suppress_time_index_sorting_warnings = value

    def restore_defaults(self):
        """
        Set all configuration settings to their default values.

        This method can be used to easily set/reset the configuration state of pynapple
        to its initial, default configuration.
        """
        self.suppress_conversion_warnings = False
        self.suppress_time_index_sorting_warnings = False


# Initialize a config instance
nap_config = PynappleConfig()
