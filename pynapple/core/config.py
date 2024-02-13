"""This module deals with package configurations. For now it includes only warning configurations.
"""


class PynappleConfig:
    """
    A class to hold configuration settings for pynapple.

    This class includes all configuration settings that control the behavior of
    pynapple. It offers a structured way to access and modify settings.

    Examples
    --------
    >>> import pynapple as nap
    >>> import jax.numpy as jnp
    >>> t = jnp.arange(3)
    >>> print(t)
    Array([0, 1, 2], dtype=int32)

    >>> # Suppress warnings when converting a non-numpy array to numpy array
    >>> nap.config.nap_config.suppress_conversion_warnings = True
    >>> nap.Ts(t=t)
    Time (s)
    0.0
    1.0
    2.0
    shape: 3

    >>> # Restore to defaults
    >>> nap.config.nap_config.restore_defaults()
    >>> nap.Ts(t=t)
    /mnt/home/gviejo/pynapple/pynapple/core/time_series.py:151: UserWarning: Converting 't' to n
    umpy.array. The provided array was of type 'ArrayImpl'.
      warnings.warn(
    Time (s)
    0.0
    1.0
    2.0
    shape: 3

    Attributes
    ----------
    suppress_conversion_warnings : boolean
        Determines whether to suppress warnings when automatically converting non-NumPy
        array-like objects to NumPy arrays. This is useful for users who frequently work with array-like objects from other
        libraries (e.g., JAX, TensorFlow) and prefer not to receive warnings for automatic
        conversions. Defaults to False, which means warnings will be shown.
    """

    def __init__(self):
        self.suppress_conversion_warnings = False

    @property
    def suppress_conversion_warnings(self):
        """
        Gets or sets the suppression state for conversion warnings. When set to True,
        warnings for automatic conversions of non-NumPy array-like objects to NumPy arrays
        are suppressed. Ensures that only boolean values are assigned.
        """
        return self._suppress_conversion_warnings

    @suppress_conversion_warnings.setter
    def suppress_conversion_warnings(self, value):
        if not isinstance(value, bool):
            raise ValueError("suppress_conversion_warnings must be a boolean value.")
        self._suppress_conversion_warnings = value

    def restore_defaults(self):
        """
        Set all configuration settings to their default values.

        This method can be used to easily set/reset the configuration state of pynapple
        to its initial, default configuration.
        """
        self.suppress_conversion_warnings = False


# Initialize a config instance
nap_config = PynappleConfig()
