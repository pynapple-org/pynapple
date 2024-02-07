"""Package configurations.
"""


class PynappleConfig:
    """
    A class to hold configuration settings for pynapple.

    This class includes all configuration settings that control the behavior of
    pynapple. It offers a structured way to access and modify settings.

    Attributes:
    -----------
    suppress_conversion_warnings (bool):
        Determines whether to suppress warnings when automatically converting non-NumPy
        array-like objects to NumPy arrays.
        This is useful for users who frequently work with array-like objects from other
        libraries (e.g., JAX, TensorFlow) and prefer not to receive warnings for automatic
        conversions. Defaults to False, which means warnings will be shown.
    """

    def __init__(self):
        self.suppress_conversion_warnings = False

    @property
    def suppress_conversion_warnings(self):
        """
        bool: Gets or sets the suppression state for conversion warnings. When set to True,
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
