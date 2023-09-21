# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-09-21 13:32:03
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-21 15:46:14

"""
    This class deals with conversion between different time units for all pynapple objects.
    It also provides a context manager that tweaks the default time units to the supported units:
    - 'us': microseconds
    - 'ms': milliseconds
    - 's': seconds  (overall default)
"""

from warnings import warn

import numpy as np

# from .time_units import format_timestamps, return_timestamps, sort_timestamps


class TsIndex(np.ndarray):
    """
    Holder for timestamps. Mimics pandas.Index. Subclass numpy.ndarray
    """

    @staticmethod
    def format_timestamps(t, units="s"):
        """
        Converts time index in pynapple in a default format

        Args:
            t: a vector (or scalar) of times
            units: the units in which times are given

        Returns:
            t: times in standard pynapple format
        """
        if units == "s":
            t = np.around(t, 9)
        elif units == "ms":
            t = np.around(t / 1.0e3, 9)
        elif units == "us":
            t = np.around(t / 1.0e6, 9)
        else:
            raise ValueError("unrecognized time units type")

        return t

    @staticmethod
    def return_timestamps(t, units="s"):
        """
        Converts time index in pynapple in a particular format

        Args:
            t: a vector (or scalar) of times
            units: the units in which times are given

        Returns:
            t: times in standard pynapple format
        """
        if units == "s":
            t = np.around(t, 9)
        elif units == "ms":
            t = np.around(t * 1.0e3, 9)
        elif units == "us":
            t = np.around(t * 1.0e6, 9)
        else:
            raise ValueError("unrecognized time units type")

        return t

    @staticmethod
    def sort_timestamps(t, give_warning=True):
        """
        Raise warning if timestamps are not sorted
        """
        if not (np.diff(t) >= 0).all():
            if give_warning:
                warn("timestamps are not sorted", UserWarning)
            t = np.sort(t)
        return t

    def __new__(cls, t, time_units="s"):
        t = t.astype(np.float64).flatten()
        t = TsIndex.format_timestamps(t, time_units)
        t = TsIndex.sort_timestamps(t)
        obj = np.asarray(t).view(cls)
        return obj

    @property
    def values(self):
        return np.asarray(self)

    def __setitem__(self):
        raise RuntimeError("TsIndex object is immutable.")

    def to_numpy(self):
        return self.values

    def in_units(self, time_units="s"):
        return TsIndex.return_timestamps(self.values, time_units)
