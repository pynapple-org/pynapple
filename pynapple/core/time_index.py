# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-09-21 13:32:03
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-25 11:28:11

"""

    Similar to pd.Index, TsIndex holds the timestamps associated with the data of a time series.
    This class deals with conversion between different time units for all pynapple objects as well
    as making sure that timestamps are property sorted before initializing any objects.    
        - `us`: microseconds
        - `ms`: milliseconds
        - `s`: seconds  (overall default)
"""

from warnings import warn

import numpy as np


class TsIndex(np.ndarray):
    """
    Holder for timestamps. Similar to pandas.Index. Subclass numpy.ndarray
    """

    @staticmethod
    def format_timestamps(t, units="s"):
        """
        Converts time index in pynapple in a default format

        Parameters
        ----------
        t : numpy.ndarray
            a vector of times
        units
            the units in which times are given

        Returns
        -------
        t : np.ndarray
            times in standard pynapple format

        Raises
        ------
        ValueError
            Description
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

        Parameters
        ----------
        t : numpy.ndarray
            a vector (or scalar) of times
        units
            the units in which times are given

        Returns
        -------
        t : numpy.ndarray
            times in standard pynapple format

        Raises
        ------
        ValueError
            IF units is not in ['s', 'ms', 'us']
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

        Parameters
        ----------
        t : numpy.ndarray
            a vector of times
        give_warning : bool, optional
            If timestamps are not sorted

        Returns
        -------
        numpy.ndarray
            Description
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
        """Returns the index as a ndarray

        Returns
        -------
        numpy.ndarray
            The timestamps in seconds
        """
        return np.asarray(self)

    def __setitem__(self, *args, **kwargs):
        raise RuntimeError("TsIndex object is not mutable.")

    def to_numpy(self):
        """Return the index as a ndarray. Useful for matplotlib.

        Returns
        -------
        numpy.ndarray
            The timestamps in seconds
        """
        return self.values

    def in_units(self, time_units="s"):
        """Return the index as a ndarray in the desired units

        Returns
        -------
        numpy.ndarray
            The timestamps in seconds
        """
        return TsIndex.return_timestamps(self.values, time_units)
