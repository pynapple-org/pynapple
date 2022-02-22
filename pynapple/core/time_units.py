import pandas as pd
import numpy as np
from warnings import warn
from pandas.core.internals import SingleBlockManager, BlockManager


class Range:
    """
    A class defining a range to restrict analyses.

    This is used as a context manager, taking a :func:`~neuroseries.interval_set.IntervalSet` as an input.
    After that, all neuroseries objects will have a property r set, that will be restricted, for example:

    .. code:: python

        with nts.Range(range_interval):
            np.testing.assert_array_almost_equal_nulp(self.tsd.r.times(), tsd_r.times())
    """
    interval = None
    cached_objects = []

    def __init__(self, a, b=None, time_units=None):
        """
        Creates a Range object
        Args:
            a: an :ref:`IntervalSet` defining the range, or the lower bound of the range
            b : if defined (defaults to :ref:`None`), contains the upper bound of the range, as a number of a Series
            (or other object with an Index)
            time_units (str): a string defining the :ref:`TimeUnits` used to define the bounds.
        """
        if b:
            start = TimeUnits.format_timestamps(np.array((a,), dtype=np.int64).ravel(), time_units)
            end = TimeUnits.format_timestamps(np.array((b,), dtype=np.int64).ravel(), time_units)
            from neuroseries.interval_set import IntervalSet
            Range.interval = IntervalSet(start, end)
        else:
            Range.interval = a

    def __enter__(self):
        return Range.interval

    def __exit__(self, exc_type, exc_val, exc_tb):
        Range.interval = None
        for i in Range.cached_objects:
            i.invalidate_restrict_cache()
        self.cached_objects = []


class TimeUnits:
    """
    This class deals with conversion between different time units for all neuroseries objects.
    It also provides a context manager that tweaks the default time units to the supported units:
    - 'us': microseconds
    - 'ms': milliseconds
    - 's': seconds  (overall default)

    The context manager is called as follows

    .. code:: python

        with nts.TimeUnits('ms'):
            t = self.tsd.times()

    """
    default_time_units = 's'

    def __init__(self, units):
        TimeUnits.default_time_units = units

    def __enter__(self):
        return self.default_time_units

    def __exit__(self, exc_type, exc_val, exc_tb):
        TimeUnits.default_time_units = 's'

    @staticmethod
    def format_timestamps(t, units=None, give_warning=True, sortt = True):
        """
        Converts numerical types to the type :func:`numpy.float64` that is used for the time index in neuroseries.

        Args:
            t: a vector (or scalar) of times
            units: the units in which times are given
            give_warning: if True, it will warn when the timestamps are not sored

        Returns:
            ts: times in standard neuroseries format
        """

        import numbers

        if not units:
            units = TimeUnits.default_time_units

        if isinstance(t, BlockManager):
            t = pd.DataFrame(t, copy=True)

        if isinstance(t, (pd.Series, pd.DataFrame)):
            t = t.index.values.astype(np.float64)

        # if isinstance(t, np.floating):
        #     t = t.round(6)

        if isinstance(t, (pd.Series, pd.DataFrame)):
            t = t.index

        if isinstance(t, numbers.Number):
            t = np.array((t,))

        t = t.astype(np.float64)

        if units == 's':
            # t *= 1000000
            t = np.around(t, 6)
            # pass
        elif units == 'ms':
            # t *= 1000
            t = np.around(t/1.0e3, 6)
        elif units == 'us':
            # pass
            t = np.around(t/1.0e6, 6)
        else:
            raise ValueError('unrecognized time units type')

        # noinspection PyUnresolvedReferences,PyTypeChecker
        ts = t.astype(np.float64).reshape((len(t),))

        if not (np.diff(ts) >= 0).all():
            if give_warning:
                warn('timestamps are not sorted', UserWarning)
            if sortt:
                ts.sort()
        return ts

    @staticmethod
    def return_timestamps(t, units=None):
        """
        package the times in the desired units
        Args:
            t: standard neuroseries times
            units: the desired units for the output

        Returns:
            ts: times in the desired format
        """
        if units is None:
            units = TimeUnits.default_time_units
        if units == 'us':
            # return t
            return t * 1.0e6
        elif units == 'ms':
            # return t / 1000.
            return t * 1.0e3
        elif units == 's':
            # return t / 1.0e6
            return t
        else:
            raise ValueError('Unrecognized units')

