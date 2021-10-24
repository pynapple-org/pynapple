import pandas as pd
import numpy as np
from warnings import warn
from pandas.core.internals import SingleBlockManager, BlockManager

use_pandas_metadata = True


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
    - 'us': microseconds (overall default)
    - 'ms': milliseconds
    - 's': seconds

    The context manager is called as follows

    .. code:: python

        with nts.TimeUnits('ms'):
            t = self.tsd.times()

    """
    default_time_units = 'us'

    def __init__(self, units):
        TimeUnits.default_time_units = units

    def __enter__(self):
        return self.default_time_units

    def __exit__(self, exc_type, exc_val, exc_tb):
        TimeUnits.default_time_units = 'us'

    @staticmethod
    def format_timestamps(t, units=None, give_warning=True):
        """
        Converts numerical types to the type :func:`numpy.int64` that is used for the time index in neuroseries.

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

        # get the times (or the index if pandas object, in int64 vector format

        if isinstance(t, BlockManager):
            t = pd.DataFrame(t, copy=True)

        if isinstance(t, (pd.Series, pd.DataFrame)):
            t = t.index.values.astype(np.int64)

        if isinstance(t, np.floating):
            t = t.round()

        if isinstance(t, (pd.Series, pd.DataFrame)):
            t = t.index

        if isinstance(t, numbers.Number):
            t = np.array((t,))

        t = t.astype(np.float64)
        if units == 'us':
            pass
        elif units == 'ms':
            t *= 1000
        elif units == 's':
            t *= 1000000
        else:
            raise ValueError('unrecognized time units type')

        # noinspection PyUnresolvedReferences,PyTypeChecker
        ts = t.astype(np.int64).reshape((len(t),))

        if not (np.diff(ts) >= 0).all():
            if give_warning:
                warn('timestamps are not sorted', UserWarning)
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
            return t
        elif units == 'ms':
            return t / 1000.
        elif units == 's':
            return t / 1.0e6
        else:
            raise ValueError('Unrecognized units')


def _get_restrict_method(align):
    if align in ('closest', 'nearest'):
        method = 'nearest'
    elif align in ('next', 'bfill', 'backfill'):
        method = 'bfill'
    elif align in ('prev', 'ffill', 'pad'):
        method = 'pad'
    else:
        raise ValueError('Unrecognized restrict align method')
    return method


class Tsd(pd.Series):
    """
    A subclass of :func:`pandas.Series` specialized for neurophysiology time series.

    Tsd provides standardized time representation, plus functions for restricting and realigning time series
    """

    def __init__(self, t, d=None, time_units=None, **kwargs):
        """
        Tsd Initializer.

        Args:
            t: an object transformable in a time series, or a :func:`~pandas.Series` equivalent (if d is None)
            d: the data in the time series
            time_units: the time units in which times are specified (has no effect if a Pandas object
            is provided as the first argument
            **kwargs: arguments that will be passed to the :func:`~pandas.Series` initializer.
        """
        if isinstance(t, (pd.Series, SingleBlockManager)):
            super().__init__(t, **kwargs)
        else:
            t = TimeUnits.format_timestamps(t, time_units)
            super().__init__(index=t, data=d, **kwargs)
        self.index.name = "Time (us)"
        self._metadata.append("nts_class")
        self.nts_class = self.__class__.__name__
        self.r_cache = None

    def times(self, units=None):
        """
        The times of the Tsd, returned as np.double in the desired time units

        Args:
            units: the desired time units

        Returns:
            ts: the times vector

        """
        return TimeUnits.return_timestamps(self.index.values.astype(np.float64), units)

    def as_series(self):
        """
        The Tsd as a :func:`pandas:pandas.Series` object

        Returns:
            ss: the series object

        """

        return pd.Series(self, copy=True)

    def as_units(self, units=None):
        """
        Returns a Series with time expressed in the desired unit.

        :param units: us, ms, or s
        :return: Series with adjusted times
        """
        ss = self.as_series()
        t = self.index.values
        t = TimeUnits.return_timestamps(t, units)
        ss.index = t
        units_str = units
        if not units_str:
            units_str = 'us'
        ss.index.name = "Time (" + units_str + ")"
        return ss

    def data(self):
        """
        The data in the Series object

        Returns: the data

        """
        return self.values

    def realign(self, t, align='closest'):
        """
        Provides a new Series only including the data points that are close to one time point in the t argument.

        Args:
            t: the aligning series, in numpy or pandas format
            align: the values accepted by :func:`pandas.Series.reindex` plus
            - next (similar to bfill)
            - prev (similar to ffill)
            - closest (similar to nearest)

        Returns:
            The realigned Tsd

        """
        method = _get_restrict_method(align)
        ix = TimeUnits.format_timestamps(t.index.values)

        rest_t = self.reindex(ix, method=method)
        return rest_t

    def restrict(self, iset, keep_labels=False):
        """
        Restricts the Tsd to a set of times delimited by a :func:`~neuroseries.interval_set.IntervalSet`

        Args:
            iset: the restricting interval set
            keep_labels:

        Returns:
        # changed col to 0
        """
        ix = iset.in_interval(self)
        tsd_r = pd.DataFrame(self, copy=True)
        col = tsd_r.columns[0]
        tsd_r['interval'] = ix
        ix = ~np.isnan(ix)
        tsd_r = tsd_r[ix]
        if not keep_labels:
            s = tsd_r.iloc[:,0]
            return Tsd(s)
        return Tsd(tsd_r, copy=True)

    def gaps(self, min_gap, method='absolute'):
        """
        finds gaps in a tsd
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet containing the gaps in the TSd
        """
        return gaps_func(self, min_gap, method)

    def support(self, min_gap, method='absolute'):
        """
        find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet
        """
        return support_func(self, min_gap, method)

    def start_time(self, units='us'):
        return self.times(units=units)[0]

    def end_time(self, units='us'):
        return self.times(units=units)[-1]

    def store(self, the_store, key, **kwargs):
        data_to_store = self.as_series()

        the_store[key] = data_to_store
        # noinspection PyProtectedMember
        metadata = {k: getattr(self, k) for k in self._metadata}
        the_store.put(key, data_to_store, metadata, **kwargs)

    @property
    def r(self):
        if Range.interval is None:
            raise ValueError('no range interval set')
        if self.r_cache is None:
            self.r_cache = self.restrict(Range.interval)
            Range.cached_objects.append(self)

        return self.r_cache

    def invalidate_restrict_cache(self):
        self.r_cache = None

    @property
    def _constructor(self):
        return Tsd


# noinspection PyAbstractClass
class TsdFrame(pd.DataFrame):
    def __init__(self, t, d=None, time_units=None, **kwargs):
        if isinstance(t, (pd.DataFrame, SingleBlockManager, BlockManager)):
            super().__init__(t, **kwargs)
        else:
            t = TimeUnits.format_timestamps(t, time_units)
            super().__init__(index=t, data=d, **kwargs)
        self.index.name = "Time (us)"
        self._metadata.append("nts_class")
        self.nts_class = self.__class__.__name__
        self.r_cache = None

    def times(self, units=None):
        return TimeUnits.return_timestamps(self.index.values.astype(np.float64), units)

    def as_dataframe(self, copy=True):
        """
        :return: copy of the data in a DataFrame (strip Tsd class label)
        """
        return pd.DataFrame(self, copy=copy)

    def as_units(self, units=None):
        """
        returns a DataFrame with time expressed in the desired unit
        :param units: us (s), ms, or s
        :return: DataFrame with adjusted times
        """
        t = self.index.values.copy()
        t = TimeUnits.return_timestamps(t, units)
        df = pd.DataFrame(index=t, data=self.values)
        units_str = units
        if not units_str:
            units_str = 'us'
        df.index.name = "Time (" + units_str + ")"
        df.columns = self.columns.copy()
        return df

    def plot(self, units=None):
        """
        makes a plot with the units of choices
        Args:
            units: us (s), ms, or s

        Returns:
            None
        """

        dz = pd.DataFrame(index=self.times(units=units), data=self.values, columns=self.columns, copy=False)
        units_str = units
        if not units_str:
            units_str = 'us'
        dz.index.name = "Time (" + units_str + ")"
        dz.plot()

    def data(self):
        if len(self.columns) == 1:
            return self.values.ravel()
        return self.values

    def realign(self, t, align='closest'):
        method = _get_restrict_method(align)
        ix = TimeUnits.format_timestamps(t)

        rest_t = self.reindex(ix, method=method, columns=self.columns.values)
        return rest_t

    def restrict(self, iset, keep_labels=False):
        ix = iset.in_interval(self)
        tsd_r = pd.DataFrame(self, copy=True)
        tsd_r['interval'] = ix
        ix = ~np.isnan(ix)
        tsd_r = tsd_r[ix]
        if not keep_labels:
            del tsd_r['interval']
        return TsdFrame(tsd_r, copy=True)

    def gaps(self, min_gap, method='absolute'):
        """
        finds gaps in a tsd
        :param self: a Tsd/TsdFrame
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet containing the gaps in the TSd
        """
        return gaps_func(self, min_gap, method)

    def support(self, min_gap, method='absolute'):
        """
        find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd
        :param min_gap: the minimum gap that will be considered
        :param method: 'absolute': min gap is expressed in time (us), 'median',
        min_gap expressed in units of the median inter-sample event
        :return: an IntervalSet
        """
        return support_func(self, min_gap, method)

    def store(self, the_store, key, **kwargs):
        data_to_store = pd.DataFrame(self)
        the_store[key] = data_to_store
        # noinspection PyProtectedMember
        metadata = {k: getattr(self, k) for k in self._metadata}
        the_store.put(key, data_to_store, metadata, **kwargs)

    def start_time(self, units='us'):
        return self.times(units=units)[0]

    def end_time(self, units='us'):
        return self.times(units=units)[-1]

    @property
    def _constructor(self):
        return TsdFrame

    @property
    def _constructor_sliced(self):
        return Tsd

    @property
    def r(self):
        if Range.interval is None:
            raise ValueError('no range interval set')
        if self.r_cache is None:
            self.r_cache = self.restrict(Range.interval)
            Range.cached_objects.append(self)

        return self.r_cache

    def invalidate_restrict_cache(self):
        self.r_cache = None


# noinspection PyAbstractClass
class Ts(Tsd):
    def __init__(self, t, time_units=None, **kwargs):
        super().__init__(t, None, time_units=time_units, **kwargs)
        self.nts_class = self.__class__.__name__


def gaps_func(data, min_gap, method='absolute'):
    """
    finds gaps in a tsd
    :param data: a Tsd/TsdFrame
    :param min_gap: the minimum gap that will be considered
    :param method: 'absolute': min gap is expressed in time (us), 'median',
    min_gap expressed in units of the median inter-sample event
    :return: an IntervalSet containing the gaps in the TSd
    """
    dt = np.diff(data.times(units='us'))

    if method == 'absolute':
        pass
    elif method == 'median':
        md = np.median(dt)
        min_gap *= md
    else:
        raise ValueError('unrecognized method')

    ix = np.where(dt > min_gap)
    t = data.times()
    st = t[ix] + 1
    en = t[(np.array(ix) + 1)] - 1
    from neuroseries.interval_set import IntervalSet
    return IntervalSet(st, en)


def support_func(data, min_gap, method='absolute'):
    """
    find the smallest (to a min_gap resolution) IntervalSet containing all the times in the Tsd
    :param data: a Tsd/TsdFrame
    :param min_gap: the minimum gap that will be considered
    :param method: 'absolute': min gap is expressed in time (us), 'median',
    min_gap expressed in units of the median inter-sample event
    :return: an IntervalSet
    """

    here_gaps = data.gaps(min_gap, method=method)
    t = data.times('us')
    from neuroseries.interval_set import IntervalSet
    span = IntervalSet(t[0] - 1, t[-1] + 1)
    support_here = span.set_diff(here_gaps)
    return support_here


# noinspection PyUnusedLocal
def filter_time_series(data, columns=None):
    pass


def store(data, the_store, key, **kwargs):
    if isinstance(data, Tsd):
        data_to_store = data.as_series()
    else:
        data_to_store = pd.DataFrame(data)

    the_store[key] = data_to_store
    # noinspection PyProtectedMember
    metadata = {k: getattr(data, k) for k in data._metadata}
    the_store.put(key, data_to_store, metadata, **kwargs)


def extract_from(storer):
    from neuroseries.interval_set import IntervalSet
    ks = storer.keys()
    extractable_classes = [Ts, Tsd, TsdFrame, IntervalSet]
    extractable_classes_id = {c.__name__: c for c in extractable_classes}

    variables = {}
    for k in ks:
        k = k[1:]
        (v, metadata) = storer.get_with_metadata(k)
        if metadata is not None and \
                        'nts_class' in metadata and \
                        metadata['nts_class'] in extractable_classes_id:
            variables[k] = extractable_classes_id[metadata['nts_class']](v)

        if hasattr(v, 'nts_class') and v.nts_class in extractable_classes_id:
            variables[k] = extractable_classes_id[v.nts_class](v)
    return variables
