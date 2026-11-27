"""

Similar to pandas.Index, `TsIndex` holds the timestamps associated with the data of a time series.
This class deals with conversion between different time units for all pynapple objects as well
as making sure that timestamps are property sorted before initializing any objects.

    - `us`: microseconds
    - `ms`: milliseconds
    - `s`: seconds  (overall default)
"""

import contextvars
from contextlib import contextmanager
from warnings import warn

import numpy as np

from .config import nap_config

# Internal, context-local flag asserting that arrays handed to a constructor are
# ALREADY canonical: float64, sorted, and (for the Tsd family) restricted to
# their time support. When set, constructors skip the corresponding O(n)
# validation/normalization steps that would otherwise be pure redundancy on data
# produced by another pynapple operation.
#
# This is a hard internal invariant contract, NOT a user-facing option: enabling
# it where the invariant does not hold yields silent data corruption rather than
# a validation error. It must only ever wrap a single reconstruction whose inputs
# are provably clean (see `trusted_construction`).
_trust_construction = contextvars.ContextVar("_trust_construction", default=False)


def is_trusted_construction():
    """Return True when inside a :func:`trusted_construction` context."""
    return _trust_construction.get()


@contextmanager
def trusted_construction(enabled=True):
    """Assert that constructor inputs are already canonical (float64, sorted,
    within support), so constructors skip revalidation.

    Token-based set/reset makes it exception-safe, correctly nested, and isolated
    per thread / async context. Wrap only the single reconstruction call, never a
    whole method body, so incidental constructions are not accidentally trusted.

    Parameters
    ----------
    enabled : bool
        When False, this is a no-op (validation runs as usual). Lets a call site
        assert cleanliness conditionally, e.g. only for order-preserving keys.
    """
    if not enabled:
        yield
        return
    token = _trust_construction.set(True)
    try:
        yield
    finally:
        _trust_construction.reset(token)


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
            return t
        #     t = np.around(t, nap_config.time_index_precision)
        elif units == "ms":
            return np.around(t / 1.0e3, nap_config.time_index_precision)
        elif units == "us":
            return np.around(t / 1.0e6, nap_config.time_index_precision)
        else:
            raise ValueError("unrecognized time units type")

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
            return t
        #     t = np.around(t)#, nap_config.time_index_precision)
        elif units == "ms":
            return np.around(t * 1.0e3, nap_config.time_index_precision)
        elif units == "us":
            return np.around(t * 1.0e6, nap_config.time_index_precision)
        else:
            raise ValueError("unrecognized time units type")

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
            if give_warning and not nap_config.suppress_time_index_sorting_warnings:
                warn("timestamps are not sorted", UserWarning)
            t = np.sort(t)
        return t

    def __new__(cls, t, time_units="s"):
        assert t.ndim == 1, "t should be 1 dimensional"
        if not _trust_construction.get():
            # canonicalize: cast to float64 (also a defensive copy so the index
            # never aliases the caller's array), convert units, ensure sorted.
            t = t.astype(np.float64)
            t = TsIndex.format_timestamps(t, time_units)
            t = TsIndex.sort_timestamps(t)
        else:
            # trusted: caller guarantees sorted, already in seconds. Still ensure
            # float64, but copy only on dtype mismatch instead of unconditionally.
            t = np.asarray(t, dtype=np.float64)
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

    def __getitem__(self, key):
        """Index timestamps, rejecting multidimensional results."""
        result = super().__getitem__(key)
        if isinstance(result, np.ndarray) and result.ndim > 1:
            raise ValueError("TsIndex must remain one-dimensional.")
        return result

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
