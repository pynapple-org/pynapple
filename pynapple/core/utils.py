# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2024-02-09 11:45:45
# @Last Modified by:   gviejo
# @Last Modified time: 2024-03-03 06:28:59

"""
    Utility functions
"""

import warnings

import numpy as np
from numba import jit

from .config import nap_config


def is_array_like(obj):
    """
    Check if an object is array-like.

    This function determines if an object has array-like properties.
    An object is considered array-like if it has attributes typically associated with arrays
    (such as `.shape`, `.dtype`, and `.ndim`), supports indexing, and is iterable.

    Parameters
    ----------
    obj : object
        The object to check for array-like properties.

    Returns
    -------
    bool
        True if the object is array-like, False otherwise.

    Notes
    -----
    This function uses a combination of checks for attributes (`shape`, `dtype`, `ndim`),
    indexability, and iterability to determine if the given object behaves like an array.
    It is designed to be flexible and work with various types of array-like objects, including
    but not limited to NumPy arrays and JAX arrays. However, it may not be full proof for all
    possible array-like types or objects that mimic these properties without being suitable for
    numerical operations.

    """
    # Check for array-like attributes
    has_shape = hasattr(obj, "shape")
    has_dtype = hasattr(obj, "dtype")
    has_ndim = hasattr(obj, "ndim")

    # Check for indexability (try to access the first element)
    try:
        obj[0]
        is_indexable = True
    except (TypeError, IndexError):
        is_indexable = False

    # Check for iterable property
    try:
        iter(obj)
        is_iterable = True
    except TypeError:
        is_iterable = False

    # not_tsd_type = not isinstance(obj, _AbstractTsd)

    return (
        has_shape
        and has_dtype
        and has_ndim
        and is_indexable
        and is_iterable
        # and not_tsd_type
    )


def convert_to_numpy(array, array_name):
    """
    Convert an input array-like object to a NumPy array.

    This function attempts to convert an input object to a NumPy array using `np.asarray`.
    If the input is not already a NumPy ndarray, it issues a warning indicating that a conversion
    has taken place and shows the original type of the input. This function is useful for
    ensuring compatibility with Numba operations in cases where the input might come from
    various array-like sources (for instance, jax.numpy.Array).

    Parameters
    ----------
    array : array_like
        The input object to convert. This can be any object that `np.asarray` is capable of
        converting to a NumPy array, such as lists, tuples, and other array-like objects,
        including those from libraries like JAX or TensorFlow that adhere to the array interface.
    array_name : str
        The name of the variable that we are converting, printed in the warning message.

    Returns
    -------
    ndarray
        A NumPy ndarray representation of the input `values`. If `values` is already a NumPy
        ndarray, it is returned unchanged. Otherwise, a new NumPy ndarray is created and returned.

    Warnings
    --------
    A warning is issued if the input `values` is not already a NumPy ndarray, indicating
    that a conversion has taken place and showing the original type of the input.

    """
    if (
        not isinstance(array, np.ndarray)
        and not nap_config.suppress_conversion_warnings
    ):
        original_type = type(array).__name__
        warnings.warn(
            f"Converting '{array_name}' to numpy.array. The provided array was of type '{original_type}'.",
            UserWarning,
        )
    return np.asarray(array)


def _split_tsd(func, tsd, indices_or_sections, axis=0):
    """
    Wrappers of numpy split functions
    """
    if func in [np.split, np.array_split, np.vsplit] and axis == 0:
        out = func._implementation(tsd.values, indices_or_sections)
        index_list = np.split(tsd.index.values, indices_or_sections)
        kwargs = {"columns": tsd.columns.values} if hasattr(tsd, "columns") else {}
        return [tsd.__class__(t=t, d=d, **kwargs) for t, d in zip(index_list, out)]
    elif func in [np.dsplit, np.hsplit]:
        out = func._implementation(tsd.values, indices_or_sections)
        kwargs = {"columns": tsd.columns.values} if hasattr(tsd, "columns") else {}
        return [tsd.__class__(t=tsd.index, d=d, **kwargs) for d in out]
    else:
        return func._implementation(tsd.values, indices_or_sections, axis)


def _concatenate_tsd(func, tsds):
    """
    Wrappers of np.concatenate and np.vstack
    """
    if isinstance(tsds, (tuple, list)):
        assert all(
            [hasattr(tsd, "nap_class") and hasattr(tsd, "values") for tsd in tsds]
        ), "Inputs should be Tsd, TsdFrame or TsdTensor"

        nap_type = np.unique([tsd.nap_class for tsd in tsds])
        assert len(nap_type) == 1, "Objects should all be the same."

        if len(tsds) > 1:
            new_index = np.hstack([tsd.index.values for tsd in tsds])
            if np.any(np.diff(new_index) <= 0):
                raise RuntimeError(
                    "The order of the Tsd index should be strictly increasing and non overlapping."
                )

            if nap_type == "Tsd":
                new_values = func._implementation(
                    [tsd.values[:, np.newaxis] for tsd in tsds]
                )
                new_values = new_values.flatten()
            else:
                new_values = func._implementation([tsd.values for tsd in tsds])

            # Joining Time support
            time_support = tsds[0].time_support
            for tsd in tsds:
                time_support = time_support.union(tsd.time_support)

            kwargs = {"columns": tsds[0].columns} if hasattr(tsds[0], "columns") else {}

            return tsds[0].__class__(
                t=new_index, d=new_values, time_support=time_support, **kwargs
            )

        else:
            return tsds[0]
    else:
        raise TypeError


@jit(nopython=True)
def _jitfix_iset(start, end):
    """
    0 - > "Some starts and ends are equal. Removing 1 microsecond!",
    1 - > "Some ends precede the relative start. Dropping them!",
    2 - > "Some starts precede the previous end. Joining them!",
    3 - > "Some epochs have no duration"

    Parameters
    ----------
    start : numpy.ndarray
        Description
    end : numpy.ndarray
        Description

    Returns
    -------
    TYPE
        Description
    """
    to_warn = np.zeros(4, dtype=np.bool_)

    m = start.shape[0]

    data = np.zeros((m, 2), dtype=np.float64)

    i = 0
    ct = 0

    while i < m:
        newstart = start[i]
        newend = end[i]

        while i < m:
            if end[i] == start[i]:
                to_warn[3] = True
                i += 1
            else:
                newstart = start[i]
                newend = end[i]
                break

        while i < m:
            if end[i] < start[i]:
                to_warn[1] = True
                i += 1
            else:
                newstart = start[i]
                newend = end[i]
                break

        while i < m - 1:
            if start[i + 1] < end[i]:
                to_warn[2] = True
                i += 1
                newend = max(end[i - 1], end[i])
            else:
                break

        if i < m - 1:
            if newend == start[i + 1]:
                to_warn[0] = True
                newend -= 1.0e-6

        data[ct, 0] = newstart
        data[ct, 1] = newend

        ct += 1
        i += 1

    data = data[0:ct]

    return (data, to_warn)


class _TsdFrameSliceHelper:
    def __init__(self, tsdframe):
        self.tsdframe = tsdframe

    def __getitem__(self, key):
        if hasattr(key, "__iter__") and not isinstance(key, str):
            for k in key:
                if k not in self.tsdframe.columns:
                    raise IndexError(str(k))
            index = self.tsdframe.columns.get_indexer(key)
        else:
            if key not in self.tsdframe.columns:
                raise IndexError(str(key))
            index = self.tsdframe.columns.get_indexer([key])

        if len(index) == 1:
            return self.tsdframe.__getitem__((slice(None, None, None), index[0]))
        else:
            return self.tsdframe.__getitem__(
                (slice(None, None, None), index), columns=key
            )


class _IntervalSetSliceHelper:
    """
    This class helps `IntervalSet` behaves like pandas.DataFrame for the `loc` function.

    Attributes
    ----------
    intervalset : `IntervalSet` to slice

    """

    def __init__(self, intervalset):
        """Class for `loc` slicing function

        Parameters
        ----------
        intervalset : IntervalSet

        """
        self.intervalset = intervalset

    def __getitem__(self, key):
        """Getters for `IntervalSet.loc`. Mimics pandas.DataFrame.

        Parameters
        ----------
        key : int, list or tuple

        Returns
        -------
        IntervalSet or Number or numpy.ndarray

        Raises
        ------
        IndexError

        """
        if key in ["start", "end"]:
            return self.intervalset[key]
        elif isinstance(key, list):
            return self.intervalset[key]
        elif isinstance(key, int):
            return self.intervalset.values[key]
        else:
            if isinstance(key, tuple):
                if len(key) == 2:
                    if key[1] not in ["start", "end"]:
                        raise IndexError
                    out = self.intervalset[key[0]][key[1]]
                    if len(out) == 1:
                        return out[0]
                    else:
                        return out
                else:
                    raise IndexError
            else:
                raise IndexError
