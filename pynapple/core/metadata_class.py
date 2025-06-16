import copy
import inspect
import itertools
import warnings
from collections import UserDict
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd
from tabulate import tabulate

from .utils import _convert_iter_to_str, _get_terminal_size


def add_meta_docstring(meta_func, sep="\n"):
    meta_doc = getattr(_MetadataMixin, meta_func).__doc__

    def _decorator(func):
        func.__doc__ = sep.join([meta_doc, func.__doc__])
        return func

    return _decorator


def add_or_convert_metadata(func):
    """
    Decorator for backwards compatibility of objects picked with older versions of pynapple.
    """

    def _decorator(self, *args, **kwargs):
        if (
            (len(args) == 1)
            and isinstance(args[0], str)
            and (
                args[0]
                in ("__getstate__", "__setstate__", "__reduce__", "__reduce_ex__")
            )
        ):
            # special case for pickling due to infinite recursion in getattr
            raise AttributeError(args[0])

        if hasattr(self, "_metadata") is False:
            # add empty metadata
            _MetadataMixin.__init__(self)

        elif isinstance(self._metadata, pd.DataFrame):
            # convert metadata to dictionary
            self.__dict__["_metadata"] = _Metadata(
                self.metadata_index,
                data={k: np.array(v) for k, v in self._metadata.items()},
            )

        return func(self, *args, **kwargs)

    return _decorator


class _MetadataMixin:
    """
    An object containing metadata functionality for TsGroup, IntervalSet, or TsdFrame objects.
    This is a private mixin that is not meant to be instantiated on its own.
    """

    metadata_index: Union[np.ndarray, pd.Index]
    """Row index for metadata DataFrame. This matches the index for TsGroup and IntervalSet, and the columns for TsdFrame."""

    def __init__(self):
        """
        Metadata initializer. This sets the metadata index using properties of the inheriting class and initialized an empty metadata DataFrame.
        Metadata can be set using the `set_info()` method.
        """
        if self.__class__.__name__ == "TsdFrame":
            # metadata index is the same as the columns for TsdFrame
            self.metadata_index = self.columns
        else:
            # metadata index is the same as the index for TsGroup and IntervalSet
            self.metadata_index = self.index

        self._metadata = _Metadata(
            self.metadata_index
        )  # pd.DataFrame(index=self.metadata_index)

    def __dir__(self):
        """
        Adds metadata columns to the list of attributes.
        """
        return sorted(list(super().__dir__()) + self.metadata_columns)

    def __setattr__(self, name, value):
        """
        Allow metadata to be added via attribute assignment after initialization, assuming the attribute is not already defined.
        Sets metadata using the `set_info()` method.
        """
        # self._initialized must be defined in the class that inherits _MetadataMixin
        # and it must be set to True after metadata is initialized
        if self._initialized:
            self.set_info(**{name: value})
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        Allows access to metadata columns as properties if the metadata name does not conflict with existing attributes.
        """
        # avoid infinite recursion when pickling due to
        # self._metadata.column having attributes '__reduce__', '__reduce_ex__'
        if name in ("__getstate__", "__setstate__", "__reduce__", "__reduce_ex__"):
            raise AttributeError(name)
        # Check if the requested attribute is part of the metadata
        if name in self.metadata_columns:
            return self.get_info(name)
        else:
            # If the attribute is not part of the metadata, raise AttributeError
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key, value):
        """
        Allows metadata to be set using key indexing. Sets metadata using the `set_info()` method.
        """
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be strings!")
        self.set_info({key: value})

    def __getitem__(self, key):
        """
        Allows metadata to be accessed using key indexing. Calls the `get_info()` method.
        """
        return self.get_info(key)

    @property
    @add_or_convert_metadata
    def metadata(self):
        """
        Returns a read-only version (copy) of the _metadata DataFrame
        """
        # return copy so that metadata cannot modified in place
        return self._metadata.as_dataframe()

    @property
    def metadata_columns(self):
        """
        List of metadata column names.
        """
        return list(self._metadata.columns)

    def _raise_invalid_metadata_column_name(self, name):
        """
        Check if metadata name is valid and raise warnings if it cannot be accessed as an attribute or key.
        Prevents metadata names for TsdFrame to match data column names, and prevents "rate" from being changed for TsGroup.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"Invalid metadata type {type(name)}. Metadata column names must be strings!"
            )
        # warnings for metadata names that cannot be accessed as attributes or keys
        if name in self._class_attributes:
            if (self.nap_class == "TsGroup") and (name == "rate"):
                # special exception for TsGroup rate attribute
                raise ValueError(
                    f"Invalid metadata name '{name}'. Cannot overwrite TsGroup 'rate'!"
                )
            else:
                # existing non-metadata attribute
                warnings.warn(
                    f"Metadata name '{name}' overlaps with an existing attribute, and cannot be accessed as an attribute or key. Use 'get_info()' to access metadata."
                )
        elif hasattr(self, "columns") and (name in self.columns):
            if self.nap_class == "TsdFrame":
                # special exception for TsdFrame columns
                raise ValueError(
                    f"Invalid metadata name '{name}'. Metadata name must differ from {list(self.columns)} column names!"
                )
            else:
                # existing non-metadata column
                warnings.warn(
                    f"Metadata name '{name}' overlaps with an existing property, and cannot be accessed as an attribute or key. Use 'get_info()' to access metadata."
                )
        # elif name in self.metadata_columns:
        #     # warnings for metadata that already exists
        #     warnings.warn(f"Overwriting existing metadata column '{name}'.")

        # warnings for metadata that cannot be accessed as attributes
        if name.replace("_", "").isalnum() is False:
            # contains invalid characters
            warnings.warn(
                f"Metadata name '{name}' contains a special character, and cannot be accessed as an attribute. Use 'get_info()' or key indexing to access metadata."
            )
        elif (name[0].isalpha() is False) and (name[0] != "_"):
            # starts with a number
            warnings.warn(
                f"Metadata name '{name}' starts with a number, and cannot be accessed as an attribute. Use 'get_info()' or key indexing to access metadata."
            )

    def _check_metadata_column_names(self, metadata=None, **kwargs):
        """
        Throw warnings when metadata names cannot be accessed as attributes or keys. Wrapper for _raise_invalid_metadata_column_name.
        """
        if (metadata is not None) and (hasattr(metadata, "keys")):
            # dictionary-like object
            [self._raise_invalid_metadata_column_name(k) for k in metadata.keys()]

        for k in kwargs:
            self._raise_invalid_metadata_column_name(k)

    def set_info(self, metadata=None, **kwargs):
        """
        Add metadata information about the object. Metadata are saved as a dictionary.

        If the metadata name does not contain special nor overlaps with class attributes,
        it can also be set using attribute assignment.

        If the metadata name does not overlap with class-reserved keys,
        it can also be set using key assignment.

        Metadata entries (excluding "rate" for `TsGroup`) are mutable and can be overwritten.

        Parameters
        ----------
        metadata : pandas.DataFrame or dict or pandas.Series, optional
            Object containing metadata information, where metadata names are extracted from column names (pandas.DataFrame), key names (dict), or index (pandas.DataFrame).

            - If a pandas.DataFrame is passed, the index must match the metadata index.
            - If a dictionary is passed, the length of each value must match the metadata index length.
            - A pandas.Series can only be passed if the object has a single interval.

        **kwargs : optional
            Key-word arguments for setting metadata. Values can be either pandas.Series, numpy.ndarray, list or tuple, and must have the same length as the metadata index.
            If pandas.Series, the index must match the metadata index.
            If the object only has one index, non-iterable values are also accepted.

        Raises
        ------
        ValueError
            - If metadata index does not match input index (pandas.DataFrame, pandas.Series)
            - If input array length does not match metadata length (numpy.ndarray, list, tuple)
        RuntimeError
            If the metadata argument is passed as a pandas.Series (for more than one metadata index), numpy.ndarray, list or tuple.
        TypeError
            If key-word arguments are not of type `pandas.Series`, `tuple`, `list`, or `numpy.ndarray` and cannot be set.
        """
        # check for duplicate names and/or formatted names that cannot be accessed as attributes or keys
        self._check_metadata_column_names(metadata, **kwargs)
        not_set = []
        if metadata is not None:
            if hasattr(metadata, "keys"):
                # metadata is a dictionary-like object
                if (
                    hasattr(metadata, "index")
                    # exlude length 1 objects in case of pandas.Series
                    and (len(self.metadata_index) > 1)
                    and not np.all(self.metadata_index == metadata.index)
                ):
                    raise ValueError("Metadata index does not match.")
                else:
                    kwargs = {**metadata, **kwargs}

            else:
                raise TypeError(
                    f"Metadata with type {type(metadata)} should be passed as a keyword argument."
                )

        if len(kwargs):
            for k, v in kwargs.items():

                if hasattr(v, "keys") and hasattr(v, "index"):
                    # pandas.Series
                    if np.all(self.metadata_index == v.index):
                        self._metadata[k] = np.array(v)
                        self._metadata[k].setflags(write=False)
                    else:
                        raise ValueError(
                            f"Metadata index does not match for argument {k}"
                        )

                elif (len(self.metadata_index) == 1) and (
                    (hasattr(v, "__iter__") is False) or isinstance(v, str)
                ):
                    # special case for single index objects for non iterable objects or strings
                    self._metadata[k] = np.array([v])
                    self._metadata[k].setflags(write=False)

                elif hasattr(v, "__len__") and not isinstance(v, dict):
                    # object that has a length
                    if len(self.metadata_index) == len(v):
                        self._metadata[k] = np.array(v)
                        self._metadata[k].setflags(write=False)
                    elif len(self.metadata_index) == 1:
                        # special case for single index objects
                        self._metadata[k] = np.array([v])
                        self._metadata[k].setflags(write=False)
                    else:
                        raise ValueError(
                            f"input array length {len(v)} does not match metadata length {len(self.metadata_index)}."
                        )

                else:
                    not_set.append({k: v})

        if not_set:
            raise TypeError(
                f"Cannot set the following metadata:\n{not_set}.\nMetadata columns provided must be  "
                f"of type `panda.Series`, `tuple`, `list`, or `numpy.ndarray`."
            )

    def get_info(self, key):
        """
        Returns metadata based on metadata column name or index.

        If the metadata name does not contain special nor overlaps with class attributes,
        it can also be accessed as an attribute.

        If the metadata name does not overlap with class-reserved keys,
        it can also be accessed as a key.

        Parameters
        ----------
        key :
            - str: metadata column name or metadata index (for TsdFrame with string column names)
            - list of str: multiple metadata column names

        Returns
        -------
        dict or np.array or Any (for single location)
            The metadata information based on the key provided.

        Raises
        ------
        IndexError
            If the metadata index is not found.
        """
        if isinstance(key, str):
            if key in self.metadata_columns:
                # single metadata column
                # wrap in list in case of a 2D array
                return pd.Series(
                    index=self.metadata_index, data=list(self._metadata[key]), name=key
                )
            else:
                raise KeyError(
                    f"Metadata column {key} not found. Metadata columns are {self.metadata_columns}"
                )

        elif isinstance(key, (list, np.ndarray)) and all(
            isinstance(k, str) for k in key
        ):
            if all(k in self.metadata_columns for k in key):
                # multiple metadata columns
                # wrap in list in case of a 2D array
                return pd.DataFrame(
                    index=self.metadata_index,
                    data={k: list(self._metadata[k]) for k in key},
                )
            else:
                raise KeyError(
                    f"Metadata column(s) {[k for k in key if k not in self.metadata_columns]} not found. Metadata columns are {self.metadata_columns}"
                )
        else:
            raise TypeError(
                f"Invalid metadata column {key}. Metadata columns must be strings."
            )

    def drop_info(self, key):
        """
        Drop metadata based on metadata column name. Operates in place.

        Parameters
        ----------
        key : (str, list)
            Metadata column name(s) to drop.

        Returns
        -------
        None
        """
        if isinstance(key, str):
            if key in self.metadata_columns:
                if (self.nap_class == "TsGroup") and (key == "rate"):
                    raise ValueError("Cannot drop TsGroup 'rate'!")
                else:
                    del self._metadata[key]
            else:
                raise KeyError(
                    f"Metadata column {key} not found. Metadata columns are {self.metadata_columns}"
                )

        elif isinstance(key, (list, np.ndarray)) and all(
            isinstance(k, str) for k in key
        ):
            if (self.nap_class == "TsGroup") and ("rate" in key):
                raise ValueError("Cannot drop TsGroup 'rate'!")

            no_drop = [k for k in key if k not in self.metadata_columns]
            if no_drop:
                raise KeyError(
                    f"Metadata column(s) {no_drop} not found. Metadata columns are {self.metadata_columns}"
                )

            for k in key:
                if k in self.metadata_columns:
                    del self._metadata[k]

        else:
            raise TypeError(
                f"Invalid metadata column {key}. Metadata columns are {self.metadata_columns}"
            )

    def groupby(self, by, get_group=None):
        """
        Group pynapple object by metadata name(s).

        Parameters
        ----------
        by : str or list of str
            Metadata name(s) to group by.
        get_group : dictionary key, optional
            Name of the group to return.

        Returns
        -------
        dict or pynapple object
            Dictionary of object indices (dictionary values) corresponding to each group (dictionary keys), or pynapple object corresponding to 'get_group' if it has been supplied.

        Raises
        ------
        ValueError
            If metadata name does not exist.
        """
        if isinstance(by, str) and by not in self.metadata_columns:
            raise ValueError(
                f"Metadata column '{by}' not found. Metadata columns are {self.metadata_columns}"
            )
        elif isinstance(by, list):
            for b in by:
                if b not in self.metadata_columns:
                    raise ValueError(
                        f"Metadata column '{b}' not found. Metadata columns are {self.metadata_columns}"
                    )
        groups = self._metadata.groupby(by)
        if get_group is not None:
            if get_group not in groups.keys():
                raise ValueError(
                    f"Group '{get_group}' not found in metadata groups. Groups are {list(groups.keys())}"
                )
            idx = groups[get_group]
            if self.nap_class == "TsdFrame":
                return self.loc[idx]
            else:
                return self[idx]
        else:
            return groups

    def groupby_apply(self, by, func, input_key=None, **func_kwargs):
        """
        Apply a function to each group in a grouped pynapple object.

        Parameters
        ----------
        by : str or list of str
            Metadata name(s) to group by.
        func : function
            Function to apply to each group.
        input_key : str or None, optional
            Input key that the grouped object will be passed as. If None, the grouped object will be passed as the first positional argument.
        **func_kwargs : optional
            Additional keyword arguments to pass to the function. Any required positional arguments that are not the grouped object should be passed as keyword arguments.

        Returns
        -------
        dict
            Dictionary of results from applying the function to each group, where the keys are the group names and the values are the results.
        """

        if input_key is not None:
            if not isinstance(input_key, str):
                raise TypeError("input_key must be a string.")
            if input_key not in inspect.signature(func).parameters:
                raise KeyError(f"{func} does not have input parameter {input_key}.")

            def anon_func(x):
                return func(**{input_key: x, **func_kwargs})

        elif func_kwargs:

            def anon_func(x):
                return func(x, **func_kwargs)

        else:
            anon_func = func

        groups = self.groupby(by)
        if self.nap_class == "TsdFrame":
            out = {k: anon_func(self.loc[v]) for k, v in groups.items()}
        else:
            out = {k: anon_func(self[v]) for k, v in groups.items()}
        return out


class _Metadata(UserDict):
    """
    A custom dictionary class for storing metadata information.

    Parameters
    ----------
    index : np.ndarray
        Object index values.
    data : dict, optional
        Dictionary containing metadata information.
        This field should only be used internally to preserve the metadata dictionary type.
    """

    def __init__(self, index, data={}):
        super().__init__(data)
        self.index = index

    def __repr__(self):
        if hasattr(self.index, "__len__"):
            # Start by determining how many columns and rows.
            # This can be unique for each object
            cols, rows = _get_terminal_size()
            # max_cols = np.maximum(cols // 12, 5)
            max_rows = np.maximum(rows - 10, 2)
            # By default, the first three columns should always show.
            data = {
                " ": self.index,
                **{k: _convert_iter_to_str(v) for k, v in self.items()},
            }
            if len(self.index) > max_rows:
                n_rows = max_rows // 2
                data = {
                    k: np.hstack((v[:n_rows], "...", v[-n_rows:]), dtype=object)
                    for k, v in data.items()
                }
            return tabulate(data, headers="keys", tablefmt="plain", numalign="left")
        else:
            keys = np.array(list(self.keys()))
            values = np.array(
                [str(v) if hasattr(v, "__len__") else v for v in self.values()]
            )
            return (
                tabulate(np.vstack((keys, values)).T, tablefmt="plain")
                + f"\nIndex: {self.index}"
            )

    def __eq__(self, other):
        """
        Check if two metadata objects are equal.
        """
        if not isinstance(other, _Metadata):
            return False
        if not np.array_equal(self.index, other.index):
            return False
        if not np.array_equal(self.columns, other.columns):
            return False
        for k in self.data:
            if not np.array_equal(self.data[k], other.data[k]):
                return False
        return True

    def __getitem__(self, key):
        """
        Wrapper around typical dictionary __getitem__ to allow for multiple key indexing.
        """
        if isinstance(key, list):
            return _Metadata(
                index=self.index, data={key: self.data[key] for key in key}
            )
        else:
            return super().__getitem__(key)

    @property
    def columns(self):
        """
        Metadata keys (columns).
        """
        return list(self.data.keys())

    @property
    def loc(self):
        """
        Pandas-like indexing for metadata.
        """
        return _MetadataLoc(self)

    @property
    def iloc(self):
        """
        Numpy-like indexing for metadata.
        """
        return _MetadataILoc(self)

    @property
    def shape(self):
        """
        Metadata shape as (n_index, n_columns).
        """
        return (len(self.index), len(self.columns))

    @property
    def dtypes(self):
        """
        Dictonary of data types for each metadata column.
        """
        return {k: self.data[k].dtype for k in self.columns}

    def drop(self, key):
        """
        Drop metadata column(s).

        Parameters
        ----------
        key : str or list of str
            Metadata column name(s) to drop.

        Returns
        -------
        None
        """
        if isinstance(key, str):
            if key in self.columns:
                del self.data[key]
            else:
                raise KeyError(
                    f"Metadata column {key} not found. Metadata columns are {self.columns}"
                )

        elif isinstance(key, (list, np.ndarray)) and all(
            isinstance(k, str) for k in key
        ):
            no_drop = [k for k in key if k not in self.columns]
            if no_drop:
                raise KeyError(
                    f"Metadata column(s) {no_drop} not found. Metadata columns are {self.columns}"
                )

            for k in key:
                if k in self.columns:
                    del self.data[k]

        else:
            raise TypeError(
                f"Invalid metadata column {key}. Metadata columns are {self.columns}"
            )
        return self

    def as_dataframe(self):
        """
        Convert metadata dictionary to a pandas DataFrame.
        """
        # convert arrays to list to avoid shape issues if metadata is a 2D array
        data = {k: list(v) for k, v in self.data.items()}
        return pd.DataFrame(data, index=self.index)

    def groupby(self, by):
        """
        Grouping function for metadata.

        Parameters
        ----------
        by : str or list of str
            Metadata column name(s) to group by.

        Returns
        -------
        dict
            Dictionary of object indices (dictionary values) corresponding to each group (dictionary keys).
        """
        if isinstance(by, str):
            # groupby single column
            return {
                (k.item() if hasattr(k, "item") else k): self.index[
                    np.where(self.data[by] == k)[0]
                ]
                for k in np.unique(self.data[by])
            }

        elif isinstance(by, list):
            # groupby multiple columns
            groups = {
                k: np.where(
                    np.all([self.data[col] == k[c] for c, col in enumerate(by)], axis=0)
                )[0]
                for k in itertools.product(*[np.unique(self.data[col]) for col in by])
            }
            # use object index, remove empty groups
            return {
                tuple(k.item() if hasattr(k, "item") else k for k in k): self.index[v]
                for k, v in groups.items()
                if len(v)
            }

    def copy(self):
        """
        Return a deep copy of the metadata object.
        """
        return copy.deepcopy(self)

    def reset_index(self):
        """
        Reset metadata index to default range index.
        """
        if hasattr(self.index, "__len__"):
            self.index = np.arange(len(self.index))
        else:
            self.index = 0
        return self

    def join(self, other):
        """
        Join metadata with another metadata object. Operates in place.
        Can only join metadata with metadata with matching index.

        When joining, columns in `other` must be unique from columns of the caller.

        Parameters
        ----------
        other : _Metadata
            Metadata object to join with.

        Returns
        -------
        _Metadata
            Joined metadata object.
        """
        if not isinstance(other, (_Metadata, pd.DataFrame)):
            raise TypeError("Can only join with another _Metadata object")

        if (len(self.index) == len(other.index)) and np.all(self.index == other.index):
            no_join = [k for k in other.columns if k in self.columns]
            if no_join:
                raise ValueError(
                    f"Metadata column(s) {no_join} already exists. Cannot join metadata."
                )
            # join metadata with matching index
            for k, v in other.items():
                self.data[k] = v
        else:
            raise ValueError("Metadata indices must match for join along axis 1.")

        return self

    def merge(self, other):
        """
        Merge metadata with another metadata object. Operates in place.
        Can only merge metadata with matching columns.

        When merging, the indices and rows of `other` are appended to the rows of the caller.
        The resulting indices are not sorted and may not be unique.

        Parameters
        ----------
        other : _Metadata
            Metadata object to merge with.

        Returns
        -------
        _Metadata
            Merged metadata object.
        """
        if not isinstance(other, (_Metadata, pd.DataFrame)):
            raise TypeError("Can only merge with another _Metadata object")

        if np.all(self.columns == other.columns):
            warnings.warn(
                "Merging metadata may result in duplicate and unsorted indices.",
            )
            # join metadata with matching columns
            self.index = np.hstack([self.index, other.index])
            for k, v in other.items():
                self.data[k] = np.hstack([self.data[k], v])
        else:
            raise ValueError("Column names must match for merge.")

        return self


class _MetadataLoc:
    """
    Helper class for pandas-like indexing of metadata.
    Assumes that index corresponds to object index values in first axis, and metadata columns in second axis.
    """

    def __init__(self, metadata):
        self.data = metadata.data
        self.keys = metadata.columns
        self.index = metadata.index
        self.index_map = {k: v for v, k in enumerate(self.index)}

    def __getitem__(self, key):
        # unpack index and columns from key
        if isinstance(key, tuple):
            if len(key) == 2:
                index = key[0]
                columns = key[1]
            else:
                raise IndexError(
                    "Too many indices for metadata.loc: Metadata is 2-dimensional"
                )
        else:
            index = key
            columns = self.keys

        if isinstance(index, (Number, str)):
            # metadata.loc[Number or str]
            idx = self.index_map[index]  # indexer from index map

        elif hasattr(index, "dtype") or isinstance(index, list):
            # list or array_like
            if np.issubdtype(np.array(index).dtype, bool):
                # metadata.loc[bool array], boolean indexing
                idx = index
                index = self.index[idx]

            else:
                # metadata.loc[array_like]
                idx = self._get_indexer(index)

        elif isinstance(index, slice) and (index == slice(None)):
            # metadata.loc[:], keep original index
            idx = index  # indexer is the slice
            index = self.index  # original index

        else:
            # metadata.loc[unknown], unknown index
            raise TypeError(f"Unknown metadata index type {type(index)}")

        if isinstance(columns, str):
            data = {columns: self.data[columns][idx]}
        else:
            data = {k: self.data[k][idx] for k in columns}

        return _Metadata(index, data)

    def _get_indexer(self, vals):
        """
        Function that maps object index values to positional index.
        """
        return [self.index_map[val] for val in vals]


class _MetadataILoc:
    """
    Helper class for numpy-like indexing of metadata.
    Assumes that indices correspond to positional index of row (object index) in first axis and positional index of column (metadata column) in second axis.
    """

    def __init__(self, metadata):
        self.data = metadata.data
        self.index = metadata.index
        self.keys = np.array(metadata.columns)

    def __getitem__(self, key):
        # unpack index and columns from key
        if isinstance(key, tuple):
            if len(key) == 2:
                idx = key[0]
                index = self.index[key[0]]
                columns = self.keys[key[1]]
                if isinstance(columns, str):
                    columns = [columns]
            else:
                raise IndexError(
                    "Too many indices for metadata.loc: Metadata is 2-dimensional"
                )
        else:
            idx = key
            index = self.index[key]
            columns = self.keys

        data = {k: self.data[k][idx] for k in columns}

        return _Metadata(index, data)
