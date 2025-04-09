import copy
import inspect
import itertools
import warnings
from collections import UserDict
from numbers import Number
from typing import Union
from tabulate import tabulate
import re

import numpy as np
import pandas as pd

from .utils import _get_terminal_size


def add_meta_docstring(meta_func, sep="\n"):
    meta_doc = getattr(_MetadataMixin, meta_func).__doc__

    def _decorator(func):
        func.__doc__ = sep.join([meta_doc, func.__doc__])
        return func

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
        if name in self._metadata.columns:
            return self._metadata[name]
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
        if metadata is not None:
            if isinstance(metadata, pd.DataFrame):
                [
                    self._raise_invalid_metadata_column_name(col)
                    for col in metadata.columns
                ]
            elif isinstance(metadata, dict):
                [self._raise_invalid_metadata_column_name(k) for k in metadata.keys()]

            elif isinstance(metadata, pd.Series) and len(self) == 1:
                [self._raise_invalid_metadata_column_name(k) for k in metadata.index]

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
            if isinstance(metadata, pd.DataFrame):
                if np.all(self.metadata_index == metadata.index.values):
                    # self._metadata[metadata.columns] = metadata
                    kwargs = {**metadata, **kwargs}
                else:
                    raise ValueError("Metadata index does not match")
            elif isinstance(metadata, (dict, _Metadata)):
                # join metadata with kwargs to use checks below
                kwargs = {**metadata, **kwargs}

            elif isinstance(metadata, pd.Series) and (len(self.metadata_index) == 1):
                # allow series to be passed if only one interval
                for key, val in metadata.items():
                    self._metadata[key] = np.array(val)
                    # self._metadata[key] = pd.Series(val, index=self.metadata_index)

            elif isinstance(metadata, (pd.Series, np.ndarray, list)):
                raise TypeError(
                    f"Metadata with type {type(metadata)} should be passed as a keyword argument."
                )
            else:
                not_set.append(metadata)
        if len(kwargs):
            for k, v in kwargs.items():

                if isinstance(v, pd.Series):
                    if np.all(self.metadata_index == v.index.values):
                        self._metadata[k] = np.array(v)
                    else:
                        raise ValueError(
                            "Metadata index does not match for argument {}".format(k)
                        )

                elif isinstance(v, (np.ndarray, list, tuple)):
                    if len(self.metadata_index) == len(v):
                        self._metadata[k] = np.array(v)
                    else:
                        raise ValueError(
                            f"input array length {len(v)} does not match metadata length {len(self.metadata_index)}."
                        )

                elif (hasattr(v, "__iter__") is False) and (
                    len(self.metadata_index) == 1
                ):
                    # if only one index and metadata is non-iterable, pack into iterable for single assignment
                    self._metadata[k] = np.array([v])

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
            - list of str: multiple metadata column names or metadata indices (for TsdFrame with string column names)
            - Number: metadata index (for TsGroup and IntervalSet)
            - list, np.ndarray, pd.Series: metadata index (for TsGroup and IntervalSet)
            - tuple: metadata index and column name (for TsGroup and IntervalSet)

        Returns
        -------
        dict or np.array or Any (for single location)
            The metadata information based on the key provided.

        Raises
        ------
        IndexError
            If the metadata index is not found.
        """
        if isinstance(key, str) and (key in self.metadata_columns):
            # single metadata column
            return self._metadata[key]

        elif (
            isinstance(key, (list, np.ndarray))
            and all(isinstance(k, str) for k in key)
            and all(k in self.metadata_columns for k in key)
        ):
            # multiple metadata columns
            return self._metadata[key]

        else:
            # everything else, use .loc
            return self._metadata.loc[key]

    def drop_info(self, key):
        """
        Drop metadata based on metadata column name

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
        This field should only be set by _MetadataLoc and _MetadataILoc in order to preserve the metadata dictionary type.
    """

    def __init__(self, index, data=None):
        super().__init__()
        self.index = index
        if data is not None:
            self.update(data)

    def __repr__(self):
        # Start by determining how many columns and rows.
        # This can be unique for each object
        cols, rows = _get_terminal_size()
        # max_cols = np.maximum(cols // 12, 5)
        max_rows = np.maximum(rows - 10, 2)
        # By default, the first three columns should always show.
        data = {"i": self.index, **self.data}
        if len(self.index) > max_rows:
            n_rows = max_rows // 2
            data = {
                k: np.hstack((v[:n_rows], "...", v[-n_rows:])) for k, v in data.items()
            }
        repstr = tabulate(data, headers="keys", tablefmt="github", stralign="right")
        # reformat table to remove some space and only have rule between first two columns
        # split at new line, remove some space
        repstr = [s[2:-2] for s in repstr.split("\n")]
        # split at first rule
        repstr = [re.split(r"\|", s, 1) for s in repstr]
        # replace remaining rules between columns, remove index label
        repstr = [
            " " * len(s[0][:-1]) + "   " + re.sub(r".\|.", "  ", s[1][1:])
            for s in repstr[:2]
        ] + [s[0][:-1] + " | " + re.sub(r".\|.", "  ", s[1][1:]) for s in repstr[2:]]
        return "\n".join(repstr)  # join back together

    def __getitem__(self, key):
        """
        Wrapper around typical dictionary __getitem__ to allow for multiple key indexing.
        """
        try:
            if isinstance(key, list):
                return _Metadata(
                    index=self.index, data={key: self.data[key] for key in key}
                )
            else:
                return super().__getitem__(key)
        except KeyError:
            raise IndexError(
                f"Metadata column(s) {key} not found. Metadata columns are {self.columns}"
            )

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
        return pd.DataFrame(self.data, index=self.index)

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
                k: self.index[np.where(self.data[by] == k)[0]]
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
            return {k: self.index[v] for k, v in groups.items() if len(v)}

    def copy(self):
        """
        Return a deep copy of the metadata object.
        """
        return copy.deepcopy(self)

    def reset_index(self):
        """
        Reset metadata index to default range index.
        """
        self.index = np.arange(len(self.index))
        return self

    def join(self, other, axis=1):
        """
        Join metadata with another metadata object. Operates in place.
        Can only join metadata with metadata with matching index along axis 1,
        and metadata with matching columns along axis 0.

        When joining columns, `other` must have unique columns from the caller.

        When joining indices, the indices and rows of `other` are appended to the rows of the caller.
        The resulting indices are not sorted and may not be unique. Presently, this feature is not used.

        Parameters
        ----------
        other : _Metadata
            Metadata object to join with.
        axis : int, optional
            Axis to join along. 0 for rows (index), 1 for columns. Default is 1.

        Returns
        -------
        _Metadata
            Joined metadata object.
        """
        if not isinstance(other, _Metadata):
            raise TypeError("Can only join with another _Metadata object")

        if axis == 0:
            if np.all(self.columns == other.columns):
                warnings.warn(
                    "Joining metadata along axis 0 may result in duplicate and unsorted indices.",
                )
                # join metadata with matching columns
                self.index = np.concatenate([self.index, other.index])
                for k, v in other.data.items():
                    self.data[k] = np.concatenate([self.data[k], v])
            else:
                raise ValueError("Column names must match for join along axis 0.")

        elif axis == 1:
            if (len(self.index) == len(other.index)) and np.all(
                self.index == other.index
            ):
                no_join = [k for k in other.columns if k in self.columns]
                if no_join:
                    raise ValueError(
                        f"Metadata column(s) {no_join} already exists. Cannot join metadata."
                    )
                # join metadata with matching index
                for k, v in other.data.items():
                    self.data[k] = v
            else:
                raise ValueError("Metadata indices must match for join along axis 1.")

        else:
            raise ValueError(
                f"Axis {axis} out of bounds for joining metadata. Must be 0 or 1."
            )

        return self

    def merge(self, other):
        if not isinstance(other, _Metadata):
            raise TypeError("Can only join with another _Metadata object")

        if np.all(self.columns == other.columns):
            warnings.warn(
                "Joining metadata along axis 0 may result in duplicate and unsorted indices.",
            )
            # join metadata with matching columns
            self.index = np.concatenate([self.index, other.index])
            for k, v in other.data.items():
                self.data[k] = np.concatenate([self.data[k], v])
        else:
            raise ValueError("Column names must match for join along axis 0.")

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
        # unpack index and columns based from key
        if isinstance(key, tuple):
            if len(key) == 2:
                index = key[0]
                columns = key[1]
            else:
                raise IndexError(
                    f"Too many indices for metadata.loc: Metadata is 2-dimensional"
                )
        else:
            index = key
            columns = self.keys

        if isinstance(index, (Number, str)):
            # metadata.loc[Number or str]
            if index in self.index:
                idx = self.index_map[index]  # indexer from index map
            else:
                # error for index not found
                raise IndexError(
                    f"Metadata index '{index}' not found. Metadata indices are {self.index}"
                )

        elif hasattr(index, "dtype") or isinstance(index, list):
            # list or array_like
            if np.issubdtype(np.array(index).dtype, bool):
                # metadata.loc[bool array], boolean indexing
                if len(index) == len(self.index):
                    idx = index

                    if hasattr(index, "index") and not isinstance(index, list):
                        # boolean values associated with an index (i.e. pandas Series)
                        if all(i in self.index for i in index.index):
                            # get index where boolean is True
                            index = np.sort(index[idx].index)
                            # use _get_indexer in case index was not sorted
                            idx = self._get_indexder(index)
                        else:
                            raise IndexError(
                                f"Index of boolean cannot be aligned to index of metadata"
                            )
                    else:
                        index = self.index[idx]

                else:
                    raise IndexError(
                        f"Boolean index length {len(index)} does not match metadata index length {len(self.index)}."
                    )

            else:
                # metadata.loc[array_like], check that values correspond to index
                if all(i in self.index for i in index):
                    idx = self._get_indexder(index)
                else:
                    # error for index not found
                    raise IndexError(
                        f"Metadata indices {[i for i in index if i not in self.index]} not found. Metadata indices are {self.index}"
                    )

        elif isinstance(index, slice) and (index == slice(None)):
            # metadata.loc[:], keep original index
            idx = index  # indexer is the slice
            index = self.index  # original index

        else:
            # metadata.loc[unknown], unknown index
            raise IndexError(f"Unknown metadata index {index}")

        # get data for given columns, check that column names are strings
        if isinstance(columns, str):
            if columns in self.keys:
                # metadata.loc[:, str], single column
                return self.data[columns][idx]
            else:
                # error for column not found
                raise KeyError(
                    f"Metadata column '{columns}' not found. Metadata columns are {self.keys}"
                )

        elif isinstance(columns, (list, np.ndarray)) and all(
            isinstance(k, str) for k in columns
        ):
            if all(k in self.keys for k in columns):
                # metadata.loc[:, [*str]], multiple columns
                data = {k: self.data[k][idx] for k in columns}
                return _Metadata(index, data)
            else:
                # informative error for columns not found
                raise KeyError(
                    f"Metadata columns {[c for c in columns if c not in self.keys]} not found. Metadata columns are {self.keys}"
                )
        else:
            # metadata.loc[:, unknown], unknown columns, probably a type error
            raise TypeError(
                f"Unknown metadata columns {columns}. Columns must be strings."
            )

    def _get_indexder(self, vals):
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
        self.keys = metadata.columns

    def __getitem__(self, key):
        if isinstance(key, Number):
            # metadata.iloc[Number], single row across all columns
            index = self.index[key]
            data = {k: [self.data[k][key]] for k in self.keys}

        elif isinstance(key, (slice, list, np.ndarray, pd.Index, pd.Series)):
            # metadata.iloc[array_like], multiple rows across all columns
            index = self.index[key]
            data = {k: self.data[k][key] for k in self.keys}

        elif isinstance(key, tuple) and len(key) == 2:
            columns = self.keys[key[1]]
            index = self.index[key[0]]

            if isinstance(key[0], Number):
                # metadata.iloc[Number, *], single row across column(s)
                data = {k: [self.data[k][key[0]]] for k in columns}
            else:
                # metadata.iloc[array_like, *], multiple rows across column(s)
                data = {k: self.data[k][key[0]] for k in columns}

        else:
            raise IndexError(f"Unknown metadata index {key}")

        return _Metadata(index, data)
