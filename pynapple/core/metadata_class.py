import warnings
from numbers import Number
from typing import Union
from collections import UserDict
from .utils import is_array_like

import numpy as np
import pandas as pd


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

        self._metadata = _Metadata()  # pd.DataFrame(index=self.metadata_index)

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
        self.set_info(**{key: value})

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
        return self._metadata.copy()

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
        elif hasattr(self, "columns") and name in self.columns:
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
        Add metadata information about the object. Metadata are saved as a pandas.DataFrame.

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
                if pd.Index.equals(self.metadata_index, metadata.index):
                    self._metadata[metadata.columns] = metadata
                else:
                    raise ValueError("Metadata index does not match")
            elif isinstance(metadata, dict):
                # merge metadata with kwargs to use checks below
                kwargs = {**metadata, **kwargs}

            elif isinstance(metadata, pd.Series) and (len(self) == 1):
                # allow series to be passed if only one interval
                for key, val in metadata.items():
                    self._metadata[key] = val

            elif isinstance(metadata, (pd.Series, np.ndarray, list)):
                raise RuntimeError("Argument should be passed as keyword argument.")
            else:
                not_set.append(metadata)
        if len(kwargs):
            for k, v in kwargs.items():

                if isinstance(v, pd.Series):
                    if pd.Index.equals(self.metadata_index, v.index):
                        self._metadata[k] = v
                    else:
                        raise ValueError(
                            "Metadata index does not match for argument {}".format(k)
                        )

                elif isinstance(v, (np.ndarray, list, tuple)):
                    if len(self.metadata_index) == len(v):
                        self._metadata[k] = v
                    else:
                        raise ValueError(
                            f"input array length {len(v)} does not match metadata length {len(self.metadata_index)}."
                        )

                elif (hasattr(v, "__iter__") is False) and (
                    len(self.metadata_index) == 1
                ):
                    # if only one index and metadata is non-iterable, pack into iterable for single assignment
                    self._metadata[k] = [v]

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
        pandas.Series or pandas.DataFrame or Any (for single location)
            The metadata information based on the key provided.

        Raises
        ------
        IndexError
            If the metadata index is not found.
        """
        # string indexing of one or more metadata columns
        if isinstance(key, str) or (
            isinstance(key, list) and all([isinstance(k, str) for k in key])
        ):
            if (
                isinstance(key, list) and all([k in self.metadata_index for k in key])
            ) or (isinstance(key, str) and (key in self.metadata_index)):
                # metadata index is a string
                return self._metadata.loc[key]
            else:
                # metadata[str] or metadata[[*str]]
                return self._metadata[key]

        elif isinstance(key, (Number, list, np.ndarray, pd.Series, pd.Index)) or (
            isinstance(key, tuple)
            and (
                isinstance(key[1], str)
                or (
                    isinstance(key[1], list)
                    and all([isinstance(k, str) for k in key[1]])
                )
            )
        ):
            # assume key is index, or tuple of index and column name
            # metadata[Number], metadata[array_like], metadata[Any, str], or metadata[Any, [*str]]
            return self._metadata.loc[key]

        elif isinstance(key, slice):
            # assume key as index slice
            # DataFrame's `loc` treats slices differently (inclusive of stop) than numpy
            # `iloc` exludes the stop index, like numpy
            return self._metadata.iloc[key]
        else:
            # we don't allow indexing columns with numbers, e.g. metadata[0,0]
            raise IndexError(f"Unknown metadata index {key}")

    def groupby(self, by, get_group=None):
        """
        Group pynapple object by metadata column(s).

        Parameters
        ----------
        by : str or list of str
            Metadata column name(s) to group by.
        get_group : dictionary key, optional
            Name of the group to return.

        Returns
        -------
        dict or pynapple object
            Dictionary of object indices (dictionary values) corresponding to each group (dictionary keys), or pynapple object corresponding to 'get_group' if it has been supplied.

        Raises
        ------
        ValueError
            If metadata column does not exist.
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
        groups = self._metadata.groupby(by).groups
        if get_group is not None:
            if get_group not in groups.keys():
                raise ValueError(
                    f"Group '{get_group}' not found in metadata groups. Groups are {list(groups.keys())}"
                )
            idx = groups[get_group]
            return self[np.array(idx)]
        else:
            return groups

    def groupby_apply(self, by, func, grouped_arg=None, **func_kwargs):
        """
        Apply a function to each group in a grouped pynapple object.

        Parameters
        ----------
        by : str or list of str
            Metadata column name(s) to group by.
        func : function
            Function to apply to each group.
        grouped_arg : str, optional
            Name of the function argument that the grouped object should be passed as. If none, the grouped object is passed as the first positional argument.
        func_kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        dict
            Dictionary of results from applying the function to each group, where the keys are the group names and the values are the results.
        """

        groups = self.groupby(by)

        if grouped_arg is None:
            out = {k: func(self[v], **func_kwargs) for k, v in groups.items()}
        else:
            out = {
                k: func(**{grouped_arg: self[v], **func_kwargs})
                for k, v in groups.items()
            }

        return out


class _Metadata(UserDict):

    def __init__(self):
        super().__init__()

    def __getitem__(self, key):
        return _MetadataSliceHelper(self)[key]

    @property
    def columns(self):
        return list(self.data.keys())

    @property
    def loc(self):
        return _MetadataSliceHelper(self)


class _MetadataSliceHelper:

    def __init__(self, metadata):
        self.metadata = metadata

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.metadata.data[key]

        elif isinstance(key, list) and all(isinstance(k, str) for k in key):
            return {k: self.metadata.data[k] for k in key}

        elif (
            isinstance(key, Number)
            or (isinstance(key, list) and isinstance(key[0], Number))
            or is_array_like(key)
        ):
            return {k: self.metadata.data[k][key] for k in self.metadata.columns}
