import warnings
from numbers import Number

import numpy as np
import pandas as pd


class _MetadataMixin:
    """
    An object containing metadata functionality for TsGroup, IntervalSet, or TsdFrame objects.

    """

    def __init__(self, metadata=None):
        """
        Metadata initializer

        Parameters
        ----------
        *args : list
            List of pandas.DataFrame
        **kwargs : dict
            Dictionary containing metadata information
        """
        if self.__class__.__name__ == "TsdFrame":
            # metadata index is the same as the columns for TsdFrame
            self.metadata_index = self.columns
        else:
            # metadata index is the same as the index for TsGroup and IntervalSet
            self.metadata_index = self.index

        self._metadata = pd.DataFrame(index=self.metadata_index)

        self.set_info(metadata)

    def __dir__(self):
        """
        Adds metadata columns to the list of attributes.
        """
        return sorted(list(super().__dir__()) + self.metadata_columns)

    def __setattr__(self, name, value):
        """
        Add metadata as an attribute assignment
        """
        # self._initialized must be defined in the class that inherits _MetadataMixin
        # and it must be set to True after metadata is initialized
        if self._initialized:
            self.set_info(**{name: value})
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        Allows dynamic access to metadata columns as properties.

        Parameters
        ----------
        name : str
            The name of the metadata column to access.

        Returns
        -------
        pandas.Series
            The series of values for the requested metadata column.

        Raises
        ------
        AttributeError
            If the requested attribute is not a metadata column.
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
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be strings!")
        self.set_info(**{key: value})

    def __getitem__(self, key):
        return self.get_info(key)

    @property
    def metadata(self):
        """
        Returns metadata DataFrame
        """
        # return copy so that metadata cannot modified in place
        return self._metadata.copy()

    @property
    def metadata_columns(self):
        """
        Returns list of metadata columns
        """
        return list(self._metadata.columns)

    def _raise_invalid_metadata_column_name(self, name):
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
        Throw warnings when metadata names cannot be accessed as attributes or keys.
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
        Add metadata information about the object.
        Metadata are saved as a DataFrame.

        Parameters
        ----------
        metadata : pandas.DataFrame or dict
            metadata information
        **kwargs
            Can be either pandas.Series, numpy.ndarray, list or tuple

        Raises
        ------
        RuntimeError
            Raise an error if
                no column labels are found when passing simple arguments,
                indexes are not equals for a pandas series,+
                not the same length when passing numpy array.
        TypeError
            If some of the provided metadata could not be set.

        Examples
        --------
        >>> import pynapple as nap
        >>> import numpy as np
        >>> tmp = { 0:nap.Ts(t=np.arange(0,200), time_units='s'),
        1:nap.Ts(t=np.arange(0,200,0.5), time_units='s'),
        2:nap.Ts(t=np.arange(0,300,0.25), time_units='s'),
        }
        >>> tsgroup = nap.TsGroup(tmp)

        To add metadata with a pandas.DataFrame:

        >>> import pandas as pd
        >>> structs = pd.DataFrame(index = [0,1,2], data=['pfc','pfc','ca1'], columns=['struct'])
        >>> tsgroup.set_info(structs)
        >>> tsgroup
          Index    Freq. (Hz)  struct
        -------  ------------  --------
              0             1  pfc
              1             2  pfc
              2             4  ca1

        To add metadata with a pd.Series, numpy.ndarray, list or tuple:

        >>> hd = pd.Series(index = [0,1,2], data = [0,1,1])
        >>> tsgroup.set_info(hd=hd)
        >>> tsgroup
          Index    Freq. (Hz)  struct      hd
        -------  ------------  --------  ----
              0             1  pfc          0
              1             2  pfc          1
              2             4  ca1          1

        """
        # check for duplicate names, otherwise "self.metadata_name"
        # syntax would behave unexpectedly.
        self._check_metadata_column_names(metadata, **kwargs)
        not_set = []
        if metadata is not None:
            if isinstance(metadata, pd.DataFrame):
                if pd.Index.equals(self._metadata.index, metadata.index):
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
                    if pd.Index.equals(self._metadata.index, v.index):
                        self._metadata[k] = v
                    else:
                        raise ValueError(
                            "Metadata index does not match for argument {}".format(k)
                        )

                elif isinstance(v, (np.ndarray, list, tuple)):
                    if len(self._metadata.index) == len(v):
                        self._metadata[k] = v
                    else:
                        raise ValueError(
                            f"input array length {len(v)} does not match metadata length {len(self._metadata.index)}."
                        )

                elif (hasattr(v, "__iter__") is False) and (
                    len(self._metadata.index) == 1
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
        Returns the metainfo located in one column.

        Parameters
        ----------
        key : str
            One of the metainfo columns name

        Returns
        -------
        pandas.Series
            The metainfo
        """
        if isinstance(key, str) or (
            isinstance(key, list) and all([isinstance(k, str) for k in key])
        ):
            # metadata[str] or metadata[[*str]]
            return self._metadata[key]
        elif isinstance(key, (Number, list, np.ndarray, pd.Series)) or (
            isinstance(key, tuple)
            and (
                isinstance(key[1], str)
                or (
                    isinstance(key[1], list)
                    and all([isinstance(k, str) for k in key[1]])
                )
            )
        ):
            # metadata[Number], metadata[array_like], metadata[Any, str], or metadata[Any, [*str]]
            return self._metadata.loc[key]
        elif isinstance(key, slice):
            # DataFrame's `loc` treats slices differently (inclusive of stop) than numpy
            # `iloc` exludes the stop index, like numpy
            return self._metadata.iloc[key]
        else:
            # we don't allow indexing columns with numbers
            raise IndexError(f"Unknown metadata index {key}")
