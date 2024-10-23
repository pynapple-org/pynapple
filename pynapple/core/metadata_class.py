from numbers import Number

import numpy as np
import pandas as pd


class _MetadataBase:
    """
    An object containing metadata for TsGroup, IntervalSet, or TsdFrame objects.

    """

    def __init__(self, *args, **kwargs):
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
        self.set_info(*args, **kwargs)

    def __dir__(self):
        """
        Adds metadata columns to the list of attributes.
        """
        return sorted(list(super().__dir__()) + self.metadata_columns)

    def __setattr__(self, name, value):
        """
        Add metadata as an attribute assignment
        """
        # self._initialized must be defined in the class that inherits _MetadataBase
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
        # technically trying to construct a dictionary will also enforce that the key is a string
        if not isinstance(key, str):
            raise TypeError("Metadata keys must be strings!")
        self.set_info(**{key: value})

    def __getitem__(self, key):
        return self.get_info(key)

    @property
    def metadata_columns(self):
        """
        Returns list of metadata columns
        """
        return list(self._metadata.columns)

    def _raise_invalid_metadata_column_name(self, name):
        if not isinstance(name, str):
            raise TypeError(
                f"Invalid metadata type {type(name)}. Metadata names must be strings!"
            )
        if hasattr(self, name) and (name not in self.metadata_columns):
            # existing non-metadata attribute
            raise ValueError(
                f"Invalid metadata name '{name}'. Metadata name must differ from "
                f"{type(self).__dict__.keys()} attribute names!"
            )
        if hasattr(self, "columns") and name in self.columns:
            # existing column (since TsdFrame columns are not attributes)
            raise ValueError(
                f"Invalid metadata name '{name}'. Metadata name must differ from "
                f"{self.columns} column names!"
            )
        if name[0].isalpha() is False:
            # starts with a number
            raise ValueError(
                f"Invalid metadata name '{name}'. Metadata name cannot start with a number"
            )
        if name.replace("_", "").isalnum() is False:
            # contains invalid characters
            raise ValueError(
                f"Invalid metadata name '{name}'. Metadata name cannot contain special characters except for underscores"
            )

    def _check_metadata_column_names(self, *args, **kwargs):
        """
        Check that metadata column names don't conflict with existing attributes, don't start with a number, and don't contain invalid characters.
        """

        for arg in args:
            if isinstance(arg, pd.DataFrame):
                [self._raise_invalid_metadata_column_name(col) for col in arg.columns]

        for k in kwargs:
            self._raise_invalid_metadata_column_name(k)

    def set_info(self, *args, **kwargs):
        """
        Add metadata information about the object.
        Metadata are saved as a DataFrame.

        Parameters
        ----------
        *args
            pandas.Dataframe or list of pandas.DataFrame
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
        self._check_metadata_column_names(*args, **kwargs)
        not_set = []
        if len(args):
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    if pd.Index.equals(self._metadata.index, arg.index):
                        self._metadata[arg.columns] = arg
                    else:
                        raise RuntimeError("Metadata index does not match")
                elif isinstance(arg, (pd.Series, np.ndarray, list)):
                    raise RuntimeError("Argument should be passed as keyword argument.")
                else:
                    not_set.append(arg)
        if len(kwargs):
            for k, v in kwargs.items():

                if isinstance(v, pd.Series):
                    if pd.Index.equals(self._metadata.index, v.index):
                        self._metadata[k] = v
                    else:
                        raise RuntimeError(
                            "Metadata index does not match for argument {}".format(k)
                        )

                elif len(self) == 1:
                    # if only one interval, allow metadata to be any type
                    self._metadata[k] = [v]

                elif isinstance(v, (np.ndarray, list, tuple)):
                    if len(self._metadata.index) == len(v):
                        self._metadata[k] = np.asarray(v)
                    else:
                        raise RuntimeError(
                            f"input array length {len(v)} does not match metadata length {len(self._metadata.index)}."
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
        Returns the metainfo located in one column.
        The key for the column frequency is "rate".

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
