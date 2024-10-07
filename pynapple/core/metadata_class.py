import numpy as np
import pandas as pd


class MetadataBase:
    """
    An object containing metadata for TsGroup or IntervalSet

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
            # metadata index is the same as the columns
            self._metadata_index = self.columns
        else:
            # what if index is not defined?
            self._metadata_index = self.index

        self._metadata = pd.DataFrame(index=self._metadata_index)
        self.set_info(*args, **kwargs)

    def __dir__(self):
        """
        Adds metadata columns to the list of attributes.
        """
        return sorted(list(super().__dir__()) + self.metadata_columns)

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
            raise ValueError("Metadata keys must be strings!")
        # replicate pandas behavior of over-writing cols
        if key in self._metadata.columns:
            old_meta = self._metadata.copy()
            self._metadata.pop(key)
            try:
                self.set_info(**{key: value})
            except Exception:
                self._metadata = old_meta
                raise
        else:
            self.set_info(**{key: value})

    def __getitem__(self, key):
        if key in self._metadata.columns:
            return self.get_info(key)
        else:
            raise KeyError(r"Key {} not in group index.".format(key))

    @property
    def metadata_columns(self):
        """
        Returns list of metadata columns
        """
        return list(self._metadata.columns)

    def _check_metadata_column_names(self, *args, **kwargs):
        invalid_cols = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                invalid_cols += [col for col in arg.columns if hasattr(self, col)]

        for k, v in kwargs.items():
            if isinstance(v, (list, np.ndarray, pd.Series)) and hasattr(self, k):
                print(hasattr(self, k))
                print(getattr(self, k))
                print(k)
                invalid_cols += [k]

        if invalid_cols:
            raise ValueError(
                f"Invalid metadata name(s) {invalid_cols}. Metadata name must differ from "
                f"{type(self).__dict__.keys()} attribute names!"
            )

    def set_info(self, *args, **kwargs):
        """
        Add metadata information about the TsGroup or IntervalSet.
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
                        self._metadata = self._metadata.join(arg)
                    else:
                        raise RuntimeError("Index are not equals")
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
                            "Index are not equals for argument {}".format(k)
                        )
                elif isinstance(v, (np.ndarray, list, tuple)):
                    if len(self._metadata.index) == len(v):
                        self._metadata[k] = np.asarray(v)
                    else:
                        raise RuntimeError("Array is not the same length.")

                # if only one interval, allow metadata to be any type
                elif len(self) == 1:
                    self._metadata[k] = v

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
        if key in [
            "freq",
            "frequency",
        ]:  # this will not be conducive for metadata of other objects
            key = "rate"
        return self._metadata[key]
