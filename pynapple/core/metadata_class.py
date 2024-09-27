from pandas import DataFrame
import pandas as pd
import numpy as np
from functools import wraps

def wrap_metadata(cls):
    """
    Decorator to add methods of Metadata to a class

    """
    @wraps(getattr(Metadata,'set_info')) # wraps original method for docstring
    def getter(self,*args,**kwargs):    
        return getattr(getattr(self,'_metadata'),'set_info')(self,*args,**kwargs)
    setattr(cls, 'set_info', getter)

    @wraps(getattr(Metadata,'get_info')) # wraps original method for docstring
    def getter(self,*args,**kwargs):    
        return getattr(getattr(self,'_metadata'),'get_info')(*args,**kwargs)
    setattr(cls, 'get_info', getter)

    @wraps(getattr(Metadata,'metadata_columns')) # wraps original method for docstring
    def getter(self):    
        return getattr(getattr(self,'_metadata'),'metadata_columns')
    setattr(cls, 'metadata_columns', property(getter))

    return cls

class Metadata(DataFrame):
    """
    A pandas DataFrame-like object containing metadata for TsGroup or IntervalSet
    
    """
    def __init__(self, ob, *args, **kwargs):
        """
        Metadata initializer
        
        Parameters
        ----------
        object : TsGroup or IntervalSet
            The object to which the metadata is attached
        kwargs : dict
            Dictionary containing metadata information

        """

        super().__init__(index=ob.index) # initialize to empty frame with index matching input object
        self.set_info(ob,*args, **kwargs)

    @property
    def metadata_columns(self):
        """
        Returns list of metadata columns
        """
        return list(self.columns)

    @staticmethod
    def _check_metadata_column_names(ob, *args, **kwargs):
        invalid_cols = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                invalid_cols += [col for col in arg.columns if  hasattr(ob, col)]

        for k, v in kwargs.items():
            if isinstance(v, (list, np.ndarray, pd.Series)) and hasattr(ob, k):
                print(hasattr(ob, k))
                print(getattr(ob, k))
                print(k)
                invalid_cols += [k]

        if invalid_cols:
            raise ValueError(
                f"Invalid metadata name(s) {invalid_cols}. Metadata name must differ from "
                f"{type(ob).__dict__.keys()} attribute names!"
            )   

    def set_info(self, ob, *args, **kwargs):
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
        self._check_metadata_column_names(ob, *args,**kwargs)
        not_set = []
        if len(args):
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    if pd.Index.equals(self.index, arg.index):
                        super().__setitem__(arg.columns,arg.values)
                    else:
                        raise RuntimeError("Index are not equals")
                elif isinstance(arg, (pd.Series, np.ndarray, list)):
                    raise RuntimeError("Argument should be passed as keyword argument.")
                else:
                    not_set.append(arg)
        if len(kwargs):
            for k, v in kwargs.items():
                if isinstance(v, pd.Series):
                    if pd.Index.equals(self.index, v.index):
                        super().__setitem__(k, v)
                    else:
                        raise RuntimeError(
                            "Index are not equals for argument {}".format(k)
                        )
                elif isinstance(v, (np.ndarray, list, tuple)):
                    if len(self) == len(v):
                        # self._metadata[k] = np.asarray(v)
                        super().__setitem__(k, np.asarray(v))

                    else:
                        raise RuntimeError("Array is not the same length.")
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
        if key in ["freq", "frequency"]:
            key = "rate"
        return self[key]
