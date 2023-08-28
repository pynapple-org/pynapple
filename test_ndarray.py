# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-18 17:38:35
# @Last Modified by:   gviejo
# @Last Modified time: 2023-08-28 11:41:08
import numpy as np
import pandas as pd
from numbers import Number
from numpy.lib.mixins import NDArrayOperatorsMixin
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, SupportsIndex
import operator

class Tsd(NDArrayOperatorsMixin):

    def __init__(self, t, d):
        if isinstance(t, Number): t = np.array([t])
        if isinstance(d, Number): d = np.array([d])
        self.t = t
        self.values = np.asarray(d)
        self.ndim = d.ndim
        self.shape = d.shape

    def __repr__(self):
        # TODO repr for all dtypes
        _str_ = "\n".join([
            "{:.6f}".format(i)+"    "+"{:.6f}".format(j) for i, j in zip(self.t, self.values)
            ])        
        return _str_

    def __str__(self):
        return self.__repr__()

    def __array__(self, dtype=None):
        return self.values.astype(dtype)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        print("In __array_ufunc__")
        print("     ufunc = ", ufunc)
        print("     method = ", method)
        print("     args = ", args)
        for inp in args: print(type(inp))
        print("     kwargs = ", kwargs)

        if method == '__call__':
            scalars = []
            for a in args:
                if isinstance(a, Number):
                    scalars.append(a)
                elif isinstance(a, self.__class__):
                    scalars.append(a.values)
                else:
                    return NotImplemented

            out = ufunc(*scalars, **kwargs)
            print("output = ", out)
            return self.__class__(self.t, out)

        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, ndarray) for t in types):
            # Defer to any non-subclasses that implement __array_function__
            return NotImplemented

        # Use NumPy's private implementation without __array_function__
        # dispatching
        return func._implementation(*args, **kwargs)

    def as_series(self):
        return pd.Series(index=self.t, data=self.values)


    def __getitem__(self, key):
        """
        Performs the operation __getitem__.
        """        
        try:
            data = self.values.__getitem__(key)
            index = self.t.__getitem__(key)            
            return Tsd(t=index, d=data)
        except:
            raise IndexError

        

tsd = Tsd(t=np.sort(np.random.uniform(0, 10, 10)), d=np.random.rand(10))

