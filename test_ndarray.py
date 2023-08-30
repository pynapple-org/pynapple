# -*- coding: utf-8 -*-
# @Author: gviejo
# @Date:   2023-08-18 17:38:35
# @Last Modified by:   gviejo
# @Last Modified time: 2023-08-30 11:21:23
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
        self.index = t
        self.values = np.asarray(d)
        self.ndim = d.ndim
        self.shape = d.shape
        self.dtype = d.dtype

    def __repr__(self):
        # TODO repr for all dtypes
        upper = "Time (s)"
        _str_ = "\n".join([
            "{:.6f}".format(i)+"    "+"{:.6f}".format(j) for i, j in zip(self.index, self.values)
            ])
        bottom = "dtype: {}".format(self.dtype)
        return "\n".join((upper, _str_, bottom))

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
            new_args = []
            for a in args:
                if isinstance(a, Number):
                    new_args.append(a)
                elif isinstance(a, self.__class__):
                    new_args.append(a.values)
                else:
                    return NotImplemented

            out = ufunc(*new_args, **kwargs)
            print("output = ", out)
            return self.__class__(self.index, out)

        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        print("In __array_function__")
        print("     func = ", func)
        print("     types = ", types)
        print("     args = ", args)
        print("     kwargs = ", kwargs)

        new_args = []
        for a in args:
            if isinstance(a, Tsd):
                new_args.append(a.values)
            else:
                new_args.append(a)
        output = func._implementation(*args, **kwargs)    

        return output


    def as_series(self):
        return pd.Series(index=self.index, data=self.values)

    def as_array(self):
        return self.values

    def __getitem__(self, key):
        """
        Performs the operation __getitem__.
        """        
        try:
            data = self.values.__getitem__(key)
            index = self.index.__getitem__(key)            
            return Tsd(t=index, d=data)
        except:
            raise IndexError

    def __setitem__(self, key, value):
        """
        Performs the operation __getitem__.
        """        
        try:
            self.values.__setitem__(key, value)
        except:
            raise IndexError

    def __len__(self):
        return len(self.values)


class TsdFrame(NDArrayOperatorsMixin):

    def __init__(self, t, d):
        if isinstance(t, Number): t = np.array([t])
        if isinstance(d, Number): d = np.array([d])
        self.index = t
        self.values = np.asarray(d)
        self.ndim = d.ndim
        self.shape = d.shape
        self.dtype = d.dtype

    def __repr__(self):
        # TODO repr for all dtypes
        upper = "Time (s)"
        _str_ = []
        for i, array in zip(self.index, self.values):
            _str_.append(
                "{:.6f}".format(i)+"    "+" ".join(["{:.6f}".format(k) for k in array])
                )
            
        _str_ = "\n".join(_str_)
        bottom = "dtype: {}".format(self.dtype)
        return "\n".join((upper, _str_, bottom))

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
            new_args = []
            for a in args:
                if isinstance(a, Number):
                    new_args.append(a)
                elif isinstance(a, self.__class__):
                    new_args.append(a.values)
                else:
                    return NotImplemented

            out = ufunc(*new_args, **kwargs)
            print("output = ", out)
            return self.__class__(self.index, out)

        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        print("In __array_function__")
        print("     func = ", func)
        print("     types = ", types)
        print("     args = ", args)
        print("     kwargs = ", kwargs)

        new_args = []
        for a in args:
            if isinstance(a, Tsd):
                new_args.append(a.values)
            else:
                new_args.append(a)

        output = func._implementation(*new_args, **kwargs)

        if isinstance(output, np.ndarray):
            if output.shape[0] == self.index.shape[0]:
                if len(output.shape) == 1:
                    return Tsd(t=self.index, d=output)
                elif len(output.shape) == 2:
                    return TsdFrame(t=self.index, d=output)
                else:
                    return output
            else:
                return output
        else:
            return output


    def as_series(self):
        return pd.Series(index=self.index, data=self.values)

    def as_array(self):
        return self.values

    def __getitem__(self, key):
        """
        Performs the operation __getitem__.
        """
        print(key)
        try:
            output = self.values.__getitem__(key)
            if isinstance(key, tuple):
                index = self.index.__getitem__(key[0])
            else:
                index = self.index.__getitem__(key)

            if all(isinstance(a, np.ndarray) for a in [index, output]):
                if output.shape[0] == index.shape[0]:
                    if len(output.shape) == 1:
                        return Tsd(t=index, d=output)
                    elif len(output.shape) == 2:
                        return TsdFrame(t=index, d=output)
                    else:
                        return output
            elif isinstance(index, Number):
                return TsdFrame(t=np.array([index]), d=np.atleast_2d(output))
            else:
                return output
        except:
            raise IndexError

    def __setitem__(self, key, value):
        """
        Performs the operation __getitem__.
        """        
        try:
            self.values.__setitem__(key, value)
        except:
            raise IndexError

    def __len__(self):
        return len(self.values)   

tsd = Tsd(t=np.sort(np.random.uniform(0, 10, 10)), d=np.random.rand(10))
tsdframe = TsdFrame(t=np.sort(np.random.uniform(0, 10, 10)), d=np.random.rand(10, 3))
