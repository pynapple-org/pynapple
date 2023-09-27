# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-09-18 18:11:24
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-27 15:03:31



import pynapple as nap
import numpy as np
import pytest
from numpy import ufunc as _ufunc
import numpy.core.umath as _umath


ufuncs = {k:obj for k, obj in _umath.__dict__.items() if isinstance(obj, _ufunc)}

tsd = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 3), time_units="s")

tsd = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 6))

@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s"),
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 5), time_units="s", columns=['a', 'b', 'c', 'd', 'e']),
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 3), time_units="s")
    ],
)
class Test_Time_Series_1:

    def test_ufuncs(self, tsd):
        a = []
        for ufunc in ufuncs.values():
            print(ufunc)
            # Bit-twiddling functions
            if ufunc.__name__ in ['bitwise_and', 'bitwise_or', 'bitwise_xor', 
                'invert', 'left_shift', 'right_shift', 'isnat', 'gcd', 'lcm', 
                'ldexp', 'arccosh']:
                a.append(ufunc.__name__)
                pass

            elif ufunc.__name__ in ['matmul', 'dot']:
                break
                if tsd.ndim > 1:
                    x = np.random.rand(*tsd.shape[1:]).T
                    out = ufunc(tsd, x)
                    assert isinstance(out, tsd.__class__)
                    np.testing.assert_array_almost_equal(out.index, tsd.index)
                    np.testing.assert_array_almost_equal(out.values, ufunc(tsd.values, x))

                with pytest.raises(TypeError):
                    ufunc(tsd, tsd)

                a.append(ufunc.__name__)

            elif ufunc.__name__ in ['logaddexp', 'logaddexp2', 'true_divide']:                
                x = np.random.rand(*tsd.shape)
                out = ufunc(tsd, x)
                assert isinstance(out, tsd.__class__)
                np.testing.assert_array_almost_equal(out.index, tsd.index)
                np.testing.assert_array_almost_equal(out.values, ufunc(tsd.values, x))

                with pytest.raises(TypeError):
                    ufunc(tsd, tsd)

                a.append(ufunc.__name__)

            elif ufunc.nin == 1 and ufunc.nout == 1:
                out = ufunc(tsd)
                assert isinstance(out, tsd.__class__)
                np.testing.assert_array_almost_equal(out.index, tsd.index)
                np.testing.assert_array_almost_equal(out.values, ufunc(tsd.values))
                a.append(ufunc.__name__)

            elif ufunc.nin == 2 and ufunc.nout == 1:
                # Testing with single number
                out = ufunc(tsd, 1)
                assert isinstance(out, tsd.__class__)
                np.testing.assert_array_almost_equal(out.index, tsd.index)
                np.testing.assert_array_almost_equal(out.values, ufunc(tsd.values, 1))

                # Testing with array            
                x = np.random.rand(*tsd.shape)
                out = ufunc(tsd, x)
                assert isinstance(out, tsd.__class__)
                np.testing.assert_array_almost_equal(out.index, tsd.index)
                np.testing.assert_array_almost_equal(out.values, ufunc(tsd.values, x))

                # Raising an error with two tsd
                with pytest.raises(TypeError):
                    ufunc(tsd, tsd)

                a.append(ufunc.__name__)

            elif ufunc.nin == 3 and ufunc.nout == 1:
                # Testing with two number
                out = ufunc(tsd, 0.2, 0.6)
                assert isinstance(out, tsd.__class__)
                np.testing.assert_array_almost_equal(out.index, tsd.index)
                np.testing.assert_array_almost_equal(out.values, ufunc(tsd.values, 0.2, 0.6))

                a.append(ufunc.__name__)

    def test_funcs(self, tsd):
        a = np.array(tsd)
        np.testing.assert_array_almost_equal(a, tsd.values)

        tsd2 = tsd.copy()
        a = np.copy(tsd.values)
        tsd2[0] = 1.0
        np.testing.assert_array_almost_equal(tsd.values, a)

        assert tsd.shape == tsd.values.shape

        if tsd.ndim>1:
            a = np.reshape(tsd, (np.prod(tsd.shape), 1))
            assert isinstance(a, np.ndarray)

        if tsd.nap_class == "TsdTensor":
            a = np.reshape(tsd, (tsd.shape[0], np.prod(tsd.shape[1:])))
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.index, tsd.index)
            np.testing.assert_array_almost_equal(a.values, np.reshape(tsd.values, (tsd.shape[0], np.prod(tsd.shape[1:]))))

        a = np.ravel(tsd)
        np.testing.assert_array_almost_equal(a, np.ravel(tsd))

        a = np.transpose(tsd)
        np.testing.assert_array_almost_equal(a, np.transpose(tsd))

        a = np.expand_dims(tsd, axis=-1)
        assert a.ndim == tsd.ndim + 1
        if a.ndim == 2:
            assert isinstance(a, nap.TsdFrame)
        else:
            assert isinstance(a, nap.TsdTensor)

        a = np.expand_dims(tsd, axis=0)
        assert isinstance(a, np.ndarray)

        with pytest.raises(TypeError):
            np.hstack((tsd, tsd2))

        with pytest.raises(TypeError):
            np.dstack((tsd, tsd2))

        if tsd.nap_class == "TsdFrame":
            a = np.column_stack((tsd[:,0],tsd[:,1]))
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.values, tsd.values[:,0:2])

        a = np.isnan(tsd)
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_equal(a.values, np.isnan(tsd.values))

    def test_split(self, tsd):
        a = np.split(tsd, 4)
        b = np.split(tsd.values, 4)
        c = np.split(tsd.index, 4)
        for i in range(4):
            np.testing.assert_array_almost_equal(a[i].values, b[i])
            np.testing.assert_array_almost_equal(a[i].index, c[i])

        if tsd.ndim>1:
            a = np.split(tsd, 1, 1)
            assert isinstance(a, list)

        a = np.array_split(tsd, 4)
        b = np.array_split(tsd.values, 4)
        c = np.array_split(tsd.index, 4)
        for i in range(4):
            np.testing.assert_array_almost_equal(a[i].values, b[i])
            np.testing.assert_array_almost_equal(a[i].index, c[i])
        if tsd.ndim>1:            
            a = np.array_split(tsd, 1, 1)
            assert isinstance(a, list)

        if tsd.ndim>1:
            a = np.vsplit(tsd, 4)
            b = np.vsplit(tsd.values, 4)
            c = np.split(tsd.index, 4)
            for i in range(4):
                np.testing.assert_array_almost_equal(a[i].values, b[i])
                np.testing.assert_array_almost_equal(a[i].index, c[i])

        if tsd.ndim == 3:
            a = np.dsplit(tsd, 1)
            b = np.dsplit(tsd.values, 1)
            c = np.split(tsd.index, 1)
            for i in range(1):
                np.testing.assert_array_almost_equal(a[i].values, b[i])
                np.testing.assert_array_almost_equal(a[i].index, c[i])

        if tsd.ndim == 2:
            a = np.hsplit(tsd, 1)
            b = np.hsplit(tsd.values, 1)
            for i in range(1):
                np.testing.assert_array_almost_equal(a[i].values, b[i])
                np.testing.assert_array_almost_equal(a[i].index, tsd.index)

    def test_operators(self, tsd):
        v = tsd.values

        a = tsd + 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v + 0.5))

        a = tsd - 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v - 0.5))

        a = tsd * 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v * 0.5))

        a = tsd / 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v / 0.5))

        a = tsd // 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v // 0.5))        

        a = tsd % 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v % 0.5))

        a = tsd ** 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        np.testing.assert_array_almost_equal(a.values, v**0.5)        

        a = tsd > 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v > 0.5))

        a = tsd >= 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v >= 0.5))

        a = tsd < 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v < 0.5))

        a = tsd <= 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v <= 0.5))        
                
        a = tsd == 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v == 0.5))

        a = tsd != 0.5
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v != 0.5)) 

    def test_slice(self, tsd):
        a = tsd[0:10]
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(tsd.index[0:10], a.index)
        np.testing.assert_array_almost_equal(tsd.values[0:10], a.values)

        if tsd.nap_class == "TsdTensor":
            a = tsd[:,0:2,2:4]
            assert isinstance(a, nap.TsdTensor)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:,0:2,2:4], a.values)

            a = tsd[:,0]
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:,0], a.values)
        
            a = tsd[:,0,0]
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:,0,0], a.values)

        if tsd.nap_class == "TsdFrame":
            a = tsd[:,0]
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:,0], a.values)

            a = tsd.loc['a']
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:,0], a.values)

            a = tsd.loc[['a', 'c']]
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:,[0,2]], a.values)

    def test_sorting(self, tsd):
        with pytest.raises(TypeError):
            np.sort(tsd)

        with pytest.raises(TypeError):
            np.lexsort(tsd)            

        with pytest.raises(TypeError):
            np.sort_complex(tsd)

        with pytest.raises(TypeError):
            np.partition(tsd)            

        with pytest.raises(TypeError):
            np.argpartition(tsd)            

    def test_searching(self, tsd):
        for func in [np.argmax, np.nanargmax, np.argmin, np.nanargmin]:

            a = func(tsd)
            assert a == func(tsd.values)

            if tsd.ndim>1:
                a = func(tsd, 1)
                if a.ndim == 1: assert isinstance(a, nap.Tsd)
                if a.ndim == 2: assert isinstance(a, nap.TsdFrame)
                np.testing.assert_array_equal(a.values, func(tsd.values, 1))
                np.testing.assert_array_almost_equal(a.index, tsd.index)
        
        for func in [np.argwhere]:
            a = func(tsd)
            np.testing.assert_array_equal(a, func(tsd.values))

        a = np.where(tsd>0.5)
        assert isinstance(a, tuple)

    def test_statistics(self, tsd):

        for func in [np.percentile, np.nanpercentile, np.quantile, np.nanquantile]:
            a = np.percentile(tsd, 50)
            assert a == np.percentile(tsd.values, 50)

            if tsd.ndim>1:
                a = np.percentile(tsd, 50, 1)
                assert isinstance(a, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))

        for func in [np.median, np.average, np.mean, np.std, np.var, np.nanmedian, np.nanmean, np.nanstd, np.nanvar]:
            a = np.percentile(tsd, 50)
            assert a == np.percentile(tsd.values, 50)

            if tsd.ndim>1:
                a = np.percentile(tsd, 50, 1)
                assert isinstance(a, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))

        if tsd.ndim == 2:
            a = np.corrcoef(tsd)
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.index, tsd.index)

            a = np.cov(tsd)
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.index, tsd.index)
        
            a = np.correlate(tsd[:,0], tsd[:,1])
            b = np.correlate(tsd[:,0].values, tsd[:,1].values)
            assert isinstance(a, np.ndarray)
            np.testing.assert_array_almost_equal(a, b)

            a = np.correlate(tsd[:,0], tsd[:,1], 'same')
            b = np.correlate(tsd[:,0].values, tsd[:,1].values, 'same')
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(a, b)

            a = np.correlate(tsd[:,0], tsd[:,1], 'full')
            b = np.correlate(tsd[:,0].values, tsd[:,1].values, 'full')
            assert isinstance(a, np.ndarray)
            np.testing.assert_array_almost_equal(a, b)

        a, bins = np.histogram(tsd)
        assert isinstance(a, np.ndarray)

        if tsd.ndim == 1:
            a = np.digitize(tsd, np.linspace(0, 1, 10))
            assert isinstance(a, nap.Tsd)

    def test_concatenate(self, tsd):
        tsd2 = tsd.__class__(t=tsd.index+150, d=tsd.values)
        a = np.concatenate((tsd, tsd2))
        assert isinstance(a, tsd.__class__)
        assert len(a) == len(tsd) + len(tsd2)
        np.testing.assert_array_almost_equal(a.values, np.concatenate((tsd.values, tsd2.values)))
        time_support = nap.IntervalSet(start=[0, 150], end=[99, 249])
        np.testing.assert_array_almost_equal(time_support.values, a.time_support.values)

        tsd3 = tsd.__class__(t=tsd.index+300, d=tsd.values)
        a = np.vstack((tsd, tsd2, tsd3))
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_almost_equal(a.values, np.concatenate((tsd.values, tsd2.values, tsd3.values)))
        time_support = nap.IntervalSet(start=[0, 150, 300], end=[99, 249, 399])
        np.testing.assert_array_almost_equal(time_support.values, a.time_support.values)

        with pytest.raises(TypeError):
            np.concatenate((tsd, tsd2), axis=1)
    
        with pytest.raises(AssertionError):
            np.concatenate((tsd, np.random.randn(10, 1)))

        if tsd.nap_class == "TsdTensor":
            with pytest.raises(AssertionError):
                np.concatenate((tsd, nap.Tsd(t=np.arange(200,300), d=np.random.randn(100))))
