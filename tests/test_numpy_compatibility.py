# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-09-18 18:11:24
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-19 18:43:51



import pynapple as nap
import numpy as np
import pytest
from numpy import ufunc as _ufunc
import numpy.core.umath as _umath


ufuncs = {k:obj for k, obj in _umath.__dict__.items() if isinstance(obj, _ufunc)}

tsd = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 3), time_units="s")

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
