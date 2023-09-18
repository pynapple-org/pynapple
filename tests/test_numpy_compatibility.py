# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2023-09-18 18:11:24
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2023-09-18 19:05:50



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
        nap.TsdFrame(t=np.arange(100), d=np.random.rand(100, 5), time_units="s"),
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 3), time_units="s")
    ],
)
class Test_Time_Series_1:
    
    def test_ufuncs(self, tsd):

        for f in ['add', 'subtract', 'multiply', 'divide', 'floor_divide', 'power', 'remainder', 'mod', 'fmod']:
            out = ufuncs[f](tsd, 1)
            assert isinstance(out, tsd.__class__)
            np.testing.assert_array_almost_equal(out.index, tsd.index)
            np.testing.assert_array_almost_equal(out.values, ufuncs[f](tsd.values, 1))

            with pytest.raises(TypeError):
                ufuncs[f](tsd, tsd)

        for f in ['negative', 'positive']:
            print(f)


        for f in ['matmul', 'logaddexp', 'logaddexp2', 'true_divide']:
            print(f)

    def test_operators(self, tsd):
        v = tsd.values

        a = tsd + 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v + 0.5))

        a = tsd - 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v - 0.5))

        a = tsd * 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v * 0.5))

        a = tsd / 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v / 0.5))

        a = tsd // 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v // 0.5))        

        a = tsd % 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v % 0.5))

        a = tsd ** 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        np.testing.assert_array_almost_equal(a.values, v**0.5)        

        a = tsd > 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v > 0.5))

        a = tsd >= 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v >= 0.5))

        a = tsd < 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v < 0.5))

        a = tsd <= 0.5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v <= 0.5))        

        tsd = nap.Tsd(t=np.arange(10), d=np.arange(10))
        v = tsd.values
        a = tsd == 5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v == 5))

        a = tsd != 5
        assert isinstance(a, nap.Tsd)
        np.testing.assert_array_almost_equal(tsd.index, a.index)
        assert np.all(a.values == (v != 5)) 
