import warnings
from numbers import Number

import numpy as np
import numpy.core.umath as _umath
import pytest
from numpy import ufunc as _ufunc

import pynapple as nap

ufuncs = {k: obj for k, obj in _umath.__dict__.items() if isinstance(obj, _ufunc)}

tsd = nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 3), time_units="s")

# tsd = nap.TsdFrame(t=np.arange(100), d=np.random.randn(100, 6))

tsd.d[tsd.values > 0.9] = np.nan


@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(100), d=np.random.rand(100), time_units="s"),
        nap.TsdFrame(
            t=np.arange(100),
            d=np.random.rand(100, 5),
            time_units="s",
            columns=["a", "b", "c", "d", "e"],
        ),
        nap.TsdTensor(t=np.arange(100), d=np.random.rand(100, 5, 3), time_units="s"),
    ],
)
class Test_Time_Series_1:

    def test_ufuncs(self, tsd):
        a = []
        for ufunc in ufuncs.values():
            print(ufunc)
            # Bit-twiddling functions
            if ufunc.__name__ in [
                "bitwise_and",
                "bitwise_or",
                "bitwise_xor",
                "invert",
                "left_shift",
                "right_shift",
                "isnat",
                "gcd",
                "lcm",
                "ldexp",
                "arccosh",
            ]:
                a.append(ufunc.__name__)
                pass

            elif ufunc.__name__ in ["matmul", "dot"]:
                break
                if tsd.ndim > 1:
                    x = np.random.rand(*tsd.shape[1:]).T
                    out = ufunc(tsd, x)
                    assert isinstance(out, tsd.__class__)
                    np.testing.assert_array_almost_equal(out.index, tsd.index)
                    np.testing.assert_array_almost_equal(
                        out.values, ufunc(tsd.values, x)
                    )

                with pytest.raises(TypeError):
                    ufunc(tsd, tsd)

                a.append(ufunc.__name__)

            elif ufunc.__name__ in ["logaddexp", "logaddexp2", "true_divide"]:
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
                np.testing.assert_array_almost_equal(
                    out.values, ufunc(tsd.values, 0.2, 0.6)
                )

                a.append(ufunc.__name__)

    def test_funcs(self, tsd):
        a = np.array(tsd)
        np.testing.assert_array_almost_equal(a, tsd.values)

        tsd2 = tsd.copy()
        a = np.copy(tsd.values)
        tsd2[0] = 1.0
        np.testing.assert_array_almost_equal(tsd.values, a)

        assert tsd.shape == tsd.values.shape

        if tsd.ndim > 1:
            a = np.reshape(tsd, (np.prod(tsd.shape), 1))
            assert isinstance(a, np.ndarray)

        if tsd.nap_class == "TsdTensor":
            a = np.reshape(tsd, (tsd.shape[0], np.prod(tsd.shape[1:])))
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.index, tsd.index)
            np.testing.assert_array_almost_equal(
                a.values, np.reshape(tsd.values, (tsd.shape[0], np.prod(tsd.shape[1:])))
            )

        a = np.ravel(tsd.values)
        np.testing.assert_array_almost_equal(a, np.ravel(tsd))

        a = np.transpose(tsd.values)
        np.testing.assert_array_almost_equal(a, np.transpose(tsd))

        a = np.expand_dims(tsd, axis=-1)
        assert a.ndim == tsd.ndim + 1
        if a.ndim == 2:
            assert isinstance(a, nap.TsdFrame)
        else:
            assert isinstance(a, nap.TsdTensor)

        a = np.expand_dims(tsd, axis=0)
        assert isinstance(a, np.ndarray)

        if tsd.nap_class == "TsdFrame":
            a = np.column_stack((tsd[:, 0], tsd[:, 1]))
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.values, tsd.values[:, 0:2])

        a = np.isnan(tsd)
        assert isinstance(a, tsd.__class__)
        np.testing.assert_array_equal(a.values, np.isnan(tsd.values))

    def test_attributes(self, tsd):
        assert tsd.min() == tsd.values.min()

        with pytest.raises(AttributeError) as e_info:
            tsd.blabla()

        assert (
            str(e_info.value) == "Time series object does not have the attribute blabla"
        )

    def test_split(self, tsd):
        a = np.split(tsd, 4)
        b = np.split(tsd.values, 4)
        c = np.split(tsd.index, 4)
        for i in range(4):
            np.testing.assert_array_almost_equal(a[i].values, b[i])
            np.testing.assert_array_almost_equal(a[i].index, c[i])

        if tsd.ndim > 1:
            a = np.split(tsd, 1, 1)
            assert isinstance(a, list)

        a = np.array_split(tsd, 4)
        b = np.array_split(tsd.values, 4)
        c = np.array_split(tsd.index, 4)
        for i in range(4):
            np.testing.assert_array_almost_equal(a[i].values, b[i])
            np.testing.assert_array_almost_equal(a[i].index, c[i])
        if tsd.ndim > 1:
            a = np.array_split(tsd, 1, 1)
            assert isinstance(a, list)

        if tsd.ndim > 1:
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

        a = tsd**0.5
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
            a = tsd[:, 0:2, 2:4]
            assert isinstance(a, nap.TsdTensor)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:, 0:2, 2:4], a.values)

            a = tsd[:, 0]
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:, 0], a.values)

            a = tsd[:, 0, 0]
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:, 0, 0], a.values)

        if tsd.nap_class == "TsdFrame":
            a = tsd[:, 0]
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:, 0], a.values)

            a = tsd.loc["a"]
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:, 0], a.values)

            a = tsd.loc[["a", "c"]]
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(tsd.index, a.index)
            np.testing.assert_array_almost_equal(tsd.values[:, [0, 2]], a.values)

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

            if tsd.ndim > 1:
                a = func(tsd, 1)
                if a.ndim == 1:
                    assert isinstance(a, nap.Tsd)
                if a.ndim == 2:
                    assert isinstance(a, nap.TsdFrame)
                np.testing.assert_array_equal(a.values, func(tsd.values, 1))
                np.testing.assert_array_almost_equal(a.index, tsd.index)

        for func in [np.argwhere]:
            a = func(tsd)
            np.testing.assert_array_equal(a, func(tsd.values))

        a = np.where(tsd > 0.5)
        assert isinstance(a, tuple)

    def test_statistics(self, tsd):

        for func in [np.percentile, np.nanpercentile, np.quantile, np.nanquantile]:
            a = np.percentile(tsd, 50)
            assert a == np.percentile(tsd.values, 50)

            if tsd.ndim > 1:
                a = np.percentile(tsd, 50, 1)
                assert isinstance(a, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))

        for func in [
            np.median,
            np.average,
            np.mean,
            np.std,
            np.var,
            np.nanmedian,
            np.nanmean,
            np.nanstd,
            np.nanvar,
        ]:
            a = np.percentile(tsd, 50)
            assert a == np.percentile(tsd.values, 50)

            if tsd.ndim > 1:
                a = np.percentile(tsd, 50, 1)
                assert isinstance(a, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))

        if tsd.ndim == 2:
            a = np.corrcoef(tsd)
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.index, tsd.index)

            a = np.cov(tsd)
            assert isinstance(a, nap.TsdFrame)
            np.testing.assert_array_almost_equal(a.index, tsd.index)

            a = np.correlate(tsd[:, 0], tsd[:, 1])
            b = np.correlate(tsd[:, 0].values, tsd[:, 1].values)
            assert isinstance(a, np.ndarray)
            np.testing.assert_array_almost_equal(a, b)

            a = np.correlate(tsd[:, 0], tsd[:, 1], "same")
            b = np.correlate(tsd[:, 0].values, tsd[:, 1].values, "same")
            assert isinstance(a, nap.Tsd)
            np.testing.assert_array_almost_equal(a, b)

            a = np.correlate(tsd[:, 0], tsd[:, 1], "full")
            b = np.correlate(tsd[:, 0].values, tsd[:, 1].values, "full")
            assert isinstance(a, np.ndarray)
            np.testing.assert_array_almost_equal(a, b)

        a, bins = np.histogram(tsd)
        assert isinstance(a, np.ndarray)

        if tsd.ndim == 1:
            a = np.digitize(tsd, np.linspace(0, 1, 10))
            assert isinstance(a, nap.Tsd)

    def test_concatenate(self, tsd):

        with pytest.raises(
            RuntimeError,
            match=r"The order of the time series indexes should be strictly increasing and non overlapping.",
        ):
            np.concatenate((tsd, tsd), 0)

        tsd2 = tsd.__class__(t=tsd.index + 150, d=tsd.values)

        a = np.concatenate((tsd, tsd2))
        assert isinstance(a, tsd.__class__)
        assert len(a) == len(tsd) + len(tsd2)
        np.testing.assert_array_almost_equal(
            a.values, np.concatenate((tsd.values, tsd2.values))
        )
        time_support = nap.IntervalSet(start=[0, 150], end=[99, 249])
        np.testing.assert_array_almost_equal(time_support.values, a.time_support.values)

        b = np.concatenate((tsd, tsd2), axis=0)
        np.testing.assert_array_almost_equal(a.values, b.values)
        np.testing.assert_array_almost_equal(a.index.values, b.index.values)

        c = np.concatenate((tsd, tsd2), 0)
        np.testing.assert_array_almost_equal(a.values, c.values)
        np.testing.assert_array_almost_equal(a.index.values, c.index.values)

        d = np.concatenate((tsd, tsd2))
        np.testing.assert_array_almost_equal(a.values, d.values)
        np.testing.assert_array_almost_equal(a.index.values, d.index.values)

        e = np.concatenate((tsd, tsd.values), 0)
        assert isinstance(e, np.ndarray)

        if tsd.ndim >= 2:
            out = np.concatenate((tsd, tsd), 1)
            assert isinstance(out, tsd.__class__)
            np.testing.assert_array_almost_equal(
                out.values, np.concatenate((tsd.values, tsd.values), 1)
            )

            out = np.concatenate((tsd.values, tsd), 1)
            assert isinstance(out, tsd.__class__)
            np.testing.assert_array_almost_equal(
                out.values, np.concatenate((tsd.values, tsd.values), 1)
            )
            np.testing.assert_array_almost_equal(tsd.index.values, out.index.values)

            msg = "Time indexes and time supports are not all equals up to pynapple precision. Returning numpy array!"
            with pytest.warns(match=msg):
                out = np.concatenate((tsd, tsd2), 1)
            assert isinstance(out, np.ndarray)

            iset = nap.IntervalSet(start=0, end=500)
            msg = "Time indexes are not all equals up to pynapple precision. Returning numpy array!"
            with pytest.warns(match=msg):
                out = np.concatenate((tsd.restrict(iset), tsd2.restrict(iset)), 1)

            msg = "Time supports are not all equals up to pynapple precision. Returning numpy array!"
            with pytest.warns(match=msg):
                out = np.concatenate((tsd, tsd.restrict(iset)), 1)

        if tsd.ndim == 3:
            out = np.concatenate((tsd, tsd), 2)
            assert isinstance(out, tsd.__class__)
            np.testing.assert_array_almost_equal(
                out.values, np.concatenate((tsd.values, tsd.values), 2)
            )
            out = np.concatenate((tsd.values, tsd), 2)
            assert isinstance(out, tsd.__class__)
            np.testing.assert_array_almost_equal(
                out.values, np.concatenate((tsd.values, tsd.values), 2)
            )
            np.testing.assert_array_almost_equal(tsd.index.values, out.index.values)

    def test_fft(self, tsd):
        with pytest.raises(TypeError):
            np.fft.fft(tsd)


@pytest.mark.parametrize(
    "func, kwargs",
    [
        ("concatenate", {}),
        ("concatenate", {"axis": 0}),
        ("concatenate", {"axis": 1}),
        ("concatenate", {"axis": 2}),
        ("stack", {}),
        ("stack", {"axis": 0}),
        ("stack", {"axis": 1}),
        ("stack", {"axis": 2}),
        ("stack", {"axis": -1}),
        ("vstack", {}),
        ("hstack", {}),
        ("dstack", {}),
        ("column_stack", {}),
    ],
)
@pytest.mark.parametrize(
    "tsds",
    [
        (
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            nap.Tsd(t=np.arange(10) + 15, d=np.random.rand(10)),
        ),
        (
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
        ),
        (
            nap.Tsd(t=np.arange(10) + 15, d=np.random.rand(10)),
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
        ),
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            nap.TsdFrame(t=np.arange(10) + 15, d=np.random.rand(10, 5)),
        ),
        (
            nap.TsdFrame(t=np.arange(10) + 15, d=np.random.rand(10, 5)),
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
        ),
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
        ),
        (
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 5, 2)),
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 5, 2)),
        ),
        (
            nap.TsdTensor(t=np.arange(10) + 15, d=np.random.rand(10, 5, 2)),
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 5, 2)),
        ),
    ],
)
def test_concatenate_all(func, kwargs, tsds):
    tsd1, tsd2 = tsds
    try:
        b = getattr(np, func)((tsd1.values, tsd2.values), **kwargs)
    except (ValueError, RuntimeError):
        pytest.skip("Skipping invalid axis operation")

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            a = getattr(np, func)((tsd1, tsd2), **kwargs)
    except (ValueError, RuntimeError) as e:
        error_msg = str(e)
        assert (
            error_msg
            == "The order of the time series indexes should be strictly increasing and non overlapping."
        )
        return

    if a.ndim == tsd1.ndim:
        if a.shape[0] == tsd1.shape[0] + tsd2.shape[0]:  # Stacking vertically
            assert isinstance(a, tsd1.__class__)
            np.testing.assert_array_almost_equal(
                a.index, np.concatenate((tsd1.index, tsd2.index))
            )
            np.testing.assert_array_almost_equal(a.values, b)
            np.testing.assert_array_equal(
                np.vstack((tsd1.time_support.values, tsd2.time_support.values)),
                a.time_support.values,
            )
        else:
            # Check if operation was allowed
            if isinstance(a, tsd1.__class__):
                np.testing.assert_array_almost_equal(tsd1.index, tsd2.index)
                np.testing.assert_array_almost_equal(a.values, b)
                np.testing.assert_array_equal(a.index, tsd1.index)
                if hasattr(tsd1, "columns") and hasattr(tsd2, "columns"):
                    np.testing.assert_array_equal(
                        a.columns, np.concatenate((tsd1.columns, tsd2.columns), axis=0)
                    )
            else:
                assert isinstance(a, np.ndarray)
                np.testing.assert_array_almost_equal(a, b)
    else:
        # Check if operation was allowed
        if hasattr(a, "nap_class"):
            np.testing.assert_array_almost_equal(tsd1.index, tsd2.index)
            np.testing.assert_array_almost_equal(a.values, b)
            np.testing.assert_array_equal(a.index, tsd1.index)
            np.testing.assert_array_equal(
                tsd1.time_support.values, tsd2.time_support.values
            )
        else:
            assert isinstance(a, np.ndarray)
            np.testing.assert_array_almost_equal(a, b)

    if len(record) > 0:
        warning_msg = str(record[0].message)
        assert warning_msg in [
            "Time indexes and time supports are not all equals up to pynapple precision. Returning numpy array!",
            "Time indexes are not all equals up to pynapple precision. Returning numpy array!",
            "Time supports are not all equals up to pynapple precision. Returning numpy array!",
        ]
        assert isinstance(a, np.ndarray)


@pytest.mark.parametrize(
    "tsd",
    [
        nap.TsdFrame(
            t=np.arange(10),
            d=np.random.rand(10, 10),
        ),
        nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 10, 10), time_units="s"),
    ],
)
@pytest.mark.parametrize(
    "func, kwargs",
    [
        ("sum", {}),
        ("sum", {"axis": 0}),
        ("sum", {"axis": 1}),
        ("sum", {"axis": -1}),
        ("sum", {"axis": (0, 1)}),
        ("sum", {"axis": None}),
    ],
)
def test_square_arrays(tsd, func, kwargs):
    a = getattr(np, func)(tsd, **kwargs)
    b = getattr(np, func)(tsd.values, **kwargs)

    if "axis" in kwargs:
        axis = kwargs["axis"]
    else:
        axis = None

    if axis is None or np.isscalar(b):
        assert np.isscalar(a)
        assert a == b
    else:
        if (axis == 0) or (isinstance(axis, tuple) and 0 in axis):
            assert isinstance(a, (np.ndarray, Number))
            np.testing.assert_array_almost_equal(a, b)
        else:
            assert not isinstance(a, np.ndarray)
            np.testing.assert_array_almost_equal(a.index, tsd.index)
            np.testing.assert_array_almost_equal(a.values, b)


@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
        nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 10)),
        nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 10, 10), time_units="s"),
    ],
)
@pytest.mark.parametrize(
    "func, kwargs, expected_type",
    [
        ("transpose", {}, (nap.Tsd, np.ndarray)),
        ("transpose", {"axes": (2, 0, 1)}, np.ndarray),
        ("transpose", {"axes": (0, 2, 1)}, nap.TsdTensor),
        ("moveaxis", {"source": 0, "destination": 1}, np.ndarray),
        ("moveaxis", {"source": 1, "destination": 0}, np.ndarray),
        ("moveaxis", {"source": 2, "destination": 1}, nap.TsdTensor),
        ("swapaxes", {"axis1": 0, "axis2": 1}, np.ndarray),
        ("swapaxes", {"axis1": 1, "axis2": 2}, nap.TsdTensor),
        ("swapaxes", {"axis1": 2, "axis2": 0}, np.ndarray),
        (
            "rollaxis",
            {"axis": 0, "start": 1},
            {"Tsd": np.ndarray, "TsdFrame": np.ndarray, "TsdTensor": np.ndarray},
        ),
        ("rollaxis", {"axis": 1, "start": 0}, np.ndarray),
        ("rollaxis", {"axis": 1, "start": 2}, (nap.TsdTensor, nap.TsdFrame)),
        ("flipud", {}, np.ndarray),
        ("fliplr", {}, (nap.TsdFrame, nap.TsdTensor)),
        ("flip", {"axis": 0}, np.ndarray),
        ("flip", {"axis": None}, np.ndarray),
        ("flip", {"axis": 1}, (nap.TsdFrame, nap.TsdTensor)),
        ("flip", {"axis": 2}, nap.TsdTensor),
        ("rot90", {}, np.ndarray),
        ("rot90", {"k": 2}, np.ndarray),
        ("roll", {"shift": 2, "axis": 0}, np.ndarray),
        ("roll", {"shift": -2, "axis": 1}, (nap.TsdFrame, nap.TsdTensor)),
        ("roll", {"shift": 1, "axis": 2}, nap.TsdTensor),
    ],
)
def test_axis_moving(tsd, func, kwargs, expected_type):
    try:
        b = getattr(np, func)(tsd.values, **kwargs)
    except (ValueError, RuntimeError):
        pytest.skip("Skipping invalid axis operation")

    a = getattr(np, func)(tsd, **kwargs)

    if isinstance(expected_type, dict):
        assert isinstance(a, expected_type[tsd.nap_class])
    else:
        assert isinstance(a, expected_type)

    if not isinstance(a, np.ndarray):
        np.testing.assert_array_almost_equal(a.index, tsd.index)

    if hasattr(a, "values"):
        np.testing.assert_array_almost_equal(a.values, b)
    else:
        np.testing.assert_array_almost_equal(a, b)


@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
        nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 10)),
        nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 10, 10)),
        nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 10, 1)),
    ],
)
@pytest.mark.parametrize(
    "func, kwargs, expected_type",
    [
        ("expand_dims", {"axis": 0}, np.ndarray),
        ("expand_dims", {"axis": 1}, (nap.TsdFrame, nap.TsdTensor)),
        ("expand_dims", {"axis": -1}, (nap.TsdFrame, nap.TsdTensor)),
        (
            "expand_dims",
            {"axis": -2},
            {"Tsd": np.ndarray, "TsdFrame": nap.TsdTensor, "TsdTensor": nap.TsdTensor},
        ),
        ("squeeze", {}, (nap.Tsd, nap.TsdFrame, nap.TsdTensor)),
        (
            "ravel",
            {},
            {"Tsd": nap.Tsd, "TsdFrame": np.ndarray, "TsdTensor": np.ndarray},
        ),
        (
            "ravel",
            {"order": "F"},
            {"Tsd": nap.Tsd, "TsdFrame": np.ndarray, "TsdTensor": np.ndarray},
        ),
        (
            "tile",
            {"reps": 2},
            {"Tsd": np.ndarray, "TsdFrame": nap.TsdFrame, "TsdTensor": nap.TsdTensor},
        ),
        (
            "tile",
            {"reps": (2, 1)},
            {"Tsd": np.ndarray, "TsdFrame": np.ndarray, "TsdTensor": nap.TsdTensor},
        ),
        (
            "tile",
            {"reps": (1, 2)},
            {"Tsd": np.ndarray, "TsdFrame": nap.TsdFrame, "TsdTensor": nap.TsdTensor},
        ),
    ],
)
def test_shape_change(tsd, func, kwargs, expected_type):
    try:
        b = getattr(np, func)(tsd.values, **kwargs)
    except (ValueError, RuntimeError):
        pytest.skip("Skipping invalid axis operation")

    a = getattr(np, func)(tsd, **kwargs)

    if isinstance(expected_type, dict):
        assert isinstance(a, expected_type[tsd.nap_class])
    else:
        assert isinstance(a, expected_type)

    if not isinstance(a, np.ndarray):
        np.testing.assert_array_almost_equal(a.index, tsd.index)

    if hasattr(a, "values"):
        np.testing.assert_array_almost_equal(a.values, b)
    else:
        np.testing.assert_array_almost_equal(a, b)


@pytest.mark.parametrize(
    "tsd, slicing, expected_type",
    [
        (
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            lambda x: x[None, :],
            np.ndarray,
        ),
        (
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            lambda x: x[:, None],
            nap.TsdFrame,
        ),
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 10)),
            lambda x: x[:, None],
            nap.TsdTensor,
        ),
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 10)),
            lambda x: x[:, :, None],
            nap.TsdTensor,
        ),
        (
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 10, 10)),
            lambda x: x[:, None],
            nap.TsdTensor,
        ),
        (
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 10, 1)),
            lambda x: x[None, :],
            np.ndarray,
        ),
    ],
)
def test_shape_change_2(tsd, slicing, expected_type):
    a = slicing(tsd)
    assert isinstance(a, expected_type)
    if hasattr(a, "index"):
        np.testing.assert_array_almost_equal(a.index, tsd.index)
    if hasattr(a, "values"):
        np.testing.assert_array_almost_equal(a.values, slicing(tsd.values))


@pytest.mark.parametrize(
    "a, b, expected_type",
    [
        (
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            float,
        ),
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            np.random.randn(5, 3),
            nap.TsdFrame,
        ),
        (
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 4, 2)),
            np.random.rand(2, 3),
            nap.TsdTensor,
        ),
        (
            np.random.rand(5, 10),
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            np.ndarray,
        ),
    ],
)
def test_dot_product(a, b, expected_type):
    out = np.dot(a, b)

    assert isinstance(out, expected_type)

    if hasattr(a, "values") and hasattr(b, "values"):
        out2 = np.dot(a.values, b.values)
    elif hasattr(a, "values"):
        out2 = np.dot(a.values, b)
    elif hasattr(b, "values"):
        out2 = np.dot(a, b.values)
    else:
        out2 = np.dot(a, b)

    if hasattr(out, "values"):
        np.testing.assert_array_almost_equal(out.values, out2)
    else:
        if isinstance(out2, float):
            assert out == out2
        else:
            np.testing.assert_array_almost_equal(out, out2)


@pytest.mark.parametrize(
    "a, b, expected_type",
    [
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            np.random.randn(5, 3),
            nap.TsdFrame,
        ),
        (
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 4, 2)),
            np.random.rand(2, 3),
            nap.TsdTensor,
        ),
        (
            np.random.rand(5, 10),
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            np.ndarray,
        ),
    ],
)
def test_matmul_product(a, b, expected_type):
    out = np.matmul(a, b)

    assert isinstance(out, expected_type)

    if hasattr(a, "values") and hasattr(b, "values"):
        out2 = np.dot(a.values, b.values)
    elif hasattr(a, "values"):
        out2 = np.dot(a.values, b)
    elif hasattr(b, "values"):
        out2 = np.dot(a, b.values)
    else:
        out2 = np.dot(a, b)

    if hasattr(out, "values"):
        np.testing.assert_array_almost_equal(out.values, out2)
    else:
        if isinstance(out2, float):
            assert out == out2
        else:
            np.testing.assert_array_almost_equal(out, out2)


@pytest.mark.parametrize(
    "a, b, subscripts, expected_type",
    [
        (
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
            "i,i->",
            float,
        ),
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            np.random.randn(5, 3),
            "ij,jk->ik",
            nap.TsdFrame,
        ),
        (
            nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 4, 2)),
            np.random.rand(2, 3),
            "ijk,kl->ijl",
            nap.TsdTensor,
        ),
        (
            np.random.rand(5, 10),
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            "ij,jk->ik",
            np.ndarray,
        ),
        (
            nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
            np.random.randn(5, 10),
            "ij,ji->i",
            nap.Tsd,
        ),
    ],
)
def test_einsum(a, b, subscripts, expected_type):
    out = np.einsum(subscripts, a, b)

    assert isinstance(out, expected_type)

    if hasattr(a, "values") and hasattr(b, "values"):
        out2 = np.einsum(subscripts, a.values, b.values)
    elif hasattr(a, "values"):
        out2 = np.einsum(subscripts, a.values, b)
    elif hasattr(b, "values"):
        out2 = np.einsum(subscripts, a, b.values)
    else:
        out2 = np.einsum(subscripts, a, b)

    if hasattr(out, "values"):
        np.testing.assert_array_almost_equal(out.values, out2)
    else:
        if isinstance(out2, float):
            assert out == out2
        else:
            np.testing.assert_array_almost_equal(out, out2)

@pytest.mark.parametrize(
    "tsd",
    [
        nap.Tsd(t=np.arange(10), d=np.random.rand(10)),
        nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 5)),
        nap.TsdFrame(t=np.arange(10), d=np.random.rand(10, 10)),
        nap.TsdTensor(t=np.arange(10), d=np.random.rand(10, 5, 2)),
    ],
)
@pytest.mark.parametrize(
    "func, kwargs, expected_type",
    [
        ("cumsum", {"axis":0}, "Tsd"),
        ("cumsum", {"axis":1}, "Tsd"),
        ("cumprod", {"axis":0}, "Tsd"),
        ("cumprod", {"axis":1}, "Tsd"),
        ("nancumsum", {"axis":0}, "Tsd"),
        ("nancumsum", {"axis":1}, "Tsd"),
        ("unwrap", {}, "Tsd"),
        ("clip", {"a_min": 0.2, "a_max": 0.8}, "Tsd"),
        ("angle", {}, "Tsd"),
        ("conj", {}, "Tsd"),
        ("real", {}, "Tsd"),
        ("imag", {}, "Tsd"),
        ("round", {"decimals": 2}, "Tsd"),
        ("fix", {}, "Tsd"),
        ("isreal", {}, "Tsd"),
        ("iscomplex", {}, "Tsd"),
    ]
)
def test_same_shape(tsd, func, kwargs, expected_type):
    try:
        b = getattr(np, func)(tsd.values, **kwargs)
    except (ValueError, RuntimeError):
        pytest.skip("Skipping invalid axis operation")

    a = getattr(np, func)(tsd, **kwargs)

    assert expected_type in a.__class__.__name__

    if expected_type == "Tsd":
        np.testing.assert_array_almost_equal(a.index, tsd.index)
        np.testing.assert_array_almost_equal(a.values, b)
    else:
        np.testing.assert_array_almost_equal(a, b)