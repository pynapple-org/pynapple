import h5py
import pandas as pd

import pynapple as nap
import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise
from pathlib import Path


@pytest.mark.parametrize(
    "time, data, expectation",
    [
        (np.arange(12), np.arange(12), does_not_raise()),
        (np.arange(12), "not_an_array", pytest.raises(TypeError, match="Data should be array-like"))
    ]
)
def test_lazy_load_hdf5_is_array(time, data, expectation):
    file_path = Path('data.h5')
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
        h5_data = h5py.File(file_path, 'r')["data"]
        with expectation:
            nap.Tsd(t=time, d=h5_data, load_array=False)
    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize(
    "time, data",
    [
        (np.arange(12), np.arange(12)),
    ]
)
@pytest.mark.parametrize("convert_flag", [True, False])
def test_lazy_load_hdf5_is_array(time, data, convert_flag):
    file_path = Path('data.h5')
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
        # get the tsd
        h5_data = h5py.File(file_path, 'r')["data"]
        tsd = nap.Tsd(t=time, d=h5_data, load_array=convert_flag)
        if convert_flag:
            assert isinstance(tsd.d, np.ndarray)
        else:
            assert isinstance(tsd.d, h5py.Dataset)
    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize("time, data", [(np.arange(12), np.arange(12))])
@pytest.mark.parametrize("cls", [nap.Tsd, nap.TsdFrame, nap.TsdTensor])
@pytest.mark.parametrize("func", [np.exp, lambda x: x*2])
def test_lazy_load_hdf5_apply_func(time, data, func,cls):
    """Apply a unary function to a lazy loaded array."""
    file_path = Path('data.h5')
    try:
        if cls is nap.TsdFrame:
            data = data[:, None]
        elif cls is nap.TsdTensor:
            data = data[:, None, None]
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
        # get the tsd
        h5_data = h5py.File(file_path, 'r')["data"]
        # lazy load and apply function
        res = func(cls(t=time, d=h5_data, load_array=False))
        assert isinstance(res, cls)
        assert isinstance(res.d, np.ndarray)
    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize("time, data", [(np.arange(12), np.arange(12))])
@pytest.mark.parametrize("cls", [nap.Tsd, nap.TsdFrame, nap.TsdTensor])
@pytest.mark.parametrize(
    "method_name, args",
    [
        ("bin_average", [0.1]),
        ("count", [0.1]),
        ("interpolate", [nap.Ts(t=np.linspace(0, 12, 50))]),
        ("convolve", [np.ones(3)]),
        ("smooth", [2]),
        ("dropna", [True]),
        ("value_from", [nap.Tsd(t=np.linspace(0, 12, 20), d=np.random.normal(size=20))]),
        ("copy", []),
        ("get", [2, 7])
    ]
)
def test_lazy_load_hdf5_apply_method(time, data, method_name, args, cls):
    file_path = Path('data.h5')
    try:
        if cls is nap.TsdFrame:
            data = data[:, None]
        elif cls is nap.TsdTensor:
            data = data[:, None, None]
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
        # get the tsd
        h5_data = h5py.File(file_path, 'r')["data"]
        # lazy load and apply function
        tsd = cls(t=time, d=h5_data, load_array=False)
        func = getattr(tsd, method_name)
        out = func(*args)
        assert isinstance(out.d, np.ndarray)
    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize("time, data", [(np.arange(12), np.arange(12))])
@pytest.mark.parametrize(
    "method_name, args, expected_out_type",
    [
        ("threshold", [3], nap.Tsd),
        ("as_series", [], pd.Series),
        ("as_units", ['ms'], pd.Series),
        ("to_tsgroup", [], nap.TsGroup)
    ]
)
def test_lazy_load_hdf5_apply_method_tsd_specific(time, data, method_name, args, expected_out_type):
    file_path = Path('data.h5')
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
        # get the tsd
        h5_data = h5py.File(file_path, 'r')["data"]
        # lazy load and apply function
        tsd = nap.Tsd(t=time, d=h5_data, load_array=False)
        func = getattr(tsd, method_name)
        assert isinstance(func(*args), expected_out_type)
    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()


@pytest.mark.parametrize("time, data", [(np.arange(12), np.arange(12))])
@pytest.mark.parametrize(
    "method_name, args, expected_out_type",
    [
        ("as_dataframe", [], pd.DataFrame),
    ]
)
def test_lazy_load_hdf5_apply_method_tsdframe_specific(time, data, method_name, args, expected_out_type):
    file_path = Path('data.h5')
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data[:, None])
        # get the tsd
        h5_data = h5py.File(file_path, 'r')["data"]
        # lazy load and apply function
        tsd = nap.TsdFrame(t=time, d=h5_data, load_array=False)
        func = getattr(tsd, method_name)
        assert isinstance(func(*args), expected_out_type)
    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()


def test_lazy_load_hdf5_tsdframe_loc():
    file_path = Path('data.h5')
    data = np.arange(10).reshape(5, 2)
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=data)
        # get the tsd
        h5_data = h5py.File(file_path, 'r')["data"]
        # lazy load and apply function
        tsd = nap.TsdFrame(t=np.arange(data.shape[0]), d=h5_data, load_array=False).loc[1]
        assert isinstance(tsd, nap.Tsd)
        assert all(tsd.d == np.array([1, 3, 5, 7, 9]))

    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()

@pytest.mark.parametrize(
    "lazy, expected_type",
    [
        (True, h5py.Dataset),
        (False, np.ndarray),
    ]
)
def test_lazy_load_nwb(lazy, expected_type):
    try:
        nwb = nap.NWBFile("tests/nwbfilestest/basic/pynapplenwb/A2929-200711.nwb", lazy_loading=lazy)
    except:
        nwb = nap.NWBFile("nwbfilestest/basic/pynapplenwb/A2929-200711.nwb", lazy_loading=lazy)

    tsd = nwb["z"]
    assert isinstance(tsd.d, expected_type)
    nwb.io.close()
