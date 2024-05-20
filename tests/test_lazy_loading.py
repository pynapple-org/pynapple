import h5py
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
            nap.Tsd(t=time, d=h5_data, conv_to_array=False)
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
        tsd = nap.Tsd(t=time, d=h5_data, conv_to_array=convert_flag)
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
        res = func(cls(t=time, d=h5_data, conv_to_array=False))
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
        ("bin_average", 0.1),
        ("count", 0.1),
        ("interpolate", nap.Ts(t=np.linspace(0, 12, 50))),
        ("convolve", np.ones(3)),
        ("smooth", 2),
        ("dropna", True),
        ("value_from", nap.Tsd(t=np.linspace(0, 12, 20), d=np.random.normal(size=20)))
    ]
)
def test_lazy_load_hdf5_apply_func(time, data, method_name, args, cls):
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
        tsd = cls(t=time, d=h5_data, conv_to_array=False)
        func = getattr(tsd, method_name)
        out = func(args)
        assert isinstance(out.d, np.ndarray)
    finally:
        # delete file
        if file_path.exists():
            file_path.unlink()
