import warnings
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from tempfile import TemporaryDirectory

import dask.array as da
import h5py
import numpy as np
import pandas as pd
import pytest
import zarr
from pynwb.testing.mock.base import mock_TimeSeries
from pynwb.testing.mock.file import mock_NWBFile

import pynapple as nap


@pytest.fixture
def dask_array_tsdframe():
    """Fixture for a Dask array."""
    array = da.random.random((100, 100), chunks=(10, 10))
    return array


@pytest.fixture
def dask_array_tsd():
    """Fixture for a Dask array."""
    array = da.random.random((100,), chunks=(10,))
    return array


@pytest.fixture
def dask_array_tsdtensor():
    """Fixture for a Dask array."""
    array = da.random.random((100, 10, 5), chunks=(10, 1, 2))
    return array


@pytest.fixture
def zarr_tsd():
    """Fixture for a Zarr array."""
    with TemporaryDirectory() as tmpdir:
        store = zarr.DirectoryStore(tmpdir)
        root = zarr.open(store, mode="w")
        array = root.create_dataset("data", shape=(100,), chunks=(10,), dtype="f8")
        array[:] = np.random.random((100,))
        yield array


@pytest.fixture
def zarr_tsdframe():
    """Fixture for a Zarr array."""
    with TemporaryDirectory() as tmpdir:
        store = zarr.DirectoryStore(tmpdir)
        root = zarr.open(store, mode="w")
        array = root.create_dataset(
            "data", shape=(100, 100), chunks=(10, 10), dtype="f8"
        )
        array[:] = np.random.random((100, 100))
        yield array


@pytest.fixture
def zarr_tsdtensor():
    """Fixture for a Zarr array."""
    with TemporaryDirectory() as tmpdir:
        store = zarr.DirectoryStore(tmpdir)
        root = zarr.open(store, mode="w")
        array = root.create_dataset(
            "data", shape=(100, 10, 2), chunks=(10, 9, 1), dtype="f8"
        )
        array[:] = np.random.random((100, 10, 2))
        yield array


@pytest.mark.parametrize(
    "time, data, expectation",
    [
        (np.arange(12), np.arange(12), does_not_raise()),
        (
            np.arange(12),
            "not_an_array",
            pytest.raises(TypeError, match="Data should be array-like"),
        ),
    ],
)
def test_lazy_load_hdf5_is_array(time, data, expectation, tmp_path):
    file_path = tmp_path / Path("data.h5")
    # try:
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
    with h5py.File(file_path, "r") as h5_data:
        with expectation:
            nap.Tsd(t=time, d=h5_data["data"], load_array=False)
    # finally:
    #     # delete file
    #     if file_path.exists():
    #         file_path.unlink()


@pytest.mark.parametrize(
    "time, data",
    [
        (np.arange(12), np.arange(12)),
    ],
)
@pytest.mark.parametrize("convert_flag", [True, False])
def test_lazy_load_hdf5_is_array(time, data, convert_flag, tmp_path):
    file_path = tmp_path / Path("data.h5")
    # try:
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
    # get the tsd
    h5_data = h5py.File(file_path, "r")["data"]
    tsd = nap.Tsd(t=time, d=h5_data, load_array=convert_flag)
    if convert_flag:
        assert not isinstance(tsd.d, h5py.Dataset)
    else:
        assert isinstance(tsd.d, h5py.Dataset)
    # finally:
    #     # delete file
    #     if file_path.exists():
    #         file_path.unlink()


@pytest.mark.parametrize("time, data", [(np.arange(12), np.arange(12))])
@pytest.mark.parametrize("cls", [nap.Tsd, nap.TsdFrame, nap.TsdTensor])
@pytest.mark.parametrize("func", [np.exp, lambda x: x * 2])
def test_lazy_load_hdf5_apply_func(time, data, func, cls, tmp_path):
    """Apply a unary function to a lazy loaded array."""
    file_path = tmp_path / Path("data.h5")
    # try:
    if cls is nap.TsdFrame:
        data = data[:, None]
    elif cls is nap.TsdTensor:
        data = data[:, None, None]
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
    # get the tsd
    h5_data = h5py.File(file_path, "r")["data"]
    # lazy load and apply function
    res = func(cls(t=time, d=h5_data, load_array=False))
    assert isinstance(res, cls)
    assert not isinstance(res.d, h5py.Dataset)
    # finally:
    #     # delete file
    #     if file_path.exists():
    #         file_path.unlink()


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
        (
            "value_from",
            [nap.Tsd(t=np.linspace(0, 12, 20), d=np.random.normal(size=20))],
        ),
        ("copy", []),
        ("get", [2, 7]),
    ],
)
def test_lazy_load_hdf5_apply_method(time, data, method_name, args, cls, tmp_path):
    file_path = tmp_path / Path("data.h5")
    # try:
    if cls is nap.TsdFrame:
        data = data[:, None]
    elif cls is nap.TsdTensor:
        data = data[:, None, None]
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
    # get the tsd
    h5_data = h5py.File(file_path, "r")["data"]
    # lazy load and apply function
    tsd = cls(t=time, d=h5_data, load_array=False)
    func = getattr(tsd, method_name)
    out = func(*args)
    assert not isinstance(out.d, h5py.Dataset)
    # finally:
    # # delete file
    # if file_path.exists():
    #     file_path.unlink()


@pytest.mark.parametrize("time, data", [(np.arange(12), np.arange(12))])
@pytest.mark.parametrize(
    "method_name, args, expected_out_type",
    [
        ("threshold", [3], nap.Tsd),
        ("as_series", [], pd.Series),
        ("as_units", ["ms"], pd.Series),
        ("to_tsgroup", [], nap.TsGroup),
    ],
)
def test_lazy_load_hdf5_apply_method_tsd_specific(
    time, data, method_name, args, expected_out_type, tmp_path
):
    file_path = tmp_path / Path("data.h5")
    # try:
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
    # get the tsd
    h5_data = h5py.File(file_path, "r")["data"]
    # lazy load and apply function
    tsd = nap.Tsd(t=time, d=h5_data, load_array=False)
    func = getattr(tsd, method_name)
    assert isinstance(func(*args), expected_out_type)
    # finally:
    #     # delete file
    #     if file_path.exists():
    #         file_path.unlink()


@pytest.mark.parametrize("time, data", [(np.arange(12), np.arange(12))])
@pytest.mark.parametrize(
    "method_name, args, expected_out_type",
    [
        ("as_dataframe", [], pd.DataFrame),
    ],
)
def test_lazy_load_hdf5_apply_method_tsdframe_specific(
    time, data, method_name, args, expected_out_type, tmp_path
):
    file_path = tmp_path / Path("data.h5")
    # try:
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data[:, None])
    # get the tsd
    h5_data = h5py.File(file_path, "r")["data"]
    # lazy load and apply function
    tsd = nap.TsdFrame(t=time, d=h5_data, load_array=False)
    func = getattr(tsd, method_name)
    assert isinstance(func(*args), expected_out_type)
    # finally:
    #     # delete file
    #     if file_path.exists():
    #         file_path.unlink()


def test_lazy_load_hdf5_tsdframe_loc(tmp_path):
    file_path = tmp_path / Path("data.h5")
    data = np.arange(10).reshape(5, 2)
    # try:
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
    # get the tsd
    h5_data = h5py.File(file_path, "r")["data"]
    # lazy load and apply function
    tsd = nap.TsdFrame(t=np.arange(data.shape[0]), d=h5_data, load_array=False).loc[1]
    assert isinstance(tsd, nap.Tsd)
    assert all(tsd.d == np.array([1, 3, 5, 7, 9]))

    # finally:
    #     # delete file
    #     if file_path.exists():
    #         file_path.unlink()


@pytest.mark.parametrize(
    "lazy",
    [
        (True),
        (False),
    ],
)
def test_lazy_load_nwb(lazy):
    try:
        nwb = nap.NWBFile(
            "tests/nwbfilestest/basic/pynapplenwb/A2929-200711.nwb", lazy_loading=lazy
        )
    except:
        nwb = nap.NWBFile(
            "nwbfilestest/basic/pynapplenwb/A2929-200711.nwb", lazy_loading=lazy
        )

    assert isinstance(nwb["z"].d, h5py.Dataset) is lazy
    nwb.close()


@pytest.mark.parametrize(
    "lazy",
    [
        (True),
        (False),
    ],
)
def test_lazy_load_function(lazy):
    try:
        nwb = nap.load_file(
            "tests/nwbfilestest/basic/pynapplenwb/A2929-200711.nwb", lazy_loading=lazy
        )
    except:
        nwb = nap.load_file(
            "nwbfilestest/basic/pynapplenwb/A2929-200711.nwb", lazy_loading=lazy
        )

    assert isinstance(nwb["z"].d, h5py.Dataset) is lazy
    nwb.close()


@pytest.mark.parametrize("data", [np.ones(10), np.ones((10, 2)), np.ones((10, 2, 2))])
def test_lazy_load_nwb_no_warnings(
    data, tmp_path
):  # tmp_path is a default fixture creating a temporary folder
    file_path = tmp_path / Path("data.h5")

    # try:
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data)
        time_series = mock_TimeSeries(name="TimeSeries", data=f["data"])
        nwbfile = mock_NWBFile()
        nwbfile.add_acquisition(time_series)
        nwb = nap.NWBFile(nwbfile)

        with warnings.catch_warnings(record=True) as w:
            tsd = nwb["TimeSeries"]
            tsd.count(0.1)
            assert isinstance(tsd.d, h5py.Dataset)

        if len(w):
            if not str(w[0].message).startswith("Converting 'd' to"):
                raise RuntimeError

    # finally:
    #     if file_path.exists():
    #         file_path.unlink()


def test_tsgroup_no_warnings(tmp_path):  # default fixture
    n_units = 2
    # try:
    for k in range(n_units):
        file_path = tmp_path / Path(f"data_{k}.h5")
        with h5py.File(file_path, "w") as f:
            f.create_dataset("spks", data=np.sort(np.random.uniform(0, 10, size=20)))
    with warnings.catch_warnings(record=True) as w:

        nwbfile = mock_NWBFile()

        for k in range(n_units):
            file_path = tmp_path / Path(f"data_{k}.h5")
            spike_times = h5py.File(file_path, "r")["spks"]
            nwbfile.add_unit(spike_times=spike_times)

        nwb = nap.NWBFile(nwbfile)
        tsgroup = nwb["units"]
        tsgroup.count(0.1)

    if len(w):
        if not str(w[0].message).startswith("Converting 'd' to"):
            raise RuntimeError

    # finally:
    #     for k in range(n_units):
    #         file_path = Path(f'data_{k}.h5')
    #         if file_path.exists():
    #             file_path.unlink()


def test_dask_lazy_loading_tsd(dask_array_tsd):
    tsd = nap.Tsd(
        t=np.arange(dask_array_tsd.shape[0]), d=dask_array_tsd, load_array=False
    )
    assert isinstance(tsd.d, da.Array)
    assert isinstance(tsd.restrict(nap.IntervalSet(0, 10)).d, da.Array)
    repr(tsd)
    assert isinstance(tsd.d, da.Array)
    assert isinstance(tsd[1:10].d, np.ndarray)
    tsd = nap.Tsd(
        t=np.arange(dask_array_tsd.shape[0]), d=dask_array_tsd, load_array=True
    )
    assert isinstance(tsd.d, np.ndarray)


def test_dask_lazy_compute_tsd(dask_array_tsd):
    tsd = nap.Tsd(
        t=np.arange(dask_array_tsd.shape[0]), d=dask_array_tsd, load_array=False
    )
    tsd = tsd + 1
    assert isinstance(tsd.d, da.Array)
    assert isinstance(tsd[:10].d, np.ndarray)
    assert tsd[:10]._load_array is True

    out = tsd.compute()
    assert isinstance(out.d, np.ndarray)
    assert isinstance(tsd.chunks, tuple)
    assert tsd._load_array is False

    out2 = tsd.map_blocks(np.exp)
    assert isinstance(out2.d, da.Array)
    assert out2._load_array is False

    assert isinstance(np.exp(tsd).d, da.Array)


def test_dask_lazy_loading_tsdframe(dask_array_tsdframe):
    tsdframe = nap.TsdFrame(
        t=np.arange(dask_array_tsdframe.shape[0]),
        d=dask_array_tsdframe,
        load_array=False,
    )
    assert isinstance(tsdframe.d, da.Array)
    assert isinstance(tsdframe.restrict(nap.IntervalSet(0, 10)).d, da.Array)
    repr(tsdframe)
    assert isinstance(tsdframe.d, da.Array)
    assert isinstance(tsdframe[1:10].d, np.ndarray)
    assert isinstance(tsdframe.loc[1].d, np.ndarray)
    tsdframe = nap.TsdFrame(
        t=np.arange(dask_array_tsdframe.shape[0]),
        d=dask_array_tsdframe,
        load_array=True,
    )
    assert isinstance(tsdframe.d, np.ndarray)


def test_dask_lazy_compute_tsdframe(dask_array_tsdframe):
    tsdframe = nap.TsdFrame(
        t=np.arange(dask_array_tsdframe.shape[0]),
        d=dask_array_tsdframe,
        load_array=False,
    )
    tsdframe = tsdframe**2
    assert isinstance(tsdframe.d, da.Array)
    assert isinstance(tsdframe[:10].d, np.ndarray)
    assert tsdframe[:10]._load_array is True

    out = tsdframe.compute()
    assert isinstance(out.d, np.ndarray)
    assert isinstance(tsdframe.chunks, tuple)
    assert tsdframe._load_array is False

    out2 = tsdframe.map_blocks(np.exp)
    assert isinstance(out2.d, da.Array)
    assert out2._load_array is False
    assert isinstance(np.exp(tsdframe).d, da.Array)


def test_dask_lazy_loading_tsdtensor(dask_array_tsdtensor):
    tsdtensor = nap.TsdTensor(
        t=np.arange(dask_array_tsdtensor.shape[0]),
        d=dask_array_tsdtensor,
        load_array=False,
    )
    assert isinstance(tsdtensor.d, da.Array)
    assert isinstance(tsdtensor.restrict(nap.IntervalSet(0, 10)).d, da.Array)
    repr(tsdtensor)
    assert isinstance(tsdtensor.d, da.Array)
    assert isinstance(tsdtensor[1:10].d, np.ndarray)
    tsdtensor = nap.TsdTensor(
        t=np.arange(dask_array_tsdtensor.shape[0]),
        d=dask_array_tsdtensor,
        load_array=True,
    )
    assert isinstance(tsdtensor.d, np.ndarray)


def test_dask_lazy_compute_tsdtensor(dask_array_tsdtensor):
    tsdtensor = nap.TsdTensor(
        t=np.arange(dask_array_tsdtensor.shape[0]),
        d=dask_array_tsdtensor,
        load_array=False,
    )
    tsdtensor = tsdtensor + 1
    assert isinstance(tsdtensor.d, da.Array)
    assert isinstance(tsdtensor[:10].d, np.ndarray)
    assert tsdtensor[:10]._load_array is True

    out = tsdtensor.compute()
    assert isinstance(out.d, np.ndarray)
    assert isinstance(tsdtensor.chunks, tuple)
    assert tsdtensor._load_array is False

    out2 = tsdtensor.map_blocks(np.exp)
    assert isinstance(out2.d, da.Array)
    assert out2._load_array is False

    assert isinstance(np.exp(tsdtensor).d, da.Array)


def test_lazy_load_zarr_tsd(zarr_tsd):
    tsd = nap.Tsd(t=np.arange(zarr_tsd.shape[0]), d=zarr_tsd, load_array=False)
    assert isinstance(tsd.d, zarr.Array)
    assert isinstance(tsd.restrict(nap.IntervalSet(0, 10)).d, np.ndarray)
    repr(tsd)
    assert isinstance(tsd.d, zarr.Array)
    assert isinstance(tsd[1:10].d, np.ndarray)
    tsd = nap.TsdFrame(t=np.arange(zarr_tsd.shape[0]), d=zarr_tsd, load_array=True)
    assert isinstance(tsd.d, np.ndarray)


def test_lazy_load_zarr_tsdframe(zarr_tsdframe):
    tsdframe = nap.TsdFrame(
        t=np.arange(zarr_tsdframe.shape[0]), d=zarr_tsdframe, load_array=False
    )
    assert isinstance(tsdframe.d, zarr.Array)
    assert isinstance(tsdframe.restrict(nap.IntervalSet(0, 10)).d, np.ndarray)
    repr(tsdframe)
    assert isinstance(tsdframe.d, zarr.Array)
    assert isinstance(tsdframe[1:10].d, np.ndarray)
    tsdframe = nap.TsdFrame(
        t=np.arange(zarr_tsdframe.shape[0]), d=zarr_tsdframe, load_array=True
    )
    assert isinstance(tsdframe.d, np.ndarray)


def test_lazy_load_zarr_tsdtensor(zarr_tsdtensor):
    tsdtensor = nap.TsdTensor(
        t=np.arange(zarr_tsdtensor.shape[0]), d=zarr_tsdtensor, load_array=False
    )
    assert isinstance(tsdtensor.d, zarr.Array)
    assert isinstance(tsdtensor.restrict(nap.IntervalSet(0, 10)).d, np.ndarray)
    repr(tsdtensor)
    assert isinstance(tsdtensor.d, zarr.Array)
    assert isinstance(tsdtensor[1:10].d, np.ndarray)
    tsdtensor = nap.TsdTensor(
        t=np.arange(zarr_tsdtensor.shape[0]), d=zarr_tsdtensor, load_array=True
    )
    assert isinstance(tsdtensor.d, np.ndarray)
