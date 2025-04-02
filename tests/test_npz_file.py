"""Tests of NPZ file functions"""

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pynapple as nap

# look for tests folder
path = Path(__file__).parent
if path.name == "pynapple":
    path = path / "tests"
path = path / "npzfilestest"

# Recursively remove the folder:
shutil.rmtree(path, ignore_errors=True)
path.mkdir(exist_ok=True, parents=True)

path2 = path.parent / "sub"
path2.mkdir(exist_ok=True, parents=True)


# Populate the folder
data = {
    "tsd": nap.Tsd(t=np.arange(100), d=np.arange(100)),
    "ts": nap.Ts(t=np.sort(np.random.rand(10) * 100)),
    "tsdframe": nap.TsdFrame(
        t=np.arange(100),
        d=np.random.rand(100, 10),
    ),
    "tsdframe_minfo": nap.TsdFrame(
        t=np.arange(100),
        d=np.random.rand(100, 10),
        metadata={"minfo": np.ones(10)},
    ),
    "tsgroup": nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 200)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
        },
    ),
    "tsgroup_minfo": nap.TsGroup(
        {
            0: nap.Ts(t=np.arange(0, 200)),
            1: nap.Ts(t=np.arange(0, 200, 0.5), time_units="s"),
            2: nap.Ts(t=np.arange(0, 300, 0.2), time_units="s"),
        },
        minfo=[1, 2, 3],
    ),
    "iset": nap.IntervalSet(start=np.array([0.0, 5.0]), end=np.array([1.0, 6.0])),
    "iset_minfo": nap.IntervalSet(
        start=np.array([0.0, 5.0]), end=np.array([1.0, 6.0]), metadata={"minfo": [1, 2]}
    ),
}
for k, d in data.items():
    d.save(path / (k + ".npz"))


@pytest.mark.parametrize("path", [path])
def test_init(path):
    tsd = nap.Tsd(t=np.arange(100), d=np.arange(100))
    file_path = path / "tsd.npz"
    tsd.save(file_path)
    file = nap.NPZFile(file_path)
    assert isinstance(file, nap.NPZFile)
    assert isinstance(file.type, np.str_)
    assert file.type == "Tsd"


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize(
    "k",
    [
        "tsd",
        "ts",
        "tsdframe",
        "tsdframe_minfo",
        "tsgroup",
        "tsgroup_minfo",
        "iset",
        "iset_minfo",
    ],
)
def test_load(path, k):
    file_path = path / (k + ".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    if hasattr(tmp, "metadata_columns") and len(tmp.metadata_columns):
        pd.testing.assert_frame_equal(tmp.metadata, data[k].metadata)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["tsgroup", "tsgroup_minfo"])
def test_load_tsgroup(path, k):
    file_path = path / (k + ".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert tmp.keys() == data[k].keys()
    assert np.all(tmp[neu] == data[k][neu] for neu in tmp.keys())
    np.testing.assert_array_almost_equal(
        tmp.time_support.values, data[k].time_support.values
    )


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["tsgroup", "tsgroup_minfo"])
def test_load_tsgroup_backward_compatibility(path, k):
    """
    For npz files saved without the _metadata keys
    """
    file_path = path / (k + ".npz")
    tmp = dict(np.load(file_path, allow_pickle=True))
    # Adding one metadata element outside the _metadata key
    tag = np.random.randn(3)
    tmp["tag"] = tag
    np.savez(file_path, **tmp)

    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert tmp.keys() == list(data[k].keys())
    assert np.all(tmp[neu] == data[k][neu] for neu in tmp.keys())
    np.testing.assert_array_almost_equal(
        tmp.time_support.values, data[k].time_support.values
    )
    assert "rate" in tmp.metadata.columns
    np.testing.assert_array_almost_equal(tmp.tag, tag)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["tsd"])
def test_load_tsd(path, k):
    file_path = path / (k + ".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert np.all(tmp.d == data[k].d)
    assert np.all(tmp.t == data[k].t)
    np.testing.assert_array_almost_equal(
        tmp.time_support.values, data[k].time_support.values
    )


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["ts"])
def test_load_ts(path, k):
    file_path = path / (k + ".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert np.all(tmp.t == data[k].t)
    np.testing.assert_array_almost_equal(
        tmp.time_support.values, data[k].time_support.values
    )


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["tsdframe", "tsdframe_minfo"])
def test_load_tsdframe(path, k):
    file_path = path / (k + ".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert np.all(tmp.t == data[k].t)
    np.testing.assert_array_almost_equal(
        tmp.time_support.values, data[k].time_support.values
    )
    assert np.all(tmp.columns == data[k].columns)
    assert np.all(tmp.d == data[k].d)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["tsdframe", "tsdframe_minfo"])
def test_load_tsdframe_backward_compatibility(path, k):
    file_path = path / (k + ".npz")
    tmp = dict(np.load(file_path, allow_pickle=True))
    tmp.pop("_metadata")
    np.savez(file_path, **tmp)
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    assert np.all(tmp.t == data[k].t)
    np.testing.assert_array_almost_equal(
        tmp.time_support.values, data[k].time_support.values
    )
    assert np.all(tmp.columns == data[k].columns)
    assert np.all(tmp.d == data[k].d)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["iset", "iset_minfo"])
def test_load_intervalset(path, k):
    file_path = path / (k + ".npz")
    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    np.testing.assert_array_almost_equal(tmp.values, data[k].values)


@pytest.mark.parametrize("path", [path])
@pytest.mark.parametrize("k", ["iset", "iset_minfo"])
def test_load_intervalset_backward_compatibility(path, k):
    file_path = path / (k + ".npz")
    tmp = dict(np.load(file_path, allow_pickle=True))
    tmp.pop("_metadata")
    np.savez(file_path, **tmp)

    file = nap.NPZFile(file_path)
    tmp = file.load()
    assert isinstance(tmp, type(data[k]))
    np.testing.assert_array_almost_equal(tmp.values, data[k].values)
    # Testing the slicing
    np.testing.assert_array_almost_equal(tmp[0].values, data[k].values[0, None])


@pytest.mark.parametrize("path", [path])
def test_load_non_npz(path):
    file_path = path / "random.npz"
    tmp = np.random.rand(100)
    np.savez(file_path, a=tmp)
    file = nap.NPZFile(file_path)

    assert file.type == "npz"
    a = file.load()
    assert isinstance(a, np.lib.npyio.NpzFile)
    np.testing.assert_array_equal(tmp, a["a"])
