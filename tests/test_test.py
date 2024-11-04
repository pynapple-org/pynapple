from numbers import Number
import inspect


import pickle
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from contextlib import nullcontext as does_not_raise
import warnings

import pynapple as nap


@pytest.fixture
def iset_meta():
    start = np.array([0, 10, 16, 25])
    end = np.array([5, 15, 20, 40])
    metadata = {"label": ["a", "b", "c", "d"], "info": np.arange(4)}
    return nap.IntervalSet(start=start, end=end, metadata=metadata)


@pytest.mark.parametrize(
    "name, set_exp, set_attr_exp, set_key_exp, get_attr_exp, get_key_exp",
    [
        # existing attribute and key
        (
            "start",
            pytest.warns(UserWarning, match="overlaps with an existing attribute"),
            pytest.raises(AttributeError, match="IntervalSet is immutable"),
            pytest.raises(RuntimeError, match="IntervalSet is immutable"),
            does_not_raise(),
            does_not_raise(),
        ),
        # existing attribute and key
        (
            "end",
            pytest.warns(UserWarning, match="overlaps with an existing attribute"),
            pytest.raises(AttributeError, match="IntervalSet is immutable"),
            pytest.raises(RuntimeError, match="IntervalSet is immutable"),
            does_not_raise(),
            does_not_raise(),
        ),
        # existing attribute
        (
            "values",
            pytest.warns(UserWarning, match="overlaps with an existing attribute"),
            pytest.raises(AttributeError, match="IntervalSet is immutable"),
            does_not_raise(),
            pytest.raises(ValueError),  # shape mismatch
            pytest.raises(AssertionError),  # we do want metadata
        ),
        # existing metdata
        (
            "label",
            does_not_raise(),
            does_not_raise(),
            does_not_raise(),
            pytest.raises(AssertionError),  # we do want metadata
            pytest.raises(AssertionError),  # we do want metadata
        ),
    ],
)
def test_iset_metadata_overlapping_names(
    iset_meta, name, set_exp, set_attr_exp, set_key_exp, get_attr_exp, get_key_exp
):
    assert hasattr(iset_meta, name)

    # warning when set
    with set_exp:
        iset_meta.set_info({name: np.ones(4)})
    # error when set as attribute
    with set_attr_exp:
        setattr(iset_meta, name, np.ones(4))
    # error when set as key
    with set_key_exp:
        iset_meta[name] = np.ones(4)
    # retrieve with get_info
    assert np.all(iset_meta.get_info(name) == np.ones(4))
    # make sure it doesn't access metadata if its an existing attribute or key
    with get_attr_exp:
        assert np.all(getattr(iset_meta, name) == np.ones(4)) == False
    # make sure it doesn't access metadata if its an existing key
    with get_key_exp:
        assert np.all(iset_meta[name] == np.ones(4)) == False
