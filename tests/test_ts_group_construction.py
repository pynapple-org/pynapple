"""All the TsGroup objects get their state from `TsGroup._init_state`.

The spy of ``conftest.py`` checks this for each test of the suite. The tests
below check the derived groups directly, and check that the spy finds the
faults that it must find.
"""

import pickle
from functools import cached_property

import numpy as np
import pytest

import pynapple as nap
from pynapple.core.ts_group import TsGroup


def _state_keys(group):
    """Keys of ``__dict__`` without the cached properties, which a group
    computes on demand."""
    cached = {
        name
        for base in type(group).__mro__
        for name, value in vars(base).items()
        if isinstance(value, cached_property)
    }
    return set(group.__dict__) - cached


@pytest.fixture
def group():
    return nap.TsGroup(
        {
            1: nap.Ts(t=np.arange(0, 10, 0.5)),
            2: nap.Ts(t=np.arange(0, 10, 1.0)),
            5: nap.Ts(t=np.arange(2, 8, 0.25)),
        },
        time_support=nap.IntervalSet(0, 10),
        metadata={"label": ["a", "b", "a"]},
    )


@pytest.mark.parametrize(
    "derive",
    [
        lambda g: g.restrict(nap.IntervalSet(2, 6)),
        lambda g: g.get(2, 6),
        lambda g: g[[1, 5]],
        lambda g: g[g.label == "a"],
        lambda g: pickle.loads(pickle.dumps(g)),
    ],
    ids=["restrict", "get", "key_subset", "metadata_subset", "pickle"],
)
def test_derived_group_has_same_attributes(group, derive):
    derived = derive(group)
    assert type(derived) is TsGroup
    assert _state_keys(derived) == _state_keys(group)


@pytest.mark.parametrize(
    "state_attrs, match",
    [
        ({"_times": np.zeros(0)}, r"Missing: \['_clusters'"),
        (
            {**{name: None for name in TsGroup._STATE_ATTRS}, "_extra": None},
            r"Not expected: \['_extra'\]",
        ),
    ],
    ids=["missing", "extra"],
)
def test_init_state_checks_attribute_names(state_attrs, match):
    obj = TsGroup.__new__(TsGroup)
    with pytest.raises(TypeError, match=match):
        obj._init_state(np.zeros(0), nap.IntervalSet(0, 1), **state_attrs)


class TestSpy:
    """The spy must find each fault below. The tests clear ``spy.created``,
    so that the teardown of the spy does not fail."""

    def test_attribute_before_init_state(self, tsgroup_spy):
        class Early(TsGroup):
            def __init__(self, *args, **kwargs):
                self.__dict__["_early"] = True
                super().__init__(*args, **kwargs)

        Early({0: nap.Ts(t=[1.0, 2.0])})
        violations = tsgroup_spy.violations()
        tsgroup_spy.created.clear()
        assert len(violations) == 1
        assert "before `_init_state`" in violations[0]
        assert "_early" in violations[0]

    def test_attribute_after_init_state(self, tsgroup_spy):
        class Late(TsGroup):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__["_late"] = True

        Late({0: nap.Ts(t=[1.0, 2.0])})
        violations = tsgroup_spy.violations()
        tsgroup_spy.created.clear()
        assert len(violations) == 1
        assert "after `_init_state`" in violations[0]
        assert "_late" in violations[0]

    def test_state_without_init_state(self, tsgroup_spy, group):
        obj = TsGroup.__new__(TsGroup)
        obj.__dict__.update(group.__dict__)
        violations = tsgroup_spy.violations()
        tsgroup_spy.created.clear()
        assert len(violations) == 1
        assert "without `_init_state`" in violations[0]

    def test_failed_construction_is_not_reported(self, tsgroup_spy):
        with pytest.raises(ValueError):
            nap.TsGroup({0.5: nap.Ts(t=[1.0])})
        assert tsgroup_spy.violations() == []

    def test_cached_properties_are_not_reported(self, tsgroup_spy, group):
        group[1]
        assert "_ragged_index" in group.__dict__
        assert tsgroup_spy.violations() == []
