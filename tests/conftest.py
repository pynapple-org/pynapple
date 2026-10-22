"""Shared fixtures of the test suite."""

from functools import cached_property

import pytest

from pynapple.core.ts_group import TsGroup


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unbuilt_tsgroup: the test builds a partial TsGroup on purpose. The "
        "TsGroup construction spy does not check it.",
    )


def _cached_property_names(klass):
    return {
        name
        for base in klass.__mro__
        for name, value in vars(base).items()
        if isinstance(value, cached_property)
    }


class TsGroupSpy:
    """Record each TsGroup that a test creates, and how it got its state.

    ``TsGroup._init_state`` must be the only code that sets the state of a new
    group (``__setstate__`` restores a saved state instead). The spy finds a
    group that got attributes without ``_init_state``, before it, or after
    it. Only the cached properties can add attributes after ``_init_state``.
    """

    def __init__(self):
        self.created = []
        self.entry = {}  # id(obj) -> keys of __dict__ when _init_state starts
        self.exit = {}  # id(obj) -> keys of __dict__ when _init_state ends
        self.restored = set()  # id(obj) restored by __setstate__

    def violations(self):
        found = []
        for obj in self.created:
            keys = set(obj.__dict__)
            if not keys:
                # the construction failed before it set any state
                continue
            name = type(obj).__name__
            if id(obj) in self.restored:
                continue
            if id(obj) not in self.entry:
                found.append(f"{name} got attributes without `_init_state`: {keys}")
                continue
            if self.entry[id(obj)]:
                found.append(
                    f"{name} got attributes before `_init_state`: "
                    f"{self.entry[id(obj)]}"
                )
            if id(obj) in self.exit:
                late = keys - self.exit[id(obj)] - _cached_property_names(type(obj))
                if late:
                    found.append(f"{name} got attributes after `_init_state`: {late}")
        return found


@pytest.fixture(autouse=True)
def tsgroup_spy(request, monkeypatch):
    spy = TsGroupSpy()

    def recording_new(cls, *args, **kwargs):
        obj = object.__new__(cls)
        spy.created.append(obj)
        return obj

    init_state = TsGroup._init_state

    def spied_init_state(self, *args, **kwargs):
        if id(self) in spy.entry:
            raise AssertionError(
                f"`_init_state` is called two times on one {type(self).__name__}."
            )
        spy.entry[id(self)] = set(self.__dict__)
        init_state(self, *args, **kwargs)
        spy.exit[id(self)] = set(self.__dict__)

    setstate = TsGroup.__setstate__

    def spied_setstate(self, state):
        spy.restored.add(id(self))
        setstate(self, state)

    TsGroup.__new__ = staticmethod(recording_new)
    monkeypatch.setattr(TsGroup, "_init_state", spied_init_state)
    monkeypatch.setattr(TsGroup, "__setstate__", spied_setstate)
    try:
        yield spy
    finally:
        del TsGroup.__new__
    if request.node.get_closest_marker("unbuilt_tsgroup") is None:
        violations = spy.violations()
        if violations:
            pytest.fail("\n".join(violations), pytrace=False)
