"""The observer is resolved once per engine, into the engine's own copy.

`_retrieve_init_parameter` accepts `workflows_core.execution_observer` or the
bare name and calls a factory; the steps initialiser gives a block whatever
object sits under `dynamic_workflows_blocks.execution_observer` without
calling it. So the engine resolves the observer once and republishes the
resolved object under both keys - into a private copy of `init_parameters`,
never the caller's dictionary: a root reuses its dictionary for the next
engine, and a written-back factory result would make every later engine
inherit the first one's observer. Nothing here imports the server.
"""

import pytest

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1.core import (
    _resolve_execution_observer,
)
from inference.core.workflows.prototypes.observer import (
    NULL_EXECUTION_OBSERVER,
    NullExecutionObserver,
)

CORE_KEY = "workflows_core.execution_observer"
DYNAMIC_KEY = "dynamic_workflows_blocks.execution_observer"

_TRIVIAL_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [],
    "outputs": [],
}


# --- the resolver on plain dictionaries -----------------------------------


def test_resolver_returns_the_null_observer_when_nothing_is_bound() -> None:
    parameters = {"workflows_core.api_key": "k"}

    observer = _resolve_execution_observer(parameters)

    assert observer is NULL_EXECUTION_OBSERVER
    assert parameters[CORE_KEY] is NULL_EXECUTION_OBSERVER
    assert parameters[DYNAMIC_KEY] is NULL_EXECUTION_OBSERVER


def test_resolver_publishes_a_namespaced_object_under_both_keys() -> None:
    bound = NullExecutionObserver()
    parameters = {CORE_KEY: bound}

    assert _resolve_execution_observer(parameters) is bound
    assert parameters[CORE_KEY] is bound
    assert parameters[DYNAMIC_KEY] is bound


def test_resolver_accepts_a_bare_key() -> None:
    # `_retrieve_init_parameter` accepts the un-namespaced name, so a host may
    # bind it that way; dynamic blocks must not be left with the null observer.
    bound = NullExecutionObserver()
    parameters = {"execution_observer": bound}

    assert _resolve_execution_observer(parameters) is bound
    assert parameters[CORE_KEY] is bound
    assert parameters[DYNAMIC_KEY] is bound


def test_resolver_calls_a_factory_once_and_publishes_the_instance() -> None:
    made = []

    def factory():
        made.append(NullExecutionObserver())
        return made[-1]

    parameters = {CORE_KEY: factory}

    observer = _resolve_execution_observer(parameters)

    assert made == [observer]
    assert parameters[CORE_KEY] is observer
    assert parameters[DYNAMIC_KEY] is observer


def test_resolver_keeps_an_explicit_dynamic_override() -> None:
    core, dynamic = NullExecutionObserver(), NullExecutionObserver()
    parameters = {CORE_KEY: core, DYNAMIC_KEY: lambda: dynamic}

    assert _resolve_execution_observer(parameters) is core
    assert parameters[CORE_KEY] is core
    assert parameters[DYNAMIC_KEY] is dynamic


# --- the engine ------------------------------------------------------------


def _engine(init_parameters: dict):
    """The versioned engine behind the facade, built from a trivial workflow."""
    return ExecutionEngine.init(
        workflow_definition=_TRIVIAL_WORKFLOW, init_parameters=init_parameters
    )._engine


def test_the_engine_defaults_to_the_null_observer() -> None:
    engine = _engine({"workflows_core.api_key": "k"})

    assert engine._execution_observer is NULL_EXECUTION_OBSERVER


@pytest.mark.parametrize("key", [CORE_KEY, "execution_observer"])
def test_the_engine_holds_the_bound_observer(key: str) -> None:
    bound = NullExecutionObserver()

    engine = _engine({"workflows_core.api_key": "k", key: bound})

    assert engine._execution_observer is bound


def test_the_engine_holds_the_instance_a_factory_made() -> None:
    made = []

    def factory():
        made.append(NullExecutionObserver())
        return made[-1]

    engine = _engine({CORE_KEY: factory})

    assert len(made) == 1
    assert engine._execution_observer is made[0]


def test_engine_initialisation_leaves_the_callers_parameters_untouched() -> None:
    """The caller's dictionary is not the engine's scratch space.

    Before this phase `ExecutionEngineV1.init` already wrote
    `dynamic_workflows_blocks.api_key` into it; with a resolved observer
    written back too, a root reusing its dictionary would hand every later
    engine the first engine's observer.
    """

    def factory():
        return NullExecutionObserver()

    parameters = {"workflows_core.api_key": "k", CORE_KEY: factory}
    before = dict(parameters)

    _engine(parameters)

    assert parameters == before, sorted(set(parameters) ^ set(before))
    assert parameters[CORE_KEY] is factory


def test_two_engines_from_one_factory_binding_each_get_their_own_observer() -> None:
    made = []

    def factory():
        made.append(NullExecutionObserver())
        return made[-1]

    parameters = {"workflows_core.api_key": "k", CORE_KEY: factory}

    first = _engine(parameters)
    second = _engine(parameters)

    assert len(made) == 2
    assert first._execution_observer is made[0]
    assert second._execution_observer is made[1]


def test_replacing_the_bound_observer_reaches_the_next_engine_only() -> None:
    first_observer, second_observer = NullExecutionObserver(), NullExecutionObserver()
    parameters = {"workflows_core.api_key": "k", CORE_KEY: first_observer}

    first = _engine(parameters)
    parameters[CORE_KEY] = second_observer
    second = _engine(parameters)

    assert first._execution_observer is first_observer
    assert second._execution_observer is second_observer


class _FalseValuedObserver(NullExecutionObserver):
    """A conforming observer whose truth value is False.

    An observer that buffers what it records is naturally sized; empty, it is
    falsy. `steps_initialiser` hands an explicit value on unchanged and the
    resolver substitutes only for `None`, so a constructor that tests truth
    instead of `is not None` would silently swap such an observer for the
    null one.
    """

    def __len__(self) -> int:
        return 0


def test_a_false_valued_observer_survives_engine_construction() -> None:
    bound = _FalseValuedObserver()
    assert not bound  # the premise: a valid observer that is falsy

    engine = _engine({"workflows_core.api_key": "k", CORE_KEY: bound})

    assert engine._execution_observer is bound
