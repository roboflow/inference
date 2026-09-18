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

from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.v1.core import _resolve_execution_observer
from roboflow_workflows.prototypes.observer import (
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


def test_resolver_keeps_an_explicit_dynamic_override() -> None:
    # Unique edge not covered by the engine-level tests below: DYNAMIC_KEY
    # bound to a CALLABLE, resolved independently of CORE_KEY.
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


# --- dynamic blocks (Task 6) -------------------------------------------------

_IDENTITY_BLOCK = """
def run(self, value) -> BlockResult:
    return {"result": value}
"""

_IDENTITY_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "dynamic_blocks_definitions": [
        {
            "type": "DynamicBlockDefinition",
            "manifest": {
                "type": "ManifestDescription",
                "block_type": "IdentityProbe",
                "inputs": {
                    "value": {
                        "type": "DynamicInputDefinition",
                        "selector_types": ["input_parameter"],
                    }
                },
                "outputs": {"result": {"type": "DynamicOutputDefinition", "kind": []}},
            },
            "code": {"type": "PythonCode", "run_function_code": _IDENTITY_BLOCK},
        }
    ],
    "steps": [{"type": "IdentityProbe", "name": "probe", "value": "$inputs.value"}],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.probe.result"}
    ],
}


def _engine_and_block(init_parameters: dict):
    """Build the engine and return it with its one (dynamic) step instance."""
    engine = ExecutionEngine.init(
        workflow_definition=_IDENTITY_WORKFLOW, init_parameters=init_parameters
    )._engine
    block = next(iter(engine._compiled_workflow.steps.values())).step
    return engine, block


def test_no_binding_gives_the_engine_and_the_block_the_null_observer() -> None:
    engine, block = _engine_and_block({"workflows_core.api_key": "k"})

    assert engine._execution_observer is NULL_EXECUTION_OBSERVER
    assert block._execution_observer is NULL_EXECUTION_OBSERVER


def test_a_namespaced_object_reaches_the_engine_and_the_block() -> None:
    bound = NullExecutionObserver()

    engine, block = _engine_and_block({"workflows_core.api_key": "k", CORE_KEY: bound})

    assert engine._execution_observer is bound
    assert block._execution_observer is bound


def test_a_bare_key_reaches_the_engine_and_the_block() -> None:
    bound = NullExecutionObserver()

    engine, block = _engine_and_block(
        {"workflows_core.api_key": "k", "execution_observer": bound}
    )

    assert engine._execution_observer is bound
    assert block._execution_observer is bound


def test_a_factory_is_called_once_and_its_result_is_shared() -> None:
    # A callable binding is invoked by `_retrieve_init_parameter`; a block must
    # receive the *instance*, never the factory.
    made = []

    def factory():
        made.append(NullExecutionObserver())
        return made[-1]

    engine, block = _engine_and_block(
        {"workflows_core.api_key": "k", CORE_KEY: factory}
    )

    assert len(made) == 1
    assert engine._execution_observer is made[0]
    assert block._execution_observer is made[0]


def test_an_explicit_dynamic_override_wins_for_dynamic_blocks_only() -> None:
    engine_observer, dynamic_observer = NullExecutionObserver(), NullExecutionObserver()

    engine, block = _engine_and_block(
        {
            "workflows_core.api_key": "k",
            CORE_KEY: engine_observer,
            DYNAMIC_KEY: dynamic_observer,
        }
    )

    assert engine._execution_observer is engine_observer
    assert block._execution_observer is dynamic_observer


def test_a_replaced_observer_reaches_the_next_engines_dynamic_block() -> None:
    """With a shared, mutated dictionary the second engine's block kept the
    first engine's automatically generated mirror - the private copy is what
    makes the replacement reach the block."""
    first_observer, second_observer = NullExecutionObserver(), NullExecutionObserver()
    parameters = {"workflows_core.api_key": "k", CORE_KEY: first_observer}

    _, first_block = _engine_and_block(parameters)
    parameters[CORE_KEY] = second_observer
    second_engine, second_block = _engine_and_block(parameters)

    assert first_block._execution_observer is first_observer
    assert second_engine._execution_observer is second_observer
    assert second_block._execution_observer is second_observer
    assert DYNAMIC_KEY not in parameters


def test_a_false_valued_observer_reaches_the_dynamic_block() -> None:
    bound = _FalseValuedObserver()

    engine, block = _engine_and_block({"workflows_core.api_key": "k", CORE_KEY: bound})

    assert engine._execution_observer is bound
    assert block._execution_observer is bound
