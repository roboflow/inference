"""State blocks on every managed-state backend: memory and owned Redis.

Not collected on its own. unit/engine/state/test_state_blocks.py (memory) and
integration/external/redis/state/test_state_blocks.py (Redis) import these
tests and provide the state fixture with one backend each.
"""

import pytest
from roboflow_workflows.v2.blocks.state.state_compare_and_set.v1 import StateCompareAndSetBlock
from roboflow_workflows.v2.blocks.state.state_get.v1 import StateGetBlock
from roboflow_workflows.v2.blocks.state.state_increment.v1 import StateIncrementBlock
from roboflow_workflows.v2.blocks.state.state_set.v1 import StateSetBlock
from roboflow_workflows.v2.data.batch import (
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.state import (
    MISSING,
    ManagedState,
    StateScopeError,
    StateValueError,
)

_CATALOGUE = Catalogue(
    [StateGetBlock, StateSetBlock, StateIncrementBlock, StateCompareAndSetBlock]
)


def _plan(steps, outputs, *, state=None):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    if state is not None:
        definition["state"] = state
    plan = compile_workflow(definition, catalogue=_CATALOGUE)

    return plan


def _sourced(value, source_id):
    sample = EntryMetadata(sample={(): SampleContext(source_id)})
    input_value = InputValue(value, metadata=sample)

    return input_value


def _state(request):
    if request.param == "memory":
        state = ManagedState()
        return state

    from roboflow_workflows.v2.engine.state.redis import RedisStateBackend

    server = request.getfixturevalue("redis_server")
    namespace = request.getfixturevalue("namespace")
    state = ManagedState(RedisStateBackend(server.url), namespace=namespace)

    return state


def test_state_blocks_chain_global_operations(state):
    plan = _plan(
        [
            {
                "type": "roboflow_core/state_increment@v1",
                "name": "count",
                "trigger": "$inputs.x",
                "key": "count",
                "amount": 2,
            },
            {
                "type": "roboflow_core/state_set@v1",
                "name": "remember",
                "trigger": "$steps.count.value",
                "key": "last",
                "value": "$inputs.x",
            },
            {
                "type": "roboflow_core/state_compare_and_set@v1",
                "name": "arm",
                "trigger": "$steps.remember.applied",
                "key": "mode",
                "expected_missing": True,
                "new": {"armed": [1]},
            },
            {
                "type": "roboflow_core/state_get@v1",
                "name": "mode",
                "trigger": "$steps.arm.applied",
                "key": "mode",
                "default": "none",
            },
        ],
        {
            "count": "$steps.count.value",
            "armed_now": "$steps.arm.applied",
            "mode": "$steps.mode.value",
        },
    )
    session = plan.create_session(resources={"managed_state": state})

    first = session.run({"x": "a"}).outputs.data
    second = session.run({"x": "b"}).outputs.data

    assert dict(first) == {"count": 2, "armed_now": True, "mode": {"armed": [1]}}
    assert dict(second) == {"count": 4, "armed_now": False, "mode": {"armed": [1]}}
    assert state.global_.get("last") == "b"


def test_compare_and_set_block_can_delete(state):
    state.global_.set("lock", "held")
    plan = _plan(
        [
            {
                "type": "roboflow_core/state_compare_and_set@v1",
                "name": "release",
                "trigger": "$inputs.x",
                "key": "lock",
                "expected": "held",
                "delete": True,
            },
        ],
        {"released": "$steps.release.applied"},
    )

    result = plan.create_session(resources={"managed_state": state}).run({"x": 1})

    assert result.outputs.data["released"] is True
    assert state.global_.get("lock", MISSING) is MISSING


def test_non_portable_value_fails_the_step(state):
    plan = _plan(
        [
            {
                "type": "roboflow_core/state_set@v1",
                "name": "put",
                "trigger": "$inputs.x",
                "key": "k",
                "value": "$inputs.x",
            },
        ],
        {"applied": "$steps.put.applied"},
    )
    session = plan.create_session(resources={"managed_state": state})

    with pytest.raises(StepExecutionError) as raised:
        session.run({"x": (1, 2)})

    assert isinstance(raised.value.__cause__, StateValueError)
    assert state.global_.get("k", MISSING) is MISSING


# Each block chooses its scope itself, so every writing block gets a case.
# Runs: x=1 from cam_a, x=2 from cam_b, x=3 from cam_a.
@pytest.mark.parametrize(
    "operation, output, outputs_by_run, stored_by_source",
    [
        pytest.param(
            {"type": "roboflow_core/state_increment@v1"},
            "value",
            [1, 1, 2],
            {"cam_a": 2, "cam_b": 1},
            id="increment",
        ),
        pytest.param(
            {"type": "roboflow_core/state_set@v1", "value": "$inputs.x"},
            "applied",
            [True, True, True],
            {"cam_a": 3, "cam_b": 2},
            id="set",
        ),
        pytest.param(
            {
                "type": "roboflow_core/state_compare_and_set@v1",
                "expected_missing": True,
                "new": "$inputs.x",
            },
            "applied",
            [True, True, False],
            {"cam_a": 1, "cam_b": 2},
            id="compare_and_set",
        ),
    ],
)
def test_source_scope_uses_the_input_source(
    state, operation, output, outputs_by_run, stored_by_source
):
    plan = _plan(
        [
            {
                **operation,
                "name": "operation",
                "trigger": "$inputs.x",
                "scope": "source",
                "key": "frames",
            },
        ],
        {"output": f"$steps.operation.{output}"},
    )
    session = plan.create_session(resources={"managed_state": state})

    outputs = [
        session.run({"x": _sourced(x, source_id)}).outputs.data["output"]
        for x, source_id in [(1, "cam_a"), (2, "cam_b"), (3, "cam_a")]
    ]

    assert outputs == outputs_by_run
    for source_id, stored in stored_by_source.items():
        assert state.for_source(source_id).get("frames") == stored
    assert state.global_.get("frames", MISSING) is MISSING


def test_source_scope_without_a_source_fails_the_step(state):
    plan = _plan(
        [
            {
                "type": "roboflow_core/state_get@v1",
                "name": "read",
                "trigger": "$inputs.x",
                "scope": "source",
                "key": "k",
            },
        ],
        {"value": "$steps.read.value"},
    )
    session = plan.create_session(resources={"managed_state": state})

    with pytest.raises(StepExecutionError) as raised:
        session.run({"x": 1})

    assert isinstance(raised.value.__cause__, StateScopeError)


def test_compare_and_set_block_resolves_selectors_at_run_time(state):
    state.global_.set("level", 3)
    plan = _plan(
        [
            {
                "type": "roboflow_core/state_compare_and_set@v1",
                "name": "bump",
                "trigger": "$inputs.x",
                "key": "level",
                "expected": "$inputs.x",
                "new": "$inputs.x",
            },
        ],
        {"applied": "$steps.bump.applied"},
    )
    session = plan.create_session(resources={"managed_state": state})

    outcomes = [session.run({"x": x}).outputs.data["applied"] for x in [None, 3]]

    assert outcomes == [False, True]
    assert state.global_.get("level") == 3


def _counting_plan():
    plan = _plan(
        [
            {
                "type": "roboflow_core/state_increment@v1",
                "name": "count",
                "trigger": "$inputs.x",
                "key": "runs",
            },
        ],
        {"count": "$steps.count.value"},
        state={"global": {"runs": 10}},
    )

    return plan


def test_session_borrows_caller_state_and_seeds_each_default_once(state):
    plan = _counting_plan()
    first = plan.create_session(resources={"managed_state": state})

    first_count = first.run({"x": 1}).outputs.data["count"]
    first.close()
    state.global_.delete("runs")
    second = plan.create_session(resources={"managed_state": state})
    second_count = second.run({"x": 1}).outputs.data["count"]
    second.close()

    assert first.owned_state is None
    assert (first_count, second_count) == (11, 1)
    assert state.global_.get("runs") == 1
