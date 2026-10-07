"""Native state blocks executed by the V2 engine."""

import pytest
from roboflow_workflows.execution_engine.v2.blocks.state import (
    StateCompareAndSetBlock,
    StateGetBlock,
    StateIncrementBlock,
    StateSetBlock,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.declaration import spec_of
from roboflow_workflows.execution_engine.v2.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.state import (
    MISSING,
    ManagedState,
    StateBackendError,
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

    from roboflow_workflows.execution_engine.v2.state.redis import RedisStateBackend

    server = request.getfixturevalue("redis_server")
    namespace = request.getfixturevalue("namespace")
    state = ManagedState(RedisStateBackend(server.url), namespace=namespace)

    return state


@pytest.fixture(params=["memory", "redis"])
def state(request):
    managed_state = _state(request)
    yield managed_state
    managed_state.backend.close()


def test_state_blocks_chain_global_operations(state):
    plan = _plan(
        [
            {
                "type": "v2/state_increment",
                "name": "count",
                "trigger": "$inputs.x",
                "key": "count",
                "amount": 2,
            },
            {
                "type": "v2/state_set",
                "name": "remember",
                "trigger": "$steps.count.value",
                "key": "last",
                "value": "$inputs.x",
            },
            {
                "type": "v2/state_compare_and_set",
                "name": "arm",
                "trigger": "$steps.remember.applied",
                "key": "mode",
                "expected_missing": True,
                "new": {"armed": [1]},
            },
            {
                "type": "v2/state_get",
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
                "type": "v2/state_compare_and_set",
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
                "type": "v2/state_set",
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


def test_source_scope_uses_the_input_source(state):
    plan = _plan(
        [
            {
                "type": "v2/state_increment",
                "name": "count",
                "trigger": "$inputs.x",
                "scope": "source",
                "key": "frames",
            },
        ],
        {"count": "$steps.count.value"},
    )
    session = plan.create_session(resources={"managed_state": state})

    counts = [
        session.run({"x": _sourced(1, source_id)}).outputs.data["count"]
        for source_id in ["cam_a", "cam_b", "cam_a"]
    ]

    assert counts == [1, 1, 2]
    assert state.for_source("cam_a").get("frames") == 2
    assert state.global_.get("frames", MISSING) is MISSING


def test_source_scope_without_a_source_fails_the_step(state):
    plan = _plan(
        [
            {
                "type": "v2/state_get",
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


def test_default_state_lives_for_one_session():
    plan = _plan(
        [
            {
                "type": "v2/state_increment",
                "name": "count",
                "trigger": "$inputs.x",
                "key": "runs",
            },
        ],
        {"count": "$steps.count.value"},
    )
    first, second = plan.create_session(), plan.create_session()

    counts = [first.run({"x": 1}).outputs.data["count"] for _ in range(3)]
    fresh = second.run({"x": 1}).outputs.data["count"]

    assert counts == [1, 2, 3]
    assert fresh == 1


@pytest.mark.parametrize(
    "raw",
    [
        {"new": 1},
        {"expected": 1, "expected_missing": True, "new": 2},
        {"expected": 1},
        {"expected": 1, "new": 2, "delete": True},
        {"expected": "$inputs.x", "expected_missing": True, "delete": True},
        {"expected_missing": True, "new": "$inputs.x", "delete": True},
    ],
    ids=[
        "no-expectation",
        "expected-and-missing",
        "no-new-value",
        "new-and-delete",
        "selector-expected-and-missing",
        "selector-new-and-delete",
    ],
)
def test_compare_and_set_params_reject_ambiguous_choices(raw):
    spec = spec_of(StateCompareAndSetBlock)

    with pytest.raises(ParamsValidationError, match="exactly one of"):
        spec.validate_params({"key": "k", **raw})


@pytest.mark.parametrize(
    "raw",
    [
        {"expected": None, "new": 1},
        {"expected_missing": True, "new": 1},
        {"expected": 1, "delete": True},
        {"expected_missing": True, "delete": True},
        {"expected": "$inputs.x", "new": "$inputs.x", "expected_missing": False},
    ],
    ids=[
        "null-expected",
        "missing-then-store",
        "delete-if-equal",
        "both-flags",
        "selectors",
    ],
)
def test_compare_and_set_params_accept_one_choice_per_side(raw):
    spec = spec_of(StateCompareAndSetBlock)

    params = spec.validate_params({"key": "k", **raw})

    assert params.key == "k"


def test_compare_and_set_block_resolves_selectors_at_run_time(state):
    state.global_.set("level", 3)
    plan = _plan(
        [
            {
                "type": "v2/state_compare_and_set",
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


def test_ambiguous_compare_and_set_step_fails_compilation():
    with pytest.raises(ParamsValidationError, match="exactly one of"):
        _plan(
            [
                {
                    "type": "v2/state_compare_and_set",
                    "name": "bump",
                    "key": "level",
                    "new": 1,
                },
            ],
            {"applied": "$steps.bump.applied"},
        )


def _counting_plan():
    plan = _plan(
        [
            {
                "type": "v2/state_increment",
                "name": "count",
                "trigger": "$inputs.x",
                "key": "runs",
            },
        ],
        {"count": "$steps.count.value"},
        state={"global": {"runs": 10}},
    )

    return plan


def test_session_closes_the_state_it_created_through_the_defaults_view():
    session = _counting_plan().create_session()

    count = session.run({"x": 1}).outputs.data["count"]
    owned = session.owned_state
    session.close()

    assert count == 11
    with pytest.raises(StateBackendError, match="closed"):
        owned.global_.get("runs")


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
