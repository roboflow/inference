"""Native state blocks executed by the V2 engine, on the in-memory backend.

Backend-independent cases live in ``tests/v2/support/state/state_blocks_contract.py``;
``integration/external/redis/state/test_state_blocks.py`` runs them on Redis.
"""

import pytest
from roboflow_workflows.v2.blocks.state.state_compare_and_set.v1 import StateCompareAndSetBlock
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError
from roboflow_workflows.v2.engine.state import StateBackendError

from tests.v2.support.state.state_blocks_contract import (  # noqa: F401  collected here
    _counting_plan,
    _plan,
    _state,
    test_compare_and_set_block_can_delete,
    test_compare_and_set_block_resolves_selectors_at_run_time,
    test_non_portable_value_fails_the_step,
    test_session_borrows_caller_state_and_seeds_each_default_once,
    test_source_scope_uses_the_input_source,
    test_source_scope_without_a_source_fails_the_step,
    test_state_blocks_chain_global_operations,
)


@pytest.fixture(params=["memory"])
def state(request):
    managed_state = _state(request)
    yield managed_state
    managed_state.backend.close()


def test_default_state_lives_for_one_session():
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


def test_ambiguous_compare_and_set_step_fails_compilation():
    with pytest.raises(ParamsValidationError, match="exactly one of"):
        _plan(
            [
                {
                    "type": "roboflow_core/state_compare_and_set@v1",
                    "name": "bump",
                    "key": "level",
                    "new": 1,
                },
            ],
            {"applied": "$steps.bump.applied"},
        )


def test_session_closes_the_state_it_created_through_the_defaults_view():
    session = _counting_plan().create_session()

    count = session.run({"x": 1}).outputs.data["count"]
    owned = session.owned_state
    session.close()

    assert count == 11
    with pytest.raises(StateBackendError, match="closed"):
        owned.global_.get("runs")
