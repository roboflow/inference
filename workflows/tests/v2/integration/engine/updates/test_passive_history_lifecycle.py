"""Passive history lifecycle: close, processing reset, failures.

A passive history plan keeps its operators in the session::

    first run ── constructs every operator (a failure rolls back the built ones)
    later runs ── reuse them; a failed run poisons them until a reset
    reset ────── retires them through the update's cleanup; the next run warms fresh
    close ────── closes each once, drops partial tails, raises after trying all
"""

from typing import Any, Dict, List, Set

import pytest
from roboflow_workflows.v2.engine import Catalogue, compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Group, Output
from roboflow_workflows.v2.engine.errors import ContractError, WorkflowExecutionError
from roboflow_workflows.v2.engine.operators.window import Window

# Operator names whose constructor or close raises; cleared around every test.
FAILING: Dict[str, Set[str]] = {"init": set(), "close": set(), "sum": set()}
# Every Tracked instance constructed, in order.
CREATED: List["Tracked"] = []


@pytest.fixture(autouse=True)
def _clear_records():
    for names in FAILING.values():
        names.clear()
    CREATED.clear()
    yield
    for names in FAILING.values():
        names.clear()
    CREATED.clear()


class Tracked(Window):
    """A Window that counts its closes and fails on request."""

    type = "test/tracked_window@v1"

    def __init__(self, *, name, params, inputs) -> None:
        if name in FAILING["init"]:
            raise RuntimeError(f"{name} constructor failed")
        super().__init__(name=name, params=params, inputs=inputs)
        self.close_calls = 0
        CREATED.append(self)

    def close(self) -> None:
        self.close_calls += 1
        super().close()
        if self.name in FAILING["close"]:
            raise ValueError(f"{self.name} close failed")


class Sum(Block):
    type = "test/sum@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        values: Group()

    def run(self, *, values):
        if "sum" in FAILING["sum"]:
            raise RuntimeError("sum failed")
        return {"out": sum(values)}


CATALOGUE = Catalogue([Sum], operators=[Tracked])


def _plan():
    """``x`` feeds a size-3 ``first`` window and a size-2 ``second`` window."""
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "x"}],
        "operators": [
            {
                "type": Tracked.type,
                "name": "first",
                "size": 3,
                "step": 1,
                "collect": {"x": "$inputs.x"},
            },
            {
                "type": Tracked.type,
                "name": "second",
                "size": 2,
                "step": 1,
                "collect": {"x": "$inputs.x"},
            },
        ],
        "steps": [{"type": Sum.type, "name": "sum", "values": "$operators.second.x"}],
        "outputs": [
            {"type": "JsonField", "name": "first", "selector": "$operators.first.x"},
            {"type": "JsonField", "name": "sum", "selector": "$steps.sum.out"},
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def _instances(session) -> Dict[str, Any]:
    slots = session._passive_history.slots
    instances = {name: slot.instance for name, slot in slots.items()}

    return instances


def _rows(session, values: List[int]) -> List[Dict[str, Any]]:
    rows = [session.run({"x": x}).rows()[0] for x in values]

    return rows


def test_close_drops_the_partial_tail_once() -> None:
    session = _plan().create_session()
    _rows(session, [1, 2])
    first = _instances(session)["first"]
    assert first._tracks  # two of three rows, warming

    session.close()
    session.close()

    assert first.counters.closed and not first.counters.finished
    assert not first._tracks
    assert first.close_calls == 1
    assert session._passive_history is None


def test_reset_restarts_warming_with_fresh_operators() -> None:
    plan = _plan()
    session = plan.create_session()
    assert _rows(session, [1, 2, 3])[-1] == {"first": [1, 2, 3], "sum": 5}
    old = _instances(session)

    receipt = session.update(plan, reset=True)
    rows = _rows(session, [4, 5, 6])

    assert receipt.processing_version == 1 and session.processing_version == 1
    assert receipt.cleanup.wait(5) and receipt.cleanup.state == "done"
    assert rows == [
        {"first": None, "sum": None},
        {"first": None, "sum": 9},
        {"first": [4, 5, 6], "sum": 11},
    ]
    new = _instances(session)
    assert all(new[name] is not old[name] for name in old)
    assert all(instance.close_calls == 1 for instance in old.values())
    assert all(instance.close_calls == 0 for instance in new.values())
    session.close()


def test_failed_run_poisons_history_until_a_reset_closes_it() -> None:
    plan = _plan()
    session = plan.create_session()
    _rows(session, [1])
    FAILING["sum"].add("sum")

    with pytest.raises(Exception, match="sum failed"):
        session.run({"x": 2})
    FAILING["sum"].clear()
    with pytest.raises(ContractError, match="processing reset") as refused:
        session.run({"x": 3})

    assert "sum failed" in str(refused.value.__cause__)
    old = _instances(session)
    assert all(instance.close_calls == 0 for instance in old.values())

    session.update(plan, reset=True)

    assert all(instance.close_calls == 1 for instance in old.values())
    assert _rows(session, [4, 5]) == [
        {"first": None, "sum": None},
        {"first": None, "sum": 9},
    ]
    session.close()


def test_close_tries_every_operator_then_raises_the_failure() -> None:
    session = _plan().create_session()
    _rows(session, [1])
    instances = _instances(session)
    FAILING["close"].add("first")

    with pytest.raises(ValueError, match="first close failed"):
        session.close()
    session.close()

    assert session.closed
    assert [instances[name].close_calls for name in ("first", "second")] == [1, 1]
    assert instances["second"].counters.closed
    assert not instances["first"].counters.closed


def test_close_with_several_failures_names_each_of_them() -> None:
    session = _plan().create_session()
    _rows(session, [1])
    FAILING["close"].update({"first", "second"})

    with pytest.raises(WorkflowExecutionError) as raised:
        session.close()

    message = str(raised.value)
    assert "first close failed" in message and "second close failed" in message
    assert isinstance(raised.value.__cause__, ValueError)
    assert session.closed


def test_reset_reports_a_throwing_close_and_the_next_run_warms_fresh() -> None:
    plan = _plan()
    session = plan.create_session()
    _rows(session, [1, 2])
    old = _instances(session)
    FAILING["close"].add("first")

    receipt = session.update(plan, reset=True)

    assert receipt.cleanup.wait(5) and receipt.cleanup.state == "failed"
    assert receipt.cleanup.errors == (
        "closing operator 'first': ValueError: first close failed",
    )
    assert [old[name].close_calls for name in ("first", "second")] == [1, 1]
    assert session._passive_history is None
    assert session.processing_version == 1
    FAILING["close"].clear()
    assert _rows(session, [3]) == [{"first": None, "sum": None}]
    assert all(
        instance is not old[name] for name, instance in _instances(session).items()
    )
    session.close()


def test_constructor_failure_closes_built_operators_and_the_next_run_retries() -> None:
    plan = _plan()
    session = plan.create_session()
    built, failing = list(plan.operators)  # construction follows plan order
    FAILING["init"].add(failing)

    with pytest.raises(WorkflowExecutionError, match=failing) as raised:
        session.run({"x": 1})

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert f"{failing} constructor failed" in str(raised.value)
    assert [instance.name for instance in CREATED] == [built]
    assert CREATED[0].close_calls == 1 and CREATED[0].counters.closed
    assert session._passive_history is None

    FAILING["init"].clear()
    assert _rows(session, [2, 3]) == [
        {"first": None, "sum": None},
        {"first": None, "sum": 5},
    ]
    assert len(CREATED) == 3
    session.close()


def test_rollback_close_failure_keeps_the_constructor_error_as_the_cause() -> None:
    plan = _plan()
    session = plan.create_session()
    built, failing = list(plan.operators)
    FAILING["init"].add(failing)
    FAILING["close"].add(built)

    with pytest.raises(WorkflowExecutionError) as raised:
        session.run({"x": 1})

    message = str(raised.value)
    assert f"{failing} constructor failed" in message
    assert f"{built} close failed" in message
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert CREATED[0].close_calls == 1
    session.close()
