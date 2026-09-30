"""Session lifecycle, host hooks, futures and payload identity."""

import json
from concurrent.futures import Future

import pytest
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import StepExecutionError
from roboflow_workflows.execution_engine.v2.plan import ExecutionObserver

from .blocks import Counter, Deferred, Echo, Failing, Mutate, Scale
from .plans import BATCH, SCALAR, PlanBuilder


def counter_plan():
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Counter, "count", at=BATCH, value="$inputs.values")
        .output("count", "$steps.count.count")
        .build()
    )

    return plan


def test_runs_of_one_session_share_block_state_and_sessions_are_isolated() -> None:
    plan = counter_plan()
    first_session = plan.create_session()
    second_session = plan.create_session()

    first = first_session.run({"values": [1, 2]}).rows()
    again = first_session.run({"values": [3]}).rows()
    fresh = second_session.run({"values": [4]}).rows()

    assert first == [{"count": 1}, {"count": 2}]
    assert again == [{"count": 3}]
    assert fresh == [{"count": 1}]
    assert (
        first_session.instances[("count",)] is not second_session.instances[("count",)]
    )


def test_run_state_is_fresh_after_a_failed_run() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Failing, "check", at=BATCH, value="$inputs.values")
        .output("value", "$steps.check.value")
        .build()
    )
    session = plan.create_session()

    with pytest.raises(StepExecutionError):
        session.run({"values": [1, -1]})
    result = session.run({"values": [5]})

    assert result.rows() == [{"value": 5}]
    assert len(session.instances[("check",)].calls) == 3


class RecordingObserver(ExecutionObserver):
    def __init__(self):
        self.events = []

    def on_run_started(self, *, session_id, run_id):
        self.events.append(("run_started",))

    def on_step_started(self, *, step, block_type):
        self.events.append(("step_started", step, block_type))

    def on_invocation(self, *, step, index, arguments, result):
        self.events.append(("invocation", step, index, dict(arguments), result))

    def on_invocation_skipped(self, *, step, index, reason):
        self.events.append(("skipped", step, index, reason))

    def on_step_finished(self, *, step, invocations, skipped):
        self.events.append(("step_finished", step, invocations, skipped))

    def on_error(self, *, error):
        self.events.append(("error", error.step_path, error.index))

    def on_run_finished(self, *, run_id, result, error):
        self.events.append(("run_finished", result is not None, type(error).__name__))


def test_observer_sees_steps_invocations_skips_and_run_end() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Echo, "echo", at=BATCH, value="$inputs.values")
        .output("value", "$steps.echo.value")
        .build()
    )
    observer = RecordingObserver()
    session = plan.create_session(observer=observer)

    session.run({"values": ["a", None]})

    assert observer.events == [
        ("run_started",),
        ("step_started", ("echo",), "test/echo@v1"),
        ("skipped", ("echo",), (1,), "empty_value"),
        ("invocation", ("echo",), (0,), {"value": "a"}, {"value": "a"}),
        ("step_finished", ("echo",), 1, 1),
        ("run_finished", True, "NoneType"),
    ]


def test_block_error_reaches_handler_and_observer_with_path_index_and_cause() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Failing, "check", at=BATCH, value="$inputs.values")
        .build()
    )
    handled = []
    observer = RecordingObserver()
    session = plan.create_session(observer=observer, error_handler=handled.append)

    with pytest.raises(StepExecutionError, match=r"negative -2") as caught:
        session.run({"values": [1, -2]})

    error = caught.value
    assert handled == [error]
    assert (error.step_path, error.index, error.block_type) == (
        ("check",),
        (1,),
        "test/failing@v1",
    )
    assert isinstance(error.__cause__, ValueError)
    assert ("error", ("check",), (1,)) in observer.events
    assert observer.events[-1] == ("run_finished", False, "StepExecutionError")


def test_error_handler_may_translate_the_error() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Failing, "check", at=BATCH, value="$inputs.values")
        .build()
    )

    def translate(error):
        raise KeyError(f"host view of {error.step_path}") from error

    session = plan.create_session(error_handler=translate)

    with pytest.raises(KeyError, match="check"):
        session.run({"values": [-1]})


def test_futures_are_resolved_before_consumers_and_before_the_result() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Deferred, "deferred", at=BATCH, value="$inputs.values")
        .step(Scale, "consumer", at=BATCH, value="$steps.deferred.doubled", factor=1)
        .output("doubled", "$steps.deferred.doubled")
        .output("pair", "$steps.deferred.pair")
        .output("children", "$steps.deferred.children")
        .output("consumed", "$steps.consumer.scaled")
        .build()
    )
    observer = RecordingObserver()
    session = plan.create_session(observer=observer)

    result = session.run({"values": [1, 2]})

    assert [call["value"] for call in session.instances[("consumer",)].calls] == [2, 4]
    assert result.rows() == [
        {"doubled": 2, "pair": [1, "fixed"], "children": [1.5], "consumed": 2},
        {"doubled": 4, "pair": [2, "fixed"], "children": [2.5], "consumed": 4},
    ]
    delivered = [event[4] for event in observer.events if event[0] == "invocation"]
    assert not any(
        isinstance(value, Future) for item in delivered for value in item.values()
    )


def test_failing_future_is_a_step_error_with_index_and_cause() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Deferred, "deferred", at=BATCH, value="$inputs.values", fail=True)
        .output("doubled", "$steps.deferred.doubled")
        .build()
    )

    with pytest.raises(StepExecutionError, match="future") as caught:
        plan.create_session().run({"values": [3]})

    assert caught.value.index == (0,)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_declared_mutation_is_visible_downstream_and_payload_identity_is_kept() -> None:
    plan = (
        PlanBuilder()
        .input("payload", SCALAR)
        .step(Mutate, "change", payload="$inputs.payload")
        .step(Echo, "read", value="$steps.change.payload")
        .output("seen", "$steps.read.value")
        .build()
    )
    payload = {"count": 0}
    inputs = {"payload": payload}

    result = plan.create_session().run(inputs)

    assert inputs == {"payload": payload} and inputs["payload"] is payload
    assert payload == {"count": 1}
    assert result.outputs.data["seen"] is payload


def test_input_preparation_does_not_rewrite_the_caller_mapping() -> None:
    plan = (
        PlanBuilder()
        .input("a", BATCH)
        .input("b", BATCH)
        .input("c", SCALAR, required=False, default=[1])
        .step(Scale, "s", at=BATCH, value="$inputs.a", factor="$inputs.b")
        .output("s", "$steps.s.scaled")
        .build()
    )
    inputs = {"a": [1, 2, 3], "b": 10}

    rows = plan.create_session().run(inputs).rows()

    assert rows == [{"s": 10}, {"s": 20}, {"s": 30}]
    assert inputs == {"a": [1, 2, 3], "b": 10}


def test_group_members_are_the_stored_payload_objects() -> None:
    plan = (
        PlanBuilder()
        .input("items", BATCH)
        .step(Echo, "echo", at=BATCH, value="$inputs.items")
        .output("items", "$steps.echo.value")
        .build()
    )
    first, second = {"id": 1}, {"id": 2}

    result = plan.create_session().run({"items": [first, second]})

    tree = result.outputs.data["items"]
    assert isinstance(tree, Batch)
    assert tree[0] is first and tree[1] is second


def test_trace_is_json_friendly_and_records_skips_and_calls() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Echo, "echo", at=BATCH, value="$inputs.values")
        .output("value", "$steps.echo.value")
        .build()
    )

    result = plan.create_session().run({"values": [1, None]})

    events = [event["event"] for event in result.trace]
    assert events == [
        "run_started",
        "step_started",
        "invocation_skipped",
        "invocation",
        "step_finished",
        "run_finished",
    ]
    json.dumps(list(result.trace))
