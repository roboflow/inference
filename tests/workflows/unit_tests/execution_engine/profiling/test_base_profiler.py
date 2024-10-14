from typing import Optional

from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    WorkflowsProfiler,
    execution_phase,
)


def test_base_profiler_on_workflow_run_start() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    profiler.start_workflow_run()
    trace = profiler.export_trace()

    # then
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "workflow_run", "Expected correct event name"
    assert trace[0]["ph"] == "B", "Expected Begin phase to be marked"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_on_workflow_run_end() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    profiler.end_workflow_run()
    trace = profiler.export_trace()

    # then
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "workflow_run", "Expected correct event name"
    assert trace[0]["ph"] == "E", "Expected Begin phase to be marked"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_on_start_execution_phase() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    profiler.start_execution_phase(
        name="some",
        categories=["a", "b"],
        metadata={"some": "data"},
    )
    trace = profiler.export_trace()

    # then
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a,b"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "B", "Expected Begin phase to be marked"
    assert trace[0]["args"] == {"some": "data"}, "Expected metadata to be present"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
        "args",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_on_end_execution_phase() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    profiler.end_execution_phase(
        name="some",
        categories=["a"],
    )

    trace = profiler.export_trace()

    # then
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "E", "Expected End phase to be marked"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_on_notify_event() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    profiler.notify_event(
        name="some",
        categories=["a"],
        metadata={"some": "data"},
    )
    trace = profiler.export_trace()

    # then
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "I", "Expected Instant event to be parked"
    assert trace[0]["args"] == {"some": "data"}, "Expected metadata to be present"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
        "args",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_execution_phase() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    with profiler.profile_execution_phase(
        name="some", categories=["a"], metadata={"some": "data"}
    ):
        pass
    trace = profiler.export_trace()

    # then
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "X", "Expected single duration event to be provided"
    assert trace[0]["args"] == {"some": "data"}, "Expected metadata to be present"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
        "args",
        "dur",
    }, "Expected all required event keys to be denoted"


@execution_phase(
    name="some",
    categories=["a"],
    metadata={"some": "value"},
    runtime_metadata=["a"],
)
def example_decorated_function(
    a: int, b: int, profiler: Optional[WorkflowsProfiler] = None
) -> int:
    return a + b


class ExampleDecoratedClass:

    @classmethod
    @execution_phase(
        name="some",
        categories=["a"],
        metadata={"some": "value"},
        runtime_metadata=["a"],
    )
    def example_cls_method(
        cls, a: int, b: int, profiler: Optional[WorkflowsProfiler] = None
    ) -> int:
        return a + b

    @execution_phase(
        name="some",
        categories=["a"],
        metadata={"some": "value"},
        runtime_metadata=["a"],
    )
    def example_instance_method(
        self, a: int, b: int, profiler: Optional[WorkflowsProfiler] = None
    ) -> int:
        return a + b


def test_base_profiler_decorating_function() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    result = example_decorated_function(a=3, b=4, profiler=profiler)
    trace = profiler.export_trace()

    # then
    assert result == 7, "Expected result to be correct"
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "X", "Expected single duration event to be provided"
    assert trace[0]["args"] == {
        "some": "value",
        "a": 3,
    }, "Expected metadata to be present"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
        "args",
        "dur",
    }, "Expected all required event keys to be denoted"


def test_function_decorator_when_profiler_do_not_provided() -> None:
    # when
    result = example_decorated_function(a=3, b=4)

    # then
    assert result == 7, "Expected result to be correct"


def test_function_decorator_when_not_named_params_given() -> None:
    # THIS TEST SHOWCASES WEAKNESS OF DECORATOR - IT ONLY OPERATES ON NAMED PARAMS
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    result = example_decorated_function(3, 4, profiler)
    trace = profiler.export_trace()

    # then
    assert result == 7, "Expected result to be correct"
    assert len(trace) == 0, "Expected nothing tracked"


def test_function_decorator_when_not_named_runtime_metadata_given() -> None:
    # THIS TEST SHOWCASES WEAKNESS OF DECORATOR - IT ONLY OPERATES ON NAMED PARAMS
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    result = example_decorated_function(3, 4, profiler=profiler)
    trace = profiler.export_trace()

    # then
    assert result == 7, "Expected result to be correct"
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "X", "Expected single duration event to be provided"
    assert trace[0]["args"] == {
        "some": "value",
        "a": None,
    }, "Expected metadata to be present, but WITHOUT 'a' parameter value!"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
        "args",
        "dur",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_decorating_instance_method() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()
    instance = ExampleDecoratedClass()

    # when
    result = instance.example_instance_method(a=3, b=4, profiler=profiler)
    trace = profiler.export_trace()

    # then
    assert result == 7, "Expected result to be correct"
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "X", "Expected single duration event to be provided"
    assert trace[0]["args"] == {
        "some": "value",
        "a": 3,
    }, "Expected metadata to be present"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
        "args",
        "dur",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_decorating_class_method() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    result = ExampleDecoratedClass.example_cls_method(a=3, b=4, profiler=profiler)
    trace = profiler.export_trace()

    # then
    assert result == 7, "Expected result to be correct"
    assert len(trace) == 1, "Expected start event registered"
    assert trace[0]["name"] == "some", "Expected correct event name"
    assert (
        trace[0]["cat"] == "a"
    ), "Expected comma-separated list of categories to be preserved"
    assert trace[0]["ph"] == "X", "Expected single duration event to be provided"
    assert trace[0]["args"] == {
        "some": "value",
        "a": 3,
    }, "Expected metadata to be present"
    assert set(trace[0].keys()) == {
        "name",
        "ph",
        "pid",
        "tid",
        "ts",
        "cat",
        "args",
        "dur",
    }, "Expected all required event keys to be denoted"


def test_base_profiler_overflowing_buffer() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init(max_runs_in_buffer=2)

    # when
    profiler.start_workflow_run()
    profiler.notify_event(name="event_1")
    profiler.end_workflow_run()
    profiler.start_workflow_run()
    profiler.notify_event(name="event_2")
    profiler.end_workflow_run()
    profiler.start_workflow_run()
    profiler.notify_event(name="event_3")
    profiler.end_workflow_run()
    trace = profiler.export_trace()

    # then
    assert len(trace) == 6, "Expected 6 out of 9 events, 3 first should be dropped"
    events_names = [e["name"] for e in trace]
    assert events_names == [
        "workflow_run",
        "event_2",
        "workflow_run",  # second run events
        "workflow_run",
        "event_3",
        "workflow_run",  # third run events
    ]


def test_base_profiler_workflow_run_start_handling_pre_workflow_start_events() -> None:
    # given
    profiler = BaseWorkflowsProfiler.init()

    # when
    profiler.notify_event(name="pre_start_event")
    profiler.start_workflow_run()
    profiler.notify_event(name="event_1")
    trace = profiler.export_trace()

    # then
    assert len(trace) == 3, "Expected three events in trace"
    events_names = [e["name"] for e in trace]
    assert events_names == ["pre_start_event", "workflow_run", "event_1"]
