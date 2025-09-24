from typing import Optional

from inference.core.workflows.execution_engine.profiling.core import (
    NullWorkflowsProfiler,
    WorkflowsProfiler,
    execution_phase,
)


def test_null_profiler_on_workflow_run_start() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    profiler.start_workflow_run()

    # then
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


def test_null_profiler_on_workflow_run_end() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    profiler.end_workflow_run()

    # then
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


def test_null_profiler_on_start_execution_phase() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    profiler.start_execution_phase(
        name="some",
        categories=["a"],
        metadata={"some": "data"},
    )

    # then
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


def test_null_profiler_on_end_execution_phase() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    profiler.end_execution_phase(
        name="some",
        categories=["a"],
        metadata={"some": "data"},
    )

    # then
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


def test_null_profiler_on_notify_event() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    profiler.notify_event(
        name="some",
        categories=["a"],
        metadata={"some": "data"},
    )

    # then
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


def test_null_profiler_execution_phase() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    with profiler.profile_execution_phase(
        name="some", categories=["a"], metadata={"some": "data"}
    ):
        pass

    # then
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


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


def test_null_profiler_decorating_function() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    result = example_decorated_function(a=3, b=4, profiler=profiler)

    # then
    assert result == 7, "Expected result to be correct"
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


def test_null_profiler_decorating_instance_method() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()
    instance = ExampleDecoratedClass()

    # when
    result = instance.example_instance_method(a=3, b=4, profiler=profiler)

    # then
    assert result == 7, "Expected result to be correct"
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"


def test_null_profiler_decorating_class_method() -> None:
    # given
    profiler = NullWorkflowsProfiler.init()

    # when
    result = ExampleDecoratedClass.example_cls_method(a=3, b=4, profiler=profiler)

    # then
    assert result == 7, "Expected result to be correct"
    assert profiler.export_trace() == [], "Expected nothing logged and no errors"
