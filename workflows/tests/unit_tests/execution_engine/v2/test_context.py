"""Execution context: explicit, nested and never leaking past errors."""

import subprocess
import sys
import threading

import pytest
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    NoExecutionContextError,
    get_execution_context,
    use_execution_context,
)

OUTER = ExecutionContext(("child", "count"), "Counting", "session-1")
INNER = ExecutionContext(
    ("child", "inner"), "Inner", "session-1", run_id="run-1", indices=[[0, 1]]
)


def test_reading_outside_any_call_is_an_explicit_error() -> None:
    # when
    with pytest.raises(NoExecutionContextError, match="No V2 execution context"):
        get_execution_context()


def test_nested_contexts_restore_the_outer_one() -> None:
    # when
    with use_execution_context(OUTER):
        seen_outer = get_execution_context()
        with use_execution_context(INNER):
            seen_inner = get_execution_context()
        seen_after = get_execution_context()

    # then
    assert seen_outer is OUTER and seen_inner is INNER and seen_after is OUTER
    assert INNER.indices == ((0, 1),) and INNER.step_selector == "$steps.child/inner"
    with pytest.raises(NoExecutionContextError):
        get_execution_context()


def test_errors_do_not_leak_the_context() -> None:
    # when
    with pytest.raises(ValueError):
        with use_execution_context(OUTER):
            raise ValueError("boom")

    # then
    with pytest.raises(NoExecutionContextError):
        get_execution_context()


def test_other_threads_do_not_see_this_threads_context() -> None:
    # given
    seen = []

    def read() -> None:
        try:
            seen.append(get_execution_context())
        except NoExecutionContextError as error:
            seen.append(error)

    # when
    with use_execution_context(OUTER):
        thread = threading.Thread(target=read)
        thread.start()
        thread.join()

    # then
    assert isinstance(seen[0], NoExecutionContextError)


def test_only_execution_contexts_can_be_entered() -> None:
    # when
    with pytest.raises(TypeError, match="ExecutionContext"):
        with use_execution_context({"step_path": ("a",)}):
            pass


def test_module_imports_nothing_heavy() -> None:
    # when
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import roboflow_workflows.execution_engine.v2.context; "
            "print(sorted(m for m in ('numpy', 'torch', 'cv2', 'supervision', "
            "'roboflow_workflows.execution_engine.v1') if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # then
    assert probe.stdout.strip() == "[]"
