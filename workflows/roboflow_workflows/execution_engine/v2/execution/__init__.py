"""Sequential executor of compiled V2 plans.

``ExecutionSession.run`` and ``RunResult.rows`` delegate here::

    run_session(session, inputs=...)
        prepare_inputs            defaults, V1 input forms, broadcasting, codecs
        for step in plan.steps    plan order; one step at a time (steps.py)
        build_result              one entry per selected port (outputs.py)

    build_rows(result, serialize=...)   V1-shaped rows (outputs.py)

The session owns the block instances, so state persists across runs of one
session; every run starts with fresh entries. Futures are resolved before any
dependent step and before the result is returned (ready boundary); no
deferred-output mode and no parallel scheduling are offered.
"""

import uuid
from typing import Any, Mapping

from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.execution.outputs import (
    build_result,
    build_rows,
)
from roboflow_workflows.execution_engine.v2.execution.steps import (
    RunState,
    execute_step,
)
from roboflow_workflows.execution_engine.v2.plan import ExecutionSession, RunResult

__all__ = ["build_rows", "run_session"]


def run_session(session: ExecutionSession, *, inputs: Mapping[str, Any]) -> RunResult:
    """Execute a session's plan once.

    Args:
        session: Session holding the plan and its block instances.
        inputs: Workflow input values by name. The mapping is not modified.

    Returns:
        The run result.

    Raises:
        WorkflowInputError: When inputs are invalid.
        StepExecutionError: When a step fails; the session's error handler and
            observer are notified first.
    """
    run_id = uuid.uuid4().hex
    observer = session.observer
    observer.on_run_started(session_id=session.session_id, run_id=run_id)
    try:
        run = RunState(
            session=session,
            run_id=run_id,
            inputs=prepare_inputs(session.plan, inputs),
        )
        run.record("run_started", run_id=run_id, session_id=session.session_id)
        for step in session.plan.steps:
            execute_step(run, step)
        result = build_result(run)
    except Exception as error:
        observer.on_run_finished(run_id=run_id, result=None, error=error)
        raise

    observer.on_run_finished(run_id=run_id, result=result, error=None)

    return result
