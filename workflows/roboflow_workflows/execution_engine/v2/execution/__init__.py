"""Executor of compiled V2 plans: one run, steps in plan order.

``ExecutionSession.run`` and ``RunResult.rows`` delegate here::

    run_session(session, inputs=...)
        prepare_inputs            defaults, V1 input forms, broadcasting, codecs
        for step in plan.steps    plan order; one step at a time (steps.py)
        build_result              one entry per selected port (outputs.py)

    build_rows(result, serialize=...)   V1-shaped rows (outputs.py)

A passive pipeline (``pipelining.passive``) prepares inputs when a caller
submits them and executes each submission with ``run_prepared`` on one of its
workers. The algorithm is the same; the run's ``Coordination`` and ``Ticket``
make its steps take their turn at shared stages, so different runs may be at
different steps at once. Serially, ``SERIAL`` gates nothing.

The session owns the block instances, so state persists across runs of one
session; every run starts with fresh entries. Futures are resolved before any
dependent step and before the result is returned (ready boundary); no
deferred-output mode is offered.
"""

import uuid
from typing import Any, Callable, Dict, Mapping, Optional

from roboflow_workflows.execution_engine.v2.context import use_pulse_run_id
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.execution.outputs import (
    build_result,
    build_rows,
)
from roboflow_workflows.execution_engine.v2.execution.steps import (
    RunState,
    execute_step,
)
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    SERIAL,
    Coordination,
    RunAborted,
    Ticket,
)
from roboflow_workflows.execution_engine.v2.plan import ExecutionSession, RunResult

__all__ = ["build_rows", "run_prepared", "run_session"]


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
    result = _run(
        session,
        entries=lambda: prepare_inputs(session.plan, inputs),
        coordination=SERIAL,
        ticket=None,
        aborted=None,
    )

    return result


def run_prepared(
    session: ExecutionSession,
    *,
    entries: Dict[str, Entry],
    coordination: Coordination,
    ticket: Ticket,
    aborted: Callable[[], Exception],
) -> RunResult:
    """Execute a session's plan once as one turn-taking run of a pipeline.

    Args:
        session: Session holding the plan and its block instances.
        entries: Inputs already validated by ``prepare_inputs``; owned by
            this run.
        coordination: The pipeline's coordination: stage gates and the
            serialized observer and error handler.
        ticket: This run's place in the order of every stage.
        aborted: Builds the error reported to the observer and raised when
            the coordination aborts this run before it completes.

    Returns:
        The run result.

    Raises:
        StepExecutionError: When a step fails; the error handler and the
            observer are notified first.
        Exception: The error built by ``aborted`` when the run stops at a
            stage because the coordination aborted.
    """
    result = _run(
        session,
        entries=lambda: entries,
        coordination=coordination,
        ticket=ticket,
        aborted=aborted,
    )

    return result


def _run(
    session: ExecutionSession,
    *,
    entries: Callable[[], Dict[str, Entry]],
    coordination: Coordination,
    ticket: Optional[Ticket],
    aborted: Optional[Callable[[], Exception]],
) -> RunResult:
    run_id = uuid.uuid4().hex
    observer = coordination.observer(session)
    with use_pulse_run_id(run_id):
        observer.on_run_started(session_id=session.session_id, run_id=run_id)
        try:
            run = RunState(
                session=session,
                run_id=run_id,
                inputs=entries(),
                coordination=coordination,
                ticket=ticket,
            )
            run.record("run_started", run_id=run_id, session_id=session.session_id)
            for step in session.plan.steps:
                execute_step(run, step)
            result = build_result(run)
        except RunAborted:
            if aborted is None:
                raise
            error = aborted()
            observer.on_run_finished(run_id=run_id, result=None, error=error)
            raise error from None
        except Exception as error:
            observer.on_run_finished(run_id=run_id, result=None, error=error)
            raise

        observer.on_run_finished(run_id=run_id, result=result, error=None)

    return result
