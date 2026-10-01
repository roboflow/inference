"""Passive pipeline: overlap whole runs of one session with bounded workers.

``ExecutionSession.pipeline`` delegates here::

    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        futures = [pipeline.submit({"image": image}) for image in images]
    results = [future.result() for future in futures]

    submit(inputs)        caller thread, one submitter at a time:
                          prepare_inputs (errors raise here, no ticket used);
                          wait for an idle worker (bounded, no queue);
                          ticket ("$passive", n) in submission order
    worker                run_prepared(...): the serial run, step by step,
                          taking its turn at every stage gate
    close()               no new submissions; wait for running ones; join;
                          then release the session

Runs of one pipeline enter every stage in submission order, one call at a
time per stage, so a stateful block sees calls in submission order while
run 1 may execute an earlier step (or phase) than run 0.

Engine-held submissions: at most ``max_in_flight`` running, plus at most one
prepared submission whose caller waits for a worker. Other blocked callers
have prepared nothing. Input objects and returned Futures belong to the
caller; the engine does not copy inputs, so reusing one mutable object in
two submissions shares it between their runs.

Failure: the first run that fails sets its ``Future`` to the error and
aborts the pipeline. Runs waiting at a stage stop with
``PipelineAbortedError``; calls already running finish (they cannot be
interrupted); later ``submit`` calls raise. ``cancel()`` aborts in the same
way without an error. The ``with`` block cancels when its body raises.

While a pipeline is open, the session refuses ``run`` and another
pipeline. User callbacks (observer, error handler) of the pipeline's runs
are serialized. ``submit``, ``close`` and leaving the ``with`` block from a
pipeline worker (for example inside an observer callback) raise
``ContractError`` instead of waiting for themselves.
"""

import threading
import time
from concurrent.futures import Future
from typing import Any, Dict, Mapping, Optional

from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.execution import run_prepared
from roboflow_workflows.execution_engine.v2.execution.entries import Entry
from roboflow_workflows.execution_engine.v2.execution.inputs import prepare_inputs
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    PASSIVE_DOMAIN,
    PipelineCounters,
    PipelinedCoordination,
    Ticket,
)
from roboflow_workflows.execution_engine.v2.pipelining.workers import (
    OwnedThreads,
    WorkerPool,
)
from roboflow_workflows.execution_engine.v2.plan import ExecutionSession, RunResult

__all__ = [
    "OUTCOMES",
    "PassivePipeline",
    "PipelineAbortedError",
    "PipelineFullError",
    "open_pipeline",
]

OUTCOMES = ("submitted", "completed", "failed", "aborted", "full")
"""Submission totals of ``PassivePipeline.outcomes``."""


class PipelineFullError(ContractError):
    """``submit`` found no idle worker: ``block=False``, or the timeout passed."""


class PipelineAbortedError(ContractError):
    """A submission stopped because its pipeline failed or was cancelled.

    Args:
        message: What happened.
        failure: The first failure of the pipeline; ``None`` after ``cancel``.
    """

    def __init__(self, message: str, *, failure: Optional[BaseException] = None):
        super().__init__(message)
        self.failure = failure


def open_pipeline(
    session: ExecutionSession, *, options: PipelineOptions
) -> "PassivePipeline":
    """Open a passive pipeline over a session's block instances.

    ``ExecutionSession.pipeline`` calls this after checking that the plan
    declares no sources and that ``options`` are ``PipelineOptions``.

    Args:
        session: Session of a plan without sources.
        options: ``max_in_flight`` bounds the submissions in progress;
            overload policies do not apply (the caller submits).

    Returns:
        The open pipeline; close it, or use it as a context manager.

    Raises:
        ContractError: When the session has an open pipeline or a ``run``
            in progress.
    """
    session._claim_pipeline()
    try:
        pipeline = PassivePipeline(session, options=options)
    except BaseException:
        session._release_pipeline()
        raise

    return pipeline


class PassivePipeline:
    """Bounded overlapping runs of one session; create with ``open_pipeline``.

    Args:
        session: Session whose block instances every run shares.
        options: Pipeline options; ``max_in_flight`` workers are started.

    Attributes:
        counters: Gauges and per-stage counts. ``executing`` and
            ``live_states`` count running submissions (at most
            ``max_in_flight``); one more prepared submission may wait for a
            worker. Counts only; no byte or device-memory bound.
    """

    def __init__(self, session: ExecutionSession, *, options: PipelineOptions):
        self.session = session
        self.options = options
        self._coordination = PipelinedCoordination(session, options=options)
        self.counters: PipelineCounters = self._coordination.counters
        self._owned = OwnedThreads()
        self._state_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._next_ordinal = 0
        self._closing = False
        self._closed = False
        self._outcomes = dict.fromkeys(OUTCOMES, 0)
        self._pool = WorkerPool(
            options.max_in_flight,
            name=f"v2-pipeline-{session.session_id[:8]}",
            owner=self._owned,
            counters=self.counters,
        )

    def __enter__(self) -> "PassivePipeline":
        return self

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        if error is not None:
            self.cancel()
        self.close()

    @property
    def failure(self) -> Optional[BaseException]:
        """The failure that aborted the pipeline; ``None`` if none did."""
        failure = self._coordination.abort_cause

        return failure

    @property
    def outcomes(self) -> Dict[str, int]:
        """Totals of ``OUTCOMES``.

        After ``close``: submitted == completed + failed + aborted. ``full``
        counts ``submit`` calls rejected with ``PipelineFullError``.
        """
        with self._state_lock:
            outcomes = dict(self._outcomes)

        return outcomes

    def submit(
        self,
        inputs: Mapping[str, Any],
        *,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> "Future[RunResult]":
        """Validate inputs and start one run as soon as a worker is idle.

        Nothing is queued: only an idle worker accepts a run. Runs take
        their turns at every stage in the order their submissions were
        accepted.

        Args:
            inputs: Workflow input values by name, as for ``session.run``.
                Values are not copied: do not mutate them until the run ends.
            block: Wait for an idle worker; otherwise fail at once.
            timeout: Longest wait in seconds when ``block``; ``None`` waits
                without limit.

        Returns:
            A future of the run's ``RunResult``. It fails with the step's
            ``StepExecutionError``, or with ``PipelineAbortedError`` when
            another submission's failure or ``cancel`` stopped this run.

        Raises:
            WorkflowInputError: When inputs are invalid; nothing is submitted
                and the pipeline stays open.
            PipelineFullError: When no worker became idle in time.
            PipelineAbortedError: When the pipeline failed or was cancelled.
            ContractError: When the pipeline is closed, or when called on a
                pipeline worker (for example from an observer callback).
        """
        self._owned.reject_wait("PassivePipeline.submit()")
        self._check_accepting()

        deadline = None if timeout is None else time.monotonic() + timeout
        if not self._submit_lock.acquire(timeout=_lock_timeout(deadline, block=block)):
            raise self._full()
        try:
            # Prepared only by the one submitter allowed to wait for a worker.
            self._check_accepting()
            entries = prepare_inputs(self.session.plan, inputs)
            future = self._accept(entries, block=block, deadline=deadline)
        finally:
            self._submit_lock.release()

        return future

    def cancel(self) -> None:
        """Abort without an error; never waits.

        No submission is accepted afterwards and blocked ``submit`` calls
        return. Runs waiting at a stage stop with ``PipelineAbortedError``;
        calls already running finish. Call ``close`` (or leave the ``with``
        block) to wait for the workers.
        """
        with self._state_lock:
            self._closing = True
        self._pool.close()
        self._coordination.abort()

    def close(self) -> None:
        """Stop accepting, wait for the runs in progress and join the workers.

        Idempotent; a concurrent second call returns once the first is done.
        The session is released only after every worker has been joined. If
        joining is interrupted, the session stays claimed and ``close`` can
        be called again. Run failures are not raised here: they are in the
        submissions' futures and in ``failure``.

        Raises:
            ContractError: When called on a pipeline worker.
        """
        self._owned.reject_wait("PassivePipeline.close()")
        with self._close_lock:
            if self._closed:
                return

            with self._state_lock:
                self._closing = True
            self._pool.close()
            # A submitter that passed the open check first returns first.
            with self._submit_lock:
                pass
            self._pool.shutdown()

            self._closed = True
            self.session._release_pipeline()

    def _accept(
        self, entries: Dict[str, Entry], *, block: bool, deadline: Optional[float]
    ) -> "Future[RunResult]":
        future: "Future[RunResult]" = Future()
        # Running from the start: an accepted run cannot be withdrawn, so its
        # ticket is always retired by the run itself or released by an abort.
        future.set_running_or_notify_cancel()
        ticket = Ticket(PASSIVE_DOMAIN, self._next_ordinal)

        def work() -> None:
            self._execute(future, entries=entries, ticket=ticket)

        if block:
            remaining = (
                None if deadline is None else max(deadline - time.monotonic(), 0)
            )
            accepted = self._pool.submit(work, timeout=remaining)
        else:
            accepted = self._pool.try_submit(work)
        if not accepted:
            self._check_accepting()
            raise self._full()

        self._next_ordinal += 1
        self._count("submitted")

        return future

    def _execute(
        self, future: "Future[RunResult]", *, entries: Dict[str, Entry], ticket: Ticket
    ) -> None:
        with self.counters.track("live_states"):
            try:
                result = run_prepared(
                    self.session,
                    entries=entries,
                    coordination=self._coordination,
                    ticket=ticket,
                    aborted=lambda: self._aborted(ticket),
                )
            except PipelineAbortedError as error:
                self._count("aborted")
                future.set_exception(error)
            except BaseException as error:
                self._fail(error)
                future.set_exception(error)
            else:
                self._count("completed")
                future.set_result(result)

    def _fail(self, error: BaseException) -> None:
        # A step failure already aborted with this error (execute_step); the
        # first abort wins, so this covers failures outside steps.
        self._coordination.abort(error)
        self._pool.close()
        with self._state_lock:
            self._outcomes["failed"] += 1
            self._closing = True

    def _aborted(self, ticket: Ticket) -> PipelineAbortedError:
        error = PipelineAbortedError(
            f"Submission {ticket.ordinal} did not complete: {self._abort_reason()}",
            failure=self.failure,
        )

        return error

    def _check_accepting(self) -> None:
        if self._coordination.aborted:
            raise PipelineAbortedError(
                f"The pipeline accepts no submissions: {self._abort_reason()}",
                failure=self.failure,
            )
        with self._state_lock:
            closing = self._closing
        if closing:
            raise ContractError("The pipeline is closed and accepts no submissions")

    def _abort_reason(self) -> str:
        failure = self.failure
        if failure is None:
            return "the pipeline was cancelled"

        reason = f"a submission failed: {type(failure).__name__}: {failure}"

        return reason

    def _full(self) -> PipelineFullError:
        self._count("full")
        error = PipelineFullError(
            f"All {self.options.max_in_flight} pipeline workers are busy"
        )

        return error

    def _count(self, outcome: str) -> None:
        with self._state_lock:
            self._outcomes[outcome] += 1


def _lock_timeout(deadline: Optional[float], *, block: bool) -> float:
    """``Lock.acquire`` timeout until ``deadline``: ``-1`` waits forever."""
    if not block:
        return 0
    if deadline is None:
        return -1

    remaining = max(deadline - time.monotonic(), 0.0)

    return remaining
