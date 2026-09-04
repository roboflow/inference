"""Small spawn-based supervisor used to de-risk video job isolation.

This module intentionally does not import the current POC worker.  It proves the
process lifecycle and grouping semantics in isolation, so the production worker
can later wrap ``JobRun`` behind the same parent/child boundary without making
the first experiment a large refactor.

The parent owns claims, access tokens, platform heartbeats and public HTTP.  A
child owns only execution state for either one job or one workspace.  CUDA must
never be initialized in the parent; ``spawn`` is mandatory.
"""

from __future__ import annotations

import enum
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional


class IsolationMode(str, enum.Enum):
    JOB = "job"
    WORKSPACE = "workspace"


class CommandType(str, enum.Enum):
    START = "start"
    CANCEL = "cancel"
    CRASH = "crash"
    STOP = "stop"


class EventType(str, enum.Enum):
    READY = "ready"
    STARTED = "started"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXITED = "exited"


@dataclass(frozen=True)
class JobDescriptor:
    job_id: str
    workspace_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class ChildEvent:
    event_type: EventType
    isolation_key: str
    job_id: Optional[str] = None
    detail: Optional[str] = None
    pid: Optional[int] = None


@dataclass
class _ChildHandle:
    process: mp.Process
    commands: Any
    jobs: set[str]


def isolation_key(job: JobDescriptor, mode: IsolationMode) -> str:
    """Return a key without putting credentials or full payloads in metadata."""

    if mode is IsolationMode.JOB:
        return f"job:{job.job_id}"
    if not job.workspace_id:
        raise ValueError("workspace isolation requires workspace_id")
    return f"workspace:{job.workspace_id}"


def _child_main(
    key: str,
    commands: Any,
    events: Any,
    runner: Callable[[JobDescriptor], None],
) -> None:
    """Run commands for one isolation domain.

    The real worker integration will replace ``runner`` with a workspace runtime
    that creates ``InferencePipeline`` instances and an MMP client.  Exceptions
    are contained here and reported without taking down sibling processes.
    """

    jobs: set[str] = set()
    events.put(ChildEvent(EventType.READY, key, pid=os.getpid()))
    try:
        while True:
            command, value = commands.get()
            command = CommandType(command)
            if command is CommandType.STOP:
                break
            if command is CommandType.CRASH:
                os._exit(int(value or 91))
            if command is CommandType.CANCEL:
                job_id = str(value)
                if job_id in jobs:
                    jobs.remove(job_id)
                    events.put(
                        ChildEvent(
                            EventType.CANCELLED,
                            key,
                            job_id=job_id,
                            pid=os.getpid(),
                        )
                    )
                continue
            if command is CommandType.START:
                job = value
                try:
                    runner(job)
                # The child boundary must contain runner failures.
                except BaseException as exc:
                    events.put(
                        ChildEvent(
                            EventType.FAILED,
                            key,
                            job_id=job.job_id,
                            detail=f"{type(exc).__name__}: {exc}",
                            pid=os.getpid(),
                        )
                    )
                else:
                    jobs.add(job.job_id)
                    events.put(
                        ChildEvent(
                            EventType.STARTED, key, job_id=job.job_id, pid=os.getpid()
                        )
                    )
    finally:
        events.put(ChildEvent(EventType.EXITED, key, pid=os.getpid()))


def _noop_runner(job: JobDescriptor) -> None:
    del job


class ProcessSupervisor:
    """Own isolated child processes and detect abnormal exits.

    This is deliberately synchronous and compact.  The live worker can poll
    ``events()`` from its existing heartbeat loop, preserving one source of
    truth for cancellation and failure reporting.
    """

    def __init__(
        self,
        mode: IsolationMode = IsolationMode.WORKSPACE,
        runner: Callable[[JobDescriptor], None] = _noop_runner,
        start_method: str = "spawn",
    ) -> None:
        self.mode = IsolationMode(mode)
        self._runner = runner
        self._ctx = mp.get_context(start_method)
        self._events = self._ctx.Queue()
        self._children: Dict[str, _ChildHandle] = {}
        self._jobs: Dict[str, str] = {}
        self._reported_dead: set[str] = set()

    def start(self, job: JobDescriptor) -> str:
        if job.job_id in self._jobs:
            raise ValueError(f"job {job.job_id!r} is already supervised")
        key = isolation_key(job, self.mode)
        handle = self._children.get(key)
        if handle is None or not handle.process.is_alive():
            commands = self._ctx.Queue()
            process = self._ctx.Process(
                target=_child_main,
                args=(key, commands, self._events, self._runner),
                name=f"video-{key}",
                daemon=False,
            )
            process.start()
            handle = _ChildHandle(process=process, commands=commands, jobs=set())
            self._children[key] = handle
            self._reported_dead.discard(key)
        handle.jobs.add(job.job_id)
        self._jobs[job.job_id] = key
        handle.commands.put((CommandType.START.value, job))
        return key

    def cancel(self, job_id: str) -> None:
        key = self._jobs.get(job_id)
        if key is None:
            return
        handle = self._children.get(key)
        if handle is not None and handle.process.is_alive():
            handle.commands.put((CommandType.CANCEL.value, job_id))
        self._forget_job(job_id, key)

    def crash_for_test(self, isolation_domain: str, exit_code: int = 91) -> None:
        """Deliberate fault injection for staging and unit tests only."""

        handle = self._children[isolation_domain]
        handle.commands.put((CommandType.CRASH.value, exit_code))

    def events(self, timeout: float = 0.0) -> list[ChildEvent]:
        result: list[ChildEvent] = []
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                event = self._events.get(timeout=remaining if timeout else 0)
            except queue.Empty:
                break
            result.append(event)
            if event.event_type in (EventType.CANCELLED, EventType.FAILED):
                if event.job_id is not None:
                    self._forget_job(event.job_id, event.isolation_key)
            if timeout == 0 or time.monotonic() >= deadline:
                break

        # A hard exit cannot publish an EXITED event, so synthesize one failure
        # per affected job.  Sibling domains remain untouched.
        for key, handle in list(self._children.items()):
            if handle.process.is_alive() or handle.process.exitcode is None:
                continue
            if key in self._reported_dead:
                continue
            self._reported_dead.add(key)
            for job_id in sorted(handle.jobs):
                result.append(
                    ChildEvent(
                        EventType.FAILED,
                        key,
                        job_id=job_id,
                        detail=(
                            "execution process exited with code "
                            f"{handle.process.exitcode}"
                        ),
                        pid=handle.process.pid,
                    )
                )
                self._forget_job(job_id, key)
        return result

    def active_domains(self) -> Dict[str, set[str]]:
        return {
            key: set(handle.jobs)
            for key, handle in self._children.items()
            if handle.process.is_alive()
        }

    def shutdown(self, timeout: float = 5.0) -> None:
        for handle in self._children.values():
            if handle.process.is_alive():
                handle.commands.put((CommandType.STOP.value, None))
        deadline = time.monotonic() + timeout
        for handle in self._children.values():
            handle.process.join(timeout=max(0.0, deadline - time.monotonic()))
            if handle.process.is_alive():
                handle.process.terminate()
                handle.process.join(timeout=1.0)
        self._children.clear()
        self._jobs.clear()

    def _forget_job(self, job_id: str, key: str) -> None:
        self._jobs.pop(job_id, None)
        handle = self._children.get(key)
        if handle is not None:
            handle.jobs.discard(job_id)

    def __enter__(self) -> "ProcessSupervisor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
