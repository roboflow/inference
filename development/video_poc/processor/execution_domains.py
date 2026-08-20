"""Feature-gated execution-domain lifecycle for the video processor POC.

The only production mode today is ``in_process``. ``workspace_probe`` starts
one deliberately empty child process per workspace so staging can exercise the
supervisor lifecycle, ownership bookkeeping, and hard-exit handling before
moving ``JobRun`` and inference state across the process boundary.

The probe is *not* an isolation or security boundary: pipelines, credentials,
models, decoders, and publishers still live in the parent process. To keep the
experiment honest, no job payload or workspace identifier is sent to the child
and diagnostics expose only an opaque domain identifier and exit code.
"""

from __future__ import annotations

import enum
import multiprocessing as mp
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


def wait_for_threads(threads: list[threading.Thread], timeout: float) -> bool:
    """Wait at most one shared deadline; return whether every thread stopped."""

    deadline = time.monotonic() + max(0.0, timeout)
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(thread.is_alive() for thread in threads)


class ExecutionDomainMode(str, enum.Enum):
    IN_PROCESS = "in_process"
    WORKSPACE_PROBE = "workspace_probe"


@dataclass(frozen=True)
class ExecutionDomainFailure:
    """Credential-free notification that one execution domain died."""

    domain_id: str
    job_ids: tuple[str, ...]
    exit_code: int

    @property
    def diagnostic(self) -> str:
        return f"workspace execution probe exited unexpectedly (code {self.exit_code})"


@dataclass
class _DomainHandle:
    workspace_id: str
    process: mp.Process
    commands: Any
    jobs: set[str]


def _probe_child(commands: Any) -> None:
    """Stay alive until stopped; never receive tenant identity or job payloads."""

    while True:
        command = commands.get()
        if command == "stop":
            return
        if isinstance(command, tuple) and command[0] == "crash":
            # Fault injection is reachable only through a test helper on the
            # parent object; the worker exposes no HTTP or CLI route for it.
            import os

            os._exit(int(command[1]))


class InProcessExecutionDomains:
    """No-op manager preserving the worker's current default behavior."""

    mode = ExecutionDomainMode.IN_PROCESS
    experimental = False

    def start_job(self, job_id: str, workspace_id: Optional[str]) -> None:
        del job_id, workspace_id

    def release_job(self, job_id: str) -> None:
        del job_id

    def poll_failures(self) -> list[ExecutionDomainFailure]:
        return []

    def snapshot(self) -> dict:
        return {"mode": self.mode.value, "experimental": False, "activeDomains": 0}

    def shutdown(self, timeout: float = 0.0) -> None:
        del timeout


class WorkspaceProbeExecutionDomains:
    """One lifecycle-only child per workspace.

    The parent retains workspace and job ownership. The child sees neither;
    it receives only lifecycle commands. A later implementation can preserve
    this API while replacing ``_probe_child`` with the real runtime entrypoint.
    """

    mode = ExecutionDomainMode.WORKSPACE_PROBE
    experimental = True

    def __init__(self, start_method: str = "spawn") -> None:
        self._ctx = mp.get_context(start_method)
        self._domains: Dict[str, _DomainHandle] = {}
        self._workspace_domains: Dict[str, str] = {}
        self._job_domains: Dict[str, str] = {}
        self._lock = threading.RLock()

    def start_job(self, job_id: str, workspace_id: Optional[str]) -> None:
        job_id = str(job_id or "").strip()
        workspace_id = str(workspace_id or "").strip()
        if not job_id:
            raise ValueError("workspace execution probe requires job id")
        if not workspace_id:
            raise ValueError("workspace execution probe requires workspace id")
        with self._lock:
            if job_id in self._job_domains:
                raise ValueError("job is already assigned to an execution domain")
            domain_id = self._workspace_domains.get(workspace_id)
            handle = self._domains.get(domain_id) if domain_id else None
            if handle is None:
                # The opaque id is safe for logs/status and cannot disclose the
                # workspace name. Workspace identity stays in the parent map.
                domain_id = f"domain-{uuid.uuid4().hex[:12]}"
                commands = self._ctx.Queue()
                process = self._ctx.Process(
                    target=_probe_child,
                    args=(commands,),
                    name=f"video-{domain_id}",
                    daemon=False,
                )
                process.start()
                handle = _DomainHandle(
                    workspace_id=workspace_id,
                    process=process,
                    commands=commands,
                    jobs=set(),
                )
                self._domains[domain_id] = handle
                self._workspace_domains[workspace_id] = domain_id
            elif not handle.process.is_alive():
                raise RuntimeError("workspace execution probe is not alive")
            handle.jobs.add(job_id)
            self._job_domains[job_id] = domain_id

    def release_job(self, job_id: str) -> None:
        with self._lock:
            domain_id = self._job_domains.pop(str(job_id), None)
            if domain_id is None:
                return
            handle = self._domains.get(domain_id)
            if handle is None:
                return
            handle.jobs.discard(str(job_id))
            if not handle.jobs:
                self._stop_domain_locked(domain_id, handle)

    def poll_failures(self) -> list[ExecutionDomainFailure]:
        failures = []
        with self._lock:
            for domain_id, handle in list(self._domains.items()):
                if handle.process.is_alive() or handle.process.exitcode is None:
                    continue
                job_ids = tuple(sorted(handle.jobs))
                exit_code = int(handle.process.exitcode)
                self._remove_domain_locked(domain_id, handle)
                if job_ids:
                    failures.append(
                        ExecutionDomainFailure(domain_id, job_ids, exit_code)
                    )
        return failures

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode.value,
                "experimental": True,
                "activeDomains": sum(
                    1 for handle in self._domains.values() if handle.process.is_alive()
                ),
            }

    def crash_workspace_for_test(
        self, workspace_id: str, exit_code: int = 91
    ) -> None:
        """Kill one probe child; intentionally not wired to the worker API."""

        with self._lock:
            domain_id = self._workspace_domains[str(workspace_id)]
            self._domains[domain_id].commands.put(("crash", int(exit_code)))

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            # Serialize teardown with release_job(). Process and Queue handles
            # are closed exactly once while ownership remains under this lock.
            for domain_id, handle in list(self._domains.items()):
                if handle.process.is_alive():
                    handle.commands.put("stop")
                    handle.process.join(timeout=max(0.0, timeout))
                if handle.process.is_alive():
                    handle.process.terminate()
                    handle.process.join(timeout=1.0)
                self._remove_domain_locked(domain_id, handle)

    def _stop_domain_locked(self, domain_id: str, handle: _DomainHandle) -> None:
        if handle.process.is_alive():
            handle.commands.put("stop")
            handle.process.join(timeout=1.0)
        if handle.process.is_alive():
            handle.process.terminate()
            handle.process.join(timeout=1.0)
        self._remove_domain_locked(domain_id, handle)

    def _remove_domain_locked(
        self, domain_id: str, handle: _DomainHandle
    ) -> None:
        self._domains.pop(domain_id, None)
        if self._workspace_domains.get(handle.workspace_id) == domain_id:
            self._workspace_domains.pop(handle.workspace_id, None)
        for job_id in tuple(handle.jobs):
            if self._job_domains.get(job_id) == domain_id:
                self._job_domains.pop(job_id, None)
        try:
            handle.commands.close()
            handle.commands.join_thread()
        except (OSError, ValueError):
            pass
        try:
            handle.process.close()
        except (OSError, ValueError):
            pass

    def __enter__(self) -> "WorkspaceProbeExecutionDomains":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


def build_execution_domains(
    mode: Optional[str], start_method: str = "spawn"
) -> InProcessExecutionDomains | WorkspaceProbeExecutionDomains:
    normalized = str(mode or ExecutionDomainMode.IN_PROCESS.value).strip().lower()
    try:
        parsed = ExecutionDomainMode(normalized)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ExecutionDomainMode)
        raise ValueError(
            f"invalid PROCESSOR_EXECUTION_DOMAIN_MODE; expected one of: {choices}"
        ) from exc
    if parsed is ExecutionDomainMode.IN_PROCESS:
        return InProcessExecutionDomains()
    return WorkspaceProbeExecutionDomains(start_method=start_method)
