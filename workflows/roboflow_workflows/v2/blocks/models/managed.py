"""Opt-in host ownership of pinned, in-process models for native V2 phases.

The optional manager package is imported only when ``ManagedModels`` is created.
The existing ``ModelCache`` remains the default. This owner is not an evictable
per-call lease: callers must finish all external output consumers before close.
"""

import hashlib
import json
import math
import re
import threading
import time
import uuid
import weakref
from typing import Any, Dict, Mapping, Optional, Tuple

from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest


def _new_manager() -> Any:
    try:
        from inference_model_manager.model_manager import ModelManager
    except ImportError as error:
        raise ImportError(
            "ManagedModels needs the optional inference-model-manager package "
            "and its runtime dependencies; the default ModelCache does not."
        ) from error

    return ModelManager()


class _ManagedCache(ModelCache):
    """Keep the existing resource type without retaining raw models on close."""

    def __init__(self, owner: "ManagedModels") -> None:
        super().__init__()
        self._owner = weakref.ref(owner)

    def get(self, request: ModelRequest) -> Any:
        owner = self._owner()
        if owner is None:
            raise RuntimeError("The managed models owner is no longer available")

        return owner._get(request)

    def supplied_requests(self) -> Tuple[ModelRequest, ...]:
        return ()


class ManagedModels:
    """Own pinned direct models and the V2 sessions that use them.

    One owner fixes credential, device and namespace policy. It never shares a
    manager with another owner, never evicts its models, and never dispatches
    inference through the manager. Blocks retain their native phase calls.

    Args:
        api_key: Host credential used for all model loads. Never put it in
            workflow JSON or use one owner across different credential policies.
        device: Explicit model device, such as ``"cpu"`` or ``"cuda:0"``.
            CUDA devices require an index; thread-relative ``"cuda"`` is refused.
        allow_untrusted_packages: Explicit host opt-in for packages whose source
            is not marked trusted. Defaults to False; never set from workflow JSON.
        namespace: Host ownership label. Separate owners always get separate
            routing keys, even when this label is identical.

    Raises:
        ValueError: When device or namespace is empty, CUDA has no valid index,
            or api_key is not a string.
        TypeError: When allow_untrusted_packages is not a bool.
        ImportError: When the optional manager or its dependencies are unavailable.

    Notes:
        Use ``create_session`` to track every session. Before ``close``, stop and
        wait active runs, finish passive/direct calls, and finish external GPU
        consumers of their outputs. Session closure alone does not prove those
        external consumers finished. Pinned models can exceed the manager's
        capacity setting; this owner is not a memory admission controller.
    """

    def __init__(
        self,
        *,
        api_key: str,
        device: str,
        namespace: str = "workflows",
        allow_untrusted_packages: bool = False,
    ):
        if not isinstance(api_key, str):
            raise ValueError("api_key must be a host-provided string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be an explicit non-empty string")
        if device != device.strip():
            raise ValueError("device must not contain surrounding whitespace")
        if (device == "cuda" or device.startswith("cuda:")) and not re.fullmatch(
            r"cuda:(0|[1-9][0-9]*)", device
        ):
            raise ValueError("CUDA device must have an explicit index, such as cuda:0")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")

        if not isinstance(allow_untrusted_packages, bool):
            raise TypeError("allow_untrusted_packages must be a bool")

        self._allow_untrusted_packages = allow_untrusted_packages
        self._api_key = api_key
        self._device = device
        self._namespace = namespace
        self._identity = uuid.uuid4().hex
        self._condition = threading.Condition(threading.RLock())
        self._state = "open"
        self._requests = 0
        self._creating = 0
        self._sessions = []
        self._manager = _new_manager()
        self._cache: Optional[ModelCache] = ModelCache(load=self._load)
        self._resource = _ManagedCache(self)

    def preload(self, request: ModelRequest) -> Any:
        """Load an exact request once and return its native model for inspection.

        Args:
            request: Model id and optional exact package id.

        Returns:
            The same native model object used by the owner's V2 sessions.

        Raises:
            RuntimeError: When the owner is closing or closed.
            Exception: When loading fails; a later call may retry.

        Notes:
            Do not use returned models after closing the owner. Retaining them
            can retain their memory after close. Inspection does not create an
            independently managed inference lease.
        """
        with self._condition:
            self._require_open()
            self._requests += 1

        try:
            return self._cache.get(request)
        finally:
            self._request_finished()

    def create_session(
        self,
        plan: Any,
        *,
        resources: Optional[Mapping[str, Any]] = None,
        **session_options: Any,
    ) -> Any:
        """Create and track a native V2 session with this owner's model resource.

        Args:
            plan: Compiled V2 plan exposing ``create_session``.
            resources: Other host resources. Model-resource overrides are refused.
            **session_options: Forwarded unchanged to the plan's session factory.

        Returns:
            The native ExecutionSession, without an inference proxy.

        Raises:
            ValueError: When resources would override a models resource.
            RuntimeError: When the owner is closing or closed.
            Exception: When the plan cannot construct the session.
        """
        supplied: Dict[str, Any] = dict(resources or {})
        if any(key == "models" or key.endswith(".models") for key in supplied):
            raise ValueError("ManagedModels owns the session's models resource")
        supplied["models"] = self._resource

        with self._condition:
            self._require_open()
            self._creating += 1

        try:
            session = plan.create_session(resources=supplied, **session_options)
            with self._condition:
                self._sessions.append(session)
            return session
        finally:
            with self._condition:
                self._creating -= 1
                self._condition.notify_all()

    def close(self, *, timeout: float = 30.0) -> None:
        """Close tracked sessions, wait pending loads, then shut down the manager.

        Args:
            timeout: Maximum seconds waiting for session construction or pending
                model loads. Does not impose a timeout on native session cleanup
                or the manager's shutdown implementation.

        Raises:
            ValueError: When timeout is negative or not finite.
            TimeoutError: When construction or model loading remains unfinished.
            RuntimeError: When another close is already in progress.
            Exception: When a session or manager cannot close. No manager shutdown
                follows a failed session close. Some sessions may already be
                closed; resolve the remaining work and retry close.

        Notes:
            Active runs must be stopped and waited by their host first. An active
            session's close refusal leaves models available to that running work.
            This method cannot track arbitrary external consumers of GPU outputs.
        """
        if (
            not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._state == "closed":
                return
            if self._state not in ("open", "shutdown_failed"):
                raise RuntimeError("ManagedModels close is already in progress")
            retry_shutdown = self._state == "shutdown_failed"
            self._state = "closing_sessions"

        shutdown_started = False
        try:
            with self._condition:
                self._wait_for(lambda: self._creating == 0, deadline=deadline)
                sessions = list(self._sessions)

            for session in sessions:
                session.close()

            with self._condition:
                self._state = "closing_loads"
                self._wait_for(lambda: self._requests == 0, deadline=deadline)

            shutdown_started = True
            self._manager.shutdown()
            with self._condition:
                self._cache = None
                self._sessions.clear()
                self._api_key = ""
                self._state = "closed"
        except BaseException:
            with self._condition:
                self._state = (
                    "shutdown_failed" if shutdown_started or retry_shutdown else "open"
                )
            raise
        finally:
            with self._condition:
                self._condition.notify_all()

    def __enter__(self) -> "ManagedModels":
        """Return the open owner for a host lifetime context."""
        with self._condition:
            self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close after the host has finished runs and external output consumers."""
        self.close()

    def _require_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("ManagedModels is closing or closed")

    def _get(self, request: ModelRequest) -> Any:
        with self._condition:
            # A refused close must not disrupt existing active sessions.
            if self._state not in ("open", "closing_sessions"):
                raise RuntimeError("ManagedModels is closing or closed")
            self._requests += 1

        try:
            return self._cache.get(request)
        finally:
            self._request_finished()

    def _request_finished(self) -> None:
        with self._condition:
            self._requests -= 1
            self._condition.notify_all()

    def _wait_for(self, predicate: Any, *, deadline: float) -> None:
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "ManagedModels still has unfinished construction or loads"
                )
            self._condition.wait(remaining)

    def _load(self, request: ModelRequest) -> Any:
        identity = json.dumps(
            [
                self._identity,
                self._namespace,
                self._device,
                request.model_id,
                request.model_package_id,
            ]
        ).encode("utf-8")
        route = "workflow-" + hashlib.sha256(identity).hexdigest()
        try:
            self._manager.load(
                route,
                api_key=self._api_key,
                model_id_or_path=request.model_id,
                model_package_id=request.model_package_id,
                backend="direct",
                device=self._device,
                pinned=True,
                warmup_iters=0,
                allow_untrusted_packages=self._allow_untrusted_packages,
            )
            backend = self._manager.get_backend(route)
            model = getattr(backend, "model", None)
            if model is None:
                raise RuntimeError(
                    "The direct manager did not expose a loaded native model"
                )
            return model
        except BaseException:
            if self._manager.get_backend(route) is not None:
                self._manager.unload(route, drain=True)
            raise
