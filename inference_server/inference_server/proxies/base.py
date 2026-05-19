"""ModelManagerProxy — L3 backend-adapter interface.

Unifies in-process `inference_model_manager.ModelManager` and the orchestrated
`ModelManagerProcess` (ZMQ + SHM) behind a single Protocol. L2 (framework
dispatcher, routers) holds a proxy and never branches on deployment mode.
L1 (per-model handlers) sees only this interface.

Why a proxy on both sides (not just orchestrated):
  - Single place to catch up when ModelManager or ModelManagerProcess change.
  - Server-only concerns (timing CSV, request-disconnect race, error mapping)
    sit at the boundary without polluting ModelManager.
  - Future server-only methods can be added to the proxy without touching
    inference_model_manager.

Hot-path return types are tuples (not dataclasses) to keep per-request
allocation cheap and to match today's `state.py` convention:
    ensure_loaded → ("model_ready",) | ("load_timeout", retry_after_s)
                  | ("error", code)
    load          → ("ok",) | ("error", code)
    unload        → ("ok",) | ("error", code)
    infer         → raw prediction object (or list for batch); errors raise

Errors on `infer` raise exceptions (TimeoutError, RuntimeError,
ClientDisconnected, MemoryError-on-payload-too-large) so the framework
layer maps them to HTTP status codes in one place.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from fastapi import Request


@runtime_checkable
class ModelManagerProxy(Protocol):
    """Backend-adapter contract. Two implementations:

    - `DirectModelManagerProxy` — wraps `ModelManager` directly. In-process,
      no SHM, no ZMQ. Used by Workflows / InferencePipeline / dev mode.
    - `OrchestratedModelManagerProxy` — wraps the FastAPI-worker side of MMP:
      DEALER socket + SHMPool attach + pending-futures recv loop. Used in
      production multi-worker uvicorn deployments.

    Both expose the same methods; callers depend only on this Protocol.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def ensure_loaded(
        self,
        model_id: str,
        instance: str = "",
        api_key: str = "",
        device: str = "",
    ) -> tuple:
        """Block until model is loaded, or return load_timeout.

        Returns:
            ("model_ready",)
            ("load_timeout", retry_after_s)
            ("error", code)
        """
        ...

    async def load(self, model_id: str, api_key: str = "") -> tuple:
        """Explicit load (POST /v2/models/load).

        Returns ("ok",) | ("error", code).
        """
        ...

    async def unload(self, model_id: str) -> tuple:
        """Explicit unload (POST /v2/models/unload).

        Returns ("ok",) | ("error", code).
        """
        ...

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    async def infer(
        self,
        *,
        model_id: str,
        image: bytes,
        task: Optional[str] = None,
        instance: str = "",
        params: Optional[dict] = None,
        request: Optional[Request] = None,
    ) -> Any:
        """Run inference on a single image. Returns the raw prediction
        object emitted by the backend model (e.g. `Detections`,
        `ClassificationPrediction`, str for text models, np.ndarray for
        embeddings). The framework's output_serializer dispatches on type.

        Batch is handled by the framework fanning out N concurrent `infer`
        calls; the proxy stays single-image.

        `request` is forwarded so the orchestrated impl can race inference
        against client disconnect; direct impl ignores it.

        Raises:
            asyncio.TimeoutError — inference exceeded timeout.
            RuntimeError         — alloc failed, backend died, deserialization failed.
            ClientDisconnected   — client gone during await (orchestrated only).
        """
        ...

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def stats(self) -> dict:
        """Full backend stats — loaded models, per-model state/device/queue,
        GPU usage. Used by `/v2/server/{info,metrics,ready}` and to look up
        `(model_id) → (model_type, task)` after first load.
        """
        ...

    async def interface(self, model_id: str) -> dict:
        """Per-model interface — tasks, param schemas, output schemas.
        Used by `/v2/models/interface`. Requires model to be loaded.
        """
        ...


class ClientDisconnected(Exception):
    """Raised by `infer` when the HTTP client disconnects mid-flight.

    Orchestrated proxy detects this via `request.is_disconnected()` poll
    racing the inference future. Direct proxy never raises this.
    """
