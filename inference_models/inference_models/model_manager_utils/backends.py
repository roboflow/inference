from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Handle to a single loaded model.

    One instance per model. Loading happens in __init__ (blocks until ready).
    ModelManager owns a dict of these and routes calls by model_id.
    """

    @abstractmethod
    def unload(self) -> None:
        """Release all resources held by this model."""
        ...

    @abstractmethod
    async def infer_async(self, request: Any, **kwargs) -> Any: ...

    @abstractmethod
    def infer_sync(self, request: Any, **kwargs) -> Any: ...


class InProcessBackend(Backend):
    """Loads and runs a model in the current process.

    Sync calls block the calling thread directly. Async calls dispatch to a
    shared thread-pool executor to avoid blocking the event loop.

    Preferred for: InferencePipeline, CPU-only / constrained hardware — any
    scenario where IPC round-trip cost per frame is unacceptable.
    """

    def __init__(self, model_id: str, api_key: str, **kwargs) -> None:
        """Loads the model in the current process."""
        ...

    def unload(self) -> None: ...

    async def infer_async(self, request: Any, **kwargs) -> Any:
        """Dispatches blocking inference to a shared thread-pool executor."""
        ...

    def infer_sync(self, request: Any, **kwargs) -> Any: ...


class MultiProcessBackend(Backend):
    """Loads a model in a dedicated spawned worker process and forwards
    inference requests over IPC (ZeroMQ or multiprocessing.Queue).

    Uses start_method="spawn" — fork is not viable because CUDA cannot be
    used after fork().

    Preferred for: HTTP server, GPU machines, multi-worker deployments.
    """

    def __init__(self, model_id: str, api_key: str, **kwargs) -> None:
        """Spawns the worker process, passes config, blocks until ready signal."""
        ...

    def unload(self) -> None:
        """Sends shutdown message to the worker process and joins it."""
        ...

    async def infer_async(self, request: Any, **kwargs) -> Any:
        """Serializes request, sends to worker via IPC, awaits result."""
        ...

    def infer_sync(self, request: Any, **kwargs) -> Any:
        """Serializes request, sends to worker via IPC, blocks for result."""
        ...
