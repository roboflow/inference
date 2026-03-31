from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Backend(ABC):
    """Handle to a single loaded model.

    One instance per model. Loading happens in __init__ (blocks until ready).
    ModelManager owns a dict of these and routes calls by model_id.
    """

    @property
    @abstractmethod
    def class_names(self) -> Optional[List[str]]:
        """Class names for the loaded model, if available."""
        ...

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """Whether this backend is in a usable state.

        For DirectBackend: whether the model object exists and didn't OOM.
        For SubprocessBackend: whether the worker process is alive and responsive.
        """
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> Optional[int]:
        """Maximum batch size this backend supports, or None if unknown.

        Used by the adaptive batcher (when added) to cap batch formation.
        None means the batcher should use its own default / the model's
        reported limit.
        """
        ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Return runtime statistics for observability.

        Expected keys (backends may add more):
            inference_count: total infer calls served
            total_latency_s:  cumulative inference wall time
            last_latency_s:   wall time of the most recent call
            errors:           total error count
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all resources held by this model."""
        ...

    @abstractmethod
    async def infer_async(self, *args, **kwargs) -> Any: ...

    @abstractmethod
    def infer_sync(self, *args, **kwargs) -> Any: ...
