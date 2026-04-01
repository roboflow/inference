import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from inference_models.backends.base import Backend


class DirectBackend(Backend):
    """Loads and runs a model in the current process.

    Sync calls block the calling thread directly. Async calls dispatch to a
    shared thread-pool executor to avoid blocking the event loop.

    Preferred for: InferencePipeline, CPU-only / constrained hardware — any
    scenario where IPC round-trip cost per frame is unacceptable.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str,
        *,
        executor: Optional[ThreadPoolExecutor] = None,
        **kwargs,
    ) -> None:
        """Loads the model in the current process.

        Args:
            executor: Shared thread-pool executor for ``infer_async``.
                      Owned by ModelManager, not by this backend.
                      None is fine if only ``infer_sync`` will be used
                      (e.g. InferencePipeline).
        """
        from inference_models.models.auto_loaders.core import AutoModel

        self._executor = executor
        self._model = AutoModel.from_pretrained(model_id, api_key=api_key, **kwargs)
        self._model_id = model_id

        self._inference_count = 0
        self._total_latency_s = 0.0
        self._last_latency_s = 0.0
        self._avg_latency_s = 0.0
        self._ram_usage_mb = 0.0
        self._gpu_usage_mb = 0.0
        self._error_count = 0

    @property
    def class_names(self) -> Optional[List[str]]:
        return getattr(self._model, "class_names", None)

    @property
    def is_healthy(self) -> bool:
        return self._model is not None

    @property
    def max_batch_size(self) -> Optional[int]:
        return getattr(self._model, "_max_batch_size", None)

    def stats(self) -> Dict[str, Any]:
        return {
            "inference_count": self._inference_count,
            "total_latency_s": self._total_latency_s,
            "last_latency_s": self._last_latency_s,
            "errors": self._error_count,
        }

    def unload(self) -> None:
        del self._model
        self._model = None

    async def infer_async(self, *args, **kwargs) -> Any:
        """Dispatches blocking inference to a shared thread-pool executor."""
        if self._executor is None:
            raise RuntimeError(
                f"infer_async called on DirectBackend('{self._model_id}') "
                f"but no executor was provided — use infer_sync or pass an "
                f"executor at construction"
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(self.infer_sync, *args, **kwargs),
        )

    def infer_sync(self, *args, **kwargs) -> Any:
        t0 = time.monotonic()
        try:
            result = self._model.infer(*args, **kwargs)
            return result
        except Exception:
            self._error_count += 1
            raise
        finally:
            elapsed = time.monotonic() - t0
            self._inference_count += 1
            self._total_latency_s += elapsed
            self._last_latency_s = elapsed
