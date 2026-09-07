"""
Asynchronous model loader to prevent workflow thread pool starvation.

This module provides a dedicated thread pool executor for model loading operations,
preventing synchronous model loads (3-19s each) from blocking the shared workflow
step execution thread pool.

Related to issue #2448: Synchronous model loads causing latency collapse.
"""
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Tuple

from inference.core.env import (
    MODEL_LOADER_MAX_WORKERS,
    MODEL_LOADER_QUEUE_SIZE,
    MODEL_LOAD_TIMEOUT,
)
from inference.core.exceptions import ModelManagerLockAcquisitionError
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import ModelEndpointType

logger = logging.getLogger(__name__)


class AsyncModelLoader:
    """Manages asynchronous model loading to prevent thread pool starvation.

    This loader runs model loading operations in a dedicated thread pool,
    separate from workflow step execution. Multiple concurrent requests for
    the same model are coalesced to wait on a single load operation.

    Attributes:
        _executor: Dedicated ThreadPoolExecutor for model loading
        _pending_loads: Map of model_id -> (Future, request_count) for coalescing
        _lock: Lock protecting _pending_loads dictionary
    """

    def __init__(
        self,
        max_workers: int = MODEL_LOADER_MAX_WORKERS,
        queue_size: int = MODEL_LOADER_QUEUE_SIZE,
    ):
        """Initialize the async model loader.

        Args:
            max_workers: Number of concurrent model loading threads
            queue_size: Maximum number of queued load requests (not enforced in ThreadPoolExecutor)
        """
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="model_loader",
        )
        self._pending_loads: Dict[str, Tuple[Future, int]] = {}
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._queue_size = queue_size
        logger.info(
            f"AsyncModelLoader initialized with {max_workers} workers, "
            f"queue_size={queue_size}"
        )

    def add_model_async(
        self,
        model_manager: ModelManager,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str] = None,
        endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[Future]:
        """Load a model asynchronously if it's not already loaded.

        This method checks if the model is already loaded. If so, returns None
        immediately. If not, it either returns an existing Future for an
        in-progress load, or submits a new load operation.

        Multiple concurrent requests for the same model are coalesced to wait
        on a single load Future, preventing duplicate loads.

        Args:
            model_manager: The ModelManager to load the model into
            model_id: Model identifier
            api_key: Roboflow API key
            model_id_alias: Optional alias for the model
            endpoint_type: Model endpoint type
            countinference: Whether to count inference
            service_secret: Optional service secret

        Returns:
            None if model is already loaded, otherwise a Future that will
            complete when the model is loaded. The Future's result is None
            on success, or raises an exception on failure.

        Example:
            ```python
            future = loader.add_model_async(manager, "my-model/1", api_key)
            if future is None:
                # Model already loaded, proceed immediately
                result = model_manager.infer(...)
            else:
                # Wait for load to complete
                future.result(timeout=120.0)
                result = model_manager.infer(...)
            ```
        """
        resolved_id = model_id if model_id_alias is None else model_id_alias

        # Fast path: check if model is already loaded without locking
        if resolved_id in model_manager:
            logger.debug(f"Model {resolved_id} already loaded (fast path)")
            return None

        with self._lock:
            # Double-check under lock (TOCTOU protection)
            if resolved_id in model_manager:
                logger.debug(f"Model {resolved_id} already loaded (locked path)")
                return None

            # Check if there's already a pending load for this model
            if resolved_id in self._pending_loads:
                existing_future, count = self._pending_loads[resolved_id]
                self._pending_loads[resolved_id] = (existing_future, count + 1)
                logger.debug(
                    f"Coalescing load request for {resolved_id} "
                    f"(total waiters: {count + 1})"
                )
                return existing_future

            # Submit new load operation
            logger.info(f"Submitting async load for model {resolved_id}")
            future = self._executor.submit(
                self._load_model_with_cleanup,
                model_manager=model_manager,
                model_id=model_id,
                api_key=api_key,
                model_id_alias=model_id_alias,
                endpoint_type=endpoint_type,
                countinference=countinference,
                service_secret=service_secret,
                resolved_id=resolved_id,
            )
            self._pending_loads[resolved_id] = (future, 1)
            return future

    def _load_model_with_cleanup(
        self,
        model_manager: ModelManager,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str],
        endpoint_type: ModelEndpointType,
        countinference: Optional[bool],
        service_secret: Optional[str],
        resolved_id: str,
    ) -> None:
        """Internal method that performs the actual load and cleans up tracking.

        This runs in a loader thread and calls the synchronous add_model method.
        After completion (success or failure), it removes the entry from
        _pending_loads so future requests know the load is complete.

        Args:
            model_manager: The ModelManager to load into
            model_id: Model identifier
            api_key: Roboflow API key
            model_id_alias: Optional alias
            endpoint_type: Model endpoint type
            countinference: Whether to count inference
            service_secret: Optional service secret
            resolved_id: Resolved model identifier (model_id or alias)

        Raises:
            Any exception raised by model_manager.add_model()
        """
        try:
            logger.debug(f"Starting load for model {resolved_id} in loader thread")
            model_manager.add_model(
                model_id=model_id,
                api_key=api_key,
                model_id_alias=model_id_alias,
                endpoint_type=endpoint_type,
                countinference=countinference,
                service_secret=service_secret,
            )
            logger.info(f"Successfully loaded model {resolved_id}")
        except Exception as e:
            logger.error(f"Failed to load model {resolved_id}: {e}", exc_info=True)
            raise
        finally:
            # Clean up pending load tracking
            with self._lock:
                if resolved_id in self._pending_loads:
                    _, count = self._pending_loads[resolved_id]
                    logger.debug(
                        f"Cleaning up load tracking for {resolved_id} "
                        f"({count} waiters)"
                    )
                    del self._pending_loads[resolved_id]

    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the loader executor.

        Args:
            wait: If True, wait for all pending loads to complete
            timeout: Maximum time to wait for shutdown (seconds)
        """
        logger.info(f"Shutting down AsyncModelLoader (wait={wait})")
        if wait and timeout:
            # Python's ThreadPoolExecutor.shutdown doesn't support timeout directly
            # but we can log a warning if it takes too long
            import time

            start = time.time()
            self._executor.shutdown(wait=True)
            elapsed = time.time() - start
            if elapsed > timeout:
                logger.warning(
                    f"AsyncModelLoader shutdown took {elapsed:.1f}s "
                    f"(timeout was {timeout}s)"
                )
        else:
            self._executor.shutdown(wait=wait)

    def get_stats(self) -> Dict[str, int]:
        """Get current statistics about the loader.

        Returns:
            Dictionary with stats:
            - pending_loads: Number of models currently being loaded
            - total_waiters: Total number of requests waiting on loads
        """
        with self._lock:
            total_waiters = sum(count for _, count in self._pending_loads.values())
            return {
                "pending_loads": len(self._pending_loads),
                "total_waiters": total_waiters,
                "max_workers": self._max_workers,
            }


# Global singleton instance
_global_loader: Optional[AsyncModelLoader] = None
_global_loader_lock = threading.Lock()


def get_global_async_loader() -> AsyncModelLoader:
    """Get or create the global AsyncModelLoader singleton.

    Returns:
        The global AsyncModelLoader instance
    """
    global _global_loader
    if _global_loader is None:
        with _global_loader_lock:
            if _global_loader is None:
                _global_loader = AsyncModelLoader()
    return _global_loader


def shutdown_global_async_loader(wait: bool = True) -> None:
    """Shutdown the global async loader if it exists.

    Args:
        wait: If True, wait for all pending loads to complete
    """
    global _global_loader
    if _global_loader is not None:
        _global_loader.shutdown(wait=wait)
        _global_loader = None
