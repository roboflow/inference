"""
Helper utilities for model loading in workflow execution.

Provides non-blocking model loading to prevent thread pool starvation.
"""
import logging
from typing import Optional

from inference.core.env import MODEL_LOAD_TIMEOUT
from inference.core.managers.async_loader import get_global_async_loader
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import ModelEndpointType

logger = logging.getLogger(__name__)


def ensure_model_loaded(
    model_manager: ModelManager,
    model_id: str,
    api_key: str,
    model_id_alias: Optional[str] = None,
    endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
    countinference: Optional[bool] = None,
    service_secret: Optional[str] = None,
    timeout: float = MODEL_LOAD_TIMEOUT,
) -> None:
    """Ensure a model is loaded, using a dedicated loader executor.

    This is a drop-in replacement for model_manager.add_model() that moves
    blocking model loads (3-19s) from the shared workflow thread pool to a
    dedicated loader executor.

    IMPORTANT: This function still blocks the calling thread via future.result().
    The workflow worker remains occupied during the load. However, the actual
    blocking I/O happens in the dedicated loader pool, which:
    1. Prevents loader pool exhaustion from degrading unrelated workflows
    2. Enables load coalescing (multiple requests → single load)
    3. Provides a foundation for future async/await integration

    For truly non-blocking behavior, this would need to return the Future to
    the workflow execution engine for async handling.

    If the model is already loaded, this returns immediately (fast path).
    If not, it submits the load to the dedicated executor and waits synchronously.

    Args:
        model_manager: The ModelManager to load into
        model_id: Model identifier
        api_key: Roboflow API key
        model_id_alias: Optional alias for the model
        endpoint_type: Model endpoint type
        countinference: Whether to count inference
        service_secret: Optional service secret
        timeout: Maximum time to wait for model load (seconds)

    Raises:
        TimeoutError: If model load exceeds timeout
        Any exception raised during model loading

    Example:
        ```python
        # In a workflow block's run() method:
        ensure_model_loaded(
            self._model_manager,
            model_id=model_id,
            api_key=self._api_key,
        )
        predictions = self._model_manager.infer_from_request_sync(...)
        ```
    """
    loader = get_global_async_loader()

    # Try async load - returns None if already loaded, Future otherwise
    future = loader.add_model_async(
        model_manager=model_manager,
        model_id=model_id,
        api_key=api_key,
        model_id_alias=model_id_alias,
        endpoint_type=endpoint_type,
        countinference=countinference,
        service_secret=service_secret,
    )

    if future is None:
        # Model was already loaded, nothing to wait for
        logger.debug(f"Model {model_id} already loaded")
        return

    # Model is being loaded - wait for completion
    resolved_id = model_id if model_id_alias is None else model_id_alias
    logger.info(
        f"Waiting for async load of model {resolved_id} (timeout={timeout}s)"
    )

    try:
        future.result(timeout=timeout)
        logger.info(f"Model {resolved_id} loaded successfully")
    except TimeoutError as e:
        logger.error(
            f"Model load timed out after {timeout}s for {resolved_id}", exc_info=True
        )
        raise TimeoutError(
            f"Model {resolved_id} failed to load within {timeout}s. "
            f"This may indicate the model is very large or the system is overloaded."
        ) from e
    except Exception as e:
        logger.error(f"Model load failed for {resolved_id}: {e}", exc_info=True)
        raise
