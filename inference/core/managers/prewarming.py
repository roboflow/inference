"""
Model pre-warming infrastructure for production deployments.

Solves the production issue from #2448:
- Eliminates cold starts by pre-loading models at startup
- Pins pre-warmed models to prevent eviction under memory pressure
- Gates readiness on successful pre-warming completion
- Provides metrics for observability

This directly addresses Point 2 from #2448:
"First-class model pre-warming: a way to declare the deployment's model set
to be loaded at startup (and pinned), with readiness gated on warm-up"
"""
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set

from inference.core.managers.base import ModelManager

logger = logging.getLogger(__name__)


class PrewarmStatus(Enum):
    """Status of the pre-warming process."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelPrewarmConfig:
    """Configuration for a single model to pre-warm."""

    model_id: str
    api_key: str
    model_id_alias: Optional[str] = None
    pin: bool = True  # Pin by default to prevent eviction
    required_for_readiness: bool = True  # Block readiness if fails


@dataclass
class PrewarmResult:
    """Result of pre-warming a single model."""

    model_id: str
    success: bool
    load_time_seconds: float
    error: Optional[Exception] = None
    pinned: bool = False


class ModelPrewarmingManager:
    """Manages model pre-warming at application startup.

    This class handles:
    1. Parallel loading of declared models at startup
    2. Pinning models to prevent LRU eviction
    3. Readiness gating on successful warmup
    4. Retry logic for failed loads
    5. Metrics collection for observability

    Example usage:
        ```python
        # In http_api.py startup
        prewarm_config = [
            ModelPrewarmConfig("yolov8n/1", api_key="key", pin=True),
            ModelPrewarmConfig("yolov8s/2", api_key="key", pin=True),
        ]

        prewarm_manager = ModelPrewarmingManager(model_manager, prewarm_config)
        prewarm_manager.warmup()  # Blocks until complete

        if not prewarm_manager.is_ready():
            logger.error("Pre-warming failed!")
            # Kubernetes liveness check fails, pod doesn't go ready
        ```

    Addresses production pain from #2448:
    - Eliminates 200-700 model loads/hour in steady state
    - Prevents GPU idle time during reloads
    - Makes latency predictable
    - Stops autoscaler thrashing
    """

    def __init__(
        self,
        model_manager: ModelManager,
        models_to_prewarm: List[ModelPrewarmConfig],
        max_parallel_loads: int = 4,
        retry_count: int = 2,
        retry_delay_seconds: float = 5.0,
    ):
        """Initialize the pre-warming manager.

        Args:
            model_manager: The model manager to load models into
            models_to_prewarm: List of models to pre-warm
            max_parallel_loads: Maximum concurrent model loads
            retry_count: Number of retries for failed loads
            retry_delay_seconds: Delay between retries
        """
        self._model_manager = model_manager
        self._models_to_prewarm = models_to_prewarm
        self._max_parallel_loads = max_parallel_loads
        self._retry_count = retry_count
        self._retry_delay_seconds = retry_delay_seconds

        self._status = PrewarmStatus.NOT_STARTED
        self._results: Dict[str, PrewarmResult] = {}
        self._lock = threading.Lock()

        # Track required models for readiness
        self._required_models = {
            cfg.model_id for cfg in models_to_prewarm
            if cfg.required_for_readiness
        }

        logger.info(
            f"ModelPrewarmingManager initialized: "
            f"models_to_prewarm={len(models_to_prewarm)}, "
            f"required_for_readiness={len(self._required_models)}, "
            f"max_parallel_loads={max_parallel_loads}"
        )

    def warmup(self, timeout: Optional[float] = None) -> bool:
        """Execute the pre-warming process.

        Loads all configured models in parallel, with retries for failures.
        Pins successfully loaded models to prevent eviction.

        Args:
            timeout: Maximum time to wait for all loads (seconds)

        Returns:
            True if all required models loaded successfully
        """
        if self._status != PrewarmStatus.NOT_STARTED:
            logger.warning(
                f"warmup() called but status is {self._status.value}, skipping"
            )
            return self.is_ready()

        with self._lock:
            self._status = PrewarmStatus.IN_PROGRESS

        logger.info(
            f"Starting pre-warming of {len(self._models_to_prewarm)} models "
            f"(parallel: {self._max_parallel_loads}, timeout: {timeout}s)..."
        )
        start_time = time.time()

        try:
            # Use ThreadPoolExecutor for parallel loads
            with ThreadPoolExecutor(max_workers=self._max_parallel_loads) as executor:
                futures = {
                    executor.submit(self._load_with_retry, cfg): cfg
                    for cfg in self._models_to_prewarm
                }

                completed = 0
                for future in as_completed(futures, timeout=timeout):
                    cfg = futures[future]
                    try:
                        result = future.result()
                        self._results[result.model_id] = result
                        completed += 1

                        status_emoji = "✅" if result.success else "❌"
                        logger.info(
                            f"{status_emoji} [{completed}/{len(self._models_to_prewarm)}] "
                            f"model_id={result.model_id}, "
                            f"load_time={result.load_time_seconds:.2f}s, "
                            f"pinned={result.pinned}, "
                            f"success={result.success}"
                        )

                        if not result.success and result.error:
                            logger.error(
                                f"Failed to pre-warm {result.model_id}: {result.error}",
                                exc_info=result.error,
                            )
                    except Exception as e:
                        logger.error(
                            f"Unexpected error processing pre-warm result for {cfg.model_id}: {e}",
                            exc_info=True,
                        )
                        self._results[cfg.model_id] = PrewarmResult(
                            model_id=cfg.model_id,
                            success=False,
                            load_time_seconds=0.0,
                            error=e,
                        )

        except TimeoutError:
            logger.error(
                f"Pre-warming timed out after {timeout}s - "
                f"{completed}/{len(self._models_to_prewarm)} completed"
            )

        except Exception as e:
            logger.error(f"Pre-warming failed with unexpected error: {e}", exc_info=True)

        finally:
            elapsed = time.time() - start_time
            success_count = sum(1 for r in self._results.values() if r.success)
            failure_count = len(self._results) - success_count

            is_ready = self.is_ready()
            final_status = PrewarmStatus.COMPLETED if is_ready else PrewarmStatus.FAILED

            with self._lock:
                self._status = final_status

            logger.info(
                f"Pre-warming {final_status.value}: "
                f"total_time={elapsed:.2f}s, "
                f"success={success_count}, "
                f"failed={failure_count}, "
                f"ready={is_ready}"
            )

        return self.is_ready()

    def _load_with_retry(self, config: ModelPrewarmConfig) -> PrewarmResult:
        """Load a single model with retry logic.

        Args:
            config: Configuration for the model to load

        Returns:
            Result of the load attempt (success/failure)
        """
        last_error = None

        for attempt in range(self._retry_count + 1):
            try:
                start = time.time()

                # Attempt to load the model
                self._model_manager.add_model(
                    model_id=config.model_id,
                    api_key=config.api_key,
                    model_id_alias=config.model_id_alias,
                )

                load_time = time.time() - start

                # Pin the model if requested (prevent eviction)
                pinned = False
                if config.pin and hasattr(self._model_manager, 'pin_model'):
                    try:
                        resolved_id = config.model_id_alias or config.model_id
                        self._model_manager.pin_model(resolved_id)
                        pinned = True
                        logger.debug(f"Pinned model {resolved_id} to prevent eviction")
                    except Exception as pin_error:
                        logger.warning(
                            f"Failed to pin model {config.model_id}: {pin_error}"
                        )

                return PrewarmResult(
                    model_id=config.model_id,
                    success=True,
                    load_time_seconds=load_time,
                    pinned=pinned,
                )

            except Exception as e:
                last_error = e
                if attempt < self._retry_count:
                    logger.warning(
                        f"Pre-warm attempt {attempt + 1}/{self._retry_count + 1} failed "
                        f"for {config.model_id}: {e}. Retrying in {self._retry_delay_seconds}s..."
                    )
                    time.sleep(self._retry_delay_seconds)
                else:
                    logger.error(
                        f"Pre-warm failed after {self._retry_count + 1} attempts "
                        f"for {config.model_id}: {e}"
                    )

        # All retries exhausted
        return PrewarmResult(
            model_id=config.model_id,
            success=False,
            load_time_seconds=0.0,
            error=last_error,
        )

    def is_ready(self) -> bool:
        """Check if the system is ready for traffic.

        Returns True only if all required-for-readiness models loaded successfully.
        Use this to gate Kubernetes readiness probes.

        Returns:
            True if all required models are loaded and ready
        """
        if self._status == PrewarmStatus.NOT_STARTED:
            return False

        # Check that all required models loaded successfully
        for model_id in self._required_models:
            result = self._results.get(model_id)
            if result is None or not result.success:
                return False

        return True

    def get_metrics(self) -> Dict[str, any]:
        """Get pre-warming metrics for observability.

        Returns:
            Dictionary with metrics:
            - status: current pre-warming status
            - total_models: total number of models configured
            - loaded: number successfully loaded
            - failed: number that failed
            - pinned: number pinned to prevent eviction
            - total_load_time: total time spent loading
            - ready: whether system is ready for traffic
        """
        loaded = sum(1 for r in self._results.values() if r.success)
        failed = len(self._results) - loaded
        pinned = sum(1 for r in self._results.values() if r.pinned)
        total_load_time = sum(
            r.load_time_seconds for r in self._results.values() if r.success
        )

        return {
            "status": self._status.value,
            "total_models": len(self._models_to_prewarm),
            "loaded": loaded,
            "failed": failed,
            "pinned": pinned,
            "total_load_time_seconds": round(total_load_time, 2),
            "ready": self.is_ready(),
            "results": [
                {
                    "model_id": r.model_id,
                    "success": r.success,
                    "load_time_seconds": round(r.load_time_seconds, 2),
                    "pinned": r.pinned,
                    "error": str(r.error) if r.error else None,
                }
                for r in self._results.values()
            ],
        }
