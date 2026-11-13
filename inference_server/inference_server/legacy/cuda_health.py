"""CUDA health checking utilities.

Provides a fast, cached health check for GPU/CUDA state. Once CUDA fails,
the context is permanently corrupted and cannot recover without process restart.
The failure state is cached to avoid repeatedly calling into a broken CUDA runtime.
"""

import logging
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CudaHealthChecker:
    """Thread-safe CUDA health checker with failure caching.

    Once a CUDA failure is detected, the result is cached permanently
    (CUDA context corruption is unrecoverable). Subsequent calls return
    the cached failure immediately without touching CUDA.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cuda_failed: bool = False
        self._failure_error: Optional[str] = None
        self._failure_time: Optional[float] = None
        self._gpu_available: Optional[bool] = None  # None = not yet checked

    def _is_gpu_environment(self) -> bool:
        """Check if we're running in a GPU environment. Cached after first call."""
        if self._gpu_available is not None:
            return self._gpu_available
        try:
            import torch

            self._gpu_available = torch.cuda.is_available()
        except ImportError:
            self._gpu_available = False
        except Exception:
            self._gpu_available = False
        return self._gpu_available

    def check_health(self) -> Tuple[bool, Optional[str]]:
        """Check CUDA health. Returns (is_healthy, error_message).

        - If not a GPU environment: returns (True, None) immediately
        - If CUDA previously failed: returns cached failure immediately
        - Otherwise: runs synchronize + mem_get_info check

        Thread-safe. The actual CUDA check is serialized by the lock to
        prevent concurrent CUDA calls during health checking.
        """
        # Fast path: not a GPU environment
        if not self._is_gpu_environment():
            return True, None

        # Fast path: already known to be failed (unrecoverable)
        if self._cuda_failed:
            return False, self._failure_error

        # Slow path: actually check CUDA
        with self._lock:
            # Double-check after acquiring lock
            if self._cuda_failed:
                return False, self._failure_error

            try:
                import torch

                # Synchronize to surface any pending async CUDA errors
                torch.cuda.synchronize()
                # Query runtime to verify it's still functional
                torch.cuda.mem_get_info()
                return True, None
            except Exception as e:
                error_msg = f"CUDA health check failed: {e}"
                logger.error(error_msg)
                self._cuda_failed = True
                self._failure_error = error_msg
                self._failure_time = time.time()
                return False, error_msg

    @property
    def is_failed(self) -> bool:
        return self._cuda_failed

    @property
    def failure_info(self) -> Optional[dict]:
        if not self._cuda_failed:
            return None
        return {
            "error": self._failure_error,
            "failed_at": self._failure_time,
        }


# Module-level singleton
_checker = CudaHealthChecker()


def check_cuda_health() -> Tuple[bool, Optional[str]]:
    """Module-level convenience function."""
    return _checker.check_health()


def get_cuda_health_checker() -> CudaHealthChecker:
    """Return the singleton for dependency injection / testing."""
    return _checker
