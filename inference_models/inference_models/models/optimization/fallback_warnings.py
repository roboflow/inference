"""Thread-safe warning de-duplication for compatibility fallbacks."""

import threading
from typing import Optional, Set, Tuple

from inference_models.models.optimization.contracts import OptimizationStage

FallbackWarningKey = Tuple[OptimizationStage, str, str, Optional[str]]


class FallbackWarningTracker:
    """Claim each distinct fallback warning once per tracker instance."""

    def __init__(self) -> None:
        self._claimed: Set[FallbackWarningKey] = set()
        self._lock = threading.Lock()

    def claim(
        self,
        *,
        stage: OptimizationStage,
        requested_id: str,
        effective_id: str,
        reason: Optional[str],
    ) -> bool:
        """Return whether the caller should emit this fallback warning.

        Args:
            stage: Inference stage that followed a fallback.
            requested_id: Originally selected implementation ID.
            effective_id: Implementation ID used after fallback.
            reason: Compatibility reason that caused the fallback.

        Returns:
            ``True`` for the first claim of a distinct warning key and
            ``False`` for subsequent claims.
        """
        key = (stage, requested_id, effective_id, reason)
        with self._lock:
            if key in self._claimed:
                return False
            self._claimed.add(key)

        return True
