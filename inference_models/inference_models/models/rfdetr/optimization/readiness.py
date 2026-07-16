"""Explicit asynchronous readiness handoff between RF-DETR stages."""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class PreprocessReadiness:
    """Readiness state recorded for one preprocessed tensor."""

    ready_event: Optional[torch.cuda.Event]
    input_kind: str
    implementation_id: str


class PreprocessReadinessTracker:
    """Track typed preprocessing readiness without mutating tensors."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[
            int, Tuple[weakref.ReferenceType[torch.Tensor], PreprocessReadiness]
        ] = {}

    def record(
        self,
        tensor: torch.Tensor,
        *,
        ready_event: Optional[torch.cuda.Event],
        input_kind: str,
        implementation_id: str,
    ) -> None:
        """Record readiness for a tensor returned by preprocessing.

        Args:
            tensor: Tensor passed to the protected forward stage.
            ready_event: Optional event the inference stream must await.
            input_kind: Canonical input-path description.
            implementation_id: Preprocessor that produced the tensor.
        """
        key = id(tensor)

        def discard(reference: weakref.ReferenceType[torch.Tensor]) -> None:
            with self._lock:
                current = self._states.get(key)
                if current is not None and current[0] is reference:
                    self._states.pop(key, None)

        reference = weakref.ref(tensor, discard)
        state = PreprocessReadiness(
            ready_event=ready_event,
            input_kind=input_kind,
            implementation_id=implementation_id,
        )
        with self._lock:
            self._states[key] = (reference, state)

    def consume(self, tensor: torch.Tensor) -> Optional[PreprocessReadiness]:
        """Consume readiness recorded for the exact tensor instance.

        Args:
            tensor: Tensor entering the protected forward stage.

        Returns:
            Recorded readiness, or ``None`` for externally supplied tensors.
        """
        key = id(tensor)
        with self._lock:
            current = self._states.pop(key, None)
        if current is None or current[0]() is not tensor:
            return None

        return current[1]
