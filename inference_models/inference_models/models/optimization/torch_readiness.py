"""Weakly referenced state handoff for exact torch tensor instances."""

from __future__ import annotations

import threading
import weakref
from typing import Dict, Generic, Optional, Tuple, TypeVar

import torch

StateT = TypeVar("StateT")


class TensorReadinessTracker(Generic[StateT]):
    """Associate one-shot readiness state with exact tensor instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[int, Tuple[weakref.ReferenceType[torch.Tensor], StateT]] = {}

    def record(self, tensor: torch.Tensor, *, state: StateT) -> None:
        """Record state for a tensor without mutating the tensor.

        Args:
            tensor: Tensor passed between asynchronous inference stages.
            state: Model-specific readiness state associated with the tensor.
        """
        key = id(tensor)

        def discard(reference: weakref.ReferenceType[torch.Tensor]) -> None:
            with self._lock:
                current = self._states.get(key)
                if current is not None and current[0] is reference:
                    self._states.pop(key, None)

        reference = weakref.ref(tensor, discard)
        with self._lock:
            self._states[key] = (reference, state)

    def consume(self, tensor: torch.Tensor) -> Optional[StateT]:
        """Consume state recorded for the exact tensor instance.

        Args:
            tensor: Tensor entering the consuming stage.

        Returns:
            Recorded state, or ``None`` when the tensor has no associated state.
        """
        key = id(tensor)
        with self._lock:
            current = self._states.pop(key, None)
        if current is None or current[0]() is not tensor:
            return None

        return current[1]
