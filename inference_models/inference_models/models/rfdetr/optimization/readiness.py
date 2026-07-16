"""RF-DETR readiness state carried by the shared tensor-state tracker."""

from dataclasses import dataclass
from typing import Optional

import torch

from inference_models.models.optimization.torch_readiness import TensorReadinessTracker


@dataclass(frozen=True)
class PreprocessReadiness:
    """Readiness state recorded for one preprocessed tensor."""

    ready_event: Optional[torch.cuda.Event]
    input_kind: str
    implementation_id: str


class PreprocessReadinessTracker(TensorReadinessTracker[PreprocessReadiness]):
    """Adapt generic exact-tensor state tracking to RF-DETR preprocessing."""

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
        state = PreprocessReadiness(
            ready_event=ready_event,
            input_kind=input_kind,
            implementation_id=implementation_id,
        )
        super().record(tensor, state=state)
