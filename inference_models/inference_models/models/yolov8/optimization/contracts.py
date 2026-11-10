"""YOLO raw-output requests and explicit postprocessing completion."""

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from inference_models import Detections
from inference_models.models.common.roboflow.model_packages import PreProcessingMetadata
from inference_models.models.optimization.contracts import ExecutionContext


@dataclass(frozen=True)
class PostprocessRequest:
    """Inputs to YOLO decoding; tensors remain owned by the caller.

    Attributes:
        model_results: Raw (B, 4 + C, N) output, or fused export output.
        pre_processing_meta: Original coordinate transforms in batch order.
        threshold: Resolved scalar or per-class confidence thresholds.
        iou_threshold: NMS intersection-over-union threshold.
        max_detections: Maximum output count per image.
        class_agnostic_nms: Whether NMS ignores predicted classes.
        fused: Whether the export already applied NMS.
        ready_event: Optional producer event to await on the execution stream.
    """

    model_results: torch.Tensor
    pre_processing_meta: List[PreProcessingMetadata]
    threshold: Union[float, torch.Tensor]
    iou_threshold: float
    max_detections: int
    class_agnostic_nms: bool
    fused: bool = False
    ready_event: Optional[torch.cuda.Event] = None


@dataclass
class PostprocessResult:
    """Pending per-call output whose host counts are readable after completion.

    Attributes:
        ready_event: CUDA completion event, or None for completed CPU work.
        details: Inspectable route and initialization facts for this call.
    """

    ready_event: Optional[torch.cuda.Event]
    _materialize: Callable[[], List[Detections]]
    details: Dict[str, Any] = field(default_factory=dict)
    _detections: Optional[List[Detections]] = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def wait(self) -> List[Detections]:
        """Complete this result once and return independently owned detections.

        Returns:
            Normal detections, ready for use on any stream or the host.
        """
        with self._lock:
            if self._detections is None:
                if self.ready_event is not None:
                    self.ready_event.synchronize()
                self._detections = self._materialize()
            detections = self._detections

        return detections


@contextmanager
def _run_on_stream(request: PostprocessRequest, context: ExecutionContext):
    output = request.model_results
    if not output.is_cuda:
        yield
        return

    stream = context.current_stream
    if stream is None:
        raise ValueError("CUDA postprocessing requires ExecutionContext.current_stream")
    if stream.device != output.device:
        raise ValueError(
            "Postprocessing stream and model output must use the same device"
        )

    with torch.cuda.device(output.device), torch.cuda.stream(stream):
        if request.ready_event is not None:
            stream.wait_event(request.ready_event)
        output.record_stream(stream)
        if isinstance(request.threshold, torch.Tensor) and request.threshold.is_cuda:
            request.threshold.record_stream(stream)
        yield


def _completion_event(context: ExecutionContext) -> Optional[torch.cuda.Event]:
    if context.current_stream is None:
        return None

    event = torch.cuda.Event()
    event.record(context.current_stream)

    return event
