"""Host drawing data of detections: one device-to-host transfer per batch.

Painters need each detection's box, class id, confidence and label text on
the host. ``prepare_drawings`` gathers them for a whole batch at once::

    device: view each tensor as bytes and concatenate them (small copies;
            strided model columns are gathered first)
    one device-to-host transfer of those bytes
    host:   copy into an immutable ``bytes`` object, view it as the arrays

Bytes, not a common dtype, so nothing is rounded: an int64 class id above
2**53 arrives as it left. Each result is a new ``DetectionDrawing`` whose
arrays are read-only views of that ``bytes`` object, so neither a painter nor
a later change of the source tensors can alter it. Nothing is cached.

The style is the V1 default of the bounding box and label blocks: the
supervision default palette by class id and 2 px square borders.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import supervision as sv
import torch
from roboflow_workflows.execution_engine.v2.kinds import Kind

from inference_models.models.base.object_detection import Detections

PALETTE = sv.ColorPalette.DEFAULT
BOX_THICKNESS = 2


@dataclass(frozen=True)
class DetectionDrawing:
    """What the painters need of one image's detections, on the host.

    Built per call by ``prepare_drawings``; never cached or shared between
    calls.

    Args:
        xyxy: Read-only ``(N, 4)`` boxes in the image's pixel coordinates,
            the dtype of ``Detections.xyxy``.
        class_id: Read-only ``(N,)`` class ids.
        confidence: Read-only ``(N,)`` confidences.
        labels: ``N`` texts ``"<class name> <confidence:.2f>"``.

    Raises:
        ValueError: On mismatched lengths, wrong ranks or writeable arrays.
    """

    xyxy: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray
    labels: Tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.labels)
        shapes_ok = (
            self.xyxy.shape == (count, 4)
            and self.class_id.shape == (count,)
            and self.confidence.shape == (count,)
        )
        if not shapes_ok:
            raise ValueError(
                f"DetectionDrawing of {count} labels has xyxy {self.xyxy.shape}, "
                f"class_id {self.class_id.shape}, confidence {self.confidence.shape}"
            )
        arrays = (self.xyxy, self.class_id, self.confidence)
        if any(array.flags.writeable for array in arrays):
            raise ValueError("DetectionDrawing arrays must be read-only")

    def __len__(self) -> int:
        return len(self.labels)


def _is_detection_drawing(payload: Any) -> bool:
    return isinstance(payload, DetectionDrawing)


DETECTION_DRAWING_KIND = Kind(
    name="detection_drawing",
    description=(
        "DetectionDrawing: one image's boxes, class ids, confidences and "
        "label texts as read-only host arrays, prepared for painters."
    ),
    validate=_is_detection_drawing,
)


def prepare_drawings(predictions: Sequence[Detections]) -> List[DetectionDrawing]:
    """Copy the drawing data of every ``Detections`` to the host at once.

    One device-to-host transfer for the whole sequence; it is complete on
    return. CUDA copies use the current stream. If a producer used another
    stream, join that stream before calling. Painters call this inside
    ``ReadyScope.work`` to use their side stream with the caller dependency;
    outside that context it uses the caller's current stream directly.

    Args:
        predictions: Native detections, all tensors on one device. Each needs
            ``image_metadata["class_names"]`` (id -> name mapping, or a list
            indexed by id) naming its class ids, unless it is empty.

    Returns:
        One new ``DetectionDrawing`` per element, in order.

    Raises:
        ValueError: On tensors on several devices, inconsistent shapes, a
            missing ``class_names`` or a class id without a name.
    """
    tensors = []
    for detections in predictions:
        tensors.extend(_drawing_tensors(detections))
    _require_one_device(tensors)

    host = _to_host(tensors)
    drawings = []
    for position, detections in enumerate(predictions):
        xyxy, class_id, confidence = host[3 * position : 3 * position + 3]
        labels = _class_and_confidence_labels(
            class_id, confidence, image_metadata=detections.image_metadata
        )
        drawing = DetectionDrawing(
            xyxy=xyxy, class_id=class_id, confidence=confidence, labels=labels
        )
        drawings.append(drawing)

    return drawings


def _drawing_tensors(detections: Detections) -> Tuple[torch.Tensor, ...]:
    count = len(detections)
    xyxy, class_id, confidence = (
        detections.xyxy,
        detections.class_id,
        detections.confidence,
    )
    if xyxy.shape != (count, 4) or class_id.shape != (count,):
        raise ValueError(
            f"Detections xyxy {tuple(xyxy.shape)} and class_id "
            f"{tuple(class_id.shape)} do not describe {count} boxes"
        )
    if confidence.shape != (count,):
        raise ValueError(
            f"Detections confidence {tuple(confidence.shape)} does not describe "
            f"{count} boxes"
        )

    return xyxy, class_id, confidence


def _require_one_device(tensors: Sequence[torch.Tensor]) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) > 1:
        raise ValueError(
            f"detections to draw must be on one device, got {sorted(map(str, devices))}"
        )


def _to_host(tensors: Sequence[torch.Tensor]) -> List[np.ndarray]:
    # Raw bytes in one buffer: small device copies, one device-to-host
    # transfer, one host copy into immutable bytes. Exact for every dtype.
    if not tensors:
        return []

    raw = [_dense(tensor).view(torch.uint8) for tensor in tensors]
    blob = torch.cat(raw).cpu().numpy().tobytes()

    arrays = []
    offset = 0
    for tensor, part in zip(tensors, raw):
        array = np.frombuffer(
            blob,
            dtype=_numpy_dtype(tensor.dtype),
            count=tensor.numel(),
            offset=offset,
        ).reshape(tuple(tensor.shape))
        arrays.append(array)
        offset += part.numel()

    return arrays


def _dense(tensor: torch.Tensor) -> torch.Tensor:
    # A 1-D view with stride 1, which a byte view needs. Not .contiguous():
    # PyTorch calls a length-1 slice contiguous whatever its stride (a real
    # model's class_id column of one detection has stride 6).
    flat = tensor.detach().reshape(-1)
    if flat.stride(0) == 1:
        return flat

    dense = torch.empty(flat.shape, dtype=flat.dtype, device=flat.device)
    dense.copy_(flat)

    return dense


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    numpy_dtype = torch.empty((), dtype=dtype).numpy().dtype

    return numpy_dtype


def _class_and_confidence_labels(
    class_id: np.ndarray,
    confidence: np.ndarray,
    *,
    image_metadata: Optional[Mapping[str, Any]],
) -> Tuple[str, ...]:
    # V1 "Class and Confidence" text; a missing name is an error, not a guess.
    if len(class_id) == 0:
        return ()

    class_names = (image_metadata or {}).get("class_names")
    if class_names is None:
        raise ValueError("predictions.image_metadata has no class_names")
    if not isinstance(class_names, Mapping):
        class_names = dict(enumerate(class_names))

    labels = []
    for one_class, one_confidence in zip(class_id, confidence):
        if int(one_class) not in class_names:
            raise ValueError(f"class id {int(one_class)} has no name in class_names")
        labels.append(f"{class_names[int(one_class)]} {one_confidence:.2f}")
    labels = tuple(labels)

    return labels
