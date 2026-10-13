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

Label names (decision 025): a row's own string ``bboxes_metadata[i]["class"]``
wins, else ``image_metadata["class_names"][class_id]``. The row name matters
when rows share an id but not a name: OCR texts replacing classes, or two
models combined with clashing ids. Colours still follow the class id.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import supervision as sv
import torch
from roboflow_workflows.v2.data.kinds import Kind
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    class_names_by_id,
    row_class_names,
)

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
        labels: ``N`` texts ``"<class name> <confidence:.2f>"``; the name is
            the row's own ``class`` when it has one.

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
        predictions: Native detections, all tensors on one device. Each row is
            named by its string ``bboxes_metadata[i]["class"]`` if present,
            else by ``image_metadata["class_names"]`` (id -> name mapping, or
            a list indexed by id).

    Returns:
        One new ``DetectionDrawing`` per element, in order.

    Raises:
        ValueError: On tensors on several devices, inconsistent shapes,
            ``bboxes_metadata`` of another length than the boxes, or a row
            that neither its own ``class`` nor ``class_names`` names.
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
            class_id,
            confidence,
            image_metadata=detections.image_metadata,
            bboxes_metadata=detections.bboxes_metadata,
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
    bboxes_metadata: Optional[Sequence[Any]] = None,
) -> Tuple[str, ...]:
    # V1 "Class and Confidence" text; a missing name is an error, not a guess.
    if len(class_id) == 0:
        return ()

    rows = _row_metadata(bboxes_metadata, count=len(class_id))
    class_ids = class_id.tolist()
    class_names = class_names_by_id(image_metadata)
    names = row_class_names(class_ids, row_metadata=rows, names_by_id=class_names)
    if None in names:
        index = names.index(None)
        _raise_unnamed(
            class_ids[index], class_names=class_names, row=rows[index], index=index
        )

    labels = tuple(
        f"{name} {one_confidence:.2f}"
        for name, one_confidence in zip(names, confidence)
    )

    return labels


def _row_metadata(
    bboxes_metadata: Optional[Sequence[Any]], *, count: int
) -> List[Mapping[str, Any]]:
    # A row that is not a mapping (None included) carries no name.
    if bboxes_metadata is None:
        return [{}] * count
    if len(bboxes_metadata) != count:
        raise ValueError(
            f"bboxes_metadata has {len(bboxes_metadata)} rows for {count} boxes"
        )

    rows = [row if isinstance(row, Mapping) else {} for row in bboxes_metadata]

    return rows


def _raise_unnamed(
    class_id: int,
    *,
    class_names: Optional[Mapping[int, Any]],
    row: Mapping[str, Any],
    index: int,
) -> None:
    if class_names is None:
        reason = f"image_metadata has no {CLASS_NAMES_KEY}"
    else:
        reason = (
            f"{CLASS_NAMES_KEY} does not name it (ids: {sorted(class_names, key=str)})"
        )
    if CLASS_NAME_KEY in row:
        reason += (
            f"; bboxes_metadata['{CLASS_NAME_KEY}'] is "
            f"{row[CLASS_NAME_KEY]!r}, not a string"
        )
    raise ValueError(f"row {index} with class id {class_id} has no name: {reason}")
