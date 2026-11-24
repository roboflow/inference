"""Result building shared by the SAM2, SAM3, interactive and video blocks.

Every SAM block gives the instance-segmentation outputs of decision 018: a
``segmentation`` view to wire to blocks and ``predictions``, its dense
serializable ``InstanceDetections``. Each output is built only when a reader
wants it. Masks and boxes are on the image's device; boxes are half-open
pixel extents of each mask, ``[0, 0, 0, 0]`` for an empty mask. Including
the full last occupied pixel (``max + 1``) deliberately corrects V1's
inclusive bounds so crops retain every pixel; prompt boxes reach the model
unchanged.

Hosted execution is deferred, not ported: V1's inference-server mode
(``StepExecutionMode.REMOTE``) and SAM3's platform proxy (``SAM3_EXEC_MODE``).
Roboflow's next inference server will define that contract; until then the
blocks run only the local model from the ``models`` resource, and a step that
asks for another mode fails validation before any request is made.
"""

import weakref
from typing import Any, Dict, Mapping, Optional

import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef, require_compatible_frame
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import SegmentationView
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAMES_KEY,
    RowSelection,
    class_names_by_id,
    take_rows,
)
from roboflow_workflows.v2.data.row_identity import ROW_RECORD_KEY, with_row_identity
from roboflow_workflows.v2.engine.declaration import Block


PREDICTION_TYPE = "instance-segmentation"

FOREGROUND = "foreground"

MASK_LOGIT_THRESHOLD = 0.0


class _Embedding:
    """One image's SAM embedding and a weak reference to its pixel tensor."""

    def __init__(self, pixels: torch.Tensor, embeddings: Any):
        self._pixels = weakref.ref(pixels)
        self.embeddings = embeddings

    def holds(self, pixels: torch.Tensor) -> bool:
        # A new tensor under a reused image_id (e.g. with_pixels) is a miss.
        held = self._pixels() is pixels

        return held


def _segmentation_result(
    predictions: InstanceDetections,
    *,
    reader: Block,
    model_id: str,
    inference_id: str,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"model_id": model_id, "inference_id": inference_id}
    if reader.wants("segmentation"):
        result["segmentation"] = SegmentationView.from_dense(predictions)
    if reader.wants("predictions"):
        result["predictions"] = predictions

    return result


def _no_instances(
    image: ImageData, *, prompt: Optional[Detections], namespace: str
) -> InstanceDetections:
    """The empty result of an image whose prompt has no rows."""
    names = None if prompt is None else class_names_by_id(prompt.image_metadata)
    # Shared selection carries the record even for an already empty prompt.
    kept = (
        None
        if prompt is None
        else take_rows(prompt, RowSelection.of(len(prompt), indices=[]))
    )
    record = None if kept is None else (kept.image_metadata or {}).get(ROW_RECORD_KEY)
    predictions = _instances(
        torch.zeros((0, *image.size_hw), dtype=torch.bool, device=image.device),
        scores=torch.zeros(0, dtype=torch.float32, device=image.device),
        class_id=torch.zeros(0, dtype=torch.int64, device=image.device),
        metadata=_metadata(image, class_names=names),
    )
    identified = _with_record(predictions, record=record, namespace=namespace)

    return identified


def _instances(
    mask: torch.Tensor,
    *,
    scores: torch.Tensor,
    class_id: torch.Tensor,
    metadata: Dict[str, Any],
) -> InstanceDetections:
    predictions = InstanceDetections(
        xyxy=_mask_boxes(mask),
        class_id=class_id,
        confidence=scores,
        mask=mask,
        image_metadata=metadata,
    )

    return predictions


def _with_record(
    predictions: InstanceDetections, *, record: Optional[dict], namespace: str
) -> InstanceDetections:
    """Carry the prompt rows' identity, or start a new one without it."""
    if record is None:
        identified = with_row_identity(predictions, namespace=namespace)
        return identified

    predictions.image_metadata[ROW_RECORD_KEY] = record

    return predictions


def _metadata(
    image: ImageData, *, class_names: Optional[Mapping[int, Any]]
) -> Dict[str, Any]:
    metadata = {
        **FrameRef.from_image(image).prediction_metadata(),
        CLASS_NAMES_KEY: dict(class_names or {}),
        "prediction_type": PREDICTION_TYPE,
    }

    return metadata


def _mask_boxes(mask: torch.Tensor) -> torch.Tensor:
    """Half-open ``xyxy`` pixel extents of each mask; zeros for an empty mask."""
    _, height, width = mask.shape
    columns_used = mask.any(dim=1)
    rows_used = mask.any(dim=2)
    x = torch.arange(width, device=mask.device)
    y = torch.arange(height, device=mask.device)
    boxes = torch.stack(
        [
            torch.where(columns_used, x, width).amin(dim=1),
            torch.where(rows_used, y, height).amin(dim=1),
            torch.where(columns_used, x, -1).amax(dim=1) + 1,
            torch.where(rows_used, y, -1).amax(dim=1) + 1,
        ],
        dim=1,
    ).to(torch.float32)
    empty = ~columns_used.any(dim=1, keepdim=True)
    boxes = boxes.masked_fill(empty, 0.0)

    return boxes


def _prompt_boxes(prompt: Any, *, block: str) -> Optional[Detections]:
    """The prompt's boxes as ``Detections``; ``None`` when there is no prompt."""
    if prompt is None:
        return None

    if isinstance(prompt, SegmentationView):
        return prompt.detections

    if isinstance(prompt, tuple):
        _, detections = prompt
        if detections is None:
            raise ValueError(
                f"{block}: the keypoint prediction has no boxes; SAM2 needs boxes "
                "as prompts"
            )
        prompt = detections
    boxes = Detections(
        xyxy=prompt.xyxy,
        class_id=prompt.class_id,
        confidence=prompt.confidence,
        image_metadata=prompt.image_metadata,
        bboxes_metadata=prompt.bboxes_metadata,
    )

    return boxes


def _require_same_frame(
    prompt: Optional[Detections], image: ImageData, *, block: str
) -> None:
    metadata = None if prompt is None else prompt.image_metadata
    try:
        require_compatible_frame(metadata, image=image)
    except ContractError as error:
        raise ValueError(
            f"{block}: {error}; wire boxes predicted on the image you segment"
        ) from error


def _unit_float(value: Any, *, name: str, block: str) -> float:
    # Direct callers bypass engine validation; list elements also use this check.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{block}: {name} must be a number 0..1, got {value!r}")

    if not 0 <= value <= 1:
        raise ValueError(f"{block}: {name} must be a number 0..1, got {value!r}")

    number = float(value)

    return number
