"""Concatenate two predictions of one image: ``roboflow_core/detections_combine@v1``.

Port of V1 ``roboflow_core/detections_combine@v1`` (``sv.Detections.merge``).
Rows of ``prediction_one`` come first, then rows of ``prediction_two``; no row
is dropped, deduplicated or relabelled. The carrier follows the inputs::

    one input has no rows     the other input, unchanged (V1 drops empty inputs)
    Detections + Detections   Detections
    dense + dense masks       dense masks, same (H, W)
    RLE + RLE masks           RLE masks, same image size; bytes are not decoded
    view + view, same grid    view; scores concatenated, nothing upscaled
    other mask mixes          dense masks; each input materialized once at
                              reference fidelity (``mask_provider_of``)
    boxes + masks             error, as in V1: all or none carry masks

"Same grid" needs two compact views with equal geometry, score type,
threshold and score dtype/shape. A dense-fallback view (``is_dense_fallback``;
``roboflow_core/roboflow_instance_segmentation_model@v4`` emits only these today) has no grid, so it
counts as dense: two of them combine to dense ``InstanceDetections``.

Both inputs must describe the same frame and live on the same device. Combine
neither maps coordinates nor copies between devices.

Metadata of the combined rows:

=================  ==========================================================
class names        ``class_id`` values are kept. The map is the union, and
                   ``prediction_one`` wins an id both maps name differently;
                   the losing rows of ``prediction_two`` keep their name in
                   row metadata ``class``.
row identity       Same namespace on both inputs: kept, ids concatenated.
                   Otherwise a fresh namespace (decision 023): the rows are
                   new semantic rows, so exports derive new ``detection_id``
                   values, and crops cut from an input do not join them.
row metadata       Concatenated; a missing list is padded with ``{}``.
                   Explicit ``detection_id`` values are kept; none is made.
=================  ==========================================================

Host work: a class-id copy of ``prediction_two`` only when the class maps
conflict. Mixed mask representations materialize on the host only where their
provider does (RLE decoding).

``geometry.stitch.concatenate`` also concatenates rows, but for another
contract: its frame comes from a reference image, a later class map wins
without per-row names, and mixed masks are encoded to RLE on the host.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.query.detections import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    boxes_of,
    row_count,
)
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef, frame_of
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
    mask_provider_of,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    ROW_RECORD_KEY,
    ROW_RECORD_VERSION,
    RowRecord,
    row_record,
    with_row_identity,
    without_row_record,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


BLOCK_TYPE = "roboflow_core/detections_combine@v1"

COMBINE_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)

_LEGACY_FRAME_KEY = "image_dimensions"

class DetectionsCombineBlock(Block):
    """Combine two predictions of the same image into one.

    Example: 2 people from one model and 3 cars from another give 5 rows,
    the people first, each with its own class name. The 5 rows are new rows:
    their exported ``detection_id`` values differ from the inputs', unless a
    row already had an explicit ``detection_id``. Crop after combining.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Detections Combine",
        "port_of": "roboflow_core/detections_combine@v1",
        "section": "transformation",
    }
    outputs = {
        "predictions": Output(
            *COMBINE_KINDS,
            description=(
                "Rows of prediction_one, then rows of prediction_two. Masks keep "
                "their representation when both inputs share it; dense-fallback "
                "views, as instance segmentation emits them, give dense masks. "
                "Rows from two different results get a new row identity: "
                "exported detection_id values are new, except explicit "
                "detection_id values in row metadata, which are kept. Crops cut "
                "before this step do not match these rows; crop after combining."
            ),
        ),
    }

    class Params(BlockParams):
        prediction_one: Ref(*COMBINE_KINDS) = Field(
            description=(
                "First predictions: detections or instance segmentation (dense, "
                "RLE or view). Its rows come first."
            ),
            examples=["$steps.my_object_detection_model.predictions"],
        )
        prediction_two: Ref(*COMBINE_KINDS) = Field(
            description=(
                "Second predictions of the same image, appended after "
                "prediction_one."
            ),
            examples=["$steps.my_object_detection_model.predictions"],
        )

    def run(self, *, prediction_one: Any, prediction_two: Any) -> Dict[str, Any]:
        """Combine two predictions.

        Args:
            prediction_one: A carrier accepted by ``combine_predictions``.
            prediction_two: A carrier accepted by ``combine_predictions``.

        Returns:
            ``predictions``: the combined rows.

        Raises:
            ValueError: When the inputs cannot be combined; see
                ``combine_predictions``.
        """
        combined = combine_predictions(prediction_one, prediction_two)

        return {"predictions": combined}

def combine_predictions(prediction_one: Any, prediction_two: Any) -> Any:
    """Concatenate the rows of two predictions of the same image.

    Args:
        prediction_one: Detections, InstanceDetections or a SegmentationView.
        prediction_two: Same carriers; its rows follow those of
            ``prediction_one``.

    Returns:
        The other input itself when one has no rows (``prediction_one`` when
        both are empty); otherwise a new carrier as in the module docstring.

    Raises:
        ValueError: When an input is another type, the frames or devices
            differ, an unknown frame cannot be checked, only one input has
            masks, or mask rasters differ in size.
    """
    for name, value in (
        ("prediction_one", prediction_one),
        ("prediction_two", prediction_two),
    ):
        if not isinstance(value, (Detections, InstanceDetections, SegmentationView)):
            raise ValueError(
                f"{BLOCK_TYPE}: {name} must be Detections, InstanceDetections or a "
                f"SegmentationView, got {type(value).__name__}"
            )

    if row_count(prediction_two) == 0:
        return prediction_one
    if row_count(prediction_one) == 0:
        return prediction_two

    boxes_one, boxes_two = boxes_of(prediction_one), boxes_of(prediction_two)
    _check_same_frame(boxes_one, boxes_two)
    _check_same_device(boxes_one, boxes_two)
    if isinstance(prediction_one, Detections) != isinstance(prediction_two, Detections):
        raise ValueError(
            f"{BLOCK_TYPE}: cannot combine object detection with instance "
            "segmentation; all or none of the predictions must carry masks, as "
            "in V1. Drop the masks or segment both"
        )

    boxes = _combined_boxes(boxes_one, boxes_two)
    if isinstance(prediction_one, Detections):
        return boxes

    if _views_share_grid(prediction_one, prediction_two):
        view = SegmentationView(
            detections=boxes,
            scores=torch.cat([prediction_one.scores, prediction_two.scores]),
            score_type=prediction_one.score_type,
            threshold=prediction_one.threshold,
            geometry=prediction_one.geometry,
            take_ownership=True,
        )
        return view

    combined = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=_combined_masks(prediction_one, prediction_two),
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )

    return combined

def _check_same_frame(boxes_one: Any, boxes_two: Any) -> None:
    frame_one, frame_two = _known_frame(boxes_one), _known_frame(boxes_two)
    if frame_one is None or frame_two is None:
        raise ValueError(
            f"{BLOCK_TYPE}: cannot check that both predictions describe the same "
            "image, because "
            f"{'prediction_one' if frame_one is None else 'prediction_two'} has no "
            "frame metadata (image_metadata with image_dimensions). Use "
            "predictions from model blocks or attach the image's frame"
        )
    if frame_one == frame_two:
        return
    frames = (
        f"image {frame_one.image_id!r} {frame_one.size_hw} vs "
        f"{frame_two.image_id!r} {frame_two.size_hw}"
    )
    if frame_one.root.frame_id != frame_two.root.frame_id:
        raise ValueError(
            f"{BLOCK_TYPE}: the predictions describe different images ({frames}). "
            "Combine predictions of the same image"
        )
    raise ValueError(
        f"{BLOCK_TYPE}: the predictions describe different frames ({frames}) of "
        f"workflow image {frame_one.root.frame_id!r}, so their coordinates differ. "
        "Combine does not map coordinates. Run both model steps on the same "
        "image or frame, for example both on the workflow image or both on the "
        "same crop. Python code can map one prediction first with "
        "roboflow_workflows.v2.data.coordinates.prediction_to_frame"
    )

def _known_frame(boxes: Any) -> Optional[FrameRef]:
    metadata = boxes.image_metadata or {}
    if FRAME_RECORD_KEY not in metadata and _LEGACY_FRAME_KEY not in metadata:
        return None

    frame = frame_of(boxes)

    return frame

def _check_same_device(boxes_one: Any, boxes_two: Any) -> None:
    if boxes_one.xyxy.device != boxes_two.xyxy.device:
        raise ValueError(
            f"{BLOCK_TYPE}: prediction_one is on {boxes_one.xyxy.device} and "
            f"prediction_two on {boxes_two.xyxy.device}. Combine needs both on "
            "one device and does not copy between devices. Choose model packages "
            "for the same device: set model_package_id on the model steps that "
            "produced these predictions"
        )

def _combined_boxes(boxes_one: Any, boxes_two: Any) -> Detections:
    """Boxes, classes, confidences and metadata of both inputs, in order."""
    rows_one, rows_two = _row_dicts(boxes_one), _row_dicts(boxes_two)
    names_one, names_two = _class_names(boxes_one), _class_names(boxes_two)
    conflicts = {
        class_id
        for class_id, name in (names_two or {}).items()
        if class_id in (names_one or {}) and names_one[class_id] != name
    }
    if conflicts:
        _keep_row_class_names(
            rows_two, class_ids=boxes_two.class_id, names=names_two, only=conflicts
        )

    image_metadata = dict(without_row_record(boxes_one.image_metadata) or {})
    if names_one is not None or names_two is not None:
        image_metadata[CLASS_NAMES_KEY] = {**(names_two or {}), **(names_one or {})}

    record_one, record_two = row_record(boxes_one), row_record(boxes_two)
    same_namespace = (
        record_one is not None
        and record_two is not None
        and record_one.namespace == record_two.namespace
    )
    if same_namespace:
        image_metadata[ROW_RECORD_KEY] = _concatenated_record(
            record_one, record_two, device=boxes_one.xyxy.device
        )

    rows = rows_one + rows_two
    combined = Detections(
        xyxy=torch.cat([boxes_one.xyxy, boxes_two.xyxy]),
        class_id=torch.cat([boxes_one.class_id, boxes_two.class_id]),
        confidence=torch.cat([boxes_one.confidence, boxes_two.confidence]),
        image_metadata=image_metadata,
        bboxes_metadata=rows if any(rows) else None,
    )
    if not same_namespace:
        combined = with_row_identity(combined)

    return combined

def _class_names(boxes: Any) -> Optional[Dict[int, str]]:
    # Integer keys, so maps decoded from JSON ("0") still clash with 0.
    names = (boxes.image_metadata or {}).get(CLASS_NAMES_KEY)
    if names is None:
        return None
    if not isinstance(names, dict):
        names = dict(enumerate(names))

    normalized = {int(class_id): name for class_id, name in names.items()}

    return normalized

def _row_dicts(boxes: Any) -> List[dict]:
    # Copies, so adding class names never writes into an input's rows.
    rows = boxes.bboxes_metadata or [None] * len(boxes)
    copied = [dict(row or {}) for row in rows]

    return copied

def _keep_row_class_names(
    rows: List[dict], *, class_ids: torch.Tensor, names: Dict[int, str], only: set
) -> None:
    for row, class_id in zip(rows, class_ids.tolist()):
        if class_id in only and CLASS_NAME_KEY not in row:
            row[CLASS_NAME_KEY] = names[class_id]

def _concatenated_record(
    record_one: RowRecord, record_two: RowRecord, *, device: torch.device
) -> Dict[str, Any]:
    ids = torch.cat(
        [
            _explicit_ids(record_one, device=device),
            _explicit_ids(record_two, device=device),
        ]
    )
    record = {
        "version": ROW_RECORD_VERSION,
        "namespace": record_one.namespace,
        "rows": record_one.rows + record_two.rows,
        "ids": ids,
    }

    return record

def _explicit_ids(record: RowRecord, *, device: torch.device) -> torch.Tensor:
    if record.ids is not None:
        return record.ids

    ids = torch.arange(record.rows, dtype=torch.int64, device=device)

    return ids

def _views_share_grid(prediction_one: Any, prediction_two: Any) -> bool:
    """Whether two views concatenate exactly, without any upscale."""
    views = (prediction_one, prediction_two)
    if not all(isinstance(view, SegmentationView) for view in views):
        return False
    if any(view.is_dense_fallback for view in views):
        return False

    shares_grid = (
        prediction_one.geometry == prediction_two.geometry
        and prediction_one.score_type == prediction_two.score_type
        and prediction_one.threshold == prediction_two.threshold
        and prediction_one.scores.dtype == prediction_two.scores.dtype
        and prediction_one.scores.shape[1:] == prediction_two.scores.shape[1:]
    )

    return shares_grid

def _combined_masks(prediction_one: Any, prediction_two: Any) -> Any:
    mask_one = getattr(prediction_one, "mask", None)
    mask_two = getattr(prediction_two, "mask", None)
    if isinstance(mask_one, InstancesRLEMasks) and isinstance(
        mask_two, InstancesRLEMasks
    ):
        _check_mask_sizes(tuple(mask_one.image_size), tuple(mask_two.image_size))
        masks = InstancesRLEMasks(
            image_size=mask_one.image_size,
            masks=list(mask_one.masks) + list(mask_two.masks),
        )
        return masks

    dense_one, dense_two = _dense_mask(prediction_one), _dense_mask(prediction_two)
    _check_mask_sizes(tuple(dense_one.shape[1:]), tuple(dense_two.shape[1:]))
    if dense_one.dtype != dense_two.dtype:
        dense_one, dense_two = dense_one.bool(), dense_two.bool()
    masks = torch.cat([dense_one, dense_two])

    return masks

def _dense_mask(prediction: Any) -> torch.Tensor:
    # Dense tensors pass through; RLE and views materialize at reference fidelity.
    if isinstance(prediction, InstanceDetections) and isinstance(
        prediction.mask, torch.Tensor
    ):
        return prediction.mask

    reference = mask_provider_of(prediction).binary(fidelity="reference")

    return reference.predictions.mask

def _check_mask_sizes(size_one: Tuple[int, ...], size_two: Tuple[int, ...]) -> None:
    if size_one != size_two:
        raise ValueError(
            f"{BLOCK_TYPE}: mask sizes differ ({size_one} vs {size_two}); both "
            "predictions must be made against the same image"
        )
