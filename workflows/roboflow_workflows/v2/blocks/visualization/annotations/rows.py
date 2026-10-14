"""Predictions as the annotation blocks read them: placed in the image, on the host.

Carriers and what each block family reads of them::

    carrier                          boxes                masks (mask_quality)
    Detections                       itself               -
    InstanceDetections, dense/RLE    its boxes            mask_provider_of(...).binary
    SegmentationView                 view.detections      mask_provider_of(...).binary
    (KeyPoints, Detections)          the Detections       -
    (KeyPoints, None)                ValueError (F8)      -

Box readers never touch masks: a view is never materialized for them.

``mask_quality`` maps the public names to the provider's fidelity::

    accurate  -> "reference"     the masks the model defines
    coarse    -> "approximate"   a view's grid cells, thresholded, then expanded

Both return one full-image mask per row. Dense and RLE carriers have no
cheaper form, so both names give the same masks for them.

Placement in the image (``placed_in``)::

    no "workflows_v2_frame" record      drawn as given, as V1
    record names the image              drawn as given (sizes must agree)
    other frame of the same named root  prediction_to_frame into the image;
                                        drawing clips to the image
    another or an unnamed root          ValueError with the remedy

Host boundary: ``read_rows`` copies one image's boxes, class ids and
confidences to the host in one transfer. Each column travels as its own
bytes (no float64 on the device, which MPS lacks), so class ids stay exact
integers. Boxes arrive as float64 if the device boxes are float64, else as
float32. Row metadata (class names, tracker ids) is already host data;
class names follow ``prediction_ops.row_class_names``.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
from roboflow_workflows.v2.data.coordinates import prediction_to_frame
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef, frame_of
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
    mask_provider_of,
)
from roboflow_workflows.v2.data.prediction_ops import (
    class_names_by_id,
    row_class_names,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

BOX_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)
MASK_KINDS = (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)
# Public mask_quality -> provider fidelity.
MASK_QUALITY_FIDELITY = {"accurate": "reference", "coarse": "approximate"}
MASK_QUALITIES = tuple(MASK_QUALITY_FIDELITY)
TRACKER_ID_KEY = "tracker_id"
# Torch dtypes whose bytes numpy reads directly; others travel as float32.
_HOST_DTYPES = {
    torch.float64: np.float64,
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

Placeable = Union[Detections, InstanceDetections, tuple]


@dataclass(frozen=True)
class DrawRows:
    """One image's rows on the host, in the image's pixels.

    Args:
        xyxy: ``(N, 4)`` boxes: float64 if the device boxes are float64,
            else float32.
        class_id: ``(N,)`` int64 class ids.
        confidence: ``(N,)`` float64 confidences.
        row_metadata: ``N`` dicts (``bboxes_metadata``; empty when absent).
        image_metadata: The predictions' ``image_metadata`` (may be empty).
    """

    xyxy: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray
    row_metadata: Sequence[dict]
    image_metadata: dict

    def __len__(self) -> int:
        return len(self.class_id)

    def class_names(self) -> List[str]:
        """Return each row's class name.

        ``prediction_ops.row_class_names``: a row's own ``"class"`` string
        wins (decision 025); otherwise ``image_metadata["class_names"]
        [class_id]`` (a dict or a list).

        Returns:
            ``N`` names.

        Raises:
            ValueError: When a row's name cannot be resolved.
        """
        class_ids = self.class_id.tolist()
        names = row_class_names(
            class_ids,
            row_metadata=self.row_metadata,
            names_by_id=class_names_by_id(self.image_metadata),
        )
        if None in names:
            raise ValueError(
                f"class id {class_ids[names.index(None)]} has no name: the row has "
                "no 'class' string and image_metadata['class_names'] does not list it"
            )
        names = [str(name) for name in names]

        return names

    def tracker_ids(self) -> Optional[np.ndarray]:
        """Return each row's tracker id (``tracker_ids_of``)."""
        tracker_ids = tracker_ids_of(self.row_metadata)

        return tracker_ids


def tracker_ids_of(row_metadata: Sequence[Optional[dict]]) -> Optional[np.ndarray]:
    """Return each row's ``tracker_id`` for color axis ``TRACK``.

    Args:
        row_metadata: ``bboxes_metadata`` rows; a ``None`` row has no id.

    Returns:
        ``(N,)`` int64 tracker ids on the host (empty for no rows), or
        ``None`` when a row has no id.
    """
    values = [(row or {}).get(TRACKER_ID_KEY) for row in row_metadata]
    if any(value is None for value in values):
        return None

    tracker_ids = np.asarray(values, dtype=np.int64)

    return tracker_ids


def boxes_of(predictions: Any, *, block: str) -> Detections:
    """Return a carrier's boxes as ``Detections`` without touching its masks.

    Args:
        predictions: A carrier listed in this module's docstring.
        block: Block type for error messages.

    Returns:
        ``Detections`` sharing the carrier's box, class, confidence tensors
        and metadata.

    Raises:
        ValueError: For keypoints without boxes or an unsupported value.
    """
    if isinstance(predictions, SegmentationView):
        boxes = predictions.detections
    elif isinstance(predictions, tuple) and len(predictions) == 2:
        key_points, boxes = predictions
        if not isinstance(key_points, KeyPoints):
            raise ValueError(f"{block}: unsupported predictions {predictions!r}")
        if boxes is None:
            raise ValueError(
                f"{block}: these keypoint predictions have no bounding boxes, "
                "which this block draws. Use roboflow_core/keypoint_visualization@v1 for "
                "keypoints, or a keypoint model that also returns boxes."
            )
    elif isinstance(predictions, (Detections, InstanceDetections)):
        boxes = predictions
    else:
        raise ValueError(
            f"{block}: expected Detections, InstanceDetections, a "
            "SegmentationView or (KeyPoints, Detections), got "
            f"{type(predictions).__name__}"
        )

    if isinstance(boxes, InstanceDetections):
        from roboflow_workflows.v2.data.masks.grid import image_detections_of

        boxes = image_detections_of(boxes)

    return boxes


def has_masks(predictions: Any) -> bool:
    """Tell whether a carrier holds instance masks (dense, RLE or a view)."""
    masked = isinstance(predictions, (InstanceDetections, SegmentationView))

    return masked


def dense_masks_of(predictions: Any, *, quality: str, block: str) -> InstanceDetections:
    """Return one dense binary mask per row at the requested ``mask_quality``.

    Args:
        predictions: ``InstanceDetections`` (dense or RLE) or a
            ``SegmentationView``.
        quality: ``accurate`` or ``coarse`` (see this module's docstring).
        block: Block type for error messages.

    Returns:
        ``InstanceDetections`` with a dense ``(N, H, W)`` mask in the
        prediction's own frame; shared and read-only.

    Raises:
        ValueError: On an unknown quality or a carrier without masks.
    """
    if not isinstance(quality, str) or quality not in MASK_QUALITY_FIDELITY:
        raise ValueError(
            f"{block}: mask_quality must be one of {list(MASK_QUALITIES)}, got "
            f"{quality!r}"
        )
    if not has_masks(predictions):
        raise ValueError(
            f"{block}: needs instance masks, got {type(predictions).__name__}"
        )

    fidelity = MASK_QUALITY_FIDELITY[quality]
    masks = mask_provider_of(predictions).binary(fidelity=fidelity)

    return masks.predictions


def placed_in(prediction: Placeable, image: ImageData, *, block: str) -> Placeable:
    """Return the prediction in the image's pixel coordinates.

    Predictions of another frame of the same workflow input are mapped into
    the image; the blocks then clip what they draw to the image, so rows
    partly outside are cut and rows fully outside draw nothing.

    Args:
        prediction: ``Detections``, ``InstanceDetections`` or a keypoint
            tuple.
        image: The image drawn on.
        block: Block type for error messages.

    Returns:
        ``prediction`` itself when it has no frame record or already
        describes ``image``; otherwise a new prediction mapped into it.

    Raises:
        ValueError: When the prediction claims the image's id with another
            size, or its workflow input is another one or has no name.
    """
    if not _has_frame_record(prediction):
        return prediction

    frame = frame_of(prediction)
    if frame.image_id == image.image_id:
        if tuple(frame.size_hw) != tuple(image.size_hw):
            raise ValueError(
                f"{block}: predictions describe image {frame.image_id!r} of "
                f"(h, w)={frame.size_hw}, but the image is {image.size_hw}"
            )
        return prediction

    prediction_root, image_root = frame.root, image.root
    same_root = (
        prediction_root.frame_id is not None
        and prediction_root.frame_id == image_root.frame_id
        and tuple(prediction_root.frame_size_hw) == tuple(image_root.frame_size_hw)
    )
    if not same_root:
        raise ValueError(
            f"{block}: predictions of image {frame.image_id!r} (input "
            f"{prediction_root.frame_id!r}, (h, w)={prediction_root.frame_size_hw}) "
            f"cannot be placed in image {image.image_id!r} (input "
            f"{image_root.frame_id!r}, (h, w)={image_root.frame_size_hw}). Draw on "
            "the image the predictions were made on (the model step's image "
            "input, e.g. $inputs.image) or on an image derived from it."
        )

    mapping = _mapping_between(prediction_root, image=image)
    placed = prediction_to_frame(
        prediction, mapping=mapping, frame=FrameRef.from_image(image)
    )

    return placed


def read_rows(boxes: Detections) -> DrawRows:
    """Copy one image's boxes, class ids and confidences to the host at once.

    The columns are packed as raw bytes into one ``(N, bytes)`` uint8 table,
    copied once and read back with their own dtypes, so no value is rounded.

    Args:
        boxes: ``Detections`` in the image's pixels.

    Returns:
        The host rows; empty arrays when there are no rows (no transfer).
    """
    count = len(boxes)
    columns = [
        boxes.xyxy.reshape(count, 4),
        boxes.class_id.reshape(count, 1),
        boxes.confidence.reshape(count, 1),
    ]
    columns = [
        column if column.dtype in _HOST_DTYPES else column.to(torch.float32)
        for column in columns
    ]
    xyxy, class_id, confidence = _host_columns(columns, count=count)

    # Boxes keep a float64 dtype, others become float32 (exact for float16 and
    # bfloat16), so anchors and centres use the arithmetic V1 used.
    box_dtype = np.float64 if xyxy.dtype == np.float64 else np.float32
    rows = DrawRows(
        xyxy=xyxy.astype(box_dtype),
        class_id=class_id[:, 0].astype(np.int64),
        confidence=confidence[:, 0].astype(np.float64),
        row_metadata=[row or {} for row in (boxes.bboxes_metadata or [{}] * count)],
        image_metadata=dict(boxes.image_metadata or {}),
    )

    return rows


def clipped_corners(xyxy: np.ndarray, size_hw: Sequence[int]) -> List[List[int]]:
    """Clip boxes to the frame and truncate them to integers, as V1 does.

    Args:
        xyxy: ``(N, 4)`` float boxes.
        size_hw: Frame ``(H, W)``.

    Returns:
        ``N`` integer ``[x1, y1, x2, y2]``; ``x2 <= x1`` marks an empty box.
    """
    height, width = int(size_hw[0]), int(size_hw[1])
    clipped = np.array(xyxy, copy=True)
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, width)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, height)
    corners = clipped.astype(int).tolist()

    return corners


def mask_extents(masks: torch.Tensor) -> List[Optional[List[int]]]:
    """Return each mask's pixel extent, with one device-to-host transfer.

    Args:
        masks: ``(N, H, W)`` bool masks on any device.

    Returns:
        Per mask, half-open ``[left, top, right, bottom]``, or ``None`` for
        an empty mask.
    """
    count, height, width = masks.shape
    if count == 0:
        return []

    rows, columns = masks.any(dim=2), masks.any(dim=1)
    top = rows.to(torch.uint8).argmax(dim=1)
    bottom = height - rows.flip(1).to(torch.uint8).argmax(dim=1)
    left = columns.to(torch.uint8).argmax(dim=1)
    right = width - columns.flip(1).to(torch.uint8).argmax(dim=1)
    present = rows.any(dim=1)
    table = torch.stack([left, top, right, bottom, present.to(left.dtype)], dim=1).cpu()
    extents = [row[:4] if row[4] else None for row in table.tolist()]

    return extents


def _host_columns(columns: List[torch.Tensor], *, count: int) -> List[np.ndarray]:
    """Copy ``(N, k)`` columns of ``_HOST_DTYPES`` to the host in one transfer.

    Layout of the packed uint8 table: the columns side by side, each as its
    ``k * itemsize`` bytes in device byte order (host and devices agree).
    """
    host_dtypes = [_HOST_DTYPES[column.dtype] for column in columns]
    if count == 0:
        empty = [
            np.zeros((0, column.shape[1]), dtype=dtype)
            for column, dtype in zip(columns, host_dtypes)
        ]
        return empty

    packed = torch.cat(
        [
            column.contiguous().view(torch.uint8).reshape(count, -1)
            for column in columns
        ],
        dim=1,
    )
    table = packed.cpu().numpy()

    host, start = [], 0
    for column, dtype in zip(columns, host_dtypes):
        width = column.shape[1] * column.element_size()
        host.append(np.ascontiguousarray(table[:, start : start + width]).view(dtype))
        start += width

    return host


def _has_frame_record(prediction: Placeable) -> bool:
    parts = prediction if isinstance(prediction, tuple) else (prediction,)
    has_record = any(
        part is not None and FRAME_RECORD_KEY in (part.image_metadata or {})
        for part in parts
    )

    return has_record


def _mapping_between(root: FrameMapping, *, image: ImageData) -> FrameMapping:
    # prediction -> root -> image: image_xy = (root_xy - o_i) / s_i.
    (prediction_sx, prediction_sy), (prediction_ox, prediction_oy) = (
        root.scale_xy,
        root.offset_xy,
    )
    (image_sx, image_sy), (image_ox, image_oy) = (
        image.root.scale_xy,
        image.root.offset_xy,
    )
    mapping = FrameMapping(
        frame_id=image.image_id,
        frame_size_hw=image.size_hw,
        scale_xy=(prediction_sx / image_sx, prediction_sy / image_sy),
        offset_xy=(
            (prediction_ox - image_ox) / image_sx,
            (prediction_oy - image_oy) / image_sy,
        ),
    )

    return mapping
