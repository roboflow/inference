"""Coordinate conversion of native predictions for V2 workflow outputs.

Predictions are the ``inference_models`` carriers used by tensor-native blocks.
``frames.root_placement_from_metadata`` reads where a prediction sits in its
workflow root: from the typed ``frames.FRAME_RECORD_KEY`` record when present,
otherwise from the standard V1 ``image_metadata`` root keys. Any image carrier
that writes either works. Legacy keys beside a record must agree with it. Root
conversion rewrites the record and, when the input had any, the legacy keys
together.

The placement's ``root`` is a ``FrameMapping``: scale ``S`` counts root pixels
per local pixel (the reciprocal of the V1 ``scaling_relative_to_root_parent``)
and ``o`` is the local origin in root pixels. Every spatial field maps with
its equation, applied per axis::

    root_xy = local_xy * S + o

This covers ``xyxy`` boxes, keypoint ``xy``, the per-box ``keypoints_xy``,
``polygon`` and oriented-box corner metadata, and masks. Keypoint covariance
maps as ``A C A^T`` with ``A = diag(S_x, S_y)``.

Masks are resampled onto the root canvas by pixel-centre nearest neighbour:
root pixel ``(X, Y)`` takes local pixel
``(floor((X + 0.5 - o_x) / S_x), floor((Y + 0.5 - o_y) / S_y))`` when that pixel
exists, and is empty otherwise. An integer translation with ``S = 1`` is an
exact paste. Other scales change the pixel grid, so converting a mask back is
not lossless. Dense masks stay on their device. RLE masks decode only the local
masks on the host and are re-encoded onto the root canvas without building the
full root canvas.

``prediction_to_frame`` applies the same field mapping with a mapping the
caller states, into any frame of the same workflow input: a crop's
predictions into its parent, or a parent's predictions into a crop.

Output options follow V1: ``coordinates_system="own"`` keeps local coordinates
and ``"parent"`` means the workflow root, not the immediate crop parent.
``"root"`` is an explicit alias of ``"parent"``. A missing option means
``"parent"``, the V1 ``JsonField`` default: V2 output declarations reuse that
V1 key verbatim.
"""

from dataclasses import replace
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
from roboflow_workflows.v2.data.frames import (
    FRAME_RECORD_KEY,
    FrameRef,
    RootPlacement,
    root_placement_from_metadata,
)
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.rle import decode_rle_masks
from supervision.config import ORIENTED_BOX_COORDINATES

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)
from inference_models.models.base.types import InstancesRLEMasks

COORDINATES_SYSTEM_KEY = "coordinates_system"
OWN_COORDINATES = "own"
PARENT_COORDINATES = "parent"
ROOT_COORDINATES = "root"
DEFAULT_COORDINATES_SYSTEM = PARENT_COORDINATES
COORDINATES_SYSTEMS = (OWN_COORDINATES, PARENT_COORDINATES, ROOT_COORDINATES)

# Image-carrier key for the actual ancestor frame. It is not a V1 key.
PARENT_FRAME_ID_KEY = "parent_frame_id"

# The frame may exceed the root canvas by this many root pixels, which absorbs
# integer rounding of resized sizes. Mask pixels beyond the canvas are clipped.
FRAME_FIT_TOLERANCE_PX = 1.0

# V1 per-box host copies of xyxy/class_id/confidence
# (core_steps/common/tensor_native.py). They are stale once boxes move.
_HOST_MIRROR_KEYS = ("_host_xyxy", "_host_class_id", "_host_confidence")

_POINT_METADATA_KEYS = (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
    ORIENTED_BOX_COORDINATES,
)
_LEGACY_FRAME_KEYS = (
    PARENT_ID_KEY,
    PARENT_FRAME_ID_KEY,
    ROOT_PARENT_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
_DETECTION_ROW_LENGTH = 7
# V1 keys of a mosaic frame; rewritten with the other frame keys.
_COMPOSITE_LEGACY_KEYS = ("is_composite", "composite_sources")

TensorNativeDetections = Union[Detections, InstanceDetections]
KeyPointPrediction = Tuple[KeyPoints, Optional[Detections]]


def convert_prediction_output(prediction: Any, options: Mapping[str, Any]) -> Any:
    """Convert a native prediction for one workflow output.

    This is the ``convert_output`` hook of native prediction kinds.

    Args:
        prediction: ``Detections``, ``InstanceDetections``, ``KeyPoints``, a
            ``(KeyPoints, Optional[Detections])`` tuple, one 7-field detection
            row, a classification prediction, or ``None``.
        options: Output declaration options. ``coordinates_system`` is
            ``"own"``, ``"parent"`` (workflow root, as in V1) or ``"root"``;
            a missing value means ``"parent"``.

    Returns:
        ``prediction`` itself for ``"own"``; otherwise the result of
        ``prediction_to_root``.

    Raises:
        ValueError: On an unknown coordinate system or invalid frame metadata.
        TypeError: On options that are not a mapping or an unsupported payload.
    """
    if not isinstance(options, Mapping):
        raise TypeError(
            f"Output options must be a mapping, got {type(options).__name__}"
        )

    coordinates_system = options.get(COORDINATES_SYSTEM_KEY, DEFAULT_COORDINATES_SYSTEM)
    if coordinates_system not in COORDINATES_SYSTEMS:
        raise ValueError(
            f"Unknown {COORDINATES_SYSTEM_KEY} {coordinates_system!r}; expected one "
            f"of {list(COORDINATES_SYSTEMS)}. 'parent' and 'root' both mean the "
            "workflow input image."
        )
    if coordinates_system == OWN_COORDINATES:
        return prediction

    converted = prediction_to_root(prediction)

    return converted


def prediction_to_root(prediction: Any) -> Any:
    """Map a native prediction from its local frame to the workflow root frame.

    Uses ``root_xy = local_xy * S + o`` for every spatial field and rewrites
    ``image_metadata`` to describe the root frame: offsets ``[0, 0]``, scales
    ``1.0``, root dimensions and ``parent_id = root_parent_id``. The input and
    its tensors, masks and metadata are never modified. Unchanged tensors
    (class IDs, confidences) are shared with the input. Per-box host mirrors
    are dropped because they would describe the old boxes.

    Floating coordinates keep their dtype and device. Integer coordinates of
    any width become ``int64`` under an integer translation and ``float32``
    otherwise, so narrow types such as ``uint8`` never wrap.

    A prediction already in its root placement is returned unchanged, so the
    conversion is idempotent. A prediction without a typed frame record or
    root geometry keys is also returned unchanged: it carries no frame to
    restore. Classification
    predictions have no geometry and are returned unchanged.

    Args:
        prediction: ``Detections`` (including QR/barcode detections),
            ``InstanceDetections`` with dense or RLE masks (including semantic
            segmentation), ``KeyPoints``, a ``(KeyPoints, Optional[Detections])``
            tuple, one 7-field detection row, a classification prediction, a
            ``SemanticSegmentationResult`` already in its root placement, or
            ``None``.

    Returns:
        A prediction of the same type in root coordinates.

    Raises:
        ValueError: On partial or malformed frame metadata, a typed frame
            record that disagrees with the legacy keys, a frame that does
            not fit its root, mask canvases that disagree with
            ``image_dimensions``, a composite (mosaic) prediction, which has
            no single root, or a semantic label grid outside its root frame.
        TypeError: On an unsupported payload type.
    """
    if prediction is None:
        return None
    if isinstance(
        prediction, (ClassificationPrediction, MultiLabelClassificationPrediction)
    ):
        return prediction
    if isinstance(prediction, (Detections, InstanceDetections)):
        converted = _detections_to_root(prediction, placement=None)
        return converted
    if isinstance(prediction, KeyPoints):
        converted = _key_points_to_root(prediction, placement=None)
        return converted
    if isinstance(prediction, tuple) and len(prediction) == 2:
        converted = _key_point_prediction_to_root(prediction)
        return converted
    if isinstance(prediction, SemanticSegmentationResult):
        _check_semantic_grid_is_root(prediction)
        return prediction
    if isinstance(prediction, tuple) and len(prediction) == _DETECTION_ROW_LENGTH:
        converted = _detection_row_to_root(prediction)
        return converted

    raise TypeError(
        f"Cannot convert {type(prediction).__name__} to root coordinates; expected "
        "an inference_models prediction, a (KeyPoints, Optional[Detections]) tuple "
        "or a 7-field detection row"
    )


def prediction_to_frame(
    prediction: Any, *, mapping: FrameMapping, frame: FrameRef
) -> Any:
    """Map a native prediction into another frame of the same workflow input.

    The caller states where the prediction's pixels lie in ``frame``:
    ``frame_xy = local_xy * mapping.scale_xy + mapping.offset_xy``. A crop at
    ``(x0, y0)`` moves its predictions into its parent with offset
    ``(x0, y0)``; a parent's prediction moves into the crop with
    ``(-x0, -y0)``. Every spatial field maps as in ``prediction_to_root``.
    Masks are resampled onto ``frame``'s canvas: pixels outside it are
    dropped, uncovered pixels are empty.

    ``image_metadata`` keeps unrelated keys and describes ``frame``: the
    typed record always, the V1 frame keys when the input had any (stale
    mosaic keys are removed). Per-box host mirrors are dropped. Unchanged
    tensors (class ids, confidences) are shared with the input, which is never
    modified. Classification predictions have no geometry and are returned
    unchanged.

    Args:
        prediction: ``Detections``, ``InstanceDetections``, ``KeyPoints``, a
            ``(KeyPoints, Optional[Detections])`` tuple, a classification
            prediction or ``None``.
        mapping: Placement of the prediction's local pixels in ``frame``;
            ``frame_id`` and ``frame_size_hw`` must equal ``frame``'s.
        frame: Frame the result describes.

    Returns:
        A prediction of the same type in ``frame`` coordinates.

    Raises:
        ValueError: When ``mapping`` names another frame or size than
            ``frame``, or masks disagree with their boxes.
        TypeError: On an unsupported payload type.
    """
    if mapping.frame_id != frame.image_id or mapping.frame_size_hw != frame.size_hw:
        raise ValueError(
            f"Mapping targets frame {mapping.frame_id!r} of (h, w)="
            f"{mapping.frame_size_hw}, but the frame is {frame.image_id!r} of "
            f"(h, w)={frame.size_hw}"
        )
    if prediction is None:
        return None
    if isinstance(
        prediction, (ClassificationPrediction, MultiLabelClassificationPrediction)
    ):
        return prediction
    if isinstance(prediction, (Detections, InstanceDetections)):
        _check_row_counts(prediction, mask=getattr(prediction, "mask", None))
        converted = _map_detections(
            prediction,
            mapping=mapping,
            image_metadata=_frame_image_metadata(
                prediction.image_metadata, frame=frame
            ),
        )
        return converted
    if isinstance(prediction, KeyPoints):
        converted = _map_key_points(
            prediction,
            mapping=mapping,
            image_metadata=_frame_image_metadata(
                prediction.image_metadata, frame=frame
            ),
        )
        return converted
    if isinstance(prediction, tuple) and len(prediction) == 2:
        key_points, detections = prediction
        converted = (
            prediction_to_frame(key_points, mapping=mapping, frame=frame),
            prediction_to_frame(detections, mapping=mapping, frame=frame),
        )
        return converted

    raise TypeError(
        f"Cannot map {type(prediction).__name__} to another frame; expected an "
        "inference_models detection, keypoint or classification prediction, or a "
        "(KeyPoints, Optional[Detections]) tuple"
    )


def _frame_image_metadata(
    image_metadata: Optional[Mapping[str, Any]], *, frame: FrameRef
) -> dict:
    """Copy metadata so it describes ``frame``; other keys are kept."""
    frame_metadata = frame.prediction_metadata()
    metadata = dict(image_metadata or {})
    if any(key in metadata for key in _LEGACY_FRAME_KEYS):
        for key in _COMPOSITE_LEGACY_KEYS:
            metadata.pop(key, None)
        metadata.update(frame_metadata)
    else:
        metadata[FRAME_RECORD_KEY] = frame_metadata[FRAME_RECORD_KEY]

    return metadata


def _detections_to_root(
    detections: TensorNativeDetections, *, placement: Optional[RootPlacement]
) -> TensorNativeDetections:
    if isinstance(detections, InstanceDetections):
        from roboflow_workflows.v2.data.masks.grid import image_predictions_of

        detections = image_predictions_of(detections)
    mask = getattr(detections, "mask", None)
    _check_row_counts(detections, mask=mask)
    if placement is None:
        placement = _read_placement(
            detections.image_metadata, mask_canvas_hw=_mask_canvas_hw(mask)
        )
    if placement is None or placement.is_root:
        return detections

    converted = _map_detections(
        detections,
        mapping=placement.root,
        image_metadata=_root_image_metadata(
            detections.image_metadata, placement=placement
        ),
    )

    return converted


def _map_detections(
    detections: TensorNativeDetections,
    *,
    mapping: FrameMapping,
    image_metadata: Optional[dict],
) -> TensorNativeDetections:
    """Map boxes, row metadata and masks with ``mapping``; masks get its canvas."""
    if isinstance(detections, InstanceDetections):
        from roboflow_workflows.v2.data.masks.grid import image_predictions_of

        detections = image_predictions_of(detections)
    xyxy = _map_points(detections.xyxy.reshape(-1, 2), root=mapping).reshape(-1, 4)
    bboxes_metadata = _map_bboxes_metadata(detections.bboxes_metadata, root=mapping)
    if isinstance(detections, InstanceDetections):
        converted = InstanceDetections(
            xyxy=xyxy,
            class_id=detections.class_id,
            confidence=detections.confidence,
            mask=_masks_to_root(detections.mask, root=mapping),
            mask_size=mapping.frame_size_hw,
            image_size=mapping.frame_size_hw,
            mask_frame_size=mapping.frame_size_hw,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
        return converted

    converted = Detections(
        xyxy=xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )

    return converted


def _key_points_to_root(
    key_points: KeyPoints, *, placement: Optional[RootPlacement]
) -> KeyPoints:
    if placement is None:
        placement = _read_placement(key_points.image_metadata, mask_canvas_hw=None)
    if placement is None or placement.is_root:
        return key_points

    converted = _map_key_points(
        key_points,
        mapping=placement.root,
        image_metadata=_root_image_metadata(
            key_points.image_metadata, placement=placement
        ),
    )

    return converted


def _map_key_points(
    key_points: KeyPoints, *, mapping: FrameMapping, image_metadata: Optional[dict]
) -> KeyPoints:
    # Hidden keypoints (confidence 0) move like visible ones; visibility is
    # derived from confidence, which is unchanged.
    xy = _map_points(key_points.xy, root=mapping)
    covariance = key_points.covariance
    if covariance is not None:
        scale = torch.as_tensor(
            mapping.scale_xy, dtype=covariance.dtype, device=covariance.device
        )
        covariance = covariance * torch.outer(scale, scale)
    key_points_metadata = (
        [dict(entry) for entry in key_points.key_points_metadata]
        if key_points.key_points_metadata is not None
        else None
    )
    converted = KeyPoints(
        xy=xy,
        class_id=key_points.class_id,
        confidence=key_points.confidence,
        image_metadata=image_metadata,
        key_points_metadata=key_points_metadata,
        covariance=covariance,
        detection_confidence=key_points.detection_confidence,
    )

    return converted


def _check_semantic_grid_is_root(prediction: SemanticSegmentationResult) -> None:
    # A label grid cannot be resampled or pasted without choosing a background
    # label, so only grids already in their root frame convert (unchanged).
    placement = _read_placement(
        prediction.image_metadata,
        mask_canvas_hw=tuple(prediction.segmentation_map.shape),
    )
    if placement is None or placement.is_root:
        return

    raise ValueError(
        "A semantic label grid of a cropped or resized image has no lossless "
        f"workflow-root form. Request {COORDINATES_SYSTEM_KEY}="
        f"'{OWN_COORDINATES}'. No helper projects a grid to the root yet."
    )


def _key_point_prediction_to_root(prediction: Tuple[Any, Any]) -> KeyPointPrediction:
    key_points, detections = prediction
    if not isinstance(key_points, KeyPoints):
        raise TypeError(
            "Keypoint prediction tuple must start with KeyPoints, got "
            f"{type(key_points).__name__}"
        )
    if detections is not None and not isinstance(detections, Detections):
        raise TypeError(
            "Keypoint prediction tuple must end with Detections or None, got "
            f"{type(detections).__name__}"
        )

    key_points_placement = _read_placement(
        key_points.image_metadata, mask_canvas_hw=None
    )
    detections_placement = (
        _read_placement(detections.image_metadata, mask_canvas_hw=None)
        if detections is not None
        else None
    )
    if (
        key_points_placement is not None
        and detections_placement is not None
        and key_points_placement.root != detections_placement.root
    ):
        raise ValueError(
            "KeyPoints and Detections of one keypoint prediction describe different "
            f"frames: {key_points_placement.root} vs {detections_placement.root}"
        )
    if key_points_placement is None and detections_placement is None:
        return prediction

    # A component without root geometry follows the placement of its sibling.
    converted_key_points = _key_points_to_root(
        key_points, placement=key_points_placement or detections_placement
    )
    converted_detections = (
        _detections_to_root(
            detections, placement=detections_placement or key_points_placement
        )
        if detections is not None
        else None
    )
    if converted_key_points is key_points and converted_detections is detections:
        return prediction

    converted = (converted_key_points, converted_detections)

    return converted


def _detection_row_to_root(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Convert one ``(xyxy, mask, class_id, confidence, tracker_id, data,
    metadata)`` row by converting a one-row carrier.

    Only ``xyxy``, ``mask``, ``data`` geometry and ``metadata`` change. The
    row's own ``class_id``, ``confidence`` and ``tracker_id`` fields are kept
    as given, even when ``data`` holds a different ``tracker_id``.
    """
    xyxy, mask, class_id, confidence, tracker_id, data, metadata = row
    if not isinstance(xyxy, torch.Tensor) or tuple(xyxy.shape) != (4,):
        raise ValueError(
            "Detection row xyxy must be a torch.Tensor of shape (4,), got "
            f"{type(xyxy).__name__} {tuple(getattr(xyxy, 'shape', ()))}"
        )

    common = dict(
        xyxy=xyxy.reshape(1, 4),
        class_id=torch.as_tensor(class_id).reshape(1),
        confidence=torch.as_tensor(confidence).reshape(1),
        image_metadata=metadata,
        bboxes_metadata=[data if data is not None else {}],
    )
    if mask is None:
        carrier = Detections(**common)
    elif isinstance(mask, Mapping):
        rle_masks = InstancesRLEMasks(
            image_size=tuple(mask["size"]), masks=[mask["counts"]]
        )
        carrier = InstanceDetections(mask=rle_masks, **common)
    else:
        carrier = InstanceDetections(mask=mask.unsqueeze(0), **common)

    converted_carrier = _detections_to_root(carrier, placement=None)
    if converted_carrier is carrier:
        return row

    converted_xyxy, converted_mask, _, _, _, converted_data, converted_metadata = next(
        iter(converted_carrier)
    )
    converted_row = (
        converted_xyxy,
        converted_mask,
        class_id,
        confidence,
        tracker_id,
        converted_data if data is not None else None,
        converted_metadata,
    )

    return converted_row


def _read_placement(
    image_metadata: Optional[Mapping[str, Any]],
    *,
    mask_canvas_hw: Optional[Tuple[int, int]],
) -> Optional[RootPlacement]:
    """Read and check the root placement; ``None`` when there is none.

    When legacy metadata omits ``image_dimensions``, the mask canvas is the
    local size, so a local mask smaller than its root is not mistaken for one
    already in root placement.
    """
    if not image_metadata:
        return None

    placement = root_placement_from_metadata(image_metadata)
    if placement is None:
        return None

    local_size_hw = placement.size_hw
    if local_size_hw is not None and mask_canvas_hw is not None:
        if local_size_hw != mask_canvas_hw:
            raise ValueError(
                f"Mask canvas (h, w)={mask_canvas_hw} disagrees with "
                f"'{IMAGE_DIMENSIONS_KEY}'={list(local_size_hw)} of the frame"
            )
    if local_size_hw is None and mask_canvas_hw is not None:
        placement = replace(placement, size_hw=mask_canvas_hw)
    _check_frame_fits_root(placement.root, local_size_hw=placement.size_hw)

    return placement


def _check_frame_fits_root(
    root: FrameMapping, *, local_size_hw: Optional[Tuple[int, int]]
) -> None:
    root_height, root_width = root.frame_size_hw
    offset_x, offset_y = root.offset_xy
    if offset_x < 0 or offset_y < 0 or offset_x > root_width or offset_y > root_height:
        raise ValueError(
            f"'{ROOT_PARENT_COORDINATES_KEY}'={[offset_x, offset_y]} lies outside the "
            f"root frame (h, w)={root.frame_size_hw}"
        )
    if local_size_hw is None:
        return

    local_height, local_width = local_size_hw
    end_x, end_y = root.map_xy(local_width, local_height)
    if (
        end_x > root_width + FRAME_FIT_TOLERANCE_PX
        or end_y > root_height + FRAME_FIT_TOLERANCE_PX
    ):
        raise ValueError(
            f"Local frame (h, w)={local_size_hw} at offset {[offset_x, offset_y]} "
            f"with {list(root.scale_xy)} root pixels per local pixel ends at "
            f"(x, y)=({end_x:g}, {end_y:g}), outside the root frame "
            f"(h, w)={root.frame_size_hw}"
        )


def _is_integer_translation(root: FrameMapping) -> bool:
    integer_offset = all(float(value).is_integer() for value in root.offset_xy)
    is_translation = root.scale_xy == (1.0, 1.0) and integer_offset

    return is_translation


def _root_image_metadata(
    image_metadata: Optional[Mapping[str, Any]], *, placement: RootPlacement
) -> Optional[dict]:
    """Copy metadata so it describes the root frame; other keys are kept.

    Legacy keys are written only when the input had some; typed-only metadata
    stays typed-only.
    """
    if image_metadata is None:
        return None

    root_id = placement.root.frame_id
    root_height, root_width = placement.root.frame_size_hw
    root_metadata = dict(image_metadata)
    if PARENT_FRAME_ID_KEY in root_metadata:
        # A root image is its own parent frame.
        root_metadata[PARENT_FRAME_ID_KEY] = root_id
    if any(key in image_metadata for key in _LEGACY_FRAME_KEYS):
        root_metadata.update(
            {
                PARENT_ID_KEY: root_id,
                IMAGE_DIMENSIONS_KEY: [root_height, root_width],
                PARENT_DIMENSIONS_KEY: [root_height, root_width],
                ROOT_PARENT_DIMENSIONS_KEY: [root_height, root_width],
                PARENT_COORDINATES_KEY: [0, 0],
                ROOT_PARENT_COORDINATES_KEY: [0, 0],
                SCALING_RELATIVE_TO_PARENT_KEY: 1.0,
                SCALING_RELATIVE_TO_ROOT_PARENT_KEY: 1.0,
            }
        )
    if root_metadata.get(FRAME_RECORD_KEY) is not None:
        # The typed record moves to the root with the legacy keys; source and
        # time stay.
        root_mapping = FrameMapping.identity(root_id, placement.root.frame_size_hw)
        root_frame = FrameRef(
            image_id=root_id,
            size_hw=placement.root.frame_size_hw,
            parent=root_mapping,
            root=root_mapping,
            video_metadata=placement.video_metadata,
        )
        root_metadata[FRAME_RECORD_KEY] = root_frame.to_record()

    return root_metadata


def _map_points(points: torch.Tensor, *, root: FrameMapping) -> torch.Tensor:
    """Map a ``(..., 2)`` tensor of ``(x, y)`` points on its own device.

    Integer points are widened before arithmetic, so a narrow storage type
    such as ``uint8`` cannot wrap: to ``int64`` for an integer translation,
    otherwise to ``float32``.
    """
    if not points.is_floating_point():
        wide_dtype = torch.int64 if _is_integer_translation(root) else torch.float32
        points = points.to(wide_dtype)
    scale = torch.as_tensor(root.scale_xy, dtype=points.dtype, device=points.device)
    offset = torch.as_tensor(root.offset_xy, dtype=points.dtype, device=points.device)
    if root.scale_xy == (1.0, 1.0):
        mapped = points + offset
    else:
        mapped = points * scale + offset

    return mapped


def _map_bboxes_metadata(
    bboxes_metadata: Optional[List[dict]], *, root: FrameMapping
) -> Optional[List[dict]]:
    if bboxes_metadata is None:
        return None

    mapped_metadata = []
    for entry in bboxes_metadata:
        entry = {
            key: value for key, value in entry.items() if key not in _HOST_MIRROR_KEYS
        }
        for key in _POINT_METADATA_KEYS:
            if entry.get(key) is not None:
                entry[key] = _map_point_payload(entry[key], root=root, key=key)
        mapped_metadata.append(entry)

    return mapped_metadata


def _map_point_payload(value: Any, *, root: FrameMapping, key: str) -> Any:
    """Map points stored as a tensor, array or (possibly ragged) nested list.

    Tensors and arrays keep their container type; Python sequences come back
    as lists. Integers stay integers (widened to 64 bits) only under an
    integer translation.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return value
        mapped = _map_points(value, root=root)
        return mapped
    if isinstance(value, (list, tuple)) and value and _is_ragged(value):
        mapped = type(value)(
            _map_point_payload(item, root=root, key=key) for item in value
        )
        return mapped

    array = np.asarray(value)
    if array.size == 0:
        return value
    if array.shape[-1] != 2 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            f"Per-box '{key}' must hold (x, y) points, got shape {array.shape} "
            f"of {array.dtype}"
        )

    if np.issubdtype(array.dtype, np.integer) and _is_integer_translation(root):
        mapped_array = array.astype(np.int64) + np.asarray(
            root.offset_xy, dtype=np.int64
        )
    else:
        dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float64
        scale = np.asarray(root.scale_xy, dtype=dtype)
        offset = np.asarray(root.offset_xy, dtype=dtype)
        mapped_array = array.astype(dtype) * scale + offset
    mapped = mapped_array.tolist() if isinstance(value, (list, tuple)) else mapped_array

    return mapped


def _is_ragged(value: Sequence[Any]) -> bool:
    first = value[0]
    if not isinstance(first, (list, tuple, np.ndarray)):
        return False

    lengths = {len(item) for item in value if hasattr(item, "__len__")}
    ragged = len(lengths) > 1 or any(not hasattr(item, "__len__") for item in value)

    return ragged


def _masks_to_root(
    mask: Union[torch.Tensor, InstancesRLEMasks, None], *, root: FrameMapping
) -> Union[torch.Tensor, InstancesRLEMasks, None]:
    if mask is None:
        return None

    local_height, local_width = _mask_canvas_hw(mask)
    root_height, root_width = root.frame_size_hw
    row_start, rows = _sample_source_indices(
        offset=root.offset_xy[1],
        scale=root.scale_xy[1],
        local_size=local_height,
        root_size=root_height,
    )
    column_start, columns = _sample_source_indices(
        offset=root.offset_xy[0],
        scale=root.scale_xy[0],
        local_size=local_width,
        root_size=root_width,
    )
    if isinstance(mask, InstancesRLEMasks):
        converted = _rle_masks_to_root(
            mask,
            rows=rows,
            columns=columns,
            window_origin_xy=(column_start, row_start),
            root_size_hw=root.frame_size_hw,
        )
        return converted

    converted = torch.zeros(
        (mask.shape[0], root_height, root_width), dtype=mask.dtype, device=mask.device
    )
    if rows.size and columns.size:
        window = mask[:, _as_index(rows, device=mask.device)]
        window = window[:, :, _as_index(columns, device=mask.device)]
        converted[
            :,
            row_start : row_start + rows.size,
            column_start : column_start + columns.size,
        ] = window

    return converted


def _sample_source_indices(
    *, offset: float, scale: float, local_size: int, root_size: int
) -> Tuple[int, np.ndarray]:
    """Local pixel index for each root pixel of the covered window, along one axis.

    ``scale`` counts root pixels per local pixel. Returns the first covered
    root pixel and the local indices from there on. Computed from metadata
    only; no mask values are read.
    """
    centres = (np.arange(root_size, dtype=np.float64) + 0.5 - offset) / scale
    sources = np.floor(centres).astype(np.int64)
    covered = np.flatnonzero((sources >= 0) & (sources < local_size))
    if covered.size == 0:
        return 0, sources[:0]

    start = int(covered[0])
    window_sources = sources[start : int(covered[-1]) + 1]

    return start, window_sources


def _as_index(
    indices: np.ndarray, *, device: torch.device
) -> Union[slice, torch.Tensor]:
    first, last = int(indices[0]), int(indices[-1])
    if last - first + 1 == indices.size and np.all(np.diff(indices) == 1):
        return slice(first, last + 1)

    index = torch.as_tensor(indices, dtype=torch.long, device=device)

    return index


def _rle_masks_to_root(
    masks: InstancesRLEMasks,
    *,
    rows: np.ndarray,
    columns: np.ndarray,
    window_origin_xy: Tuple[int, int],
    root_size_hw: Tuple[int, int],
) -> InstancesRLEMasks:
    if not masks.masks:
        return InstancesRLEMasks(image_size=root_size_hw, masks=[])

    # Explicit host boundary: RLE is host data, so only the local masks are
    # validated and decoded here. The root canvas is emitted as runs.
    local_masks = decode_rle_masks(masks)
    windows = local_masks[:, rows][:, :, columns]
    encoded = [
        _encode_window_on_canvas(
            window, origin_xy=window_origin_xy, canvas_size_hw=root_size_hw
        )
        for window in windows
    ]
    converted = InstancesRLEMasks(image_size=root_size_hw, masks=encoded)

    return converted


def _encode_window_on_canvas(
    window: np.ndarray, *, origin_xy: Tuple[int, int], canvas_size_hw: Tuple[int, int]
) -> bytes:
    """Compressed COCO RLE of a canvas that is empty outside ``window``.

    Only the columns crossing the window are materialised.
    """
    canvas_height, canvas_width = canvas_size_hw
    origin_x, origin_y = origin_xy
    window_height, window_width = window.shape
    band = np.zeros((canvas_height, window_width), dtype=np.uint8)
    band[origin_y : origin_y + window_height, :] = window

    # COCO RLE is column-major and starts with a run of zeros.
    values = band.ravel(order="F")
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [values.size]))
    counts = np.diff(boundaries).tolist()
    if values.size and values[0]:
        counts.insert(0, 0)
    if not counts:
        counts = [0]

    counts[0] += origin_x * canvas_height
    trailing_zeros = (canvas_width - origin_x - window_width) * canvas_height
    if len(counts) % 2:
        counts[-1] += trailing_zeros
    elif trailing_zeros:
        counts.append(trailing_zeros)

    encoded = mask_utils.frPyObjects(
        {"counts": counts, "size": [canvas_height, canvas_width]},
        canvas_height,
        canvas_width,
    )["counts"]

    return encoded


def _mask_canvas_hw(
    mask: Union[torch.Tensor, InstancesRLEMasks, None],
) -> Optional[Tuple[int, int]]:
    if mask is None:
        return None
    if isinstance(mask, InstancesRLEMasks):
        return int(mask.mask_size[0]), int(mask.mask_size[1])
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
        raise ValueError(
            "Instance masks must be InstancesRLEMasks or an (N, H, W) torch.Tensor, "
            f"got {type(mask).__name__} {tuple(getattr(mask, 'shape', ()))}"
        )

    return int(mask.shape[1]), int(mask.shape[2])


def _check_row_counts(
    detections: TensorNativeDetections,
    *,
    mask: Union[torch.Tensor, InstancesRLEMasks, None],
) -> None:
    rows = int(detections.xyxy.shape[0])
    if isinstance(mask, InstancesRLEMasks):
        mask_rows = len(mask.masks)
    elif isinstance(mask, torch.Tensor):
        mask_rows = int(mask.shape[0])
    else:
        return
    if mask_rows != rows:
        raise ValueError(f"Prediction has {rows} boxes but {mask_rows} masks")
