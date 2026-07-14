"""Shared helpers for tensor-native predictions (the inference_models dataclasses
used under ENABLE_TENSOR_DATA_REPRESENTATION).

These consolidate logic that otherwise gets copy-pasted into every tensor-native
block sibling: selecting a subset of detections by boolean mask or index list,
and normalising the keypoint-detection input shape. Kept here (rather than on the
inference_models dataclasses) to avoid bloating those types with workflow-specific
concerns.

Supported prediction shapes:
- ``inference_models.Detections``                (object detection)
- ``inference_models.InstanceDetections``        (instance segmentation; dense or RLE masks)
- ``inference_models.KeyPoints``                 (keypoints, standalone)
- ``Tuple[KeyPoints, Optional[Detections]]``     (keypoint-detection workflow kind)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from pycocotools import mask as mask_utils
from supervision.config import ORIENTED_BOX_COORDINATES

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.workflows.execution_engine.constants import (
    CLASS_ID_KEY,
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    CONFIDENCE_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

TensorNativeDetections = Union[Detections, InstanceDetections]
KeyPointPrediction = Tuple[KeyPoints, Optional[Detections]]
TensorNativePrediction = Union[
    Detections, InstanceDetections, KeyPoints, KeyPointPrediction
]


def mask_to_indices(mask: Union[np.ndarray, Sequence[bool]]) -> List[int]:
    """Convert a boolean mask (numpy array, list, or torch tensor) into the list
    of surviving row indices, in ascending order."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().to("cpu").numpy()
    return np.nonzero(np.asarray(mask))[0].tolist()


class _RowSelection:
    """Normalised row selection shared by the ``take_*`` helpers.

    Tensor fields are selected with a torch selector (a boolean mask stays a
    boolean mask - it is moved to the field's device ONCE and reused, so a
    device-resident mask never round-trips through a python list). Python-list
    fields (``bboxes_metadata`` / RLE masks / ``key_points_metadata``) need
    concrete row positions, which are pulled to host lazily and cached - a
    selection that only touches tensor fields never pays the transfer.
    Known-empty selections use zero-row tensor views without creating a selector.
    """

    def __init__(
        self,
        selector: Optional[torch.Tensor],
        host_indices: Optional[List[int]],
        is_identity: bool,
        is_empty: bool,
        total_rows: int,
    ) -> None:
        self._selector = selector
        self._host_indices = host_indices
        self.is_identity = is_identity
        self.is_empty = is_empty
        self._total_rows = total_rows

    @classmethod
    def from_indices(cls, indices: Sequence[int], total_rows: int) -> "_RowSelection":
        indices = [int(index) for index in indices]
        if not indices:
            if total_rows == 0:
                # Selecting nothing from nothing: identity avoids even the
                # zero-row slice views.
                return cls(None, [], True, False, total_rows)
            return cls(None, [], False, True, total_rows)
        is_identity = indices == list(range(total_rows))
        selector = None if is_identity else torch.as_tensor(indices, dtype=torch.long)
        return cls(selector, indices, is_identity, False, total_rows)

    @classmethod
    def from_mask(
        cls,
        mask: Union[np.ndarray, Sequence[bool], torch.Tensor],
        total_rows: int,
    ) -> "_RowSelection":
        if not isinstance(mask, torch.Tensor) or mask.numel() != total_rows:
            # Host-born masks (and the legacy mismatched-length edge) keep the
            # exact index-list semantics of the old implementation.
            return cls.from_indices(mask_to_indices(mask), total_rows)
        mask = mask.detach().to(dtype=torch.bool).reshape(-1)
        nnz = int(mask.sum())
        if nnz == total_rows:
            return cls(None, None, True, False, total_rows)
        if nnz == 0:
            return cls(None, [], False, True, total_rows)
        return cls(mask, None, False, False, total_rows)

    def select_tensor(self, field: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return field
        if self.is_empty:
            return field[0:0]
        if self._selector.device != field.device:
            # Move once, reuse for every subsequent field of this prediction.
            self._selector = self._selector.to(field.device)
        return field[self._selector]

    def host_indices(self) -> List[int]:
        if self._host_indices is None:
            if self.is_identity:
                self._host_indices = list(range(self._total_rows))
            elif self.is_empty:
                self._host_indices = []
            else:
                # The single (lazy) device->host transfer of this selection.
                self._host_indices = (
                    torch.nonzero(self._selector, as_tuple=False)
                    .reshape(-1)
                    .cpu()
                    .tolist()
                )
        return self._host_indices


def _take_detections(
    detections: TensorNativeDetections,
    selection: _RowSelection,
) -> TensorNativeDetections:
    """Shared gather body for ``Detections`` / ``InstanceDetections``.

    Per-detection state (``bboxes_metadata``) and masks (dense torch or RLE) are
    carried over for the surviving rows; ``image_metadata`` is shared as-is.
    The surviving ``bboxes_metadata`` dicts are COPIED (not shared by reference)
    so a downstream block that mutates a selected box's metadata (e.g. assigns a
    ``tracker_id``) cannot leak the mutation back into the source prediction.
    An identity selection skips the tensor gathers entirely.
    """
    bboxes_metadata = None
    if detections.bboxes_metadata is not None:
        bboxes_metadata = [
            dict(detections.bboxes_metadata[i]) for i in selection.host_indices()
        ]
    if isinstance(detections, InstanceDetections):
        mask_field = detections.mask
        if isinstance(mask_field, InstancesRLEMasks):
            new_mask: Union[torch.Tensor, InstancesRLEMasks] = InstancesRLEMasks(
                image_size=mask_field.image_size,
                masks=[mask_field.masks[i] for i in selection.host_indices()],
            )
        else:
            new_mask = selection.select_tensor(mask_field)
        return InstanceDetections(
            xyxy=selection.select_tensor(detections.xyxy),
            class_id=selection.select_tensor(detections.class_id),
            confidence=selection.select_tensor(detections.confidence),
            mask=new_mask,
            image_metadata=detections.image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=selection.select_tensor(detections.xyxy),
        class_id=selection.select_tensor(detections.class_id),
        confidence=selection.select_tensor(detections.confidence),
        image_metadata=detections.image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _take_key_points(
    key_points: KeyPoints,
    selection: _RowSelection,
) -> KeyPoints:
    """Shared gather body for ``KeyPoints`` (instance dimension).

    The surviving ``key_points_metadata`` dicts are COPIED (not shared by
    reference) so downstream mutation cannot leak back into the source; the
    auxiliary per-instance tensors (populated by RF-DETR) are sliced in
    lockstep so lossless selections stay lossless."""
    key_points_metadata = None
    if key_points.key_points_metadata is not None:
        key_points_metadata = [
            dict(key_points.key_points_metadata[i]) for i in selection.host_indices()
        ]
    covariance = key_points.covariance
    if covariance is not None:
        covariance = selection.select_tensor(covariance)
    detection_confidence = key_points.detection_confidence
    if detection_confidence is not None:
        detection_confidence = selection.select_tensor(detection_confidence)
    return KeyPoints(
        xy=selection.select_tensor(key_points.xy),
        class_id=selection.select_tensor(key_points.class_id),
        confidence=selection.select_tensor(key_points.confidence),
        image_metadata=key_points.image_metadata,
        key_points_metadata=key_points_metadata,
        covariance=covariance,
        detection_confidence=detection_confidence,
    )


def take_detections_by_indices(
    detections: TensorNativeDetections,
    indices: Sequence[int],
) -> TensorNativeDetections:
    """Select rows of a ``Detections`` / ``InstanceDetections`` by index list."""
    selection = _RowSelection.from_indices(indices, int(detections.xyxy.shape[0]))
    return _take_detections(detections, selection)


def take_key_points_by_indices(
    key_points: KeyPoints,
    indices: Sequence[int],
) -> KeyPoints:
    """Select instances of a ``KeyPoints`` by index list."""
    selection = _RowSelection.from_indices(indices, int(key_points.xy.shape[0]))
    return _take_key_points(key_points, selection)


def _prediction_row_count(prediction: TensorNativePrediction) -> int:
    if isinstance(prediction, tuple):
        key_points, _ = prediction
        return int(key_points.xy.shape[0])
    if isinstance(prediction, KeyPoints):
        return int(prediction.xy.shape[0])
    return int(prediction.xyxy.shape[0])


def _take_prediction(
    prediction: TensorNativePrediction,
    selection: _RowSelection,
) -> TensorNativePrediction:
    if isinstance(prediction, tuple):
        key_points, detections = prediction
        sliced_key_points = _take_key_points(key_points, selection)
        sliced_detections = (
            _take_detections(detections, selection) if detections is not None else None
        )
        return sliced_key_points, sliced_detections
    if isinstance(prediction, KeyPoints):
        return _take_key_points(prediction, selection)
    return _take_detections(prediction, selection)


def take_prediction_by_indices(
    prediction: TensorNativePrediction,
    indices: Sequence[int],
) -> TensorNativePrediction:
    """Select a subset of any tensor-native prediction by index list, returning
    the same shape. For the keypoint-detection tuple, both the ``KeyPoints`` and
    the bbox ``Detections`` components are sliced consistently."""
    selection = _RowSelection.from_indices(indices, _prediction_row_count(prediction))
    return _take_prediction(prediction, selection)


def take_prediction_by_mask(
    prediction: TensorNativePrediction,
    mask: Union[np.ndarray, Sequence[bool], torch.Tensor],
) -> TensorNativePrediction:
    """Select a subset of any tensor-native prediction by boolean mask.

    A torch mask is used AS the selector: tensor fields are boolean-indexed on
    their own device with no host round trip; the row positions only reach the
    host (once, lazily) when a python-list field of the prediction needs them.
    Non-torch masks (and torch masks whose length does not match the
    prediction) keep the legacy index-list semantics."""
    selection = _RowSelection.from_mask(mask, _prediction_row_count(prediction))
    return _take_prediction(prediction, selection)


def split_key_point_prediction(
    prediction: Union[TensorNativeDetections, KeyPointPrediction],
) -> Tuple[Optional[KeyPoints], TensorNativeDetections]:
    """Normalise a block input to its bounding-box component.

    Returns ``(key_points, detections)`` where ``key_points`` is the
    ``KeyPoints`` component when the input is a keypoint-detection tuple
    (``(KeyPoints, Detections)``), otherwise ``None``. Blocks that only need
    bounding boxes operate on the returned ``detections`` and re-wrap with the
    returned ``key_points`` to preserve keypoints downstream. Raises if a
    keypoint tuple lacks its bbox component.
    """
    if isinstance(prediction, tuple):
        key_points, detections = prediction
        if detections is None:
            raise ValueError(
                "Keypoint prediction is missing the bounding-box component "
                "required by this block."
            )
        return key_points, detections
    return None, prediction


def _read_root_coordinates_shift(
    image_metadata: Optional[dict],
) -> Optional[Tuple[float, float]]:
    """Return the ``(shift_x, shift_y)`` offset that maps this prediction's local
    coordinates back to the root (workflow input) image, or ``None`` when no shift
    is needed.

    The offset is the crop origin recorded at
    ``image_metadata[ROOT_PARENT_COORDINATES_KEY] = [left_top_x, left_top_y]`` by
    ``build_native_image_metadata``. ``None`` is returned (a no-op) only when the
    key is absent or the prediction is provably root-anchored already. A zero
    offset alone is NOT that proof: a crop taken at ``(0, 0)`` still has
    crop-sized dimensions and its own lineage, and its masks still need
    re-embedding onto the root canvas - so identity additionally requires the
    image dimensions to match the root dimensions and the parent to BE the root.
    A ``(0.0, 0.0)`` return means "no coordinate shift, but the conversion
    (mask re-embedding + metadata rewrite) must still run".
    """
    if not image_metadata:
        return None
    root_coordinates = image_metadata.get(ROOT_PARENT_COORDINATES_KEY)
    if not root_coordinates:
        return None
    shift_x, shift_y = float(root_coordinates[0]), float(root_coordinates[1])
    if shift_x != 0.0 or shift_y != 0.0:
        return shift_x, shift_y
    image_dimensions = image_metadata.get(IMAGE_DIMENSIONS_KEY)
    root_dimensions = image_metadata.get(ROOT_PARENT_DIMENSIONS_KEY)
    dimensions_confirm_root = (
        image_dimensions is None
        or root_dimensions is None
        or list(image_dimensions) == list(root_dimensions)
    )
    parent_is_root = image_metadata.get(PARENT_ID_KEY) == image_metadata.get(
        ROOT_PARENT_ID_KEY
    )
    if dimensions_confirm_root and parent_is_root:
        return None
    return shift_x, shift_y


def _native_image_metadata_in_root_coordinates(image_metadata: dict) -> dict:
    """Copy ``image_metadata`` and rewrite the lineage so it describes a
    root-anchored prediction: the parent/root origin offsets collapse to ``[0, 0]``
    and the reported image dimensions become the root-parent dimensions. Mirrors
    the metadata rewrite that ``sv_detections_to_root_coordinates`` performs via
    ``attach_parent_coordinates_to_detections`` (utils.py)."""
    new_metadata = dict(image_metadata)
    root_dimensions = new_metadata.get(ROOT_PARENT_DIMENSIONS_KEY)
    if root_dimensions is not None:
        new_metadata[IMAGE_DIMENSIONS_KEY] = list(root_dimensions)
    new_metadata[PARENT_COORDINATES_KEY] = [0, 0]
    new_metadata[ROOT_PARENT_COORDINATES_KEY] = [0, 0]
    if ROOT_PARENT_ID_KEY in new_metadata:
        new_metadata[PARENT_ID_KEY] = new_metadata[ROOT_PARENT_ID_KEY]
    if root_dimensions is not None:
        new_metadata[PARENT_DIMENSIONS_KEY] = list(root_dimensions)
    return new_metadata


# Per-box geometry payloads shifted alongside xyxy on root conversion - the same
# set the crop side localizes (dynamic_crop subtracts the crop origin from
# keypoints, polygons AND oriented-box corners; root conversion adds it back).
# The numpy ``sv_detections_to_root_coordinates`` shifts the same three keys.
_GEOMETRY_KEYS_SHIFTED_TO_ROOT = (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
    ORIENTED_BOX_COORDINATES,
)


def _add_offset(
    value: Union[List, np.ndarray], offset_xy: np.ndarray
) -> Union[List, np.ndarray]:
    """Add ``offset_xy`` ([x, y]) to a coordinate container - the exact inverse of
    ``dynamic_crop/v1_tensor._subtract_offset``, with the same container- and
    dtype-preservation rules: python lists stay lists, numpy arrays stay arrays,
    integer coordinates stay integers (crop origins are integral, so the shift is
    lossless), floats keep float precision. Empty/None payloads pass through."""
    if value is None:
        return value
    was_list = isinstance(value, list)
    array = np.asarray(value)
    if array.size == 0:
        return value
    is_integer = np.issubdtype(array.dtype, np.integer)
    if is_integer:
        shifted = array.astype(np.int64) + offset_xy.astype(np.int64)
    else:
        shifted = array.astype(float) + offset_xy
    return shifted.tolist() if was_list else shifted


def _shift_bboxes_metadata_to_root_coordinates(
    bboxes_metadata: Optional[List[dict]],
    shift_x: float,
    shift_y: float,
) -> Optional[List[dict]]:
    """Copy the per-box metadata entries, shifting the geometry payloads the
    tensor-native serialiser reads back (``keypoints_xy``, ``polygon``) - the
    mirror of the per-row shifts numpy's ``sv_detections_to_root_coordinates``
    applies to ``sv.Detections.data``."""
    if bboxes_metadata is None:
        return None
    offset_xy = np.asarray([shift_x, shift_y])
    shifted_entries = []
    for entry in bboxes_metadata:
        entry = dict(entry)
        for key in _GEOMETRY_KEYS_SHIFTED_TO_ROOT:
            if key in entry:
                entry[key] = _add_offset(entry[key], offset_xy)
        shifted_entries.append(entry)
    return shifted_entries


def _shift_native_masks_to_root_coordinates(
    mask: Union[torch.Tensor, InstancesRLEMasks, None],
    shift_x: float,
    shift_y: float,
    root_dimensions: Optional[Sequence[int]],
) -> Union[torch.Tensor, InstancesRLEMasks, None]:
    """Re-anchor crop-local instance masks onto the root-image canvas, mirroring
    the numpy path's paste into an ``np.full((H_root, W_root), False)`` base.
    Dense torch masks are pasted into a zeros canvas on the same device; RLE
    masks are re-embedded without densifying the full canvas via
    ``embed_rle_masks_in_larger_canvas``."""
    if mask is None:
        return None
    if root_dimensions is None:
        raise ValueError(
            "Cannot shift tensor-native instance masks to root coordinates: the "
            f"prediction carries a non-zero root offset but no "
            f"`image_metadata['{ROOT_PARENT_DIMENSIONS_KEY}']` to size the root "
            "canvas. The producer block must attach the root lineage keys "
            "(see build_native_image_metadata)."
        )
    root_height, root_width = int(root_dimensions[0]), int(root_dimensions[1])
    offset_x, offset_y = int(shift_x), int(shift_y)
    if isinstance(mask, InstancesRLEMasks):
        return embed_rle_masks_in_larger_canvas(
            masks=mask,
            offset_xy=(offset_x, offset_y),
            target_size_hw=(root_height, root_width),
        )
    _, mask_height, mask_width = mask.shape
    if offset_x < 0 or offset_y < 0:
        raise ValueError(
            f"Cannot shift tensor-native instance masks to root coordinates: got "
            f"negative crop offset ({offset_x}, {offset_y}); offsets must be "
            "non-negative."
        )
    if offset_x + mask_width > root_width or offset_y + mask_height > root_height:
        raise ValueError(
            f"Crop-local masks of size (h={mask_height}, w={mask_width}) at offset "
            f"(x={offset_x}, y={offset_y}) do not fit into the root canvas of size "
            f"(H={root_height}, W={root_width})."
        )
    anchored = torch.zeros(
        (mask.shape[0], root_height, root_width), dtype=mask.dtype, device=mask.device
    )
    anchored[:, offset_y : offset_y + mask_height, offset_x : offset_x + mask_width] = (
        mask
    )
    return anchored


def _shift_native_detections_to_root_coordinates(
    detections: TensorNativeDetections,
    shift_x: float,
    shift_y: float,
) -> TensorNativeDetections:
    shift = torch.as_tensor(
        [shift_x, shift_y, shift_x, shift_y],
        dtype=detections.xyxy.dtype,
        device=detections.xyxy.device,
    )
    shifted_xyxy = detections.xyxy + shift
    # Root canvas dims must be read BEFORE the metadata rewrite below collapses
    # the lineage to the root frame ([0, 0] offsets, root dims everywhere).
    root_dimensions = (
        detections.image_metadata.get(ROOT_PARENT_DIMENSIONS_KEY)
        if detections.image_metadata is not None
        else None
    )
    image_metadata = (
        _native_image_metadata_in_root_coordinates(detections.image_metadata)
        if detections.image_metadata is not None
        else None
    )
    bboxes_metadata = _shift_bboxes_metadata_to_root_coordinates(
        bboxes_metadata=detections.bboxes_metadata,
        shift_x=shift_x,
        shift_y=shift_y,
    )
    if isinstance(detections, InstanceDetections):
        return InstanceDetections(
            xyxy=shifted_xyxy,
            class_id=detections.class_id,
            confidence=detections.confidence,
            mask=_shift_native_masks_to_root_coordinates(
                mask=detections.mask,
                shift_x=shift_x,
                shift_y=shift_y,
                root_dimensions=root_dimensions,
            ),
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=shifted_xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _shift_native_key_points_to_root_coordinates(
    key_points: KeyPoints,
    shift_x: float,
    shift_y: float,
) -> KeyPoints:
    shift = torch.as_tensor(
        [shift_x, shift_y],
        dtype=key_points.xy.dtype,
        device=key_points.xy.device,
    )
    image_metadata = (
        _native_image_metadata_in_root_coordinates(key_points.image_metadata)
        if key_points.image_metadata is not None
        else None
    )
    key_points_metadata = (
        [dict(entry) for entry in key_points.key_points_metadata]
        if key_points.key_points_metadata is not None
        else None
    )
    return KeyPoints(
        xy=key_points.xy + shift,
        class_id=key_points.class_id,
        confidence=key_points.confidence,
        image_metadata=image_metadata,
        key_points_metadata=key_points_metadata,
        # Auxiliary RF-DETR tensors ride along unchanged - a pure translation
        # affects neither positional covariance nor detection confidence.
        covariance=key_points.covariance,
        detection_confidence=key_points.detection_confidence,
    )


def native_detections_to_root_coordinates(
    prediction: TensorNativePrediction,
) -> TensorNativePrediction:
    """Shift a tensor-native prediction from its crop-local coordinates back to the
    root (workflow input) image coordinates, returning a copy.

    Reads the crop origin from
    ``image_metadata[ROOT_PARENT_COORDINATES_KEY] = [shift_x, shift_y]`` and adds
    ``[shift_x, shift_y, shift_x, shift_y]`` to ``xyxy`` (and ``[shift_x, shift_y]``
    to keypoint ``xy``). Instance masks are re-anchored onto the root-image canvas
    (dense torch masks pasted into a zeros canvas; RLE masks re-embedded without
    densifying), and the per-box ``keypoints_xy`` / ``polygon`` payloads in
    ``bboxes_metadata`` are shifted - matching the mask paste and the ``data``-key
    shifts of numpy's ``sv_detections_to_root_coordinates``. The keypoint-detection
    tuple ``(KeyPoints, Detections)`` has both components shifted consistently.
    This is the tensor-native mirror of ``sv_detections_to_root_coordinates``
    (utils.py), used by the execution-engine output constructor when an output is
    requested in PARENT coordinates.

    No-op (the input is returned unchanged) when the prediction carries no root
    offset — i.e. the key is absent or the shift is ``[0, 0]`` (already
    root-anchored, e.g. a model run directly on the workflow input image).
    """
    if isinstance(prediction, tuple):
        key_points, detections = prediction
        shift = _read_root_coordinates_shift(
            key_points.image_metadata if key_points is not None else None
        )
        if shift is None and detections is not None:
            shift = _read_root_coordinates_shift(detections.image_metadata)
        if shift is None:
            return prediction
        shift_x, shift_y = shift
        shifted_key_points = (
            _shift_native_key_points_to_root_coordinates(key_points, shift_x, shift_y)
            if key_points is not None
            else None
        )
        shifted_detections = (
            _shift_native_detections_to_root_coordinates(detections, shift_x, shift_y)
            if detections is not None
            else None
        )
        return shifted_key_points, shifted_detections
    if isinstance(prediction, KeyPoints):
        shift = _read_root_coordinates_shift(prediction.image_metadata)
        if shift is None:
            return prediction
        return _shift_native_key_points_to_root_coordinates(prediction, *shift)
    shift = _read_root_coordinates_shift(prediction.image_metadata)
    if shift is None:
        return prediction
    return _shift_native_detections_to_root_coordinates(prediction, *shift)


def instance_mask_to_numpy(
    detections: InstanceDetections,
    index: int,
) -> np.ndarray:
    """Materialise a single instance's mask as a 2-D ``np.ndarray`` of bool
    ``(H, W)``. RLE masks are decoded one instance at a time so the full stack
    is never materialised at once (the same convention as the serialiser).

    The dense seg-adapter mask is already ``torch.bool`` (binarised via
    ``.gt_(threshold).to(dtype=torch.bool)`` in the post-processing path), so no
    ``.astype(bool)`` is applied — the serialiser's dense branch makes the same
    assumption.
    """
    mask = detections.mask
    if isinstance(mask, InstancesRLEMasks):
        return coco_rle_masks_to_numpy_mask(
            InstancesRLEMasks(image_size=mask.image_size, masks=[mask.masks[index]])
        )[0]
    return mask[index].detach().to("cpu").numpy()


def build_native_image_metadata(
    image: WorkflowImageData,
    class_names: Dict[int, str],
    prediction_type: str,
    inference_id: Optional[str] = None,
) -> dict:
    """Build the per-image ``image_metadata`` dict carried by a tensor-native
    ``Detections`` prediction produced by a model block.

    Holds the ``class_id -> name`` map (required by the tensor-native serialiser),
    the image dimensions, and the parent/root lineage needed for crop-aware
    coordinate recovery downstream. Mirrors the convention in
    ``formatters/vlm_as_detector/v1_tensor.py``, but with a parametrised
    ``prediction_type``. The image shape is read without forcing a device->host
    materialization (so tensor-only inputs stay on device).
    """
    height, width = image._read_shape_without_materialization()
    parent = image.parent_metadata
    root = image.workflow_root_ancestor_metadata
    parent_coordinates = parent.origin_coordinates
    root_coordinates = root.origin_coordinates
    metadata = {
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: prediction_type,
        IMAGE_DIMENSIONS_KEY: [height, width],
        PARENT_ID_KEY: parent.parent_id,
        PARENT_COORDINATES_KEY: [
            parent_coordinates.left_top_x,
            parent_coordinates.left_top_y,
        ],
        PARENT_DIMENSIONS_KEY: [
            parent_coordinates.origin_height,
            parent_coordinates.origin_width,
        ],
        ROOT_PARENT_ID_KEY: root.parent_id,
        ROOT_PARENT_COORDINATES_KEY: [
            root_coordinates.left_top_x,
            root_coordinates.left_top_y,
        ],
        ROOT_PARENT_DIMENSIONS_KEY: [
            root_coordinates.origin_height,
            root_coordinates.origin_width,
        ],
    }
    if inference_id is not None:
        metadata[INFERENCE_ID_KEY] = inference_id
    return metadata


def attach_native_detection_metadata(
    detections: Detections,
    image: WorkflowImageData,
    class_names: Dict[int, str],
    prediction_type: str,
    inference_id: Optional[str] = None,
) -> Detections:
    """LOCAL-path helper for model blocks that get a native ``Detections`` straight
    from an inference_models adapter's ``run_tensor_native_inference``.

    The adapter fills ``xyxy`` / ``class_id`` / ``confidence`` (and sometimes a few
    per-detection ``bboxes_metadata`` fields) but knows nothing about the workflow
    image lineage. This attaches the workflow ``image_metadata`` and guarantees each
    detection carries a ``detection_id`` (generated when missing), preserving any
    keys the model already set (e.g. EasyOCR's per-box ``text``). Mutates and
    returns the same object.
    """
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=class_names,
        prediction_type=prediction_type,
        inference_id=inference_id,
    )
    number_of_detections = int(detections.xyxy.shape[0])
    if number_of_detections == 0:
        detections.bboxes_metadata = None
        return detections
    existing = detections.bboxes_metadata
    bboxes_metadata = []
    for index in range(number_of_detections):
        entry = (
            dict(existing[index])
            if existing is not None and index < len(existing)
            else {}
        )
        entry.setdefault(DETECTION_ID_KEY, str(uuid4()))
        bboxes_metadata.append(entry)
    detections.bboxes_metadata = bboxes_metadata
    return detections


def native_detections_from_inference_predictions(
    image: WorkflowImageData,
    predictions: List[dict],
    prediction_type: str,
    class_names: Optional[Dict[int, str]] = None,
    inference_id: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Detections:
    """REMOTE-path helper: build a native ``Detections`` from standard inference
    object-detection prediction dicts (center ``x``/``y``/``width``/``height``,
    ``confidence``, ``class_id``, optional ``class`` name and ``detection_id``).

    Boxes are converted from center form to corner ``xyxy``. When ``class_names``
    is not supplied it is derived from the predictions' own ``class_id``/``class``
    pairs so the serialiser can resolve every id. A ``detection_id`` is preserved
    when present, otherwise generated. When ``device`` is not supplied the output
    tensors are pinned to ``WORKFLOWS_IMAGE_TENSOR_DEVICE`` (so a REMOTE/HTTP
    detection result lands on the same device as the LOCAL-path siblings rather
    than silently staying on CPU).
    """
    if device is None:
        device = WORKFLOWS_IMAGE_TENSOR_DEVICE
    xyxy: List[List[float]] = []
    class_id: List[int] = []
    confidence: List[float] = []
    bboxes_metadata: List[dict] = []
    derived_class_names: Dict[int, str] = {}
    for prediction in predictions:
        center_x = float(prediction[X_KEY])
        center_y = float(prediction[Y_KEY])
        box_width = float(prediction[WIDTH_KEY])
        box_height = float(prediction[HEIGHT_KEY])
        xyxy.append(
            [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ]
        )
        prediction_class_id = int(prediction.get(CLASS_ID_KEY, 0))
        class_id.append(prediction_class_id)
        confidence.append(float(prediction.get(CONFIDENCE_KEY, 1.0)))
        if CLASS_NAME_KEY in prediction:
            derived_class_names[prediction_class_id] = str(prediction[CLASS_NAME_KEY])
        bboxes_metadata.append(
            {DETECTION_ID_KEY: str(prediction.get(DETECTION_ID_KEY) or uuid4())}
        )
    number_of_detections = len(xyxy)
    resolved_class_names = (
        class_names if class_names is not None else derived_class_names
    )
    image_metadata = build_native_image_metadata(
        image=image,
        class_names=resolved_class_names,
        prediction_type=prediction_type,
        inference_id=inference_id,
    )
    return Detections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32, device=device).reshape(-1, 4),
        class_id=torch.as_tensor(class_id, dtype=torch.long, device=device).reshape(-1),
        confidence=torch.as_tensor(
            confidence, dtype=torch.float32, device=device
        ).reshape(-1),
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata if number_of_detections > 0 else None,
    )


def _column_to_runs(column: np.ndarray) -> List[Tuple[int, int]]:
    """Split a single 1-D column into (value, run_length) pairs of consecutive equal values."""
    if column.shape[0] == 0:
        return []
    change_points = np.flatnonzero(column[1:] != column[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [column.shape[0]]))
    return [
        (int(column[start]), int(end - start))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _embed_single_mask_counts(
    column_major_slice: np.ndarray,
    offset_xy: Tuple[int, int],
    target_size_hw: Tuple[int, int],
) -> List[int]:
    """Build the column-major uncompressed COCO counts for one slice placed onto the canvas.

    ``column_major_slice`` is the dense (h, w) slice already laid out as a list of its w
    columns (each of height h). The big (H, W) canvas is never densified: we emit run
    lengths directly and merge adjacent equal-value runs.
    """
    h, w = column_major_slice.shape
    x0, y0 = offset_xy
    target_h, target_w = target_size_hw

    bottom_zeros = target_h - h - y0
    leading_zero_pixels = x0 * target_h
    trailing_zero_pixels = (target_w - w - x0) * target_h

    # (value, run_length) pairs in column-major order across the whole canvas.
    runs: List[Tuple[int, int]] = []
    if leading_zero_pixels > 0:
        runs.append((0, leading_zero_pixels))
    for column_index in range(w):
        column = column_major_slice[:, column_index]
        if y0 > 0:
            runs.append((0, y0))
        runs.extend(_column_to_runs(column))
        if bottom_zeros > 0:
            runs.append((0, bottom_zeros))
    if trailing_zero_pixels > 0:
        runs.append((0, trailing_zero_pixels))

    # COCO uncompressed counts are alternating run lengths starting with a zero run.
    merged_counts: List[int] = [0]
    current_value = 0
    for value, length in runs:
        if length == 0:
            continue
        if value == current_value:
            merged_counts[-1] += length
        else:
            merged_counts.append(length)
            current_value = value
    return merged_counts


def embed_rle_masks_in_larger_canvas(
    masks: InstancesRLEMasks,
    offset_xy: Tuple[int, int],
    target_size_hw: Tuple[int, int],
) -> InstancesRLEMasks:
    """Place N slice-resolution RLE masks onto a larger all-zeros canvas, in RLE.

    Each input mask sits at slice resolution ``masks.image_size == (h, w)``. The returned
    masks live on a ``(H, W)`` canvas with the slice's top-left at ``offset_xy == (x0, y0)``;
    everything outside the slice is zero. COCO RLE here is column-major (fortran), matching
    ``torch_mask_to_coco_rle``. The big canvas is never densified -- only the small slice is
    decoded to dense and run lengths are emitted directly onto the canvas.

    Lives on the workflows side (rather than ``inference_models``) so the SAHI stitch block
    introduces no new dependency on ``inference_models`` internals.
    """
    h, w = masks.image_size
    x0, y0 = offset_xy
    target_h, target_w = target_size_hw

    if x0 < 0 or y0 < 0:
        raise ValueError(
            f"embed_rle_masks_in_larger_canvas got negative offset_xy={offset_xy}; "
            f"offsets must be non-negative."
        )
    if x0 + w > target_w or y0 + h > target_h:
        raise ValueError(
            f"Slice of size (h={h}, w={w}) at offset_xy=(x0={x0}, y0={y0}) does not fit "
            f"into target canvas of size (H={target_h}, W={target_w}): requires "
            f"x0+w={x0 + w} <= W={target_w} and y0+h={y0 + h} <= H={target_h}."
        )

    if len(masks.masks) == 0:
        return InstancesRLEMasks(image_size=(target_h, target_w), masks=[])

    # Decode only the small (h, w) slices to dense; the big canvas stays in run-list form.
    dense_slices = coco_rle_masks_to_numpy_mask(masks).astype(np.uint8)

    embedded: List[bytes] = []
    for dense_slice in dense_slices:
        # Lay the (h, w) slice out as columns (column j of height h) -> shape (h, w).
        counts = _embed_single_mask_counts(
            column_major_slice=dense_slice,
            offset_xy=offset_xy,
            target_size_hw=target_size_hw,
        )
        rle = mask_utils.frPyObjects(
            {"counts": counts, "size": [target_h, target_w]}, target_h, target_w
        )
        embedded.append(rle["counts"])

    return InstancesRLEMasks(image_size=(target_h, target_w), masks=embedded)


def build_native_key_points(
    per_instance_xy: List[Optional[List[List[float]]]],
    per_instance_confidence: List[Optional[List[float]]],
    object_class_ids: List[Any],
    image_metadata: dict,
) -> KeyPoints:
    """Rebuild a padded native ``KeyPoints`` from per-instance keypoint lists (the
    flattened ``keypoints_xy`` / ``keypoints_confidence`` shape the keypoint
    producer writes into ``bboxes_metadata``). Mirrors
    ``keypoint_detection/v3_tensor._native_key_points_from_inference_predictions``:
    ragged per-instance keypoint counts are zero-padded to a uniform ``K`` with
    confidence ``0.0`` in the padding rows. ``class_id`` is the per-instance
    *object* class id (one per skeleton), matching the bbox ``Detections.class_id``.

    Extracted (verbatim) from ``fusion/detections_list_rollup/v1_tensor.py`` so the
    dynamic-block representation boundary can rebuild the ``(KeyPoints, Detections)``
    tuple the visualizer siblings require.
    """
    number_of_instances = len(object_class_ids)
    normalised_xy = [list(xy) if xy else [] for xy in per_instance_xy]
    normalised_confidence = [
        list(conf) if conf else [] for conf in per_instance_confidence
    ]
    max_key_points = max((len(xy) for xy in normalised_xy), default=0)
    xy_tensor = torch.zeros(
        (number_of_instances, max_key_points, 2), dtype=torch.float32
    )
    confidence_tensor = torch.zeros(
        (number_of_instances, max_key_points), dtype=torch.float32
    )
    for index in range(number_of_instances):
        keypoint_count = len(normalised_xy[index])
        if keypoint_count > 0:
            xy_tensor[index, :keypoint_count] = torch.as_tensor(
                normalised_xy[index], dtype=torch.float32
            )
        confidence_count = len(normalised_confidence[index])
        if confidence_count > 0:
            confidence_tensor[index, :confidence_count] = torch.as_tensor(
                normalised_confidence[index], dtype=torch.float32
            )
    class_id_tensor = torch.as_tensor(
        [int(value) for value in object_class_ids], dtype=torch.long
    ).reshape(-1)
    return KeyPoints(
        xy=xy_tensor,
        class_id=class_id_tensor,
        confidence=confidence_tensor,
        image_metadata=image_metadata,
    )
