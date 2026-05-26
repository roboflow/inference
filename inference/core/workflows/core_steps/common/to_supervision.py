"""Workflows-side conversion between `inference_models` native prediction
types and `sv.Detections` / dicts, with `image_metadata` and
`bboxes_metadata` folded into the per-detection `sv.Detections.data` dict.

Used by Phase 5 consumer block tensor siblings that wrap existing sv-shaped
implementations: convert at the input boundary, run the numpy logic, convert
back (when downstream is tensor-aware). The materialization cost is paid by
the block itself — no engine coercion per `[ITERATE PRED.6]`.

`inference_models.{Detections,InstanceDetections,KeyPoints}.to_supervision()`
does NOT propagate metadata, so this module owns the merge.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import supervision as sv
import torch

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
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask


PredictionWithMetadata = Union[
    Detections,
    InstanceDetections,
    KeyPoints,
    MultiLabelClassificationPrediction,
    SemanticSegmentationResult,
]


TENSOR_NATIVE_PREDICTION_KEY = "__tensor_native_prediction__"


def detections_to_supervision_with_metadata(pred: Detections) -> sv.Detections:
    sv_det = pred.to_supervision()
    _fold_metadata(sv_det, pred.image_metadata, pred.bboxes_metadata)
    return sv_det


def build_dual_detections(pred: Detections) -> sv.Detections:
    """Build a dual-representation prediction for tensor-mode producers.

    Returns an `sv.Detections` (so every existing sv-shaped consumer works
    unchanged) with the original `inference_models.Detections` stashed in
    `.data[TENSOR_NATIVE_PREDICTION_KEY]`. Tensor-aware consumers read the
    native source from there to avoid re-materialisation; everything else
    just uses the sv interface.

    `image_metadata` is broadcast per-detection and `bboxes_metadata` is
    folded per-detection into `.data` — identical to the existing
    `detections_to_supervision_with_metadata` semantics.
    """
    sv_det = detections_to_supervision_with_metadata(pred)
    sv_det.data[TENSOR_NATIVE_PREDICTION_KEY] = pred
    return sv_det


def build_dual_instance_detections(pred: InstanceDetections) -> sv.Detections:
    """Same as `build_dual_detections` for instance segmentation. RLE masks
    are rasterised to dense numpy at this boundary so the sv.Detections
    consumers see `.mask` as a (n, H, W) bool array. Tensor-aware
    consumers reach `.data[TENSOR_NATIVE_PREDICTION_KEY].mask` to recover
    the RLE form."""
    sv_det = instance_detections_to_supervision_with_metadata(pred)
    sv_det.data[TENSOR_NATIVE_PREDICTION_KEY] = pred
    return sv_det


def build_dual_key_points(pred: KeyPoints) -> sv.KeyPoints:
    """`sv.KeyPoints` form of the tensor `KeyPoints`. Tensor source
    stashed in `.data` for tensor-aware consumers."""
    sv_kp = key_points_to_supervision_with_metadata(pred)
    sv_kp.data[TENSOR_NATIVE_PREDICTION_KEY] = pred
    return sv_kp


def build_dual_classification_dicts(
    pred: ClassificationPrediction,
    *,
    class_names: Optional[Dict[int, str]] = None,
) -> List[dict]:
    """Per-image list of numpy-mode classification dicts. Each dict
    additionally carries the per-image slice of the tensor source under
    TENSOR_NATIVE_PREDICTION_KEY."""
    dicts = classification_prediction_to_dict_per_image(
        pred, class_names=class_names
    )
    # Stash a per-image ClassificationPrediction slice so tensor-aware
    # downstream blocks can keep operating on the tensor form.
    class_ids = pred.class_id
    confidences = pred.confidence
    for i, entry in enumerate(dicts):
        sliced = ClassificationPrediction(
            class_id=class_ids[i : i + 1],
            confidence=confidences[i : i + 1],
            images_metadata=[
                (pred.images_metadata or [{}] * len(dicts))[i]
            ],
        )
        entry[TENSOR_NATIVE_PREDICTION_KEY] = sliced
    return dicts


def build_dual_multi_label_dict(
    pred: MultiLabelClassificationPrediction,
    *,
    class_names: Optional[Dict[int, str]] = None,
) -> dict:
    """Numpy-mode multi-label dict with the tensor source stashed in it."""
    out = multi_label_classification_to_dict(pred, class_names=class_names)
    out[TENSOR_NATIVE_PREDICTION_KEY] = pred
    return out


def build_dual_semantic_segmentation(
    pred: SemanticSegmentationResult,
) -> SemanticSegmentationResult:
    """SemSeg's numpy-mode output is itself a specialised structure (not
    sv.Detections) — passthrough the `inference_models` native form.
    Tensor backing is the value itself; consumers that need the
    numpy-mode rendering call to_supervision() ad hoc."""
    return pred


def instance_detections_to_supervision_with_metadata(
    pred: InstanceDetections,
) -> sv.Detections:
    """Materialises RLE masks to dense numpy at this boundary — this is the
    intended point where the dense cost is paid in tensor-mode workflows."""
    if isinstance(pred.mask, InstancesRLEMasks):
        dense_mask = coco_rle_masks_to_numpy_mask(pred.mask)
        sv_det = sv.Detections(
            xyxy=pred.xyxy.detach().cpu().numpy(),
            class_id=pred.class_id.detach().cpu().numpy(),
            confidence=pred.confidence.detach().cpu().numpy(),
            mask=dense_mask,
        )
    else:
        sv_det = pred.to_supervision()
    _fold_metadata(sv_det, pred.image_metadata, pred.bboxes_metadata)
    return sv_det


def key_points_to_supervision_with_metadata(pred: KeyPoints) -> sv.KeyPoints:
    sv_kp = pred.to_supervision()
    # sv.KeyPoints has its own .data dict (same convention as Detections);
    # write image_metadata into per-instance arrays.
    n = len(sv_kp.class_id) if sv_kp.class_id is not None else 0
    if pred.image_metadata:
        for key, value in pred.image_metadata.items():
            sv_kp.data[key] = np.array([value] * n, dtype=object)
    if pred.key_points_metadata:
        keys = {k for d in pred.key_points_metadata for k in d}
        for key in keys:
            sv_kp.data[key] = np.array(
                [d.get(key) for d in pred.key_points_metadata], dtype=object
            )
    return sv_kp


def multi_label_classification_to_dict(
    pred: MultiLabelClassificationPrediction,
    *,
    class_names: Optional[Dict[int, str]] = None,
) -> dict:
    """Numpy-mode multi-label classification emits a dict-keyed structure.
    Mirror it for backwards compat."""
    names = class_names or {}
    predictions: Dict[str, dict] = {}
    predicted_classes: List[str] = []
    class_ids = pred.class_ids.detach().cpu().tolist()
    confidences = pred.confidence.detach().cpu().tolist()
    for class_id, conf in zip(class_ids, confidences):
        name = names.get(int(class_id), str(class_id))
        predictions[name] = {"class_id": int(class_id), "confidence": float(conf)}
        predicted_classes.append(name)
    out: dict = {"predictions": predictions, "predicted_classes": predicted_classes}
    if pred.image_metadata:
        out.update(pred.image_metadata)
    return out


def classification_prediction_to_dict_per_image(
    pred: ClassificationPrediction,
    *,
    class_names: Optional[Dict[int, str]] = None,
) -> List[dict]:
    """Numpy-mode single-label classification emits a list-of-dicts (one per
    image)."""
    names = class_names or {}
    class_ids = pred.class_id.detach().cpu().tolist()
    confidences = pred.confidence.detach().cpu().tolist()
    images_meta = pred.images_metadata or [{} for _ in class_ids]
    out: List[dict] = []
    for class_id, conf, image_meta in zip(class_ids, confidences, images_meta):
        name = names.get(int(class_id), str(class_id))
        entry = {
            "top": name,
            "confidence": float(conf),
            "class_id": int(class_id),
            "class": name,
        }
        if image_meta:
            entry.update(image_meta)
        out.append(entry)
    return out


def sv_detections_to_inference_models_detections(
    sv_det: sv.Detections,
    *,
    inherit_image_metadata: Optional[dict] = None,
) -> Detections:
    """Reverse direction: build `inference_models.Detections` from an
    `sv.Detections`. Used by consumer block tensor siblings that wrap an
    sv-shaped implementation and emit predictions for downstream tensor
    consumers.

    Per-detection keys in `sv_det.data` (DETECTION_ID, PARENT_ID, …) land in
    `bboxes_metadata`. Optional `inherit_image_metadata` (e.g. carrying
    inference_id / model_id / class_names from the upstream prediction)
    is preserved in `image_metadata`.
    """
    n = len(sv_det)
    if n == 0:
        xyxy_t = torch.zeros((0, 4), dtype=torch.float32)
        class_id_t = torch.zeros((0,), dtype=torch.int64)
        confidence_t = torch.zeros((0,), dtype=torch.float32)
        bboxes_metadata = None
    else:
        xyxy_t = torch.from_numpy(np.asarray(sv_det.xyxy, dtype=np.float32))
        class_id_t = torch.from_numpy(
            np.asarray(
                sv_det.class_id if sv_det.class_id is not None else np.zeros(n),
                dtype=np.int64,
            )
        )
        confidence_t = torch.from_numpy(
            np.asarray(
                sv_det.confidence if sv_det.confidence is not None else np.zeros(n),
                dtype=np.float32,
            )
        )
        bboxes_metadata = [
            {key: sv_det.data[key][i] for key in sv_det.data} for i in range(n)
        ]
    return Detections(
        xyxy=xyxy_t,
        class_id=class_id_t,
        confidence=confidence_t,
        image_metadata=dict(inherit_image_metadata) if inherit_image_metadata else None,
        bboxes_metadata=bboxes_metadata,
    )


def _fold_metadata(
    sv_det: sv.Detections,
    image_metadata: Optional[dict],
    bboxes_metadata: Optional[List[dict]],
) -> None:
    n = len(sv_det)
    if image_metadata:
        for key, value in image_metadata.items():
            sv_det.data[key] = np.array([value] * n, dtype=object)
    if bboxes_metadata:
        keys = {k for d in bboxes_metadata for k in d}
        for key in keys:
            sv_det.data[key] = np.array(
                [d.get(key) for d in bboxes_metadata], dtype=object
            )


# ---------------------------------------------------------------------------
# Convenience: detect inference_models-native input vs sv.Detections
# ---------------------------------------------------------------------------


def to_supervision_with_metadata(value: Any) -> Any:
    """Generic dispatch for consumer block tensor siblings. If `value` is an
    inference_models native prediction, convert it to the corresponding
    `sv.*` shape with metadata folded. Otherwise return unchanged
    (passthrough for already-sv inputs and unrelated types)."""
    if isinstance(value, Detections):
        return detections_to_supervision_with_metadata(value)
    if isinstance(value, InstanceDetections):
        return instance_detections_to_supervision_with_metadata(value)
    if isinstance(value, KeyPoints):
        return key_points_to_supervision_with_metadata(value)
    return value
