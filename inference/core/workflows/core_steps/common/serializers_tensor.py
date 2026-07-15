from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
import torch

from inference.core.workflows.core_steps.common.serializers import (
    _attach_parent_metadata_to_detection_dict,
    mask_to_polygon,
    serialise_image,
)
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections as _serialise_legacy_sv_detections,
)
from inference.core.workflows.core_steps.common.serializers import serialize_timestamp
from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_INFERENCE_RESPONSE,
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_INFERENCE_RESPONSE,
    AREA_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
    CLASS_ID_KEY,
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    CLASSIFICATION_STYLE_FORMATTER,
    CLASSIFICATION_STYLE_KEY,
    CLASSIFICATION_STYLE_MODEL,
    CONFIDENCE_KEY,
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PARENT_ORIGIN_KEY,
    PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE,
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY,
    POLYGON_KEY_IN_INFERENCE_RESPONSE,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    RLE_MASK_KEY_IN_INFERENCE_RESPONSE,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    ROOT_PARENT_ORIGIN_KEY,
    SERIALISE_POLYGONS_KEY,
    SMOOTHED_SPEED_KEY_IN_INFERENCE_RESPONSE,
    SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
    SMOOTHED_VELOCITY_KEY_IN_INFERENCE_RESPONSE,
    SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
    SPEED_KEY_IN_INFERENCE_RESPONSE,
    SPEED_KEY_IN_SV_DETECTIONS,
    TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
    TRACKER_ID_KEY,
    VELOCITY_KEY_IN_INFERENCE_RESPONSE,
    VELOCITY_KEY_IN_SV_DETECTIONS,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

TensorNativeDetections = Union[Detections, InstanceDetections]

# D4 discriminator signals. A single native ``ClassificationPrediction`` /
# ``MultiLabelClassificationPrediction`` type is emitted by producers whose flag-OFF
# JSON has two INCOMPATIBLE shapes (see ``serialise_native_classification``): the
# numpy ``*InferenceResponse`` "model" shape and the hand-built ``vlm_as_classifier``
# "formatter" shape. Since the tensors cannot self-describe which shape they want,
# we infer it from metadata the producers already attach - the model producers carry
# a confidence threshold and/or ``root_parent_id`` and/or ``time``; the vlm formatter
# carries none of these.
_CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY = "classification_confidence_threshold"
_TIME_KEY = "time"


def serialise_sv_detections(
    detections: TensorNativeDetections,
) -> dict:
    """Serialise native ``inference_models.Detections`` / ``InstanceDetections`` into
    the response dict shape produced by the numpy ``serialise_sv_detections``.

    The name is retained for loader symbol-swap compatibility (it is load-bearing) -
    despite the ``sv`` prefix it now consumes native tensor objects, not
    ``sv.Detections``.
    """
    serialized_image_metadata, serialized_detections, _ = _serialise_sv_detections(
        detections=detections
    )
    return {"image": serialized_image_metadata, "predictions": serialized_detections}


def _serialise_sv_detections(
    detections: TensorNativeDetections,
    emit_polygons: Optional[bool] = None,
) -> Tuple[dict, List[dict], List[int]]:
    """Shared core for the detection serialisers.

    Returns the serialised image metadata, the per-box prediction dicts (after the
    polygon-None skip) and the ORIGINAL row index of each surviving prediction so
    downstream serialisers (e.g. RLE) can re-align masks despite skipped instances.

    ``emit_polygons`` controls the per-instance mask -> ``points`` polygon work
    (the RLE-decode + ``mask_to_polygon``/``findContours`` cost):

    * ``None`` (default): resolve from ``image_metadata[SERIALISE_POLYGONS_KEY]``,
      defaulting to ``True`` when the key is ABSENT. Absent-key behaviour is
      byte-identical to the historical serialiser (polygons computed & emitted,
      instances with an empty contour dropped). A producer that sets the key to
      ``False`` gets the polygon work skipped: instances are kept (no drop) and
      no ``points`` field is emitted.
    * ``False`` (forced): callers that discard the polygon anyway (the native RLE
      serialiser) pass this to avoid the wasted O(N*H*W) computation. Overrides the
      metadata flag.
    """
    if not isinstance(detections, (Detections, InstanceDetections)):
        # C2: this typed path only accepts the native tensor objects. ``sv.Detections``
        # is routed to the numpy serialiser by ``serialize_wildcard_kind`` instead, so
        # the message must NOT claim this function accepts it.
        raise ValueError(
            f"serialise_sv_detections(...) expected `inference_models.Detections` "
            f"or `inference_models.InstanceDetections`, got {type(detections)}."
        )
    image_metadata = detections.image_metadata or {}
    if emit_polygons is None:
        # Absent key -> True -> historical behaviour (byte-identical). Producer may
        # set it to False to disable polygon emission.
        emit_polygons = image_metadata.get(SERIALISE_POLYGONS_KEY, True)
    detections_number = int(detections.xyxy.shape[0])
    bboxes_metadata = detections.bboxes_metadata
    if bboxes_metadata is None:
        bboxes_metadata = [{} for _ in range(detections_number)]
    class_names_mapping = None
    if detections_number > 0:
        class_names_mapping = image_metadata.get(CLASS_NAMES_KEY)
        if class_names_mapping is None:
            raise ValueError(
                f"Serialising tensor-native detections, but "
                f"`image_metadata['{CLASS_NAMES_KEY}']` is missing - the producer "
                f"block must attach the class_id -> name mapping."
            )
    boxes = detections.xyxy.detach().cpu().tolist()
    confidences = detections.confidence.detach().cpu().tolist()
    class_ids = [int(value) for value in detections.class_id.detach().cpu().tolist()]
    serialized_detections = []
    kept_indices: List[int] = []
    for index in range(detections_number):
        data = bboxes_metadata[index]
        detection_dict = {}
        x1, y1, x2, y2 = (float(coordinate) for coordinate in boxes[index])
        detection_dict[WIDTH_KEY] = abs(x2 - x1)
        detection_dict[HEIGHT_KEY] = abs(y2 - y1)
        detection_dict[X_KEY] = x1 + detection_dict[WIDTH_KEY] / 2
        detection_dict[Y_KEY] = y1 + detection_dict[HEIGHT_KEY] / 2
        detection_dict[CONFIDENCE_KEY] = float(confidences[index])
        detection_dict[CLASS_ID_KEY] = class_ids[index]
        if isinstance(detections, InstanceDetections) and emit_polygons:
            polygon = _resolve_instance_polygon(
                mask=detections.mask, index=index, data=data
            )
            if polygon is None:
                # ignoring the whole instance - mirrors the numpy serialiser
                continue
            detection_dict[POLYGON_KEY] = []
            for x, y in polygon:
                detection_dict[POLYGON_KEY].append(
                    {
                        X_KEY: float(x),
                        Y_KEY: float(y),
                    }
                )
        if data.get("tracker_id") is not None:
            detection_dict[TRACKER_ID_KEY] = int(data["tracker_id"])
        # C1: a producer may carry an arbitrary per-box label on the box metadata;
        # prefer it, otherwise fall back to the class_id -> name mapping.
        if CLASS_NAME_KEY in data:
            detection_dict[CLASS_NAME_KEY] = str(data[CLASS_NAME_KEY])
        else:
            detection_dict[CLASS_NAME_KEY] = _resolve_class_name(
                class_id=class_ids[index], class_names_mapping=class_names_mapping
            )
        if DETECTION_ID_KEY not in data:
            raise ValueError(
                f"Serialising tensor-native detections, but "
                f"`bboxes_metadata['{DETECTION_ID_KEY}']` is missing for detection "
                f"at index {index} - the producer block must attach it."
            )
        detection_dict[DETECTION_ID_KEY] = str(data[DETECTION_ID_KEY])
        if PATH_DEVIATION_KEY_IN_SV_DETECTIONS in data:
            detection_dict[PATH_DEVIATION_KEY_IN_INFERENCE_RESPONSE] = data[
                PATH_DEVIATION_KEY_IN_SV_DETECTIONS
            ]
        if TIME_IN_ZONE_KEY_IN_SV_DETECTIONS in data:
            detection_dict[TIME_IN_ZONE_KEY_IN_INFERENCE_RESPONSE] = data[
                TIME_IN_ZONE_KEY_IN_SV_DETECTIONS
            ]
        if POLYGON_KEY_IN_SV_DETECTIONS in data:
            detection_dict[POLYGON_KEY_IN_INFERENCE_RESPONSE] = (
                np.asarray(data[POLYGON_KEY_IN_SV_DETECTIONS])
                .astype(float)
                .round()
                .astype(int)
                .tolist()
            )
        if (
            BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS in data
            and BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS in data
        ):
            detection_dict[BOUNDING_RECT_ANGLE_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_RECT_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_HEIGHT_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS
            ]
            detection_dict[BOUNDING_RECT_WIDTH_KEY_IN_INFERENCE_RESPONSE] = data[
                BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS
            ]
        if PARENT_ID_KEY in image_metadata:
            detection_dict[PARENT_ID_KEY] = str(image_metadata[PARENT_ID_KEY])
        # Add parent origin metadata if detection is based on a crop/slice
        if (
            PARENT_ID_KEY in image_metadata
            and ROOT_PARENT_ID_KEY in image_metadata
            and str(image_metadata[PARENT_ID_KEY])
            != str(image_metadata[ROOT_PARENT_ID_KEY])
        ):
            _attach_parent_metadata_to_detection_dict(
                detection_dict=detection_dict,
                data=image_metadata,
                coordinates_key=PARENT_COORDINATES_KEY,
                dimensions_key=PARENT_DIMENSIONS_KEY,
                origin_key=PARENT_ORIGIN_KEY,
            )
            detection_dict[ROOT_PARENT_ID_KEY] = str(image_metadata[ROOT_PARENT_ID_KEY])
            _attach_parent_metadata_to_detection_dict(
                detection_dict=detection_dict,
                data=image_metadata,
                coordinates_key=ROOT_PARENT_COORDINATES_KEY,
                dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
                origin_key=ROOT_PARENT_ORIGIN_KEY,
            )
        if (
            KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS in data
            and KEYPOINTS_XY_KEY_IN_SV_DETECTIONS in data
        ):
            kp_class_id = data[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS]
            kp_class_name = data[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS]
            kp_confidence = data[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS]
            kp_xy = data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS]
            detection_dict[KEYPOINTS_KEY_IN_INFERENCE_RESPONSE] = []
            for (
                keypoint_class_id,
                keypoint_class_name,
                keypoint_confidence,
                (x, y),
            ) in zip(kp_class_id, kp_class_name, kp_confidence, kp_xy):
                detection_dict[KEYPOINTS_KEY_IN_INFERENCE_RESPONSE].append(
                    {
                        "class_id": int(keypoint_class_id),
                        "class": str(keypoint_class_name),
                        "confidence": float(keypoint_confidence),
                        "x": float(x),
                        "y": float(y),
                    }
                )
        if DETECTED_CODE_KEY in data:
            detection_dict[DETECTED_CODE_KEY] = data[DETECTED_CODE_KEY]
        if VELOCITY_KEY_IN_SV_DETECTIONS in data:
            detection_dict[VELOCITY_KEY_IN_INFERENCE_RESPONSE] = _to_plain_list(
                data[VELOCITY_KEY_IN_SV_DETECTIONS]
            )
        if SPEED_KEY_IN_SV_DETECTIONS in data:
            detection_dict[SPEED_KEY_IN_INFERENCE_RESPONSE] = float(
                data[SPEED_KEY_IN_SV_DETECTIONS]
            )
        if SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS in data:
            detection_dict[SMOOTHED_VELOCITY_KEY_IN_INFERENCE_RESPONSE] = (
                _to_plain_list(data[SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS])
            )
        if SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS in data:
            detection_dict[SMOOTHED_SPEED_KEY_IN_INFERENCE_RESPONSE] = float(
                data[SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS]
            )
        if AREA_KEY_IN_SV_DETECTIONS in data:
            detection_dict[AREA_KEY_IN_INFERENCE_RESPONSE] = float(
                data[AREA_KEY_IN_SV_DETECTIONS]
            )
        if AREA_CONVERTED_KEY_IN_SV_DETECTIONS in data:
            detection_dict[AREA_CONVERTED_KEY_IN_INFERENCE_RESPONSE] = float(
                data[AREA_CONVERTED_KEY_IN_SV_DETECTIONS]
            )
        serialized_detections.append(detection_dict)
        kept_indices.append(index)
    serialized_image_metadata = {
        "width": None,
        "height": None,
    }
    image_dimensions = image_metadata.get(IMAGE_DIMENSIONS_KEY)
    if image_dimensions is not None:
        serialized_image_metadata = {
            "width": int(image_dimensions[1]),
            "height": int(image_dimensions[0]),
        }
    return serialized_image_metadata, serialized_detections, kept_indices


def serialise_native_rle_detections(detections: InstanceDetections) -> dict:
    """C3 native RLE serialiser for the rle-instance-seg and semantic-seg kinds.

    Emits the same per-box prediction dicts as ``serialise_sv_detections`` (re-using
    its class-name / detection_id / image_metadata logic) but attaches the COCO RLE
    pulled straight from the carried ``InstancesRLEMasks`` instead of a ``points``
    polygon. Polygon computation is skipped up front (``emit_polygons=False``) so the
    O(N*H*W) mask->polygon work is never spent on a value that would only be popped;
    as a consequence no instance is dropped for an empty contour and every box keeps
    its RLE, attached by ORIGINAL row index.
    """
    if not isinstance(detections, InstanceDetections):
        raise ValueError(
            f"serialise_native_rle_detections(...) expected "
            f"`inference_models.InstanceDetections`, got {type(detections)}."
        )
    if not isinstance(detections.mask, InstancesRLEMasks):
        raise ValueError(
            "serialise_native_rle_detections(...) requires the instance masks to be "
            f"carried as `InstancesRLEMasks`, got {type(detections.mask)}."
        )
    serialized_image_metadata, serialized_detections, kept_indices = (
        _serialise_sv_detections(detections=detections, emit_polygons=False)
    )
    rle_masks = detections.mask.to_coco_rle_masks()
    for detection_dict, original_index in zip(serialized_detections, kept_indices):
        rle = rle_masks[original_index]
        counts = rle.get("counts")
        if isinstance(counts, bytes):
            rle = {"size": rle["size"], "counts": counts.decode("utf-8")}
        detection_dict[RLE_MASK_KEY_IN_INFERENCE_RESPONSE] = rle
    return {"image": serialized_image_metadata, "predictions": serialized_detections}


# Exported under the legacy numpy name so existing loader/imports keep working; now
# points at the native implementation rather than the numpy alias.
serialise_rle_sv_detections = serialise_native_rle_detections


def _resolve_instance_polygon(
    mask: Union[torch.Tensor, InstancesRLEMasks],
    index: int,
    data: dict,
) -> Optional[Any]:
    declared_polygon = data.get(POLYGON_KEY_IN_SV_DETECTIONS)
    if declared_polygon is not None and len(declared_polygon) > 2:
        return declared_polygon
    if isinstance(mask, InstancesRLEMasks):
        instance_mask = coco_rle_masks_to_numpy_mask(
            InstancesRLEMasks(image_size=mask.image_size, masks=[mask.masks[index]])
        )[0]
    else:
        instance_mask = mask[index].detach().cpu().numpy()
    return mask_to_polygon(mask=instance_mask)


def _resolve_class_name(class_id: int, class_names_mapping: dict) -> str:
    class_name = class_names_mapping.get(class_id)
    if class_name is None:
        raise ValueError(
            f"Serialising tensor-native detections, class_id={class_id} is missing "
            f"from the class_names mapping "
            f"(keys present: {sorted(class_names_mapping.keys())})."
        )
    return str(class_name)


def _to_plain_list(value: Any) -> list:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def serialise_native_classification(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
) -> dict:
    """Serialise a native single-label ``ClassificationPrediction`` (single-row,
    bs=1) or a native ``MultiLabelClassificationPrediction`` into the response dict
    the numpy classification blocks produce flag-OFF.

    The ``class_id -> name`` map is read from the prediction's metadata
    (``image_metadata[CLASS_NAMES_KEY]``). Single-label carries PLURAL
    ``images_metadata`` (list, [0] used); multi-label carries SINGULAR
    ``image_metadata`` (dict).

    IMPORTANT - two INCOMPATIBLE flag-OFF shapes share this one serialiser, because
    the tensor-native ``CLASSIFICATION_PREDICTION_KIND`` is produced both by model
    blocks and by hand-built formatter blocks:

    * "model" shape - roboflow multi_class / multi_label classification model blocks
      (and visual_search_classifier). Flag-OFF these ``model_dump(by_alias=True,
      exclude_none=True)`` a numpy ``*InferenceResponse`` then append
      ``prediction_type`` / ``parent_id`` / ``root_parent_id``. Dict is
      ``inference_id``-first (``time`` right after when present); single-label is
      confidence-threshold FILTERED, ``round(..., 4)`` and SORTED desc; a
      ``prediction_type`` key is appended.
    * "formatter" shape - ``vlm_as_classifier`` (D4). Flag-OFF it hand-builds the
      dict and passes it through UNSERIALISED, so it is ``image``-first, in INSERTION
      (== class_id) order, UNROUNDED, with NO threshold filter and NO
      ``prediction_type``. Key order single-label
      ``image, predictions, top, confidence, inference_id, parent_id``; multi-label
      ``image, predictions, predicted_classes, inference_id, parent_id``.

    A native prediction cannot self-describe which shape it wants, so the producer
    stamps an explicit ``CLASSIFICATION_STYLE_KEY`` (``"model"`` / ``"formatter"``)
    into ``image_metadata`` and the serialiser reads it (see
    ``_wants_model_style_classification``, which falls back to the original
    metadata heuristic when the key is absent). NOTE: the formatter shape still
    cannot reproduce the numpy vlm ``class_id = -1`` for an out-of-list class,
    because the native producer collapses such classes into a dense
    ``len(classes)`` id before building the tensors (in-list classes ARE
    byte-identical).
    """
    if isinstance(prediction, ClassificationPrediction):
        metadata_list = prediction.images_metadata or [{}]
        image_metadata = metadata_list[0] or {}
    elif isinstance(prediction, MultiLabelClassificationPrediction):
        image_metadata = prediction.image_metadata or {}
    else:
        raise ValueError(
            f"serialise_native_classification(...) expected "
            f"`inference_models.ClassificationPrediction` or "
            f"`inference_models.MultiLabelClassificationPrediction`, "
            f"got {type(prediction)}."
        )

    class_names_mapping = image_metadata.get(CLASS_NAMES_KEY)
    if class_names_mapping is None:
        raise ValueError(
            f"Serialising tensor-native classification, but "
            f"`image_metadata['{CLASS_NAMES_KEY}']` is missing - the producer "
            f"block must attach the class_id -> name mapping."
        )

    serialized_image_metadata = {"width": None, "height": None}
    image_dimensions = image_metadata.get(IMAGE_DIMENSIONS_KEY)
    if image_dimensions is not None:
        serialized_image_metadata = {
            "width": int(image_dimensions[1]),
            "height": int(image_dimensions[0]),
        }

    if _wants_model_style_classification(image_metadata):
        return _serialise_classification_model_style(
            prediction=prediction,
            image_metadata=image_metadata,
            class_names_mapping=class_names_mapping,
            serialized_image_metadata=serialized_image_metadata,
        )
    return _serialise_classification_formatter_style(
        prediction=prediction,
        image_metadata=image_metadata,
        class_names_mapping=class_names_mapping,
        serialized_image_metadata=serialized_image_metadata,
    )


def _wants_model_style_classification(image_metadata: dict) -> bool:
    """D4 / lane 1b: choose which flag-OFF classification shape to reproduce.

    ``True`` -> numpy ``*InferenceResponse``-derived "model" shape. ``False`` ->
    hand-built ``vlm_as_classifier`` "formatter" shape.

    Primary signal: the explicit ``CLASSIFICATION_STYLE_KEY`` the producer stamps
    into ``image_metadata`` (``"model"`` / ``"formatter"``). This replaces the
    original brittle heuristic, which inferred the shape from incidental metadata.

    Fallback (key ABSENT or an unrecognised value): the original heuristic. The vlm
    formatter attaches only CLASS_NAMES / PREDICTION_TYPE / IMAGE_DIMENSIONS /
    INFERENCE_ID / PARENT_ID, while the model producers additionally attach a
    confidence threshold (multi_class model blocks), ``root_parent_id`` (all model
    blocks + visual_search_classifier) and/or ``time`` - presence of ANY of the
    three signals the model shape. Keeping the heuristic as a fallback makes wiring
    the explicit key non-breaking: an un-wired producer, and the pinned key-ordering
    tests (which set ``root_parent_id``), keep today's output.
    """
    style = image_metadata.get(CLASSIFICATION_STYLE_KEY)
    if style == CLASSIFICATION_STYLE_MODEL:
        return True
    if style == CLASSIFICATION_STYLE_FORMATTER:
        return False
    return (
        image_metadata.get(_CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY) is not None
        or image_metadata.get(ROOT_PARENT_ID_KEY) is not None
        or image_metadata.get(_TIME_KEY) is not None
    )


def _serialise_classification_model_style(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
    image_metadata: dict,
    class_names_mapping: dict,
    serialized_image_metadata: dict,
) -> dict:
    """Numpy ``*InferenceResponse``-derived "model" shape (unchanged behaviour).

    Key order mirrors the numpy ``model_dump(by_alias=True, exclude_none=True)``
    followed by the block's in-place ``prediction_type`` / ``parent_id`` /
    ``root_parent_id`` writes: ``inference_id`` first, ``time`` next when present,
    then ``image``, the predictions section, and the appended lineage keys.
    """
    result: dict = {}
    # numpy dump order puts inference_id BEFORE image (InferenceResponse base
    # fields precede CvInferenceResponse.image in the pydantic MRO).
    if image_metadata.get(INFERENCE_ID_KEY) is not None:
        result[INFERENCE_ID_KEY] = image_metadata[INFERENCE_ID_KEY]
    # `time` sits between inference_id and image in the numpy dump
    # (InferenceResponse declares inference_id, frame_id, time before
    # CvInferenceResponse.image; frame_id is None-excluded in workflows).
    if image_metadata.get(_TIME_KEY) is not None:
        result[_TIME_KEY] = image_metadata[_TIME_KEY]
    result["image"] = serialized_image_metadata

    if isinstance(prediction, ClassificationPrediction):
        # confidence: (1, num_classes) full softmax; class_id: (1,)
        confidence_vector = prediction.confidence.detach().cpu().reshape(-1).tolist()
        # Mirror the numpy `prepare_classification_response` cutoff - drop classes
        # whose (raw) score is below the resolved threshold when the producer attached
        # it; otherwise keep the full softmax. Rounding/sort match the numpy
        # `ClassificationInferenceResponse` (round(score, 4); sort desc by confidence).
        confidence_threshold = image_metadata.get(
            _CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY
        )
        individual_classes_predictions = []
        for class_id, score in enumerate(confidence_vector):
            class_score = float(score)
            if confidence_threshold is not None and class_score < confidence_threshold:
                continue
            individual_classes_predictions.append(
                {
                    "class": str(class_names_mapping.get(class_id, class_id)),
                    "class_id": class_id,
                    "confidence": round(class_score, 4),
                }
            )
        individual_classes_predictions = sorted(
            individual_classes_predictions,
            key=lambda item: item["confidence"],
            reverse=True,
        )
        result["predictions"] = individual_classes_predictions
        result["top"] = (
            individual_classes_predictions[0]["class"]
            if individual_classes_predictions
            else ""
        )
        result["confidence"] = (
            individual_classes_predictions[0]["confidence"]
            if individual_classes_predictions
            else 0.0
        )
    else:
        # MultiLabel: confidence (num_classes,) sigmoid; class_ids = predicted ids
        confidence_vector = prediction.confidence.detach().cpu().reshape(-1).tolist()
        # Same opt-in cutoff as the multi-class branch above: producers that
        # gap-fill a dense vector (e.g. visual_search_classifier) attach the
        # threshold so their synthetic zero-confidence entries are dropped;
        # model predictions attach no threshold and keep every class.
        confidence_threshold = image_metadata.get(
            _CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY
        )
        predictions_dict = {
            str(class_names_mapping.get(class_id, class_id)): {
                "confidence": float(score),
                "class_id": class_id,
            }
            for class_id, score in enumerate(confidence_vector)
            if confidence_threshold is None or float(score) >= confidence_threshold
        }
        predicted_classes = [
            str(class_names_mapping.get(int(class_id), int(class_id)))
            for class_id in prediction.class_ids.detach().cpu().tolist()
        ]
        result["predictions"] = predictions_dict
        result["predicted_classes"] = predicted_classes

    # Append order mirrors the numpy block's in-place writes after model_dump:
    # attach_prediction_type_info, then parent_id, then root_parent_id
    # (inference_id was already emitted first, matching the dump order).
    for source_key in (
        PREDICTION_TYPE_KEY,
        PARENT_ID_KEY,
        ROOT_PARENT_ID_KEY,
    ):
        if image_metadata.get(source_key) is not None:
            result[source_key] = image_metadata[source_key]
    return result


def _serialise_classification_formatter_style(
    prediction: Union[ClassificationPrediction, MultiLabelClassificationPrediction],
    image_metadata: dict,
    class_names_mapping: dict,
    serialized_image_metadata: dict,
) -> dict:
    """D4: reproduce the hand-built ``vlm_as_classifier`` flag-OFF dict byte-for-byte.

    Single-label key order: ``image, predictions, top, confidence, inference_id,
    parent_id``. Multi-label: ``image, predictions, predicted_classes, inference_id,
    parent_id``. Predictions are in INSERTION (== class_id) order, confidences are
    UNROUNDED, there is no threshold filter and no ``prediction_type`` key. Per-entry
    order matches the numpy vlm path (single-label ``class, class_id, confidence``;
    multi-label ``confidence, class_id``).

    Residual divergences (reported, not fixable in the serialiser): (a) an out-of-list
    class gets a dense ``len(classes)`` id from the producer instead of the numpy
    ``-1`` (in-list classes are byte-identical); (b) confidences are read back through
    a float32 tensor, so their least-significant digits can differ from the numpy
    float. ``top`` / ``confidence`` are taken from the native argmax id, matching the
    numpy path even when the top class was out-of-list.
    """
    result: dict = {"image": serialized_image_metadata}

    if isinstance(prediction, ClassificationPrediction):
        confidence_vector = prediction.confidence.detach().cpu().reshape(-1).tolist()
        result["predictions"] = [
            {
                "class": class_names_mapping.get(class_id, class_id),
                "class_id": class_id,
                "confidence": float(score),
            }
            for class_id, score in enumerate(confidence_vector)
        ]
        top_ids = prediction.class_id.detach().cpu().reshape(-1).tolist()
        top_id = int(top_ids[0]) if top_ids else None
        if top_id is not None and 0 <= top_id < len(confidence_vector):
            result["top"] = class_names_mapping.get(top_id, top_id)
            result["confidence"] = float(confidence_vector[top_id])
        else:
            # Empty prediction - numpy vlm never emits this (there is always a top),
            # but keep the numpy `ClassificationInferenceResponse` empty fallback.
            result["top"] = ""
            result["confidence"] = 0.0
    else:
        confidence_vector = prediction.confidence.detach().cpu().reshape(-1).tolist()
        result["predictions"] = {
            class_names_mapping.get(class_id, class_id): {
                "confidence": float(score),
                "class_id": class_id,
            }
            for class_id, score in enumerate(confidence_vector)
        }
        result["predicted_classes"] = [
            class_names_mapping.get(int(class_id), int(class_id))
            for class_id in prediction.class_ids.detach().cpu().tolist()
        ]

    # Trailing keys, matching the numpy vlm dict (always present there).
    result[INFERENCE_ID_KEY] = image_metadata.get(INFERENCE_ID_KEY)
    result[PARENT_ID_KEY] = image_metadata.get(PARENT_ID_KEY)
    return result


def serialise_native_keypoint_detection(
    prediction: Tuple[KeyPoints, Optional[Detections]],
) -> dict:
    """Tuple-aware serialiser for the keypoint-detection kind. The native prediction
    is a ``(KeyPoints, Detections)`` tuple; the per-instance keypoint payload is
    carried on the bbox ``Detections`` ``bboxes_metadata``, so unwrap and defer to
    ``serialise_sv_detections``."""
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise ValueError(
            f"serialise_native_keypoint_detection(...) expected a "
            f"Tuple[KeyPoints, Detections], got {type(prediction)}."
        )
    _, detections = prediction
    if detections is None:
        raise ValueError(
            "Keypoint prediction is missing the bounding-box component required for "
            "serialisation."
        )
    return serialise_sv_detections(detections)


def serialise_native_embedding(value: torch.Tensor) -> list:
    """C4 serialiser for the embedding kind. The native CLIP/PE embedding is a
    (possibly CUDA) ``torch.Tensor``; emit the numpy-faithful ``List[float]``."""
    return value.detach().cpu().tolist()


def serialise_native_tensor(value: torch.Tensor) -> list:
    """C4 serialiser for the tensor kind (e.g. depth-estimation normalized depth,
    raw model tensors). The native value is a (possibly CUDA/MPS) ``torch.Tensor``;
    emit a JSON-serialisable nested list."""
    return value.detach().cpu().tolist()


def serialize_wildcard_kind(value: Any) -> Any:
    """Tensor-aware wildcard (`*`) serialiser — same public name as the numpy
    sibling in ``serializers.py``; the loader swaps the symbol under
    ``ENABLE_TENSOR_DATA_REPRESENTATION``.

    Extends the numpy contract with arms for the native tensor-path values that
    can legitimately reach a wildcard output — native detections, the
    ``(KeyPoints, Detections)`` keypoint tuple, classification predictions
    (e.g. wildcard outputs of ``legacy_compatibility`` dynamic blocks after
    boundary conversion) and bare ``torch.Tensor`` values (serialised exactly
    like the ``tensor`` kind). Legacy values keep the numpy behavior verbatim,
    including the ``sv.Detections`` arm (defensive: non-swapped paths can still
    route sv under the flag) — routed to the NUMPY detections serialiser, since
    this module's same-name ``serialise_sv_detections`` consumes native objects.
    Tuples other than the keypoint prediction pass through untouched, mirroring
    the numpy contract.
    """
    if isinstance(value, WorkflowImageData):
        return serialise_image(image=value)
    if isinstance(value, (Detections, InstanceDetections)):
        return serialise_sv_detections(value)
    if _is_native_key_point_prediction(value):
        # A (KeyPoints, None) tuple raises inside the kind serialiser — same
        # loud behavior as a declared keypoint output missing its bbox component.
        return serialise_native_keypoint_detection(prediction=value)
    if isinstance(
        value, (ClassificationPrediction, MultiLabelClassificationPrediction)
    ):
        return serialise_native_classification(prediction=value)
    if isinstance(value, torch.Tensor):
        return serialise_native_tensor(value=value)
    if isinstance(value, dict):
        return {key: serialize_wildcard_kind(value=item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_wildcard_kind(value=element) for element in value]
    if isinstance(value, sv.Detections):
        return _serialise_legacy_sv_detections(detections=value)
    if isinstance(value, datetime):
        return serialize_timestamp(timestamp=value)
    return value


def _is_native_key_point_prediction(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], KeyPoints)
        and (value[1] is None or isinstance(value[1], (Detections, InstanceDetections)))
    )
