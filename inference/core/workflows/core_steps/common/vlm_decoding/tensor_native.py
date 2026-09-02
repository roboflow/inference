"""Tensor-native carriers for decoded VLM predictions.

Under ``ENABLE_TENSOR_DATA_REPRESENTATION`` the ``object_detection_prediction``
and ``classification_prediction`` kinds are carried by the native
``inference_models`` dataclasses, not by ``sv.Detections`` / a plain dict: the
kind serializers registered in ``core_steps/loader.py`` and every tensor
consumer (e.g. the visualization blocks reading ``.image_metadata``) expect
them. A VLM block therefore needs NO ``_tensor`` sibling — the shared decoder
converts here, in one place, right before handing ``predictions`` back.

The conversion re-uses the numpy carriers the decoders already built rather
than re-parsing the model answer, so there is exactly one parser per box
format, and reproduces the output of the deprecated tensor formatter blocks
(``formatters/vlm_as_detector/v2_tensor.py`` and
``formatters/vlm_as_classifier/v2_tensor.py``) field for field:

* detections carry ``image_metadata`` with the ``class_id -> name`` map,
  prediction type, dimensions, inference id and the parent/root lineage, plus
  per-box ``detection_id``/``class`` on ``bboxes_metadata``;
* classification carries the dense, ``class_id``-indexed confidence vector and
  is tagged ``CLASSIFICATION_STYLE_FORMATTER`` so
  ``serializers_tensor.serialise_native_classification`` reproduces the "D4 /
  formatter shape" dict — i.e. the serialized ``predictions`` JSON stays
  identical with the flag on and off. The one documented residual divergence
  is inherited from the native structure: an out-of-list class gets a dense
  ``len(classes)`` id instead of the numpy ``-1``, because the native
  confidence vector is indexed by ``class_id``.

``torch`` and ``inference_models`` are OPTIONAL dependencies (the slim
``inference-core`` artifact ships without them), so the imports are guarded
and every entry point falls back to the numpy carrier when they are missing or
the flag is off.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    CLASSIFICATION_STYLE_FORMATTER,
    CLASSIFICATION_STYLE_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

try:
    import torch

    from inference_models.models.base.classification import (
        ClassificationPrediction,
        MultiLabelClassificationPrediction,
    )
    from inference_models.models.base.object_detection import Detections

    TENSOR_NATIVE_CARRIERS_AVAILABLE = True
except ImportError:
    torch = None
    ClassificationPrediction = None
    MultiLabelClassificationPrediction = None
    Detections = None
    TENSOR_NATIVE_CARRIERS_AVAILABLE = False

DETECTION_PREDICTION_TYPE = "object-detection"
CLASSIFICATION_PREDICTION_TYPE = "classification"


def tensor_native_carriers_enabled() -> bool:
    """Whether decoded predictions must be handed back as native carriers."""
    return ENABLE_TENSOR_DATA_REPRESENTATION and TENSOR_NATIVE_CARRIERS_AVAILABLE


def to_tensor_native_predictions(
    predictions: Any,
    image: WorkflowImageData,
    classes: Optional[List[str]],
) -> Any:
    """Convert a decoded prediction into its tensor-native carrier.

    A no-op returning ``predictions`` unchanged when the tensor
    representation is off or the optional dependencies are missing, so the
    numpy path stays byte-for-byte as it was.

    Args:
        predictions: Decoded prediction - ``sv.Detections`` for detection
            tasks, the classification dict for classification tasks.
        image: Workflow image the prediction refers to.
        classes: Class names the block asked the model for; used to populate
            the ``class_id -> name`` map of an empty detection prediction.

    Returns:
        The native ``Detections`` / ``ClassificationPrediction`` /
        ``MultiLabelClassificationPrediction``, or the input unchanged.
    """
    if not tensor_native_carriers_enabled():
        return predictions
    if isinstance(predictions, sv.Detections):
        return native_detections_from_sv_detections(
            detections=predictions,
            image=image,
            classes=classes,
        )
    if isinstance(predictions, dict):
        return native_classification_from_prediction(prediction=predictions)
    return predictions


def native_detections_from_sv_detections(
    detections: sv.Detections,
    image: WorkflowImageData,
    classes: Optional[List[str]],
) -> "Detections":
    """Rebuild ``sv.Detections`` decoded from a VLM answer as native detections.

    Mirrors ``native_detections_from_parsed`` /``empty_native_detections`` of
    ``formatters/vlm_as_detector/v2_tensor.py``: same tensors, same
    ``image_metadata`` keys and same per-box ``bboxes_metadata``. The per-box
    ``detection_id`` values already generated on the numpy carrier are reused
    so both representations of one decode agree.

    Args:
        detections: Detections built by ``build_detections``.
        image: Workflow image the detections refer to.
        classes: Class names the block asked the model for.

    Returns:
        The native ``inference_models.Detections``.
    """
    image_height, image_width = image._read_shape_without_materialization()
    class_name = [
        str(value) for value in detections.data.get(CLASS_NAME_DATA_FIELD, [])
    ]
    class_id = (
        detections.class_id
        if detections.class_id is not None
        else np.empty(0, dtype=int)
    )
    confidence = (
        detections.confidence if detections.confidence is not None else np.empty(0)
    )
    inference_ids = detections.data.get(INFERENCE_ID_KEY, [])
    inference_id = str(inference_ids[0]) if len(inference_ids) > 0 else ""
    if len(detections) == 0:
        # Matches `empty_native_detections`: with no box to resolve, the map is
        # seeded from the requested class list rather than left empty.
        class_names = {
            class_index: class_name_value
            for class_index, class_name_value in enumerate(classes or [])
        }
        bboxes_metadata = None
    else:
        # `class_names` maps each resolved class_id -> the class name the VLM
        # produced, built from the pairs actually present so the serialiser can
        # resolve every id, including unmapped ids (class_id == -1).
        class_names = {
            int(detection_class_id): str(detection_class_name)
            for detection_class_id, detection_class_name in zip(class_id, class_name)
        }
        # The per-box VLM label is carried on bboxes_metadata[i]["class"] so
        # distinct unmapped labels (all sharing class_id == -1) survive: the
        # serialiser prefers this per-box label over the class_id -> name map.
        detection_ids = detections.data.get(DETECTION_ID_KEY, [])
        bboxes_metadata = [
            {
                DETECTION_ID_KEY: str(detection_ids[index]),
                CLASS_NAME_KEY: class_name[index],
            }
            for index in range(len(detections))
        ]
    image_metadata = _build_detections_image_metadata(
        image=image,
        image_height=image_height,
        image_width=image_width,
        inference_id=inference_id,
        class_names=class_names,
    )
    return Detections(
        xyxy=torch.as_tensor(
            np.asarray(detections.xyxy),
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ).reshape(-1, 4),
        class_id=torch.as_tensor(
            np.asarray(class_id),
            dtype=torch.long,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ).reshape(-1),
        confidence=torch.as_tensor(
            np.asarray(confidence),
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ).reshape(-1),
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _build_detections_image_metadata(
    image: WorkflowImageData,
    image_height: int,
    image_width: int,
    inference_id: str,
    class_names: Dict[int, str],
) -> dict:
    """Per-image detection state - verbatim port of ``build_image_metadata``
    from ``formatters/vlm_as_detector/v2_tensor.py``."""
    parent = image.parent_metadata
    root = image.workflow_root_ancestor_metadata
    parent_coordinates = parent.origin_coordinates
    root_coordinates = root.origin_coordinates
    return {
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: DETECTION_PREDICTION_TYPE,
        IMAGE_DIMENSIONS_KEY: [image_height, image_width],
        INFERENCE_ID_KEY: inference_id,
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


def native_classification_from_prediction(prediction: dict):
    """Rebuild a decoded classification dict as its native carrier.

    Mirrors ``parse_multi_class_classification_results`` /
    ``parse_multi_label_classification_results`` of
    ``formatters/vlm_as_classifier/v2_tensor.py``.

    Args:
        prediction: Classification dict built by ``decode_classification``.

    Returns:
        ``MultiLabelClassificationPrediction`` when the dict carries
        ``predicted_classes``, otherwise ``ClassificationPrediction``.
    """
    if "predicted_classes" in prediction:
        return _native_multi_label_classification(prediction=prediction)
    return _native_single_label_classification(prediction=prediction)


def _densify_class_ids(
    named_class_ids: List[Tuple[str, int]],
) -> Dict[str, int]:
    """Give every class a non-negative dense id, in insertion order.

    The numpy dict marks a class the block did not ask for with
    ``class_id == -1``; the native carriers index their confidence vector BY
    class id, so such a class has to take the next free dense slot - exactly
    what the tensor formatter does when it appends an out-of-list class at
    ``len(class2id_mapping)``.
    """
    next_free_class_id = sum(
        1 for _, class_id in named_class_ids if class_id is not None and class_id >= 0
    )
    class_name_to_id: Dict[str, int] = {}
    for class_name, class_id in named_class_ids:
        if class_id is None or class_id < 0:
            class_id = next_free_class_id
            next_free_class_id += 1
        class_name_to_id[class_name] = int(class_id)
    return class_name_to_id


def _build_classification_image_metadata(
    prediction: dict,
    class_names: Dict[int, str],
) -> dict:
    serialized_image = prediction.get("image") or {}
    return {
        CLASS_NAMES_KEY: class_names,
        # Explicit style tag -> the serialiser reproduces the formatter shape.
        CLASSIFICATION_STYLE_KEY: CLASSIFICATION_STYLE_FORMATTER,
        PREDICTION_TYPE_KEY: CLASSIFICATION_PREDICTION_TYPE,
        IMAGE_DIMENSIONS_KEY: [
            serialized_image.get("height"),
            serialized_image.get("width"),
        ],
        INFERENCE_ID_KEY: prediction.get(INFERENCE_ID_KEY),
        PARENT_ID_KEY: prediction.get(PARENT_ID_KEY),
    }


def _native_single_label_classification(
    prediction: dict,
) -> "ClassificationPrediction":
    entries = prediction["predictions"]
    class_name_to_id = _densify_class_ids(
        named_class_ids=[(entry["class"], entry["class_id"]) for entry in entries]
    )
    confidences = {entry["class"]: float(entry["confidence"]) for entry in entries}
    class_names = {
        class_id: class_name for class_name, class_id in class_name_to_id.items()
    }
    confidence_vector = [0.0] * len(class_names)
    for class_name, class_id in class_name_to_id.items():
        confidence_vector[class_id] = confidences[class_name]
    top_class_id = class_name_to_id[prediction["top"]]
    return ClassificationPrediction(
        class_id=torch.tensor(
            [top_class_id],
            dtype=torch.long,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        confidence=torch.tensor(
            [confidence_vector],
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        images_metadata=[
            _build_classification_image_metadata(
                prediction=prediction, class_names=class_names
            )
        ],
    )


def _native_multi_label_classification(
    prediction: dict,
) -> "MultiLabelClassificationPrediction":
    entries = prediction["predictions"]
    class_name_to_id = _densify_class_ids(
        named_class_ids=[
            (class_name, entry["class_id"]) for class_name, entry in entries.items()
        ]
    )
    class_names = {
        class_id: class_name for class_name, class_id in class_name_to_id.items()
    }
    confidence_vector = [0.0] * len(class_names)
    for class_name, class_id in class_name_to_id.items():
        confidence_vector[class_id] = float(entries[class_name]["confidence"])
    predicted_class_ids = [
        class_name_to_id[class_name] for class_name in prediction["predicted_classes"]
    ]
    return MultiLabelClassificationPrediction(
        class_ids=torch.tensor(
            predicted_class_ids,
            dtype=torch.long,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        confidence=torch.tensor(
            confidence_vector,
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        image_metadata=_build_classification_image_metadata(
            prediction=prediction, class_names=class_names
        ),
    )
