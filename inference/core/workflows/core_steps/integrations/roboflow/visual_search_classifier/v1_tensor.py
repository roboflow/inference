"""Tensor-native sibling of ``visual_search_classifier/v1.py``, loaded when
``ENABLE_TENSOR_DATA_REPRESENTATION`` is on.

The block itself is an external-API integration (Roboflow project image
search) - the search flow, candidate formatting and all non-prediction outputs
are verbatim copies of the numpy source. The single tensor-native change is the
``predictions`` output: under the flag the classification kind carries native
``inference_models`` objects, and ``KINDS_SERIALIZERS`` swaps to
``serialise_native_classification`` (which raises on plain dicts), so this
block builds a native ``ClassificationPrediction`` /
``MultiLabelClassificationPrediction`` instead of the legacy response dict.

Reconstruction follows the ``_native_classification_from_inference_response``
precedent (models/roboflow/multi_class_classification/v1_tensor.py): a dense
confidence vector indexed by ``class_id`` with the matched class' confidence,
plus the ``class_id -> name`` map and lineage keys in the metadata.

KNOWN serialized-output divergences vs the numpy sibling (surface before merge):

- P1 (rounding): ``serialise_native_classification`` rounds confidences to 4
  decimal places and round-trips through float32; the numpy block emits the
  raw float64 confidence unrounded. The native serializer convention matches
  the model classification blocks, not this block's hand-built dict.
- P2 (single-label, zero confidence): when the matched candidate has no usable
  ``score`` (confidence == 0.0) no serializer threshold can be attached, so
  gap class ids ``0..class_id-1`` (named ``str(id)``) appear in the serialized
  ``predictions`` list with confidence 0.0; numpy emits only the matched entry.
  With a positive confidence the attached
  ``classification_confidence_threshold`` filters the gap entries and the
  output matches numpy's single-entry shape.
- P3 (multi-label): ``serialise_native_classification`` has no threshold on
  the multi-label branch and emits an entry for EVERY dense-vector index, so
  non-matched (gap) class ids appear with confidence 0.0; numpy emits only the
  matched classes. ``predicted_classes`` matches numpy exactly. Fixing this
  cleanly needs a serializer-side rule (e.g. skip ids absent from
  ``class_names``), which affects other producers - decide there, not here.
"""

import base64
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

import torch
from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from inference.core.env import (
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.roboflow_api import (
    get_roboflow_workspace,
    search_project_images_at_roboflow,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.utils.preprocess import downscale_image_keeping_aspect_ratio
from inference.core.workflows.core_steps.common.utils import run_in_parallel
from inference.core.workflows.core_steps.integrations.roboflow.visual_search.helpers import (
    build_visual_search_candidate_image,
    format_visual_search_candidate,
)
from inference.core.workflows.core_steps.integrations.roboflow.visual_search_classifier.classification_annotations import (
    parse_visual_search_classification,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

# Same literal as models/roboflow/multi_class_classification/v1_tensor.py - the
# serializer reads it from image metadata to filter sub-threshold classes.
CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY = "classification_confidence_threshold"

SHORT_DESCRIPTION = (
    "Classify an image by finding the most visually similar annotated image."
)

LONG_DESCRIPTION = """
Search a Roboflow classification project for the visually closest image and return
the matched image's classification annotation as a standard classification
prediction. Single-label annotations are returned in single-label classification
shape, and multi-label annotations are returned in multi-label classification
shape. Classification confidence is derived from the best candidate's public
Roboflow project search `score` field when present.

This block performs an external visual search API call and is not intended for
real-time or high-throughput workloads. To bound latency, query images are
downscaled with aspect ratio preserved to a maximum side length of 224 pixels by
default. Increase `max_image_size` when finer visual matching is more important
than lower latency.

## How This Block Works

This block uses Roboflow project image search:

1. Receives an input image from the workflow
2. Downscales the query image if its larger side exceeds `max_image_size`
3. Sends the query image to Roboflow project search as `image_base64`
4. Requests candidate image fields and classification labels or annotations
5. Uses the best candidate's annotation as the predicted class
6. Maps the API `score` field to classification confidence when present
7. Returns the result through the standard `predictions` classification output

The target project must expose classification annotation data in visual search
results. This block does not train a model, create annotations, consume raw
search backend relevance scores, or manage the visual search index.
"""

TopK = Annotated[int, Field(ge=1, le=50)]
MaxImageSize = Annotated[int, Field(ge=1)]

VISUAL_SEARCH_CLASSIFIER_FIELDS = [
    "id",
    "name",
    "filename",
    "url",
    "user_metadata",
    "tags",
    "width",
    "height",
    "aspectRatio",
    "score",
    "labels",
    "annotations",
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Visual Search Classifier",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-tags",
                "blockPriority": 4,
                "inference": True,
                "requires_rf_key": True,
            },
        }
    )
    type: Literal["roboflow_core/visual_search_classifier@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image to classify using visual search.",
        examples=["$inputs.image", "$steps.crop.crops"],
    )
    target_project: Union[Selector(kind=[ROBOFLOW_PROJECT_KIND]), str] = Field(
        description="Roboflow classification project URL slug to search.",
        examples=["reference-images", "$inputs.target_project"],
    )
    workspace: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description=(
            "Optional Roboflow workspace URL slug that owns the target project. "
            "If not provided, the workspace is resolved from the request API key."
        ),
        examples=["my-workspace", "$inputs.workspace"],
    )
    top_k: Union[TopK, Selector(kind=[INTEGER_KIND])] = Field(
        default=1,
        description=(
            "Number of visually similar image candidates to request. The nearest "
            "candidate is used for classification."
        ),
        examples=[1, 5, "$inputs.top_k"],
    )
    max_image_size: Union[MaxImageSize, Selector(kind=[INTEGER_KIND])] = Field(
        default=224,
        description=(
            "Maximum side length, in pixels, for the visual search query image. "
            "Images larger than this are downscaled with aspect ratio preserved "
            "before search. Increase this value for finer matching at higher "
            "latency."
        ),
        examples=[224, 640, "$inputs.max_image_size"],
    )

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["image"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[TENSOR_NATIVE_CLASSIFICATION_PREDICTION_KIND]
            ),
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(name="candidate_found", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="class_found", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="best_candidate", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="candidates", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="best_candidate_image", kind=[IMAGE_KIND]),
            OutputDefinition(name="visual_search_score", kind=[FLOAT_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowVisualSearchClassifierBlockV1(WorkflowBlock):
    def __init__(self, api_key: Optional[str]):
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: Union[WorkflowImageData, Batch[WorkflowImageData]],
        target_project: str,
        workspace: Optional[str] = None,
        top_k: int = 1,
        max_image_size: int = 224,
    ) -> BlockResult:
        if self._api_key is None:
            raise ValueError(
                "Roboflow Visual Search Classifier block cannot run without a "
                "Roboflow API key. Visit https://docs.roboflow.com/api-reference/"
                "authentication#retrieve-an-api-key to learn how to retrieve one."
            )

        if isinstance(image, Batch):
            tasks = [
                partial(
                    self._classify_single_image,
                    image=single_image,
                    workspace=workspace,
                    target_project=target_project,
                    top_k=top_k,
                    max_image_size=max_image_size,
                )
                for single_image in image
            ]
            return run_in_parallel(
                tasks=tasks,
                max_workers=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            )

        return self._classify_single_image(
            image=image,
            workspace=workspace,
            target_project=target_project,
            top_k=top_k,
            max_image_size=max_image_size,
        )

    def _classify_single_image(
        self,
        image: WorkflowImageData,
        target_project: str,
        workspace: Optional[str],
        top_k: int,
        max_image_size: int,
    ) -> Dict[str, Any]:
        inference_id = f"{uuid4()}"
        try:
            workspace = self._resolve_workspace(workspace=workspace)
            response = search_project_images_at_roboflow(
                api_key=self._api_key,
                workspace=workspace,
                project=target_project,
                image_base64=_prepare_query_image_base64(
                    image=image, max_image_size=max_image_size
                ),
                limit=top_k,
                fields=VISUAL_SEARCH_CLASSIFIER_FIELDS,
            )
            candidates = [
                _format_visual_search_classifier_candidate(candidate=candidate)
                for candidate in response.get("results", [])
            ]
            if not candidates:
                return _empty_result(
                    inference_id=inference_id,
                    error_status=False,
                    message="No visually similar images found.",
                )

            best_candidate = candidates[0]
            classification = parse_visual_search_classification(
                candidate=best_candidate
            )
            if classification is None:
                return _candidate_without_class_result(
                    inference_id=inference_id,
                    best_candidate=best_candidate,
                    candidates=candidates,
                )

            return {
                "predictions": _build_classification_prediction(
                    image=image,
                    classification=classification,
                    inference_id=inference_id,
                    confidence=_normalise_visual_search_confidence(
                        score=best_candidate.get("score")
                    ),
                ),
                INFERENCE_ID_KEY: inference_id,
                "candidate_found": True,
                "class_found": True,
                "best_candidate": best_candidate,
                "candidates": candidates,
                "best_candidate_image": build_visual_search_candidate_image(
                    candidate=best_candidate,
                    fallback_parent_id="visual_search_classifier_candidate",
                ),
                "visual_search_score": _normalise_visual_search_score(
                    score=best_candidate.get("score")
                ),
                "error_status": False,
                "message": "Visual search classification completed.",
            }
        except Exception as error:
            return _empty_result(
                inference_id=inference_id,
                error_status=True,
                message=f"Visual search classification failed: {error}",
            )

    def _resolve_workspace(self, workspace: Optional[str]) -> str:
        if workspace:
            return workspace
        return get_roboflow_workspace(api_key=self._api_key)


def _prepare_query_image_base64(
    image: WorkflowImageData,
    max_image_size: int,
) -> str:
    # The search API needs JPEG bytes anyway, so a tensor-only input image is
    # materialised to numpy here - a deliberate host copy, same as every
    # remote-execution path.
    source_image = image.numpy_image
    query_image = downscale_image_keeping_aspect_ratio(
        image=source_image,
        desired_size=(max_image_size, max_image_size),
    )
    if query_image.shape[:2] == source_image.shape[:2]:
        return image.base64_image
    return base64.b64encode(
        encode_image_to_jpeg_bytes(query_image, jpeg_quality=95)
    ).decode("ascii")


def _build_classification_prediction(
    image: WorkflowImageData,
    classification: Dict[str, Any],
    inference_id: str,
    confidence: float,
) -> Union[ClassificationPrediction, MultiLabelClassificationPrediction]:
    height, width = image.numpy_image.shape[:2]
    if classification["type"] == "multi_label":
        return _build_multi_label_classification_prediction(
            image=image,
            classification=classification,
            inference_id=inference_id,
            width=width,
            height=height,
            confidence=confidence,
        )

    class_entry = classification["classes"][0]
    class_name = class_entry["class"]
    class_id = int(class_entry["class_id"])
    # Dense vector indexed by class_id; ids below the matched one are gap-filled
    # (str(id) names, 0.0 confidence) - the full class list is not available on
    # this path (same convention as the classifiers' remote-response converter).
    num_classes = class_id + 1
    class_names = {
        gap_class_id: str(gap_class_id) for gap_class_id in range(num_classes)
    }
    class_names[class_id] = str(class_name)
    confidence_vector = [0.0] * num_classes
    confidence_vector[class_id] = confidence
    image_metadata = _build_image_metadata(
        image=image,
        class_names=class_names,
        inference_id=inference_id,
        width=width,
        height=height,
    )
    if confidence > 0.0:
        # Filters the gap-filled zero-confidence classes at serialization so the
        # output keeps numpy's single-entry shape (see P2 in the module docstring).
        image_metadata[CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY] = confidence
    return ClassificationPrediction(
        class_id=torch.tensor(
            [class_id], dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        # float64 for the same reason as the multi-label sibling: float32 would
        # push the API-provided confidence below its own serialization
        # threshold and off the numpy byte-parity value.
        confidence=torch.tensor(
            [confidence_vector],
            dtype=torch.float64,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        images_metadata=[image_metadata],
    )


def _build_multi_label_classification_prediction(
    image: WorkflowImageData,
    classification: Dict[str, Any],
    inference_id: str,
    width: int,
    height: int,
    confidence: float,
) -> MultiLabelClassificationPrediction:
    class_entries = classification["classes"]
    num_classes = (
        max(int(class_entry["class_id"]) for class_entry in class_entries) + 1
        if class_entries
        else 0
    )
    class_names = {
        gap_class_id: str(gap_class_id) for gap_class_id in range(num_classes)
    }
    confidence_vector = [0.0] * num_classes
    predicted_class_ids = []
    for class_entry in class_entries:
        class_id = int(class_entry["class_id"])
        class_names[class_id] = str(class_entry["class"])
        confidence_vector[class_id] = confidence
        predicted_class_ids.append(class_id)
    image_metadata = _build_image_metadata(
        image=image,
        class_names=class_names,
        inference_id=inference_id,
        width=width,
        height=height,
    )
    if confidence > 0.0:
        # Filters the gap-filled zero-confidence classes at serialization so the
        # output keeps numpy's real-labels-only shape (see P2 in the module
        # docstring) - same opt-in as the single-class path below.
        image_metadata[CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY] = confidence
    return MultiLabelClassificationPrediction(
        class_ids=torch.tensor(
            predicted_class_ids, dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        # float64: the confidence is an API-provided python float; float32
        # storage would shift it below its own serialization threshold and off
        # the numpy byte-parity value (0.82 -> 0.8199999928...).
        confidence=torch.tensor(
            confidence_vector,
            dtype=torch.float64,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        # MultiLabel carries SINGULAR image_metadata (dict) - see
        # serialise_native_classification.
        image_metadata=image_metadata,
    )


def _build_image_metadata(
    image: WorkflowImageData,
    class_names: Dict[int, str],
    inference_id: str,
    width: int,
    height: int,
) -> dict:
    return {
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: "classification",
        IMAGE_DIMENSIONS_KEY: [height, width],
        INFERENCE_ID_KEY: inference_id,
        PARENT_ID_KEY: image.parent_metadata.parent_id,
        ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
    }


def _format_visual_search_classifier_candidate(
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    formatted_candidate = format_visual_search_candidate(
        candidate=candidate,
        extra_fields=_visual_search_classifier_extra_fields(candidate=candidate),
    )
    formatted_candidate["score"] = _normalise_visual_search_score(
        score=candidate.get("score")
    )
    return formatted_candidate


def _visual_search_classifier_extra_fields(candidate: Dict[str, Any]) -> List[str]:
    extra_fields = ["score", "labels", "annotations"]
    if "classification" in candidate:
        extra_fields.append("classification")
    return extra_fields


def _normalise_visual_search_score(score: Any) -> Optional[float]:
    try:
        raw_score = float(score)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(raw_score):
        return None
    return raw_score


def _normalise_visual_search_confidence(score: Any) -> float:
    raw_score = _normalise_visual_search_score(score=score)
    if raw_score is None:
        return 0.0
    # Convert the project-search match score into a bounded confidence-like value.
    return max(0.0, min(1.0, raw_score / 2.0))


def _empty_result(
    inference_id: str,
    error_status: bool,
    message: str,
) -> Dict[str, Any]:
    return {
        "predictions": None,
        INFERENCE_ID_KEY: inference_id,
        "candidate_found": False,
        "class_found": False,
        "best_candidate": {},
        "candidates": [],
        "best_candidate_image": None,
        "visual_search_score": None,
        "error_status": error_status,
        "message": message,
    }


def _candidate_without_class_result(
    inference_id: str,
    best_candidate: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "predictions": None,
        INFERENCE_ID_KEY: inference_id,
        "candidate_found": True,
        "class_found": False,
        "best_candidate": best_candidate,
        "candidates": candidates,
        "best_candidate_image": build_visual_search_candidate_image(
            candidate=best_candidate,
            fallback_parent_id="visual_search_classifier_candidate",
        ),
        "visual_search_score": _normalise_visual_search_score(
            score=best_candidate.get("score")
        ),
        "error_status": True,
        "message": (
            "Best visual search candidate does not include classification labels "
            "or annotations."
        ),
    }
