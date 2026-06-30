from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from inference.core.roboflow_api import search_project_images_at_roboflow
from inference.core.workflows.execution_engine.constants import (
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
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

SHORT_DESCRIPTION = (
    "Classify an image by finding the most visually similar annotated image."
)

LONG_DESCRIPTION = """
Search a Roboflow classification project for the visually closest image and return
the matched image's classification annotation as a standard classification
prediction. Single-label annotations are returned in single-label classification
shape, and multi-label annotations are returned in multi-label classification
shape.

## How This Block Works

This block uses Roboflow project image search:

1. Receives an input image from the workflow
2. Sends the image to Roboflow project search as `image_base64`
3. Requests candidate image fields, visual search score, and classification annotation
4. Uses the best candidate's annotation as the predicted class
5. Returns the result through the standard `predictions` classification output

The target project must expose classification annotation data in visual search
results. This block does not train a model, create annotations, or manage the
visual search index.
"""

TopK = Annotated[int, Field(ge=1, le=50)]

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
    "classification",
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
    workspace: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Roboflow workspace URL slug that owns the target project.",
        examples=["my-workspace", "$inputs.workspace"],
    )
    target_project: Union[Selector(kind=[ROBOFLOW_PROJECT_KIND]), str] = Field(
        description="Roboflow classification project URL slug to search.",
        examples=["reference-images", "$inputs.target_project"],
    )
    top_k: Union[TopK, Selector(kind=[INTEGER_KIND])] = Field(
        default=1,
        description=(
            "Number of visually similar image candidates to request. The nearest "
            "candidate is used for classification."
        ),
        examples=[1, 5, "$inputs.top_k"],
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
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
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
        workspace: str,
        target_project: str,
        top_k: int = 1,
    ) -> BlockResult:
        if self._api_key is None:
            raise ValueError(
                "Roboflow Visual Search Classifier block cannot run without a "
                "Roboflow API key. Visit https://docs.roboflow.com/api-reference/"
                "authentication#retrieve-an-api-key to learn how to retrieve one."
            )

        if isinstance(image, Batch):
            return [
                self._classify_single_image(
                    image=single_image,
                    workspace=workspace,
                    target_project=target_project,
                    top_k=top_k,
                )
                for single_image in image
            ]

        return self._classify_single_image(
            image=image,
            workspace=workspace,
            target_project=target_project,
            top_k=top_k,
        )

    def _classify_single_image(
        self,
        image: WorkflowImageData,
        workspace: str,
        target_project: str,
        top_k: int,
    ) -> Dict[str, Any]:
        inference_id = f"{uuid4()}"
        try:
            response = search_project_images_at_roboflow(
                api_key=self._api_key,
                workspace=workspace,
                project=target_project,
                image_base64=image.base64_image,
                limit=top_k,
                fields=VISUAL_SEARCH_CLASSIFIER_FIELDS,
            )
            candidates = [
                _format_candidate(candidate)
                for candidate in response.get("results", [])
            ]
            if not candidates:
                return _empty_result(
                    inference_id=inference_id,
                    error_status=False,
                    message="No visually similar images found.",
                )

            best_candidate = candidates[0]
            classification = _normalise_classification_annotation(
                classification=best_candidate.get("classification")
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
                ),
                INFERENCE_ID_KEY: inference_id,
                "candidate_found": True,
                "class_found": True,
                "best_candidate": best_candidate,
                "candidates": candidates,
                "best_candidate_image": _build_best_candidate_image(best_candidate),
                "visual_search_score": best_candidate["score"],
                "error_status": False,
                "message": "Visual search classification completed.",
            }
        except Exception as error:
            return _empty_result(
                inference_id=inference_id,
                error_status=True,
                message=f"Visual search classification failed: {error}",
            )


def _format_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    filename = candidate.get("filename") or candidate.get("name")
    return {
        "image_id": candidate.get("id"),
        "image_url": candidate.get("url"),
        "name": candidate.get("name") or filename,
        "filename": filename,
        "metadata": candidate.get("user_metadata") or {},
        "tags": candidate.get("tags") or [],
        "width": candidate.get("width"),
        "height": candidate.get("height"),
        "aspect_ratio": candidate.get("aspectRatio"),
        "score": candidate.get("score"),
        "classification": candidate.get("classification"),
    }


def _normalise_classification_annotation(
    classification: Any,
) -> Optional[Dict[str, Any]]:
    if isinstance(classification, dict) and _has_single_label_annotation(
        classification=classification
    ):
        return {
            "type": "single_label",
            "classes": [_normalise_class_entry(classification)],
        }

    class_entries = _extract_multi_label_class_entries(classification=classification)
    if not class_entries:
        return None

    return {
        "type": "multi_label",
        "classes": class_entries,
    }


def _has_single_label_annotation(classification: Dict[str, Any]) -> bool:
    return (
        isinstance(classification.get("class"), str)
        and len(classification["class"]) > 0
    )


def _extract_multi_label_class_entries(classification: Any) -> List[Dict[str, Any]]:
    if isinstance(classification, list):
        return _deduplicate_class_entries(
            class_entries=[
                class_entry
                for raw_class_entry in classification
                if (class_entry := _normalise_class_entry(raw_class_entry)) is not None
            ]
        )

    if not isinstance(classification, dict):
        return []

    predictions = classification.get("predictions")
    for classes_key in ("predicted_classes", "classes", "labels"):
        classes = classification.get(classes_key)
        if not isinstance(classes, list):
            continue
        class_entries = [
            class_entry
            for raw_class_entry in classes
            if (
                class_entry := _normalise_class_entry(
                    raw_class_entry,
                    prediction_details=_get_prediction_details(
                        predictions=predictions,
                        raw_class_entry=raw_class_entry,
                    ),
                )
            )
            is not None
        ]
        return _deduplicate_class_entries(class_entries=class_entries)

    return []


def _get_prediction_details(
    predictions: Any,
    raw_class_entry: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(predictions, dict):
        return None
    class_name = _get_class_name(raw_class_entry=raw_class_entry)
    if class_name is None:
        return None
    prediction_details = predictions.get(class_name)
    if not isinstance(prediction_details, dict):
        return None
    return prediction_details


def _normalise_class_entry(
    raw_class_entry: Any,
    prediction_details: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    class_name = _get_class_name(raw_class_entry=raw_class_entry)
    if class_name is None:
        return None
    raw_class_id = None
    if isinstance(raw_class_entry, dict):
        raw_class_id = raw_class_entry.get("class_id")
    if raw_class_id is None and prediction_details is not None:
        raw_class_id = prediction_details.get("class_id")
    return {
        "class": class_name,
        "class_id": _normalise_class_id(raw_class_id),
    }


def _get_class_name(raw_class_entry: Any) -> Optional[str]:
    if isinstance(raw_class_entry, str) and raw_class_entry:
        return raw_class_entry
    if not isinstance(raw_class_entry, dict):
        return None
    class_name = (
        raw_class_entry.get("class")
        or raw_class_entry.get("class_name")
        or raw_class_entry.get("name")
    )
    if not isinstance(class_name, str) or not class_name:
        return None
    return class_name


def _deduplicate_class_entries(
    class_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result = []
    seen_classes = set()
    for class_entry in class_entries:
        class_name = class_entry["class"]
        if class_name in seen_classes:
            continue
        result.append(class_entry)
        seen_classes.add(class_name)
    return result


def _build_classification_prediction(
    image: WorkflowImageData,
    classification: Dict[str, Any],
    inference_id: str,
) -> Dict[str, Any]:
    height, width = image.numpy_image.shape[:2]
    if classification["type"] == "multi_label":
        return _build_multi_label_classification_prediction(
            image=image,
            classification=classification,
            inference_id=inference_id,
            width=width,
            height=height,
        )

    class_entry = classification["classes"][0]
    class_name = class_entry["class"]
    class_id = class_entry["class_id"]
    return {
        "image": {"width": width, "height": height},
        "predictions": [{"class": class_name, "class_id": class_id, "confidence": 1.0}],
        "top": class_name,
        "confidence": 1.0,
        PREDICTION_TYPE_KEY: "classification",
        INFERENCE_ID_KEY: inference_id,
        PARENT_ID_KEY: image.parent_metadata.parent_id,
        ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
    }


def _build_multi_label_classification_prediction(
    image: WorkflowImageData,
    classification: Dict[str, Any],
    inference_id: str,
    width: int,
    height: int,
) -> Dict[str, Any]:
    predictions = {
        class_entry["class"]: {
            "class_id": class_entry["class_id"],
            "confidence": 1.0,
        }
        for class_entry in classification["classes"]
    }
    return {
        "image": {"width": width, "height": height},
        "predictions": predictions,
        "predicted_classes": list(predictions.keys()),
        PREDICTION_TYPE_KEY: "classification",
        INFERENCE_ID_KEY: inference_id,
        PARENT_ID_KEY: image.parent_metadata.parent_id,
        ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
    }


def _normalise_class_id(class_id: Any) -> int:
    if class_id is None:
        return -1
    try:
        return int(class_id)
    except (TypeError, ValueError):
        return -1


def _build_best_candidate_image(
    candidate: Dict[str, Any],
) -> Optional[WorkflowImageData]:
    image_url = candidate.get("image_url")
    if not image_url:
        return None
    parent_id = (
        candidate.get("image_id")
        or candidate.get("filename")
        or "visual_search_classifier_candidate"
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=str(parent_id)),
        image_reference=image_url,
    )


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
        "best_candidate_image": _build_best_candidate_image(best_candidate),
        "visual_search_score": best_candidate["score"],
        "error_status": True,
        "message": (
            "Best visual search candidate does not include a classification "
            "annotation."
        ),
    }
