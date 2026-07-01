import math
from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from inference.core.roboflow_api import (
    get_roboflow_workspace,
    search_project_images_at_roboflow,
)
from inference.core.workflows.core_steps.integrations.roboflow.visual_search.helpers import (
    build_visual_search_candidate_image,
    format_visual_search_candidate,
)
from inference.core.workflows.core_steps.integrations.roboflow.visual_search_classifier.classification_annotations import (
    parse_visual_search_classification,
)
from inference.core.workflows.execution_engine.constants import (
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
shape. Classification confidence is derived from the best candidate's public
Roboflow project search `score` field when present.

## How This Block Works

This block uses Roboflow project image search:

1. Receives an input image from the workflow
2. Sends the image to Roboflow project search as `image_base64`
3. Requests candidate image fields and classification labels or annotations
4. Uses the best candidate's annotation as the predicted class
5. Maps the API `score` field to classification confidence when present
6. Returns the result through the standard `predictions` classification output

The target project must expose classification annotation data in visual search
results. This block does not train a model, create annotations, consume raw
search backend relevance scores, or manage the visual search index.
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
        target_project: str,
        workspace: Optional[str] = None,
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
        target_project: str,
        workspace: Optional[str],
        top_k: int,
    ) -> Dict[str, Any]:
        inference_id = f"{uuid4()}"
        try:
            workspace = self._resolve_workspace(workspace=workspace)
            response = search_project_images_at_roboflow(
                api_key=self._api_key,
                workspace=workspace,
                project=target_project,
                image_base64=image.base64_image,
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


def _build_classification_prediction(
    image: WorkflowImageData,
    classification: Dict[str, Any],
    inference_id: str,
    confidence: float,
) -> Dict[str, Any]:
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
    class_id = class_entry["class_id"]
    return {
        "image": {"width": width, "height": height},
        "predictions": [
            {"class": class_name, "class_id": class_id, "confidence": confidence}
        ],
        "top": class_name,
        "confidence": confidence,
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
    confidence: float,
) -> Dict[str, Any]:
    predictions = {
        class_entry["class"]: {
            "class_id": class_entry["class_id"],
            "confidence": confidence,
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
    # The project search API exposes visual similarity scores on a [0, 2] scale.
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
