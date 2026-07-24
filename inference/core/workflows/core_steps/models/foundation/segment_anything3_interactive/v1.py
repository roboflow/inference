from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Type, Union

import requests
import supervision as sv
from pydantic import ConfigDict, Field, model_validator

from inference.core import logger
from inference.core.entities.requests.sam2 import (
    Box,
    Point,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.responses.sam2 import Sam2SegmentationPrediction
from inference.core.env import (
    API_BASE_URL,
    CORE_MODEL_SAM3_ENABLED,
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    SAM3_EXEC_MODE,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import build_roboflow_api_headers
from inference.core.utils.url_utils import wrap_url
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    convert_sam2_segmentation_response_to_inference_instances_seg_response,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LABELED_POINTS_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.offline import ensure_builtin_remote_execution_allowed
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient

DETECTIONS_CLASS_NAME_FIELD = "class_name"
DETECTION_ID_FIELD = "detection_id"

SAM3_INTERACTIVE_MODEL_ID = "sam3/sam3_interactive"

SHORT_DESCRIPTION = (
    "Segment a specific object with SAM3 using point and/or bounding box prompts."
)

LONG_DESCRIPTION = """
Run the interactive (promptable visual segmentation) head of Segment Anything 3 (SAM3) on an image.

Unlike the SAM 3 concept segmentation block (which takes text or exemplar prompts and returns
ALL instances of a concept), this block performs SAM2-style interactive segmentation: each prompt
targets ONE object and the model returns a single mask for it.

Two prompt inputs are supported (at least one must be provided):
- **points**: a list of labeled 2D points defining a single object. Positive points mark the
  object to segment, negative points mark regions to exclude (useful to refine the mask).
- **boxes**: detections from another model. Each bounding box becomes a separate prompt and
  the model segments the object inside it. Class names of the boxes are forwarded to the
  predicted masks.
"""


def _as_sam2_points(points: List[Any]) -> List[Point]:
    """Normalise points given as dicts or (x, y[, positive]) sequences into Sam2 Points."""
    result = []
    for raw_point in points:
        if isinstance(raw_point, Point):
            result.append(raw_point)
            continue
        if isinstance(raw_point, dict):
            if "x" not in raw_point or "y" not in raw_point:
                raise ValueError(
                    f"Each point prompt must define `x` and `y` coordinates - got: {raw_point}"
                )
            x, y = raw_point["x"], raw_point["y"]
            positive = raw_point.get("positive", True)
        elif isinstance(raw_point, (list, tuple)) and len(raw_point) in {2, 3}:
            x, y = raw_point[0], raw_point[1]
            positive = raw_point[2] if len(raw_point) == 3 else True
        else:
            raise ValueError(
                f"Invalid point prompt: {raw_point}. Expected dict with `x`, `y` and optional "
                f"`positive` keys, or a sequence of (x, y) or (x, y, positive)."
            )
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError(f"Point coordinates must be numbers - got: {raw_point}")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"Point coordinates must be numbers - got: {raw_point}")
        result.append(Point(x=float(x), y=float(y), positive=bool(positive)))
    return result


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM 3 Interactive",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Sam",
                "SAM3",
                "segment anything",
                "segment anything 3",
                "point prompt",
                "interactive segmentation",
                "PVS",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fa-solid fa-eye",
                "blockPriority": 9.47,
                "needsGPU": True,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/sam3_interactive@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    points: Optional[Union[List[Any], Selector(kind=[LABELED_POINTS_KIND])]] = Field(
        default=None,
        title="Point Prompts",
        description="Labeled points defining a single object to segment. "
        "Each point is {'x': ..., 'y': ..., 'positive': ...} in absolute pixel coordinates - "
        "positive points mark the object, negative points mark regions to exclude. "
        "Plain (x, y) or (x, y, positive) sequences are also accepted.",
        examples=[
            [{"x": 320, "y": 240, "positive": True}],
            "$inputs.points",
        ],
        json_schema_extra={"always_visible": True},
    )
    boxes: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(  # type: ignore
        default=None,
        description="Bounding boxes (from another model) to use as prompts - "
        "the model segments the object inside each box",
        examples=["$steps.object_detection_model.predictions"],
        json_schema_extra={"always_visible": True},
    )
    threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        default=0.0,
        description="Minimum confidence threshold for predicted masks",
        examples=[0.3],
    )
    multimask_output: Union[Optional[bool], Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Flag to determine whether to use SAM3 internal multimask or single mask mode. "
        "For ambiguous prompts (like a single point) setting to True is recommended.",
        examples=[True, "$inputs.multimask_output"],
    )

    @model_validator(mode="after")
    def _validate_points(self) -> "BlockManifest":
        if isinstance(self.points, list):
            _as_sam2_points(self.points)
        return self

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "boxes"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        restrictions = [
            RuntimeRestriction(
                severity=Severity.HARD,
                note="Requires a GPU; run_locally() loads a model that needs CUDA.",
                applies_to_runtimes=[Runtime.SELF_HOSTED_CPU],
                applies_to_step_execution_modes=[StepExecutionMode.LOCAL],
            ),
        ]
        if not CORE_MODEL_SAM3_ENABLED:
            restrictions.append(
                RuntimeRestriction(
                    severity=Severity.HARD,
                    note=(
                        "CORE_MODEL_SAM3_ENABLED=False on Roboflow Hosted "
                        "Serverless: the SAM3 endpoint is not registered, so "
                        "run_remotely() returns 404."
                    ),
                    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
                    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
                )
            )
        return restrictions

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return list of model_id variants that can satisfy this block."""
        return [SAM3_INTERACTIVE_MODEL_ID]


class SegmentAnything3InteractiveBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch[sv.Detections]],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        if SAM3_EXEC_MODE == "remote":
            logger.debug(
                "Running SAM3 Interactive via inference proxy (SAM3_EXEC_MODE=remote)"
            )
            return self.run_via_request(
                images=images,
                points=points,
                boxes=boxes,
                threshold=threshold,
                multimask_output=multimask_output,
            )
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                points=points,
                boxes=boxes,
                threshold=threshold,
                multimask_output=multimask_output,
            )
        if self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                points=points,
                boxes=boxes,
                threshold=threshold,
                multimask_output=multimask_output,
            )
        raise ValueError(f"Unknown step execution mode: {self._step_execution_mode}")

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch[sv.Detections]],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        if boxes is None:
            boxes = [None] * len(images)

        self._model_manager.add_model(
            model_id=SAM3_INTERACTIVE_MODEL_ID,
            api_key=self._api_key,
        )

        predictions = []
        for single_image, boxes_for_image in zip(images, boxes):
            groups = self._build_prompt_groups(
                boxes_for_image=boxes_for_image, points=points
            )
            segmentation_predictions, class_ids, class_names, detection_ids = (
                [],
                [],
                [],
                [],
            )
            for group in groups:
                inference_request = Sam2SegmentationRequest(
                    image=single_image.to_inference_format(numpy_preferred=True),
                    model_id=SAM3_INTERACTIVE_MODEL_ID,
                    api_key=self._api_key,
                    source="workflow-execution",
                    prompts=Sam2PromptSet(prompts=group.prompts),
                    multimask_output=multimask_output,
                )
                segmentation_response = self._model_manager.infer_from_request_sync(
                    SAM3_INTERACTIVE_MODEL_ID, inference_request
                )
                segmentation_predictions.extend(segmentation_response.predictions)
                class_ids.extend(group.class_ids)
                class_names.extend(group.class_names)
                detection_ids.extend(group.detection_ids)
            prediction = (
                convert_sam2_segmentation_response_to_inference_instances_seg_response(
                    sam2_segmentation_predictions=segmentation_predictions,
                    image=single_image,
                    prompt_class_ids=class_ids,
                    prompt_class_names=class_names,
                    prompt_detection_ids=detection_ids,
                    threshold=threshold,
                )
            )
            predictions.append(prediction)

        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch[sv.Detections]],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        ensure_builtin_remote_execution_allowed("SAM3 Interactive remote execution")
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()

        if boxes is None:
            boxes = [None] * len(images)

        predictions = []
        for single_image, boxes_for_image in zip(images, boxes):
            groups = self._build_prompt_groups(
                boxes_for_image=boxes_for_image, points=points
            )
            segmentation_predictions, class_ids, class_names, detection_ids = (
                [],
                [],
                [],
                [],
            )
            for group in groups:
                result = client.sam3_visual_segment(
                    inference_input=single_image.base64_image,
                    prompts=[
                        prompt.dict(exclude_none=True) for prompt in group.prompts
                    ],
                    multimask_output=multimask_output,
                )
                segmentation_predictions.extend(_parse_segmentation_predictions(result))
                class_ids.extend(group.class_ids)
                class_names.extend(group.class_names)
                detection_ids.extend(group.detection_ids)
            prediction = (
                convert_sam2_segmentation_response_to_inference_instances_seg_response(
                    sam2_segmentation_predictions=segmentation_predictions,
                    image=single_image,
                    prompt_class_ids=class_ids,
                    prompt_class_names=class_names,
                    prompt_detection_ids=detection_ids,
                    threshold=threshold,
                )
            )
            predictions.append(prediction)

        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def run_via_request(
        self,
        images: Batch[WorkflowImageData],
        points: Optional[List[Any]],
        boxes: Optional[Batch[sv.Detections]],
        threshold: float,
        multimask_output: bool,
    ) -> BlockResult:
        ensure_builtin_remote_execution_allowed(
            "SAM3 Interactive inference proxy execution"
        )
        endpoint = f"{API_BASE_URL}/inferenceproxy/sam3-pvs"

        if boxes is None:
            boxes = [None] * len(images)

        predictions = []
        for single_image, boxes_for_image in zip(images, boxes):
            groups = self._build_prompt_groups(
                boxes_for_image=boxes_for_image, points=points
            )
            segmentation_predictions, class_ids, class_names, detection_ids = (
                [],
                [],
                [],
                [],
            )
            for group in groups:
                payload = {
                    "image": {"type": "base64", "value": single_image.base64_image},
                    "prompts": Sam2PromptSet(prompts=group.prompts).dict(
                        exclude_none=True
                    ),
                    "multimask_output": multimask_output,
                }
                try:
                    headers = {"Content-Type": "application/json"}
                    if ROBOFLOW_INTERNAL_SERVICE_NAME:
                        headers["X-Roboflow-Internal-Service-Name"] = (
                            ROBOFLOW_INTERNAL_SERVICE_NAME
                        )
                    if ROBOFLOW_INTERNAL_SERVICE_SECRET:
                        headers["X-Roboflow-Internal-Service-Secret"] = (
                            ROBOFLOW_INTERNAL_SERVICE_SECRET
                        )
                    headers = build_roboflow_api_headers(explicit_headers=headers)
                    response = requests.post(
                        wrap_url(f"{endpoint}?api_key={self._api_key}"),
                        json=payload,
                        headers=headers,
                        timeout=60,
                    )
                    response.raise_for_status()
                    resp_json = response.json()
                except Exception as e:
                    raise Exception(f"SAM3 interactive request failed: {e}")

                segmentation_predictions.extend(
                    _parse_segmentation_predictions(resp_json)
                )
                class_ids.extend(group.class_ids)
                class_names.extend(group.class_names)
                detection_ids.extend(group.detection_ids)
            prediction = (
                convert_sam2_segmentation_response_to_inference_instances_seg_response(
                    sam2_segmentation_predictions=segmentation_predictions,
                    image=single_image,
                    prompt_class_ids=class_ids,
                    prompt_class_names=class_names,
                    prompt_detection_ids=detection_ids,
                    threshold=threshold,
                )
            )
            predictions.append(prediction)

        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    @staticmethod
    def _build_prompt_groups(
        boxes_for_image: Optional[sv.Detections],
        points: Optional[List[Any]],
    ) -> List["_PromptGroup"]:
        # The SAM prompt encoder batches a request's prompts into a single
        # tensor, so one request cannot mix box-carrying and box-less prompts -
        # box and point prompts are therefore issued as separate requests.
        groups: List[_PromptGroup] = []
        if boxes_for_image is not None and len(boxes_for_image) > 0:
            prompts: List[Sam2Prompt] = []
            class_ids: List[Optional[int]] = []
            class_names: List[str] = []
            detection_ids: List[Optional[str]] = []
            for xyxy, _, _, class_id, _, bbox_data in boxes_for_image:
                x1, y1, x2, y2 = xyxy
                width = x2 - x1
                height = y2 - y1
                prompts.append(
                    Sam2Prompt(
                        box=Box(
                            x=float(x1 + width / 2),
                            y=float(y1 + height / 2),
                            width=float(width),
                            height=float(height),
                        )
                    )
                )
                class_ids.append(class_id)
                class_names.append(bbox_data[DETECTIONS_CLASS_NAME_FIELD])
                detection_ids.append(bbox_data[DETECTION_ID_FIELD])
            groups.append(
                _PromptGroup(
                    prompts=prompts,
                    class_ids=class_ids,
                    class_names=class_names,
                    detection_ids=detection_ids,
                )
            )
        if points:
            groups.append(
                _PromptGroup(
                    prompts=[Sam2Prompt(points=_as_sam2_points(points))],
                    class_ids=[0],
                    class_names=["foreground"],
                    detection_ids=[None],
                )
            )
        if boxes_for_image is None and not points:
            raise ValueError(
                "SAM 3 Interactive block requires at least one prompt - "
                "provide `points` and/or `boxes` input."
            )
        # boxes input connected but no detections for this image -> no prompts,
        # runners emit empty predictions
        return groups

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="instance-segmentation",
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [{"predictions": prediction} for prediction in predictions]


@dataclass(frozen=True)
class _PromptGroup:
    prompts: List[Sam2Prompt]
    class_ids: List[Optional[int]]
    class_names: List[str]
    detection_ids: List[Optional[str]]


def _parse_segmentation_predictions(
    response: Union[dict, list],
) -> List[Sam2SegmentationPrediction]:
    if isinstance(response, list):
        raw_predictions = response
    else:
        raw_predictions = response.get("predictions", [])
    return [
        Sam2SegmentationPrediction(
            masks=raw_prediction.get("masks", []),
            confidence=raw_prediction.get("confidence", 0.0),
        )
        for raw_prediction in raw_predictions
    ]
