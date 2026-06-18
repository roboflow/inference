"""Tensor-native sibling of `roboflow_core/roboflow_object_detection_model@v3`.

Under ENABLE_TENSOR_DATA_REPRESENTATION this block emits a native
``inference_models.Detections`` (torch tensors on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``)
under ``TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND`` instead of ``sv.Detections``.

- LOCAL: ``ModelManager.run_tensor_native_inference`` returns ``List[Detections]``
  straight from the adapter (xyxy / class_id / confidence only). The block applies
  ``class_filter`` natively (the adapter/model does NOT read it on this path) and
  attaches the producer contract (``image_metadata[class_names]`` + per-box
  ``detection_id``) the tensor serialiser requires, via ``attach_native_detection_metadata``.
- REMOTE: standard inference prediction dicts are rebuilt into a native ``Detections``
  via ``native_detections_from_inference_predictions`` (never ``sv.Detections``).
"""

import uuid
from typing import Dict, List, Literal, Optional, Type, Union

import torch
from pydantic import ConfigDict, Field, PositiveInt, model_validator

from inference.core.env import (
    HOSTED_DETECT_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    attach_native_detection_metadata,
    native_detections_from_inference_predictions,
    take_prediction_by_mask,
)
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.object_detection import Detections
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

PREDICTION_TYPE = "object-detection"

LONG_DESCRIPTION = """
Run inference on a object-detection model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Object Detection Model",
            "version": "v3",
            "short_description": "Predict the location of objects with bounding boxes.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["yolo", "rfdetr", "rf-detr"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 0,
                "inference": True,
                "popular": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_object_detection_model@v3"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence_mode: Union[
        Literal["best", "default", "custom"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="best",
        description="How confidence thresholds are determined.",
        json_schema_extra={
            "always_visible": True,
            "values_metadata": {
                "best": {
                    "name": "Best (Recommended)",
                    "description": "Use F1-optimal thresholds from model evaluation.",
                },
                "default": {
                    "name": "Default",
                    "description": "Use the model's built-in default threshold.",
                },
                "custom": {
                    "name": "Custom",
                    "description": "Specify a custom confidence threshold.",
                },
            },
        },
    )
    custom_confidence: Union[
        Optional[FloatZeroToOne],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Custom confidence threshold for predictions.",
        examples=[0.3, "$inputs.confidence_threshold"],
        json_schema_extra={
            "relevant_for": {
                "confidence_mode": {"values": ["custom"], "required": True},
            },
        },
    )
    class_filter: Union[Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])] = (
        Field(
            default=None,
            description="List of accepted classes. Classes must exist in the model's training set.",
            examples=[["a", "b", "c"], "$inputs.class_filter"],
        )
    )
    iou_threshold: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=300,
        description="Maximum number of detections to return.",
        examples=[300, "$inputs.max_detections"],
    )
    class_agnostic_nms: Union[Optional[bool], Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to specify if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    max_candidates: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    disable_active_learning: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to disable project-level active learning for this block.",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        Selector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for active learning, if enabled.",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.confidence_mode == "custom" and self.custom_confidence is None:
            raise ValueError(
                "`custom_confidence` is required when `confidence_mode` is 'custom'"
            )
        return self

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["object-detection"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="inference_id", kind=[INFERENCE_ID_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowObjectDetectionModelBlockV3(WorkflowBlock):

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
        model_id: str,
        confidence_mode: str,
        custom_confidence: Optional[float],
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        confidence = (
            custom_confidence if confidence_mode == "custom" else confidence_mode
        )
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["best", "default"]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        # Feed the representation already materialised on the images to avoid forcing a
        # numpy->device conversion: GPU tensors (RGB) only when every image in the batch
        # already has one, otherwise the numpy frames (BGR) — matching what the numpy
        # block hands the model.
        if all(image.is_tensor_materialised() for image in images):
            model_inputs = [image.tensor_image for image in images]
            image_color_format = "rgb"
        else:
            model_inputs = [image.numpy_image for image in images]
            image_color_format = "bgr"
        self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
        detections_batch: List[Detections] = (
            self._model_manager.run_tensor_native_inference(
                model_id=model_id,
                images=model_inputs,
                input_color_format=image_color_format,
                confidence=confidence,
                iou_threshold=iou_threshold,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                max_detections=max_detections,
                max_candidates=max_candidates,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        )
        class_names = _class_names_map(self._model_manager.get_class_names(model_id))
        results: List[dict] = []
        for image, detections in zip(images, detections_batch):
            # The adapter/model does NOT honour class_filter on the native path, so
            # filter here (the only place it is applied for LOCAL execution).
            detections = _filter_classes_native(detections, class_filter, class_names)
            # Reuse the adapter-provided inference id when present (numpy parity:
            # numpy v3 surfaces ``p.get(INFERENCE_ID_KEY)`` off the model dump);
            # the tensor-native adapter normally attaches none, so fall back to a
            # freshly minted uuid that is then shared with ``image_metadata``.
            inference_id = getattr(detections, "inference_id", None) or str(
                uuid.uuid4()
            )
            detections = attach_native_detection_metadata(
                detections=detections,
                image=image,
                class_names=class_names,
                prediction_type=PREDICTION_TYPE,
                inference_id=inference_id,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": detections,
                    "model_id": model_id,
                }
            )
        return results

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["best", "default"]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_DETECT_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        client_config = InferenceConfiguration(
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        non_empty_inference_images = [i.base64_image for i in images]
        predictions = client.infer(
            inference_input=non_empty_inference_images,
            model_id=model_id,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        return self._post_process_remote_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
            model_id=model_id,
        )

    def _post_process_remote_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        class_filter: Optional[List[str]],
        model_id: str,
    ) -> BlockResult:
        results: List[dict] = []
        for image, response in zip(images, predictions):
            inference_id = response.get(INFERENCE_ID_KEY) or str(uuid.uuid4())
            detection_dicts = response.get("predictions", [])
            if class_filter:
                detection_dicts = [
                    d for d in detection_dicts if d.get("class") in class_filter
                ]
            detections = native_detections_from_inference_predictions(
                image=image,
                predictions=detection_dicts,
                prediction_type=PREDICTION_TYPE,
                inference_id=inference_id,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            )
            results.append(
                {
                    "inference_id": inference_id,
                    "predictions": detections,
                    "model_id": model_id,
                }
            )
        return results


def _class_names_map(class_names: List[str]) -> Dict[int, str]:
    return {index: name for index, name in enumerate(class_names)}


def _filter_classes_native(
    detections: Detections,
    class_filter: Optional[List[str]],
    class_names: Dict[int, str],
) -> Detections:
    if not class_filter:
        return detections
    accepted = set(class_filter)
    accepted_ids = sorted(
        class_id for class_id, name in class_names.items() if name in accepted
    )
    if not accepted_ids:
        return take_prediction_by_mask(
            detections,
            torch.zeros_like(detections.class_id, dtype=torch.bool),
        )
    accepted_tensor = torch.as_tensor(accepted_ids, device=detections.class_id.device)
    keep = torch.isin(detections.class_id, accepted_tensor)
    return take_prediction_by_mask(detections, keep)
