from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, PositiveInt

from inference.core.entities.requests.inference import (
    KeypointsDetectionInferenceRequest,
)
from inference.core.env import (
    HOSTED_DETECT_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    add_inference_keypoints_to_sv_detections,
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    filter_out_unwanted_classes_from_sv_detections_batch,
)
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
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
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on a keypoint detection model hosted on or uploaded to Roboflow.

## What is Keypoint Detection?

Keypoint detection is a computer vision task that identifies **specific points of interest** on objects and connects them to form **skeletons** or structural representations. Unlike object detection (which only provides bounding boxes), keypoint detection:
- **Detects individual keypoints** (specific points like joints, corners, or landmarks)
- **Connects keypoints** to form skeletons or structures showing relationships between points
- **Provides confidence scores** for each keypoint indicating visibility and accuracy

For example, a person pose estimation model detects keypoints like "left shoulder", "right elbow", "left knee" and connects them to show the person's pose skeleton. Each keypoint has coordinates (x, y) and a confidence score.

## How This Block Works

This block takes one or more images as input and runs them through a trained keypoint detection model. The model analyzes the image and returns detections where each detection contains:
- A **bounding box** (coordinates defining a rectangle around the detected object)
- **Keypoints** (a list of specific points with x, y coordinates and confidence scores)
- **Skeleton connections** (lines connecting keypoints to form structures)
- A **class label** (the name of what was detected, e.g., "person", "hand")
- A **confidence score** for the overall detection

The block applies post-processing techniques like Non-Maximum Suppression (NMS) to filter duplicate detections and ensures keypoints are properly structured and connected.

## Inputs and Outputs

**Input:**
- **images**: One or more images to analyze (can be from workflow inputs or previous steps)

**Output:**
- **predictions**: A `sv.Detections` object containing all detected objects with their bounding boxes, keypoints, skeletons, classes, and confidence scores
- **inference_id**: A unique identifier for this inference run (string value)

## Key Configuration Options

- **model_id**: The identifier for your Roboflow model (format: `workspace/project/version`)
- **confidence**: Minimum confidence threshold for detections (0.0-1.0, default: 0.4) - detections below this threshold are filtered out
- **keypoint_confidence**: Minimum confidence threshold for individual keypoints (0.0-1.0, default: 0.0) - keypoints below this threshold are marked as not visible
- **class_filter**: Optional list of classes to include - if specified, only these classes will be returned
- **iou_threshold**: Intersection over Union threshold for NMS (default: 0.3) - controls how much bounding boxes can overlap before being merged
- **max_detections**: Maximum number of detections to return per image (default: 300)
- **class_agnostic_nms**: If true, NMS ignores class labels when merging overlapping boxes (default: False)
- **max_candidates**: Maximum number of candidates as NMS input to be taken into account (default: 3000)
- **disable_active_learning**: Boolean flag to disable project-level active learning for this block (default: True)
- **active_learning_target_dataset**: Target dataset for active learning, if enabled (optional)

## Common Use Cases

- **Pose Estimation**: Detecting human poses for fitness tracking, sports analysis, or animation (e.g., yoga pose analysis, athlete performance monitoring)
- **Gesture Recognition**: Identifying hand gestures and finger positions for sign language recognition or touchless interfaces
- **Sports Analytics**: Analyzing athlete movements, tracking player positions, or measuring biomechanics
- **Animation and Gaming**: Capturing motion for character animation or creating interactive experiences
- **Healthcare and Rehabilitation**: Monitoring patient movements, assessing physical therapy progress, or analyzing gait patterns
- **Industrial Quality Control**: Detecting keypoints on manufactured parts to verify assembly correctness or measure dimensions

## Model Sources

You can use:
- Models from your private Roboflow account (requires authentication)
- Public models from [Roboflow Universe](https://universe.roboflow.com) (no authentication needed for public models)

## Requirements

You will need to set your Roboflow API key in your Inference environment to use private models. To learn more about setting your Roboflow API key, [refer to the Inference documentation](https://inference.roboflow.com/quickstart/configure_api_key/).

## Connecting to Other Blocks

The keypoint detection results from this block can be connected to:
- **Visualization blocks** to draw skeletons, keypoints, and bounding boxes on images (Keypoint Visualization)
- **Tracking blocks** to track skeletons and poses across video frames
- **Filter blocks** to filter detections based on keypoint confidence, pose characteristics, or class
- **Measurement blocks** to calculate angles, distances, or other geometric properties from keypoint positions
- **Classification blocks** to classify poses or gestures based on keypoint configurations
- **Transformation blocks** to modify or normalize keypoint coordinates
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Keypoint Detection Model",
            "version": "v1",
            "short_description": "Predict skeletons on objects.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 4,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal[
        "roboflow_core/roboflow_keypoint_detection_model@v1",
        "RoboflowKeypointDetectionModel",
        "KeypointsDetectionModel",
    ]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions.",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    keypoint_confidence: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Confidence threshold to predict a keypoint as visible.",
        examples=[0.3, "$inputs.keypoint_confidence"],
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
    class_agnostic_nms: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
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

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions", kind=[KEYPOINT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowKeypointDetectionModelBlockV1(WorkflowBlock):

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
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        keypoint_confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
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
                keypoint_confidence=keypoint_confidence,
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
                keypoint_confidence=keypoint_confidence,
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
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        keypoint_confidence: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
        request = KeypointsDetectionInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=inference_images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            keypoint_confidence=keypoint_confidence,
            source="workflow-execution",
        )
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        predictions = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
        )

    def run_remotely(
        self,
        images: Batch[Optional[WorkflowImageData]],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        keypoint_confidence: Optional[float],
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
            keypoint_confidence_threshold=keypoint_confidence,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        inference_images = [i.base64_image for i in images]
        predictions = client.infer(
            inference_input=inference_images,
            model_id=model_id,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        return self._post_process_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        class_filter: Optional[List[str]],
    ) -> BlockResult:
        inference_ids = [p.get(INFERENCE_ID_KEY, None) for p in predictions]
        detections = convert_inference_detections_batch_to_sv_detections(predictions)
        for prediction, image_detections in zip(predictions, detections):
            add_inference_keypoints_to_sv_detections(
                inference_prediction=prediction["predictions"],
                detections=image_detections,
            )
        detections = attach_prediction_type_info_to_sv_detections_batch(
            predictions=detections,
            prediction_type="keypoint-detection",
        )
        detections = filter_out_unwanted_classes_from_sv_detections_batch(
            predictions=detections,
            classes_to_accept=class_filter,
        )
        detections = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=detections,
        )
        return [
            {"inference_id": inference_id, "predictions": image_detections}
            for inference_id, image_detections in zip(inference_ids, detections)
        ]
