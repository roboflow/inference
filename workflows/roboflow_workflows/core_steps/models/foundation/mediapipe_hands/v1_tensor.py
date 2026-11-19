"""Tensor-native sibling of `roboflow_core/mediapipe_hand_landmarks@v1`.

Same manifest, models and per-video region reuse as the numpy block
(``hands_pipeline`` is shared). Upstream hand boxes arrive as tensor-native
predictions and the output is the native ``(KeyPoints, Detections)`` keypoint
prediction, with each instance's keypoints flattened into the box metadata
the tensor serialiser reads.
"""

from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
from pydantic import ConfigDict, Field, PositiveInt
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.tensor_native import (
    attach_native_detection_metadata,
    split_key_point_prediction,
)
from roboflow_workflows.core_steps.models.foundation.mediapipe_hands.hands_pipeline import (
    HandResult,
    HandsSession,
    ProviderHandModels,
    run_hands_on_frame,
)
from roboflow_workflows.core_steps.models.foundation.mediapipe_hands.v1 import (
    DEFAULT_HAND_LANDMARKS_MODEL_ID,
    DEFAULT_PALM_DETECTOR_MODEL_ID,
    HAND_PRESENCE_KEY,
    HANDEDNESS_CONFIDENCE_KEY,
    LANDMARKS_Z_KEY,
    LONG_DESCRIPTION,
    SHORT_DESCRIPTION,
    WORLD_LANDMARKS_KEY,
)
from roboflow_workflows.core_steps.models.workload_presets import (
    REMOTE_STEP_EXECUTION_NOT_SUPPORTED,
)
from roboflow_workflows.execution_engine.constants import (
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ImageInputField,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    ModelExecutionLocation,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
    roboflow_platform_model,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider

from inference_models import Detections, KeyPoints

TensorNativePrediction = Union[Detections, Tuple[KeyPoints, Detections]]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "MediaPipe Hand Landmarks",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "MediaPipe",
                "hands",
                "hand landmarks",
                "fingers",
                "keypoints",
                "gloves",
                "pose",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-hand",
                "blockPriority": 12,
                "inference": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/mediapipe_hand_landmarks@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    hand_detections: Optional[
        Selector(
            kind=[
                TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description=(
            "Hand boxes from an upstream detection model. When set, these "
            "replace MediaPipe's palm detector, which is then not loaded."
        ),
        examples=["$steps.hand_detector.predictions"],
        json_schema_extra={"always_visible": True},
    )
    max_num_hands: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=2,
        description="Most hands returned per image.",
        examples=[2, "$inputs.max_num_hands"],
    )
    confidence: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.5,
        description=(
            "Minimum palm detector score. Not used with `hand_detections`; "
            "filter those boxes upstream."
        ),
        examples=[0.5, "$inputs.confidence"],
    )
    hand_presence_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = (
        Field(
            default=0.5,
            description=(
                "Minimum hand presence from the landmark model. Hands below "
                "it are dropped, and on video stop being tracked."
            ),
            examples=[0.5, "$inputs.hand_presence_threshold"],
        )
    )
    image_is_mirrored: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description=(
            "Report handedness for a mirrored (selfie) view, MediaPipe's own "
            "convention. Leave off for a normal camera."
        ),
        examples=[False, "$inputs.image_is_mirrored"],
    )
    reuse_hand_regions: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description=(
            "On video, reuse the previous frame's hand regions instead of "
            "detecting hands on every frame."
        ),
        examples=[True, "$inputs.reuse_hand_regions"],
    )
    palm_detector_model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = Field(
        default=DEFAULT_PALM_DETECTOR_MODEL_ID,
        description="Model ID of MediaPipe's palm detector.",
        examples=[DEFAULT_PALM_DETECTOR_MODEL_ID],
    )
    hand_landmarks_model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            default=DEFAULT_HAND_LANDMARKS_MODEL_ID,
            description="Model ID of MediaPipe's hand landmarks model.",
            examples=[DEFAULT_HAND_LANDMARKS_MODEL_ID],
        )
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images", "hand_detections"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        """Return the block's coarse execution restrictions.

        Returns:
            Restrictions that apply on this host, each with a stable ``code``.
        """
        return [REMOTE_STEP_EXECUTION_NOT_SUPPORTED]

    def discover_dependent_resources(self) -> Optional[List[DependentResource]]:
        """Declare the landmark model, and the palm detector when it is used.

        With ``hand_detections`` connected the palm detector is never loaded,
        so it is not a dependency.

        Returns:
            The models the block's LOCAL run path loads.
        """
        models = [self.hand_landmarks_model_id]
        if self.hand_detections is None:
            models.insert(0, self.palm_detector_model_id)
        return [
            roboflow_platform_model(
                model_id, execution_location=ModelExecutionLocation.LOCAL
            )
            for model_id in models
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        """Declare the block's work: model inference.

        Returns:
            The operations the block performs.
        """
        return [WorkOperation.MODEL_INFERENCE]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Declare that REMOTE step execution is not supported.

        Args:
            ignore_environment_restrictions: If True, return every declaration
                with its condition intact (the portable view). If False,
                evaluate configuration predicates against this host and drop
                entries that definitively do not apply here.

        Returns:
            The step's restrictions.
        """
        return actual_restrictions_of(
            declared=[REMOTE_STEP_EXECUTION_NOT_SUPPORTED],
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )


class MediaPipeHandLandmarksBlockV1(WorkflowBlock):
    """Hand landmarks per image, with hand regions reused across video frames."""

    _REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE = (
        "MediaPipe Hand Landmarks only supports LOCAL workflow step execution: "
        "it passes hand regions between its two models and keeps them across "
        "video frames, which the remote model API cannot carry. Set "
        "WORKFLOWS_STEP_EXECUTION_MODE=local to use this block."
    )

    def __init__(
        self,
        model_manager: ModelsProvider,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._sessions: Dict[str, HandsSession] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        hand_detections: Optional[Batch[Optional[TensorNativePrediction]]],
        max_num_hands: int,
        confidence: float,
        hand_presence_threshold: float,
        image_is_mirrored: bool,
        reuse_hand_regions: bool,
        palm_detector_model_id: str,
        hand_landmarks_model_id: str,
    ) -> BlockResult:
        if self._step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(self._REMOTE_EXECUTION_NOT_SUPPORTED_MESSAGE)
        from inference_models.models.mediapipe_hands.hand_landmarks_onnx import (
            HAND_CLASS_NAMES,
            HAND_LANDMARK_NAMES,
        )

        models = ProviderHandModels(
            model_manager=self._model_manager,
            api_key=self._api_key,
            palm_detector_model_id=palm_detector_model_id,
            hand_landmarks_model_id=hand_landmarks_model_id,
        )
        uses_upstream_boxes = hand_detections is not None
        if hand_detections is None:
            hand_detections = [None] * len(images)
        predictions = []
        for image, detections in zip(images, hand_detections):
            hand_boxes, box_scores = None, None
            if uses_upstream_boxes:
                hand_boxes = np.zeros((0, 4), dtype=np.float32)
                if detections is not None:
                    hand_boxes, box_scores = _boxes_and_scores(detections)
            session, frame_number = self._session_for(
                image=image,
                reuse_hand_regions=reuse_hand_regions,
                models=models,
            )
            hands = run_hands_on_frame(
                image.numpy_image,
                models=models,
                session=session,
                frame_number=frame_number,
                hand_boxes=hand_boxes,
                box_scores=box_scores,
                max_num_hands=max_num_hands,
                confidence=confidence,
                hand_presence_threshold=hand_presence_threshold,
                image_is_mirrored=image_is_mirrored,
            )
            predictions.append(
                _hands_to_native_prediction(
                    hands=hands,
                    image=image,
                    class_names=HAND_CLASS_NAMES,
                    landmark_names=HAND_LANDMARK_NAMES,
                )
            )
        return [{"predictions": prediction} for prediction in predictions]

    def _session_for(
        self,
        *,
        image: WorkflowImageData,
        reuse_hand_regions: bool,
        models: ProviderHandModels,
    ) -> Tuple[Optional[HandsSession], int]:
        metadata = image.video_metadata
        frame_number = metadata.frame_number if metadata is not None else None
        if not reuse_hand_regions or frame_number is None:
            return None, 0
        # Regions found with other models are not this video's regions.
        key = (
            f"{metadata.video_identifier}|{models.palm_detector_model_id}"
            f"|{models.hand_landmarks_model_id}"
        )
        session = self._sessions.setdefault(key, HandsSession())
        if frame_number <= session.last_frame_number:
            session.hands = []
            session.last_frame_number = -1
        return session, frame_number


def _boxes_and_scores(
    prediction: TensorNativePrediction,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    _, detections = split_key_point_prediction(prediction)
    boxes = detections.xyxy.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
    if detections.confidence is None:
        return boxes, None
    return boxes, detections.confidence.detach().cpu().numpy().astype(np.float32)


def _hands_to_native_prediction(
    *,
    hands: List[HandResult],
    image: WorkflowImageData,
    class_names: List[str],
    landmark_names: List[str],
) -> Tuple[KeyPoints, Detections]:
    count, landmarks = len(hands), len(landmark_names)
    class_id = torch.as_tensor([hand.class_id for hand in hands], dtype=torch.int64)
    scores = torch.as_tensor([hand.score for hand in hands], dtype=torch.float32)
    metadata = [
        {
            HAND_PRESENCE_KEY: hand.hand_presence,
            HANDEDNESS_CONFIDENCE_KEY: hand.handedness_confidence,
            LANDMARKS_Z_KEY: hand.landmarks_z.tolist(),
            WORLD_LANDMARKS_KEY: hand.world_landmarks.tolist(),
        }
        for hand in hands
    ]
    key_points = KeyPoints(
        xy=torch.as_tensor(
            np.asarray([hand.key_points_xy for hand in hands], dtype=np.float32)
        ).reshape(count, landmarks, 2),
        class_id=class_id.reshape(count),
        confidence=scores.reshape(count, 1).expand(count, landmarks).clone(),
        key_points_metadata=[dict(entry) for entry in metadata],
        detection_confidence=scores.reshape(count),
    )
    detections = Detections(
        xyxy=torch.as_tensor(
            np.asarray([hand.xyxy for hand in hands], dtype=np.float32)
        ).reshape(count, 4),
        class_id=class_id.reshape(count),
        confidence=scores.reshape(count),
        bboxes_metadata=metadata or None,
    )
    detections = attach_native_detection_metadata(
        detections=detections,
        image=image,
        class_names=dict(enumerate(class_names)),
        prediction_type="keypoint-detection",
    )
    for hand, entry in zip(hands, detections.bboxes_metadata or []):
        entry[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = list(range(landmarks))
        entry[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = list(landmark_names)
        entry[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = [hand.score] * landmarks
        entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = hand.key_points_xy.tolist()
    return key_points, detections
