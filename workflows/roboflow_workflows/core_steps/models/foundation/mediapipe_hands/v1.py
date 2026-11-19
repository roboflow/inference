"""MediaPipe Hands landmarks workflow block.

Runs MediaPipe's hand landmarker as two Roboflow models, the palm detector and
the hand landmarks model, both ONNX conversions of Google's TFLite graphs. Hand
regions come from the palm detector or from any upstream detector's boxes; on
video, the regions the previous frame's landmarks give are reused, so the
detector only runs while fewer hands are tracked than asked for.
"""

import uuid
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, PositiveInt
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
)
from roboflow_workflows.core_steps.models.foundation.mediapipe_hands.hands_pipeline import (
    HandResult,
    HandsSession,
    ProviderHandModels,
    run_hands_on_frame,
)
from roboflow_workflows.core_steps.models.workload_presets import (
    REMOTE_STEP_EXECUTION_NOT_SUPPORTED,
)
from roboflow_workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
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
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
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
from supervision.config import CLASS_NAME_DATA_FIELD

DEFAULT_PALM_DETECTOR_MODEL_ID = "mediapipe-palm-detector/full"
DEFAULT_HAND_LANDMARKS_MODEL_ID = "mediapipe-hand-landmarks/full"

HAND_PRESENCE_KEY = "hand_presence"
HANDEDNESS_CONFIDENCE_KEY = "handedness_confidence"
LANDMARKS_Z_KEY = "landmarks_z"
WORLD_LANDMARKS_KEY = "world_landmarks"

SHORT_DESCRIPTION = (
    "Find hands and their 21 landmarks with MediaPipe's hand landmarker."
)

LONG_DESCRIPTION = """
Find hands and return 21 landmarks per hand (the wrist and four joints per
finger), labelled `left_hand` or `right_hand`. This is Google's MediaPipe hand
landmarker (Apache 2.0) with its two models converted to ONNX, so it runs on
the same execution providers as other Roboflow models and needs no `mediapipe`
library.

**Where hands come from**

* By default, MediaPipe's palm detector finds them.
* Connect `hand_detections` to use boxes from any detection model instead.
  MediaPipe's palm detector rarely finds gloved hands; a model trained on your
  own gloved hands, followed by this block, often does. The palm detector is
  then never loaded. Each box gets two landmark passes: the first finds the
  hand's orientation, the second runs on the rotated region it gives.

**Video**

When frames come from a video (`InferencePipeline`), the block keeps each
video's hand regions and reuses them on the next frame, the way MediaPipe
tracks hands: the detector only runs while fewer than `max_num_hands` hands are
tracked, and a hand is dropped once its presence falls below
`hand_presence_threshold`. Turn this off with `reuse_hand_regions`. Region
reuse is not identity tracking; chain a tracker block such as ByteTrack after
this one for stable `tracker_id`s. Hand scores (detection score times
presence) are often around 0.5, below ByteTrack's default activation
thresholds, so lower `track_activation_threshold` and
`high_conf_det_threshold` (0.3 works) or it will not start tracks.

**Output**

Keypoint detections in image pixels. Every landmark carries the hand's
score, since MediaPipe predicts no per-landmark visibility. Each detection
also carries `hand_presence`, `handedness_confidence`, `landmarks_z` (21
relative depths, in the same pixel scale as x) and `world_landmarks` (21 x 3,
metres, origin at the hand's centre) in its data. Those four are available to
blocks in the same workflow but are not included in serialised outputs.

Handedness is reported for the person in a normal camera view. Set
`image_is_mirrored` for a mirrored selfie view, MediaPipe's own convention.

Runs with LOCAL step execution only.
"""


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
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
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
                name="predictions", kind=[KEYPOINT_DETECTION_PREDICTION_KIND]
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
        hand_detections: Optional[Batch[Optional[sv.Detections]]],
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
                    hand_boxes, box_scores = detections.xyxy, detections.confidence
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
                _hands_to_sv_detections(
                    hands=hands,
                    image=image,
                    class_names=HAND_CLASS_NAMES,
                    landmark_names=HAND_LANDMARK_NAMES,
                )
            )
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions, prediction_type="keypoint-detection"
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images, predictions=predictions
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


def _hands_to_sv_detections(
    *,
    hands: List[HandResult],
    image: WorkflowImageData,
    class_names: List[str],
    landmark_names: List[str],
) -> sv.Detections:
    count, landmarks = len(hands), len(landmark_names)
    height, width = image.numpy_image.shape[:2]
    class_id = np.asarray([hand.class_id for hand in hands], dtype=int)
    scores = np.asarray([hand.score for hand in hands], dtype=np.float32)
    detections = sv.Detections(
        xyxy=np.asarray([hand.xyxy for hand in hands], dtype=np.float32).reshape(
            count, 4
        ),
        confidence=scores,
        class_id=class_id,
    )
    detections[CLASS_NAME_DATA_FIELD] = np.asarray(
        [class_names[index] for index in class_id], dtype=object
    ).reshape(count)
    detections[DETECTION_ID_KEY] = np.asarray(
        [str(uuid.uuid4()) for _ in hands], dtype=object
    ).reshape(count)
    detections[IMAGE_DIMENSIONS_KEY] = np.asarray(
        [[height, width]] * count, dtype=int
    ).reshape(count, 2)
    detections[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.asarray(
        [hand.key_points_xy for hand in hands], dtype=np.float32
    ).reshape(count, landmarks, 2)
    detections[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = np.repeat(
        scores[:, None], landmarks, axis=1
    ).reshape(count, landmarks)
    detections[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = np.tile(
        np.arange(landmarks), (count, 1)
    )
    detections[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = np.tile(
        np.asarray(landmark_names, dtype=object), (count, 1)
    )
    detections[HAND_PRESENCE_KEY] = np.asarray(
        [hand.hand_presence for hand in hands], dtype=np.float32
    ).reshape(count)
    detections[HANDEDNESS_CONFIDENCE_KEY] = np.asarray(
        [hand.handedness_confidence for hand in hands], dtype=np.float32
    ).reshape(count)
    detections[LANDMARKS_Z_KEY] = np.asarray(
        [hand.landmarks_z for hand in hands], dtype=np.float32
    ).reshape(count, landmarks)
    detections[WORLD_LANDMARKS_KEY] = np.asarray(
        [hand.world_landmarks for hand in hands], dtype=np.float32
    ).reshape(count, landmarks, 3)
    return detections
