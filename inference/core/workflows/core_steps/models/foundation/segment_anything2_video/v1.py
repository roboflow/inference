"""SAM2 Video Tracker workflow block.

Wraps ``SegmentAnything2Video`` (the streaming SAM2 camera predictor) so
it can be used from a workflow driven by ``InferencePipeline``.  The
pipeline calls ``run`` once per frame with a ``WorkflowImageData`` that
carries ``video_metadata``; this block keeps a SAM2 session per
``video_identifier`` and either re-prompts or propagates existing tracks
on each frame.

Prompt modes:

``first_frame``
    Prompts are taken from ``boxes`` on the first frame of each video
    (or whenever the session for a ``video_identifier`` is fresh) and
    then the block tracks every subsequent frame without re-prompting.

``every_n_frames``
    Re-prompt every ``prompt_interval`` frames.  Useful when the scene
    drifts or new objects may appear; between re-prompts the block
    simply propagates.

``every_frame``
    Convert each incoming ``boxes`` batch to masks via SAM2 on every
    frame.  The temporal memory is still refreshed from the new prompts
    so object ids remain stable within each segment, but you effectively
    get per-frame SAM2 mask prediction rather than tracking.

``boxes`` is ignored on frames where we only propagate.
"""

import uuid
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core import logger
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

DETECTIONS_CLASS_NAME_FIELD = "class_name"

PromptMode = Literal["first_frame", "every_n_frames", "every_frame"]

SHORT_DESCRIPTION = (
    "Segment and track objects across video frames using SAM2's streaming "
    "camera predictor."
)

LONG_DESCRIPTION = """
Run Segment Anything 2 on a live video stream frame by frame, maintaining
per-video temporal memory so object identities are preserved across frames.

Feed box detections from an upstream detector (e.g. a YOLO block) as prompts.
The block keeps a SAM2 session per `video_identifier` and, depending on
`prompt_mode`, either re-seeds the prompts periodically or simply
propagates existing tracks.  Object identities (`tracker_id`) are assigned
by SAM2 itself and are stable across frames while the session is alive.

This block is intended for use with `InferencePipeline`, which delivers one
frame at a time and tags each frame with video metadata.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM2 Video Tracker",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "SAM2",
                "segment anything",
                "video",
                "tracking",
                "META",
            ],
            "ui_manifest": {
                "section": "video",
                "icon": "fa-brands fa-meta",
                "blockPriority": 9.4,
                "needsGPU": True,
                "inference": True,
                "trackers": True,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/segment_anything_2_video@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    boxes: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        description=(
            "Bounding boxes to use as SAM2 prompts.  Only read on frames "
            "where the block re-prompts (see `prompt_mode`)."
        ),
        examples=["$steps.object_detection_model.predictions"],
        default=None,
        json_schema_extra={"always_visible": True},
    )
    version: Union[
        Selector(kind=[STRING_KIND]),
        Literal["hiera_large", "hiera_small", "hiera_tiny", "hiera_b_plus"],
    ] = Field(
        default="hiera_tiny",
        description=(
            "SAM2 model variant to load.  One of hiera_large, hiera_small, "
            "hiera_tiny, hiera_b_plus."
        ),
        examples=["hiera_large"],
    )
    prompt_mode: Literal["first_frame", "every_n_frames", "every_frame"] = Field(
        default="first_frame",
        description=(
            "When to consume `boxes` as SAM2 prompts.  `first_frame` seeds "
            "the session once and then tracks; `every_n_frames` re-seeds "
            "every `prompt_interval` frames; `every_frame` re-seeds on "
            "every frame.  On frames where re-seeding does not happen, "
            "`boxes` is ignored and the block simply propagates."
        ),
    )
    prompt_interval: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=30,
        description=(
            "For `prompt_mode=every_n_frames`: re-prompt every N frames."
        ),
        examples=[30],
    )
    threshold: Union[
        Selector(kind=[FLOAT_KIND]),
        float,
    ] = Field(
        default=0.0,
        description="Minimum confidence for emitted masks.",
        examples=[0.0],
    )

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
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        return [
            "sam2/hiera_large",
            "sam2/hiera_small",
            "sam2/hiera_tiny",
            "sam2/hiera_b_plus",
        ]


class SegmentAnything2VideoBlockV1(WorkflowBlock):
    """Stateful SAM2 video tracking block."""

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        if step_execution_mode is not StepExecutionMode.LOCAL:
            raise NotImplementedError(
                "SAM2 Video Tracker only supports LOCAL workflow step "
                "execution.  Remote execution would ship each frame to a "
                "separate process and break the per-video SAM2 session "
                "that holds the temporal memory.  Set "
                "WORKFLOWS_STEP_EXECUTION_MODE=local (or run on a "
                "dedicated deployment) to use this block."
            )
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode
        self._video_model = None
        self._current_model_id: Optional[str] = None
        self._last_frame_number: Dict[str, int] = {}
        self._frames_since_prompt: Dict[str, int] = {}
        # Per video_id, map SAM2 obj_id -> class metadata from prompting.
        self._obj_id_metadata: Dict[str, Dict[int, Dict[str, object]]] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _get_video_model(self, version: str):
        model_id = f"sam2/{version}"
        if self._video_model is None or self._current_model_id != model_id:
            from inference.models.sam2.segment_anything2_video import (
                SegmentAnything2Video,
            )

            self._video_model = SegmentAnything2Video(
                model_id=model_id, api_key=self._api_key
            )
            self._current_model_id = model_id
            self._last_frame_number.clear()
            self._frames_since_prompt.clear()
            self._obj_id_metadata.clear()
        return self._video_model

    def run(
        self,
        images: Batch[WorkflowImageData],
        boxes: Optional[Batch[sv.Detections]],
        version: str,
        prompt_mode: PromptMode,
        prompt_interval: int,
        threshold: float,
    ) -> BlockResult:
        video_model = self._get_video_model(version=version)

        if boxes is None:
            boxes = [None] * len(images)

        batch_detections: List[sv.Detections] = []
        for single_image, boxes_for_image in zip(images, boxes):
            metadata = single_image.video_metadata
            video_id = metadata.video_identifier
            frame_number = metadata.frame_number or 0

            should_reset, should_prompt = self._session_decision(
                video_id=video_id,
                frame_number=frame_number,
                prompt_mode=prompt_mode,
                prompt_interval=prompt_interval,
                has_boxes=boxes_for_image is not None and len(boxes_for_image) > 0,
                has_session=video_model.has_session(video_id),
            )

            if should_reset:
                video_model.reset_session(video_id)
                self._obj_id_metadata.pop(video_id, None)
                self._frames_since_prompt[video_id] = 0

            frame_np = single_image.numpy_image
            if should_prompt:
                boxes_xyxy, per_box_meta = _extract_box_prompts(boxes_for_image)
                masks, obj_ids = video_model.prompt_and_track(
                    video_id=video_id,
                    frame=frame_np,
                    boxes_xyxy=boxes_xyxy,
                    clear_old_prompts=True,
                )
                self._obj_id_metadata[video_id] = dict(
                    zip([int(i) for i in obj_ids.tolist()], per_box_meta)
                )
                self._frames_since_prompt[video_id] = 0
            elif video_model.has_session(video_id):
                masks, obj_ids = video_model.track(
                    video_id=video_id, frame=frame_np
                )
                self._frames_since_prompt[video_id] = (
                    self._frames_since_prompt.get(video_id, 0) + 1
                )
            else:
                masks = np.zeros(
                    (0, frame_np.shape[0], frame_np.shape[1]), dtype=bool
                )
                obj_ids = np.zeros((0,), dtype=np.int64)

            self._last_frame_number[video_id] = frame_number

            detections = _masks_to_sv_detections(
                masks=masks,
                obj_ids=obj_ids,
                image=single_image,
                obj_id_metadata=self._obj_id_metadata.get(video_id, {}),
                threshold=threshold,
            )
            batch_detections.append(detections)

        batch_detections = attach_prediction_type_info_to_sv_detections_batch(
            predictions=batch_detections,
            prediction_type="instance-segmentation",
        )
        batch_detections = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=batch_detections,
        )
        return [{"predictions": pred} for pred in batch_detections]

    def _session_decision(
        self,
        video_id: str,
        frame_number: int,
        prompt_mode: PromptMode,
        prompt_interval: int,
        has_boxes: bool,
        has_session: bool,
    ) -> Tuple[bool, bool]:
        """Decide whether to reset the session and whether to re-prompt.

        A reset happens when ``frame_number`` rolls back (source stream
        restarted) or when we don't yet have a session for this video.
        """
        last = self._last_frame_number.get(video_id)
        reset = last is None or frame_number < last or not has_session

        if prompt_mode == "every_frame":
            return reset, has_boxes
        if prompt_mode == "every_n_frames":
            frames_since = self._frames_since_prompt.get(video_id, 0)
            due = reset or frames_since >= max(1, prompt_interval)
            return reset, due and has_boxes
        return reset, reset and has_boxes  # first_frame


def _extract_box_prompts(
    boxes_for_image: Optional[sv.Detections],
) -> Tuple[List[Tuple[float, float, float, float]], List[Dict[str, object]]]:
    """Return ``(boxes_xyxy, per_box_metadata)`` parallel lists.

    ``per_box_metadata`` holds the class id / class name / parent
    detection id for each prompt so we can decorate outgoing masks with
    the same class labels as the upstream detector.
    """
    if boxes_for_image is None or len(boxes_for_image) == 0:
        return [], []
    boxes_xyxy: List[Tuple[float, float, float, float]] = []
    metas: List[Dict[str, object]] = []
    for xyxy, _mask, confidence, class_id, _tracker_id, data in boxes_for_image:
        x1, y1, x2, y2 = xyxy
        boxes_xyxy.append((float(x1), float(y1), float(x2), float(y2)))
        metas.append(
            {
                "class_id": int(class_id) if class_id is not None else 0,
                "class_name": (
                    data.get(DETECTIONS_CLASS_NAME_FIELD, "foreground")
                    if isinstance(data, dict)
                    else "foreground"
                ),
                "parent_id": (
                    data.get("detection_id") if isinstance(data, dict) else None
                ),
                "confidence": float(confidence) if confidence is not None else 1.0,
            }
        )
    return boxes_xyxy, metas


def _masks_to_sv_detections(
    masks: np.ndarray,
    obj_ids: np.ndarray,
    image: WorkflowImageData,
    obj_id_metadata: Dict[int, Dict[str, object]],
    threshold: float,
) -> sv.Detections:
    """Build an ``sv.Detections`` of instance-seg predictions with tracker_ids.

    We emit one detection per SAM2 object (not per polygon contour).
    Keeping it at object-level preserves the one-to-one relationship
    with tracker ids; downstream consumers that need polygons can
    re-extract them from the binary mask.
    """
    h, w = image.numpy_image.shape[:2]
    if masks.shape[0] == 0:
        empty = sv.Detections.empty()
        empty[DETECTION_ID_KEY] = np.array([], dtype=object)
        empty[PARENT_ID_KEY] = np.array([], dtype=object)
        empty[IMAGE_DIMENSIONS_KEY] = np.zeros((0, 2), dtype=int)
        empty.data[DETECTIONS_CLASS_NAME_FIELD] = np.array([], dtype=object)
        return empty

    xyxy: List[List[float]] = []
    confidences: List[float] = []
    class_ids: List[int] = []
    class_names: List[str] = []
    tracker_ids: List[int] = []
    detection_ids: List[str] = []
    parent_ids: List[str] = []
    kept_masks: List[np.ndarray] = []

    for mask, obj_id in zip(masks, obj_ids.tolist()):
        meta = obj_id_metadata.get(int(obj_id), {})
        confidence = float(meta.get("confidence", 1.0))
        if confidence < threshold:
            continue
        # Compute xyxy from the mask; skip empty masks.
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        xyxy.append(
            [
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
            ]
        )
        confidences.append(confidence)
        class_ids.append(int(meta.get("class_id", 0) or 0))
        class_names.append(str(meta.get("class_name", "foreground")))
        tracker_ids.append(int(obj_id))
        parent = meta.get("parent_id")
        parent_ids.append(str(parent) if parent is not None else "")
        detection_ids.append(str(uuid.uuid4()))
        kept_masks.append(mask.astype(bool))

    if not kept_masks:
        empty = sv.Detections.empty()
        empty[DETECTION_ID_KEY] = np.array([], dtype=object)
        empty[PARENT_ID_KEY] = np.array([], dtype=object)
        empty[IMAGE_DIMENSIONS_KEY] = np.zeros((0, 2), dtype=int)
        empty.data[DETECTIONS_CLASS_NAME_FIELD] = np.array([], dtype=object)
        return empty

    detections = sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        mask=np.stack(kept_masks, axis=0),
        confidence=np.asarray(confidences, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=int),
        tracker_id=np.asarray(tracker_ids, dtype=int),
    )
    detections.data[DETECTIONS_CLASS_NAME_FIELD] = np.asarray(
        class_names, dtype=object
    )
    detections[DETECTION_ID_KEY] = np.asarray(detection_ids, dtype=object)
    detections[PARENT_ID_KEY] = np.asarray(parent_ids, dtype=object)
    detections[IMAGE_DIMENSIONS_KEY] = np.asarray(
        [[h, w]] * len(detections), dtype=int
    )
    return detections
